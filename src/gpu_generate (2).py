"""GPU JOB (runs on t4v3): greedy generation for one model, preemption-safe and OOM-safe.

Design guarantees
  * Checkpoint: finished generations are flushed to the mounted blob output every
    --ckpt_minutes (default 15) as immutable part files (write .tmp -> fsync -> rename),
    and also on SIGTERM and at the end. A preemption loses <= 15 min of work.
  * Resume: on start, every item_id already present in part-*.jsonl is skipped
    (truncated lines are ignored, stale .tmp files removed), so retries continue where
    the previous attempt stopped instead of starting over.
  * OOM: batches are sized by a token budget derived from free GPU memory. On CUDA OOM
    the budget is halved and the SAME items are retried (never dropped). The learned
    budget is persisted in state.json so the next attempt does not repeat the OOM.
  * Completeness: number of outputs is asserted == batch size; `_DONE` is written only
    when every item_id in the prompt file has a record. Otherwise exit code 1 so the
    watcher resubmits.
"""
import argparse
import gc
import glob
import json
import os
import re
import signal
import sys
import time
import uuid

import pandas as pd
import torch

ANS_TAIL = re.compile(r"(?i:answer)\s*[:\-]?[^A-Za-z\n]*(?:[A-D][^A-Za-z\n]*){3}[A-D]")  # v3: 4 boxes, labelled or not


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- checkpoint writer
class ShardWriter:
    def __init__(self, out_dir, run_id, ckpt_minutes, max_records=2000):
        self.out_dir, self.run_id = out_dir, run_id
        self.every = ckpt_minutes * 60
        self.max_records = max_records
        self.buf, self.last = [], time.time()
        self.n_written = 0

    def add(self, recs):
        self.buf.extend(recs)
        if time.time() - self.last >= self.every or len(self.buf) >= self.max_records:
            self.flush()

    def flush(self):
        self.last = time.time()
        if not self.buf:
            return
        name = f"part-{time.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}.jsonl"
        final = os.path.join(self.out_dir, name)
        data = "".join(json.dumps(r) + "\n" for r in self.buf)
        tmp = final + ".tmp"
        try:
            with open(tmp, "w") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, final)
        except OSError:                       # some blob mounts do not support rename
            with open(final, "w") as f:
                f.write(data)
        self.n_written += len(self.buf)
        log(f"[ckpt] flushed {len(self.buf)} records -> {name} (session total {self.n_written})")
        self.buf = []


def load_done(out_dir):
    for t in glob.glob(os.path.join(out_dir, "*.tmp")):
        try:
            os.remove(t)
        except OSError:
            pass
    done = set()
    for p in glob.glob(os.path.join(out_dir, "part-*.jsonl")):
        with open(p) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["item_id"])
                except Exception:
                    continue                  # truncated last line from a killed write
    return done


def read_state(out_dir):
    p = os.path.join(out_dir, "state.json")
    try:
        return json.load(open(p))
    except Exception:
        return {}


def write_state(out_dir, st):
    p = os.path.join(out_dir, "state.json")
    with open(p, "w") as f:
        json.dump(st, f)


# ----------------------------------------------------------------------------- model
def load_model(model_dir, quant, dtype_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    dtype = getattr(torch, dtype_name)
    kw = dict(device_map={"": 0}, low_cpu_mem_usage=True, attn_implementation="sdpa")
    if quant == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype)
        kw["torch_dtype"] = dtype
    else:
        kw["torch_dtype"] = dtype
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kw)
    except (ValueError, KeyError) as e:     # e.g. Gemma 3 4B/12B multimodal checkpoints
        log("AutoModelForCausalLM failed, trying ImageTextToText:", repr(e)[:200])
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_dir, **kw)
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok, dtype


def per_token_bytes(model, dtype):
    cfg = getattr(model.config, "text_config", None) or model.config
    L = cfg.num_hidden_layers
    heads = cfg.num_attention_heads
    kvh = getattr(cfg, "num_key_value_heads", None) or heads
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // heads
    elt = torch.finfo(dtype).bits // 8
    kv = 2 * L * kvh * hd * elt
    act = (2 * getattr(cfg, "intermediate_size", 4 * cfg.hidden_size) + 6 * cfg.hidden_size) * elt
    return kv + act


class AnswerStop:
    """Per-sequence stop once a full 'Answer: X X X X X' line has been generated."""

    def __init__(self, tok, start_len):
        self.tok, self.start = tok, start_len

    def __call__(self, input_ids, scores, **kw):
        tail = input_ids[:, max(self.start, input_ids.shape[1] - 24):]
        texts = self.tok.batch_decode(tail, skip_special_tokens=True)
        return torch.tensor([bool(ANS_TAIL.search(t)) for t in texts], device=input_ids.device)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_key", required=True)
    ap.add_argument("--quant", default="none")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--max_bs", type=int, default=64)
    ap.add_argument("--mem_fraction", type=float, default=0.70)
    ap.add_argument("--ckpt_minutes", type=float, default=15)
    ap.add_argument("--max_passes", type=int, default=3)
    a = ap.parse_args()

    from transformers import StoppingCriteriaList
    os.makedirs(a.out_dir, exist_ok=True)
    run_id = os.environ.get("AZUREML_RUN_ID", "local") + "-" + uuid.uuid4().hex[:6]
    pf = a.prompts if a.prompts.endswith(".parquet") else glob.glob(os.path.join(a.prompts, "*.parquet"))[0]
    prompts = pd.read_parquet(pf)
    all_ids = set(prompts.item_id)
    assert len(all_ids) == len(prompts), "duplicate item_id in prompt file"

    if os.path.exists(os.path.join(a.out_dir, "_DONE")) and load_done(a.out_dir) >= all_ids:
        log("already complete (_DONE present)"); return

    writer = ShardWriter(a.out_dir, run_id, a.ckpt_minutes)

    def on_term(signum, frame):
        log(f"signal {signum}: flushing before exit")
        writer.flush()
        sys.exit(143)
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    model, tok, dtype = load_model(a.model_dir, a.quant, a.dtype)
    free, total = torch.cuda.mem_get_info()
    ptb = per_token_bytes(model, dtype)
    init_budget = max(512, int(free * a.mem_fraction / ptb))
    st = read_state(a.out_dir)
    budget = min(st.get("token_budget", init_budget), init_budget)
    log(f"model={a.model_key} free={free/2**30:.1f}GB bytes/token={ptb} budget={budget} tokens")

    for pas in range(a.max_passes):
        done = load_done(a.out_dir) | {r["item_id"] for r in writer.buf}
        todo = prompts[~prompts.item_id.isin(done)].copy()
        log(f"pass {pas}: done={len(done)} todo={len(todo)} / {len(all_ids)}")
        if todo.empty:
            break
        todo["cost"] = todo.prompt_len + todo.max_new
        todo = todo.sort_values(["cost", "item_id"]).reset_index(drop=True)
        i, streak = 0, 0
        t0 = time.time()
        while i < len(todo):
            # form batch under token budget (sorted -> last row has max length)
            bs = 1
            while (i + bs < len(todo) and bs < a.max_bs and
                   (bs + 1) * (todo.prompt_len[i + bs] + todo.max_new[i + bs]) <= budget):
                bs += 1
            rows = todo.iloc[i:i + bs]
            ids = [json.loads(x) for x in rows.input_ids]
            L = max(map(len, ids))
            pad = tok.pad_token_id
            inp = torch.tensor([[pad] * (L - len(x)) + x for x in ids], device="cuda")
            att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in ids], device="cuda")
            max_new = int(rows.max_new.max())
            try:
                with torch.inference_mode():
                    out = model.generate(
                        input_ids=inp, attention_mask=att, max_new_tokens=max_new,
                        do_sample=False, temperature=None, top_p=None, top_k=None,
                        pad_token_id=pad, use_cache=True,
                        stopping_criteria=StoppingCriteriaList([AnswerStop(tok, L)]))
            except torch.cuda.OutOfMemoryError:
                del inp, att
                gc.collect(); torch.cuda.empty_cache()
                if bs == 1:
                    log(f"OOM at batch size 1 for {rows.item_id.iloc[0]}; recording error")
                    writer.add([dict(item_id=rows.item_id.iloc[0], text="", n_gen=0,
                                     stopped_by_answer=False, batch_size=1, error="oom_bs1",
                                     run_id=run_id)])
                    i += 1
                    continue
                budget = max(256, budget // 2)
                st["token_budget"] = budget; write_state(a.out_dir, st)
                log(f"OOM with bs={bs}: budget -> {budget}; retrying same items")
                streak = 0
                continue
            gen = out[:, L:].cpu()
            assert gen.shape[0] == bs, "output count != batch size"
            texts = tok.batch_decode(gen, skip_special_tokens=True)
            recs = []
            for r, t, g in zip(rows.itertuples(), texts, gen):
                n_gen = int((g != pad).sum())
                recs.append(dict(item_id=r.item_id, text=t, n_gen=n_gen,
                                 stopped_by_answer=bool(ANS_TAIL.search(t)),
                                 batch_size=bs, run_id=run_id))
            assert len(recs) == bs
            writer.add(recs)
            i += bs
            streak += 1
            if streak >= 20 and budget < init_budget:      # cautious recovery
                budget = min(init_budget, int(budget * 1.15)); streak = 0
            if i % 200 < bs:
                rate = i / max(1e-6, time.time() - t0)
                log(f"  {i}/{len(todo)} items  bs={bs}  {rate:.2f} items/s  "
                    f"peak={torch.cuda.max_memory_allocated()/2**30:.1f}GB")
                with open(os.path.join(a.out_dir, "progress.json"), "w") as f:
                    json.dump(dict(done=len(done) + i, total=len(all_ids), budget=budget,
                                   run_id=run_id, ts=time.time()), f)
        writer.flush()

    writer.flush()
    done = load_done(a.out_dir)
    missing = all_ids - done
    if missing:
        log(f"INCOMPLETE: {len(missing)} items missing -> exit 1 for resubmission")
        sys.exit(1)
    with open(os.path.join(a.out_dir, "_DONE"), "w") as f:
        json.dump(dict(n=len(all_ids), run_id=run_id, ts=time.time()), f)
    log(f"COMPLETE: {len(all_ids)} items")


if __name__ == "__main__":
    main()
