"""STEP 3 (CPU, multiprocess): render model-specific prompts (chat template) and token budgets.

All tokenization happens on the CPU so the GPU job only loads ids and generates.
Usage: python step3_render_prompts.py --tag pilot|main [--models pilot|all|k1,k2] [--workers 24]
Outputs: data/<tag>/prompts/<key>.parquet (also uploaded to blob for the GPU job).
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from config import DATA_ROOT, LOCAL_SCRATCH, PILOT_MODELS, blob_models, blob_prompts, data_dir


def _tokenizer_dir(key):
    """Download only tokenizer/config files of the staged model (small)."""
    import storage
    d = LOCAL_SCRATCH / "tok" / key
    if not (d / "tokenizer_config.json").exists():
        d.mkdir(parents=True, exist_ok=True)
        for n in storage.list_blobs(blob_models(key) + "/"):
            base = n.rsplit("/", 1)[-1]
            if base.endswith((".safetensors", ".bin", "_UPLOAD_OK")) or "/" in n[len(blob_models(key)) + 1:]:
                continue
            with open(d / base, "wb") as f:
                f.write(storage.container().download_blob(n).readall())
    return d


def render(entry, tag):
    from transformers import AutoTokenizer
    key = entry["key"]
    tok = AutoTokenizer.from_pretrained(str(_tokenizer_dir(key)), trust_remote_code=False)
    items = pd.read_parquet(data_dir(tag) / "items.parquet")
    use_chat = entry["kind"] == "chat" and tok.chat_template is not None
    texts = []
    for u, pre in zip(items.user_text, items.assistant_prefix):
        if use_chat:
            kw = {"enable_thinking": False} if entry["family"] == "qwen3" else {}
            t = tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,
                                        add_generation_prompt=True, **kw)
            texts.append(t + pre)
        else:
            texts.append(u + "\n" + pre)
    add_special = not use_chat           # chat template already contains BOS
    enc = tok(texts, add_special_tokens=add_special)["input_ids"]
    gold = tok(list(items.gold_completion), add_special_tokens=False)["input_ids"]
    out = pd.DataFrame(dict(
        item_id=items.item_id, pid=items.pid, k=items.k, cond=items.cond,
        input_ids=[json.dumps(x) for x in enc],
        prompt_len=[len(x) for x in enc],
        max_new=[int(len(g) * 1.5) + 32 for g in gold]))
    out_dir = data_dir(tag) / "prompts"
    out_dir.mkdir(exist_ok=True)
    p = out_dir / f"{key}.parquet"
    out.to_parquet(p, index=False)
    import storage
    storage.upload_file(p, blob_prompts(tag, key))
    return key, len(out), int(out.prompt_len.max()), int(out.max_new.max()), use_chat


def main(tag, which, workers):
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text())
    keys = list(manifest)
    if which == "pilot":
        keys = [k for k in keys if k in PILOT_MODELS]
    elif which != "all":
        keys = which.split(",")
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(render, manifest[k], tag): k for k in keys}
        for f in as_completed(futs):
            try:
                k, n, pl, mn, chat = f.result()
                print(f"[ok] {k:24s} items={n} max_prompt={pl} max_new={mn} chat={chat}")
            except Exception as e:
                print(f"[FAIL] {futs[f]}: {e!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main", choices=["pilot", "main"])
    ap.add_argument("--models", default="all")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    main(a.tag, a.models, a.workers)
