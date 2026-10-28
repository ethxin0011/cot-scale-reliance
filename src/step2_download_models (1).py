"""STEP 2 (CPU compute instance): download model weights from Hugging Face and stage them in blob.

GPU jobs then mount the staged folder (no HF download on the expensive/preemptible GPU node).
Resume-safe: a model with blob marker `_UPLOAD_OK` is skipped.

Usage: python step2_download_models.py --models all|pilot|<key,key> [--workers 3]
Needs env HF_TOKEN (Gemma 3 and Llama are gated: accept their licenses on huggingface.co first).
Writes data/manifest.json with every expanded model entry (incl. OLMo checkpoints, N, D).
"""
import argparse
import json
import os
import re
import shutil
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import (DATA_ROOT, LOCAL_SCRATCH, MODELS, OLMO_CKPT_QUANTILES, PILOT_MODELS,
                    blob_models)

IGNORE = ["original/*", "*.pth", "*.gguf", "*.onnx", "*.msgpack", "*.h5", "*.ot", "*.tflite"]
EXCLUDE_PARAM = re.compile(r"(embed_tokens|lm_head|wte|wpe|vision|multi_modal_projector|siglip)")


# ----------------------------------------------------------------------------- OLMo checkpoint expansion
def olmo_revisions(repo, token):
    from huggingface_hub import list_repo_refs
    refs = list_repo_refs(repo, token=token)
    cands = []
    for b in refs.branches:
        if not b.name.startswith("stage1-"):
            continue
        mt = re.search(r"tokens(\d+)B", b.name)
        if mt:
            cands.append((int(mt.group(1)), b.name))
    cands.sort()
    if not cands:
        raise RuntimeError(f"no stage1 revisions found for {repo}")
    picked = []
    for q in OLMO_CKPT_QUANTILES:
        tok, name = cands[min(len(cands) - 1, int(q * (len(cands) - 1)))]
        if name not in [p[1] for p in picked]:
            picked.append((tok, name))
    return picked


def expand_registry(token):
    out = []
    for m in MODELS:
        if m["rev"] != "AUTO":
            out.append(dict(m))
            continue
        for tok, name in olmo_revisions(m["hf"], token):
            e = dict(m, key=f"{m['key']}-{tok}B", rev=name, D=tok * 1e9, ckpt="intermediate")
            out.append(e)
        out.append(dict(m, key=f"{m['key']}-main", rev="main", ckpt="final"))
    return out


# ----------------------------------------------------------------------------- param count from safetensors headers
def count_params(folder):
    total = non_emb = 0
    for f in Path(folder).rglob("*.safetensors"):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            c = 1
            for d in meta["shape"]:
                c *= d
            total += c
            if not EXCLUDE_PARAM.search(name):
                non_emb += c
    return total, non_emb


# ----------------------------------------------------------------------------- one model
def stage(entry, token):
    import storage
    from huggingface_hub import list_repo_files, snapshot_download
    key = entry["key"]
    marker = f"{blob_models(key)}/_UPLOAD_OK"
    if storage.exists(marker):
        info = json.loads(storage.read_text(marker))
        return dict(entry, **info, status="cached")
    files = list_repo_files(entry["hf"], revision=entry["rev"], token=token)
    ignore = list(IGNORE)
    if any(f.endswith(".safetensors") for f in files):
        ignore += ["*.bin"]
    local = LOCAL_SCRATCH / "models" / key
    snapshot_download(entry["hf"], revision=entry["rev"], local_dir=str(local),
                      token=token, ignore_patterns=ignore, max_workers=8)
    total, non_emb = count_params(local)
    if total == 0:
        raise RuntimeError(f"{key}: no safetensors found; cannot count params")
    n_files = storage.upload_dir(local, blob_models(key))
    info = dict(N_total=total, N=non_emb, n_files=n_files)
    storage.upload_bytes(json.dumps(info).encode(), marker)   # marker LAST => upload complete
    shutil.rmtree(local, ignore_errors=True)
    return dict(entry, **info, status="uploaded")


def main(which, workers):
    token = os.environ.get("HF_TOKEN")
    reg = expand_registry(token)
    if which == "pilot":
        reg = [m for m in reg if m["key"] in PILOT_MODELS]
    elif which != "all":
        keys = set(which.split(","))
        reg = [m for m in reg if m["key"] in keys]
    LOCAL_SCRATCH.mkdir(parents=True, exist_ok=True)
    man_path = DATA_ROOT / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(stage, m, token): m["key"] for m in reg}
        for f in as_completed(futs):
            k = futs[f]
            try:
                e = f.result()
                manifest[k] = e
                print(f"[ok] {k:24s} {e['status']:8s} N={e['N']/1e9:.2f}B D={e['D']/1e12:.2f}T")
            except Exception as ex_:  # keep going; re-run the cell to retry failures
                print(f"[FAIL] {k}: {ex_!r}")
            DATA_ROOT.mkdir(parents=True, exist_ok=True)
            man_path.write_text(json.dumps(manifest, indent=1))
    print(f"manifest: {len(manifest)} models -> {man_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    main(a.models, a.workers)
