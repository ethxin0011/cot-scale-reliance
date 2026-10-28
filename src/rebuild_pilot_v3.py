"""Pilot v3: regenerate problems (4 boxes) + items with s5_task v3, archive old outputs, clear blob results.
Run from src/:  python rebuild_pilot_v3.py [--tag pilot]
Keeps the same pids / k values / count; only problem contents change (seeded, reproducible)."""
import argparse, random, shutil, time, zlib
import pandas as pd
import config, storage, s5_task

ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="pilot"); a = ap.parse_args()
tag = a.tag
assert getattr(s5_task, "NBOX", 5) == 4, "s5_task.py is not v3: upload the patched file first"
d = config.data_dir(tag)

# ---- 1. problems (same pid/k; new 4-box contents)
pp = d / "problems.parquet"
old = pd.read_parquet(pp)
v = 1
while (d / f"problems_v{v}.parquet").exists():
    v += 1
shutil.copy(pp, d / f"problems_v{v}.parquet")
rows = []
for pid, k in zip(old.pid, old.k):
    rng = random.Random(zlib.crc32(f"{tag}-v3-{pid}".encode()))
    rows.append(s5_task.make_problem(rng, int(k), pid))
new = pd.DataFrame(rows)
extra = [c for c in old.columns if c not in new.columns]
new = new.merge(old[["pid"] + extra], on="pid", how="left") if extra else new
new = new[[c for c in old.columns if c in new.columns] + [c for c in new.columns if c not in old.columns]]
new.to_parquet(pp, index=False)
print("problems regenerated:", len(new), "| k counts:", new.k.value_counts().sort_index().to_dict())

# ---- 2. items
ip = d / "items.parquet"
old_items = pd.read_parquet(ip)
shutil.copy(ip, d / f"items_v{v}.parquet")
items = s5_task.build_items(new)
cols = ["user_text", "assistant_prefix", "gold_completion"]
out = old_items.drop(columns=[c for c in cols if c in old_items.columns]).merge(
    items[["item_id"] + cols], on="item_id", how="left")
assert len(out) == len(old_items) and out[cols].notna().all().all(), "item mismatch"
out[list(old_items.columns)].to_parquet(ip, index=False)
print("items rebuilt:", len(out), "| question sample:\n" + out.user_text.iloc[0].split("Now solve this problem.")[-1])

# ---- 3. archive local outputs
arch = config.DATA_ROOT / f"{tag}_archive"
arch.mkdir(exist_ok=True)
for sub in ("prompts", "raw", "scored"):
    if (d / sub).exists():
        dst = arch / f"{sub}_{time.strftime('%m%d_%H%M%S')}"
        shutil.move(str(d / sub), str(dst)); print("archived", sub, "->", dst.name)

# ---- 4. delete blob results + _DONE markers
pref = config.blob_prompts(tag, "X").rsplit("/prompts/", 1)[0] + "/results/"
names = list(storage.list_blobs(pref))
for n in names:
    storage.container().delete_blob(n)
print(f"deleted {len(names)} blobs under {pref}")
print("DONE")
