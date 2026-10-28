"""STEP 1 (CPU): generate S5 problems and model-agnostic items for all 4 conditions.

Usage: python step1_build_data.py --tag pilot|main
Outputs: data/<tag>/problems.parquet, data/<tag>/items.parquet
"""
import argparse
import json
import random

import pandas as pd

from config import K_GRID, N_PER_K, PILOT_N_PER_K, SEED, data_dir
from s5_task import build_items, fmt, parse_output


def main(tag):
    n = PILOT_N_PER_K if tag == "pilot" else N_PER_K
    rng = random.Random(SEED + (17 if tag == "pilot" else 0))
    from s5_task import make_problem
    probs = [make_problem(rng, k, f"k{k:02d}-{i:04d}") for k in K_GRID for i in range(n)]
    df = pd.DataFrame(probs)
    items = build_items(df)

    # ---- self-checks: guaranteed y' != y, and the parser round-trips every gold completion
    assert (df.y != df.y_edit).all()
    by_pid = df.set_index("pid")
    bad = 0
    for r in items.itertuples():
        p = parse_output(r.gold_completion)
        gold_ans = by_pid.at[r.pid, "y_edit"] if r.cond in ("edit", "fresh") else by_pid.at[r.pid, "y"]
        if p["answer"] != gold_ans:
            bad += 1
    assert bad == 0, f"parser failed on {bad} gold completions"

    out = data_dir(tag)
    df.to_parquet(out / "problems.parquet", index=False)
    items.to_parquet(out / "items.parquet", index=False)
    print(f"[{tag}] problems={len(df)} items={len(items)} -> {out}")
    print("example edit prompt tail:\n", items[items.cond == 'edit'].iloc[3].assistant_prefix)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main", choices=["pilot", "main"])
    main(ap.parse_args().tag)
