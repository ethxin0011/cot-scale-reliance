"""STEP 5 (CPU): download generations, check coverage, parse, classify, GPT fallback + audit.

Usage:
  python step5_score.py --tag pilot --gate          # pilot go/no-go report
  python step5_score.py --tag main [--no_gpt]
Outputs: data/<tag>/scored/<key>.parquet, data/<tag>/coverage.csv, data/<tag>/parser_audit.csv
"""
import argparse
import glob
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from config import (AUDIT_SAMPLE_PER_MODEL, DATA_ROOT, GATE_MIN_EDIT_ELIGIBLE,
                    GATE_MIN_FORMAT_COMPLIANCE, PILOT_MODELS, SEED, blob_results, data_dir)
from s5_task import classify_edit, first_post_edit_path, parse_output


def fetch(tag, key):
    import storage
    local = data_dir(tag) / "raw" / key
    local.mkdir(parents=True, exist_ok=True)
    storage.download_prefix(blob_results(tag, key) + "/", local, suffix=".jsonl")
    recs = []
    for p in sorted(glob.glob(str(local / "part-*.jsonl"))):
        for line in open(p):
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df["is_err"] = df.get("error", pd.Series([None] * len(df))).notna()
    return df.sort_values("is_err").drop_duplicates("item_id", keep="first")


def score_frame(gen, items, probs):
    P = probs.set_index("pid")
    rows = []
    for r in items.merge(gen[["item_id", "text", "n_gen", "is_err"]], on="item_id", how="left").itertuples():
        pr = P.loc[r.pid]
        text = r.text if isinstance(r.text, str) else ""
        ps = parse_output(text)
        states = json.loads(pr.states)
        row = dict(item_id=r.item_id, pid=r.pid, k=r.k, m=r.m, cond=r.cond,
                   has_output=isinstance(r.text, str), n_gen=r.n_gen, answer=ps["answer"],
                   step_k=ps["steps"].get(r.k), n_steps=ps["n_steps"], llm_parsed=False, text=text)
        if r.cond == "cot":
            row["prefix_ok"] = all(ps["steps"].get(t) == states[t - 1] for t in range(1, r.m + 1))
        rows.append(row)
    return pd.DataFrame(rows)


def finalize(df, probs):
    P = probs.set_index("pid")
    y = df.pid.map(P.y); ye = df.pid.map(P.y_edit)
    target = y.where(df.cond.isin(["cot", "direct"]), ye)
    df["compliant"] = df.answer.notna()
    df["correct"] = (df.answer == target) & df.compliant
    oc, path = [], []
    for r in df.itertuples():
        if r.cond == "edit":
            pr = P.loc[r.pid].to_dict(); pr["k"] = r.k; pr["m"] = r.m
            parsed = dict(answer=r.answer, steps={r.k: r.step_k} if r.step_k else {})
            oc.append(classify_edit(parsed, pr))
            path.append(first_post_edit_path(parse_output(r.text), pr))
        else:
            oc.append(None); path.append(None)
    df["outcome"] = oc
    df["post_edit_path"] = path
    return df


def score_model(tag, key, use_gpt):
    items = pd.read_parquet(data_dir(tag) / "items.parquet")
    probs = pd.read_parquet(data_dir(tag) / "problems.parquet")
    gen = fetch(tag, key)
    if gen.empty:
        return key, None, dict(model=key, n_expected=len(items), n_got=0, coverage=0.0)
    cov = dict(model=key, n_expected=len(items), n_got=int(gen.item_id.isin(items.item_id).sum()),
               n_err=int(gen.is_err.sum()))
    cov["coverage"] = cov["n_got"] / cov["n_expected"]
    df = score_frame(gen, items, probs)
    cov["regex_compliance_cot"] = float(df[df.cond == "cot"].answer.notna().mean())
    audit = None
    if use_gpt:
        import llm_assist
        miss = df[df.answer.isna() & (df.text.str.len() > 0)]
        if len(miss):
            res = llm_assist.extract_many(list(miss.text), list(miss.k))
            for idx, r in zip(miss.index, res):
                if r.get("final_answer"):
                    df.at[idx, "answer"] = r["final_answer"]; df.at[idx, "llm_parsed"] = True
                if r.get("step_k") and not df.at[idx, "step_k"]:
                    df.at[idx, "step_k"] = r["step_k"]
        ok = df[df.answer.notna() & ~df.llm_parsed]
        smp = ok.sample(min(AUDIT_SAMPLE_PER_MODEL, len(ok)), random_state=SEED)
        res = llm_assist.extract_many(list(smp.text), list(smp.k))
        audit = pd.DataFrame(dict(model=key, item_id=smp.item_id.values,
                                  regex=smp.answer.values,
                                  gpt=[r.get("final_answer") for r in res]))
        audit["agree"] = audit.regex == audit.gpt
        cov["audit_agree"] = float(audit.agree.mean())
    df = finalize(df, probs)
    out = data_dir(tag) / "scored"; out.mkdir(exist_ok=True)
    df.drop(columns=["text"]).to_parquet(out / f"{key}.parquet", index=False)
    return key, audit, cov


def gate_report(tag, keys):
    print("\n==== PILOT GATE ====")
    ok_all = True
    for k in keys:
        p = data_dir(tag) / "scored" / f"{k}.parquet"
        if not p.exists():
            print(f"{k}: not scored"); ok_all = False; continue
        d = pd.read_parquet(p)
        cot = d[d.cond == "cot"]; dr = d[d.cond == "direct"]
        comp = cot.compliant.mean()
        elig = cot.prefix_ok.mean()
        acc_c = cot.groupby("k").correct.mean().round(2).to_dict()
        acc_d = dr.groupby("k").correct.mean().round(2).to_dict()
        oc = d[d.cond == "edit"].outcome.value_counts(normalize=True).round(2).to_dict()
        flag = comp >= GATE_MIN_FORMAT_COMPLIANCE and elig >= GATE_MIN_EDIT_ELIGIBLE
        ok_all &= flag
        print(f"\n{k}: {'PASS' if flag else 'CHECK'}  compliance={comp:.2f}  prefix_ok={elig:.2f}")
        print("   CoT acc by k   :", acc_c)
        print("   direct acc by k:", acc_d)
        print("   edit outcomes  :", oc)
    print("\nGATE:", "PASS - run main" if ok_all else
          "CHECK - inspect failing models (prompt format / dtype) before main run")


def main(tag, which, use_gpt, gate, workers):
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text())
    keys = list(manifest)
    if which == "pilot" or (which == "all" and tag == "pilot"):
        keys = [k for k in keys if k in PILOT_MODELS]
    elif which != "all":
        keys = which.split(",")
    covs, audits = [], []
    with ProcessPoolExecutor(workers) as ex:
        for key, audit, cov in ex.map(score_model, [tag] * len(keys), keys, [use_gpt] * len(keys)):
            covs.append(cov)
            if audit is not None:
                audits.append(audit)
            print(f"[scored] {key}: coverage={cov['coverage']:.3f} "
                  f"compliance={cov.get('regex_compliance_cot', float('nan')):.3f} "
                  f"audit={cov.get('audit_agree', float('nan')):.3f}")
    pd.DataFrame(covs).to_csv(data_dir(tag) / "coverage.csv", index=False)
    if audits:
        pd.concat(audits).to_csv(data_dir(tag) / "parser_audit.csv", index=False)
    if gate:
        gate_report(tag, keys)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main", choices=["pilot", "main"])
    ap.add_argument("--models", default="all")
    ap.add_argument("--no_gpt", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    main(a.tag, a.models, not a.no_gpt, a.gate, a.workers)
