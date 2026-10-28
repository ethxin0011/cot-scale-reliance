"""Per-model step accuracy + 2 raw samples. Run from src/: python diag_v3.py [tag]"""
import glob, json, os, sys
sys.path.insert(0, os.getcwd())
import pandas as pd
import config, s5_task
tag = sys.argv[1] if len(sys.argv) > 1 else "pilot"
D = config.data_dir(tag)
prob = pd.read_parquet(D / "problems.parquet").set_index("pid")
for m in sorted(os.path.basename(p) for p in glob.glob(str(D / "raw" / "*"))):
    R = pd.concat(pd.read_json(f, lines=True) for f in glob.glob(str(D / "raw" / m / "*.jsonl")))
    R["pid"], R["cond"] = R.item_id.str.split("|").str[0], R.item_id.str.split("|").str[1]
    R["k"] = R.pid.map(prob.k)
    c = R[R.cond == "cot"]
    acc = {}
    for t in (1, 2, 3, 4):
        h = [s5_task.parse_output(x)["steps"].get(t) == json.loads(prob.loc[p, "states"])[t - 1]
             for p, x in zip(c.pid, c.text) if len(json.loads(prob.loc[p, "states"])) >= t]
        acc[t] = round(sum(h) / max(len(h), 1), 2)
    print("\n" + "#" * 80 + f"\n{m}  CoT step-t accuracy: {acc}")
    for _, r in c[c.k == 2].head(2).iterrows():
        print(f"--- true={json.loads(prob.loc[r.pid, 'states'])}\n{str(r.text)[:260].rstrip()}")
