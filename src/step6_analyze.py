"""STEP 6 (CPU, 48 cores): fit the CoT-reliance scaling law, test H1-H3, held-out prediction.

Definitions (per model):
  k_d   = number of swaps at which DIRECT (no-CoT) accuracy falls to 50%  (logistic fit in log k)
  eligible edit item = clean CoT trace correct through step m  AND  fresh-continuation correct
  R     = 1 if the edited trace's answer follows the edit (y'), else 0   (invalid excluded)

Models (logistic, cluster-robust SE by problem id, all with is_base control):
  H1 : R ~ x                         where x = log k - log k_d   (relative difficulty only)
  H2 : R ~ x + log N + log D         (size / data matter beyond relative difficulty)
  C  : R ~ x + log(6 N D)            (compute-only; H3 says this fits as well as H2)
  ND : R ~ log k + log N + log D     (no k_d; pure scaling-variable model)
Out-of-sample: fit on split=='fit' (Qwen3 + OLMo 2), predict split=='heldout' (Gemma 3, Llama 3);
and predict the final OLMo checkpoints from intermediate ones.
Usage: python step6_analyze.py --tag main [--boot 1000]
"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

from config import DATA_ROOT, K_GRID, data_dir

warnings.filterwarnings("ignore")
FORMULAS = {
    "H1": ["x", "is_base"],
    "H2": ["x", "logN", "logD", "is_base"],
    "C": ["x", "logC", "is_base"],
    "ND": ["logk", "logN", "logD", "is_base"],
}


# ----------------------------------------------------------------------------- k_d per model
def fit_kd(direct):
    """Penalised logistic of direct correctness on log k; returns (k_d, censored, slope)."""
    x = np.log(direct.k.values.astype(float)); y = direct.correct.values.astype(float)
    if y.mean() == 0:
        return 0.5, True, np.nan
    if y.mean() == 1:
        return 200.0, True, np.nan

    def nll(w):
        z = w[0] + w[1] * x
        return np.sum(np.logaddexp(0, z) - y * z) + 1e-3 * (w ** 2).sum()
    w = minimize(nll, [0.0, -1.0], method="BFGS").x
    if w[1] >= 0:
        return 200.0, True, w[1]
    kd = float(np.exp(-w[0] / w[1]))
    cens = not (min(K_GRID) <= kd <= max(K_GRID))
    return float(np.clip(kd, 0.5, 200)), cens, float(w[1])


# ----------------------------------------------------------------------------- data assembly
def assemble(tag):
    man = json.loads((DATA_ROOT / "manifest.json").read_text())
    cov = pd.read_csv(data_dir(tag) / "coverage.csv").set_index("model")
    rows, mstats = [], []
    for key, e in man.items():
        p = data_dir(tag) / "scored" / f"{key}.parquet"
        if not p.exists():
            continue
        if key not in cov.index or cov.at[key, "coverage"] < 1.0:
            print(f"[exclude] {key}: incomplete coverage"); continue
        d = pd.read_parquet(p)
        kd, cens, slope = fit_kd(d[d.cond == "direct"])
        cot = d[d.cond == "cot"].set_index("pid"); fr = d[d.cond == "fresh"].set_index("pid")
        ed = d[d.cond == "edit"].copy()
        ed["eligible"] = ed.pid.map(cot.prefix_ok).fillna(False).astype(bool) & \
            ed.pid.map(fr.correct).fillna(False).astype(bool)
        ed = ed[ed.eligible & (ed.outcome != "invalid")]
        ed["R"] = (ed.outcome == "follows_edit").astype(int)
        for c in ["family", "kind", "split", "N", "D", "quant"]:
            ed[c] = e[c]
        ed["model"] = key
        ed["kd"] = kd
        ed["ckpt"] = e.get("ckpt", "final")
        rows.append(ed)
        mstats.append(dict(model=key, family=e["family"], kind=e["kind"], split=e["split"],
                           N=e["N"], D=e["D"], quant=e["quant"], ckpt=e.get("ckpt", "final"),
                           kd=kd, kd_censored=cens, direct_slope=slope,
                           cot_acc=d[d.cond == "cot"].correct.mean(),
                           direct_acc=d[d.cond == "direct"].correct.mean(),
                           n_eligible=len(ed), R=ed.R.mean() if len(ed) else np.nan))
    df = pd.concat(rows, ignore_index=True)
    df["logk"] = np.log(df.k); df["logN"] = np.log(df.N); df["logD"] = np.log(df.D)
    df["logC"] = np.log(6 * df.N * df.D); df["x"] = df.logk - np.log(df.kd)
    df["is_base"] = (df.kind == "base").astype(int)
    return df, pd.DataFrame(mstats)


# ----------------------------------------------------------------------------- fitting
def fit(df, cols, robust=True):
    import statsmodels.api as sm
    X = sm.add_constant(df[cols].astype(float), has_constant="add")
    if df[cols].nunique().min() <= 1:                       # drop constant columns (e.g. pilot)
        keep = [c for c in cols if df[c].nunique() > 1]
        X = sm.add_constant(df[keep].astype(float), has_constant="add")
    m = sm.Logit(df.R.values, X)
    try:
        r = m.fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(df.pid)[0]}) if robust \
            else m.fit(disp=0)
    except Exception:
        r = m.fit_regularized(disp=0, alpha=1e-4)
    return r


def predict(res, df):
    import statsmodels.api as sm
    cols = [c for c in res.params.index if c != "const"]
    X = sm.add_constant(df[cols].astype(float), has_constant="add")
    return np.clip(res.predict(X), 1e-6, 1 - 1e-6)


def scores(y, p, df):
    ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    br = np.mean((p - y) ** 2)
    cell = pd.DataFrame(dict(m=df.model.values, k=df.k.values, y=y, p=p)).groupby(["m", "k"]).mean()
    return dict(logloss=ll, brier=br, cell_mae=float((cell.y - cell.p).abs().mean()), n=len(y))


def boot_one(df, b, cols):
    rng = np.random.default_rng(b)
    pids = df.pid.unique()
    pick = rng.choice(pids, len(pids), replace=True)
    cnt = pd.Series(pick).value_counts()
    idx = np.concatenate([np.repeat(df.index[df.pid == p].values, c) for p, c in cnt.items()])
    try:
        return fit(df.loc[idx], cols, robust=False).params.to_dict()
    except Exception:
        return None


# ----------------------------------------------------------------------------- figures
def figures(df, ms, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fam_c = {"qwen3": "tab:blue", "olmo2": "tab:green", "gemma3": "tab:red", "llama3": "tab:purple"}
    cell = df.groupby(["model", "family", "k", "kd"]).R.mean().reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for mdl, g in cell.groupby("model"):
        c = fam_c.get(g.family.iloc[0], "gray")
        ax[0].plot(g.k, g.R, "-o", color=c, alpha=.6, ms=3)
        ax[1].plot(g.k / g.kd, g.R, "-o", color=c, alpha=.6, ms=3)
    ax[0].set_xscale("log"); ax[1].set_xscale("log")
    ax[0].set_xlabel("k (swaps)"); ax[1].set_xlabel("k / k_d (relative difficulty)")
    for a in ax:
        a.set_ylabel("CoT reliance R"); a.set_ylim(-.02, 1.02)
    ax[0].set_title("Raw"); ax[1].set_title("Collapse test (H1)")
    for f, c in fam_c.items():
        ax[1].plot([], [], color=c, label=f)
    ax[1].legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "fig1_reliance_collapse.png", dpi=160)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for f, g in ms.groupby("family"):
        ax[0].scatter(g.N, g.kd, color=fam_c.get(f, "gray"), label=f)
        ax[1].scatter(g.D, g.kd, color=fam_c.get(f, "gray"))
    for a, lab in zip(ax, ["N (non-embedding params)", "D (pretraining tokens)"]):
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel(lab); a.set_ylabel("k_d")
    ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "fig2_kd_scaling.png", dpi=160)
    plt.close("all")


def outcome_figure(tag, models, out):
    import matplotlib.pyplot as plt
    cats = ["follows_edit", "repair", "dissociation", "retained_unexplained", "wrong_other", "invalid"]
    rows = []
    for m in models:
        d = pd.read_parquet(data_dir(tag) / "scored" / f"{m}.parquet")
        v = d[d.cond == "edit"].outcome.value_counts(normalize=True)
        rows.append({c: v.get(c, 0) for c in cats} | {"model": m})
    t = pd.DataFrame(rows).set_index("model")[cats]
    ax = t.plot.bar(stacked=True, figsize=(12, 4), width=.85)
    ax.set_ylabel("share of edited traces"); ax.legend(fontsize=7, ncol=3)
    plt.tight_layout(); plt.savefig(out / "fig3_outcomes.png", dpi=160); plt.close("all")
    t.to_csv(out / "outcome_decomposition.csv")
    return t


# ----------------------------------------------------------------------------- main
def main(tag, n_boot, workers):
    from joblib import Parallel, delayed
    out = data_dir(tag) / "analysis"; out.mkdir(exist_ok=True)
    df, ms = assemble(tag)
    ms.to_csv(out / "model_stats.csv", index=False)
    fitd = df[df.split == "fit"]; held = df[df.split == "heldout"]
    lines = [f"# CoT-reliance scaling law — results ({tag})\n",
             f"Eligible edit items: fit={len(fitd)}, heldout={len(held)}; models={df.model.nunique()}\n"]

    res = {}
    for name, cols in FORMULAS.items():
        res[name] = fit(fitd, cols)
        lines.append(f"\n## {name}: R ~ {' + '.join(cols)}\n```\n{res[name].summary2().tables[1].round(4).to_string()}\n```")
        lines.append(f"AIC={res[name].aic:.1f}  BIC={res[name].bic:.1f}  pseudo-R2={res[name].prsquared:.3f}")

    # H2 vs H1 likelihood-ratio test (non-robust likelihoods)
    l1 = fit(fitd, FORMULAS["H1"], robust=False).llf; l2 = fit(fitd, FORMULAS["H2"], robust=False).llf
    lr = 2 * (l2 - l1); p = chi2.sf(lr, 2)
    lines.append(f"\n## Hypothesis tests\n- H2 vs H1 LR={lr:.2f}, df=2, p={p:.3g} "
                 f"({'size/data matter beyond relative difficulty' if p < 0.05 else 'no evidence beyond relative difficulty (H1 holds)'})")
    lines.append(f"- H3 (compute-only): AIC(C)-AIC(H2) = {res['C'].aic - res['H2'].aic:.1f} "
                 f"({'compute-only adequate' if res['C'].aic - res['H2'].aic < 2 else 'N and D act separately'})")

    # bootstrap CIs for H2
    if n_boot > 0:
        bs = Parallel(n_jobs=workers)(delayed(boot_one)(fitd.reset_index(drop=True), b, FORMULAS["H2"])
                                      for b in range(n_boot))
        bs = pd.DataFrame([b for b in bs if b])
        ci = bs.quantile([0.025, 0.975]).T
        ci.columns = ["lo95", "hi95"]
        ci.to_csv(out / "h2_bootstrap_ci.csv")
        lines.append(f"\n## H2 bootstrap 95% CI over problems (B={len(bs)})\n```\n{ci.round(4).to_string()}\n```")

    # held-out family prediction
    if len(held):
        lines.append("\n## Held-out families (Gemma 3, Llama 3) — fit on Qwen3+OLMo 2 only")
        tab = []
        base = np.full(len(held), fitd.R.mean())
        tab.append(dict(model="constant", **scores(held.R.values, base, held)))
        for name in FORMULAS:
            tab.append(dict(model=name, **scores(held.R.values, predict(res[name], held), held)))
        t = pd.DataFrame(tab); t.to_csv(out / "heldout_prediction.csv", index=False)
        lines.append("```\n" + t.round(4).to_string(index=False) + "\n```")

    # OLMo: predict final checkpoint from intermediate ones
    olmo = fitd[fitd.family == "olmo2"]
    if (olmo.ckpt == "final").any() and (olmo.ckpt == "intermediate").any():
        tr = fitd[~((fitd.family == "olmo2") & (fitd.ckpt == "final"))]
        te = olmo[olmo.ckpt == "final"]
        tab = [dict(model=n, **scores(te.R.values, predict(fit(tr, c), te), te)) for n, c in FORMULAS.items()]
        t = pd.DataFrame(tab); t.to_csv(out / "olmo_final_prediction.csv", index=False)
        lines.append("\n## Predict final OLMo 2 checkpoints from intermediate ones\n```\n"
                     + t.round(4).to_string(index=False) + "\n```")

    # k_d law
    import statsmodels.formula.api as smf
    kd = ms[(ms.split == "fit") & ~ms.kd_censored].copy()
    if len(kd) >= 4:
        kd["logkd"] = np.log(kd.kd); kd["logN"] = np.log(kd.N); kd["logD"] = np.log(kd.D)
        kd["is_base"] = (kd.kind == "base").astype(int)
        r = smf.ols("logkd ~ logN + logD + is_base", kd).fit(cov_type="HC3")
        lines.append(f"\n## k_d law: log k_d = log c + a log N + b log D (+ base)\n```\n{r.summary2().tables[1].round(4).to_string()}\n```")

    # quantisation control
    q = ms[ms.model.isin(["qwen3-4b", "qwen3-4b-nf4"])]
    if len(q) == 2:
        lines.append("\n## Quantisation control (Qwen3-4B fp16 vs nf4)\n```\n"
                     + q[["model", "kd", "R", "cot_acc", "direct_acc"]].round(3).to_string(index=False) + "\n```")

    figures(df, ms, out)
    oc = outcome_figure(tag, list(ms.model), out)
    lines.append("\n## Edit-outcome decomposition (all edited traces)\n```\n" + oc.round(3).to_string() + "\n```")
    lines.append("\n## Per-model statistics\n```\n" + ms.round(3).to_string(index=False) + "\n```")
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nall outputs in {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main", choices=["pilot", "main"])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=44)
    a = ap.parse_args()
    main(a.tag, a.boot, a.workers)
