"""step7_paper.py - final scoring summary, statistics, scaling fits and figures for the paper.
Usage:  python step7_paper.py --tag main
Reads   data/<tag>/scored/*.parquet + data/manifest.json
Writes  data/paper/ (CSV tables, PNG figures, paper_numbers.md) + data/paper_kit.zip
"""
import argparse, json, math, shutil, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(0)
NBOOT = 1000
CAP_THRESH = 0.5          # step-1 accuracy needed to count a model as "capable"
CHANCE = 1 / 24           # 4 boxes -> 24 permutations
ABLATION = {"qwen3-4b-nf4"}   # quantization ablation, not on the main curve


def family(k):
    for f in ("qwen3", "gemma3", "llama3", "olmo2"):
        if k.startswith(f): return f
    return "other"


def stage(k):
    if k.startswith("olmo2"): return "base"          # OLMo-2 pretraining checkpoints
    return "base" if k.endswith("-base") else "post-trained"


def boot_ci(df, val):
    """95% bootstrap CI of mean(val), resampling problems (pid). val: 0/1 Series aligned with df."""
    v = pd.Series(val, index=df.index).astype(float)
    m = v.notna()
    if m.sum() == 0: return (np.nan, np.nan)
    g = pd.DataFrame({"pid": df.pid[m].values, "v": v[m].values}).groupby("pid")["v"].agg(["sum", "count"])
    if len(g) < 2: return (np.nan, np.nan)
    idx = RNG.integers(0, len(g), size=(NBOOT, len(g)))
    s, c = g["sum"].values[idx].sum(1), g["count"].values[idx].sum(1)
    b = s / np.maximum(c, 1)
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def fmean(s):
    s = pd.Series(s).dropna().astype(float)
    return float(s.mean()) if len(s) else np.nan


def load(tag):
    man = json.loads((config.DATA_ROOT / "manifest.json").read_text())
    rows = {}
    for f in sorted((config.data_dir(tag) / "scored").glob("*.parquet")):
        S = pd.read_parquet(f)
        cov = fmean(S["has_output"]) if "has_output" in S else 1.0
        if cov < 0.999:
            print(f"[skip] {f.stem}: coverage {cov:.3f} (incomplete)"); continue
        rows[f.stem] = S
    return man, rows


def model_summary(key, S, man):
    cot, dr, ed = S[S.cond == "cot"], S[S.cond == "direct"], S[S.cond == "edit"]
    ok_pid = set(cot[cot.prefix_ok.fillna(False).astype(bool)].pid)
    ed_cond = ed[ed.pid.isin(ok_pid)]
    fe = lambda d: fmean(d.outcome == "follows_edit") if len(d) else np.nan
    m = man.get(key, {})
    r = dict(model=key, family=family(key), stage=stage(key),
             N_B=round(m.get("N", np.nan) / 1e9, 3) if m.get("N") else np.nan,
             D_T=round(m.get("D", np.nan) / 1e12, 2) if m.get("D") else np.nan,
             quant=m.get("quant", ""), n_items=len(S),
             compliance=fmean(S.compliant),
             step1_acc=fmean(cot.prefix_ok),
             cot_acc=fmean(cot.correct), direct_acc=fmean(dr.correct),
             cot_acc_k2=fmean(cot[cot.k == 2].correct), direct_acc_k2=fmean(dr[dr.k == 2].correct),
             cot_acc_kmax=fmean(cot[cot.k == cot.k.max()].correct),
             direct_acc_k4plus=fmean(dr[dr.k >= 4].correct),
             follows_edit=fe(ed), follows_edit_cond=fe(ed_cond), n_edit_cond=len(ed_cond))
    r["cot_gain"] = r["cot_acc"] - r["direct_acc"]
    r["capable"] = bool(r["step1_acc"] >= CAP_THRESH)
    r["follows_edit_lo"], r["follows_edit_hi"] = boot_ci(ed, ed.outcome == "follows_edit")
    r["follows_edit_cond_lo"], r["follows_edit_cond_hi"] = boot_ci(ed_cond, ed_cond.outcome == "follows_edit")
    r["cot_acc_lo"], r["cot_acc_hi"] = boot_ci(cot, cot.correct)
    return r


def by_k(key, S):
    out = []
    for (cond, k), d in S[S.cond.isin(["cot", "direct"])].groupby(["cond", "k"]):
        out.append(dict(model=key, cond=cond, k=int(k), acc=fmean(d.correct), n=len(d)))
    return out


def edit_breakdown(key, S):
    ed = S[S.cond == "edit"]
    vc = ed.outcome.fillna("none").value_counts(normalize=True)
    return dict(model=key, **{f"p_{c}": round(float(v), 4) for c, v in vc.items()})


def fit_saturating(x, y):
    """y = lo + (1-lo) / (1 + exp(-a (log10 N - b)))  ; returns params, R^2"""
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return None
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.log10(x[m]), y[m]
    if len(x) < 4: return None
    f = lambda t, lo, a, b: lo + (1 - lo) / (1 + np.exp(-a * (t - b)))
    try:
        p, _ = curve_fit(f, x, y, p0=[0.1, 5, 9.2], bounds=([0, 0.1, 8], [1, 100, 11]), maxfev=20000)
    except Exception:
        return None
    pred = f(x, *p)
    ss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - pred) ** 2).sum() / ss if ss > 0 else np.nan
    return dict(floor=p[0], slope=p[1], midpoint_N_B=10 ** p[2] / 1e9, r2=r2, n=len(x), f=f, p=p)


def spearman(x, y):
    x, y = pd.Series(x, dtype=float), pd.Series(y, dtype=float)
    m = x.notna() & y.notna()
    return float(x[m].rank().corr(y[m].rank())) if m.sum() >= 3 else np.nan


def main(tag):
    out = config.DATA_ROOT / "paper"; out.mkdir(parents=True, exist_ok=True)
    man, rows = load(tag)
    print(f"complete models: {len(rows)}")

    T = pd.DataFrame([model_summary(k, S, man) for k, S in rows.items()])
    T = T.sort_values(["family", "stage", "N_B", "model"]).reset_index(drop=True)
    T.to_csv(out / "table_models.csv", index=False)
    K = pd.DataFrame([r for k, S in rows.items() for r in by_k(k, S)])
    K.to_csv(out / "table_acc_by_k.csv", index=False)
    E = pd.DataFrame([edit_breakdown(k, S) for k, S in rows.items()]).fillna(0)
    E.to_csv(out / "table_edit_outcomes.csv", index=False)

    # base vs post-trained (Qwen3 pairs)
    q = T[(T.family == "qwen3") & (~T.model.isin(ABLATION))]
    pairs = []
    for n, d in q.groupby("N_B"):
        b, p = d[d.stage == "base"], d[d.stage == "post-trained"]
        if len(b) and len(p):
            b, p = b.iloc[0], p.iloc[0]
            pairs.append(dict(N_B=n, base=b.model, post=p.model,
                              step1_base=b.step1_acc, step1_post=p.step1_acc,
                              fe_base=b.follows_edit, fe_post=p.follows_edit,
                              fe_cond_base=b.follows_edit_cond, fe_cond_post=p.follows_edit_cond,
                              diff_fe_cond=p.follows_edit_cond - b.follows_edit_cond))
    P = pd.DataFrame(pairs); P.to_csv(out / "table_base_vs_post.csv", index=False)

    # scaling fits
    fits, curves = [], {}
    main_T = T[~T.model.isin(ABLATION)]
    for fam in ["qwen3", "gemma3", "llama3", "olmo2", "ALL"]:
        d = main_T if fam == "ALL" else main_T[main_T.family == fam]
        for metric in ["follows_edit", "step1_acc", "cot_acc_kmax", "direct_acc_k2"]:
            rec = dict(family=fam, metric=metric, n_models=int(d[metric].notna().sum()),
                       spearman_vs_N=spearman(d.N_B, d[metric]))
            ft = fit_saturating(d.N_B * 1e9, d[metric]) if fam != "ALL" else None
            if ft:
                rec.update(floor=ft["floor"], slope=ft["slope"], midpoint_N_B=ft["midpoint_N_B"], r2=ft["r2"])
                curves[(fam, metric)] = ft
            fits.append(rec)
        dc = d[d.capable]
        fits.append(dict(family=fam, metric="follows_edit_cond (capable only)",
                         n_models=int(dc.follows_edit_cond.notna().sum()),
                         spearman_vs_N=spearman(dc.N_B, dc.follows_edit_cond)))
    F = pd.DataFrame(fits); F.to_csv(out / "table_scaling_fits.csv", index=False)

    # OLMo token axis (limitation section)
    O = T[T.family == "olmo2"][["model", "N_B", "D_T", "compliance", "step1_acc", "cot_acc_k2",
                                 "direct_acc_k2", "follows_edit", "capable"]]
    O.to_csv(out / "table_olmo_checkpoints.csv", index=False)

    # ---------------- figures ----------------
    col = {"qwen3": "C0", "gemma3": "C1", "llama3": "C2", "olmo2": "C3"}
    mk = {"post-trained": "o", "base": "s"}

    fig, ax = plt.subplots(figsize=(6, 4))
    for (fam, st), d in main_T.groupby(["family", "stage"]):
        d = d.sort_values("N_B")
        ax.errorbar(d.N_B, d.follows_edit, yerr=[d.follows_edit - d.follows_edit_lo, d.follows_edit_hi - d.follows_edit],
                    fmt=mk[st] + ("-" if st == "post-trained" else "--"), color=col.get(fam, "k"),
                    label=f"{fam} {st}", capsize=2, ms=5)
    if ("qwen3", "follows_edit") in curves:
        ft = curves[("qwen3", "follows_edit")]
        xs = np.linspace(np.log10(main_T.N_B.min() * 1e9), np.log10(main_T.N_B.max() * 1e9), 100)
        ax.plot(10 ** xs / 1e9, ft["f"](xs, *ft["p"]), ":", color="C0", alpha=.7, label=f"Qwen3 fit (R²={ft['r2']:.2f})")
    ax.set_xscale("log"); ax.set_xlabel("Non-embedding parameters (B)"); ax.set_ylabel("P(answer follows edited step)")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=7); ax.grid(alpha=.3); ax.set_title("CoT reliance vs. model size")
    fig.tight_layout(); fig.savefig(out / "fig1_reliance_vs_size.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for (fam, st), d in main_T.groupby(["family", "stage"]):
        d = d.sort_values("N_B")
        ax.plot(d.N_B, d.cot_acc_kmax, mk[st] + "-", color=col.get(fam, "k"), label=f"{fam} {st} CoT")
        ax.plot(d.N_B, d.direct_acc_k4plus, mk[st] + ":", color=col.get(fam, "k"), alpha=.6)
    ax.axhline(CHANCE, color="gray", lw=1, ls="--", label="chance (1/24)")
    ax.set_xscale("log"); ax.set_xlabel("Non-embedding parameters (B)"); ax.set_ylabel("Accuracy")
    ax.set_title("CoT at k=24 (solid) vs direct, k≥4 (dotted)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "fig2_cot_vs_direct.png", dpi=200); plt.close(fig)

    show = [m for m in main_T.sort_values(["family", "N_B"]).model if stage(m) == "post-trained"]
    ncol = 4; nrow = math.ceil(len(show) / ncol) or 1
    fig, axs = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.4 * nrow), sharey=True, squeeze=False)
    for i, m in enumerate(show):
        a = axs[i // ncol][i % ncol]
        for cond, ls in [("cot", "-o"), ("direct", "--s")]:
            d = K[(K.model == m) & (K.cond == cond)].sort_values("k")
            a.plot(d.k, d.acc, ls, ms=3, label=cond)
        a.axhline(CHANCE, color="gray", lw=.8, ls=":"); a.set_title(m, fontsize=8); a.set_ylim(0, 1.05); a.grid(alpha=.3)
    for j in range(len(show), nrow * ncol): axs[j // ncol][j % ncol].axis("off")
    axs[0][0].legend(fontsize=7); fig.supxlabel("k (number of swaps)"); fig.supylabel("Accuracy")
    fig.tight_layout(); fig.savefig(out / "fig3_acc_by_k.png", dpi=200); plt.close(fig)

    if len(P):
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.plot(P.N_B, P.fe_cond_post, "o-", label="post-trained")
        ax.plot(P.N_B, P.fe_cond_base, "s--", label="base")
        ax.set_xscale("log"); ax.set_ylim(0, 1.05); ax.grid(alpha=.3); ax.legend()
        ax.set_xlabel("Qwen3 parameters (B)"); ax.set_ylabel("Follows edit | correct prefix")
        ax.set_title("Post-training effect (Qwen3)")
        fig.tight_layout(); fig.savefig(out / "fig4_base_vs_post.png", dpi=200); plt.close(fig)

    if len(O):
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for n, d in O.groupby("N_B"):
            d = d.sort_values("D_T"); ax.plot(d.D_T, d.step1_acc, "o-", label=f"OLMo-2 {n:.1f}B")
        ax.axhline(CAP_THRESH, color="gray", ls="--", lw=1, label="capability threshold")
        ax.set_xlabel("Training tokens (T)"); ax.set_ylabel("Step-1 accuracy"); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7); ax.grid(alpha=.3); ax.set_title("OLMo-2 checkpoints (token axis)")
        fig.tight_layout(); fig.savefig(out / "fig5_olmo_tokens.png", dpi=200); plt.close(fig)

    # ---------------- numbers for the text ----------------
    fmt = lambda v: "nan" if v != v else f"{v:.3f}"
    L = ["# Key numbers for the paper", "", f"Models complete: {len(T)} | capable (step1>={CAP_THRESH}): {int(T.capable.sum())}",
         f"Items per model: {int(T.n_items.median())} | bootstrap: {NBOOT} resamples over problems | chance = 1/24", ""]
    L.append("## Per model"); L.append("")
    L.append("| model | N(B) | stage | step1 | CoT acc k=24 | direct k=2 | direct k>=4 | follows_edit [95% CI] | follows_edit given correct prefix |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in T.iterrows():
        L.append(f"| {r.model} | {r.N_B} | {r.stage} | {fmt(r.step1_acc)} | {fmt(r.cot_acc_kmax)} | {fmt(r.direct_acc_k2)} | "
                 f"{fmt(r.direct_acc_k4plus)} | {fmt(r.follows_edit)} [{fmt(r.follows_edit_lo)}, {fmt(r.follows_edit_hi)}] | "
                 f"{fmt(r.follows_edit_cond)} (n={int(r.n_edit_cond)}) |")
    L += ["", "## Scaling fits (saturating logistic in log N; Spearman vs N)", ""]
    for _, r in F.iterrows():
        L.append(f"- {r.family} / {r.metric}: n={r.n_models}, Spearman={fmt(r.spearman_vs_N)}"
                 + (f", R²={fmt(r.r2)}, midpoint={r.midpoint_N_B:.2f}B, floor={fmt(r.floor)}" if "r2" in r and r.r2 == r.r2 else ""))
    if len(P):
        L += ["", "## Base vs post-trained (Qwen3, follows_edit | correct prefix)", ""]
        for _, r in P.iterrows():
            L.append(f"- {r.N_B}B: base {fmt(r.fe_cond_base)} vs post {fmt(r.fe_cond_post)} (diff {fmt(r.diff_fe_cond)})")
    L += ["", "## Max direct accuracy at k>=4 across all models", "",
          f"{fmt(T.direct_acc_k4plus.max())} ({T.loc[T.direct_acc_k4plus.idxmax(), 'model']}) vs chance {CHANCE:.3f}"]
    (out / "paper_numbers.md").write_text("\n".join(L))

    z = shutil.make_archive(str(config.DATA_ROOT / "paper_kit"), "zip", out)
    print(T[["model", "N_B", "stage", "step1_acc", "cot_acc_kmax", "direct_acc_k2", "follows_edit", "follows_edit_cond", "capable"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n" + F.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwritten -> {out}\nzip     -> {z}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="main")
    main(ap.parse_args().tag)
