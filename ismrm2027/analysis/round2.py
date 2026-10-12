"""
Round-2 follow-up: per-voxel-SD likelihood vs fixed SD-50, 3 seeds each, all test sigmas.

Reads posterior means/SDs from preds/ (written by evaluate.py) and computes per parameter:
raw coverage, raw width, box-clipped width (interval intersected with the physiological box),
accuracy, D/T2 reversion diagnostics, and coverage stratified by true epithelium fraction.
Writes results/round2_long.csv, results/round2_tables.md, figures/fig5_vfraction_sigma_sweep.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import piva_core as pc

HERE = Path(__file__).parent
PRED, R, FIG = HERE / "preds", HERE / "results", HERE / "figures"
P = pc.PARAM_NAMES
SIG = pc.SIGMA_TEST_LEVELS
MODELS = {
    "Chol+ILR SD-50": "gen-chol-ilr-gauss50-clean_s{}",
    "Chol+ILR per-voxel SD": "gen-chol-ilr-gaussPV-clean_s{}",
    "Diag+softmax SD-50": "gen-diag-softmax-gauss50-clean_s{}",
    "Diag+softmax per-voxel SD": "gen-diag-softmax-gaussPV-clean_s{}",
    "Cholesky original (MSE+0.1KL)": "chol_s{}",
    "Diagonal original (MSE+0.1KL)": "diag_s{}",
}
CRB = pd.read_csv(R / "crb_widths.csv")
UNIFORM_BOX_PCT = 100 * 3.92 / np.sqrt(12)  # 3.92 x SD of a uniform over the box = 113%


def rows_for(label, pat, seed, cohort):
    co = pc.build_test_cohort(cohort)
    y = co["y"]
    vep_hi = y[:, 6] > np.median(y[:, 6])
    out = []
    for s in SIG:
        z = np.load(PRED / f"{pat.format(seed)}_{cohort}_sig{s}.npz")
        m, sd = z["mean"], z["sd"]
        lo, hi = m - 1.96 * sd, m + 1.96 * sd
        cov = (y >= lo) & (y <= hi)
        clipw = np.minimum(hi, pc.RANGE_HI) - np.maximum(lo, pc.RANGE_LO)
        for j, p in enumerate(P):
            t = y[:, j]
            out.append({
                "model": label, "seed": seed, "cohort": cohort, "sigma": s, "param": p,
                "raw_cov": cov[:, j].mean(),
                "raw_width_pct": 100 * 3.92 * sd[:, j].mean() / pc.FULL_RANGE[j],
                "clip_width_pct": 100 * clipw[:, j].mean() / pc.FULL_RANGE[j],
                "cov_vep_low": cov[~vep_hi, j].mean(), "cov_vep_high": cov[vep_hi, j].mean(),
                "clipw_vep_low": 100 * clipw[~vep_hi, j].mean() / pc.FULL_RANGE[j],
                "clipw_vep_high": 100 * clipw[vep_hi, j].mean() / pc.FULL_RANGE[j],
                "R2": pc.r2_1d(t, m[:, j]), "MAE": float(np.abs(t - m[:, j]).mean()),
                "bias": float((m[:, j] - t).mean()),
                "pred_spread_ratio": float(m[:, j].std() / t.std()),       # 0 = constant prediction
                "mean_offset_from_centre_pct": float(100 * np.abs(m[:, j] - pc.PRIOR_CENTRE[j]).mean() / pc.FULL_RANGE[j]),
                "rho_sd_err": float(spearmanr(sd[:, j], np.abs(t - m[:, j]))[0]),
            })
    return out


def ms(df, col, d=2):
    g = df.groupby("model", sort=False)[col]
    return g.mean(), g.std()


def fmt(m, s, d):
    return f"{m:.{d}f} ± {s:.{d}f}"


def main():
    rows = []
    for label, pat in MODELS.items():
        for seed in [0, 1, 2]:
            if (PRED / f"{pat.format(seed)}_large2000_sig0.05.npz").exists():
                for c in ["large2000", "orig240"]:
                    rows += rows_for(label, pat, seed, c)
    df = pd.DataFrame(rows)
    df.to_csv(R / "round2_long.csv", index=False)
    agg = df.groupby(["model", "cohort", "sigma", "param"]).agg(["mean", "std", "count"])
    md = []
    L = df[df.cohort == "large2000"]
    order = list(MODELS)

    # (a) sigma sweep, v and all params, raw coverage and clipped width vs CRB
    for metric, title, d in [("raw_cov", "raw 95% coverage", 3), ("clip_width_pct", "box-clipped width, % of range", 1),
                             ("raw_width_pct", "unclipped width (3.92 x SD), % of range", 1)]:
        md += [f"**(a) {title}, large2000, mean ± SD over 3 seeds**", "",
               "| Model | σ | " + " | ".join(P) + " |", "|" + "---|" * 11]
        for mo in order:
            for s in SIG:
                sub = L[(L.model == mo) & np.isclose(L.sigma, s)]
                if sub.empty:
                    continue
                g = sub.groupby("param")[metric]
                md.append(f"| {mo} | {s} | " + " | ".join(fmt(g.mean()[p], g.std()[p], d) for p in P) + " |")
        if metric != "raw_cov":
            for s in SIG:
                c = CRB[np.isclose(CRB.sigma, s)].set_index("param")["bcrb_rician_expectedF_fullprior_pct"]
                md.append(f"| **Information limit (BCRB)** | {s} | " + " | ".join(f"{c[p]:.1f}" for p in P) + " |")
        md.append("")

    # (c) accuracy
    for metric, title, d in [("R2", "per-parameter R²", 3), ("MAE", "per-parameter MAE", 3)]:
        md += [f"**(c) {title}, large2000, σ = 0.05, mean ± SD over 3 seeds**", "",
               "| Model | " + " | ".join(P) + " |", "|" + "---|" * 10]
        for mo in order:
            sub = L[(L.model == mo) & np.isclose(L.sigma, 0.05)]
            g = sub.groupby("param")[metric]
            md.append(f"| {mo} | " + " | ".join(fmt(g.mean()[p], g.std()[p], d) for p in P) + " |")
        md.append("")
    md += ["**(c) v R² across σ, large2000**", "", "| Model | σ | v_ep | v_st | v_lum |", "|---|---|---|---|---|"]
    for mo in order:
        for s in SIG:
            sub = L[(L.model == mo) & np.isclose(L.sigma, s)]
            g = sub.groupby("param")["R2"]
            md.append(f"| {mo} | {s} | " + " | ".join(fmt(g.mean()[p], g.std()[p], 3) for p in ["v_ep", "v_st", "v_lum"]) + " |")
    md.append("")

    # (d) D/T2 reversion
    md += ["**(d) D/T2 behaviour, large2000, σ = 0.05** (spread = SD of posterior means / SD of truths, 0 means a constant prediction; "
           f"unclipped width of a uniform over the box = {UNIFORM_BOX_PCT:.0f}%)", "",
           "| Model | metric | " + " | ".join(P[:6]) + " |", "|" + "---|" * 8]
    for mo in order:
        sub = L[(L.model == mo) & np.isclose(L.sigma, 0.05)]
        for metric, d in [("pred_spread_ratio", 2), ("raw_width_pct", 0), ("rho_sd_err", 2)]:
            g = sub.groupby("param")[metric]
            md.append(f"| {mo} | {metric} | " + " | ".join(fmt(g.mean()[p], g.std()[p], d) for p in P[:6]) + " |")
    md.append("")

    # (e) stratified by epithelium fraction
    for s in [0.05, 0.08]:
        md += [f"**(e) Raw coverage stratified by true v_ep (below / above cohort median), large2000, σ = {s}**", "",
               "| Model | stratum | " + " | ".join(P) + " |", "|" + "---|" * 11]
        for mo in order:
            sub = L[(L.model == mo) & np.isclose(L.sigma, s)]
            for col, lab in [("cov_vep_low", "v_ep low"), ("cov_vep_high", "v_ep high")]:
                g = sub.groupby("param")[col]
                md.append(f"| {mo} | {lab} | " + " | ".join(fmt(g.mean()[p], g.std()[p], 2) for p in P) + " |")
        md.append("")
    md += ["**(e) Box-clipped width stratified by v_ep, large2000, σ = 0.05, v parameters**", "",
           "| Model | stratum | v_ep | v_st | v_lum |", "|---|---|---|---|---|"]
    for mo in order:
        sub = L[(L.model == mo) & np.isclose(L.sigma, 0.05)]
        for col, lab in [("clipw_vep_low", "v_ep low"), ("clipw_vep_high", "v_ep high")]:
            g = sub.groupby("param")[col]
            md.append(f"| {mo} | {lab} | " + " | ".join(fmt(g.mean()[p], g.std()[p], 1) for p in ["v_ep", "v_st", "v_lum"]) + " |")
    md.append("")
    (R / "round2_tables.md").write_text("\n".join(md))

    # figure
    show = [("Chol+ILR SD-50", "#E07B39", "s"), ("Chol+ILR per-voxel SD", "#2F6DB5", "o")]
    fig, axes = plt.subplots(2, 3, figsize=(8, 4.8), sharex=True)
    for k, p in enumerate(["v_ep", "v_st", "v_lum"]):
        for mo, col, mk in show:
            sub = L[(L.model == mo) & (L.param == p)].groupby("sigma")
            for row, metric in enumerate(["raw_cov", "clip_width_pct"]):
                mu, sd = sub[metric].mean(), sub[metric].std()
                axes[row, k].errorbar(mu.index, mu.values, yerr=sd.values, color=col, marker=mk, ms=4, capsize=2, label=mo)
        c = CRB[CRB.param == p].sort_values("sigma")
        axes[1, k].plot(c.sigma, c.bcrb_rician_expectedF_fullprior_pct, "k--", lw=1.2, label="information limit (BCRB)")
        axes[0, k].axhline(0.95, color="#777", lw=0.8, ls=":")
        axes[0, k].set_ylim(0.6, 1.0)
        axes[0, k].set_title(p.replace("_lum", "_lu"), fontsize=10)
        axes[1, k].set_xlabel("test noise σ (SNR = 1/σ)")
        axes[1, k].set_ylim(0, 60)
        for r in range(2):
            axes[r, k].grid(alpha=0.3)
            axes[r, k].spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("raw 95% coverage")
    axes[1, 0].set_ylabel("interval width,\n% of range (box-clipped)")
    h, l = axes[1, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, fontsize=7.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Volume-fraction posteriors, PIVA-CholeskyILR, no post-hoc calibration (mean ± SD, 3 seeds)", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "fig5_vfraction_sigma_sweep.png", dpi=300)
    plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
