"""Aggregate results/metrics_long.csv + calibration_long.csv into markdown tables, summary CSVs and figures."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import piva_core as pc

HERE = Path(__file__).parent
R = HERE / "results"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
L = pd.read_csv(R / "metrics_long.csv")
C = pd.read_csv(R / "calibration_long.csv")
CRB = pd.read_csv(R / "crb_widths.csv")
P = pc.PARAM_NAMES
CORE = ["Const prior centre", "Const training mean", "NLLS", "PIA", "PIVA-Diagonal", "PIVA-CholeskyILR"]
md = []


def agg(df):
    g = df.groupby(["model", "cohort", "sigma", "metric"])["value"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["std"] = out["std"].fillna(0.0)
    return out


A = agg(L[L.cohort != "train"])
A.to_csv(R / "summary_mean_sd.csv", index=False)


def fmt(m, s, n, d):
    if n <= 1:
        return f"{m:.{d}f}"
    return f"{m:.{d}f} ± {s:.{d}f}"


def table(models, cohort, sigma, metrics, header, d=3, colnames=None):
    rows = []
    for mo in models:
        sub = A[(A.model == mo) & (A.cohort == cohort) & np.isclose(A.sigma, sigma)]
        if sub.empty:
            continue
        cells = []
        for me in metrics:
            r = sub[sub.metric == me]
            cells.append("n/a" if r.empty else fmt(r["mean"].iloc[0], r["std"].iloc[0], int(r["count"].iloc[0]), d))
        n = int(sub["count"].max())
        rows.append(f"| {mo} | {n} | " + " | ".join(cells) + " |")
    cn = colnames or metrics
    return [header, "", "| Model | n seeds | " + " | ".join(cn) + " |", "|" + "---|" * (len(cn) + 2)] + rows + [""]


def per_param_tables(cohort, sigma, models=CORE):
    out = []
    out += table(models, cohort, sigma, [f"{p}_R2" for p in P], f"**Per-parameter R², cohort {cohort}, σ = {sigma}**", 3, P)
    out += table(models, cohort, sigma, [f"{p}_MAE" for p in P],
                 f"**Per-parameter MAE (D in µm²/ms, T2 in ms, v fraction), cohort {cohort}, σ = {sigma}**", 3, P)
    out += table(models, cohort, sigma, [f"{p}_bias" for p in P], f"**Per-parameter bias (mean pred − true), cohort {cohort}, σ = {sigma}**", 3, P)
    return out


def pooled_table(cohort, sigma, models=CORE):
    ms = ["pooled_D_R2", "pooled_T2_R2", "pooled_v_R2", "pooled_D_MAE", "pooled_T2_MAE", "pooled_v_MAE"]
    return table(models, cohort, sigma, ms, f"**Repo-style pooled metrics, cohort {cohort}, σ = {sigma}**", 3,
                 ["D R²", "T2 R²", "v R²", "D MAE", "T2 MAE", "v MAE"])


def uq_table(cohort, sigma, models):
    out = [f"**Uncertainty, cohort {cohort}, σ = {sigma}** (mean ± SD over seeds; coverage target 0.95; width = 3.92 × mean posterior SD × τ as % of physiological range; "
           "ρ = Spearman correlation of posterior SD with |error| per parameter)", "",
           "| Param | Model | raw cov | raw width % | ρ(sd, err) | v2.0 τ (voxel split, grid≤25) | v2.0 cal cov | v2.0 width % | sep-cohort τ | sep-cohort cal cov [patient-bootstrap 95% CI] | sep-cohort width % |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in P:
        for mo in models:
            u = L[(L.model == mo) & (L.cohort == cohort) & np.isclose(L.sigma, sigma)]
            c = C[(C.model == mo) & (C.cohort == cohort) & np.isclose(C.sigma, sigma) & (C.param == p)]
            if c.empty:
                continue
            ms = lambda k: fmt(c[k].mean(), c[k].std() if len(c) > 1 else 0, len(c), 3 if "cov" in k else 1)
            rho = u[u.metric == f"{p}_err_sd_spearman"]["value"]
            ci = f"[{c['sep_cal_cov_lo'].mean():.2f}, {c['sep_cal_cov_hi'].mean():.2f}]"
            atmax = int(c["v20_tau_at_max"].sum())
            out.append(f"| {p} | {mo} | {ms('raw_cov')} | {ms('raw_width_pct')} | "
                       f"{fmt(rho.mean(), rho.std() if len(rho) > 1 else 0, len(rho), 2)} | {ms('v20_tau')}"
                       f"{' (' + str(atmax) + '/' + str(len(c)) + ' at grid max)' if atmax else ''} | {ms('v20_cal_cov')} | {ms('v20_width_pct')} | "
                       f"{ms('sep_tau')} | {ms('sep_cal_cov')} {ci} | {ms('sep_width_pct')} |")
    return out + [""]


def reference_widths(cohort, sigma):
    fx = L[(L.model.str.startswith("Fixed")) & (L.cohort == cohort) & np.isclose(L.sigma, sigma)].set_index("metric")["value"]
    cr = CRB[np.isclose(CRB.sigma, sigma)].set_index("param")
    out = [f"**Reference intervals, cohort {cohort}, σ = {sigma}** (data-independent fixed interval = central 95% of the training-set parameter distribution; "
           "BCRB = Bayesian Cramér-Rao width 3.92 × SD from `fisher_crb.py`, expected Rician Fisher info over the large cohort + full training prior covariance)", "",
           "| Param | Fixed interval coverage | Fixed interval width % | BCRB width % (Rician, full prior) | fisher2.py width % (reproduced) | prior-only width % | BCRB SD / prior SD |",
           "|---|---|---|---|---|---|---|"]
    for p in P:
        out.append(f"| {p} | {fx[f'{p}_rawcov']:.3f} | {fx[f'{p}_rawwidth_pct']:.1f} | {cr.loc[p, 'bcrb_rician_expectedF_fullprior_pct']:.1f} | "
                   f"{cr.loc[p, 'fisher2_reproduced_pct']:.1f} | {cr.loc[p, 'prior_full_only_pct']:.1f} | {cr.loc[p, 'shrink_ratio_bcrb_rician_full']:.2f} |")
    return out + [""]


def nlls_table():
    out = ["**NLLS fit diagnostics** (failed = curve_fit exception, masked by `valid`; bound = at least one of 8 params on its box bound; v_lum<0 = simplex violated)", "",
           "| Cohort | σ | n | failed (masked) | ≥1 param at bound | v_lum < 0 |", "|---|---|---|---|---|---|"]
    for c in ["orig240", "large2000"]:
        for s in pc.SIGMA_TEST_LEVELS:
            d = L[(L.model == "NLLS") & (L.cohort == c) & np.isclose(L.sigma, s)].set_index("metric")["value"]
            out.append(f"| {c} | {s} | {int(d['n'])} | {int(d['n_failed'])} | {int(d['n_any_bound'])} | {int(d['n_vlum_neg'])} |")
    return out + [""]


def train_table():
    t = L[L.cohort == "train"].pivot_table(index=["model", "seed"], columns="metric", values="value").reset_index()
    out = ["**Training summary** (final-epoch signal MSE and KL; wall time on 1 CPU thread with 8 jobs in parallel)", "",
           "| Model | seed | final signal MSE | final KL | train time (min) |", "|---|---|---|---|---|"]
    for _, r in t.iterrows():
        out.append(f"| {r['model']} | {int(r['seed'])} | {r['final_signal_mse']:.1f} | {r['final_kl']:.2f} | {r['train_seconds'] / 60:.0f} |")
    return out + [""]


def extras_table(cohort, sigma):
    ex = sorted([m for m in L.model.unique() if m.startswith("X:")])
    base = ["PIVA-Diagonal", "PIVA-CholeskyILR"]
    out = [f"**Reviewer experiments vs originals, cohort {cohort}, σ = {sigma}** (accuracy = per-parameter R²; seeds averaged where >1)", ""]
    out += table(base + ex, cohort, sigma, [f"{p}_R2" for p in P], "R² per parameter", 3, P)
    out += table(base + ex, cohort, sigma, [f"{p}_rawcov" for p in P], "Raw 95% coverage per parameter", 3, P)
    out += table(base + ex, cohort, sigma, [f"{p}_rawwidth_pct" for p in P], "Raw 95% width, % of range", 1, P)
    out += table(base + ex, cohort, sigma, [f"{p}_err_sd_spearman" for p in P], "Spearman ρ(posterior SD, |error|)", 2, P)
    cc = C[(C.cohort == cohort) & np.isclose(C.sigma, sigma) & C.model.isin(base + ex)]
    piv = cc.groupby(["model", "param"])[["sep_cal_cov", "sep_width_pct"]].mean().reset_index()
    out += ["Separate-cohort calibrated coverage / width %", "", "| Model | " + " | ".join(P) + " |", "|" + "---|" * 10]
    for mo in base + ex:
        s = piv[piv.model == mo].set_index("param")
        if s.empty:
            continue
        out.append(f"| {mo} | " + " | ".join(f"{s.loc[p, 'sep_cal_cov']:.2f} / {s.loc[p, 'sep_width_pct']:.0f}" for p in P) + " |")
    return out + [""]


# ── figures ─────────────────────────────────────────────────────────────────
COLORS = {"Const prior centre": "#9e9e9e", "Const training mean": "#cfcfcf", "NLLS": "#C0504D", "PIA": "#4472C4",
          "PIVA-Diagonal": "#ED7D31", "PIVA-CholeskyILR": "#70AD47"}
LBL = [p.replace("_lum", "_lu").replace("T2", "T₂") for p in P]


def fig_accuracy(cohort="large2000", sigma=0.05):
    models = CORE
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True)
    x = np.arange(9)
    w = 0.8 / len(models)
    ref = A[(A.model == "Const prior centre") & (A.cohort == cohort) & np.isclose(A.sigma, sigma)].set_index("metric")["mean"]
    for i, mo in enumerate(models):
        sub = A[(A.model == mo) & (A.cohort == cohort) & np.isclose(A.sigma, sigma)].set_index("metric")
        r2 = np.array([sub.loc[f"{p}_R2", "mean"] for p in P])
        r2e = np.array([sub.loc[f"{p}_R2", "std"] for p in P])
        mae = np.array([sub.loc[f"{p}_MAE", "mean"] / ref[f"{p}_MAE"] for p in P])
        maee = np.array([sub.loc[f"{p}_MAE", "std"] / ref[f"{p}_MAE"] for p in P])
        xo = x - 0.4 + w * (i + 0.5)
        axes[0].bar(xo, np.clip(r2, -2.5, 1), w, yerr=r2e, color=COLORS[mo], label=mo, capsize=1.5, error_kw={"lw": 0.7})
        axes[1].bar(xo, mae, w, yerr=maee, color=COLORS[mo], capsize=1.5, error_kw={"lw": 0.7})
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_ylim(-2.5, 1)
    axes[0].set_ylabel("per-parameter R²\n(NLLS bars clipped at −2.5)")
    axes[1].axhline(1, color="k", lw=0.8, ls="--")
    axes[1].set_ylabel("MAE / MAE of\nprior-centre constant")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(LBL)
    axes[0].legend(ncol=3, fontsize=7.5, frameon=False, loc="upper left")
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Per-parameter accuracy, {cohort} test cohort, σ = {sigma} (mean ± SD over seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_per_parameter_accuracy.png", dpi=300)
    plt.close(fig)


def fig_uq(cohort="large2000", sigma=0.05):
    models = ["PIVA-Diagonal", "PIVA-CholeskyILR"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True)
    x = np.arange(9)
    w = 0.2
    fx = L[(L.model.str.startswith("Fixed")) & (L.cohort == cohort) & np.isclose(L.sigma, sigma)].set_index("metric")["value"]
    crb = CRB[np.isclose(CRB.sigma, sigma)].set_index("param")
    for i, mo in enumerate(models):
        c = C[(C.model == mo) & (C.cohort == cohort) & np.isclose(C.sigma, sigma)].groupby("param")
        m, s = c.mean(numeric_only=True).loc[P], c.std(numeric_only=True).loc[P].fillna(0)
        col = COLORS[mo]
        axes[0].bar(x - 0.3 + 2 * i * w, m["raw_cov"], w, yerr=s["raw_cov"], color=col, alpha=0.45, label=f"{mo} raw", capsize=1.5, error_kw={"lw": 0.7})
        lo, hi = m["sep_cal_cov"] - m["sep_cal_cov_lo"], m["sep_cal_cov_hi"] - m["sep_cal_cov"]
        axes[0].bar(x - 0.3 + (2 * i + 1) * w, m["sep_cal_cov"], w, yerr=[lo, hi], color=col, label=f"{mo} calibrated", capsize=1.5, error_kw={"lw": 0.7})
        axes[1].bar(x - 0.3 + 2 * i * w, m["raw_width_pct"], w, color=col, alpha=0.45, yerr=s["raw_width_pct"], capsize=1.5, error_kw={"lw": 0.7})
        axes[1].bar(x - 0.3 + (2 * i + 1) * w, m["sep_width_pct"], w, color=col, yerr=s["sep_width_pct"], capsize=1.5, error_kw={"lw": 0.7})
    axes[0].axhline(0.95, color="k", ls="--", lw=0.9)
    axes[0].set_ylabel("95% interval coverage")
    axes[0].set_ylim(0, 1.05)
    axes[1].scatter(x, [crb.loc[p, "bcrb_rician_expectedF_fullprior_pct"] for p in P], marker="_", s=300, color="k", zorder=5,
                    label="information limit (Bayesian CRB)")
    axes[1].scatter(x, [fx[f"{p}_rawwidth_pct"] for p in P], marker="x", s=30, color="#7030A0", zorder=5,
                    label="fixed data-independent interval")
    axes[1].axhline(100, color="#999", lw=0.8, ls=":")
    axes[1].set_ylabel("interval width,\n% of physiological range")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(LBL)
    axes[0].legend(ncol=4, fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    axes[1].legend(fontsize=7, frameon=False, loc="upper left")
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Posterior calibration, {cohort}, σ = {sigma}. Calibrated = per-parameter τ fit on a separate cohort;\n"
                 "error bars: seed SD (raw), patient-bootstrap 95% CI (calibrated)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_coverage_width.png", dpi=300)
    plt.close(fig)


def fig_sigma(cohort="large2000"):
    models = ["NLLS", "PIA", "PIVA-Diagonal", "PIVA-CholeskyILR"]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    for k, blk in enumerate(["ep", "st", "lum"]):
        ax = axes[k]
        for mo in models:
            ys, es = [], []
            for s in pc.SIGMA_TEST_LEVELS:
                sub = A[(A.model == mo) & (A.cohort == cohort) & np.isclose(A.sigma, s) & (A.metric == f"v_{blk}_R2")]
                ys.append(sub["mean"].iloc[0])
                es.append(sub["std"].iloc[0])
            ax.errorbar(pc.SIGMA_TEST_LEVELS, ys, yerr=es, marker="o", ms=3, color=COLORS[mo], label=mo, capsize=2)
        ax.set_title(f"v_{blk.replace('lum', 'lu')}", fontsize=9)
        ax.set_xlabel("test σ")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("R²")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    (FIG / "supplementary").mkdir(exist_ok=True)
    fig.savefig(FIG / "supplementary" / "fig3_vfraction_r2_vs_sigma.png", dpi=300)
    plt.close(fig)


def fig_likelihood(cohort="large2000", sigma=0.05):
    """Experiment A: raw coverage and raw width vs likelihood scale, Cholesky+ILR (one seed for X: runs)."""
    runs = [("PIVA-CholeskyILR", "MSE + 0.1·KL\n(original, SD≈0.9)"),
            ("X:Chol+ILR|gaussSD15.49|clean", "Gauss SD 15.5"),
            ("X:Chol+ILR|gaussSD28.28|clean", "Gauss SD 28.3"),
            ("X:Chol+ILR|gaussSD50|clean", "Gauss SD 50"),
            ("X:Chol+ILR|gaussSD50|noisy", "Gauss SD 50,\nnoisy target"),
            ("X:Chol+ILR|ricianSD50|noisy", "Rician SD 50,\nnoisy target")]
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(runs)))
    crb = CRB[np.isclose(CRB.sigma, sigma)].set_index("param")
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.4), sharex=True)
    x = np.arange(9)
    w = 0.8 / len(runs)
    for i, (mo, lab) in enumerate(runs):
        sub = A[(A.model == mo) & (A.cohort == cohort) & np.isclose(A.sigma, sigma)].set_index("metric")["mean"]
        xo = x - 0.4 + w * (i + 0.5)
        axes[0].bar(xo, [sub[f"{p}_rawcov"] for p in P], w, color=cols[i], label=lab)
        axes[1].bar(xo, [sub[f"{p}_rawwidth_pct"] for p in P], w, color=cols[i])
    axes[0].axhline(0.95, color="k", ls="--", lw=0.9)
    axes[0].set_ylabel("raw 95% coverage")
    axes[1].scatter(x, [crb.loc[p, "bcrb_rician_expectedF_fullprior_pct"] for p in P], marker="_", s=300, color="k", zorder=5,
                    label="information limit (Bayesian CRB)")
    axes[1].axhline(113, color="#999", lw=0.8, ls=":", label="uniform over the tanh box (113%)")
    axes[1].set_ylabel("raw width,\n% of physiological range")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(LBL)
    axes[0].legend(ncol=3, fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    axes[1].legend(fontsize=7, frameon=False, loc="upper right")
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Likelihood scale, PIVA-CholeskyILR, {cohort}, σ = {sigma} (no post-hoc calibration)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_likelihood_scale.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    for coh in ["orig240", "large2000"]:
        md += [f"### Cohort {coh}, σ = 0.05", ""]
        md += per_param_tables(coh, 0.05)
        md += pooled_table(coh, 0.05)
    for s in [0.01, 0.03, 0.08]:
        md += [f"### Cohort large2000, σ = {s}", ""]
        md += table(CORE, "large2000", s, [f"{p}_R2" for p in P], f"**Per-parameter R², large2000, σ = {s}**", 3, P)
        md += table(CORE, "large2000", s, [f"{p}_MAE" for p in P], f"**Per-parameter MAE, large2000, σ = {s}**", 3, P)
        md += pooled_table("large2000", s)
    for s in [0.01, 0.03, 0.08]:
        md += table(CORE, "orig240", s, [f"{p}_R2" for p in P], f"**Per-parameter R², orig240, σ = {s}**", 3, P)
    md += ["### Uncertainty", ""]
    for coh in ["orig240", "large2000"]:
        md += uq_table(coh, 0.05, ["PIVA-Diagonal", "PIVA-CholeskyILR"])
        md += reference_widths(coh, 0.05)
    md += table(["PIVA-Diagonal", "PIVA-CholeskyILR"], "large2000", 0.05,
                [f"repo_calr_{b}" for b in ["D", "T2", "v"]] + [f"pooled_cov_{b}" for b in ["D", "T2", "v"]],
                "**Repo-style block summaries, large2000, σ = 0.05** (repo calibration r = Pearson of voxel-mean SD vs voxel-mean |error| per block)", 3,
                ["cal r D", "cal r T2", "cal r v", "cov D", "cov T2", "cov v"])
    md += nlls_table()
    md += train_table()
    md += ["### Reviewer experiments", ""]
    for coh in ["orig240", "large2000"]:
        md += extras_table(coh, 0.05)
    (R / "tables.md").write_text("\n".join(md))
    fig_accuracy()
    fig_uq()
    fig_sigma()
    fig_likelihood()
    print("wrote", R / "tables.md")
