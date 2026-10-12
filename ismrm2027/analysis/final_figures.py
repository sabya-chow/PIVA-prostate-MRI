"""
Final submission figures (ISMRM 2027 abstract v5 spec, checklist section B) and the A6 stratum numbers.

Writes figures/final/fig1_schematic.png ... fig5_phantom.png, preview_3x2.png, final_numbers.json.
Inference only: reads existing checkpoints in runs/ and posterior predictions in preds/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
import numpy as np
import pandas as pd
import torch

import piva_core as pc
import extras_models as em

torch.set_num_threads(4)
torch.set_flush_denormal(True)
HERE = Path(__file__).parent
OUT = HERE / "figures" / "final"
OUT.mkdir(parents=True, exist_ok=True)
PRED, R = HERE / "preds", HERE / "results"
P = pc.PARAM_NAMES
LAB = [r"$D_\mathrm{ep}$", r"$D_\mathrm{st}$", r"$D_\mathrm{lum}$", r"$T_{2,\mathrm{ep}}$", r"$T_{2,\mathrm{st}}$",
       r"$T_{2,\mathrm{lum}}$", r"$v_\mathrm{ep}$", r"$v_\mathrm{st}$", r"$v_\mathrm{lum}$"]
LO, HI, RNG = pc.RANGE_LO, pc.RANGE_HI, pc.FULL_RANGE
BENIGN, CANCER = 0.232, 0.488  # Chatterjee 2018 epithelium volume fractions, used as signal fractions
NUM = {}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "xtick.labelsize": 10.5, "ytick.labelsize": 10, "legend.fontsize": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})

CO = pc.build_test_cohort("large2000")
Y, PIDS = CO["y"], CO["pids"]
UP = np.unique(PIDS)
IDX = {p: np.where(PIDS == p)[0] for p in UP}
BOOT = np.random.default_rng(0).choice(UP, size=(1000, len(UP)), replace=True)


def load(tag, seed, sig=0.05, cohort="large2000"):
    d = np.load(PRED / f"{tag}_s{seed}_{cohort}_sig{sig}.npz")
    return d["mean"], d["sd"]


def interval_stats(m, sd, tau=1.0):
    lo, hi = m - 1.96 * tau * sd, m + 1.96 * tau * sd
    cov = (Y >= lo) & (Y <= hi)
    w = (np.minimum(hi, HI) - np.maximum(lo, LO)) / RNG * 100
    return cov.astype(float), w


def boot_ci(per_voxel_by_seed):
    """per_voxel_by_seed: (n_seed, N, 9). Point = mean over seeds and voxels; CI = patient bootstrap of seed-mean."""
    x = per_voxel_by_seed.mean(0)
    point = x.mean(0)
    bs = np.array([x[np.concatenate([IDX[p] for p in b])].mean(0) for b in BOOT])
    return point, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)


def save(fig, name):
    f = OUT / name
    fig.savefig(f, dpi=300, facecolor="white")
    plt.close(fig)
    return f


# ─────────────────────────────────────────────────────────────── Fig. 2 accuracy
def fig2():
    L = pd.read_csv(R / "summary_mean_sd.csv")
    L = L[(L.cohort == "large2000") & np.isclose(L.sigma, 0.05)]
    models = [("Const prior centre", "constant (prior centre)", "#9e9e9e"),
              ("Const training mean", "constant (training mean)", "#cfcfcf"),
              ("NLLS", "NLLS", "#C0504D"), ("PIA", "PIA", "#4472C4"),
              ("PIVA-Diagonal", "PIVA-Diagonal", "#ED7D31"), ("PIVA-CholeskyILR", "PIVA-CholeskyILR", "#70AD47")]
    ref = L[L.model == "Const prior centre"].set_index("metric")["mean"]
    fig, ax = plt.subplots(2, 1, figsize=(7.5, 5.2), sharex=True)
    x, w = np.arange(9), 0.8 / len(models)
    for i, (mo, lab, col) in enumerate(models):
        s = L[L.model == mo].set_index("metric")
        r2 = np.array([s.loc[f"{p}_R2", "mean"] for p in P]); r2e = np.array([s.loc[f"{p}_R2", "std"] for p in P])
        mae = np.array([s.loc[f"{p}_MAE", "mean"] / ref[f"{p}_MAE"] for p in P])
        maee = np.array([s.loc[f"{p}_MAE", "std"] / ref[f"{p}_MAE"] for p in P])
        xo = x - 0.4 + w * (i + 0.5)
        ax[0].bar(xo, np.clip(r2, -2.5, 1), w, yerr=r2e, color=col, label=lab, capsize=1.2, error_kw={"lw": 0.6})
        ax[1].bar(xo, mae, w, yerr=maee, color=col, capsize=1.2, error_kw={"lw": 0.6})
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_ylim(-2.5, 1)
    ax[0].set_ylabel("R² (per parameter)")
    ax[0].text(8.45, -2.3, "NLLS bars for D and T2\nclipped at −2.5", fontsize=9, color="#C0504D", ha="right", va="bottom")
    ax[1].axhline(1, color="k", lw=0.9, ls="--")
    ax[1].set_ylabel("MAE / MAE of constant")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(LAB, fontsize=12)
    ax[0].legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
    for a in ax:
        a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return save(fig, "fig2_accuracy.png")


# ─────────────────────────────────────────────────────────────── Fig. 3 original-loss calibration
def fig3():
    C = pd.read_csv(R / "calibration_long.csv")
    C = C[(C.cohort == "large2000") & np.isclose(C.sigma, 0.05)]
    crb = pd.read_csv(R / "crb_widths.csv")
    crb = crb[np.isclose(crb.sigma, 0.05)].set_index("param").loc[P, "bcrb_rician_expectedF_fullprior_pct"].values
    _, _, ytr = pc.build_training_set()
    flo, fhi = np.percentile(ytr, 2.5, 0), np.percentile(ytr, 97.5, 0)
    fixed_w = 100 * (fhi - flo) / RNG
    fixed_cov = ((Y >= flo) & (Y <= fhi)).mean(0)
    models = [("diag", "PIVA-Diagonal", "#ED7D31"), ("chol", "PIVA-CholeskyILR", "#70AD47")]
    res = {}
    for tag, name, _ in models:
        rc, rw, cc, cw = [], [], [], []
        for s in range(3):
            m, sd = load(tag, s)
            tau = C[(C.model == name) & (C.seed == s)].set_index("param").loc[P, "sep_tau"].values
            a, b = interval_stats(m, sd)
            c, d = interval_stats(m, sd, tau)
            rc.append(a); rw.append(b); cc.append(c); cw.append(d)
        res[name] = {k: boot_ci(np.array(v)) for k, v in
                     [("raw_cov", rc), ("raw_w", rw), ("cal_cov", cc), ("cal_w", cw)]}
    fig, ax = plt.subplots(2, 1, figsize=(7.5, 5.8), sharex=True)
    x, w = np.arange(9), 0.2
    for i, (tag, name, col) in enumerate(models):
        for k, (cv, wd, alpha, lab) in enumerate([("raw_cov", "raw_w", 0.45, "raw"), ("cal_cov", "cal_w", 1.0, "after σ-scaling")]):
            xo = x - 0.3 + (2 * i + k) * w
            for row, key in [(0, cv), (1, wd)]:
                pt, lo, hi = res[name][key]
                ax[row].bar(xo, pt, w, yerr=[pt - lo, hi - pt], color=col, alpha=alpha,
                            label=f"{name}, {lab}" if row == 0 else None, capsize=1.2, error_kw={"lw": 0.6})
    ax[0].axhline(0.95, color="k", ls="--", lw=0.9)
    ax[0].set_ylim(0, 1.05)
    ax[0].set_ylabel("95% interval coverage")
    ax[1].scatter(x, crb, marker="_", s=320, lw=2.2, color="k", zorder=5, label="information limit (BCRB)")
    ax[1].scatter(x, fixed_w, marker="x", s=36, lw=1.6, color="#7030A0", zorder=5, label="fixed interval (ignores signal)")
    ax[1].set_ylim(0, 100)
    ax[1].set_ylabel("width, % of prior box")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(LAB, fontsize=12)
    ax[0].legend(ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
    ax[1].legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=9.5)
    for a in ax:
        a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    f = save(fig, "fig3_calibration_original_loss.png")
    NUM["fig3"] = {name: {k: {p: [round(float(v[0][j]), 3), round(float(v[1][j]), 3), round(float(v[2][j]), 3)]
                              for j, p in enumerate(P)} for k, v in d.items()} for name, d in res.items()}
    NUM["fig3"]["fixed_interval"] = {"coverage": dict(zip(P, np.round(fixed_cov, 3).tolist())),
                                     "width_pct": dict(zip(P, np.round(fixed_w, 1).tolist()))}
    NUM["fig3"]["bcrb_width_pct"] = dict(zip(P, np.round(crb, 1).tolist()))
    return f


# ─────────────────────────────────────────────────────────────── Fig. 4 sigma sweep
def fig4():
    crb = pd.read_csv(R / "crb_widths.csv")
    series = [("gen-chol-ilr-gauss50-clean", "fixed likelihood SD 50", "#E07B39", "s"),
              ("gen-chol-ilr-gaussPV-clean", "per-voxel likelihood SD", "#2F6DB5", "o")]
    sig = pc.SIGMA_TEST_LEVELS
    fig, ax = plt.subplots(2, 3, figsize=(7.5, 4.9), sharex=True)
    out = {}
    for tag, lab, col, mk in series:
        cov = np.zeros((3, 4, 9)); wid = np.zeros((3, 4, 9))
        for s in range(3):
            for k, sg in enumerate(sig):
                m, sd = load(tag, s, sg)
                c, w = interval_stats(m, sd)
                cov[s, k], wid[s, k] = c.mean(0), w.mean(0)
        out[lab] = {"coverage_mean": cov.mean(0)[:, 6:].round(3).tolist(), "coverage_sd": cov.std(0, ddof=1)[:, 6:].round(3).tolist(),
                    "width_mean": wid.mean(0)[:, 6:].round(1).tolist()}
        for j in range(3):
            ax[0, j].errorbar(sig, cov.mean(0)[:, 6 + j], yerr=cov.std(0, ddof=1)[:, 6 + j], color=col, marker=mk, ms=4.5, capsize=2, lw=1.6, label=lab)
            ax[1, j].errorbar(sig, wid.mean(0)[:, 6 + j], yerr=wid.std(0, ddof=1)[:, 6 + j], color=col, marker=mk, ms=4.5, capsize=2, lw=1.6, label=lab)
    for j in range(3):
        c = crb[crb.param == P[6 + j]].sort_values("sigma")
        ax[1, j].plot(c.sigma, c.bcrb_rician_expectedF_fullprior_pct, "k--", lw=1.3, label="information limit (BCRB; lower bound on error)")
        ax[0, j].axhline(0.95, color="#777", lw=0.9, ls=":")
        ax[0, j].set_ylim(0.6, 1.0)
        ax[1, j].set_ylim(0, 60)
        ax[0, j].set_title(LAB[6 + j], fontsize=13)
        ax[1, j].set_xlabel("test noise σ")
        ax[1, j].set_xticks(sig)
        ax[1, j].set_xticklabels(["0.01", "0.03", "0.05", "0.08"])
        for r in range(2):
            ax[r, j].grid(alpha=0.3)
        if j:
            ax[0, j].set_yticklabels([]); ax[1, j].set_yticklabels([])
    ax[0, 0].set_ylabel("raw 95% coverage")
    ax[1, 0].set_ylabel("width, % of prior box")
    h, l = ax[1, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.0), fontsize=9.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    NUM["fig4"] = {"sigma": sig, "params": P[6:], **out,
                   "bcrb_width": {p: crb[crb.param == p].sort_values("sigma").bcrb_rician_expectedF_fullprior_pct.round(1).tolist() for p in P[6:]}}
    return save(fig, "fig4_sigma_sweep.png")


# ─────────────────────────────────────────────────────────────── Fig. 1 schematic
def fig1():
    fig = plt.figure(figsize=(9, 6), dpi=200)  # 1800 x 1200 px
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 18); ax.set_ylim(0, 12); ax.axis("off")
    fs = 14
    dark = "#333333"

    def grid(x0, y0, size, color="#B0B0B0"):
        c = size / 4
        for i in range(4):
            for j in range(4):
                ax.add_patch(Rectangle((x0 + j * c, y0 + i * c), c * 0.88, c * 0.88, color=color, lw=0))

    def box(x, y, w, h, text, fc="#F3F3F3", ec=dark, size=fs, lw=1.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, linespacing=1.15)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=dark, shrinkA=0, shrinkB=0))

    yc = 8.6
    grid(0.6, yc - 0.75, 1.3)
    ax.text(1.25, yc - 1.05, "16 signals\n4 b × 4 ΔTE", ha="center", va="top", fontsize=fs)
    arrow(1.98, yc, 2.42, yc)
    ax.add_patch(Polygon([[2.48, yc - 1.5], [2.48, yc + 1.5], [4.1, yc + 0.75], [4.1, yc - 0.75]], closed=True, fc="#DDE7F5", ec=dark, lw=1.4))
    ax.text(3.25, yc, "encoder\n(MLP)", ha="center", va="center", fontsize=fs - 3)
    arrow(4.15, yc + 0.3, 4.45, yc + 1.05); arrow(4.15, yc - 0.3, 4.45, yc - 1.05)
    box(4.5, yc + 0.7, 2.6, 0.8, "μ (8–9-D)")
    box(4.5, yc - 2.0, 2.6, 1.6, "Σ = LLᵀ\n(Cholesky)\nor diag(σ²)")
    arrow(7.2, yc + 1.1, 7.75, yc + 0.45); arrow(7.2, yc - 1.2, 7.75, yc - 0.45)
    ax.add_patch(Circle((8.45, yc), 0.72, fc="#FFF4D6", ec=dark, lw=1.4))
    ax.text(8.45, yc, "z = μ\n+ Lε", ha="center", va="center", fontsize=fs)
    ys = [yc + 1.45, yc, yc - 1.55]
    labels = ["tanh → D (3)", "tanh → T2 (3)", "log-ratio →\nsoftmax → v (3)"]
    for yy, t in zip(ys, labels):
        arrow(9.2, yc, 9.65, yy)
        h = 1.25 if "\n" in t else 0.9
        box(9.7, yy - h / 2, 3.3, h, t)
        arrow(13.1, yy, 13.5, yc)
    box(13.55, yc - 1.2, 2.5, 2.4, "fixed\nsignal\nequation\nS(D, T2, v)", fc="#DFF0D8", ec="#2E7D32", lw=2)
    ax.text(14.8, yc - 1.45, "physics,\nno weights", ha="center", va="top", fontsize=12.5, color="#2E7D32")
    arrow(16.15, yc, 16.45, yc)
    grid(16.5, yc - 0.6, 1.2)
    ax.text(17.1, yc - 0.9, "Ŝ", ha="center", va="top", fontsize=fs + 2)
    # loss block
    ax.plot([13.55, 13.55, 17.7, 17.7], [5.75, 5.5, 5.5, 5.75], color=dark, lw=1.3)
    ax.text(17.7, 5.05, "loss = −log p(x | θ) + KL[q ‖ N(0, I)]", ha="right", va="top", fontsize=fs)
    ax.text(17.7, 4.15, "original: MSE + 0.1·KL ⇒ implied noise SD 0.9", ha="right", va="top", fontsize=fs, color="#C62828")
    ax.text(17.7, 3.3, "matched: Gaussian SD = noise SD (50, or per voxel)", ha="right", va="top", fontsize=fs, color="#2E7D32")
    ax.text(17.7, 2.45, "true noise SD 10–80 (signal units)", ha="right", va="top", fontsize=fs, color=dark)
    # prior box table, bottom left
    ax.text(0.3, 5.05, "prior box", ha="left", va="top", fontsize=fs, weight="bold")
    ax.text(0.3, 4.35, "(centre ± half-width)", ha="left", va="top", fontsize=12.5)
    rows = [("", "D (µm²/ms)", "T2 (ms)"), ("ep", "0.5 ± 0.2", "45 ± 25"), ("st", "1.2 ± 0.5", "70 ± 30"), ("lum", "2.85 ± 0.15", "750 ± 250")]
    xs = [0.3, 1.35, 4.0]
    for i, r in enumerate(rows):
        for j, t in enumerate(r):
            ax.text(xs[j], 3.45 - i * 0.78, t, ha="left", va="center", fontsize=fs, weight="bold" if (i == 0 or j == 0) else "normal")
    ax.plot([0.3, 6.0], [3.05, 3.05], color="#999999", lw=0.8)
    return _save_dpi(fig, "fig1_schematic.png", 200)


def _save_dpi(fig, name, dpi):
    f = OUT / name
    fig.savefig(f, dpi=dpi, facecolor="white")
    plt.close(fig)
    return f


# ─────────────────────────────────────────────────────────────── Fig. 5 phantom
def make_phantom(seed=2027, n=64, radius=12):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    disc = ((xx - (n - 1) / 2) ** 2 + (yy - (n - 1) / 2) ** 2) <= radius ** 2
    # mean v of Dirichlet(1 + 18 m) is (1 + 18 m) / 21; solve for m given target region means
    v_bg, v_ca = np.array([BENIGN, 0.45, 1 - BENIGN - 0.45]), np.array([CANCER, 0.40, 1 - CANCER - 0.40])
    m_bg, m_ca = (21 * v_bg - 1) / 18, (21 * v_ca - 1) / 18
    N = n * n
    dflat = disc.ravel()
    v = np.empty((N, 3), np.float32)
    v[~dflat] = rng.dirichlet(1 + 18 * m_bg, size=(~dflat).sum())
    v[dflat] = rng.dirichlet(1 + 18 * m_ca, size=dflat.sum())
    # one "patient": generator's patient shift + voxel spread, clipped to the box
    d_shift = rng.normal(0, [0.025, 0.060, 0.020]); t2_shift = rng.normal(0, [4.0, 6.0, 55.0])
    pD = np.clip(pc.D_MEAN_3 + d_shift, pc.D_LOW_3, pc.D_HIGH_3); pT = np.clip(pc.T2_MEAN_3 + t2_shift, pc.T2_LOW_3, pc.T2_HIGH_3)
    D = np.clip(rng.normal(pD, pc.D_VOXEL_SD_3, size=(N, 3)), pc.D_LOW_3, pc.D_HIGH_3).astype(np.float32)
    T2 = np.clip(rng.normal(pT, pc.T2_VOXEL_SD_3, size=(N, 3)), pc.T2_LOW_3, pc.T2_HIGH_3).astype(np.float32)
    S = pc.pia3_signal_np(D, T2, v)
    X = pc.add_rician_noise(S / pc.SIGNAL_SCALE, 0.05, seed=seed + 1)
    y = np.concatenate([D, T2, v], 1)
    return X, y, disc


def run_model(tag, seed, X):
    if tag == "chol":
        m = pc.PIVA_CholeskyILR()
        path = HERE / "runs" / f"chol_s{seed}.pt"
    else:
        cov, vmap = ("chol", "ilr")
        m = em.GenericPIVA(cov=cov, vmap=vmap)
        path = HERE / "runs" / f"{tag}_s{seed}.pt"
    m.load_state_dict(torch.load(path, weights_only=True))
    return pc.mc_predict_piva(m, X, n_mc=25, seed=1234)


def fig5():
    X, y, disc = make_phantom()
    d = disc.ravel()
    tags = [("chol", "original loss"), ("gen-chol-ilr-gauss50-clean", "matched SD 50"), ("gen-chol-ilr-gaussPV-clean", "matched per-voxel SD")]
    stats, maps = {}, {}
    for tag, lab in tags:
        rows = []
        for s in range(3):
            mean, sd = run_model(tag, s, X)
            hw = 1.96 * sd
            if s == 0:
                maps[tag] = (mean, hw)
            j = 6
            lo, hi = mean[:, j] - hw[:, j], mean[:, j] + hw[:, j]
            cov = (y[:, j] >= lo) & (y[:, j] <= hi)
            t = 4  # T2_st
            covt = (y[:, t] >= mean[:, t] - hw[:, t]) & (y[:, t] <= mean[:, t] + hw[:, t])
            rows.append({
                "vep_mean_bg": mean[~d, j].mean(), "vep_mean_disc": mean[d, j].mean(),
                "vep_contrast": mean[d, j].mean() - mean[~d, j].mean(),
                "vep_halfwidth_bg": hw[~d, j].mean(), "vep_halfwidth_disc": hw[d, j].mean(),
                "vep_cov_bg": cov[~d].mean(), "vep_cov_disc": cov[d].mean(),
                "disc_excludes_benign": ((lo[d] > BENIGN) | (hi[d] < BENIGN)).mean(),
                "bg_excludes_cancer": ((lo[~d] > CANCER) | (hi[~d] < CANCER)).mean(),
                "T2st_mean": mean[:, t].mean(), "T2st_sd_of_means": mean[:, t].std(),
                "T2st_halfwidth": hw[:, t].mean(), "T2st_cov": covt.mean(),
                "T2st_corr_true": float(np.corrcoef(mean[:, t], y[:, t])[0, 1]),
            })
        df = pd.DataFrame(rows)
        stats[lab] = {"seed0": {k: round(float(v), 4) for k, v in df.iloc[0].items()},
                      "mean_3seeds": {k: round(float(v), 4) for k, v in df.mean().items()},
                      "sd_3seeds": {k: round(float(v), 4) for k, v in df.std(ddof=1).items()}}
    truth = {"vep_true_bg": float(y[~d, 6].mean()), "vep_true_disc": float(y[d, 6].mean()),
             "vep_true_contrast": float(y[d, 6].mean() - y[~d, 6].mean()),
             "T2st_true_mean": float(y[:, 4].mean()), "T2st_true_sd": float(y[:, 4].std()),
             "n_disc": int(d.sum()), "n_bg": int((~d).sum()), "phantom_seed": 2027, "noise_seed": 2028, "sigma": 0.05}
    NUM["fig5"] = {"truth": truth, **stats}

    n = disc.shape[0]
    img = lambda a: a.reshape(n, n)
    mo, mm = maps["chol"], maps["gen-chol-ilr-gauss50-clean"]
    rows = [(6, (0, 0.7), (0, 0.3), r"$v_\mathrm{ep}$", "", [0, 0.35, 0.7], [0, 0.15, 0.3]),
            (4, (40, 100), (0, 40), r"$T_{2,\mathrm{st}}$", " (ms)", [40, 70, 100], [0, 20, 40])]
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(12.5, 5.4))
    gs = GridSpec(2, 9, figure=fig, width_ratios=[1, 1, 1, 1, 1, 0.1, 0.06, 0.32, 0.06], wspace=0.06, hspace=0.12,
                  left=0.05, right=0.93, top=0.86, bottom=0.03)
    titles = ["truth", "mean\noriginal loss", "95% half-width\noriginal loss", "mean\nmatched SD 50", "95% half-width\nmatched SD 50"]
    for r, (j, vr, hr, name, unit, tv, th) in enumerate(rows):
        panels = [(img(y[:, j]), vr, "viridis"), (img(mo[0][:, j]), vr, "viridis"), (img(mo[1][:, j]), hr, "magma"),
                  (img(mm[0][:, j]), vr, "viridis"), (img(mm[1][:, j]), hr, "magma")]
        ims = []
        for c, (a, (lo, hi), cm) in enumerate(panels):
            axc = fig.add_subplot(gs[r, c])
            ims.append(axc.imshow(a, vmin=lo, vmax=hi, cmap=cm, interpolation="nearest"))
            axc.set_xticks([]); axc.set_yticks([])
            for sp in axc.spines.values():
                sp.set_visible(False)
            if r == 0:
                axc.set_title(titles[c], fontsize=13)
            axc.text(2, 2, "abcde"[c] if r == 0 else "fghij"[c], color="white", fontsize=14, weight="bold", va="top")
            if c == 0:
                axc.set_ylabel(name + unit, fontsize=15)
        cax1 = fig.add_subplot(gs[r, 6]); cax2 = fig.add_subplot(gs[r, 8])
        cb1 = fig.colorbar(ims[0], cax=cax1, ticks=tv); cb2 = fig.colorbar(ims[2], cax=cax2, ticks=th)
        cb1.ax.tick_params(labelsize=11)
        cb2.ax.tick_params(labelsize=11)
        cb1.set_label("mean", fontsize=11); cb2.set_label("95% half-width", fontsize=11)
    f = save(fig, "fig5_phantom.png")

    # preview: 2 x 3, 3:2
    fig = plt.figure(figsize=(9, 6))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 0.07], wspace=0.08, hspace=0.12, left=0.01, right=0.87, top=0.89, bottom=0.04)
    j = 6
    pv = [[(img(y[:, j]), (0, 0.7), "viridis", "truth"), (img(mo[0][:, j]), (0, 0.7), "viridis", "original"),
           (img(mm[0][:, j]), (0, 0.7), "viridis", "matched")],
          [None, (img(mo[1][:, j]), (0, 0.3), "magma", ""), (img(mm[1][:, j]), (0, 0.3), "magma", "")]]
    for r in range(2):
        for c in range(3):
            a = fig.add_subplot(gs[r, c])
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
            if pv[r][c] is None:
                a.axis("off")
                a.text(0.5, 0.5, "epithelium\nfraction:\nestimate (top)\nand 95%\nuncertainty\n(bottom)", ha="center", va="center", fontsize=21, transform=a.transAxes, linespacing=1.15)
                continue
            arr, (lo, hi), cm, t = pv[r][c]
            im = a.imshow(arr, vmin=lo, vmax=hi, cmap=cm, interpolation="nearest")
            if t:
                a.set_title(t, fontsize=26)
        cb = fig.colorbar(im, cax=fig.add_subplot(gs[r, 3]), ticks=[0, 0.35, 0.7] if r == 0 else [0, 0.15, 0.3])
        cb.ax.tick_params(labelsize=21)
    fp = OUT / "preview_3x2.png"
    fig.savefig(fp, dpi=200, facecolor="white")
    plt.close(fig)
    return f, fp


# ─────────────────────────────────────────────────────────────── A6 strata
def strata():
    ve = Y[:, 6]
    q25, q50, q75 = np.percentile(ve, [25, 50, 75])
    defs = {
        "v_ep < 0.18": ve < 0.18,
        "near benign: 0.18 < v_ep < 0.28": (ve > 0.18) & (ve < 0.28),
        "near cancer-like: 0.44 < v_ep < 0.54": (ve > 0.44) & (ve < 0.54),
        f"Q1: v_ep < {q25:.3f}": ve < q25,
        f"Q2: {q25:.3f}-{q50:.3f}": (ve >= q25) & (ve < q50),
        f"Q3: {q50:.3f}-{q75:.3f}": (ve >= q50) & (ve < q75),
        f"Q4: v_ep >= {q75:.3f}": ve >= q75,
        "all": np.ones_like(ve, bool),
    }
    out = {"quartiles": [round(float(q), 4) for q in (q25, q50, q75)]}
    for tag, lab in [("gen-chol-ilr-gauss50-clean", "CholeskyILR SD-50"), ("gen-chol-ilr-gaussPV-clean", "CholeskyILR per-voxel SD"),
                     ("chol", "CholeskyILR original loss")]:
        res = {}
        for sname, mask in defs.items():
            cov, bias = [], []
            for s in range(3):
                m, sd = load(tag, s)
                c, _ = interval_stats(m, sd)
                cov.append(c[mask, 6:].mean(0)); bias.append((m[mask, 6] - Y[mask, 6]).mean())
            cov = np.array(cov)
            res[sname] = {"n": int(mask.sum()), "cov_v_ep_v_st_v_lum_mean": cov.mean(0).round(3).tolist(),
                          "cov_sd": cov.std(0, ddof=1).round(3).tolist(), "v_ep_bias_mean": round(float(np.mean(bias)), 3)}
        out[lab] = res
    NUM["A6"] = out


if __name__ == "__main__":
    files = [fig1(), fig2(), fig3(), fig4(), *fig5()]
    strata()
    (OUT / "final_numbers.json").write_text(json.dumps(NUM, indent=1))
    from PIL import Image
    for f in files:
        im = Image.open(f)
        print(f.name, im.size, f"{f.stat().st_size / 1024:.0f} KB")
