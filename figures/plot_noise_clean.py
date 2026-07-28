"""
Clean, publication-quality rRMSE vs Noise plots.
Figure 1: Full range log-log
Figure 2: Realistic noise zoom with human-readable x-axis labels
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "results", "noise_evaluation_results_e2e.json")

with open(JSON_PATH) as f:
    data = json.load(f)

sigma = np.array(data["noise_levels"])

methods = [
    ("NLLS",              "nlls",     "s", "--",  1.8, 5.5, "black",   "white"),
    ("MLP-PIA",           "mlp",      "^", ":",   1.2, 5,   "#888888", "white"),
    ("MLP-PIA + Refiner", "mlp_ref",  "v", "-.",  1.2, 5,   "#555555", "white"),
    ("CNN-PIA",           "cnn",      "D", ":",   1.2, 4.5, "#999999", "white"),
    ("CNN-PIA + Refiner", "cnn_ref",  "o", "-",   2.0, 5.5, "black",  "black"),
]


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(colors="#333333", labelsize=9)
    ax.grid(True, which="major", color="#d4d4d4", lw=0.5)
    ax.grid(True, which="minor", color="#eeeeee", lw=0.3, ls=":")


def plot_on_ax(ax, region, legend=False):
    for label, key, marker, ls, lw, ms, color, fill in methods:
        y = np.array(data[f"{key}_{region}"])
        z = 10 if label in ("NLLS", "CNN-PIA + Refiner") else 5
        ax.plot(sigma, y, color=color, marker=marker, ls=ls,
                lw=lw, ms=ms, label=label, zorder=z,
                markerfacecolor=fill, markeredgecolor=color,
                markeredgewidth=1.1)
    if legend:
        ax.legend(fontsize=8, frameon=True, facecolor="white",
                  edgecolor="#cccccc", loc="upper left",
                  borderpad=0.7, handlelength=2.5)


# ── Figure 1: Full range log-log ──
def make_full_plot(region, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    plot_on_ax(ax, region, legend=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Noise Level ($\\sigma$)", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    style_ax(ax)
    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, filename)
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 2: Realistic noise zoom with human-readable labels ──

# Realistic subset: sigma >= 0.02
REALISTIC_IDX = [i for i, s in enumerate(sigma) if s >= 0.02]
REALISTIC_SIGMA = sigma[REALISTIC_IDX]

# Human-readable labels: sigma value + SNR + clinical description
HUMAN_LABELS = {
    0.02:  "σ=0.02\nSNR 50\nLow",
    0.033: "σ=0.033\nSNR 30\nMild",
    0.05:  "σ=0.05\nSNR 20\nModerate",
    0.10:  "σ=0.10\nSNR 10\nHigh",
    0.15:  "σ=0.15\nSNR 7\nVery High",
    0.20:  "σ=0.20\nSNR 5\nSevere",
    0.25:  "σ=0.25\nSNR 4\nExtreme",
}


def make_zoom_plot(region, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(8, 5))

    x_pos = np.arange(len(REALISTIC_SIGMA))

    for label, key, marker, ls, lw, ms, color, fill in methods:
        y_full = np.array(data[f"{key}_{region}"])
        y = y_full[REALISTIC_IDX]
        z = 10 if label in ("NLLS", "CNN-PIA + Refiner") else 5
        ax.plot(x_pos, y, color=color, marker=marker, ls=ls,
                lw=lw, ms=ms, label=label, zorder=z,
                markerfacecolor=fill, markeredgecolor=color,
                markeredgewidth=1.1)

    # Cap y-axis to DL range so the separation is clear
    dl_max = max(
        max(np.array(data[f"mlp_{region}"])[REALISTIC_IDX]),
        max(np.array(data[f"mlp_ref_{region}"])[REALISTIC_IDX]),
        max(np.array(data[f"cnn_{region}"])[REALISTIC_IDX]),
        max(np.array(data[f"cnn_ref_{region}"])[REALISTIC_IDX]),
    )
    ax.set_ylim(-0.02, dl_max * 1.15)

    # Human-readable x-axis
    tick_labels = [HUMAN_LABELS.get(s, f"σ={s}") for s in REALISTIC_SIGMA]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tick_labels, fontsize=8, ha="center")
    ax.set_xlabel("Noise Condition", fontsize=11, fontweight="bold")

    # Note that NLLS goes off chart
    nlls_vals = np.array(data[f"nlls_{region}"])[REALISTIC_IDX]
    for i, v in enumerate(nlls_vals):
        if v > dl_max * 1.15:
            ax.annotate(f"{v:.1f}", xy=(x_pos[i], dl_max * 1.1),
                        fontsize=7, ha="center", va="bottom", color="black",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", lw=0.5))

    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    style_ax(ax)
    ax.grid(visible=False, which="minor")

    ax.legend(fontsize=8, frameon=True, facecolor="white",
              edgecolor="#cccccc", loc="upper left",
              borderpad=0.7, handlelength=2.5)

    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, filename)
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out}")


# ── Generate all 4 figures ──
if __name__ == '__main__':
    make_full_plot("total", "rRMSE (Total Tissue)",
                   "rRMSE vs. Noise Level \u2014 Total Tissue",
                   "rrmse_noise_total_full.png")
    make_full_plot("tumor", "rRMSE (Tumor Core)",
                   "rRMSE vs. Noise Level \u2014 Tumor Core",
                   "rrmse_noise_tumor_full.png")

    make_zoom_plot("total", "rRMSE (Total Tissue)",
                   "rRMSE at Realistic Noise Levels \u2014 Total Tissue",
                   "rrmse_noise_total_zoom.png")
    make_zoom_plot("tumor", "rRMSE (Tumor Core)",
                   "rRMSE at Realistic Noise Levels \u2014 Tumor Core",
                   "rrmse_noise_tumor_zoom.png")

    print("Done.")
