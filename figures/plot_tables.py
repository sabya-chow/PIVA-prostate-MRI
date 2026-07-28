"""
Publication tables — booktabs style (three-line: top, midrule, bottom).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def make_table(rows, col_labels, title, subtitle, filename,
               figsize=(6.5, 2.2), bold_last_row=True, fontsize=9.5):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc='center',
        colLoc='center',
        loc='center',
        edges='open',  # no cell borders at all
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.0, 1.55)

    n_cols = len(col_labels)
    n_rows = len(rows)

    for j in range(n_cols):
        cell = table[0, j]
        cell.set_text_props(fontweight='bold', fontsize=fontsize)
        cell.set_facecolor('white')
        if j == 0:
            cell._loc = 'left'
            cell.PAD = 0.03

    for i in range(n_rows):
        is_last = (i == n_rows - 1) and bold_last_row
        for j in range(n_cols):
            cell = table[i + 1, j]
            cell.set_facecolor('white')
            if is_last:
                cell.set_text_props(fontweight='bold')
            if j == 0:
                cell._loc = 'left'
                cell.PAD = 0.03

    # Draw three rules manually: top, midrule, bottom
    bbox = table.get_window_extent(fig.canvas.get_renderer())
    inv = ax.transData.inverted()
    bb = inv.transform(bbox)
    x0, x1 = bb[0][0], bb[1][0]

    # Get y positions from cell bboxes
    header_cell = table[0, 0]
    first_data = table[1, 0]
    last_data = table[n_rows, 0]

    def cell_y(cell, edge='bottom'):
        cb = cell.get_window_extent(fig.canvas.get_renderer())
        cb_data = inv.transform(cb)
        return cb_data[0][1] if edge == 'bottom' else cb_data[1][1]

    y_top = cell_y(header_cell, 'top')
    y_mid = cell_y(header_cell, 'bottom')
    y_bot = cell_y(last_data, 'bottom')

    lw_thick = 1.2
    lw_thin = 0.6

    ax.plot([x0, x1], [y_top, y_top], color='black', lw=lw_thick, clip_on=False)
    ax.plot([x0, x1], [y_mid, y_mid], color='black', lw=lw_thin, clip_on=False)
    ax.plot([x0, x1], [y_bot, y_bot], color='black', lw=lw_thick, clip_on=False)

    fig.text(0.5, 0.97, title, ha='center', va='top',
             fontsize=11, fontweight='bold', family='serif')
    if subtitle:
        fig.text(0.5, 0.90, subtitle, ha='center', va='top',
                 fontsize=7.5, color='#666666', family='serif', style='italic')

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    out = os.path.join(SCRIPT_DIR, filename)
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close()
    print(f"Saved: {out}")


# ── Table 1: Inference Time ──
make_table(
    rows=[
        ["NLLS",              "187.51",  "\u2014",     "1\u00d7"],
        ["MLP-PIA",           "0.480",   "0.073", "2,584\u00d7"],
        ["CNN-PIA",           "0.102",   "0.008", "23,185\u00d7"],
        ["MLP-PIA + Refiner", "0.792",   "0.095", "1,974\u00d7"],
        ["CNN-PIA + Refiner", "0.389",   "0.029", "6,431\u00d7"],
    ],
    col_labels=["Method", "CPU (s)", "GPU (s)", "Speedup"],
    title="Inference Time per Patient (200\u00d7200 slice)",
    subtitle="GPU: NVIDIA Quadro T2000",
    filename="inference_time_table.png",
)


# ── Table 2: rRMSE ──
JSON_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "results", "noise_evaluation_results_e2e.json")
with open(JSON_PATH) as f:
    data = json.load(f)

sigma = np.array(data["noise_levels"])
selected = [0.02, 0.05, 0.1, 0.2]
indices = [list(sigma).index(s) for s in selected]

methods = [
    ("NLLS",              "nlls"),
    ("MLP-PIA",           "mlp"),
    ("MLP-PIA + Refiner", "mlp_ref"),
    ("CNN-PIA",           "cnn"),
    ("CNN-PIA + Refiner", "cnn_ref"),
]

col_labels = ["Method"]
for s in selected:
    snr = int(round(1.0 / s))
    col_labels.append(f"\u03c3={s} (SNR {snr})")

rrmse_rows = []
for label, key in methods:
    row = [label]
    for idx in indices:
        total = data[f"{key}_total"][idx]
        tumor = data[f"{key}_tumor"][idx]
        row.append(f"{total:.3f} / {tumor:.3f}")
    rrmse_rows.append(row)

make_table(
    rows=rrmse_rows,
    col_labels=col_labels,
    title="rRMSE at Clinical Noise Levels (Total / Tumor)",
    subtitle="Lower is better \u2022 400 test patients",
    filename="rrmse_comparison_table.png",
    figsize=(8, 2.4),
    fontsize=9,
)

print("Done.")
