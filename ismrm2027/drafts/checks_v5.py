"""Writer checks for abstract_v5: 3-seed clinical anchor (exclusion rate, contrast) and
3-seed SD-50/per-voxel clipped D/T2 widths. Run: conda run -n pvae python checks_v5.py"""
import sys
import numpy as np
A = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "analysis")
sys.path.insert(0, A)
import piva_core as pc  # noqa: E402
y = pc.build_test_cohort("large2000")["y"]
ve = y[:, 6]
ben, can = (ve > 0.18) & (ve < 0.28), (ve > 0.44) & (ve < 0.54)
print(f"n benign-like {ben.sum()}, cancer-like {can.sum()}")
for tag in ["chol", "gen-chol-ilr-gauss50-clean", "gen-chol-ilr-gaussPV-clean"]:
    rows = []
    for s in range(3):
        d = np.load(f"{A}/preds/{tag}_s{s}_large2000_sig0.05.npz")
        m, sd = d["mean"][:, 6], d["sd"][:, 6]
        lo, hi = m - 1.96 * sd, m + 1.96 * sd
        rows.append([m[ben].mean(), m[can].mean(), m[can].mean() - m[ben].mean(),
                     ((ve[can] >= lo[can]) & (ve[can] <= hi[can])).mean(),
                     ((lo[can] > 0.232) | (hi[can] < 0.232)).mean(),
                     ((lo[ben] > 0.488) | (hi[ben] < 0.488)).mean()])
    r = np.array(rows)
    print(tag, "per seed [est benign, est cancer, contrast, cancer cov, cancer excl benign, benign excl cancer]")
    print(np.round(r, 3)); print("mean", np.round(r.mean(0), 3), "sd", np.round(r.std(0, ddof=1), 3))
