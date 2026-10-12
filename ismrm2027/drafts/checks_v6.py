"""Writer check for abstract_v6 (A2): per-voxel CholeskyILR fraction RMSE vs BCRB-implied SD
at each test sigma, 3 seeds. Run: conda run -n pvae python checks_v6.py"""
import sys
import numpy as np
import pandas as pd
A = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "analysis")
sys.path.insert(0, A)
import piva_core as pc  # noqa: E402
y = pc.build_test_cohort("large2000")["y"]
crb = pd.read_csv(f"{A}/results/crb_widths.csv")
for tag in ["gen-chol-ilr-gaussPV-clean", "gen-chol-ilr-gauss50-clean"]:
    for sig in [0.01, 0.03, 0.05, 0.08]:
        c = crb[crb.sigma == sig].set_index("param")
        bsd = np.array([c.loc[p, "bcrb_rician_expectedF_fullprior_pct"] for p in ["v_ep", "v_st", "v_lum"]]) / 100 / 3.92
        r = []
        for s in range(3):
            m = np.load(f"{A}/preds/{tag}_s{s}_large2000_sig{sig}.npz")["mean"][:, 6:9]
            r.append(np.sqrt(((m - y[:, 6:9]) ** 2).mean(0)))
        r = np.mean(r, 0)
        print(f"{tag} sigma={sig}: RMSE v_ep/st/lum {np.round(r,3)}  BCRB SD {np.round(bsd,3)}  ratio {np.round(r/bsd,2)}")
