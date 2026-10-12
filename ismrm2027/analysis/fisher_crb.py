"""
Experiment E: information-limit widths, reviewing support_scripts/fisher2.py.

fisher2.py computes a Bayesian (van Trees style) bound sqrt(diag(inv(F + P^-1))) with
  * F = J^T J / sigma^2 at ONE point (prior centre, v = 1/3 each), Gaussian noise;
  * P = diagonal Gaussian prior from the test-cohort marginal SDs; 8 params (no v_lum).
This script reproduces that number and then removes the three approximations:
  (1) F averaged over the actual cohort voxels (expected Fisher, as the van Trees bound requires),
  (2) Rician instead of Gaussian per-channel Fisher information,
  (3) full prior covariance from the training set (v_ep and v_st are strongly anti-correlated),
and adds v_lum via the delta method (v_lum = 1 - v_ep - v_st).
Output: results/crb_widths.csv with 95% width (3.92 * SD) as % of the physiological range.
"""
import json
from pathlib import Path

import numpy as np
from scipy.special import i0e, i1e

import piva_core as pc

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
b, te = pc.b_grid, pc.te_grid
RNG8 = pc.FULL_RANGE[:8].astype(float)


def sig(p):
    D, T = p[0:3], p[3:6]
    v = np.array([p[6], p[7], 1 - p[6] - p[7]])
    return (v[None, :] * np.exp(-b[:, None] * D[None, :] / 1000) * np.exp(-te[:, None] / T[None, :])).sum(1)


def jac(p):
    J = np.zeros((16, 8))
    for k in range(8):
        h = 1e-5 * RNG8[k]
        pp, pm = p.copy(), p.copy()
        pp[k] += h
        pm[k] -= h
        J[:, k] = (sig(pp) - sig(pm)) / (2 * h)
    return J


# Rician Fisher information for nu, relative to the Gaussian value 1/sigma^2, as a function of snr = nu/sigma.
def rician_ratio_table():
    snrs = np.linspace(0, 40, 401)
    x = np.linspace(1e-6, 60, 60001)  # x in units of sigma
    dx = x[1] - x[0]
    out = []
    for a in snrs:
        z = x * a
        logp = np.log(x) - (x ** 2 + a ** 2) / 2 + np.log(i0e(z)) + z
        p = np.exp(logp)
        score = -a + x * i1e(z) / i0e(z)
        out.append(np.sum(p * score ** 2) * dx)
    return snrs, np.array(out)


SNRS, RATIO = rician_ratio_table()


def fisher(p, sigma, noise="gauss"):
    J = jac(p)
    if noise == "gauss":
        w = np.ones(16)
    else:
        w = np.interp(sig(p) / sigma, SNRS, RATIO, right=1.0)
    return (J * w[:, None]).T @ J / sigma ** 2


def to9(C8):
    """8x8 covariance of (D, T2, v_ep, v_st) -> 9 SDs including v_lum = 1 - v_ep - v_st."""
    a = np.zeros(8)
    a[6] = a[7] = -1
    return np.r_[np.sqrt(np.diag(C8)), np.sqrt(a @ C8 @ a)]


def main():
    res = {}
    # (0) reproduce fisher2.py exactly
    sd_pop = np.array([0.047, 0.139, 0.035, 7.4, 9.9, 88.5, 0.17, 0.20])
    p0 = np.r_[pc.D_MEAN_3, pc.T2_MEAN_3, 1 / 3, 1 / 3].astype(float)
    rows = []
    _, _, _, _, y_tr, _ = pc.generate_patient_clean_cohort(80, 512, seed=0)
    th_tr = y_tr[:, :8].astype(float)
    P_full = np.cov(th_tr.T)
    test = pc.generate_patient_clean_cohort(200, 10, seed=2027)[4][:, :8].astype(float)
    for sigma in [0.01, 0.03, 0.05, 0.08]:
        F0 = fisher(p0, sigma)
        fisher2 = to9(np.linalg.inv(F0 + np.diag(1 / sd_pop ** 2)))
        prior_only = to9(np.diag(sd_pop ** 2))
        # expected Fisher over cohort voxels
        Fg = np.mean([fisher(t, sigma, "gauss") for t in test], axis=0)
        Fr = np.mean([fisher(t, sigma, "rician") for t in test], axis=0)
        bcrb_g_diag = to9(np.linalg.inv(Fg + np.diag(1 / sd_pop ** 2)))
        bcrb_r_full = to9(np.linalg.inv(Fr + np.linalg.inv(P_full)))
        prior_full = to9(P_full)
        # per-voxel local (Laplace-style) posterior SD with Rician Fisher + full prior, averaged as RMS
        loc = np.sqrt(np.mean([to9(np.linalg.inv(fisher(t, sigma, "rician") + np.linalg.inv(P_full))) ** 2
                               for t in test], axis=0))
        # unregularised CRB (no prior) at expected Fisher, Rician
        crb_noprior = to9(np.linalg.inv(Fr))
        for j, name in enumerate(pc.PARAM_NAMES):
            pct = lambda s: float(100 * 3.92 * s[j] / pc.FULL_RANGE[j])
            rows.append({"sigma": sigma, "param": name,
                         "fisher2_reproduced_pct": pct(fisher2), "fisher2_prior_only_pct": pct(prior_only),
                         "bcrb_gauss_expectedF_diagprior_pct": pct(bcrb_g_diag),
                         "bcrb_rician_expectedF_fullprior_pct": pct(bcrb_r_full),
                         "local_rician_fullprior_rms_pct": pct(loc),
                         "prior_full_only_pct": pct(prior_full),
                         "crb_rician_noprior_pct": pct(crb_noprior),
                         "shrink_ratio_bcrb_rician_full": float(bcrb_r_full[j] / prior_full[j])})
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "crb_widths.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(df[df.sigma == 0.05].round(2).to_string(index=False))
    # sanity: Rician/Gaussian Fisher ratio at a few SNRs
    print("Rician/Gauss Fisher ratio at snr 0,0.5,1,2,5,20:", np.round(np.interp([0, .5, 1, 2, 5, 20], SNRS, RATIO), 3))


if __name__ == "__main__":
    main()
