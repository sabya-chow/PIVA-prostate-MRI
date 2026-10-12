"""
Evaluate every checkpoint in runs/ plus NLLS (preds/) and baselines on every cohort and noise level.

Writes:
  results/metrics_long.csv    one row per (model, seed, cohort, sigma, metric)
  results/calibration_long.csv per-parameter calibration results (v2.0 replica and separate-cohort tau)
  preds/<tag>_<cohort>_sig<s>.npz   posterior means / SDs for figures
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import piva_core as pc
import extras_models as em

torch.set_num_threads(1)
torch.set_flush_denormal(True)  # inference only; avoids the subnormal slowdown seen in training
HERE = Path(__file__).parent
import os
RUNS = HERE / os.environ.get("EVAL_RUNS", "runs")
PREDS = HERE / os.environ.get("EVAL_PREDS", "preds")
OUT = HERE / os.environ.get("EVAL_OUT", "results")
PREDS.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
N_MC, MC_SEED = 25, 1234
TEST_COHORTS = ["orig240", "large2000"]
CAL_FOR = {"orig240": "calib240", "large2000": "calib2000"}

COH = {c: pc.build_test_cohort(c) for c in ["orig240", "large2000", "calib240", "calib2000"]}
_, _, y_train = pc.build_training_set()
TRAIN_MEAN = y_train.mean(0)
FIXED_LO = np.percentile(y_train, 2.5, axis=0)
FIXED_HI = np.percentile(y_train, 97.5, axis=0)


def load_model(tag):
    meta = json.load(open(RUNS / f"{tag}.json"))
    a = meta["args"]
    if a["model"] == "pia":
        m = pc.PIA()
    elif a["model"] == "diag":
        m = pc.PIVA_Diagonal()
    elif a["model"] == "chol":
        m = pc.PIVA_CholeskyILR()
    else:
        m = em.GenericPIVA(cov=a["cov"], vmap=a["vmap"])
    m.load_state_dict(torch.load(RUNS / f"{tag}.pt", weights_only=True))
    m.eval()
    return m, a, meta


def model_label(a):
    if a["model"] == "pia":
        return "PIA"
    if a["model"] == "diag":
        return "PIVA-Diagonal"
    if a["model"] == "chol":
        return "PIVA-CholeskyILR"
    cov = "Chol" if a["cov"] == "chol" else "Diag"
    v = "ILR" if a["vmap"] == "ilr" else "softmax"
    if a.get("lik_mode") == "pervoxel":
        lik = f"{a['lik']}SD-pervoxel"
    else:
        lik = f"{a['lik']}SD{a['lik_sd']:g}" if a["lik_sd"] is not None else f"MSE+b{a['beta']:g}KL"
    return f"X:{cov}+{v}|{lik}|{a['target']}"


def tau_grid_fit(y, m, sd, grid, target=0.95):
    """Vectorised equivalent of NB2 calibrate_model's per-parameter grid search."""
    ratio = np.abs(y - m) / (1.96 * np.maximum(sd, 1e-12))  # covered at tau iff ratio <= tau
    taus = np.empty(y.shape[1])
    for j in range(y.shape[1]):
        r = np.sort(ratio[:, j])
        covs = np.searchsorted(r, grid, side="right") / len(r)
        taus[j] = grid[int(np.argmin(np.abs(covs - target)))]
    return taus


def patient_boot_ci(covered, pids, n_boot=1000, seed=0):
    """Patient-clustered bootstrap 95% CI of coverage, per parameter. covered: (N, 9) bool."""
    rng = np.random.default_rng(seed)
    up = np.unique(pids)
    idx_by_p = {p: np.where(pids == p)[0] for p in up}
    stats = []
    for _ in range(n_boot):
        samp = rng.choice(up, size=len(up), replace=True)
        idx = np.concatenate([idx_by_p[p] for p in samp])
        stats.append(covered[idx].mean(0))
    stats = np.array(stats)
    return np.percentile(stats, 2.5, axis=0), np.percentile(stats, 97.5, axis=0)


GRID_V20 = np.linspace(0.5, 25.0, 500)
GRID_WIDE = np.linspace(0.1, 200.0, 4000)


def calib_rows(label, seed, cohort, sigma, y, m, sd, pids, cal):
    rows = []
    # (i) exact replica of NB2 cell [18]: voxel-level split of the test cohort, tau grid [0.5, 25]
    r = pc.calibrate(y, m, sd, pids, split="voxel", grid=GRID_V20, seed=0)
    # (ii) same, wide grid, patient-level split
    r2 = pc.calibrate(y, m, sd, pids, split="patient", grid=GRID_WIDE, seed=0)
    # (iii) experiment C: tau fit on a separate calibration cohort, evaluated on the whole test cohort
    tau_c = tau_grid_fit(cal["y"], cal["m"], cal["sd"], GRID_WIDE)
    covered = (np.abs(y - m) <= 1.96 * tau_c * sd)
    lo, hi = patient_boot_ci(covered, pids)
    raw_cov = (np.abs(y - m) <= 1.96 * sd)
    rlo, rhi = patient_boot_ci(raw_cov, pids)
    for j, pn in enumerate(pc.PARAM_NAMES):
        rows.append({"model": label, "seed": seed, "cohort": cohort, "sigma": sigma, "param": pn,
                     "raw_cov": raw_cov[:, j].mean(), "raw_cov_lo": rlo[j], "raw_cov_hi": rhi[j],
                     "raw_width_pct": 100 * 3.92 * sd[:, j].mean() / pc.FULL_RANGE[j],
                     "v20_tau": r["tau"][j], "v20_tau_at_max": bool(r["tau_at_grid_max"][j]),
                     "v20_cal_cov": r["cal_cov"][j], "v20_width_pct": r["cal_width_pct"][j],
                     "pat_tau": r2["tau"][j], "pat_cal_cov": r2["cal_cov"][j], "pat_width_pct": r2["cal_width_pct"][j],
                     "sep_tau": tau_c[j], "sep_tau_at_max": bool(tau_c[j] >= GRID_WIDE[-2] - 1e-9),
                     "sep_cal_cov": covered[:, j].mean(), "sep_cal_cov_lo": lo[j], "sep_cal_cov_hi": hi[j],
                     "sep_width_pct": 100 * 3.92 * tau_c[j] * sd[:, j].mean() / pc.FULL_RANGE[j]})
    return rows


def metric_rows(label, seed, cohort, sigma, d, kind):
    return [{"model": label, "seed": seed, "cohort": cohort, "sigma": sigma, "kind": kind, "metric": k, "value": v}
            for k, v in d.items()]


def eval_tag(tag):
    long, cal = [], []
    m, a, meta = load_model(tag)
    label, seed = model_label(a), a["seed"]
    hist = meta["history"][-1]
    long += metric_rows(label, seed, "train", np.nan, {"final_signal_mse": hist["signal_mse"],
                                                       "final_kl": hist.get("kl", np.nan),
                                                       "train_seconds": meta["train_seconds"]}, "train")
    for c in TEST_COHORTS:
        y, pids = COH[c]["y"], COH[c]["pids"]
        for s in pc.SIGMA_TEST_LEVELS:
            X = COH[c]["X"][s]
            if a["model"] == "pia":
                mean = pc.predict_pia(m, X)
                long += metric_rows(label, seed, c, s, pc.per_param_metrics(y, mean), "acc")
                continue
            mean, sd = pc.mc_predict_piva(m, X, n_mc=N_MC, seed=MC_SEED)
            np.savez(PREDS / f"{tag}_{c}_sig{s}.npz", mean=mean, sd=sd)
            long += metric_rows(label, seed, c, s, pc.per_param_metrics(y, mean), "acc")
            long += metric_rows(label, seed, c, s, pc.uq_metrics(y, mean, sd), "uq")
            cm, csd = pc.mc_predict_piva(m, COH[CAL_FOR[c]]["X"][s], n_mc=N_MC, seed=MC_SEED + 1)
            cal += calib_rows(label, seed, c, s, y, mean, sd, pids,
                              {"y": COH[CAL_FOR[c]]["y"], "m": cm, "sd": csd})
    print("evaluated", tag, flush=True)
    return long, cal


def main():
    long, cal = [], []
    # baselines
    for c in TEST_COHORTS:
        y = COH[c]["y"]
        for s in pc.SIGMA_TEST_LEVELS:
            long += metric_rows("Const prior centre", 0, c, s, pc.per_param_metrics(y, np.broadcast_to(pc.PRIOR_CENTRE, y.shape)), "acc")
            long += metric_rows("Const training mean", 0, c, s, pc.per_param_metrics(y, np.broadcast_to(TRAIN_MEAN, y.shape)), "acc")
            cov = ((y >= FIXED_LO) & (y <= FIXED_HI)).mean(0)
            covd = {f"{p}_rawcov": float(cov[j]) for j, p in enumerate(pc.PARAM_NAMES)}
            covd.update({f"{p}_rawwidth_pct": float(100 * (FIXED_HI[j] - FIXED_LO[j]) / pc.FULL_RANGE[j])
                         for j, p in enumerate(pc.PARAM_NAMES)})
            long += metric_rows("Fixed interval (train 2.5-97.5%)", 0, c, s, covd, "uq")
            # NLLS
            f = PREDS / f"nlls_{c}_sig{s}.npz"
            if f.exists():
                z = np.load(f)
                d = pc.per_param_metrics(y, z["pred"], z["valid"])
                d.update({"n_failed": int((~z["valid"]).sum()), "n_any_bound": int((z["n_at_bound"] > 0).sum()),
                          "n_vlum_neg": int((z["pred"][:, 8] < 0).sum()), "n": len(z["valid"])})
                long += metric_rows("NLLS", 0, c, s, d, "acc")
    # trained models, one process per checkpoint
    from multiprocessing import Pool
    tags = sorted(js.stem for js in RUNS.glob("*.json"))
    with Pool(int(os.environ.get("EVAL_PROCS", "8"))) as pool:
        for lo, ca in pool.imap_unordered(eval_tag, tags):
            long += lo
            cal += ca
    pd.DataFrame(long).to_csv(OUT / "metrics_long.csv", index=False)
    pd.DataFrame(cal).to_csv(OUT / "calibration_long.csv", index=False)


if __name__ == "__main__":
    main()
