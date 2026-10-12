"""NLLS (NB1 cell [10] logic) on every cohort and noise level. Saves predictions, valid mask, bound hits."""
import sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import piva_core as pc

OUT = Path(__file__).parent / "preds"
OUT.mkdir(exist_ok=True)

if __name__ == "__main__":
    cohorts = sys.argv[1:] or ["orig240", "large2000"]
    for c in cohorts:
        co = pc.build_test_cohort(c)
        for s in pc.SIGMA_TEST_LEVELS:
            t0 = time.time()
            with Pool(int(__import__("os").environ.get("NLLS_PROCS", "2"))) as pool:
                res = pool.map(pc.nlls_fit_one, list(co["X"][s]), chunksize=20)
            pred = np.stack([r[0] for r in res]); valid = np.array([r[1] for r in res]); nb = np.array([r[2] for r in res])
            np.savez(OUT / f"nlls_{c}_sig{s}.npz", pred=pred, valid=valid, n_at_bound=nb)
            print(f"{c} sigma={s}: n={len(valid)} failed={int((~valid).sum())} any_param_at_bound={int((nb>0).sum())} "
                  f"v_lum<0={int((pred[:,8]<0).sum())} [{time.time()-t0:.0f}s]", flush=True)
