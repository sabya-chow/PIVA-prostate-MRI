"""Train one model with one seed and save the checkpoint.

Core audit (original recipe):
  python train_one.py --model {pia,diag,chol} --seed 0
Reviewer-requested experiments (changes the recipe; tag says how):
  python train_one.py --model generic --cov {diag,chol} --vmap {softmax,ilr} --seed 0
        [--lik_sd 50] [--lik {gauss,rician}] [--target {clean,noisy}] [--beta 0.1]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

import piva_core as pc

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["pia", "diag", "chol", "generic"])
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--out", default="runs")
ap.add_argument("--cov", default="diag")
ap.add_argument("--vmap", default="softmax")
ap.add_argument("--lik_sd", type=float, default=None)
ap.add_argument("--lik", default="gauss")
ap.add_argument("--target", default="clean")
ap.add_argument("--beta", type=float, default=0.1)
ap.add_argument("--lik_mode", default="fixed", choices=["fixed", "pervoxel"],
                help="pervoxel: Gaussian NLL with each training voxel's own simulated noise SD (ignores --lik_sd)")
args = ap.parse_args()

torch.set_num_threads(args.threads)
torch.set_num_interop_threads(1)
# Diagonal runs stalled on the first queue (epoch 60 then >2.5 h with no progress, stuck in backward
# matmuls), consistent with subnormal floats (e.g., vanishing gradients through saturated tanh).
# Benchmarked on this i9: subnormal inputs make sgemm ~225x slower; FTZ removes it.
# PIVA_NO_FTZ=1 disables it (used only for the A/B reproduction test).
if os.environ.get("PIVA_NO_FTZ") != "1":
    torch.set_flush_denormal(True)
out = Path(__file__).parent / args.out
out.mkdir(exist_ok=True)
if args.model == "generic":
    tag = f"gen-{args.cov}-{args.vmap}"
    if args.lik_mode == "pervoxel":
        tag += f"-{args.lik}PV"
    else:
        tag += f"-{args.lik}{args.lik_sd:g}" if args.lik_sd is not None else f"-mseb{args.beta:g}"
    tag += f"-{args.target}_s{args.seed}"
else:
    tag = f"{args.model}_s{args.seed}"
logf = open(out / f"{tag}.log", "w")


def log(msg):
    msg = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(msg, flush=True)
    logf.write(msg + "\n")
    logf.flush()
    # Abort a diverged run at the next 20-epoch log line instead of burning hours on NaNs.
    if "epoch" in msg and "nan" in msg.lower():
        logf.write("ABORT: NaN loss\n"); logf.flush()
        sys.exit(3)


X_train, S_clean_tr, y_train = pc.build_training_set()
log(f"{tag}: X_train {X_train.shape}  seed={args.seed}  threads={args.threads}  batch={pc.BATCH_SIZE}  args={vars(args)}")

pc.set_all_seeds(args.seed)          # seeds network init
if args.lik_mode == "pervoxel":
    import extras_models as em
    # third tensor carries each voxel's noise SD instead of y (y is only for monitoring)
    loader = pc.make_loader(X_train, S_clean_tr, em.training_sigmas(len(X_train)), seed=args.seed)
else:
    loader = pc.make_loader(X_train, S_clean_tr, y_train, seed=args.seed)  # seeds batch order
t0 = time.time()
if args.model == "pia":
    model = pc.PIA()
    hist = pc.train_pia(model, loader, n_epochs=args.epochs, log=log)
elif args.model == "diag":
    model = pc.PIVA_Diagonal()
    hist = pc.train_piva(model, loader, pc.kl_divergence_diag, n_epochs=args.epochs, log=log, label="PIVA-Diag")
elif args.model == "chol":
    model = pc.PIVA_CholeskyILR()
    hist = pc.train_piva(model, loader, pc.kl_divergence_full, n_epochs=args.epochs, log=log, label="PIVA-Chol")
else:
    import extras_models as em
    model = em.GenericPIVA(cov=args.cov, vmap=args.vmap)
    hist = em.train_piva_generic(model, loader, n_epochs=args.epochs, beta=args.beta,
                                 lik_sd="pervoxel" if args.lik_mode == "pervoxel" else args.lik_sd,
                                 target=args.target, lik=args.lik, log=log, label=tag)
dt = time.time() - t0
log(f"done in {dt:.1f}s")
torch.save(model.state_dict(), out / f"{tag}.pt")
json.dump({"tag": tag, "args": vars(args), "train_seconds": dt, "history": hist}, open(out / f"{tag}.json", "w"))
