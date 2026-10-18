#!/usr/bin/env python3
"""Fit the two directions from the saved states (CPU, seconds).

  theta (Outside Truth) = unit(mean(h | true statement) - mean(h | false statement))
  u     (unsureness)    = unit(mean(h | swapped: model_wrong + random_wrong) - mean(h | control))

The README's "Reasoned Truth" direction is -u. Steering is done along u and the
plots flip the sign of r.

For the steering test u is CROSS-FITTED: one u per fold in `swap.folds`, fit with
that fold's problems held out. directions/folds.json maps each problem to the
direction used when steering it.

    python scripts/04_build_directions.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from truthsteer import paths  # noqa: E402


def mean_diff(X, pos):
    diff = X[pos].mean(0) - X[~pos].mean(0)
    return diff / diff.norm(), float(diff.norm())


def main():
    c = paths.cfg()
    L = c["layer"]
    os.makedirs(paths.DIRECTIONS, exist_ok=True)

    Xt = torch.load(os.path.join(paths.STATES, f"truth_layer{L}.pt"))
    yt = torch.from_numpy(pd.read_parquet(os.path.join(paths.STATES, "truth_meta.parquet"))
                          ["label"].values.astype(bool))
    theta, d = mean_diff(Xt, yt)
    torch.save(theta, os.path.join(paths.DIRECTIONS, "theta.pt"))

    Xu = torch.load(os.path.join(paths.STATES, f"swap_layer{L}.pt"))
    mu = pd.read_parquet(os.path.join(paths.STATES, "swap_meta.parquet"))
    swapped = torch.from_numpy((mu.condition != "control").values)
    u, du = mean_diff(Xu, swapped)
    torch.save(u, os.path.join(paths.DIRECTIONS, "unsure.pt"))

    mapping, fold_cos = {}, {}
    for fold, held in c["swap"]["folds"].items():
        keep = torch.from_numpy(~mu.problem_id.isin(held).values)
        if not keep.any() or keep.all():
            continue           # smoke runs with only a few CoTs
        uf, _ = mean_diff(Xu[keep], swapped[keep])
        rel = f"directions/unsure_{fold}.pt"
        torch.save(uf, os.path.join(paths.DATA, rel))
        fold_cos[fold] = float(uf @ theta)
        mapping.update({p: rel for p in held})
    with open(os.path.join(paths.DIRECTIONS, "folds.json"), "w") as f:
        json.dump({"problem_to_direction": mapping, "default": "directions/unsure.pt"}, f, indent=2)

    stats = {"layer": L, "n_truth": int(len(yt)), "theta_mean_diff_norm": d,
             "n_swap_states": int(len(mu)), "u_mean_diff_norm": du,
             "cos_theta_u": float(u @ theta), "cos_theta_u_folds": fold_cos}
    with open(os.path.join(paths.DIRECTIONS, "directions.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    if fold_cos:
        print(f"[04] fold cos(theta, u) range {np.min(list(fold_cos.values())):+.3f} .. "
              f"{np.max(list(fold_cos.values())):+.3f}")


if __name__ == "__main__":
    main()
