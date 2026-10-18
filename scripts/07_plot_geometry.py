#!/usr/bin/env python3
"""Figs 1-3 and direction_stats.json (CPU).

  fig1  MATH true/false statements in their own PCA space (Outside Truth)
  fig2  answer-swapped CoT prefixes in their own PCA space (Reasoned Truth)
  fig3  each dataset projected into the OTHER one's PCA space

Each PCA axis is sign-oriented so the class used to fit the direction (true
statements; swapped answers) has the larger mean.

    python scripts/07_plot_geometry.py
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from truthsteer import paths  # noqa: E402
from truthsteer.plotstyle import AQUA, BLUE, GHOST, ORANGE, SURFACE, style  # noqa: E402


def auc(pos, neg):
    """P(score_pos > score_neg), ties count half."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    ranks = rankdata(np.concatenate([pos, neg]))
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class PCA2:
    """Top-2 PCA via the covariance eigendecomposition, signs oriented so the
    positive class mean sits on the + side of each axis."""

    def __init__(self, X, pos_mask):
        X = torch.as_tensor(X, dtype=torch.float64)
        self.mu = X.mean(0)
        Xc = X - self.mu
        evals, evecs = torch.linalg.eigh(Xc.T @ Xc / (len(X) - 1))
        order = torch.argsort(evals, descending=True)
        self.evr = (evals[order[:2]] / evals.sum()).numpy()
        E = evecs[:, order[:2]]
        Z = Xc @ E
        self.E = E * torch.where(Z[pos_mask].mean(0) >= Z[~pos_mask].mean(0), 1.0, -1.0)

    def __call__(self, X):
        return ((torch.as_tensor(X, dtype=torch.float64) - self.mu) @ self.E).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=paths.REPORT)
    ap.add_argument("--n-plot", type=int, default=4000, help="truth points drawn (PCA uses all)")
    args = ap.parse_args()
    L = paths.cfg()["layer"]
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)
    plt = style()

    Xt = torch.load(os.path.join(paths.STATES, f"truth_layer{L}.pt")).to(torch.float32)
    yt = pd.read_parquet(os.path.join(paths.STATES, "truth_meta.parquet"))["label"].values.astype(bool)
    Xu = torch.load(os.path.join(paths.STATES, f"swap_layer{L}.pt")).to(torch.float32)
    mu = pd.read_parquet(os.path.join(paths.STATES, "swap_meta.parquet"))
    cond = mu["condition"].values
    yu = cond != "control"
    theta = torch.load(os.path.join(paths.DIRECTIONS, "theta.pt")).to(torch.float64)
    u = torch.load(os.path.join(paths.DIRECTIONS, "unsure.pt")).to(torch.float64)
    folds = {os.path.basename(f)[len("unsure_"):-3]: torch.load(f).to(torch.float64)
             for f in sorted(glob.glob(os.path.join(paths.DIRECTIONS, "unsure_f*.pt")))}

    pt, pu = PCA2(Xt, torch.from_numpy(yt)), PCA2(Xu, torch.from_numpy(yu))
    Zt_t, Zt_u = pt(Xt), pu(Xt)   # truth data in truth / unsure space
    Zu_u, Zu_t = pu(Xu), pt(Xu)   # unsure data in unsure / truth space
    sub = np.zeros(len(yt), bool)
    sub[np.random.default_rng(0).choice(len(yt), min(args.n_plot, len(yt)), replace=False)] = True

    def scatter(ax, Z, mask, color, label, s, alpha):
        ax.scatter(Z[mask, 0], Z[mask, 1], s=s, c=color, alpha=alpha, lw=0.3,
                   edgecolors=SURFACE, label=label, rasterized=True)

    def truth_layer(ax, Z, ghost=False):
        if ghost:
            ax.scatter(Z[sub, 0], Z[sub, 1], s=6, c=GHOST, alpha=0.35, lw=0,
                       label="MATH Statements (Reference)", rasterized=True)
            return
        scatter(ax, Z, sub & yt, BLUE, "True Statement", 8, 0.35)
        scatter(ax, Z, sub & ~yt, ORANGE, "False Statement", 8, 0.35)

    def unsure_layer(ax, Z, ghost=False):
        if ghost:
            ax.scatter(Z[:, 0], Z[:, 1], s=14, c=GHOST, alpha=0.5, lw=0,
                       label="MATH Answers (Reference)", rasterized=True)
            return
        scatter(ax, Z, cond == "control", BLUE, "Unswapped Correct Answer", 22, 0.75)
        scatter(ax, Z, cond == "model_wrong", ORANGE, "Model-Wrong Swap", 22, 0.75)
        scatter(ax, Z, cond == "random_wrong", AQUA, "Random-Wrong Swap", 22, 0.75)

    def finish(ax, space, y):
        ax.set_xlabel(f"{space} PC1")
        ax.set_ylabel(f"{space} PC2")
        leg = ax.legend(*ax.get_legend_handles_labels(), loc="upper center",
                        bbox_to_anchor=(0.5, y), ncol=1, markerscale=1.6)
        for h in leg.legend_handles:
            h.set_alpha(1)

    for fname, space, draw in [
            ("fig1_outside_truth_pca.png", "Outside Truth", lambda ax: truth_layer(ax, Zt_t)),
            ("fig2_reasoned_truth_pca.png", "Reasoned Truth", lambda ax: unsure_layer(ax, Zu_u))]:
        fig, ax = plt.subplots(figsize=(5.2, 5.4))
        draw(ax)
        finish(ax, space, -0.14)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "figures", fname), dpi=120)
        plt.close(fig)
        print(f"[07] wrote {fname}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 5.6))
    unsure_layer(a1, Zu_u, ghost=True)
    truth_layer(a1, Zt_u)
    a1.set_title("Outside Truth Statements in Reasoned Truth PCA", fontsize=12)
    finish(a1, "Reasoned Truth", -0.13)
    truth_layer(a2, Zt_t, ghost=True)
    unsure_layer(a2, Zu_t)
    a2.set_title("MATH Answers in Outside Truth PCA", fontsize=12)
    finish(a2, "Outside Truth", -0.13)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "figures", "fig3_cross_pca.png"), dpi=120)
    plt.close(fig)
    print("[07] wrote fig3_cross_pca.png")

    Xt64, Xu64 = Xt.to(torch.float64), Xu.to(torch.float64)
    st, su = (Xt64 @ theta).numpy(), (Xu64 @ theta).numpy()
    ut, uu = (Xt64 @ u).numpy(), (Xu64 @ u).numpy()
    cos_folds = {k: float(v @ theta) for k, v in folds.items()}
    stats = {
        "layer": L, "n_truth": int(len(yt)), "n_truth_true": int(yt.sum()),
        "n_unsure_states": int(len(yu)), "n_unsure_cots": int(mu.cot_id.nunique()),
        "cos_theta_u_full": float(u @ theta),
        "cos_theta_reasoned_truth (= -u)": float(-u @ theta),
        "cos_theta_u_folds": cos_folds,
        "cos_theta_truth_pc1": float(theta @ pt.E[:, 0]), "cos_theta_truth_pc2": float(theta @ pt.E[:, 1]),
        "cos_u_unsure_pc1": float(u @ pu.E[:, 0]), "cos_u_unsure_pc2": float(u @ pu.E[:, 1]),
        "cos_truth_pc1_unsure_pc1": float(pt.E[:, 0] @ pu.E[:, 0]),
        "evr_truth_pc12": pt.evr.tolist(), "evr_unsure_pc12": pu.evr.tolist(),
        "auc_truth_data_along_theta (true>false)": auc(st[yt], st[~yt]),
        "auc_truth_data_along_u (true>false)": auc(ut[yt], ut[~yt]),
        "auc_unsure_data_along_u (swapped>control)": auc(uu[yu], uu[~yu]),
        "auc_unsure_data_along_theta (swapped>control)": auc(su[yu], su[~yu]),
        "note": "AUCs on the data each direction was fit on are in-sample (optimistic).",
    }
    with open(os.path.join(args.out, "direction_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
