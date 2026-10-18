#!/usr/bin/env python3
"""Figs 4-6 and steering_strength.csv (CPU).

  fig4  fraction of completed responses (finished before the token cap)
  fig5  uses of "Wait" per 100 words
  fig6  fraction of initially INCORRECT answers that end on the ground truth
        (completed AND grader matches_gt) - needs 06_grade_followups.py

Both directions share the truth run's r = 0 baseline. The Reasoned Truth direction
is -u, so the unsure run's +r is plotted at -r. Intervals: Wilson for fractions,
bootstrap over generations for the Wait rate.

    python scripts/08_plot_steering.py
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from truthsteer import paths  # noqa: E402
from truthsteer.plotstyle import BLUE, INK2, ORANGE, SURFACE, style, wilson  # noqa: E402

KEYS = ["problem_id", "solution_type", "r", "sample_idx"]
# label, color, marker, x-offset (nudged so coincident points stay visible)
SERIES = {"truth": ("Outside Truth Direction", BLUE, "o", 0.0),
          "unsure": ("Reasoned Truth Direction", ORANGE, "s", -0.008)}
BASES = {"correct_base_solution": "Correct MATH Answers",
         "incorrect_base_solution": "Incorrect MATH Answers"}
WAIT = re.compile(r"\bWait\b")


def load(direction):
    files = sorted(glob.glob(os.path.join(paths.STEER, f"generations_{direction}", "*.parquet")))
    if not files:
        sys.exit(f"no generations for {direction} - run 05_steer_followups.py --direction {direction}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    gp = os.path.join(paths.STEER, f"grades_{direction}.parquet")
    if os.path.exists(gp):
        g = pd.read_parquet(gp)[KEYS + ["verdict", "matches_gt"]]
        df = df.merge(g, on=KEYS, how="left", validate="one_to_one")
    else:
        df["verdict"], df["matches_gt"] = None, None
    return df


def series_frame():
    truth, unsure = load("truth"), load("unsure")
    base = truth[truth.r == 0]
    unsure = pd.concat([base, unsure]).assign(r=lambda d: -d.r + 0.0)
    out = pd.concat([truth.assign(direction="truth"), unsure.assign(direction="unsure")],
                    ignore_index=True)
    out["graded"] = out["verdict"].notna()
    out["completed"] = ~out["truncated"].astype(bool)
    out["corrected"] = out["completed"] & out["matches_gt"].fillna(False).astype(bool)
    out["n_wait"] = out.response_text.map(lambda t: len(WAIT.findall(t)))
    out["n_words"] = out.response_text.map(lambda t: len(t.split()))
    return out


def summarize(df, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for (d, st, r), g in df.groupby(["direction", "solution_type", "r"]):
        row = {"direction": d, "solution_type": st, "r": r, "n": len(g),
               "problems": g.problem_id.nunique(), "n_graded": int(g.graded.sum())}
        for m in ("completed", "corrected"):
            p, lo, hi = wilson(int(g[m].sum()), len(g))
            if m == "corrected" and not g.graded.all():
                p = lo = hi = np.nan
            row.update({m: p, f"{m}_lo": lo, f"{m}_hi": hi})
        w, n = g.n_wait.values, g.n_words.values
        idx = rng.integers(0, len(g), (n_boot, len(g)))
        boot = 100 * w[idx].sum(1) / n[idx].sum(1)
        row.update({"wait_per_100w": 100 * w.sum() / n.sum(),
                    "wait_per_100w_lo": np.percentile(boot, 2.5),
                    "wait_per_100w_hi": np.percentile(boot, 97.5)})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["direction", "solution_type", "r"])


def figure(plt, s, metric, ylabel, fname, out, bases, ylim):
    """One panel per base type, side by side, sharing y-limits."""
    fig, axes = plt.subplots(1, len(bases), figsize=(5.2 * len(bases), 4.2), squeeze=False)
    for ax, st in zip(axes[0], bases):
        for d, (label, color, mk, dx) in SERIES.items():
            g = s[(s.direction == d) & (s.solution_type == st)]
            ax.fill_between(g.r + dx, g[f"{metric}_lo"], g[f"{metric}_hi"], color=color, alpha=0.15, lw=0)
            ax.plot(g.r + dx, g[metric], color=color, lw=2, marker=mk, ms=7, mec=SURFACE, mew=1.5,
                    label=label)
        ax.axvline(0, color=INK2, lw=0.8, ls=":")
        ticks = [-0.4, -0.25, -0.1, 0, 0.1, 0.25, 0.4]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.set_xlabel("Steering Strength r")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_title(BASES[st], fontsize=12.5)
    axes[0][0].legend(loc="best", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", fname), dpi=120)
    plt.close(fig)
    print(f"[08] wrote {fname}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=paths.REPORT)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)
    plt = style(legend_fontsize=12.5)
    s = summarize(series_frame())
    s.to_csv(os.path.join(args.out, "steering_strength.csv"), index=False)
    pd.set_option("display.width", 200)
    print(s[["direction", "solution_type", "r", "n", "completed", "corrected", "wait_per_100w"]]
          .round(3).to_string(index=False))

    both = [b for b in BASES if b in set(s.solution_type)]
    figure(plt, s, "completed", "Fraction of Completed Responses", "fig4_completed_vs_r.png",
           args.out, both, (-0.03, 1.03))
    figure(plt, s, "wait_per_100w", 'Uses of "Wait" per 100 Words', "fig5_wait_vs_r.png",
           args.out, both, (0, 1.08 * s.wait_per_100w_hi.max()))
    inc = s[s.solution_type == "incorrect_base_solution"]
    if len(inc) and inc.corrected.notna().all():
        figure(plt, s, "corrected", "Fraction of Corrected Responses",
               "fig6_corrected_vs_r_incorrect.png", args.out, ["incorrect_base_solution"], (-0.01, 0.3))
    else:
        print("[08] skipped fig6: incorrect-base generations are not all graded "
              "(run 06_grade_followups.py for both directions)")


if __name__ == "__main__":
    main()
