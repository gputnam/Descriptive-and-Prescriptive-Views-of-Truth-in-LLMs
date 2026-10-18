"""Shared figure style."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE, GHOST = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#c9c8c2"


def style(legend_fontsize=11):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 12.5, "legend.fontsize": legend_fontsize, "legend.frameon": False,
    })
    return plt


def wilson(k, n, z=1.96):
    """(p, lo, hi): Wilson score interval for k successes out of n."""
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return p, max(0.0, c - h), min(1.0, c + h)
