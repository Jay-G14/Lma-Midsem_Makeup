"""One look for every figure in the make-up mid-sem report (used by part_a/aggregate_ada.py and
report/make_figures.py).

Colours are fixed by the job they do and were checked for colour-vision deficiency (OKLab distance
under simulated protanopia/deuteranopia, Machado et al. 2009) before use:
  - model identity: full finetune = neutral grey; LoRA r = 1, 4, 16 = one blue ramp, light -> dark as
    the rank grows; DPO beta = 0.1 = orange; DPO beta = 0.5 = aqua. A model keeps its colour in every figure.
  - correctness (outcome bars, the C1 grid): correct = blue, wrong = red, no answer = light grey, always
    with a symbol or a legend label as well, so colour is never the only cue (green vs red fails for
    deuteranopes, so it is not used).
  - other categories (Parts B and C): blue, orange, aqua in that order.
  - magnitudes (heatmaps): one blue ramp, light = low, dark = high.
Figures are drawn at their printed size (7 in = the report's text width), so 8 pt here is 8 pt on paper.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402

# ink and chrome
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

# categorical slots (in this order)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

# model identity
FULL = "#52514e"
RANK = {1: "#86b6ef", 4: "#2a78d6", 16: "#104281"}
BETA = {"0.1": ORANGE, "0.5": AQUA}

# correctness
CORRECT, CORRECT_LIGHT, WRONG, NONE = "#184f95", "#6da7ec", "#d03b3b", "#c3c2b7"
CORRECT_CELL = "#2a78d6"

# sequential blue ramp (steps 100 ... 700)
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)

TEXT_WIDTH_IN = 7.0


def apply():
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 8, "text.color": INK,
        "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlepad": 5,
        "axes.labelsize": 8, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "lines.linewidth": 1.6, "lines.markersize": 5, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.fontsize": 7, "legend.frameon": False, "legend.handlelength": 1.8, "legend.borderaxespad": 0.3,
        "errorbar.capsize": 2, "hatch.linewidth": 0.6,
    })


def text_on(fill):
    """Near-black or white label for text set inside a filled cell, by the fill's luminance."""
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in to_rgb(fill))
    return INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.32 else "#ffffff"


def point(ax, x, y, color, size=5.5, marker="o", hollow=False, zorder=3, **kw):
    """A marker with a white ring, so it stays legible where it crosses a line or another marker."""
    return ax.plot([x], [y], marker=marker, linestyle="none", markersize=size,
                   markerfacecolor=SURFACE if hollow else color, markeredgecolor=color if hollow else SURFACE,
                   markeredgewidth=1.3 if hollow else 1.0, zorder=zorder, **kw)


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
