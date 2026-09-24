"""Shared matplotlib style for all report figures.

Single accent color with neutral grays; series are distinguished by line
style and marker shape rather than by extra colors. Uses TeX Gyre Pagella,
the same Palatino design as the LaTeX report.
"""

from pathlib import Path

import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACCENT = "#064A56"
GRAY = "#6B7A7D"
GRID = "#DCE4E6"

_TEXGYRE = Path("/usr/local/texlive/2025basic/texmf-dist/fonts/opentype/public/tex-gyre")


def apply() -> None:
    if _TEXGYRE.is_dir():
        for face in ("regular", "bold", "italic", "bolditalic"):
            path = _TEXGYRE / f"texgyrepagella-{face}.otf"
            if path.exists():
                font_manager.fontManager.addfont(path)
    plt.rcParams.update({
        "font.family": "TeX Gyre Pagella" if _TEXGYRE.is_dir() else "serif",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.edgecolor": "#9DAFB3",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "xtick.color": GRAY,
        "ytick.color": GRAY,
        "text.color": "#1E2A2C",
        "axes.labelcolor": "#1E2A2C",
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
