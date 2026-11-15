#!/usr/bin/env python3
"""scripts/paper_style.py

V2 Phase 6D: one shared IEEE-style matplotlib configuration module, used
by every plotting script in this phase (scripts/generate_v2_phase6d_figures.py)
so no figure duplicates these constants. Read-only: never touches any
data/results file.

Fixed per-arm identity (Okabe-Ito categorical palette, validated
colorblind-safe via the dataviz skill's validator -- all 6 checks PASS):
    ppo_v2             #D55E00  circle marker   no hatch
    hpa_v2_cpu_memory  #0072B2  square marker    '//' hatch
    hpa_v2_cpu_only    #009E73  triangle marker  'xx' hatch
The same color+marker+hatch triple is used for a given arm in EVERY
figure this phase produces -- never reassigned per-figure.
"""

from __future__ import annotations

from typing import Dict, Tuple

import matplotlib
import matplotlib.font_manager as fm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# -- IEEE column widths (inches) --
SINGLE_COL_WIDTH_IN = 3.5
DOUBLE_COL_WIDTH_IN = 7.16

# -- fixed per-arm identity: color (Okabe-Ito), marker, hatch --
ARM_IDS: Tuple[str, ...] = ("ppo_v2", "hpa_v2_cpu_memory", "hpa_v2_cpu_only")
ARM_LABELS: Dict[str, str] = {
    "ppo_v2": "PPO (multi-metric, 11 signals)",
    "hpa_v2_cpu_memory": "HPA CPU+Memory (resource baseline)",
    "hpa_v2_cpu_only": "HPA CPU-only (Kubernetes default)",
}
ARM_COLORS: Dict[str, str] = {
    "ppo_v2": "#D55E00", "hpa_v2_cpu_memory": "#0072B2", "hpa_v2_cpu_only": "#009E73",
}
ARM_MARKERS: Dict[str, str] = {"ppo_v2": "o", "hpa_v2_cpu_memory": "s", "hpa_v2_cpu_only": "^"}
ARM_HATCHES: Dict[str, str] = {"ppo_v2": "", "hpa_v2_cpu_memory": "//", "hpa_v2_cpu_only": "xx"}

SLA_LINE_COLOR = "#555555"
FAILED_SESSION_COLOR = "#999999"


def _sans_serif_family() -> str:
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in ("Helvetica", "DejaVu Sans"):
        if candidate in available:
            return candidate
    return "sans-serif"


def apply_style() -> None:
    """Call once, before creating any figure in a plotting script."""
    family = _sans_serif_family()
    plt.rcParams.update({
        "font.family": family,
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,  # embed as real text, not outlined paths
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })


def new_figure(width_in: float = SINGLE_COL_WIDTH_IN, height_in: float = 2.6, **kwargs):
    fig, ax = plt.subplots(figsize=(width_in, height_in), **kwargs)
    return fig, ax


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)


def save_figure(fig, path_without_ext) -> Tuple[str, str]:
    """Saves a PDF (vector) and a 300dpi PNG side by side. No in-figure
    title is ever set by this helper -- captions live in FIGURES.md.
    `bbox_inches="tight"` so a figure-level legend placed outside the axes
    (e.g. above a stacked-area plot where an in-axes legend would sit on
    top of the data) is never clipped at the figure edge."""
    fig.tight_layout()
    pdf_path = f"{path_without_ext}.pdf"
    png_path = f"{path_without_ext}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def arm_color(arm_id: str) -> str:
    return ARM_COLORS[arm_id]


def arm_marker(arm_id: str) -> str:
    return ARM_MARKERS[arm_id]


def arm_hatch(arm_id: str) -> str:
    return ARM_HATCHES[arm_id]


def arm_label(arm_id: str) -> str:
    return ARM_LABELS[arm_id]
