"""components/layout.py -- Structural chrome shared across pages: top bar,
instrument metric strip, and the Auto-mode pipeline stepper. Pure
rendering; takes plain data in, no provider/business-logic coupling.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import streamlit as st


def render_topbar(mode_label: str, is_live: bool) -> None:
    """Thin full-width header (sec.5 TOP BAR)."""
    mode_class = "du-mode-live" if is_live else "du-mode-demo"
    st.markdown(
        f"""
        <div class="du-topbar">
          <div>
            <div class="du-topbar-title">Adaptive Kubernetes Autoscaling</div>
            <div class="du-topbar-name">Experiment Console</div>
            <div class="du-topbar-sub">PPO vs Kubernetes HPA &middot; Controlled Workload Evaluation</div>
          </div>
          <div class="du-mode-indicator {mode_class}">{mode_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_strip(cells: List[Tuple[str, str]]) -> None:
    """A slim horizontal instrument strip (sec.7): [(label, value), ...]."""
    cells_html = "".join(
        f'<div class="du-metric-cell"><div class="du-metric-label">{label}</div>'
        f'<div class="du-metric-value">{value}</div></div>'
        for label, value in cells
    )
    st.markdown(f'<div class="du-metric-strip">{cells_html}</div>', unsafe_allow_html=True)


def render_pipeline(steps: List[Tuple[str, str]], current_index: Optional[int]) -> None:
    """Horizontal pipeline stepper for Auto mode (sec.12).

    `steps`: [(key, label), ...] in order.
    `current_index`: index of the currently active step, or None/len(steps)
    if the whole pipeline is complete (every step renders as done)."""
    n = len(steps)
    parts = []
    for i, (_key, label) in enumerate(steps):
        if current_index is None or i < current_index:
            cls, dot = "done", "✓"
        elif i == current_index:
            cls, dot = "active", "●"
        else:
            cls, dot = "", "○"
        parts.append(f'<div class="du-pipeline-step {cls}"><div class="du-pipeline-dot">{dot}</div>{label}</div>')
        if i < n - 1:
            connector_done = "done" if (current_index is None or i < current_index) else ""
            parts.append(f'<div class="du-pipeline-connector {connector_done}"></div>')
    st.markdown(f'<div class="du-pipeline">{"".join(parts)}</div>', unsafe_allow_html=True)
