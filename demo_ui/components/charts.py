"""components/charts.py -- Reusable Plotly time-series charts (sec.8-9, 14).

Every chart shares the same timestamp basis as the evidence log
(components/logs.py) and supports overlaying EventMarker vertical lines, so
a demo/guide can visually correlate a log line with the exact point on a
graph it corresponds to. Visually integrated with the page: surface-colored
background, thin axes, subtle grid, restrained engineering series colors
(never a rainbow palette).

Dependency: plotly (not yet installed in this repo's .venv as of this
phase -- see demo_ui/README.md).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import plotly.graph_objects as go

from demo_ui.models import EventMarker
from demo_ui.theme import CHART_COLORS, PALETTE

_MARKER_COLORS = {
    "PPO_SCALE_UP": PALETTE["vermilion"],
    "PPO_SCALE_DOWN": PALETTE["vermilion"],
    "HPA_SCALE_UP": PALETTE["ochre"],
    "HPA_SCALE_DOWN": PALETTE["ochre"],
    "DEPLOYMENT_REPLICA_CHANGE": PALETTE["petrol"],
    "POD_READY": PALETTE["olive"],
}

_SERIES_PALETTE = [PALETTE["petrol"], PALETTE["vermilion"], PALETTE["olive"], PALETTE["ochre"]]

_BASE_LAYOUT = dict(
    paper_bgcolor=PALETTE["surface"],
    plot_bgcolor=PALETTE["surface"],
    font=dict(color=PALETTE["ink"], family='"SFMono-Regular", Menlo, Consolas, monospace', size=11),
    margin=dict(l=44, r=12, t=28, b=32),
    xaxis=dict(gridcolor=PALETTE["hairline"], title=None, linecolor=PALETTE["hairline"], showline=True, zeroline=False),
    yaxis=dict(gridcolor=PALETTE["hairline"], linecolor=PALETTE["hairline"], showline=True, zeroline=False),
    legend=dict(orientation="h", y=-0.22, font=dict(size=10.5), bgcolor="rgba(0,0,0,0)"),
    hovermode="x unified",
)

_TITLE_FONT = dict(size=12, family='"Avenir Next", "Inter", sans-serif')


def _title(text: str) -> dict:
    return dict(text=text, font=_TITLE_FONT, x=0.01, xanchor="left")


def _add_event_markers(fig: go.Figure, events: Optional[Sequence[EventMarker]]) -> None:
    if not events:
        return
    for ev in events:
        color = _MARKER_COLORS.get(ev.category, PALETTE["muted"])
        fig.add_vline(x=ev.timestamp, line_width=1, line_dash="dot", line_color=color, opacity=0.6)


def line_chart(
    title: str,
    timestamps: List[float],
    series: dict,  # {label: values}
    y_title: str = "",
    events: Optional[Sequence[EventMarker]] = None,
) -> go.Figure:
    fig = go.Figure()
    for i, (label, values) in enumerate(series.items()):
        fig.add_trace(go.Scatter(
            x=timestamps, y=values, mode="lines", name=label,
            line=dict(color=_SERIES_PALETTE[i % len(_SERIES_PALETTE)], width=1.6),
        ))
    _add_event_markers(fig, events)
    fig.update_layout(title=_title(title), yaxis_title=y_title, **_BASE_LAYOUT)
    return fig


def step_chart(
    title: str,
    timestamps: List[float],
    values: List[int],
    y_title: str = "replicas",
    events: Optional[Sequence[EventMarker]] = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=values, mode="lines", name=y_title,
        line=dict(color=CHART_COLORS["ppo"], width=1.6, shape="hv"),
    ))
    _add_event_markers(fig, events)
    fig.update_layout(title=_title(title), yaxis_title=y_title, **_BASE_LAYOUT)
    return fig


def overlay_line_chart(
    title: str,
    timestamps_a: List[float], values_a: List[float], label_a: str,
    timestamps_b: List[float], values_b: List[float], label_b: str,
    y_title: str = "",
) -> go.Figure:
    """For the final Auto comparison page's HPA-vs-PPO overlays (sec.14)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps_a, y=values_a, mode="lines", name=label_a,
        line=dict(color=CHART_COLORS["hpa"], width=1.6, dash="solid"),
    ))
    fig.add_trace(go.Scatter(
        x=timestamps_b, y=values_b, mode="lines", name=label_b,
        line=dict(color=CHART_COLORS["ppo"], width=1.6, dash="dot"),
    ))
    fig.update_layout(title=_title(title), yaxis_title=y_title, **_BASE_LAYOUT)
    return fig
