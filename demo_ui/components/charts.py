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


def _add_empty_state(fig: go.Figure, primary: str, secondary: Optional[str] = None) -> None:
    """Overlays a compact, centered two-line message on an otherwise-empty
    chart WITHOUT inserting any fake data point -- the axes/grid/legend/
    title/frame stay exactly as they would with real data, so the page
    never jumps/reflows once real samples arrive and this call stops being
    made. Kept short and vertically stacked (never a long sentence spanning
    the plot width) so it reads like a monitoring dashboard's "no
    datapoints" state, not an error banner."""
    text = f"<b>{primary}</b>"
    if secondary:
        text += f"<br><span style='font-size:10px;color:{PALETTE['muted']}'>{secondary}</span>"
    fig.add_annotation(
        text=text, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, align="center", font=dict(size=12, color=PALETTE["muted"]),
    )


def _apply_empty_axes(fig: go.Figure, y_range: Optional[Sequence[float]], x_title: str = "Elapsed time (s)") -> None:
    """Pins a stable, sensible presentation range for an empty chart --
    axis range here is display metadata, never measured data -- so Plotly
    never falls back to a strange auto-range like x:0..6/y:-1..4 for a
    trace with zero points. Only applied when there is no real data; once
    real samples exist, callers skip this and let Plotly auto-fit them."""
    fig.update_xaxes(range=[0, 60], title=dict(text=x_title, font=dict(size=10, color=PALETTE["muted"])))
    if y_range is not None:
        fig.update_yaxes(range=list(y_range))


# All traces below use mode="lines+markers", not "lines" -- a centralized
# fix, not a per-metric special case: Plotly only draws a connecting line
# segment between 2+ points, so a series with exactly one valid sample
# (common for a short demo episode where only one decision step produced a
# non-null value, e.g. P95 at t=0 is null until the first real traffic
# window closes) rendered NOTHING under "lines" mode -- not an empty-state
# message, not a point, just a blank plot. With markers included, 0 points
# still shows nothing (the empty-state annotation path handles that
# separately), exactly 1 point shows a single visible marker at its real
# coordinate, and 2+ points render the normal line+marker series -- no
# special-casing, no fabricated second point, no backfill to t=0.
def line_chart(
    title: str,
    timestamps: List[float],
    series: dict,  # {label: values}
    y_title: str = "",
    events: Optional[Sequence[EventMarker]] = None,
    empty_message: Optional[str] = None,
    empty_detail: Optional[str] = None,
    empty_y_range: Optional[Sequence[float]] = None,
) -> go.Figure:
    fig = go.Figure()
    has_data = bool(timestamps)
    for i, (label, values) in enumerate(series.items()):
        fig.add_trace(go.Scatter(
            x=timestamps, y=values, mode="lines+markers", name=label,
            line=dict(color=_SERIES_PALETTE[i % len(_SERIES_PALETTE)], width=1.6),
            marker=dict(size=5, color=_SERIES_PALETTE[i % len(_SERIES_PALETTE)]),
        ))
    _add_event_markers(fig, events)
    fig.update_layout(title=_title(title), yaxis_title=y_title, **_BASE_LAYOUT)
    if not has_data:
        _apply_empty_axes(fig, empty_y_range)
        if empty_message:
            _add_empty_state(fig, empty_message, empty_detail)
    return fig


def step_chart(
    title: str,
    timestamps: List[float],
    values: List[int],
    y_title: str = "replicas",
    events: Optional[Sequence[EventMarker]] = None,
    empty_message: Optional[str] = None,
    empty_detail: Optional[str] = None,
    empty_y_range: Optional[Sequence[float]] = None,
) -> go.Figure:
    fig = go.Figure()
    has_data = bool(timestamps)
    fig.add_trace(go.Scatter(
        x=timestamps, y=values, mode="lines+markers", name=y_title,
        line=dict(color=CHART_COLORS["ppo"], width=1.6, shape="hv"),
        marker=dict(size=5, color=CHART_COLORS["ppo"]),
    ))
    _add_event_markers(fig, events)
    fig.update_layout(title=_title(title), yaxis_title=y_title, **_BASE_LAYOUT)
    if not has_data:
        _apply_empty_axes(fig, empty_y_range)
        if empty_message:
            _add_empty_state(fig, empty_message, empty_detail)
    return fig


def overlay_line_chart(
    title: str,
    timestamps_a: List[float], values_a: List[float], label_a: str,
    timestamps_b: List[float], values_b: List[float], label_b: str,
    y_title: str = "",
    empty_message: Optional[str] = None,
    empty_detail: Optional[str] = None,
    empty_y_range: Optional[Sequence[float]] = None,
) -> go.Figure:
    """For the final Auto comparison page's HPA-vs-PPO overlays (sec.14).
    Either series may be empty (e.g. only HPA has run so far) -- that
    series is simply not drawn, never backfilled with fake points. The
    empty-state message/axis pinning only applies when BOTH are empty; a
    partial pair (one real series, one missing) shows the real series at
    its natural scale with no overlay message on the chart itself."""
    fig = go.Figure()
    if timestamps_a:
        fig.add_trace(go.Scatter(
            x=timestamps_a, y=values_a, mode="lines+markers", name=label_a,
            line=dict(color=CHART_COLORS["hpa"], width=1.6, dash="solid"),
            marker=dict(size=5, color=CHART_COLORS["hpa"]),
        ))
    if timestamps_b:
        fig.add_trace(go.Scatter(
            x=timestamps_b, y=values_b, mode="lines+markers", name=label_b,
            line=dict(color=CHART_COLORS["ppo"], width=1.6, dash="dot"),
            marker=dict(size=5, color=CHART_COLORS["ppo"]),
        ))
    fig.update_layout(title=_title(title), yaxis_title=y_title, **_BASE_LAYOUT)
    if not timestamps_a and not timestamps_b:
        _apply_empty_axes(fig, empty_y_range)
        if empty_message:
            _add_empty_state(fig, empty_message, empty_detail)
    return fig
