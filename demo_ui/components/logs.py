"""components/logs.py -- Event timeline / evidence panel (sec.11): a dark
graphite/terminal-style panel with aligned monospace columns. Every text
role (timestamp, source/category, message, emphasis) gets its own class
with an explicit color in theme.py -- never relies on inherited/cascaded
text color, since this panel sits on a dark background inside an otherwise
light-themed page. Not raw stdout: every row is a typed LogEvent with
timestamp/source/level/event_type/message/metadata, sharing the same
timestamp basis as the charts in components/charts.py.
"""

from __future__ import annotations

import html
from typing import List

import streamlit as st

from demo_ui.models import LogEvent, LogSource


def _format_row(ev: LogEvent) -> str:
    ts = html.escape(f"{ev.timestamp:08.3f}")
    source = html.escape(f"{ev.source.value:<10}")
    event = html.escape(ev.event_type)
    message = html.escape(ev.message)
    return (
        f'<div class="du-evidence-row">'
        f'<span class="du-evidence-ts">{ts}</span>'
        f'<span class="du-evidence-source">{source}</span>'
        f'<span class="du-evidence-event">{event}</span>'
        f'<span class="du-evidence-message">{message}</span>'
        f'</div>'
    )


def render_log_viewer(logs: List[LogEvent], key_prefix: str = "du_logs") -> None:
    st.markdown('<div class="du-section-label">Event Timeline</div>', unsafe_allow_html=True)

    all_sources = ["ALL"] + [s.value for s in LogSource]
    selected = st.radio("Filter", all_sources, horizontal=True, key=f"{key_prefix}_filter", label_visibility="collapsed")

    filtered = logs if selected == "ALL" else [ev for ev in logs if ev.source.value == selected]
    filtered = sorted(filtered, key=lambda e: e.timestamp)

    rows = "".join(_format_row(ev) for ev in filtered) or '<div class="du-evidence-row du-evidence-empty">No events.</div>'
    st.markdown(f'<div class="du-evidence-panel">{rows}</div>', unsafe_allow_html=True)
