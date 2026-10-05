"""pages/history.py -- Compact experiment log table (sec.15). Fixture-backed;
does not create or modify real scientific experiment records. The
`experiments/<id>/...` layout documented in sec.17 is for future ResultStore
implementations only.
"""

from __future__ import annotations

import streamlit as st

from demo_ui.components.layout import render_topbar
from demo_ui.fixtures import build_experiment_history


def render() -> None:
    render_topbar(mode_label="DEMO", is_live=False)

    records = build_experiment_history()

    rows = []
    for rec in records:
        controllers = "/".join(c.value for c in rec.controllers)
        status_class = "status-ok" if rec.valid else "status-error"
        rows.append(
            f"<tr><td>{rec.experiment_id.split('_')[-1]}</td>"
            f"<td>{rec.traffic_profile.value.capitalize()}</td>"
            f"<td>{controllers}</td>"
            f"<td>{rec.timestamp_label}</td>"
            f"<td><span class='du-badge {status_class}'>{'VALID' if rec.valid else 'INVALID'}</span></td></tr>"
        )

    st.markdown(
        f"""
        <table class="du-history-table">
          <tr><th>ID</th><th>Workload</th><th>Mode</th><th>Time</th><th>Status</th></tr>
          {''.join(rows)}
        </table>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="du-section-label" style="margin-top:1rem;">Select to View</div>', unsafe_allow_html=True)
    options = [r.experiment_id for r in records]
    st.selectbox("Experiment", options, key="du_history_select", label_visibility="collapsed")
