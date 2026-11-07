"""pages/history.py -- Compact experiment log table (sec.15).

Shows two clearly-separated sections: real LIVE experiment runs (from
`experiment_store`, newest first -- these are actual executions launched
from the Experiment page) and DEMO fixture history (unchanged, clearly
labeled). Fixture rows are never presented as LIVE results. Does not
create or modify real scientific experiment records -- `experiment_store`
is a demo-only run-history store, not a scientific ledger.
"""

from __future__ import annotations

import time

import streamlit as st

from demo_ui import experiment_store
from demo_ui.components.layout import render_topbar
from demo_ui.fixtures import build_experiment_history


def _render_live_history() -> None:
    st.markdown('<div class="du-section-label">Live Experiment Runs</div>', unsafe_allow_html=True)
    runs = experiment_store.completed_runs()
    if not runs:
        st.markdown('<div class="du-panel mono">No LIVE experiment history yet.</div>', unsafe_allow_html=True)
        return

    rows = []
    for rec in runs:
        profile_label = rec["profile_name"].replace("_", " ").title()
        ts = time.strftime("%H:%M:%S", time.localtime(rec.get("completed_at") or rec["started_at"]))
        status = rec.get("status", "unknown")
        status_class = "status-ok" if status == "completed" else "status-error"
        summary = rec.get("summary") or {}
        metrics = (
            f'replicas={summary.get("mean_replicas", "–")} '
            f'p95={summary.get("mean_latency_p95_ms", "–")} '
            f'errors={summary.get("error_rate_rps", "–")}'
        ) if summary else "–"
        rows.append(
            f"<tr><td>{rec['run_id']}</td><td>{profile_label}</td><td>{rec['controller']}</td>"
            f"<td>{ts}</td><td><span class='du-badge {status_class}'>{status.upper()}</span></td>"
            f"<td class='mono' style='font-size:11px;'>{metrics}</td></tr>"
        )
    st.markdown(
        f"""
        <table class="du-history-table">
          <tr><th>Run ID</th><th>Workload</th><th>Controller</th><th>Time</th><th>Status</th><th>Metrics</th></tr>
          {''.join(rows)}
        </table>
        """,
        unsafe_allow_html=True,
    )


def _render_fixture_history() -> None:
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


def render() -> None:
    render_topbar(mode_label="DEMO", is_live=False)

    _render_live_history()

    with st.expander("Demo fixture data (not real LIVE results)"):
        st.caption("Optional simulated reference data -- not generated until you load it.")
        if st.button("Load demo fixture data", key="du_load_history_fixture"):
            st.session_state["du_show_history_fixture"] = True
        if st.session_state.get("du_show_history_fixture", False):
            _render_fixture_history()
