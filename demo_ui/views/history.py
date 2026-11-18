"""pages/history.py -- Compact experiment log table (sec.15).

Shows real LIVE experiment runs made by the current system (from
`experiment_store`, newest first -- these are actual executions launched
from the Experiment page). Does not create or modify real scientific
experiment records -- `experiment_store` is a demo-only run-history store,
not a scientific ledger.
"""

from __future__ import annotations

import time

import streamlit as st

from demo_ui import experiment_store
from demo_ui.components.controls_v2 import display_arm_label
from demo_ui.components.layout import render_topbar


def _current_system_runs():
    """Only runs made by the current system -- a run from a different
    generation must never appear in this history."""
    return [r for r in experiment_store.completed_runs() if r.get("generation") == "v2"]


def render() -> None:
    render_topbar(mode_label="Live cluster", is_live=True)

    st.markdown('<div class="du-section-label">Experiment Runs</div>', unsafe_allow_html=True)
    runs = _current_system_runs()
    if not runs:
        st.markdown('<div class="du-panel mono">No experiment history yet.</div>', unsafe_allow_html=True)
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
            f"<tr><td>{rec['run_id']}</td><td>{profile_label}</td><td>{display_arm_label(rec['controller'])}</td>"
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
