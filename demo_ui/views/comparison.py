"""pages/comparison.py -- Final comparison page (sec.13-14, 16): looks
like a scientific result, not another monitoring dashboard.

When real LIVE experiment runs exist (`experiment_store`), the page leads
with an actual-data comparison driven by the latest completed run and its
workload-compatible counterpart (same profile/hash/seed/duration) -- never
a fabricated graph for an arm that hasn't actually run that workload. Below
that sits the pre-registered evaluation campaign's own analysis output.
"""

from __future__ import annotations

import time

import streamlit as st

from demo_ui import experiment_store
from demo_ui.components.charts import overlay_line_chart
from demo_ui.components.layout import render_topbar
from demo_ui.config_v2 import load_config_v2_safe
from demo_ui.views import comparison_v2

_METRIC_LABELS = [
    ("mean_replicas", "Mean replicas"),
    ("max_replicas", "Max replicas"),
    ("scaling_events", "Scaling events"),
    ("mean_latency_p95_ms", "P95 latency (ms)"),
    ("success_rate_rps", "Success rate (rps)"),
    ("error_rate_rps", "Error rate (rps)"),
]


def _fmt_metric(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def _latest_current_system_run():
    """The most recently started completed run made by the current
    system (never a run from a different generation). Task sec.3: an
    `invalid_no_traffic` run (zero/near-zero traffic throughout -- never a
    real measurement) is never chosen as the workload this section leads
    with, even if it is the most recent terminal run -- falls back to the
    next-most-recent genuinely-evaluable run, or `None` if every run so
    far is invalidated (the empty state below, never a fabricated
    comparison)."""
    candidates = [
        r for r in experiment_store.completed_runs()
        if r.get("generation") == "v2" and r.get("status") != experiment_store.INVALID_NO_TRAFFIC
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["started_at"])


def _render_live_comparison() -> bool:
    """Always renders the Live Comparison section -- chart frames/axes are
    shown even with zero runs, per the "dashboard structure always exists,
    data appears only once real" requirement. Returns True if at least one
    real completed run exists (used only to adjust the spacing below)."""
    st.markdown('<div class="du-section-label">Live Comparison</div>', unsafe_allow_html=True)
    latest = _latest_current_system_run()
    has_any = latest is not None

    if not has_any:
        st.caption("No compatible experiment pair yet.")
        c1, c2 = st.columns(2)
        empty_primary, empty_detail = "No comparison data", "Run both arms with the same workload"
        c1.plotly_chart(overlay_line_chart("Replicas", [], [], "HPA", [], [], "PPO", "replicas", empty_message=empty_primary, empty_detail=empty_detail, empty_y_range=(0, 8)), use_container_width=True)
        c2.plotly_chart(overlay_line_chart("P95 Latency", [], [], "HPA", [], [], "PPO", "ms", empty_message=empty_primary, empty_detail=empty_detail, empty_y_range=(0, 500)), use_container_width=True)
        return False

    profile_label = latest["profile_name"].replace("_", " ").title()
    st.markdown(f'<div class="mono" style="font-size:13px;">Comparing workload: <b>{profile_label}</b></div>', unsafe_allow_html=True)

    pair = experiment_store.latest_compatible_pair(latest)
    hpa_run, ppo_run = pair["hpa"], pair["ppo"]

    def _run_label(label: str, rec) -> str:
        if rec is None:
            return f"{label} run: not available"
        ts = time.strftime("%H:%M:%S", time.localtime(rec.get("completed_at") or rec["started_at"]))
        return f"{label} run: {rec['run_id']} ({ts})"

    st.caption(f"{_run_label('HPA', hpa_run)}  ·  {_run_label('PPO', ppo_run)}")

    missing_msg = None
    if hpa_run is None:
        missing_msg = "Matching HPA run not available yet." if ppo_run is not None else "No compatible experiment pair yet."
    elif ppo_run is None:
        missing_msg = "Matching PPO run not available yet."

    if hpa_run is not None or ppo_run is not None:
        available = "HPA result available" if hpa_run is not None else None
        if ppo_run is not None:
            available = f"{available} · PPO result available" if available else "PPO result available"
        line = available or ""
        if missing_msg and missing_msg != "No compatible experiment pair yet.":
            line = f"{line} · {missing_msg}" if line else missing_msg
        if line:
            st.caption(line)

    # Pre-existing field-name mismatch fixed in passing while verifying
    # comparison pairing end-to-end (task sec.7): `EpisodeSample`/
    # `scripts/run_demo_episode_v2.py`'s own per-decision sample schema
    # names this field `desired_replicas` (see `_to_status_sample`), never
    # `replicas_desired` -- the old key name here meant `extract_series()`
    # always returned `([], [])` for the Replicas overlay chart even for a
    # correctly-paired HPA/PPO run, silently empty rather than erroring.
    ts_hr, hr = experiment_store.extract_series(hpa_run, "desired_replicas")
    ts_hl, hl = experiment_store.extract_series(hpa_run, "p95_latency_ms")
    ts_pr, pr = experiment_store.extract_series(ppo_run, "desired_replicas")
    ts_pl, pl = experiment_store.extract_series(ppo_run, "p95_latency_ms")

    c1, c2 = st.columns(2)
    c1.plotly_chart(overlay_line_chart("Replicas", ts_hr, hr, "HPA", ts_pr, pr, "PPO", "replicas", empty_message="No comparison data", empty_detail="Run both arms with the same workload", empty_y_range=(0, 8)), use_container_width=True)
    c2.plotly_chart(overlay_line_chart("P95 Latency", ts_hl, hl, "HPA", ts_pl, pl, "PPO", "ms", empty_message="No comparison data", empty_detail="Run both arms with the same workload", empty_y_range=(0, 500)), use_container_width=True)

    if hpa_run is not None and ppo_run is not None:
        hpa_summary, ppo_summary = hpa_run.get("summary") or {}, ppo_run.get("summary") or {}
        rows = "".join(
            f"<tr><td>{label}</td><td>{_fmt_metric(hpa_summary.get(key))}</td><td>{_fmt_metric(ppo_summary.get(key))}</td></tr>"
            for key, label in _METRIC_LABELS
        )
        st.markdown(
            f"""
            <table class="du-cmp-table" style="margin-top:0.5rem;">
              <tr><th>Metric</th><th>HPA</th><th>PPO</th></tr>
              {rows}
            </table>
            """,
            unsafe_allow_html=True,
        )
    return True


def render() -> None:
    render_topbar(mode_label="Live cluster", is_live=True)

    _render_live_comparison()

    st.markdown('<div class="du-section-label" style="margin-top:1rem;">Evaluation (CORR-60/61 pre-registered campaign)</div>', unsafe_allow_html=True)
    cfg, error = load_config_v2_safe()
    if error is not None:
        st.caption(error)
    else:
        comparison_v2.render(cfg.evaluation_campaign_dir)
