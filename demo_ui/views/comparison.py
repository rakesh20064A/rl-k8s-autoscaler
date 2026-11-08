"""pages/comparison.py -- Final Auto comparison page (sec.13-14, 16): looks
like a scientific result, not another monitoring dashboard. Validity header,
clean comparison table (tiny delta indicators, never a fully-colored
table), overlay graphs, expandable evidence, and an editorial automatic
conclusion.

When real LIVE experiment runs exist (`experiment_store`), the page leads
with an actual-data comparison driven by the latest completed run and its
workload-compatible counterpart (same profile/hash/seed/duration) -- never
a fabricated graph for a controller that hasn't actually run that workload.
The original fixture-driven comparison below remains, clearly labeled as
DEMO/simulated, for when no real runs exist yet.
"""

from __future__ import annotations

import time

import streamlit as st

from demo_ui import experiment_store
from demo_ui.comparison_engine import compare
from demo_ui.components.charts import overlay_line_chart
from demo_ui.components.layout import render_topbar
from demo_ui.conclusion_engine import generate_conclusion
from demo_ui.config_v2 import load_config_v2_safe
from demo_ui.fixtures import build_fixture_pair
from demo_ui.models import TrafficProfile
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


def _render_live_comparison() -> bool:
    """Always renders the Live Comparison section -- chart frames/axes are
    shown even with zero runs, per the "dashboard structure always exists,
    data appears only once real" requirement. Returns True if at least one
    real completed run exists (used only to adjust the fixture section's
    spacing below)."""
    st.markdown('<div class="du-section-label">Live Comparison</div>', unsafe_allow_html=True)
    latest = experiment_store.latest_run()
    has_any = latest is not None and latest.get("status") == "completed"

    if not has_any:
        st.caption("No compatible experiment pair yet.")
        c1, c2 = st.columns(2)
        empty_primary, empty_detail = "No comparison data", "Run HPA and PPO with the same workload"
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

    ts_hr, hr = experiment_store.extract_series(hpa_run, "replicas_desired")
    ts_hl, hl = experiment_store.extract_series(hpa_run, "latency_p95_ms")
    ts_pr, pr = experiment_store.extract_series(ppo_run, "replicas_desired")
    ts_pl, pl = experiment_store.extract_series(ppo_run, "latency_p95_ms")

    c1, c2 = st.columns(2)
    c1.plotly_chart(overlay_line_chart("Replicas", ts_hr, hr, "HPA", ts_pr, pr, "PPO", "replicas", empty_message="No comparison data", empty_detail="Run HPA and PPO with the same workload", empty_y_range=(0, 8)), use_container_width=True)
    c2.plotly_chart(overlay_line_chart("P95 Latency", ts_hl, hl, "HPA", ts_pl, pl, "PPO", "ms", empty_message="No comparison data", empty_detail="Run HPA and PPO with the same workload", empty_y_range=(0, 500)), use_container_width=True)

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


def _delta_html(pct: float) -> str:
    cls = "du-delta-pos" if pct > 0 else "du-delta-neg" if pct < 0 else ""
    return f'<span class="{cls}">{pct:+.1f}%</span>'


def _render_table(result, hpa_run, ppo_run) -> None:
    rows = []
    for m in result.metrics:
        rows.append(f"<tr><td>{m.name}</td><td>{m.hpa_value:,.1f}</td><td>{m.ppo_value:,.1f}</td><td>{_delta_html(m.pct_difference)}</td></tr>")

    rs_delta = ((result.ppo_replica_seconds - result.hpa_replica_seconds) / result.hpa_replica_seconds * 100) if result.hpa_replica_seconds else 0.0
    rows.append(f"<tr><td>Avg replicas</td><td>{result.hpa_avg_replicas:,.2f}</td><td>{result.ppo_avg_replicas:,.2f}</td>"
                f"<td>{_delta_html((result.ppo_avg_replicas - result.hpa_avg_replicas) / result.hpa_avg_replicas * 100 if result.hpa_avg_replicas else 0)}</td></tr>")
    rows.append(f"<tr><td>Peak replicas</td><td>{result.hpa_peak_replicas}</td><td>{result.ppo_peak_replicas}</td><td></td></tr>")
    rows.append(f"<tr><td>Replica-seconds</td><td>{result.hpa_replica_seconds:,.0f}</td><td>{result.ppo_replica_seconds:,.0f}</td><td>{_delta_html(rs_delta)}</td></tr>")
    rows.append(f"<tr><td>Scaling events</td><td>{result.hpa_scaling_events}</td><td>{result.ppo_scaling_events}</td><td></td></tr>")

    st.markdown(
        f"""
        <table class="du-cmp-table">
          <tr><th>Metric</th><th>HPA</th><th>PPO</th><th>&Delta;</th></tr>
          {''.join(rows)}
        </table>
        """,
        unsafe_allow_html=True,
    )


def _render_conclusion(conclusion) -> None:
    dims_html = "".join(f"<div style='margin-top:0.35rem;'><b>{d.name}:</b> {d.verdict}</div>" for d in conclusion.dimensions)
    st.markdown(
        f"""
        <div class="du-conclusion">
          <div class="du-conclusion-headline">Experiment Observation</div>
          {conclusion.headline}
          {dims_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if conclusion.is_demo_data:
        st.markdown(
            '<div class="du-offline-banner" style="margin-top:0.6rem;">SIMULATED DATA — NOT A SCIENTIFIC RESULT</div>',
            unsafe_allow_html=True,
        )


def render() -> None:
    render_topbar(mode_label="DEMO", is_live=False)

    _render_live_comparison()

    with st.expander("Demo fixture data (simulated, not real LIVE results)"):
        st.caption("Optional simulated reference data -- not generated until you load it.")
        if st.button("Load demo fixture data", key="du_load_comparison_fixture"):
            st.session_state["du_show_comparison_fixture"] = True
        if st.session_state.get("du_show_comparison_fixture", False):
            _render_fixture_comparison()

    st.markdown('<div class="du-section-label" style="margin-top:1rem;">V2 Evaluation (CORR-60/61 pre-registered campaign)</div>', unsafe_allow_html=True)
    cfg, error = load_config_v2_safe()
    if error is not None:
        st.caption(error)
    else:
        comparison_v2.render(cfg.evaluation_campaign_dir)


def _render_fixture_comparison() -> None:
    profile_choice = st.selectbox("Traffic profile", [p.value.capitalize() for p in TrafficProfile], key="du_cmp_profile", label_visibility="collapsed")
    profile = TrafficProfile(profile_choice.lower())

    hpa_run, ppo_run = build_fixture_pair(profile)
    result = compare(hpa_run, ppo_run)
    conclusion = generate_conclusion(result, profile, is_demo_data=True)

    same = hpa_run.schedule_hash == ppo_run.schedule_hash
    st.markdown(
        f"""
        <div class="du-panel" style="display:flex;justify-content:space-between;align-items:center;">
          <div>
            <div style="font-size:18px;font-weight:700;">{profile.value.upper()}</div>
            <div class="mono" style="font-size:12px;color:var(--muted);">Seed {hpa_run.seed} &middot; {hpa_run.duration_s:.0f}s</div>
          </div>
          <div style="text-align:right;">
            <div class="du-section-label">Traffic Match</div>
            <span class="du-badge {'status-ok' if same else 'status-error'}">{'VERIFIED' if same else 'MISMATCH'}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="mono" style="font-size:11px;color:var(--muted);margin:0.3rem 0 0.8rem 0;">'
        f'HPA sha256 {hpa_run.schedule_hash} &nbsp;&middot;&nbsp; PPO sha256 {ppo_run.schedule_hash}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="du-section-label">Comparison</div>', unsafe_allow_html=True)
    _render_table(result, hpa_run, ppo_run)

    st.markdown('<div class="du-section-label" style="margin-top:0.8rem;">Overlay Graphs</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(overlay_line_chart("Replicas", hpa_run.timestamps, hpa_run.replicas, "HPA", ppo_run.timestamps, ppo_run.replicas, "PPO", "replicas"), use_container_width=True)
    c2.plotly_chart(overlay_line_chart("P95 Latency", hpa_run.timestamps, hpa_run.latency_p95_ms, "HPA", ppo_run.timestamps, ppo_run.latency_p95_ms, "PPO", "ms"), use_container_width=True)
    st.plotly_chart(overlay_line_chart("CPU", hpa_run.timestamps, hpa_run.cpu_pct, "HPA", ppo_run.timestamps, ppo_run.cpu_pct, "PPO", "%"), use_container_width=True)

    st.markdown('<div class="du-section-label" style="margin-top:0.8rem;">Evidence</div>', unsafe_allow_html=True)
    with st.expander("HPA Events"):
        for ev in hpa_run.events:
            st.markdown(f'<div class="mono" style="font-size:12px;">{ev.timestamp:8.1f}s  {ev.category}  {ev.label}</div>', unsafe_allow_html=True)
    with st.expander("PPO Decisions"):
        if ppo_run.ppo:
            for d in ppo_run.ppo.decisions:
                st.markdown(
                    f'<div class="mono" style="font-size:12px;">{d.timestamp:8.1f}s  before={d.replicas_before} '
                    f'action={d.action.value} after={d.replicas_after} reward={d.reward:+.3f}</div>',
                    unsafe_allow_html=True,
                )
    with st.expander("Experiment Provenance"):
        st.markdown(
            f'<div class="mono" style="font-size:12px;">controller_configs: HPA(target={hpa_run.hpa.cpu_target_pct if hpa_run.hpa else "-"}%), '
            f'PPO(seed={ppo_run.seed})<br/>min_replicas=1 max_replicas=8 initial_replicas=1<br/>'
            f'application: demo-app</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="du-section-label" style="margin-top:0.8rem;">Conclusion</div>', unsafe_allow_html=True)
    _render_conclusion(conclusion)
