"""pages/live_details.py -- Technical inspection page (sec.16).

READ-ONLY monitor of whatever Experiment is doing. This page has NO
controller/traffic-profile/DEMO-LIVE selector of its own, and the primary
content is NEVER gated behind `st.session_state["du_data_source"]` --
Experiment is the only control plane, and relying on that shared key here
was itself the bug that let the fixture dashboard render by default on a
fresh Streamlit session (before Experiment's own widget had ever run once
to seed it). The primary view always shows experiment_store's real
active/latest run -- `—`/empty-state placeholders, never fixture numbers,
until that run actually exists. Fixture/demo content, if kept at all, lives
only inside one collapsed expander at the bottom, clearly labeled.
"""

from __future__ import annotations

import streamlit as st

from demo_ui import experiment_store
from demo_ui.components.charts import line_chart, step_chart
from demo_ui.components.controller_status import render_hpa_panel, render_ppo_panel
from demo_ui.components.layout import render_metric_strip, render_topbar
from demo_ui.components.logs import render_log_viewer
from demo_ui.components.pods import render_pod_panel
from demo_ui.fixtures import build_all_fixture_summaries
from demo_ui.fixtures_v2 import ARMS_V2, PROFILES_V2, build_fixture_v2_run
from demo_ui.live_providers import KubernetesClusterProvider, NOT_AVAILABLE, PrometheusMetricsProvider
from demo_ui.models import Controller, TrafficProfile
from demo_ui.views import live_details_v2

# Presentation-only axis ranges for an EMPTY chart (never measured data) --
# keeps a stable, sensible frame instead of Plotly's arbitrary auto-range
# for a trace with zero points. Real data overrides these entirely once it
# exists (callers only pass these when there is no data yet).
_REPLICAS_EMPTY_RANGE = (0, 8)  # autoscaler's own canonical min1/max8
_LATENCY_EMPTY_RANGE = (0, 500)
_CPU_EMPTY_RANGE = (0, 1)


def _fmt(value, suffix: str = "") -> str:
    return NOT_AVAILABLE if value is None else f"{value:.1f}{suffix}"


def _run_status_label(rec: dict) -> str:
    return (rec.get("stage") or rec.get("status", "unknown")).upper()


def _render_run_banner(rec: dict | None, is_active: bool) -> None:
    if rec is None:
        st.markdown(
            '<div class="du-panel mono" style="margin-bottom:0.6rem;">No LIVE experiment has been run yet.</div>',
            unsafe_allow_html=True,
        )
        return
    profile_label = rec["profile_name"].replace("_", " ").title()
    prefix = "Current run" if is_active else "Latest completed run"
    st.markdown(
        f'<div class="du-panel mono" style="margin-bottom:0.6rem;">{prefix}: LIVE &middot; '
        f'{rec["controller"]} &middot; {profile_label} &middot; <b>{_run_status_label(rec)}</b> '
        f'&middot; {rec["run_id"]}</div>',
        unsafe_allow_html=True,
    )


def _render_run_telemetry(rec: dict | None) -> None:
    """Metric cards + charts for the SAME run the banner names. Every
    value here comes from that run's own persisted samples -- never the
    instantaneous cluster read (see `_render_current_cluster_state` for
    that, kept deliberately separate) and never a fixture."""
    st.markdown('<div class="du-section-label" style="margin-top:0.6rem;">Experiment Telemetry</div>', unsafe_allow_html=True)

    has_samples = bool((rec.get("summary") or {}).get("samples")) if rec is not None else False
    empty_detail = None
    if rec is None:
        empty_primary = "No data available"
        empty_detail = "Run an experiment to view this metric"
    elif not has_samples:
        empty_primary = "Waiting for first measurement…"
    else:
        empty_primary = None

    ts_r, replicas = experiment_store.extract_series(rec, "replicas_desired")
    ts_l, p95 = experiment_store.extract_series(rec, "latency_p95_ms")
    ts_c, cpu = experiment_store.extract_series(rec, "cpu_cores")
    ts_q, rps = experiment_store.extract_series(rec, "request_rate_rps")

    render_metric_strip([
        ("Controller", rec["controller"] if rec is not None else "—"),
        ("Traffic", rec["profile_name"].replace("_", " ").title() if rec is not None else "—"),
        ("RPS", f"{rps[-1]:.1f}" if rps else "—"),
        ("CPU (cores)", f"{cpu[-1]:.3f}" if cpu else "—"),
        ("Replicas", str(replicas[-1]) if replicas else "—"),
        ("P95", f"{p95[-1]:.0f} ms" if p95 else "—"),
    ])

    c1, c2 = st.columns(2)
    c1.plotly_chart(
        step_chart("Replicas over time", ts_r, replicas, "replicas",
                   empty_message=empty_primary, empty_detail=empty_detail, empty_y_range=_REPLICAS_EMPTY_RANGE),
        use_container_width=True,
    )
    c2.plotly_chart(
        line_chart("P95 Latency", ts_l, {"p95": p95}, "ms",
                   empty_message=empty_primary, empty_detail=empty_detail, empty_y_range=_LATENCY_EMPTY_RANGE),
        use_container_width=True,
    )
    st.plotly_chart(
        line_chart("CPU", ts_c, {"cpu": cpu}, "cores",
                   empty_message=empty_primary, empty_detail=empty_detail, empty_y_range=_CPU_EMPTY_RANGE),
        use_container_width=True,
    )


def _render_experiment_events(rec: dict | None) -> None:
    st.markdown('<div class="du-section-label" style="margin-top:0.5rem;">Experiment Events</div>', unsafe_allow_html=True)
    summary = (rec.get("summary") or {}) if rec is not None else {}
    scaling_events = summary.get("scaling_events")
    if rec is None or scaling_events is None:
        st.markdown('<div class="du-panel mono">No experiment events yet.</div>', unsafe_allow_html=True)
        return
    st.markdown(f'<div class="du-panel mono">{scaling_events} scaling event(s) recorded this run.</div>', unsafe_allow_html=True)


def _render_current_cluster_state() -> None:
    """Deliberately separate from, and never mixed into, the experiment
    telemetry above: this is the live cluster's state RIGHT NOW, which may
    have nothing to do with any experiment run (e.g. idle between runs).
    Still real, read-only Kubernetes/Prometheus data -- never fabricated --
    just explicitly scoped so it can never be mistaken for the run's own
    historical measurements."""
    st.markdown('<div class="du-section-label" style="margin-top:0.8rem;">Current Cluster State</div>', unsafe_allow_html=True)
    st.caption("Requires an already-running, externally-managed Prometheus port-forward on localhost:9090. This page never starts or stops one.")

    k8s = KubernetesClusterProvider()
    prom = PrometheusMetricsProvider()

    dep = k8s.get_deployment_replica_state()
    hpa = k8s.get_hpa_snapshot()
    pods = k8s.get_pods()
    restarts = k8s.get_pod_restart_counts()
    snap = prom.get_live_snapshot()
    events = k8s.get_recent_events()

    ready_count = sum(1 for p in pods if p.state.value == "READY")
    render_metric_strip([
        ("Pods ready", f"{ready_count}/{len(pods)}" if pods else NOT_AVAILABLE),
        ("Deployment", f"{dep['desired']}/{dep['ready']}" if dep else NOT_AVAILABLE),
        ("HPA present", "Yes" if hpa is not None else "No"),
        ("Prometheus", "Reachable" if snap.target_up else f"DOWN: {snap.target_health_message}"),
    ])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="du-section-label" style="margin-top:0.5rem;">HPA</div>', unsafe_allow_html=True)
        if hpa is not None:
            render_hpa_panel(hpa)
        else:
            st.markdown(f'<div class="du-panel mono">{NOT_AVAILABLE}</div>', unsafe_allow_html=True)
        st.markdown('<div class="du-section-label" style="margin-top:0.5rem;">PPO</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="du-panel mono">{NOT_AVAILABLE} — no live PPO decision feed wired up in this phase</div>', unsafe_allow_html=True)
    with col2:
        render_pod_panel(pods, restarts=restarts)

    st.markdown('<div class="du-section-label" style="margin-top:0.5rem;">Recent Kubernetes Events</div>', unsafe_allow_html=True)
    if events:
        rows = "\n".join(f"{e['lastTimestamp']:<22} {e['type']:<8} {e['reason']:<20} {e['message']}" for e in events)
        st.markdown(f'<pre class="mono" style="font-size:12px;">{rows}</pre>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="du-panel mono">{NOT_AVAILABLE}</div>', unsafe_allow_html=True)


def _render_fixture_dashboard() -> None:
    summaries = build_all_fixture_summaries()

    col1, col2 = st.columns(2)
    profile_choice = col1.selectbox("Traffic profile", [p.value.capitalize() for p in TrafficProfile], key="du_live_profile")
    controller_choice = col2.selectbox("Controller", [Controller.HPA.value, Controller.PPO.value], key="du_live_controller")

    profile = TrafficProfile(profile_choice.lower())
    controller = Controller(controller_choice)
    run = summaries[f"{profile.value}_{controller.value.lower()}"]

    st.markdown('<div class="du-section-label">Cluster / Traffic</div>', unsafe_allow_html=True)
    render_metric_strip([
        ("Controller", run.controller.value),
        ("Traffic", run.traffic_profile.value.capitalize()),
        ("RPS", f"{run.rps[-1]:.1f}" if run.rps else "–"),
        ("CPU", f"{run.cpu_pct[-1]:.0f}%" if run.cpu_pct else "–"),
        ("Replicas", str(run.replicas[-1]) if run.replicas else "–"),
        ("P95", f"{run.latency_p95_ms[-1]:.0f} ms" if run.latency_p95_ms else "–"),
    ])

    st.markdown('<div class="du-section-label" style="margin-top:0.6rem;">Charts</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(line_chart("Traffic (req/s)", run.timestamps, {"rps": run.rps}, "req/s", run.events), use_container_width=True)
    c2.plotly_chart(line_chart("CPU (%)", run.timestamps, {"cpu": run.cpu_pct}, "%", run.events), use_container_width=True)

    c3, c4 = st.columns(2)
    c3.plotly_chart(step_chart("Replicas", run.timestamps, run.replicas, "replicas", run.events), use_container_width=True)
    c4.plotly_chart(
        line_chart(
            "Latency",
            run.timestamps,
            {"avg": run.latency_avg_ms, "p95": run.latency_p95_ms, "p99": run.latency_p99_ms},
            "ms",
            run.events,
        ),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="du-section-label">HPA / PPO</div>', unsafe_allow_html=True)
        if run.hpa is not None:
            render_hpa_panel(run.hpa)
        if run.ppo is not None:
            render_ppo_panel(run.ppo)
    with col2:
        render_pod_panel(run.pods)

    render_log_viewer(run.logs)


def _render_v2_fixture_dashboard() -> None:
    col1, col2 = st.columns(2)
    profile_choice = col1.selectbox(
        "Demo traffic profile", [p.replace("_", " ").title() for p in PROFILES_V2], key="du_live_v2_profile",
    )
    arm_choice = col2.selectbox("Arm", [a.replace("_", " ").upper() for a in ARMS_V2], key="du_live_v2_arm")
    profile_name = profile_choice.lower().replace(" ", "_")
    arm = arm_choice.lower().replace(" ", "_")
    rec = build_fixture_v2_run(arm, profile_name)
    live_details_v2.render(rec, is_active=False)


def render() -> None:
    render_topbar(mode_label="LIVE", is_live=True)

    rec = experiment_store.active_run()
    is_active = rec is not None
    if rec is None:
        rec = experiment_store.latest_run()

    if rec is not None and rec.get("generation") == "v2":
        # V2 Phase 5E: a V2 run (LIVE or DEMO-fixture-originated) is the
        # most recent/active one -- delegate entirely to the V2 renderer.
        # V1's own rendering path below is completely unchanged for any
        # V1-only history.
        live_details_v2.render(rec, is_active)
    else:
        _render_run_banner(rec, is_active)
        _render_run_telemetry(rec)
        _render_experiment_events(rec)

    _render_current_cluster_state()

    with st.expander("Demo fixture data (simulated, not real LIVE results)"):
        st.caption("Optional simulated reference data -- not generated until you load it.")
        if st.button("Load demo fixture data", key="du_load_live_fixture"):
            st.session_state["du_show_live_fixture"] = True
        if st.session_state.get("du_show_live_fixture", False):
            _render_fixture_dashboard()

    with st.expander("V2 demo fixture data (simulated, not evaluation evidence)"):
        st.caption("Optional simulated V2 reference data -- not generated until you load it.")
        if st.button("Load V2 demo fixture data", key="du_load_live_fixture_v2"):
            st.session_state["du_show_live_fixture_v2"] = True
        if st.session_state.get("du_show_live_fixture_v2", False):
            _render_v2_fixture_dashboard()
