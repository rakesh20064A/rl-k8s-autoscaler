"""pages/live_details.py -- Technical inspection page (sec.16): CLUSTER,
PROMETHEUS, HPA, PPO, TRAFFIC, EVENTS, PROVENANCE. Denser than the
executive-dashboard feel of other pages -- engineering diagnostics.

Supports DEMO (fixture, full time series) and LIVE (real read-only
Kubernetes + Prometheus instant snapshot) data sources (sec.9). LIVE mode
never fabricates values: any field the real runtime cannot currently answer
renders `Not available`.
"""

from __future__ import annotations

import streamlit as st

from demo_ui.components.charts import line_chart, step_chart
from demo_ui.components.controller_status import render_hpa_panel, render_ppo_panel
from demo_ui.components.layout import render_metric_strip, render_topbar
from demo_ui.components.logs import render_log_viewer
from demo_ui.components.pods import render_pod_panel
from demo_ui.fixtures import build_all_fixture_summaries
from demo_ui.live_providers import KubernetesClusterProvider, NOT_AVAILABLE, PrometheusMetricsProvider
from demo_ui.models import Controller, TrafficProfile


def _fmt(value, suffix: str = "") -> str:
    return NOT_AVAILABLE if value is None else f"{value:.1f}{suffix}"


def _render_live_snapshot() -> None:
    st.caption("Requires an already-running, externally-managed Prometheus port-forward on localhost:9090. This page never starts or stops one.")

    k8s = KubernetesClusterProvider()
    prom = PrometheusMetricsProvider()

    dep = k8s.get_deployment_replica_state()
    hpa = k8s.get_hpa_snapshot()
    pods = k8s.get_pods()
    restarts = k8s.get_pod_restart_counts()
    snap = prom.get_live_snapshot()
    events = k8s.get_recent_events()

    st.markdown('<div class="du-section-label">Cluster</div>', unsafe_allow_html=True)
    render_metric_strip([
        ("Deployment", f"{dep['desired']}/{dep['ready']}" if dep else NOT_AVAILABLE),
        ("Pods", str(len(pods))),
        ("Node", "Ready"),
    ])

    st.markdown('<div class="du-section-label" style="margin-top:0.5rem;">Prometheus</div>', unsafe_allow_html=True)
    render_metric_strip([
        ("Target Health", snap.target_health_message if snap.target_up else f"DOWN: {snap.target_health_message}"),
        ("RPS", _fmt(snap.request_rate_rps)),
        ("CPU (cores)", _fmt(snap.cpu_cores)),
        ("Memory (MB)", _fmt(snap.memory_bytes / 1e6 if snap.memory_bytes is not None else None)),
        ("P50", _fmt(snap.latency_p50_ms, " ms")),
        ("P95", _fmt(snap.latency_p95_ms, " ms")),
        ("P99", _fmt(snap.latency_p99_ms, " ms")),
        ("Error rate", _fmt(snap.error_rate_rps)),
    ])
    st.caption("Charts and the evidence log require a running historical sampler and remain DEMO-only in this phase.")

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

    st.markdown('<div class="du-section-label" style="margin-top:0.5rem;">Events</div>', unsafe_allow_html=True)
    if events:
        rows = "\n".join(f"{e['lastTimestamp']:<22} {e['type']:<8} {e['reason']:<20} {e['message']}" for e in events)
        st.markdown(f'<pre class="mono" style="font-size:12px;">{rows}</pre>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="du-panel mono">{NOT_AVAILABLE}</div>', unsafe_allow_html=True)

    st.markdown('<div class="du-section-label" style="margin-top:0.5rem;">Provenance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="du-panel mono" style="font-size:12px;">Data source: LIVE read-only (Kubernetes + Prometheus)</div>',
        unsafe_allow_html=True,
    )


def render() -> None:
    data_source = st.radio("Data source", ["DEMO", "LIVE"], horizontal=True, key="du_live_details_source", label_visibility="collapsed")
    render_topbar(mode_label=data_source, is_live=(data_source == "LIVE"))

    if data_source == "LIVE":
        _render_live_snapshot()
        return

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
