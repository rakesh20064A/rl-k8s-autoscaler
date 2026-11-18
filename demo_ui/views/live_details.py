"""pages/live_details.py -- Technical inspection page (sec.16).

READ-ONLY monitor of whatever Experiment is doing. This page has NO
controller/profile/data-source selector of its own, and the primary
content is NEVER gated behind `st.session_state["du_data_source"]` --
Experiment is the only control plane, and relying on that shared key here
was itself the bug that let the fixture dashboard render by default on a
fresh Streamlit session (before Experiment's own widget had ever run once
to seed it). The primary view always shows experiment_store's real
active/latest run -- `—`/empty-state placeholders, never fixture numbers,
until that run actually exists. Fixture/sample-data content, if kept at
all, lives only inside one collapsed expander at the bottom, clearly
labeled as simulated.
"""

from __future__ import annotations

import streamlit as st

from demo_ui import experiment_store
from demo_ui.components.controller_status import render_hpa_panel
from demo_ui.components.controls_v2 import display_arm_label
from demo_ui.components.layout import render_metric_strip, render_topbar
from demo_ui.components.pods import render_pod_panel
from demo_ui.fixtures_v2 import ARMS_V2, PROFILES_V2, build_fixture_v2_run
from demo_ui.live_providers import KubernetesClusterProvider, NOT_AVAILABLE, PrometheusMetricsProvider
from demo_ui.views import live_details_v2


def _only_current_system(rec: dict | None) -> dict | None:
    """A run from a different generation must never be shown as the
    current/latest run on this page."""
    if rec is not None and rec.get("generation") != "v2":
        return None
    return rec


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


def _render_sample_data_dashboard() -> None:
    st.caption("SIMULATED DATA -- not evaluation evidence.")
    col1, col2 = st.columns(2)
    profile_choice = col1.selectbox(
        "Demo traffic profile", [p.replace("_", " ").title() for p in PROFILES_V2], key="du_live_v2_profile",
    )
    arm_labels = [display_arm_label(a) for a in ARMS_V2]
    arm_choice = col2.selectbox("Arm", arm_labels, key="du_live_v2_arm")
    profile_name = profile_choice.lower().replace(" ", "_")
    arm = ARMS_V2[arm_labels.index(arm_choice)]
    rec = build_fixture_v2_run(arm, profile_name)
    live_details_v2.render(rec, is_active=False)


def render() -> None:
    render_topbar(mode_label="Live cluster", is_live=True)

    rec = _only_current_system(experiment_store.active_run())
    is_active = rec is not None
    if rec is None:
        rec = _only_current_system(experiment_store.latest_run())

    live_details_v2.render(rec, is_active)
    _render_current_cluster_state()

    with st.expander("Sample data (offline) -- simulated, not real LIVE results"):
        st.caption("Optional simulated reference data -- not generated until you load it.")
        if st.button("Load sample data", key="du_load_live_fixture_v2"):
            st.session_state["du_show_live_fixture_v2"] = True
        if st.session_state.get("du_show_live_fixture_v2", False):
            _render_sample_data_dashboard()
