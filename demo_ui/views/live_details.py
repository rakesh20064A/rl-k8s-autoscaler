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

from demo_ui.components.controller_status import render_hpa_panel
from demo_ui.components.controls_v2 import display_arm_label
from demo_ui.components.layout import render_metric_strip, render_topbar
from demo_ui.components.pods import render_pod_panel
from demo_ui.fixtures_v2 import ARMS_V2, build_fixture_v2_run
from demo_ui.live_providers import KubernetesClusterProvider, NOT_AVAILABLE, PrometheusMetricsProvider
from demo_ui.views import live_details_v2
from scripts.generate_demo_profiles import (
    DEFAULT_LOAD,
    DEFAULT_PATTERN,
    DEFAULT_RESOURCE,
    LOAD_LABELS,
    LOADS,
    PATTERN_LABELS,
    PATTERNS,
    RESOURCE_LABELS,
    RESOURCES,
    profile_slug,
)


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

    # Task sec.4: the confusing "HPA not available"/"PPO not available"
    # pair this section used to ALWAYS show side by side (one or the
    # other -- often both -- unavailable depending on which controller
    # happened to hold the cluster right now) is removed. The run-
    # specific decision/HPA panel for whichever controller actually
    # produced the SELECTED run lives in `live_details_v2.render_primary()`
    # above, which already shows exactly one, correctly labeled panel --
    # this section is scoped to the live cluster's CURRENT raw state only
    # (see the module docstring), so it shows the real HPA snapshot when
    # one happens to be present and nothing else.
    if hpa is not None:
        st.markdown('<div class="du-section-label" style="margin-top:0.5rem;">HPA</div>', unsafe_allow_html=True)
        render_hpa_panel(hpa)
    render_pod_panel(pods, restarts=restarts)

    st.markdown('<div class="du-section-label" style="margin-top:0.5rem;">Recent Kubernetes Events</div>', unsafe_allow_html=True)
    if events:
        rows = "\n".join(f"{e['lastTimestamp']:<22} {e['type']:<8} {e['reason']:<20} {e['message']}" for e in events)
        st.markdown(f'<pre class="mono" style="font-size:12px;">{rows}</pre>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="du-panel mono">{NOT_AVAILABLE}</div>', unsafe_allow_html=True)


def _render_sample_data_dashboard() -> None:
    st.caption("SIMULATED DATA -- not evaluation evidence.")
    col1, col2, col3, col4 = st.columns(4)
    pattern_labels = [PATTERN_LABELS[p] for p in PATTERNS]
    resource_labels = [RESOURCE_LABELS[r] for r in RESOURCES]
    load_labels = [LOAD_LABELS[lvl] for lvl in LOADS]
    pattern_choice = col1.selectbox(
        "Traffic pattern", pattern_labels, index=PATTERNS.index(DEFAULT_PATTERN), key="du_live_v2_pattern",
    )
    resource_choice = col2.selectbox(
        "Resource type", resource_labels, index=RESOURCES.index(DEFAULT_RESOURCE), key="du_live_v2_resource",
    )
    load_choice = col3.selectbox(
        "Load level", load_labels, index=LOADS.index(DEFAULT_LOAD), key="du_live_v2_load",
    )
    arm_labels = [display_arm_label(a) for a in ARMS_V2]
    arm_choice = col4.selectbox("Arm", arm_labels, key="du_live_v2_arm")

    pattern = PATTERNS[pattern_labels.index(pattern_choice)]
    resource = RESOURCES[resource_labels.index(resource_choice)]
    load = LOADS[load_labels.index(load_choice)]
    profile_name = profile_slug(pattern, resource, load)
    arm = ARMS_V2[arm_labels.index(arm_choice)]
    rec = build_fixture_v2_run(arm, profile_name)
    live_details_v2.render(rec, is_active=False)


def render() -> None:
    render_topbar(mode_label="Live cluster", is_live=True)

    # Task sec.4: the run selector (default = the active run if one
    # exists, else the latest completed run) now lives inside
    # `render_primary()` itself, which also filters to v2-generation runs
    # the same way `_only_current_system()` used to here.
    live_details_v2.render_primary()
    _render_current_cluster_state()

    with st.expander("Sample data (offline) -- simulated, not real LIVE results"):
        st.caption("Optional simulated reference data -- not generated until you load it.")
        if st.button("Load sample data", key="du_load_live_fixture_v2"):
            st.session_state["du_show_live_fixture_v2"] = True
        if st.session_state.get("du_show_live_fixture_v2", False):
            _render_sample_data_dashboard()
