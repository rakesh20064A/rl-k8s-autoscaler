"""components/controls.py -- Horizontal experiment-control strip (sec.5-7,
9, 19), replacing a stack of default Streamlit widgets with one compact row:
CONTROLLER / MODE / TRAFFIC / STATUS / START-STOP.

Buttons here call ONLY an ExperimentController implementation passed in by
the caller (FixtureExperimentController or LiveExperimentController, both
dry-run/no-op against real infrastructure in this phase). A
`du_run_in_progress` session-state flag guards against Streamlit reruns
re-triggering `start()` a second time for the same click (sec.19).

Dependency note: requires `streamlit` (install via demo_ui/requirements-ui.txt
into a SEPARATE venv -- never the project's scientific .venv).
"""

from __future__ import annotations

from typing import Optional, Tuple

import streamlit as st

from demo_ui.fixtures import TRAFFIC_PROFILE_DESCRIPTIONS
from demo_ui.models import Controller, TrafficProfile
from demo_ui.providers import ExperimentController


def render_data_source_toggle() -> str:
    """Returns 'DEMO' or 'LIVE'. Never silently mixes the two -- the caller
    must branch on this return value and render an unambiguous banner."""
    return st.radio(
        "Data source", ["DEMO", "LIVE"], horizontal=True, key="du_data_source", label_visibility="collapsed",
    )


def render_controls(controller_impl: ExperimentController) -> Tuple[Optional[Controller], bool, TrafficProfile, bool, bool]:
    """Renders the horizontal control strip and returns the CURRENT widget
    state plus whether START/STOP were just clicked this render:

        (selected_controller_or_None, auto_on, traffic_profile, start_clicked, stop_clicked)

    `selected_controller_or_None` is None when auto_on is True (Auto mode
    selects its own HPA-then-PPO sequence; sec.6)."""
    already_running = st.session_state.get("du_run_in_progress", False)

    st.markdown('<div class="du-control-strip">', unsafe_allow_html=True)
    c_controller, c_mode, c_traffic, c_status, c_action = st.columns([2, 2, 2.2, 1.6, 1.4])

    # Mode is read from session state ahead of the Controller column so the
    # Controller cell can render correctly on the FIRST pass too (Streamlit
    # evaluates columns top-to-bottom in source order, left-to-right in
    # layout) -- the Mode widget itself is rendered below and writes back to
    # this same key.
    auto_on = st.session_state.get("du_mode_choice", "Manual") == "Auto"

    with c_controller:
        if auto_on:
            st.markdown('<span class="du-badge">Managed by Auto</span>', unsafe_allow_html=True)
            selected_controller = None
        else:
            choice = st.radio(
                "Controller", [Controller.HPA.value, Controller.PPO.value],
                horizontal=True, key="du_controller_choice", label_visibility="collapsed",
            )
            selected_controller = Controller(choice)

    with c_mode:
        st.markdown('<div class="du-section-label">Mode</div>', unsafe_allow_html=True)
        mode_choice = st.radio(
            "Mode", ["Manual", "Auto"], horizontal=True, key="du_mode_choice", label_visibility="collapsed",
            index=1 if auto_on else 0,
        )
        auto_on = mode_choice == "Auto"

    with c_traffic:
        st.markdown('<div class="du-section-label">Traffic</div>', unsafe_allow_html=True)
        profile_choice = st.selectbox(
            "Traffic Profile", [p.value.capitalize() for p in TrafficProfile],
            key="du_profile_choice", label_visibility="collapsed",
        )
        traffic_profile = TrafficProfile(profile_choice.lower())

    with c_status:
        st.markdown('<div class="du-section-label">Status</div>', unsafe_allow_html=True)
        status_label = "RUNNING" if already_running else "READY"
        status_class = "status-warn" if already_running else "status-ok"
        st.markdown(f'<span class="du-badge {status_class}">{status_label}</span>', unsafe_allow_html=True)

    with c_action:
        st.markdown('<div class="du-section-label">&nbsp;</div>', unsafe_allow_html=True)
        if already_running:
            stop_clicked = st.button("STOP", use_container_width=True)
            start_clicked = False
        else:
            start_clicked = st.button("START", use_container_width=True)
            stop_clicked = False

    st.markdown(
        f'<div style="font-size:12px;color:var(--muted);margin:0.3rem 0 0.5rem 0;">'
        f'{TRAFFIC_PROFILE_DESCRIPTIONS[traffic_profile]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if start_clicked and not already_running:
        st.session_state["du_run_in_progress"] = True
        controller_impl.start(selected_controller or Controller.HPA, traffic_profile)
    if stop_clicked:
        st.session_state["du_run_in_progress"] = False
        controller_impl.stop()

    return selected_controller, auto_on, traffic_profile, start_clicked, stop_clicked
