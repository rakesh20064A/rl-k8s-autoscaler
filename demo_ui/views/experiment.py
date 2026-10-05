"""pages/experiment.py -- Main experiment control page (sec.5-7, 12).

Top bar -> horizontal control strip -> (Auto mode) pipeline stepper -> run
state panel. Drives the offline state machine (SingleControllerRun or
AutoComparisonRun) using ONLY the FixtureExperimentController. No live
infrastructure call exists in this module -- a parallel LiveExperimentController
plan preview is shown read-only when LIVE data source is selected, but it is
never executed.
"""

from __future__ import annotations

import streamlit as st

from demo_ui.components.controls import render_controls, render_data_source_toggle
from demo_ui.components.layout import render_pipeline, render_topbar
from demo_ui.live_adapter import LiveExperimentController
from demo_ui.models import AutoRunState, Controller, ResetSnapshot, SingleRunState
from demo_ui.providers import FixtureExperimentController
from demo_ui.state_machine import AutoComparisonRun, SingleControllerRun

_AUTO_PIPELINE_STEPS = [
    ("hpa", "HPA"), ("reset", "RESET"), ("ppo", "PPO"), ("validate", "VALIDATE"), ("result", "RESULT"),
]

_AUTO_STATE_TO_PIPELINE_INDEX = {
    AutoRunState.IDLE: 0,
    AutoRunState.PREPARING_HPA: 0,
    AutoRunState.HPA_RUNNING: 0,
    AutoRunState.HPA_COMPLETE: 1,
    AutoRunState.RESETTING: 1,
    AutoRunState.BASELINE_STABLE: 1,
    AutoRunState.PREPARING_PPO: 2,
    AutoRunState.PPO_RUNNING: 2,
    AutoRunState.PPO_COMPLETE: 3,
    AutoRunState.VALIDATING: 3,
    AutoRunState.COMPARISON_READY: 4,
}


def _render_reset_panel(snapshot: ResetSnapshot) -> None:
    st.markdown('<div class="du-section-label">Resetting Environment</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="du-panel mono" style="display:flex;gap:2rem;">'
        f'<span>Traffic&nbsp;&nbsp;{snapshot.traffic_rps:.0f} req/s</span>'
        f'<span>Replicas&nbsp;&nbsp;{snapshot.replicas}</span>'
        f'<span>Application&nbsp;&nbsp;{"Ready" if snapshot.application_ready else "Not Ready"}</span>'
        f'<span>Metrics&nbsp;&nbsp;{"Stabilizing" if snapshot.metrics_stabilizing else "Stable"}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render() -> None:
    if "du_controller_impl" not in st.session_state:
        st.session_state.du_controller_impl = FixtureExperimentController()
    if "du_live_plan_impl" not in st.session_state:
        st.session_state.du_live_plan_impl = LiveExperimentController()
    if "du_single_run" not in st.session_state:
        st.session_state.du_single_run = None
    if "du_auto_run" not in st.session_state:
        st.session_state.du_auto_run = None

    data_source = render_data_source_toggle()
    render_topbar(mode_label="LIVE" if data_source == "LIVE" else "DEMO", is_live=(data_source == "LIVE"))

    controller_impl = st.session_state.du_controller_impl
    selected_controller, auto_on, traffic_profile, start_clicked, stop_clicked = render_controls(controller_impl)

    if data_source == "LIVE":
        if start_clicked:
            st.session_state.du_live_plan_impl.start(selected_controller or Controller.HPA, traffic_profile)
        if st.session_state.du_live_plan_impl.planned_operations:
            with st.expander("Planned live operations (not executed in this phase)"):
                for line in st.session_state.du_live_plan_impl.describe_plan():
                    st.markdown(f'<div class="mono" style="font-size:12px;">{line}</div>', unsafe_allow_html=True)

    if start_clicked:
        if auto_on:
            st.session_state.du_auto_run = AutoComparisonRun()
            st.session_state.du_single_run = None
        else:
            st.session_state.du_single_run = SingleControllerRun(controller=selected_controller)
            st.session_state.du_auto_run = None

    if stop_clicked:
        if st.session_state.du_single_run is not None:
            st.session_state.du_single_run.stop()
        controller_impl.stop()

    single_run = st.session_state.du_single_run
    auto_run = st.session_state.du_auto_run

    if single_run is not None:
        st.markdown(
            f'<div class="du-panel" style="margin-top:0.4rem;">'
            f'<span class="du-section-label">Run State</span><br/>'
            f'<span class="mono" style="font-size:16px;font-weight:700;">{single_run.state.value}</span></div>',
            unsafe_allow_html=True,
        )
        if single_run.state == SingleRunState.RESETTING and single_run.reset_snapshot:
            _render_reset_panel(single_run.reset_snapshot)
        if single_run.is_terminal():
            st.session_state["du_run_in_progress"] = False
        if st.button("Advance (demo step)", key="du_advance_single") and not single_run.is_terminal():
            single_run.advance()
            st.rerun()

    elif auto_run is not None:
        pipeline_index = _AUTO_STATE_TO_PIPELINE_INDEX.get(auto_run.state, 0)
        render_pipeline(_AUTO_PIPELINE_STEPS, None if auto_run.is_terminal() else pipeline_index)
        st.caption(f"Auto run state: `{auto_run.state.value}`")

        if auto_run.state == AutoRunState.RESETTING and auto_run.reset_snapshot is None:
            _render_reset_panel(ResetSnapshot(traffic_rps=0.0, replicas=1, application_ready=False, metrics_stabilizing=True))
        if auto_run.state == AutoRunState.VALIDATING:
            if st.button("Validate fair-replay (demo)", key="du_validate_auto"):
                auto_run.validate(same_schedule_verified=True)
                st.rerun()
        elif not auto_run.is_terminal():
            if st.button("Advance (demo step)", key="du_advance_auto"):
                auto_run.advance()
                st.rerun()
        if auto_run.is_terminal():
            st.session_state["du_run_in_progress"] = False
            st.success("COMPARISON_READY — see the Comparison page.")
    else:
        st.caption("No run in progress. Configure the controls above and press START.")
