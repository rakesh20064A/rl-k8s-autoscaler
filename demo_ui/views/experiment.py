"""pages/experiment.py -- Main experiment control page (sec.5-7, 12).

Top bar -> horizontal control strip -> (Auto mode) pipeline stepper -> run
state panel. In DEMO mode, drives the offline state machine
(SingleControllerRun/AutoComparisonRun) via FixtureExperimentController. In
LIVE mode (Manual only -- Auto is not yet wired to real execution),
START/STOP drive RealExperimentController (demo_ui/live_execution_adapter.py),
which launches a real, isolated, non-scientific subprocess
(scripts/run_demo_episode.py).
"""

from __future__ import annotations

import streamlit as st

from demo_ui import experiment_store
from demo_ui.components.controls import render_controls, render_data_source_toggle
from demo_ui.components.controls_v2 import render_controls_v2
from demo_ui.components.layout import render_pipeline, render_topbar
from demo_ui.config_v2 import load_config_v2_safe
from demo_ui.fixtures_v2 import build_fixture_v2_run
from demo_ui.live_execution_adapter import RealExperimentController, RealExperimentControllerV2
from demo_ui.models import AutoRunState, ResetSnapshot, SingleRunState
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


@st.fragment(run_every=3)
def _render_live_status_panel(real_ctrl: RealExperimentController) -> None:
    """Periodically re-executes on its own (every 3s) WITHOUT rerunning the
    rest of the page -- `render_controls()` (and its START/STOP click
    handling) lives outside this fragment, so these refreshes can never
    re-invoke `.start()` or launch a duplicate subprocess. This is what
    keeps status/log output moving while a run is active instead of being
    frozen until the user happens to touch some other widget."""
    real_status = real_ctrl.status()
    status_label = real_status.get("status", "unknown")
    # The subprocess needs a few seconds (imports, PPO.load, etc.) before
    # it writes its first status file -- show "starting", never a bare
    # "unknown", so a just-clicked START reads as acknowledged immediately.
    display_label = "starting" if (real_ctrl.is_running and status_label == "unknown") else status_label

    # "Current run" banner (sec.8 label requirement): read from
    # experiment_store, the same record Live Details/History/Comparison
    # read, so this page's own label is never a separate source of truth.
    rec = experiment_store.get_run(real_ctrl.run_id) if real_ctrl.run_id else None
    if rec is not None:
        profile_label = rec["profile_name"].replace("_", " ").title()
        st.markdown(
            f'<div class="mono" style="font-size:12px;color:var(--muted);">'
            f'Current run: LIVE &middot; {rec["controller"]} &middot; {profile_label} &middot; '
            f'<b>{display_label.upper()}</b> &middot; {rec["run_id"]}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="du-panel mono" style="margin-top:0.4rem;">LIVE run status: <b>{display_label.upper()}</b></div>',
        unsafe_allow_html=True,
    )
    if status_label == "error":
        st.error(real_status.get("error", "unknown error"))
    elif status_label == "completed":
        st.success("Episode completed.")

    # Mirrors the DEMO state machine's own `if ...is_terminal(): du_run_in_progress
    # = False` pattern elsewhere in this file -- once the real subprocess has
    # actually finished, the control strip must flip back to READY/START on
    # its own, without a manual STOP click. `st.rerun()` (default scope="app")
    # is needed, not just the fragment's own periodic refresh: the control
    # strip with the START/STOP button lives OUTSIDE this fragment, in the
    # main page body, which a fragment-local rerun never touches. Guarded by
    # `was_running` so this fires exactly once at the transition, not on
    # every subsequent 3s tick after completion.
    was_running = st.session_state.get("du_run_in_progress", False)
    if status_label in ("completed", "error", "exited") and was_running:
        st.session_state["du_run_in_progress"] = False
        st.rerun()

    with st.expander("Recent log output", expanded=real_ctrl.is_running):
        lines = real_ctrl.recent_log_lines()
        if not lines:
            st.caption("No output yet." if real_ctrl.is_running else "No output.")
        for line in lines:
            st.markdown(f'<div class="mono" style="font-size:11px;">{line}</div>', unsafe_allow_html=True)


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


def _render_generation_toggle() -> str:
    """V2 Phase 5E: analogous to `render_data_source_toggle()` -- a plain,
    non-widget-bound public key (`du_generation`) is the durable source of
    truth, the widget itself gets a private key, same reasoning as that
    function's own docstring (a widget-bound key is cleared whenever this
    exact call is skipped on a rerun, which never happens here since only
    this page calls it, but the pattern is kept consistent). Default "V2"
    per this phase's explicit spec; V1 remains fully selectable and its own
    code path below is completely unchanged when selected."""
    st.session_state.setdefault("du_generation", "V2")
    value = st.segmented_control(
        "Generation", ["V1", "V2"], default=st.session_state["du_generation"],
        selection_mode="single", required=True,
        key="du_generation_widget", label_visibility="collapsed",
    )
    st.session_state["du_generation"] = value
    return value


@st.fragment(run_every=3)
def _render_v2_live_status_panel(real_ctrl: RealExperimentControllerV2) -> None:
    """V2 analogue of `_render_live_status_panel()` above -- same polling
    fragment pattern, plus surfacing a `"refused"` status (the
    training/evaluation-protection check inside
    `scripts/run_demo_episode_v2.py` failing closed) distinctly."""
    real_status = real_ctrl.status()
    status_label = real_status.get("status", "unknown")
    display_label = "starting" if (real_ctrl.is_running and status_label == "unknown") else status_label

    rec = experiment_store.get_run(real_ctrl.run_id) if real_ctrl.run_id else None
    if rec is not None:
        profile_label = rec["profile_name"].replace("_", " ").title()
        st.markdown(
            f'<div class="mono" style="font-size:12px;color:var(--muted);">'
            f'Current run: V2 LIVE &middot; {rec["controller"]} &middot; {profile_label} &middot; '
            f'<b>{display_label.upper()}</b> &middot; {rec["run_id"]}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="du-panel mono" style="margin-top:0.4rem;">LIVE run status: <b>{display_label.upper()}</b></div>',
        unsafe_allow_html=True,
    )
    if status_label == "refused":
        st.error(f"Refused to start: {real_status.get('error', 'unknown reason')}")
    elif status_label == "error":
        st.error(real_status.get("error", "unknown error"))
    elif status_label == "completed":
        st.success("Demo episode completed. DEMO ONLY -- not evaluation evidence.")

    was_running = st.session_state.get("du_v2_run_in_progress", False)
    if status_label in ("completed", "error", "exited", "refused") and was_running:
        st.session_state["du_v2_run_in_progress"] = False
        st.rerun()

    with st.expander("Recent log output", expanded=real_ctrl.is_running):
        lines = real_ctrl.recent_log_lines()
        if not lines:
            st.caption("No output yet." if real_ctrl.is_running else "No output.")
        for line in lines:
            st.markdown(f'<div class="mono" style="font-size:11px;">{line}</div>', unsafe_allow_html=True)


def _render_v2() -> None:
    """V2 Phase 5E: the V2 control panel -- a single arm, one of the 3
    short DEMO-ONLY profiles, DEMO (fixture) or LIVE. Writes LIVE runs into
    `experiment_store` exactly like V1 does (same API), tagged
    `generation="v2"` so Live Details/Comparison/History can tell them
    apart. Does not use the Auto/pipeline machinery -- out of this phase's
    scope."""
    if "du_v2_real_controller_impl" not in st.session_state:
        st.session_state.du_v2_real_controller_impl = RealExperimentControllerV2()

    # A config error (missing/malformed demo_ui/config_v2.yaml) must never
    # crash this page -- DEMO still works either way; only LIVE is forced
    # off, with the error itself as the disabled-reason message.
    cfg, config_error = load_config_v2_safe()
    live_enabled = cfg.live_enabled if cfg is not None else False
    live_disabled_reason = cfg.live_disabled_reason if cfg is not None else config_error
    model_path = cfg.model_path if cfg is not None else None

    st.markdown('<div class="du-section-label" style="margin-top:0.6rem;">Data Source</div>', unsafe_allow_html=True)
    data_source = render_data_source_toggle()
    is_live = data_source == "LIVE"

    active_rec = experiment_store.active_run() if is_live else None
    if active_rec is not None and active_rec.get("generation") != "v2":
        active_rec = None  # a V1 active run must never disable the V2 controls

    arm, profile_name, start_clicked, stop_clicked = render_controls_v2(
        live_enabled, live_disabled_reason, is_live, active_rec,
    )

    if is_live:
        real_ctrl = st.session_state.du_v2_real_controller_impl
        if start_clicked and live_enabled:
            st.session_state["du_v2_run_in_progress"] = True
            real_ctrl.start(arm, profile_name, model_path)
            st.rerun()
        if stop_clicked:
            st.session_state["du_v2_run_in_progress"] = False
            real_ctrl.stop()
            st.rerun()
        _render_v2_live_status_panel(real_ctrl)
        return

    if start_clicked:
        # DEMO loading is instantaneous (no subprocess) -- `du_v2_run_in_progress`
        # still flips to True so the control strip shows RUNNING/STOP,
        # consistent with the LIVE path, with STOP acting as "clear".
        st.session_state["du_v2_run_in_progress"] = True
        st.session_state["du_v2_demo_record"] = build_fixture_v2_run(arm, profile_name)
        st.rerun()
    if stop_clicked:
        st.session_state["du_v2_run_in_progress"] = False
        st.session_state.pop("du_v2_demo_record", None)
        st.rerun()

    record = st.session_state.get("du_v2_demo_record")
    if record is not None:
        profile_label = profile_name.replace("_", " ").title()
        st.success(
            f"DEMO fixture loaded: {arm} / {profile_label} — see Live Details for the full view. "
            f"DEMO ONLY, not evaluation evidence."
        )
    else:
        st.caption("No V2 demo run loaded. Configure the controls above and press START.")


def render() -> None:
    if "du_controller_impl" not in st.session_state:
        st.session_state.du_controller_impl = FixtureExperimentController()
    if "du_real_controller_impl" not in st.session_state:
        st.session_state.du_real_controller_impl = RealExperimentController()
    if "du_single_run" not in st.session_state:
        st.session_state.du_single_run = None
    if "du_auto_run" not in st.session_state:
        st.session_state.du_auto_run = None

    # Badge reflects the last-known value so it's accurate even on the
    # SAME rerun that processes a click on the selector rendered below --
    # Streamlit updates session_state for a clicked widget before the
    # script reruns, so this is never a render behind (same pattern already
    # used for `auto_on` ahead of the Mode widget in controls.py).
    prior_data_source = st.session_state.get("du_data_source", "DEMO")
    render_topbar(mode_label=prior_data_source, is_live=(prior_data_source == "LIVE"))

    st.markdown('<div class="du-section-label" style="margin-top:0.6rem;">Generation</div>', unsafe_allow_html=True)
    generation = _render_generation_toggle()
    if generation == "V2":
        _render_v2()
        return

    # ---- V1 path (UNCHANGED from before V2 Phase 5E) ----
    st.markdown('<div class="du-section-label" style="margin-top:0.6rem;">Data Source</div>', unsafe_allow_html=True)
    data_source = render_data_source_toggle()

    is_live = data_source == "LIVE"
    controller_impl = st.session_state.du_real_controller_impl if is_live else st.session_state.du_controller_impl

    # Active-run authority (see controls.py docstring): while a real LIVE
    # run is in progress, its OWN stored identity -- not local widget
    # state -- is what Controller/Traffic must display, including on a
    # fresh session_state (e.g. after a Streamlit restart) that never saw
    # this run's own START click.
    active_rec = experiment_store.active_run() if is_live else None

    selected_controller, auto_on, traffic_profile, start_clicked, stop_clicked = render_controls(
        controller_impl, is_live=is_live, active_rec=active_rec,
    )

    if is_live:
        if auto_on:
            st.warning("Auto (HPA-then-PPO) mode is not wired to real execution yet. Use Manual mode for a live run.")
        else:
            _render_live_status_panel(st.session_state.du_real_controller_impl)
        return

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
