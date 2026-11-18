"""pages/experiment.py -- Main experiment control page (sec.5-7, 12).

Single control panel for the PPO multi-metric autoscaler: one of its two
arms, one of its 3 short DEMO-ONLY profiles, Live cluster or Sample data
(offline). START/STOP drive RealExperimentControllerV2
(demo_ui/live_execution_adapter.py), which launches a real, isolated,
non-scientific subprocess (scripts/run_demo_episode_v2.py) when running
against the live cluster.
"""

from __future__ import annotations

import streamlit as st

from demo_ui import experiment_store
from demo_ui.components.charts import line_chart
from demo_ui.components.controls import render_data_source_toggle
from demo_ui.components.controls_v2 import display_arm_label, render_controls_v2
from demo_ui.components.layout import render_topbar
from demo_ui.config_v2 import load_config_v2_safe
from demo_ui.fixtures_v2 import build_fixture_v2_run
from demo_ui.live_execution_adapter import RealExperimentControllerV2
from demo_ui.profile_preview import compute_rps_curve
from scripts.generate_demo_profiles import PATTERN_LABELS

# Short, templated-by-resource-type descriptions of what each
# pattern x resource combination exercises. Not 36 bespoke sentences --
# templated per resource type with the pattern label substituted in, which
# reads naturally for all 6 patterns without hand-writing each one.
_RESOURCE_BLURB_TEMPLATES = {
    "cpu": "{pattern} CPU load: drives compute pressure a CPU-aware autoscaler sees directly.",
    "memory": "{pattern} memory work: a CPU-only autoscaler barely sees this load; memory-aware scaling matters here.",
    "mixed": "{pattern} mixed CPU+memory load: stresses both signals together, the hardest case to scale well.",
}


def _combo_description(pattern_label: str, resource: str) -> str:
    template = _RESOURCE_BLURB_TEMPLATES.get(resource, "{pattern} load on the demo workload.")
    return template.format(pattern=pattern_label)


@st.fragment(run_every=3)
def _render_live_status_panel(real_ctrl: RealExperimentControllerV2) -> None:
    """Periodically re-executes on its own (every 3s) WITHOUT rerunning the
    rest of the page -- `render_controls_v2()` (and its START/STOP click
    handling) lives outside this fragment, so these refreshes can never
    re-invoke `.start()` or launch a duplicate subprocess. This is what
    keeps status/log output moving while a run is active instead of being
    frozen until the user happens to touch some other widget.

    Surfaces a `"refused"` status (the training/evaluation-protection
    check inside `scripts/run_demo_episode_v2.py` failing closed) distinctly."""
    real_status = real_ctrl.status()
    status_label = real_status.get("status", "unknown")
    display_label = "starting" if (real_ctrl.is_running and status_label == "unknown") else status_label

    rec = experiment_store.get_run(real_ctrl.run_id) if real_ctrl.run_id else None
    if rec is not None:
        profile_label = rec["profile_name"].replace("_", " ").title()
        st.markdown(
            f'<div class="mono" style="font-size:12px;color:var(--muted);">'
            f'Current run: Live cluster &middot; {display_arm_label(rec["controller"])} &middot; {profile_label} &middot; '
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


def _render_traffic_preview(profile_name: str, pattern: str, resource: str, load: str) -> None:
    """Preview of the currently selected demo traffic combination -- a
    request-rate-over-time chart computed from the ACTUAL generated
    profile file (never a hand-drawn approximation), plus a one-line
    description of what the combination exercises. Shown regardless of
    data source, and regardless of whether a run is active, so changing
    the selects always previews what pressing START will actually send."""
    st.markdown('<div class="du-section-label" style="margin-top:0.6rem;">Traffic Preview</div>', unsafe_allow_html=True)
    try:
        timestamps, rps = compute_rps_curve(profile_name)
    except Exception:  # noqa: BLE001 -- preview must never crash the page
        st.caption(f"Preview unavailable for profile {profile_name!r}.")
        return

    st.plotly_chart(
        line_chart(f"Request rate over time -- {profile_name}", timestamps, {"request_rate_rps": rps}, "req/s"),
        width="stretch",
    )
    pattern_label = PATTERN_LABELS.get(pattern, pattern)
    st.caption(_combo_description(pattern_label, resource))
    if load == "high":
        st.caption("High load: up to 70 req/s, like the stress test")


def render() -> None:
    if "du_v2_real_controller_impl" not in st.session_state:
        st.session_state.du_v2_real_controller_impl = RealExperimentControllerV2()

    prior_data_source = st.session_state.get("du_data_source", "Live cluster")
    render_topbar(mode_label=prior_data_source, is_live=(prior_data_source == "Live cluster"))

    # A config error (missing/malformed demo_ui/config_v2.yaml) must never
    # crash this page -- Sample data (offline) still works either way; only
    # Live cluster is forced off, with the error itself as the
    # disabled-reason message.
    cfg, config_error = load_config_v2_safe()
    live_enabled = cfg.live_enabled if cfg is not None else False
    live_disabled_reason = cfg.live_disabled_reason if cfg is not None else config_error
    model_path = cfg.model_path if cfg is not None else None

    st.markdown('<div class="du-section-label" style="margin-top:0.6rem;">Data Source</div>', unsafe_allow_html=True)
    data_source = render_data_source_toggle()
    is_live = data_source == "Live cluster"

    active_rec = experiment_store.active_run() if is_live else None
    if active_rec is not None and active_rec.get("generation") != "v2":
        active_rec = None  # a run from a different generation must never disable these controls

    arm, profile_name, pattern, resource, load, start_clicked, stop_clicked = render_controls_v2(
        live_enabled, live_disabled_reason, is_live, active_rec,
    )

    _render_traffic_preview(profile_name, pattern, resource, load)

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
        _render_live_status_panel(real_ctrl)
        return

    if start_clicked:
        # Sample data (offline) loading is instantaneous (no subprocess) --
        # `du_v2_run_in_progress` still flips to True so the control strip
        # shows RUNNING/STOP, consistent with the LIVE path, with STOP
        # acting as "clear".
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
            f"SIMULATED DATA loaded: {display_arm_label(arm)} / {profile_label} — see Live Details for the full view. "
            f"DEMO ONLY, not evaluation evidence."
        )
    else:
        st.caption("No demo run loaded. Configure the controls above and press START.")
