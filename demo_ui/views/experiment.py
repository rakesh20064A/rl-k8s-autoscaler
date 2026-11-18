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
from scripts.v2_live_cluster_state import ensure_canonical_idle_state, no_runner_process_alive

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


def _render_live_status_panel(real_ctrl: RealExperimentControllerV2) -> None:
    """Called (never decorated directly -- see `render()`) via
    `st.fragment(run_every=...)` applied FUNCTIONALLY at the call site, so
    the interval can be `None` (no periodic reruns at all) once no run is
    active -- task sec.3's "stop polling once the run reaches a terminal
    state". While a run IS active, `render()` passes `run_every=3`: this
    fragment then re-executes every ~3s WITHOUT rerunning the rest of the
    page -- `render_controls_v2()` (and its START/STOP click handling)
    lives outside this fragment, so these refreshes can never re-invoke
    `.start()` or launch a duplicate subprocess. This is what keeps
    status/log output moving while a run is active instead of being
    frozen until the user happens to touch some other widget.

    RE-ATTACH (task sec.2/4): every tick re-derives the active run from
    `experiment_store.reconcile()` -- never from `real_ctrl`'s own
    in-memory state, which is empty/stale right after a Streamlit restart.
    If the run this `real_ctrl` is (or should be) tracking differs from
    what the store says is active, `real_ctrl.attach()` re-hydrates it
    first. If no run is active, `real_ctrl` falls back to whatever it was
    last tracking, then to the store's own latest run, so the panel still
    shows a finished run's real outcome (COMPLETED, STOPPED, ...) instead
    of reverting to "UNKNOWN" the moment it stops being active.

    Surfaces a `"refused"` status (the training/evaluation-protection
    check inside `scripts/run_demo_episode_v2.py` failing closed) distinctly."""
    reconciled = experiment_store.reconcile()
    active_rec_now = reconciled["active"]

    rec = active_rec_now
    if rec is None and real_ctrl.run_id is not None:
        rec = experiment_store.get_run(real_ctrl.run_id)
    if rec is None:
        latest = experiment_store.latest_run()
        rec = latest if (latest is not None and latest.get("generation") == "v2") else None

    if rec is not None and real_ctrl.run_id != rec["run_id"]:
        real_ctrl.attach(rec)
    real_ctrl.is_running = active_rec_now is not None

    if active_rec_now is not None:
        real_ctrl.status()  # polls the live status file -- the one thing that keeps the store's record fresh while a run is active
        rec = experiment_store.get_run(real_ctrl.run_id)
    elif reconciled["newly_stale"]:
        # Task sec.4: a run was just found dead with no terminal status of
        # its own -- if nothing is left running anywhere, make sure the
        # live cluster did not get left mid-run.
        cleanup = ensure_canonical_idle_state()
        if cleanup.get("ran"):
            st.info(f"Recovered the live cluster to its idle state after an interrupted run: {cleanup}.")

    status_label = rec.get("status", "unknown") if rec is not None else "unknown"
    display_label = (rec.get("stage") or status_label) if rec is not None else "unknown"

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
        st.error(f"Refused to start: {(rec or {}).get('error', 'unknown reason')}")
    elif status_label == "error":
        st.error((rec or {}).get("error", "unknown error"))
    elif status_label == "completed":
        st.success("Demo episode completed. DEMO ONLY -- not evaluation evidence.")
    elif status_label == "stopped":
        st.info("Run stopped by user.")
    elif status_label == experiment_store.STALE_STATUS:
        st.warning((rec or {}).get("error") or "Run was interrupted.")
    elif status_label == experiment_store.INVALID_NO_TRAFFIC:
        st.warning((rec or {}).get("error") or "Run invalidated: no traffic reached the app during this run.")

    # Task sec.5: the REAL root cause of "STOP/selects stuck after
    # COMPLETED" -- `reconcile()` at the top of this function is almost
    # always the FIRST thing to discover a status-file transition to
    # terminal (it runs before `real_ctrl.status()` below ever gets a
    # chance to), which means `active_rec_now` is ALREADY `None` and line
    # ~80 above has ALREADY set `real_ctrl.is_running = False` by the time
    # this point is reached -- in the SAME tick. Gating the one-shot
    # full-page `st.rerun()` on `real_ctrl.is_running` (its PRE-prior-line
    # value, which this tick already overwrote) therefore almost never
    # fires: the outer `render()`'s own `active_rec` (computed once per
    # full-page rerun, BEFORE this fragment runs) stays stuck on the
    # last-seen non-terminal value forever, which is exactly why the
    # control strip kept showing STOP with all four selects still locked
    # even though this panel already displayed COMPLETED/FAILED/etc.
    # correctly. The fix derives the one-shot trigger from the STORE
    # (`rec["run_id"]`/`status_label`) instead of that stale in-memory
    # flag: a run_id+status pair is synced (full-page rerun fired) EXACTLY
    # ONCE -- the session_state key here is a pure debounce guard against
    # looping, never a competing source of truth for whether controls are
    # locked (that is still `active_rec is not None`, re-derived from the
    # store on every rerun in `render()`/`controls_v2.render_controls_v2`).
    if rec is not None and status_label in experiment_store.TERMINAL_STATUSES:
        sync_key = f"{rec['run_id']}:{status_label}"
        if st.session_state.get("du_v2_last_synced_terminal") != sync_key:
            st.session_state["du_v2_last_synced_terminal"] = sync_key
            real_ctrl.is_running = False
            st.session_state["du_v2_run_in_progress"] = False  # never read for the LIVE button anymore (see controls_v2.py), cleared so a later DEMO-mode run starts from a clean slate
            # Default (app) scope: the WHOLE page reruns, not just this
            # fragment -- `render()` recomputes `active_rec` fresh from the
            # now-terminal store record (None) on that rerun, so the control
            # strip's START/STOP button and this fragment's own `run_every`
            # (task sec.3) both pick up the terminal state on this one
            # rerun, with no further periodic ticking needed afterward.
            st.rerun()

    # Task sec.2: last ~30 lines, read fresh from the log file every tick
    # above (never only once the run has finished) -- `recent_log_lines()`
    # re-reads the file from disk each call, so this is a genuine tail,
    # not a one-time snapshot.
    with st.expander("Recent log output", expanded=real_ctrl.is_running):
        lines = real_ctrl.recent_log_lines(n=30)
        if not lines:
            st.caption("No output yet." if real_ctrl.is_running else "No output.")
        for line in lines:
            st.markdown(f'<div class="mono" style="font-size:11px;">{line}</div>', unsafe_allow_html=True)


def _render_recover_control() -> None:
    """RECOVER (task sec.6): enabled only when no `run_demo_episode_v2.py`
    process is alive anywhere on this host -- runs the SAME reconciliation
    + canonical-cleanup logic the terminal `scripts/demo_recover.py` runs,
    and reports exactly what it did/found. Never touches the cluster while
    a runner might still be mid-cleanup itself."""
    runner_alive = not no_runner_process_alive()
    if st.button("RECOVER", key="du_v2_recover_button", disabled=runner_alive,
                 help="Reconcile run history and restore the idle cluster state. Disabled while a run is active."):
        reconciled = experiment_store.reconcile()
        cleanup = ensure_canonical_idle_state()
        st.session_state["du_v2_last_recover_result"] = {
            "newly_terminal": [r["run_id"] for r in reconciled["newly_terminal"]],
            "newly_stale": [r["run_id"] for r in reconciled["newly_stale"]],
            "cleanup": cleanup,
        }
        st.rerun()
    result = st.session_state.get("du_v2_last_recover_result")
    if result is not None:
        st.caption(f"Last RECOVER result: {result}")


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
        # RE-ATTACH before handling STOP (task sec.2/5): a fresh
        # `real_ctrl` (e.g. right after a Streamlit restart) has no
        # `run_id` of its own yet. Without this, clicking STOP on a run
        # this session never itself started would silently no-op --
        # `stop()` has nothing to signal without a persisted record to
        # resolve a pid/pgid from. `_render_live_status_panel()` below
        # repeats this attach on every 3s tick, but STOP must see it on
        # THIS exact rerun, synchronously, before `stop()` is called.
        if active_rec is not None and real_ctrl.run_id != active_rec["run_id"]:
            real_ctrl.attach(active_rec)
        if start_clicked and live_enabled:
            st.session_state["du_v2_run_in_progress"] = True
            real_ctrl.start(arm, profile_name, model_path)
            st.rerun()
        if stop_clicked:
            st.session_state["du_v2_run_in_progress"] = False
            real_ctrl.stop()
            st.rerun()
        # Task sec.3: `run_every` is computed HERE, fresh on every full-app
        # rerun, from the SAME store-backed `active_rec` signal the control
        # strip's START/STOP button now derives from (never a session_state
        # flag) -- polling runs every ~3s while a run is active and stops
        # entirely (no fragment reruns at all) once it is not, applying
        # `st.fragment(...)` functionally rather than as a fixed decorator
        # so this interval can change between reruns.
        st.fragment(run_every=3 if active_rec is not None else None)(_render_live_status_panel)(real_ctrl)
        _render_recover_control()
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
