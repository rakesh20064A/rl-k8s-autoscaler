"""components/controls.py -- Horizontal experiment-control strip (sec.5-7,
9, 19), replacing a stack of default Streamlit widgets with one compact row:
CONTROLLER / MODE / TRAFFIC / STATUS / START-STOP.

Buttons here call ONLY an ExperimentController implementation passed in by
the caller (FixtureExperimentController or RealExperimentController). A
`du_run_in_progress` session-state flag guards against Streamlit reruns
re-triggering `start()` a second time for the same click (sec.19).

STATE PERSISTENCE: every selection widget here (Controller, Mode, Traffic)
uses the SAME decoupled pattern as `render_data_source_toggle()` below --
a plain, non-widget-bound "public" session_state key is the durable source
of truth (`du_selected_controller`/`du_selected_mode`/`du_selected_profile`),
seeded via `setdefault` ONLY if missing, and each widget gets its own
PRIVATE key purely for rendering. Binding a widget's `key` directly to the
public value was the actual bug reported: Streamlit clears a widget-bound
key whenever that exact widget call is skipped on a script run (true of
every OTHER page, since this module is only called from Experiment), which
silently reset Controller/Traffic/Mode to their first option on navigation.

ACTIVE-RUN AUTHORITY: while `active_rec` (an `experiment_store` record) is
passed in, its `controller`/`profile_name` override whatever is in the
public keys and the Controller/Traffic widgets render `disabled=True` --
the user can never visually drift the display away from the real
executing configuration, and a missing/reset session_state can never
invent a wrong default on top of a genuinely active process (recovery
always prefers the run store over local widget state).

Dependency note: requires `streamlit` (install via demo_ui/requirements-ui.txt
into a SEPARATE venv -- never the project's scientific .venv).
"""

from __future__ import annotations

from typing import Optional, Tuple

import streamlit as st

from demo_ui.fixtures import TRAFFIC_PROFILE_DESCRIPTIONS
from demo_ui.models import Controller, TrafficProfile
from demo_ui.providers import ExperimentController

# Demo-only traffic options shown ONLY when is_live=True (never in DEMO
# mode, never part of the scientific TrafficProfile enum/contracts). Label
# -> the internal profile name scripts/run_demo_episode.py resolves outside
# the training-eligibility firewall (see its DEMO_ONLY_PROFILES).
_LIVE_ONLY_PROFILE_LABELS = {"Smoke Test (40s)": "demo_smoke_test"}

# label <-> raw profile-name string, both directions, covering the six
# scientific TrafficProfile values and the demo-only extra.
_LABEL_TO_RAW = {p.value.capitalize(): p.value for p in TrafficProfile}
_LABEL_TO_RAW.update(_LIVE_ONLY_PROFILE_LABELS)
_RAW_TO_LABEL = {raw: label for label, raw in _LABEL_TO_RAW.items()}


def render_data_source_toggle() -> str:
    """Returns 'DEMO' or 'LIVE'. Never silently mixes the two -- the caller
    must branch on this return value and render an unambiguous banner.

    Uses `st.segmented_control` rather than `st.radio`: the radiogroup's
    option text proved unreliable to keep visible across browsers (Safari)
    even after broadening theme.py's forced-color CSS, so this control
    deliberately avoids that DOM/CSS path rather than patching it further.

    `st.session_state["du_data_source"]` -- the public contract this whole
    app reads -- is deliberately NOT used as the widget's own `key`.
    Streamlit clears a widget-bound key's value whenever that exact widget
    call is skipped on a script run (confirmed directly: navigating to
    another page -- whose `render()` never calls this function -- leaves
    `du_data_source` as None on return, reverting the segmented control to
    its first option). Since `app.py` only calls this function from the
    Experiment page, a key bound 1:1 to the widget call is cleared on every
    OTHER page's rerun. The widget instead gets its own private key
    (`du_data_source_widget`), seeded each render from the public value,
    and its result is written back into `du_data_source` explicitly -- a
    plain dict entry Streamlit's widget-lifecycle cleanup never touches, so
    it survives navigating away and back."""
    st.session_state.setdefault("du_data_source", "DEMO")
    value = st.segmented_control(
        "Data source", ["DEMO", "LIVE"], default=st.session_state["du_data_source"],
        selection_mode="single", required=True,
        key="du_data_source_widget", label_visibility="collapsed",
    )
    st.session_state["du_data_source"] = value
    return value


def render_controls(
    controller_impl: ExperimentController, is_live: bool = False, active_rec: Optional[dict] = None,
) -> Tuple[Optional[Controller], bool, Optional[TrafficProfile], bool, bool]:
    """Renders the horizontal control strip and returns the CURRENT widget
    state plus whether START/STOP were just clicked this render:

        (selected_controller_or_None, auto_on, traffic_profile, start_clicked, stop_clicked)

    `selected_controller_or_None` is None when auto_on is True (Auto mode
    selects its own HPA-then-PPO sequence; sec.6). `traffic_profile` is
    None only when `is_live` and a demo-only option (`_LIVE_ONLY_PROFILE_LABELS`)
    was picked -- never in DEMO mode, where the dropdown never offers one.

    `active_rec`: pass the current `experiment_store.active_run()` record
    when one exists (LIVE mode only) -- its controller/profile become the
    displayed, non-editable truth for this render (see module docstring)."""
    already_running = st.session_state.get("du_run_in_progress", False)
    controls_disabled = active_rec is not None

    st.session_state.setdefault("du_selected_mode", "Manual")
    st.session_state.setdefault("du_selected_controller", Controller.HPA.value)
    st.session_state.setdefault("du_selected_profile", TrafficProfile.CONSTANT.value)

    st.markdown('<div class="du-control-strip">', unsafe_allow_html=True)
    c_controller, c_mode, c_traffic, c_status, c_action = st.columns([2, 2, 2.2, 1.6, 1.4])

    # Mode is resolved ahead of the Controller column so that column renders
    # correctly on the FIRST pass too (Streamlit evaluates columns top-to-
    # bottom in source order). An active run is always Manual (Auto isn't
    # wired to live execution), so it forces Manual outright.
    auto_on = (not controls_disabled) and st.session_state["du_selected_mode"] == "Auto"

    with c_controller:
        if auto_on:
            st.markdown('<span class="du-badge">Managed by Auto</span>', unsafe_allow_html=True)
            selected_controller = None
        else:
            options = [Controller.HPA.value, Controller.PPO.value]
            current = active_rec["controller"] if controls_disabled else st.session_state["du_selected_controller"]
            if current not in options:
                current = options[0]
            if controls_disabled:
                # `index=` alone is NOT enough here: once a widget's key
                # already holds a value from an earlier render (e.g. "HPA"
                # from before this run started), Streamlit honors the
                # EXISTING session_state[key] on every subsequent rerun and
                # silently ignores a new `index=` -- `index`/`value`/`default`
                # only take effect the very first time a key is created.
                # Pinning the key directly is what actually forces the
                # display to the active run's real controller.
                st.session_state["du_controller_widget"] = current
            choice = st.radio(
                "Controller", options, index=options.index(current),
                horizontal=True, key="du_controller_widget", label_visibility="collapsed",
                disabled=controls_disabled,
            )
            effective = current if controls_disabled else choice
            if not controls_disabled:
                st.session_state["du_selected_controller"] = choice
            selected_controller = Controller(effective)

    with c_mode:
        st.markdown('<div class="du-section-label">Mode</div>', unsafe_allow_html=True)
        mode_options = ["Manual", "Auto"]
        current_mode = "Manual" if controls_disabled else st.session_state["du_selected_mode"]
        mode_choice = st.radio(
            "Mode", mode_options, horizontal=True, key="du_mode_widget", label_visibility="collapsed",
            index=mode_options.index(current_mode), disabled=controls_disabled,
        )
        if not controls_disabled:
            st.session_state["du_selected_mode"] = mode_choice
            auto_on = mode_choice == "Auto"

    with c_traffic:
        st.markdown('<div class="du-section-label">Traffic</div>', unsafe_allow_html=True)
        choices = [p.value.capitalize() for p in TrafficProfile]
        if is_live:
            choices = choices + list(_LIVE_ONLY_PROFILE_LABELS)
        current_raw = active_rec["profile_name"] if controls_disabled else st.session_state["du_selected_profile"]
        current_label = _RAW_TO_LABEL.get(current_raw, choices[0])
        if current_label not in choices:
            # e.g. a demo-only profile was persisted but is_live is now False
            current_label = choices[0]
        if controls_disabled:
            # Same reasoning as the Controller widget above: pin the key
            # directly, `index=` is ignored once the key already has a
            # cached value from before this run started.
            st.session_state["du_profile_widget"] = current_label
        profile_choice = st.selectbox(
            "Traffic Profile", choices, index=choices.index(current_label),
            key="du_profile_widget", label_visibility="collapsed", disabled=controls_disabled,
        )
        effective_label = current_label if controls_disabled else profile_choice
        if not controls_disabled:
            st.session_state["du_selected_profile"] = _LABEL_TO_RAW[profile_choice]
        if effective_label in _LIVE_ONLY_PROFILE_LABELS:
            traffic_profile = None
            demo_only_profile_name = _LIVE_ONLY_PROFILE_LABELS[effective_label]
            profile_description = "Demo-only 15s smoke/connectivity check -- not a scientific traffic profile."
        else:
            traffic_profile = TrafficProfile(effective_label.lower())
            demo_only_profile_name = None
            profile_description = TRAFFIC_PROFILE_DESCRIPTIONS[traffic_profile]

    with c_status:
        st.markdown('<div class="du-section-label">Status</div>', unsafe_allow_html=True)
        status_label = "RUNNING" if already_running else "READY"
        status_class = "status-warn" if already_running else "status-ok"
        st.markdown(f'<span class="du-badge {status_class}">{status_label}</span>', unsafe_allow_html=True)

    with c_action:
        st.markdown('<div class="du-section-label">&nbsp;</div>', unsafe_allow_html=True)
        # Explicit, stable keys: this code slot alternates between two
        # different button labels (START/STOP) depending on `already_running`,
        # so each needs its own unambiguous identity across reruns.
        if already_running:
            stop_clicked = st.button("STOP", use_container_width=True, key="du_stop_button")
            start_clicked = False
        else:
            start_clicked = st.button("START", use_container_width=True, key="du_start_button")
            stop_clicked = False

    st.markdown(
        f'<div style="font-size:12px;color:var(--muted);margin:0.3rem 0 0.5rem 0;">'
        f'{profile_description}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if start_clicked and not already_running:
        st.session_state["du_run_in_progress"] = True
        if hasattr(controller_impl, "set_profile_override"):
            controller_impl.set_profile_override(demo_only_profile_name)
        # Captures the CURRENTLY SELECTED configuration exactly once, at
        # the moment of this click -- `controller_impl.start()` mints the
        # run_id/launches the subprocess with these exact values, and every
        # later rerun reflects that run via `active_rec` (above), never by
        # re-reading these (possibly since-changed) widgets again.
        controller_impl.start(selected_controller or Controller.HPA, traffic_profile or TrafficProfile.CONSTANT)
        # Root cause of "needs two clicks": `already_running` above was read
        # (and the START button drawn) BEFORE this line flips
        # du_run_in_progress to True, so without forcing another rerun right
        # now, the control strip keeps showing "START" -- looking exactly
        # like the click did nothing -- until some UNRELATED later
        # interaction happens to rerun the page. The subprocess is already
        # launched at this point; this rerun only makes the UI reflect it
        # immediately, on the same click, rather than needing a second one.
        st.rerun()
    if stop_clicked:
        st.session_state["du_run_in_progress"] = False
        controller_impl.stop()
        st.rerun()

    return selected_controller, auto_on, traffic_profile, start_clicked, stop_clicked
