"""components/controls_v2.py -- V2 (multi-metric autoscaler) experiment
control strip, parallel to controls.py's V1 strip but simpler (no Auto
mode -- the task scope for V2 is a single-arm demo run).

Same session-state discipline as controls.py: a durable public key per
selectable field (`du_v2_selected_arm`/`du_v2_selected_pattern`/
`du_v2_selected_resource`/`du_v2_selected_load`), seeded via `setdefault`
only if missing, each widget using its own private key.

Demo traffic is chosen via 3 independent selects -- pattern, resource
type, load level -- whose combination picks one generated profile file
from `scripts/generate_demo_profiles.py`'s 36-profile catalog (plus the
3 legacy `demo_*_short` profiles, reachable only via an already-active
run's own `profile_name`, never newly selectable here)."""

from __future__ import annotations

from typing import Optional, Tuple

import streamlit as st

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
    parse_profile_name,
    profile_slug,
)

ARM_LABELS = {"ppo_v2": "PPO", "hpa_v2_cpu_memory": "HPA (CPU+memory)"}
_LABEL_TO_ARM = {v: k for k, v in ARM_LABELS.items()}

_LABEL_TO_PATTERN = {v: k for k, v in PATTERN_LABELS.items()}
_LABEL_TO_RESOURCE = {v: k for k, v in RESOURCE_LABELS.items()}
_LABEL_TO_LOAD = {v: k for k, v in LOAD_LABELS.items()}


def display_arm_label(raw: str) -> str:
    """Maps an internal arm identifier to a clean, display-ready label --
    the ONE place that formatting happens, so every page shows the same
    thing for the same arm. Falls back to a generic cleanup for any arm
    not in `ARM_LABELS` rather than ever showing the raw internal code."""
    if raw in ARM_LABELS:
        return ARM_LABELS[raw]
    return raw.replace("_v2", "").replace("_", " ").strip().upper()


def render_controls_v2(
    live_enabled: bool, live_disabled_reason: Optional[str], is_live: bool, active_rec: Optional[dict] = None,
) -> Tuple[str, str, str, str, str, bool, bool]:
    """Returns (arm, profile_name, pattern, resource, load, start_clicked,
    stop_clicked).

    `live_enabled=False` disables START while `is_live=True` and shows
    `live_disabled_reason` -- DEMO mode (is_live=False) is never affected."""
    already_running = st.session_state.get("du_v2_run_in_progress", False)
    controls_disabled = active_rec is not None

    st.session_state.setdefault("du_v2_selected_arm", "ppo_v2")
    st.session_state.setdefault("du_v2_selected_pattern", DEFAULT_PATTERN)
    st.session_state.setdefault("du_v2_selected_resource", DEFAULT_RESOURCE)
    st.session_state.setdefault("du_v2_selected_load", DEFAULT_LOAD)

    st.markdown('<div class="du-control-strip">', unsafe_allow_html=True)
    c_arm, c_pattern, c_resource, c_load, c_status, c_action = st.columns(
        [1.6, 1.7, 1.3, 1.1, 1.3, 1.3]
    )

    with c_arm:
        st.markdown('<div class="du-section-label">Arm</div>', unsafe_allow_html=True)
        options = list(ARM_LABELS.values())
        current_raw = active_rec["controller"] if controls_disabled else st.session_state["du_v2_selected_arm"]
        current_label = ARM_LABELS.get(current_raw, options[0])
        if controls_disabled:
            st.session_state["du_v2_arm_widget"] = current_label
        choice = st.radio(
            "Arm", options, index=options.index(current_label), horizontal=True,
            key="du_v2_arm_widget", label_visibility="collapsed", disabled=controls_disabled,
        )
        effective_label = current_label if controls_disabled else choice
        if not controls_disabled:
            st.session_state["du_v2_selected_arm"] = _LABEL_TO_ARM[choice]
        arm = _LABEL_TO_ARM[effective_label]

    # When an active run is locked in, parse its own profile_name back
    # into (pattern, resource, load) for display; a legacy `demo_*_short`
    # profile has no such decomposition, so each select falls back to its
    # own default rather than crashing.
    if controls_disabled:
        parsed = parse_profile_name(active_rec["profile_name"])
        locked_pattern, locked_resource, locked_load = parsed if parsed is not None else (
            DEFAULT_PATTERN, DEFAULT_RESOURCE, DEFAULT_LOAD,
        )
    else:
        locked_pattern = locked_resource = locked_load = None

    with c_pattern:
        st.markdown('<div class="du-section-label">Traffic pattern</div>', unsafe_allow_html=True)
        choices = [PATTERN_LABELS[p] for p in PATTERNS]
        current_raw_pattern = locked_pattern if controls_disabled else st.session_state["du_v2_selected_pattern"]
        current_label = PATTERN_LABELS.get(current_raw_pattern, choices[0])
        if controls_disabled:
            st.session_state["du_v2_pattern_widget"] = current_label
        choice = st.selectbox(
            "Traffic pattern", choices, index=choices.index(current_label),
            key="du_v2_pattern_widget", label_visibility="collapsed", disabled=controls_disabled,
        )
        effective_label = current_label if controls_disabled else choice
        if not controls_disabled:
            st.session_state["du_v2_selected_pattern"] = _LABEL_TO_PATTERN[choice]
        pattern = _LABEL_TO_PATTERN[effective_label]

    with c_resource:
        st.markdown('<div class="du-section-label">Resource type</div>', unsafe_allow_html=True)
        choices = [RESOURCE_LABELS[r] for r in RESOURCES]
        current_raw_resource = locked_resource if controls_disabled else st.session_state["du_v2_selected_resource"]
        current_label = RESOURCE_LABELS.get(current_raw_resource, choices[0])
        if controls_disabled:
            st.session_state["du_v2_resource_widget"] = current_label
        choice = st.selectbox(
            "Resource type", choices, index=choices.index(current_label),
            key="du_v2_resource_widget", label_visibility="collapsed", disabled=controls_disabled,
        )
        effective_label = current_label if controls_disabled else choice
        if not controls_disabled:
            st.session_state["du_v2_selected_resource"] = _LABEL_TO_RESOURCE[choice]
        resource = _LABEL_TO_RESOURCE[effective_label]

    with c_load:
        st.markdown('<div class="du-section-label">Load level</div>', unsafe_allow_html=True)
        choices = [LOAD_LABELS[lvl] for lvl in LOADS]
        current_raw_load = locked_load if controls_disabled else st.session_state["du_v2_selected_load"]
        current_label = LOAD_LABELS.get(current_raw_load, choices[0])
        if controls_disabled:
            st.session_state["du_v2_load_widget"] = current_label
        choice = st.selectbox(
            "Load level", choices, index=choices.index(current_label),
            key="du_v2_load_widget", label_visibility="collapsed", disabled=controls_disabled,
        )
        effective_label = current_label if controls_disabled else choice
        if not controls_disabled:
            st.session_state["du_v2_selected_load"] = _LABEL_TO_LOAD[choice]
        load = _LABEL_TO_LOAD[effective_label]

    profile_name = active_rec["profile_name"] if controls_disabled else profile_slug(pattern, resource, load)

    with c_status:
        st.markdown('<div class="du-section-label">Status</div>', unsafe_allow_html=True)
        status_label = "RUNNING" if already_running else "READY"
        status_class = "status-warn" if already_running else "status-ok"
        st.markdown(f'<span class="du-badge {status_class}">{status_label}</span>', unsafe_allow_html=True)

    with c_action:
        st.markdown('<div class="du-section-label">&nbsp;</div>', unsafe_allow_html=True)
        start_disabled = is_live and not live_enabled
        if already_running:
            stop_clicked = st.button("STOP", width="stretch", key="du_v2_stop_button")
            start_clicked = False
        else:
            start_clicked = st.button("START", width="stretch", key="du_v2_start_button", disabled=start_disabled)
            stop_clicked = False

    st.markdown(
        '<div style="font-size:12px;color:var(--muted);margin:0.3rem 0 0.2rem 0;">'
        'DEMO ONLY -- not evaluation evidence. ~8 minute run.</div>',
        unsafe_allow_html=True,
    )
    if is_live and not live_enabled:
        st.markdown(
            f'<div style="font-size:12px;color:var(--muted);margin:0 0 0.5rem 0;">{live_disabled_reason}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    return arm, profile_name, pattern, resource, load, start_clicked, stop_clicked
