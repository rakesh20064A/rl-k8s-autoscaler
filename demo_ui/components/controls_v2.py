"""components/controls_v2.py -- V2 (multi-metric autoscaler) experiment
control strip, parallel to controls.py's V1 strip but simpler (no Auto
mode -- the task scope for V2 is a single-arm demo run).

Same session-state discipline as controls.py: a durable public key
(`du_v2_selected_arm`/`du_v2_selected_profile`), seeded via `setdefault`
only if missing, each widget using its own private key.
"""

from __future__ import annotations

from typing import Optional, Tuple

import streamlit as st

ARM_LABELS = {"ppo_v2": "PPO-V2", "hpa_v2_cpu_memory": "HPA-V2 (CPU+memory)"}
PROFILE_LABELS = {
    "demo_cpu_short": "CPU (short)", "demo_memory_short": "Memory (short)", "demo_mixed_short": "Mixed (short)",
}
_LABEL_TO_ARM = {v: k for k, v in ARM_LABELS.items()}
_LABEL_TO_PROFILE = {v: k for k, v in PROFILE_LABELS.items()}


def render_controls_v2(
    live_enabled: bool, live_disabled_reason: Optional[str], is_live: bool, active_rec: Optional[dict] = None,
) -> Tuple[str, str, bool, bool]:
    """Returns (arm, profile_name, start_clicked, stop_clicked).

    `live_enabled=False` disables START while `is_live=True` and shows
    `live_disabled_reason` -- DEMO mode (is_live=False) is never affected."""
    already_running = st.session_state.get("du_v2_run_in_progress", False)
    controls_disabled = active_rec is not None

    st.session_state.setdefault("du_v2_selected_arm", "ppo_v2")
    st.session_state.setdefault("du_v2_selected_profile", "demo_cpu_short")

    st.markdown('<div class="du-control-strip">', unsafe_allow_html=True)
    c_arm, c_profile, c_status, c_action = st.columns([2.4, 2.2, 1.6, 1.4])

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

    with c_profile:
        st.markdown('<div class="du-section-label">Demo Traffic</div>', unsafe_allow_html=True)
        choices = list(PROFILE_LABELS.values())
        current_raw_profile = active_rec["profile_name"] if controls_disabled else st.session_state["du_v2_selected_profile"]
        current_profile_label = PROFILE_LABELS.get(current_raw_profile, choices[0])
        if controls_disabled:
            st.session_state["du_v2_profile_widget"] = current_profile_label
        profile_choice = st.selectbox(
            "Demo Traffic Profile", choices, index=choices.index(current_profile_label),
            key="du_v2_profile_widget", label_visibility="collapsed", disabled=controls_disabled,
        )
        effective_profile_label = current_profile_label if controls_disabled else profile_choice
        if not controls_disabled:
            st.session_state["du_v2_selected_profile"] = _LABEL_TO_PROFILE[profile_choice]
        profile_name = _LABEL_TO_PROFILE[effective_profile_label]

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

    return arm, profile_name, start_clicked, stop_clicked
