"""components/pods.py -- Compact pod status list (sec.10): infrastructure
objects, not decorative cards. No animation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import streamlit as st

from demo_ui.models import PodState, PodStatus

_STATE_DOT = {
    PodState.READY: ("●", "ready"),
    PodState.STARTING: ("◐", "starting"),
    PodState.TERMINATING: ("◐", "terminating"),
    PodState.FAILED: ("×", "failed"),
}


def render_pod_panel(pods: List[PodStatus], deployment_name: str = "demo-app", restarts: Optional[Dict[str, int]] = None) -> None:
    st.markdown(f'<div class="du-section-label">Deployment {deployment_name}</div>', unsafe_allow_html=True)
    if not pods:
        st.markdown('<div class="du-panel mono" style="font-size:12.5px;">No pods reported.</div>', unsafe_allow_html=True)
        return

    restarts = restarts or {}
    rows = []
    for pod in pods:
        dot, cls = _STATE_DOT[pod.state]
        short_name = pod.name.split("-")[-1]
        restart_count = restarts.get(pod.name)
        restart_label = f"{restart_count} restarts" if restart_count is not None else ""
        rows.append(
            f'<div class="du-pod-row">'
            f'<span><span class="du-pod-dot {cls}">{dot}</span>{short_name}</span>'
            f'<span>{pod.state.value}</span>'
            f'<span>{restart_label}</span>'
            f'</div>'
        )
    st.markdown(f'<div class="du-panel">{"".join(rows)}</div>', unsafe_allow_html=True)
