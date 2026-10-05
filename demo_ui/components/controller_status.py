"""components/controller_status.py -- One compact instrument panel per
controller (sec.9). Visualization/data-model only; does not read live PPO
internals (the project's training code is never modified to expose these
fields during training).
"""

from __future__ import annotations

import streamlit as st

from demo_ui.models import HpaSnapshot, PpoSnapshot


def render_hpa_panel(hpa: HpaSnapshot) -> None:
    st.markdown(
        f"""
        <div class="du-panel">
          <div class="du-section-label">HPA Controller</div>
          <div style="display:flex;justify-content:space-between;margin-top:0.3rem;">
            <div>
              <div class="du-metric-label">CPU</div>
              <div class="mono" style="font-size:18px;font-weight:700;">{hpa.current_cpu_pct:.0f} / {hpa.cpu_target_pct:.0f}%</div>
            </div>
            <div>
              <div class="du-metric-label">Replicas</div>
              <div class="mono" style="font-size:18px;font-weight:700;">{hpa.current_replicas} → {hpa.desired_replicas}</div>
            </div>
          </div>
          <div class="mono" style="font-size:12px;color:var(--muted);margin-top:0.4rem;">
            MIN {hpa.min_replicas} &nbsp;&middot;&nbsp; MAX {hpa.max_replicas}
          </div>
          <div class="mono" style="font-size:12px;color:var(--muted);margin-top:0.2rem;">
            Last scaling event: {hpa.latest_scaling_event or '–'}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ppo_panel(ppo: PpoSnapshot) -> None:
    latest_action = ppo.latest_action.value.replace("_", " ") if ppo.latest_action else "–"
    last = ppo.decisions[-1] if ppo.decisions else None
    transition = f"{last.replicas_before} → {last.replicas_after}" if last else "–"
    reward = f"{last.reward:+.2f}" if last else "–"
    ts = f"{last.timestamp:.0f}s" if last else "–"

    st.markdown(
        f"""
        <div class="du-panel">
          <div class="du-section-label">PPO Controller</div>
          <div class="du-metric-label" style="margin-top:0.3rem;">Observation</div>
          <div style="display:flex;gap:1.4rem;font-family:'SFMono-Regular',monospace;font-size:13px;margin-top:0.15rem;">
            <span>CPU&nbsp;<b>{ppo.observation_timestamp if ppo.observation_timestamp is not None else '–'}</b></span>
          </div>
          <div class="du-metric-label" style="margin-top:0.5rem;">Last Decision</div>
          <div class="mono" style="font-size:16px;font-weight:700;color:var(--vermilion);">{latest_action}</div>
          <div class="mono" style="font-size:13px;margin-top:0.2rem;">{transition} &nbsp;&middot;&nbsp; Reward {reward}</div>
          <div class="mono" style="font-size:12px;color:var(--muted);margin-top:0.2rem;">{ts}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if ppo.decisions:
        with st.expander("Decision history"):
            rows = "\n".join(
                f"{d.timestamp:8.1f}s  before={d.replicas_before}  action={d.action.value:<10}  "
                f"after={d.replicas_after}  reward={d.reward:+.3f}"
                for d in ppo.decisions
            )
            st.markdown(f'<pre class="mono" style="font-size:12px;">{rows}</pre>', unsafe_allow_html=True)
