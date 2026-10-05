"""components/provenance.py -- Fair traffic replay / experiment validity
panel (sec.13, 16). Displays schedule hashes and whether HPA/PPO used the
identical traffic schedule. Fixture hashes only in this phase -- no live
replay is implemented.
"""

from __future__ import annotations

import streamlit as st

from demo_ui.models import RunSummary


def render_fairness_panel(hpa: RunSummary, ppo: RunSummary) -> None:
    same = hpa.schedule_hash == ppo.schedule_hash
    badge_class = "status-ok" if same else "status-error"
    verdict = "VERIFIED" if same else "MISMATCH"

    st.markdown(
        f"""
        <div class="du-panel">
          <div class="du-section-label">Traffic Match</div>
          <div style="display:flex;justify-content:space-between;align-items:baseline;">
            <div class="mono" style="font-size:12.5px;color:var(--muted);">
              {hpa.traffic_profile.value.capitalize()} &middot; Seed {hpa.seed} &middot; {hpa.duration_s:.0f}s
            </div>
            <span class="du-badge {badge_class}">{verdict}</span>
          </div>
          <div class="mono" style="font-size:11.5px;color:var(--muted);margin-top:0.3rem;">
            HPA&nbsp;sha256&nbsp;{hpa.schedule_hash}<br/>
            PPO&nbsp;sha256&nbsp;{ppo.schedule_hash}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if hpa.is_demo_data or ppo.is_demo_data:
        st.caption("Simulated schedule hash for demonstration — not a cryptographic provenance artifact.")
