"""demo_ui/app.py -- Streamlit entrypoint.

HARD BOUNDARY: this module (and everything it imports) must never start a
Prometheus port-forward, bind port 9090, call kubectl, start/stop the
traffic generator, start/stop PPO training, or create/remove the HPA. It is
wired ONLY to the Fixture* providers in this phase.

Not run during this phase (streamlit is not yet installed in this repo's
.venv) -- see demo_ui/README.md for how to launch it once the dependency is
added, and do so only when no live rollout is in progress.
"""

from __future__ import annotations

import streamlit as st

from demo_ui import theme
from demo_ui.views import comparison, experiment, history, live_details, training_v2

st.set_page_config(page_title="RL Autoscaler Lab", layout="wide")

PAGES = {
    "Experiment": experiment,
    "Live Details": live_details,
    "Comparison": comparison,
    "Training": training_v2,
    "History": history,
}


def main() -> None:
    st.markdown(theme.inject_css(), unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div style="font-size:11px;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted);">'
        'RL Autoscaler</div><div style="font-size:16px;font-weight:700;margin-bottom:0.6rem;">Experiment Console</div>',
        unsafe_allow_html=True,
    )
    choice = st.sidebar.radio("Page", list(PAGES.keys()), label_visibility="collapsed")
    PAGES[choice].render()


if __name__ == "__main__":
    main()
