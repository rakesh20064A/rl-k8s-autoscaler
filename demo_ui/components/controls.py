"""components/controls.py -- the data-source toggle shared by the
Experiment page (sec.5-7, 9, 19).

STATE PERSISTENCE: `st.session_state["du_data_source"]` -- the public
contract this whole app reads -- is deliberately NOT used as the widget's
own `key`. Streamlit clears a widget-bound key's value whenever that exact
widget call is skipped on a script run (confirmed directly: navigating to
another page -- whose `render()` never calls this function -- leaves
`du_data_source` as None on return, reverting the segmented control to its
first option). Since `app.py` only calls this function from the Experiment
page, a key bound 1:1 to the widget call is cleared on every OTHER page's
rerun. The widget instead gets its own private key (`du_data_source_widget`),
seeded each render from the public value, and its result is written back
into `du_data_source` explicitly -- a plain dict entry Streamlit's
widget-lifecycle cleanup never touches, so it survives navigating away and
back.

Dependency note: requires `streamlit` (install via demo_ui/requirements-ui.txt
into a SEPARATE venv -- never the project's scientific .venv).
"""

from __future__ import annotations

import streamlit as st

LIVE_CLUSTER = "Live cluster"
SAMPLE_DATA_OFFLINE = "Sample data (offline)"


def render_data_source_toggle() -> str:
    """Returns "Live cluster" or "Sample data (offline)". Never silently
    mixes the two -- the caller must branch on this return value and
    render an unambiguous banner. Defaults to "Live cluster".

    Uses `st.segmented_control` rather than `st.radio`: the radiogroup's
    option text proved unreliable to keep visible across browsers (Safari)
    even after broadening theme.py's forced-color CSS, so this control
    deliberately avoids that DOM/CSS path rather than patching it further.
    """
    st.session_state.setdefault("du_data_source", LIVE_CLUSTER)
    value = st.segmented_control(
        "Data source", [LIVE_CLUSTER, SAMPLE_DATA_OFFLINE], default=st.session_state["du_data_source"],
        selection_mode="single", required=True,
        key="du_data_source_widget", label_visibility="collapsed",
    )
    st.session_state["du_data_source"] = value
    return value
