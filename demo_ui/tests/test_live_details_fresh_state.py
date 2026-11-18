"""Coverage for the "Live Details must never default to sample data"
correctness phase: a fresh browser/restarted Streamlit session (zero
experiment_store records) must render the real/empty LIVE view as PRIMARY
content -- never gated behind a session_state key that only Experiment
happens to seed -- with sample-data content confined to one collapsed
expander.
"""

from __future__ import annotations

import inspect
import os

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
pytest.importorskip("plotly")

from pathlib import Path  # noqa: E402

from streamlit.testing.v1 import AppTest  # noqa: E402

import demo_ui.views.live_details as live_details  # noqa: E402
from demo_ui import experiment_store  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


class TestNoIndependentControlsInPrimaryPath:
    """Source-level, not just rendering-level: precise regardless of how
    Streamlit's AppTest flattens widgets inside a collapsed expander (it
    still instantiates them -- only the browser hides them visually)."""

    def test_render_function_itself_has_no_widget_calls(self):
        src = inspect.getsource(live_details.render)
        for forbidden in [".selectbox(", ".radio(", ".segmented_control("]:
            assert forbidden not in src

    def test_primary_helper_has_no_widget_calls(self):
        src = inspect.getsource(live_details._render_current_cluster_state)
        for forbidden in [".selectbox(", ".radio(", ".segmented_control("]:
            assert forbidden not in src

    def test_only_sample_data_dashboard_defines_selectors(self):
        src = inspect.getsource(live_details._render_sample_data_dashboard)
        assert ".selectbox(" in src  # confined here, and ONLY here

    def test_sample_data_dashboard_call_is_gated_behind_opt_in_flag(self):
        """Being inside `with st.expander(...)` is NOT sufficient isolation
        -- Streamlit executes an expander's body every rerun regardless of
        collapsed/expanded state, so `_render_sample_data_dashboard()` must
        only be reachable through an explicit opt-in flag, never
        unconditionally."""
        src = inspect.getsource(live_details.render)
        assert "du_show_live_fixture_v2" in src
        assert 'if st.session_state.get("du_show_live_fixture_v2"' in src

    def test_render_never_branches_on_data_source(self):
        """The actual bug: `if data_source == "LIVE": ... else: <sample>`
        made sample data the default whenever du_data_source wasn't set yet
        (e.g. Live Details visited before Experiment ever ran once)."""
        src = inspect.getsource(live_details.render)
        assert "du_data_source" not in src
        assert "data_source" not in src


class TestFreshStoreRendering:
    @pytest.fixture(autouse=True)
    def _fresh_store(self, tmp_path, monkeypatch):
        store_path = tmp_path / "runs.json"
        monkeypatch.setattr(experiment_store, "STORE_PATH", store_path)
        self.store_path = store_path

    def test_fresh_header(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Live Details").run()
        assert not at.exception, at.exception
        html = "".join(m.value for m in at.markdown)
        assert "No experiment has been run yet." in html

    def test_fresh_experiment_cards_are_dashes(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Live Details").run()
        assert not at.exception, at.exception
        first_strip = next(m.value for m in at.markdown if 'du-metric-label">Arm</div>' in m.value)
        assert 'du-metric-label">Arm</div><div class="du-metric-value">—</div>' in first_strip

    def test_sample_data_dashboard_not_rendered_before_opt_in(self):
        """The actual regression: merely nesting inside a collapsed
        `st.expander` still executes `build_fixture_v2_run()` and renders
        its output every rerun. Nothing sample-data-related may appear
        anywhere on the page until the user explicitly clicks the load
        button."""
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Live Details").run()
        assert not at.exception, at.exception
        assert at.session_state.get("du_show_live_fixture_v2", False) is False
        # The opt-in button itself must be present (it's the only
        # sample-data widget allowed to exist before opt-in).
        assert any(b.label == "Load sample data" for b in at.button)
        selectbox_keys = {sb.key for sb in at.selectbox}
        assert selectbox_keys == set()  # sample-data dashboard's own selectboxes not yet instantiated

    def test_sample_data_dashboard_appears_after_opt_in_without_disturbing_primary_state(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Live Details").run()
        primary_before = next(m.value for m in at.markdown if 'du-metric-label">Arm</div>' in m.value)

        at.button(key="du_load_live_fixture_v2").click().run()
        assert not at.exception, at.exception
        assert at.session_state["du_show_live_fixture_v2"] is True
        selectbox_keys = {sb.key for sb in at.selectbox}
        assert selectbox_keys == {"du_live_v2_profile", "du_live_v2_arm"}

        primary_after = next(m.value for m in at.markdown if 'du-metric-label">Arm</div>' in m.value)
        assert primary_after == primary_before  # opting into sample data never touches the real LIVE state

    def test_sample_data_content_confined_to_one_expander(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Live Details").run()
        assert not at.exception, at.exception
        labels = [e.label for e in at.expander]
        assert labels == ["Sample data (offline) -- simulated, not real LIVE results"]

    def test_navigation_preserves_same_active_run(self):
        run_id = experiment_store.start_run("hpa_v2_cpu_memory", "demo_cpu_short", path=self.store_path)
        # A live pid is required for `active_run()`'s liveness check to
        # treat this as a genuinely running process, not a stale record.
        experiment_store.update_run(
            run_id, generation="v2", stage="EPISODE_RUNNING", workload_hash="h1", workload_seed=42,
            workload_duration_s=40.0, pid=os.getpid(), path=self.store_path,
        )
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Live Details").run()
        html1 = "".join(m.value for m in at.markdown)
        assert run_id in html1

        at.sidebar.radio[0].set_value("Experiment").run()
        at.sidebar.radio[0].set_value("Live Details").run()
        assert not at.exception, at.exception
        html2 = "".join(m.value for m in at.markdown)
        assert run_id in html2
