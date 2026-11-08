"""Coverage for the "Live Details must never default to fixture data"
correctness phase: a fresh browser/restarted Streamlit session (no
du_data_source key set, zero experiment_store records) must render the
real/empty LIVE view as PRIMARY content -- never gated behind a
session_state key that only Experiment happens to seed -- with fixture
content confined to one collapsed expander.
"""

from __future__ import annotations

import inspect

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

    def test_primary_helpers_have_no_widget_calls(self):
        for fn in (
            live_details._render_run_banner,
            live_details._render_run_telemetry,
            live_details._render_experiment_events,
            live_details._render_current_cluster_state,
        ):
            src = inspect.getsource(fn)
            for forbidden in [".selectbox(", ".radio(", ".segmented_control("]:
                assert forbidden not in src, f"{fn.__name__} must not define its own controls"

    def test_only_fixture_dashboard_defines_selectors(self):
        src = inspect.getsource(live_details._render_fixture_dashboard)
        assert ".selectbox(" in src  # confined here, and ONLY here

    def test_fixture_dashboard_call_is_gated_behind_opt_in_flag(self):
        """Being inside `with st.expander(...)` is NOT sufficient isolation
        -- Streamlit executes an expander's body every rerun regardless of
        collapsed/expanded state, so `_render_fixture_dashboard()` (and the
        `build_all_fixture_summaries()` it calls) must only be reachable
        through an explicit opt-in flag, never unconditionally."""
        src = inspect.getsource(live_details.render)
        assert "du_show_live_fixture" in src
        assert 'if st.session_state.get("du_show_live_fixture"' in src

    def test_render_never_branches_on_data_source(self):
        """The actual bug: `if data_source == "LIVE": ... else: <fixture>`
        made fixture data the default whenever du_data_source wasn't set
        yet (e.g. Live Details visited before Experiment ever ran once)."""
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
        assert "No LIVE experiment has been run yet." in html

    def test_fresh_experiment_cards_are_dashes(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Live Details").run()
        assert not at.exception, at.exception
        # The FIRST rendered metric strip (by document order, which
        # precedes the fixture expander) is the primary Experiment
        # Telemetry one. "du-metric-label" also appears in theme.py's own
        # <style> block (as a CSS class definition), so anchor on the
        # actual cell text, not just the class name.
        first_strip = next(m.value for m in at.markdown if 'du-metric-label">Controller</div>' in m.value)
        assert 'du-metric-label">Controller</div><div class="du-metric-value">—</div>' in first_strip
        assert 'du-metric-label">Replicas</div><div class="du-metric-value">—</div>' in first_strip
        assert "25" not in first_strip and "188" not in first_strip  # the exact fixture numbers reported as leaking

    def test_fixture_dashboard_not_rendered_before_opt_in(self):
        """The actual regression: merely nesting inside a collapsed
        `st.expander` still executes `build_all_fixture_summaries()` and
        renders its output every rerun. Nothing fixture-related may appear
        anywhere on the page until the user explicitly clicks the load
        button."""
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Live Details").run()
        assert not at.exception, at.exception
        html = "".join(m.value for m in at.markdown)
        assert "25.0" not in html
        assert "188%" not in html
        assert "Constant" not in html
        assert at.session_state.get("du_show_live_fixture", False) is False
        # The opt-in button itself must be present (it's the only fixture
        # widget allowed to exist before opt-in).
        assert any(b.label == "Load demo fixture data" for b in at.button)
        selectbox_keys = {sb.key for sb in at.selectbox}
        assert selectbox_keys == set()  # fixture's own selectboxes not yet instantiated

    def test_fixture_dashboard_appears_after_opt_in_without_disturbing_primary_state(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Live Details").run()
        primary_before = next(m.value for m in at.markdown if 'du-metric-label">Controller</div>' in m.value)

        at.button(key="du_load_live_fixture").click().run()
        assert not at.exception, at.exception
        assert at.session_state["du_show_live_fixture"] is True
        html = "".join(m.value for m in at.markdown)
        assert "Constant" in html  # fixture dropdown's own default option now rendered
        selectbox_keys = {sb.key for sb in at.selectbox}
        assert selectbox_keys == {"du_live_profile", "du_live_controller"}

        primary_after = next(m.value for m in at.markdown if 'du-metric-label">Controller</div>' in m.value)
        assert primary_after == primary_before  # opting into fixture data never touches the real LIVE state

    def test_fresh_events_placeholder(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Live Details").run()
        assert not at.exception, at.exception
        html = "".join(m.value for m in at.markdown)
        assert "No experiment events yet." in html

    def test_fixture_content_confined_to_one_expander(self):
        """V2 Phase 5E: a second, clearly V2-labeled expander was added
        alongside V1's own (unchanged) one -- both are fixture-only,
        collapsed by default, never shown as primary content."""
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Live Details").run()
        assert not at.exception, at.exception
        labels = [e.label for e in at.expander]
        assert labels == [
            "Demo fixture data (simulated, not real LIVE results)",
            "V2 demo fixture data (simulated, not evaluation evidence)",
        ]

    def test_navigation_preserves_same_active_run(self):
        run_id = experiment_store.start_run("HPA", "demo_smoke_test", path=self.store_path)
        experiment_store.update_run(
            run_id, stage="EPISODE_RUNNING", workload_hash="h1", workload_seed=42,
            workload_duration_s=40.0, path=self.store_path,
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
