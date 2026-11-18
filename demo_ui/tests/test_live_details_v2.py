"""demo_ui/tests/test_live_details_v2.py

V2 Phase 5E: Live Details (V2) renders with fixtures, with a real
experiment_store record, and with no data (empty state) -- never crashes
on a missing field.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

from demo_ui import experiment_store  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


class TestRenderWithFixtureData:
    def test_ppo_v2_fixture_renders_without_exception(self):
        # live_details_v2.render() is a pure function of (rec, is_active);
        # exercised directly inside a bare Streamlit script context via
        # AppTest.from_function.
        def _script():
            from demo_ui.fixtures_v2 import build_fixture_v2_run
            from demo_ui.views import live_details_v2

            live_details_v2.render(build_fixture_v2_run("ppo_v2", "demo_cpu_short"), is_active=False)

        at = AppTest.from_function(_script, default_timeout=30)
        at.run()
        assert not at.exception

    def test_hpa_v2_fixture_renders_without_exception(self):
        def _script():
            from demo_ui.fixtures_v2 import build_fixture_v2_run
            from demo_ui.views import live_details_v2

            live_details_v2.render(build_fixture_v2_run("hpa_v2_cpu_memory", "demo_memory_short"), is_active=False)

        at = AppTest.from_function(_script, default_timeout=30)
        at.run()
        assert not at.exception

    def test_ppo_fixture_shows_decision_explanation_panel(self):
        def _script():
            from demo_ui.fixtures_v2 import build_fixture_v2_run
            from demo_ui.views import live_details_v2

            live_details_v2.render(build_fixture_v2_run("ppo_v2", "demo_mixed_short"), is_active=False)

        at = AppTest.from_function(_script, default_timeout=30)
        at.run()
        html = "".join(m.value for m in at.markdown)
        assert "PPO Decision Explanation" in html
        assert "Chosen action" in html

    def test_hpa_fixture_shows_hpa_status_panel(self):
        def _script():
            from demo_ui.fixtures_v2 import build_fixture_v2_run
            from demo_ui.views import live_details_v2

            live_details_v2.render(build_fixture_v2_run("hpa_v2_cpu_memory", "demo_cpu_short"), is_active=False)

        at = AppTest.from_function(_script, default_timeout=30)
        at.run()
        html = "".join(m.value for m in at.markdown)
        assert "HPA Status" in html
        assert "demo-app-hpa-v2" in html


class TestRenderWithNoData:
    def test_none_record_renders_empty_state_without_exception(self):
        def _script():
            from demo_ui.views import live_details_v2

            live_details_v2.render(None, is_active=False)

        at = AppTest.from_function(_script, default_timeout=30)
        at.run()
        assert not at.exception
        html = "".join(m.value for m in at.markdown)
        assert "No experiment has been run yet" in html

    def test_record_with_no_samples_yet_renders_without_exception(self):
        def _script():
            from demo_ui.views import live_details_v2

            rec = {
                "run_id": "r1", "controller": "ppo_v2", "profile_name": "demo_cpu_short",
                "status": "running", "stage": "STARTING", "data_source": "LIVE", "generation": "v2",
                "summary": None,
            }
            live_details_v2.render(rec, is_active=True)

        at = AppTest.from_function(_script, default_timeout=30)
        at.run()
        assert not at.exception


class TestFullPageWithV2ActiveRun:
    def test_live_details_page_delegates_to_v2_when_active_run_is_v2(self, tmp_path, monkeypatch):
        monkeypatch.setattr(experiment_store, "STORE_PATH", tmp_path / "runs.json")
        run_id = experiment_store.start_run("ppo_v2", "demo_cpu_short")
        # A live pid is required for `active_run()`'s liveness check to
        # treat this as a genuinely running process, not a stale record.
        experiment_store.update_run(run_id, generation="v2", stage="EPISODE_RUNNING", pid=os.getpid())

        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Live Details").run()
        assert not at.exception
        html = "".join(m.value for m in at.markdown)
        assert run_id in html


class TestRunSelectorLabelFormat:
    """Task sec.4's exact format: "PPO · Bursts / CPU / High · 18 Nov
    12:40 · completed"."""

    def test_label_format_for_a_36_catalog_profile(self):
        from datetime import datetime

        from demo_ui.views import live_details_v2

        started_at = datetime(2026, 11, 18, 12, 40, 0).timestamp()
        rec = {
            "controller": "ppo_v2", "profile_name": "demo_bursts_cpu_high",
            "started_at": started_at, "status": "completed",
        }
        label = live_details_v2.run_selector_label(rec)
        assert label == "PPO · Bursts / CPU / High · 18 Nov 12:40 · completed"

    def test_label_format_for_hpa_running(self):
        from datetime import datetime

        from demo_ui.views import live_details_v2

        started_at = datetime(2026, 1, 5, 9, 5, 0).timestamp()
        rec = {
            "controller": "hpa_v2_cpu_memory", "profile_name": "demo_constant_memory_normal",
            "started_at": started_at, "status": "running",
        }
        label = live_details_v2.run_selector_label(rec)
        assert label.startswith("HPA · Constant / Memory / Normal · 5 Jan 09:05 · running")


class TestRunSelectorDefaultsAndNoRedundantPanels:
    def test_selector_defaults_to_the_active_run(self, tmp_path, monkeypatch):
        monkeypatch.setattr(experiment_store, "STORE_PATH", tmp_path / "runs.json")
        old_id = experiment_store.start_run("hpa_v2_cpu_memory", "demo_cpu_short")
        experiment_store.update_run(old_id, generation="v2", status="completed", summary={})
        active_id = experiment_store.start_run("ppo_v2", "demo_memory_short")
        experiment_store.update_run(active_id, generation="v2", stage="EPISODE_RUNNING", pid=os.getpid())

        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Live Details").run()
        assert not at.exception
        selector = at.selectbox(key="du_v2_live_details_run_selector")
        assert selector.value == active_id

    def test_selector_never_defaults_to_an_invalid_no_traffic_run(self, tmp_path, monkeypatch):
        """Task sec.3: an `invalid_no_traffic` run must never be the
        auto-selected default, even when it is the most recently
        completed run and no run is currently active."""
        monkeypatch.setattr(experiment_store, "STORE_PATH", tmp_path / "runs.json")
        valid_id = experiment_store.start_run("ppo_v2", "demo_memory_short")
        experiment_store.update_run(valid_id, generation="v2", status="completed", summary={})
        invalid_id = experiment_store.start_run("hpa_v2_cpu_memory", "demo_cpu_short")
        experiment_store.update_run(invalid_id, generation="v2", status=experiment_store.INVALID_NO_TRAFFIC, summary={})

        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Live Details").run()
        assert not at.exception
        selector = at.selectbox(key="du_v2_live_details_run_selector")
        assert selector.value == valid_id
        # Still selectable -- just never the default.
        assert any("invalid_no_traffic" in opt for opt in selector.options)
        assert len(selector.options) == 2

    def test_no_not_available_panels_for_a_completed_ppo_run(self, tmp_path, monkeypatch):
        monkeypatch.setattr(experiment_store, "STORE_PATH", tmp_path / "runs.json")
        run_id = experiment_store.start_run("ppo_v2", "demo_cpu_short")
        experiment_store.update_run(run_id, generation="v2", status="completed", summary={})

        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Live Details").run()
        assert not at.exception
        html = "".join(m.value for m in at.markdown)
        assert "no live PPO decision feed" not in html

    def test_no_not_available_panels_for_a_completed_hpa_run(self, tmp_path, monkeypatch):
        monkeypatch.setattr(experiment_store, "STORE_PATH", tmp_path / "runs.json")
        run_id = experiment_store.start_run("hpa_v2_cpu_memory", "demo_cpu_short")
        experiment_store.update_run(run_id, generation="v2", status="completed", summary={})

        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Live Details").run()
        assert not at.exception
        html = "".join(m.value for m in at.markdown)
        assert "no live PPO decision feed" not in html

    def test_no_selector_widget_when_store_is_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(experiment_store, "STORE_PATH", tmp_path / "runs.json")

        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Live Details").run()
        assert not at.exception
        selector_keys = {sb.key for sb in at.selectbox}
        assert "du_v2_live_details_run_selector" not in selector_keys
        html = "".join(m.value for m in at.markdown)
        assert "No experiment has been run yet." in html
