"""demo_ui/tests/test_live_details_v2.py

V2 Phase 5E: Live Details (V2) renders with fixtures, with a real
experiment_store record, and with no data (empty state) -- never crashes
on a missing field.
"""

from __future__ import annotations

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
        assert "No V2 experiment has been run yet" in html

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
        experiment_store.update_run(run_id, generation="v2", stage="EPISODE_RUNNING")

        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Live Details").run()
        assert not at.exception
        html = "".join(m.value for m in at.markdown)
        assert "V2 LIVE" in html or "V2 DEMO" in html
