"""Streamlit AppTest smoke tests (sec.4, 20). These run the real app
scripts headlessly (no browser, no server socket) to catch rendering
exceptions and verify basic structure. They require `streamlit` and
`plotly`, so this module is skipped entirely when those are not installed
(the project's scientific .venv does not have them by design -- these
tests are meant to be run from the separate UI venv, e.g.:

    /tmp/rl-k8s-demo-ui-venv/bin/python -m pytest demo_ui/tests/test_app_smoke.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _run_page(page_label: str) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception, f"app.py raised on initial run: {at.exception}"
    at.sidebar.radio[0].set_value(page_label).run()
    assert not at.exception, f"{page_label} page raised: {at.exception}"
    return at


def _all_markdown_html(at: AppTest) -> str:
    return "".join(m.value for m in at.markdown)


def _switch_to_v1(at: AppTest) -> AppTest:
    """V2 Phase 5E: Generation defaults to "V2" -- these V1-specific tests
    (Auto mode, a V1-only feature) must first switch back to "V1", whose
    own code path is otherwise completely unchanged by this phase."""
    at.segmented_control(key="du_generation_widget").set_value("V1").run()
    assert not at.exception
    return at


class TestAppSmoke:
    def test_app_loads_without_exception(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        assert not at.exception

    def test_topbar_present_on_every_page(self):
        for page in ["Experiment", "Live Details", "Comparison", "History"]:
            at = _run_page(page)
            assert "Experiment Console" in _all_markdown_html(at)

    def test_experiment_page_renders_control_strip(self):
        at = _run_page("Experiment")
        html = _all_markdown_html(at)
        assert "du-control-strip" in html

    def test_live_details_page_renders_cluster_section(self):
        at = _run_page("Live Details")
        assert "Cluster" in _all_markdown_html(at)

    def test_comparison_page_renders_table(self):
        at = _run_page("Comparison")
        assert "du-cmp-table" in _all_markdown_html(at)

    def test_history_page_renders_table(self):
        at = _run_page("History")
        assert "du-history-table" in _all_markdown_html(at)

    def test_mode_indicator_present_on_experiment_page(self):
        at = _run_page("Experiment")
        assert "du-mode-indicator" in _all_markdown_html(at)

    def test_start_button_does_not_crash_single_run(self):
        at = _run_page("Experiment")
        at.button[0].click().run()  # START
        assert not at.exception

    def test_auto_mode_switch_does_not_crash(self):
        at = _switch_to_v1(_run_page("Experiment"))
        mode_radio = next(r for r in at.radio if r.key == "du_mode_widget")
        mode_radio.set_value("Auto").run()
        assert not at.exception

    def test_auto_mode_start_shows_pipeline(self):
        at = _switch_to_v1(_run_page("Experiment"))
        mode_radio = next(r for r in at.radio if r.key == "du_mode_widget")
        mode_radio.set_value("Auto").run()
        at.button[0].click().run()  # START, now in Auto mode
        assert not at.exception
        assert "du-pipeline" in _all_markdown_html(at)
