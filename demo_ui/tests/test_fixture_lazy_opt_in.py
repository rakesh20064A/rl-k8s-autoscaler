"""Coverage for lazy/opt-in fixture rendering on Comparison and History
(the same "collapsed expander still executes its body every rerun"
correctness fix applied to Live Details). On a fresh run-store, neither
page may render any fixture metric, graph trace, or row until the user
explicitly clicks "Load demo fixture data".
"""

from __future__ import annotations

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
pytest.importorskip("plotly")

from pathlib import Path  # noqa: E402

from streamlit.testing.v1 import AppTest  # noqa: E402

from demo_ui import experiment_store  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def _fresh_store(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_store, "STORE_PATH", tmp_path / "runs.json")


class TestComparisonLazyFixture:
    def test_fresh_comparison_has_no_fixture_content(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Comparison").run()
        assert not at.exception, at.exception
        assert len(at.selectbox) == 0  # fixture profile selectbox not instantiated
        assert any(b.label == "Load demo fixture data" for b in at.button)
        html = "".join(m.value for m in at.markdown)
        assert "schedule_hash" not in html.lower()

    def test_comparison_fixture_appears_after_opt_in(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Comparison").run()
        at.button(key="du_load_comparison_fixture").click().run()
        assert not at.exception, at.exception
        assert at.session_state["du_show_comparison_fixture"] is True
        assert any(sb.label == "Traffic profile" for sb in at.selectbox)


class TestHistoryLazyFixture:
    def test_fresh_history_has_no_fixture_rows(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("History").run()
        assert not at.exception, at.exception
        html = "".join(m.value for m in at.markdown)
        assert "VALID" not in html and "INVALID" not in html
        assert "No LIVE experiment history yet." in html
        assert any(b.label == "Load demo fixture data" for b in at.button)
        assert len(at.selectbox) == 0

    def test_history_fixture_rows_appear_after_opt_in(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("History").run()
        at.button(key="du_load_history_fixture").click().run()
        assert not at.exception, at.exception
        assert at.session_state["du_show_history_fixture"] is True
        html = "".join(m.value for m in at.markdown)
        assert "VALID" in html or "INVALID" in html
