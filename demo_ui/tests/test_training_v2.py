"""demo_ui/tests/test_training_v2.py

V2 Phase 5E: Training (V2) page -- renders with the real completed
final_min2 sessions (read-only; skipped if none exist yet), with a
fixture, and with no data (clean empty state). Never crashes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
REAL_SESSIONS_DIR = Path("persistent-data/rl_training_v2/final_min2/sessions")


def _all_text(at: AppTest) -> str:
    return "".join(m.value for m in at.markdown) + "".join(c.value for c in at.caption)


class TestTrainingPageInApp:
    def test_training_page_reachable_from_sidebar_and_does_not_crash(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Training").run()
        assert not at.exception

    def test_training_page_shows_v2_training_curve_heading(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Training").run()
        html = "".join(m.value for m in at.markdown)
        assert "V2 Training Curve" in html


class TestRenderWithRealSessions:
    def test_renders_with_real_completed_sessions_if_any_exist(self):
        from demo_ui.views.training_v2 import _load_real_curve

        completed = []
        try:
            completed = _load_real_curve()
        except Exception:  # noqa: BLE001
            pass
        if not completed:
            pytest.skip("no completed V2 final_min2 sessions exist yet")

        def _script():
            from demo_ui.views import training_v2

            training_v2.render()

        at = AppTest.from_function(_script, default_timeout=30)
        at.run()
        assert not at.exception
        html = "".join(m.value for m in at.markdown)
        assert "Sessions completed" in html
        assert f"{len(completed)} / " in html


class TestRenderWithFixture:
    def test_fixture_curve_renders_without_exception(self):
        def _script():
            from demo_ui.views.training_v2 import _build_fixture_curve, _render_charts, _render_table

            curve = _build_fixture_curve()
            _render_charts(curve)
            _render_table(curve)

        at = AppTest.from_function(_script, default_timeout=30)
        at.run()
        assert not at.exception

    def test_fixture_curve_has_expected_session_count(self):
        from demo_ui.views.training_v2 import _build_fixture_curve

        curve = _build_fixture_curve(n_sessions=6)
        assert len(curve) == 6
        assert curve[0]["session_index"] == 0
        assert all("entropy_loss" in row for row in curve)


class TestRenderWithNoData:
    def test_empty_curve_renders_empty_state_without_exception(self, monkeypatch):
        def _script():
            from demo_ui.views import training_v2

            training_v2._load_real_curve = lambda: []  # noqa: SLF001
            training_v2.render()

        at = AppTest.from_function(_script, default_timeout=30)
        at.run()
        assert not at.exception
        html = "".join(m.value for m in at.markdown)
        assert "No completed V2 final_min2 sessions yet" in html

    def test_empty_state_offers_demo_fixture_expander(self):
        def _script():
            from demo_ui.views import training_v2

            training_v2._load_real_curve = lambda: []  # noqa: SLF001
            training_v2.render()

        at = AppTest.from_function(_script, default_timeout=30)
        at.run()
        labels = [e.label for e in at.expander]
        assert any("Demo fixture data" in label for label in labels)

    def test_read_failure_shows_message_not_crash(self):
        def _script():
            from demo_ui.views import training_v2

            def _raise():
                raise RuntimeError("simulated read race with the live campaign")

            training_v2._load_real_curve = _raise  # noqa: SLF001
            training_v2.render()

        at = AppTest.from_function(_script, default_timeout=30)
        at.run()
        assert not at.exception
        assert "Could not read training sessions" in _all_text(at)
