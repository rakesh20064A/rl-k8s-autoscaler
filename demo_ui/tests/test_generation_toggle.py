"""demo_ui/tests/test_generation_toggle.py

V2 Phase 5E: the Experiment page's Generation selector -- V2 default,
switching to V1 restores V1's own exact behaviour, the V2 LIVE-disabled
message shows when the model/epsilon files are missing (the normal state
before training finishes), and V2 DEMO works regardless.
"""

from __future__ import annotations

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _goto_experiment(at: AppTest) -> AppTest:
    at.run()
    at.sidebar.radio[0].set_value("Experiment").run()
    return at


class TestGenerationDefaultsToV2:
    def test_generation_widget_defaults_to_v2(self):
        at = _goto_experiment(AppTest.from_file(APP_PATH, default_timeout=30))
        seg = at.segmented_control(key="du_generation_widget")
        assert seg.value == "V2"

    def test_v2_control_strip_renders_by_default(self):
        at = _goto_experiment(AppTest.from_file(APP_PATH, default_timeout=30))
        # The V2 arm selector offers PPO-V2 / HPA-V2, never V1's HPA/PPO radio.
        arm_radio = at.radio(key="du_v2_arm_widget")
        assert set(arm_radio.options) == {"PPO-V2", "HPA-V2 (CPU+memory)"}


class TestSwitchingToV1RestoresV1Behaviour:
    def test_switching_to_v1_shows_the_v1_controller_radio(self):
        at = _goto_experiment(AppTest.from_file(APP_PATH, default_timeout=30))
        at.segmented_control(key="du_generation_widget").set_value("V1").run()
        assert not at.exception
        controller_radio = at.radio(key="du_controller_widget")
        assert set(controller_radio.options) == {"HPA", "PPO"}

    def test_v1_auto_mode_still_works_after_switch(self):
        at = _goto_experiment(AppTest.from_file(APP_PATH, default_timeout=30))
        at.segmented_control(key="du_generation_widget").set_value("V1").run()
        mode_radio = at.radio(key="du_mode_widget")
        mode_radio.set_value("Auto").run()
        assert not at.exception
        at.button[0].click().run()
        assert not at.exception
        assert "du-pipeline" in "".join(m.value for m in at.markdown)

    def test_v1_start_button_still_works_after_switch(self):
        at = _goto_experiment(AppTest.from_file(APP_PATH, default_timeout=30))
        at.segmented_control(key="du_generation_widget").set_value("V1").run()
        at.button[0].click().run()  # START (Manual, HPA, Constant -- V1 defaults)
        assert not at.exception

    def test_switching_back_to_v2_restores_v2_controls(self):
        at = _goto_experiment(AppTest.from_file(APP_PATH, default_timeout=30))
        at.segmented_control(key="du_generation_widget").set_value("V1").run()
        at.segmented_control(key="du_generation_widget").set_value("V2").run()
        assert not at.exception
        arm_radio = at.radio(key="du_v2_arm_widget")
        assert set(arm_radio.options) == {"PPO-V2", "HPA-V2 (CPU+memory)"}


class TestV2LiveDisabledMessage:
    def test_live_disabled_message_shown_when_model_and_epsilon_missing(self):
        """The normal state before training finishes: config_v2.yaml points
        at session_047's checkpoint and the official epsilon file, neither
        of which exists yet -- V2 LIVE must show a clear, specific message,
        never a crash and never a silently-enabled START."""
        at = _goto_experiment(AppTest.from_file(APP_PATH, default_timeout=30))
        at.segmented_control(key="du_data_source_widget").set_value("LIVE").run()
        assert not at.exception
        html = "".join(m.value for m in at.markdown)
        assert "V2 LIVE is disabled" in html
        assert "model checkpoint not found" in html
        assert "epsilon calibration file not found" in html

    def test_start_button_disabled_in_v2_live_when_not_enabled(self):
        at = _goto_experiment(AppTest.from_file(APP_PATH, default_timeout=30))
        at.segmented_control(key="du_data_source_widget").set_value("LIVE").run()
        start_button = [b for b in at.button if b.label == "START"][0]
        assert start_button.disabled is True


class TestV2DemoWorksRegardlessOfLiveReadiness:
    def test_v2_demo_start_loads_a_fixture_without_exception(self):
        at = _goto_experiment(AppTest.from_file(APP_PATH, default_timeout=30))
        # Data source defaults to DEMO -- start directly.
        start_button = [b for b in at.button if b.label == "START"][0]
        start_button.click().run()
        assert not at.exception
        success_text = "".join(s.value for s in at.success)
        assert "DEMO fixture loaded" in success_text
        assert "DEMO ONLY, not evaluation evidence" in success_text

    def test_v2_demo_stop_clears_the_loaded_fixture(self):
        at = _goto_experiment(AppTest.from_file(APP_PATH, default_timeout=30))
        start_button = [b for b in at.button if b.label == "START"][0]
        start_button.click().run()
        stop_button = [b for b in at.button if b.label == "STOP"][0]
        stop_button.click().run()
        assert not at.exception
        caption_text = "".join(c.value for c in at.caption)
        assert "No V2 demo run loaded" in caption_text
