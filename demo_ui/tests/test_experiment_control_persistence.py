"""Coverage for Experiment control persistence across navigation -- the
general fix (not controller/profile-specific) for Controller/Mode/Traffic
resetting to HPA/Manual/Constant whenever any OTHER page's render() ran in
between, caused by binding those widgets' `key=` directly to the value
Experiment itself reads (Streamlit clears a widget-bound key on any rerun
that skips that exact widget call). Also covers "active run is
authoritative": while a real LIVE run is in progress, its OWN stored
identity -- never local widget state -- drives what Controller/Traffic
display, including recovery when session_state was never populated by
this session's own START click.
"""

from __future__ import annotations

import unittest.mock as mock

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
pytest.importorskip("plotly")

from pathlib import Path  # noqa: E402

from streamlit.testing.v1 import AppTest  # noqa: E402

from demo_ui import experiment_store  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

ALL_SIX_PROFILES = ["Constant", "Ramp", "Burst", "Step", "Periodic", "Mixed"]
ALL_PAGES = ["Live Details", "History", "Comparison"]


@pytest.fixture()
def mocked_subprocess(tmp_path, monkeypatch):
    store_path = tmp_path / "runs.json"
    monkeypatch.setattr(experiment_store, "STORE_PATH", store_path)
    monkeypatch.setattr("demo_ui.live_execution_adapter.experiment_store", experiment_store)

    with mock.patch("demo_ui.live_execution_adapter.subprocess.Popen") as m_popen, \
         mock.patch("demo_ui.live_execution_adapter.tempfile.mkstemp") as m_mkstemp, \
         mock.patch("demo_ui.live_execution_adapter.os.close"), \
         mock.patch("builtins.open", mock.mock_open()):
        status_file = tmp_path / "status.json"
        log_file = tmp_path / "log.log"
        m_mkstemp.side_effect = lambda **kw: (1, str(status_file)) if "status" in kw.get("prefix", "") else (2, str(log_file))
        m_popen.return_value.poll.return_value = None
        m_popen.return_value.communicate.return_value = ("", "")
        m_popen.return_value.returncode = 1
        m_popen.return_value.__enter__.return_value = m_popen.return_value
        yield store_path


def _go_to_experiment_live(at):
    at.sidebar.radio[0].set_value("Experiment").run()
    at.segmented_control(key="du_data_source_widget").set_value("LIVE").run()


def _select(at, controller_label, profile_label):
    at.radio(key="du_controller_widget").set_value(controller_label).run()
    at.selectbox(key="du_profile_widget").set_value(profile_label).run()


def _current_selection(at):
    return (
        at.radio(key="du_controller_widget").value,
        at.selectbox(key="du_profile_widget").value,
    )


class TestSelectionSurvivesNavigation:
    @pytest.mark.parametrize("controller,profile", [
        ("PPO", "Smoke Test (40s)"),
        ("HPA", "Ramp"),
        ("PPO", "Burst"),
        ("HPA", "Periodic"),
        ("PPO", "Mixed"),
        ("HPA", "Step"),
    ])
    def test_single_hop_navigation_preserves_selection(self, controller, profile):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        _go_to_experiment_live(at)
        _select(at, controller, profile)
        assert _current_selection(at) == (controller, profile)

        at.sidebar.radio[0].set_value("Live Details").run()
        at.sidebar.radio[0].set_value("Experiment").run()
        assert not at.exception, at.exception
        assert _current_selection(at) == (controller, profile)

    def test_navigation_through_every_page_preserves_selection(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        _go_to_experiment_live(at)
        _select(at, "PPO", "Burst")

        for page in ALL_PAGES:
            at.sidebar.radio[0].set_value(page).run()
            assert not at.exception, at.exception

        at.sidebar.radio[0].set_value("Experiment").run()
        assert not at.exception, at.exception
        assert _current_selection(at) == ("PPO", "Burst")
        assert at.session_state["du_data_source"] == "LIVE"

    @pytest.mark.parametrize("profile", ALL_SIX_PROFILES)
    def test_every_scientific_profile_round_trips(self, profile):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        _go_to_experiment_live(at)
        _select(at, "HPA", profile)
        at.sidebar.radio[0].set_value("History").run()
        at.sidebar.radio[0].set_value("Experiment").run()
        assert not at.exception, at.exception
        assert _current_selection(at) == ("HPA", profile)

    @pytest.mark.parametrize("controller", ["HPA", "PPO"])
    def test_both_controllers_round_trip(self, controller):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        _go_to_experiment_live(at)
        _select(at, controller, "Ramp")
        at.sidebar.radio[0].set_value("Comparison").run()
        at.sidebar.radio[0].set_value("Experiment").run()
        assert not at.exception, at.exception
        assert _current_selection(at) == (controller, "Ramp")

    def test_demo_live_toggle_persists_alongside_controls(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        _go_to_experiment_live(at)
        _select(at, "PPO", "Mixed")
        at.sidebar.radio[0].set_value("Live Details").run()
        at.sidebar.radio[0].set_value("Experiment").run()
        assert not at.exception, at.exception
        assert at.session_state["du_data_source"] == "LIVE"
        assert _current_selection(at) == ("PPO", "Mixed")


class TestActiveRunAuthority:
    def test_started_run_identity_matches_store(self, mocked_subprocess):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        _go_to_experiment_live(at)
        _select(at, "PPO", "Smoke Test (40s)")
        at.button(key="du_start_button").click().run()
        assert not at.exception, at.exception

        active = experiment_store.active_run(mocked_subprocess)
        assert active["controller"] == "PPO"
        assert active["profile_name"] == "demo_smoke_test"

    def test_controls_reflect_active_run_while_running(self, mocked_subprocess):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        _go_to_experiment_live(at)
        _select(at, "PPO", "Smoke Test (40s)")
        at.button(key="du_start_button").click().run()
        assert not at.exception, at.exception
        assert _current_selection(at) == ("PPO", "Smoke Test (40s)")

    def test_running_controls_cannot_visually_drift_even_if_widgets_changed(self, mocked_subprocess):
        """The precise regression: HPA+Constant appearing while PPO+Smoke
        is actually executing. Disabled widgets cannot be changed via
        `.set_value()` in a real browser, but asserting `disabled=True`
        here directly proves the display is locked to the active run."""
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        _go_to_experiment_live(at)
        _select(at, "PPO", "Smoke Test (40s)")
        at.button(key="du_start_button").click().run()
        assert not at.exception, at.exception

        controller_widget = at.radio(key="du_controller_widget")
        profile_widget = at.selectbox(key="du_profile_widget")
        assert controller_widget.disabled is True
        assert profile_widget.disabled is True
        assert controller_widget.value == "PPO"
        assert profile_widget.value == "Smoke Test (40s)"

    def test_recovery_from_active_run_when_public_selection_keys_absent(self, mocked_subprocess):
        """Simulates a Streamlit restart mid-run: session_state never saw
        this run's own START click (no du_selected_controller/profile set
        by THIS session), but the run store still says PPO/Ramp is active.
        The UI must reconstruct that identity from the store, not invent
        HPA/Constant defaults."""
        experiment_store.start_run("PPO", "ramp", path=mocked_subprocess)
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        _go_to_experiment_live(at)
        assert not at.exception, at.exception
        assert _current_selection(at) == ("PPO", "Ramp")

    def test_completion_preserves_last_selection_not_defaults(self, mocked_subprocess):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        _go_to_experiment_live(at)
        _select(at, "PPO", "Smoke Test (40s)")
        at.button(key="du_start_button").click().run()
        run_id = experiment_store.active_run(mocked_subprocess)["run_id"]

        with mock.patch("demo_ui.live_execution_adapter.RealExperimentController.status",
                        return_value={"status": "completed", "summary": {"workload_hash": "h", "workload_seed": 1, "workload_duration_seconds": 40.0}}):
            experiment_store.update_run(run_id, status="completed", path=mocked_subprocess)
            at.run()
            assert not at.exception, at.exception

        assert _current_selection(at) == ("PPO", "Smoke Test (40s)")
        controller_widget = at.radio(key="du_controller_widget")
        assert controller_widget.disabled is False  # re-enabled once terminal

    def test_user_can_explicitly_change_selection_after_completion(self, mocked_subprocess):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        _go_to_experiment_live(at)
        _select(at, "PPO", "Smoke Test (40s)")
        at.button(key="du_start_button").click().run()
        run_id = experiment_store.active_run(mocked_subprocess)["run_id"]

        with mock.patch("demo_ui.live_execution_adapter.RealExperimentController.status",
                        return_value={"status": "completed", "summary": {}}):
            experiment_store.update_run(run_id, status="completed", path=mocked_subprocess)
            at.run()

        _select(at, "HPA", "Ramp")
        assert _current_selection(at) == ("HPA", "Ramp")

    def test_no_duplicate_start_on_repeated_reruns_while_running(self, mocked_subprocess):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        _go_to_experiment_live(at)
        _select(at, "PPO", "Smoke Test (40s)")
        at.button(key="du_start_button").click().run()
        assert len(experiment_store._load(mocked_subprocess)) == 1
        for page in ALL_PAGES:
            at.sidebar.radio[0].set_value(page).run()
            at.sidebar.radio[0].set_value("Experiment").run()
        assert not at.exception, at.exception
        assert len(experiment_store._load(mocked_subprocess)) == 1

    def test_start_uses_exactly_the_visible_configuration(self, mocked_subprocess):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        _go_to_experiment_live(at)
        _select(at, "HPA", "Burst")
        at.button(key="du_start_button").click().run()
        active = experiment_store.active_run(mocked_subprocess)
        assert active["controller"] == "HPA"
        assert active["profile_name"] == "burst"
