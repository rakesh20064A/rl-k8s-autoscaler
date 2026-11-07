"""Integration coverage for the cross-page experiment-run synchronization
(Experiment is the only control plane; Live Details/History/Comparison are
read-only consumers of demo_ui.experiment_store). Uses Streamlit's AppTest
with the real subprocess mocked out (never launches a real episode) and a
per-test `experiment_store` path so tests never share state.
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


@pytest.fixture()
def mocked_subprocess(tmp_path, monkeypatch):
    """Patches experiment_store's default path to a per-test file AND
    mocks the real subprocess lifecycle, so AppTest exercises the real
    RealExperimentController/experiment_store code paths without ever
    launching scripts/run_demo_episode.py."""
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
        # Patching subprocess.Popen patches it process-wide (live_providers.py's
        # `subprocess.run(["kubectl", ...])` also goes through Popen
        # internally) -- give it a harmless communicate()/returncode so the
        # real-K8s-reading code paths exercised when other pages render just
        # see "no data" instead of crashing on an unconfigured MagicMock.
        m_popen.return_value.communicate.return_value = ("", "")
        m_popen.return_value.returncode = 1
        m_popen.return_value.__enter__.return_value = m_popen.return_value
        yield store_path


def _drive_start(at, controller_label: str, profile_label: str):
    at.sidebar.radio[0].set_value("Experiment").run()
    at.segmented_control(key="du_data_source_widget").set_value("LIVE").run()
    at.radio(key="du_controller_widget").set_value(controller_label).run()
    at.selectbox(key="du_profile_widget").set_value(profile_label).run()
    start_btn = [b for b in at.button if b.label == "START"][0]
    start_btn.click().run()
    assert not at.exception, at.exception


class TestExperimentIsControlPlane:
    def test_start_creates_store_record_and_live_details_follows_it(self, mocked_subprocess):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        _drive_start(at, "HPA", "Constant")

        active = experiment_store.active_run(mocked_subprocess)
        assert active is not None
        assert active["controller"] == "HPA"
        assert active["profile_name"] == "constant"
        run_id = active["run_id"]

        # Navigate to Live Details: must show the SAME run, no new process.
        at.sidebar.radio[0].set_value("Live Details").run()
        assert not at.exception, at.exception
        html = "".join(m.value for m in at.markdown)
        assert run_id in html
        assert "HPA" in html
        assert "Constant" in html

        # No INDEPENDENT controller/traffic/data-source selector on Live
        # Details: the fixture dashboard's own selectboxes aren't even
        # instantiated until explicit opt-in (see test_fixture_lazy_opt_in.py
        # / test_live_details_fresh_state.py), so none exist here at all --
        # never Experiment's own widget keys either.
        assert len(at.segmented_control) == 0
        assert len(at.selectbox) == 0

    def test_navigation_back_to_experiment_preserves_run(self, mocked_subprocess):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        _drive_start(at, "HPA", "Constant")
        run_id = experiment_store.active_run(mocked_subprocess)["run_id"]

        at.sidebar.radio[0].set_value("Live Details").run()
        at.sidebar.radio[0].set_value("Experiment").run()
        assert not at.exception, at.exception
        assert at.session_state["du_data_source"] == "LIVE"
        html = "".join(m.value for m in at.markdown)
        assert run_id in html

    def test_history_gains_record_only_after_completion(self, mocked_subprocess):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        _drive_start(at, "HPA", "Constant")

        at.sidebar.radio[0].set_value("History").run()
        assert not at.exception, at.exception
        assert "No LIVE experiment history yet" in "".join(m.value for m in at.markdown)

        run_id = experiment_store.active_run(mocked_subprocess)["run_id"]
        experiment_store.update_run(
            run_id, status="completed",
            workload_hash="h1", workload_seed=42, workload_duration_s=600.0,
            summary={"mean_replicas": 3.0, "mean_latency_p95_ms": 120.0, "error_rate_rps": 0.0},
            path=mocked_subprocess,
        )
        at.sidebar.radio[0].set_value("History").run()
        assert not at.exception, at.exception
        html = "".join(m.value for m in at.markdown)
        assert run_id in html
        assert "COMPLETED" in html

    def test_comparison_reports_missing_counterpart_then_pairs_after_second_run(self, mocked_subprocess):
        hpa_id = experiment_store.start_run("HPA", "constant", path=mocked_subprocess)
        experiment_store.update_run(
            hpa_id, status="completed", workload_hash="h1", workload_seed=42,
            workload_duration_s=600.0, summary={"mean_replicas": 3.0}, path=mocked_subprocess,
        )

        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Comparison").run()
        assert not at.exception, at.exception
        text = "".join(m.value for m in at.markdown) + "".join(c.value for c in at.caption)
        assert hpa_id in text
        assert "HPA result available" in text
        assert "Matching PPO run not available yet." in text

        ppo_id = experiment_store.start_run("PPO", "constant", path=mocked_subprocess)
        experiment_store.update_run(
            ppo_id, status="completed", workload_hash="h1", workload_seed=42,
            workload_duration_s=600.0, summary={"mean_replicas": 1.0}, path=mocked_subprocess,
        )
        at.sidebar.radio[0].set_value("Comparison").run()
        assert not at.exception, at.exception
        text = "".join(m.value for m in at.markdown) + "".join(c.value for c in at.caption)
        assert hpa_id in text and ppo_id in text
        assert "not available yet" not in text

    def test_comparison_does_not_reuse_incompatible_prior_workload(self, mocked_subprocess):
        old_ppo = experiment_store.start_run("PPO", "ramp", path=mocked_subprocess)
        experiment_store.update_run(
            old_ppo, status="completed", workload_hash="hRamp", workload_seed=9,
            workload_duration_s=60.0, summary={}, path=mocked_subprocess,
        )
        new_hpa = experiment_store.start_run("HPA", "constant", path=mocked_subprocess)
        experiment_store.update_run(
            new_hpa, status="completed", workload_hash="hConst", workload_seed=42,
            workload_duration_s=600.0, summary={}, path=mocked_subprocess,
        )

        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Comparison").run()
        assert not at.exception, at.exception
        text = "".join(m.value for m in at.markdown) + "".join(c.value for c in at.caption)
        assert old_ppo not in text  # the ramp PPO run must never appear paired with the constant HPA run
        assert "Matching PPO run not available yet." in text

    def test_no_duplicate_start_on_plain_rerun(self, mocked_subprocess):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        _drive_start(at, "HPA", "Constant")
        assert len(experiment_store._load(mocked_subprocess)) == 1
        at.run()
        assert not at.exception, at.exception
        assert len(experiment_store._load(mocked_subprocess)) == 1

    def test_fixture_history_never_labeled_as_live(self, mocked_subprocess):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("History").run()
        assert not at.exception, at.exception
        labels = [e.label for e in at.expander]
        assert any("Demo fixture data" in label and "not real LIVE results" in label for label in labels)
        assert "No LIVE experiment history yet" in "".join(m.value for m in at.markdown)
