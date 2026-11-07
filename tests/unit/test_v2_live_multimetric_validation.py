"""tests/unit/test_v2_live_multimetric_validation.py

V2 Phase 4D-D1: fully offline tests for
scripts/v2_live_multimetric_validation.py -- no real cluster, no real
Prometheus, no real subprocess. Every kubectl/HTTP/Prometheus dependency
is injected as a fake. Exercises: profile resolution (the new, firewall-
free v2_validation namespace), the pure safety/parsing/sample-row helper
functions, and the `ValidationRunner`'s scenario/lock/cleanup orchestration
end-to-end against fakes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pytest

import scripts.v2_live_multimetric_validation as mod
from autoscaler.src.rl_env.controller_exclusivity import (
    CONTROLLER_NONE,
    CONTROLLER_PPO,
    read_lock,
)
from autoscaler.src.rl_env.observation import ObservationBuildResult
from autoscaler.src.rl_env.types import ReplicaState

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# -- profile resolution --------------------------------------------------


class TestProfileResolution:
    def test_resolves_real_validation_profiles(self):
        for name in ("constant_cpu", "constant_memory", "constant_mixed"):
            meta = mod.resolve_validation_profile_metadata(name)
            assert meta.name == name
            assert meta.duration_seconds == 300.0

    def test_raises_on_unknown_profile(self):
        with pytest.raises(FileNotFoundError):
            mod.resolve_validation_profile_metadata("does_not_exist")

    def test_validation_profiles_are_a_separate_namespace_from_training(self):
        """Never reused from, or registered in, either firewalled registry."""
        assert mod.VALIDATION_PROFILE_DIR.name == "v2_validation"
        assert mod.VALIDATION_PROFILE_DIR != PROJECT_ROOT / "configs" / "traffic_profiles" / "v2"


# -- pure helpers ----------------------------------------------------------


class TestEvaluateSafetyAbort:
    def test_no_abort_when_healthy(self):
        since, reason = mod.evaluate_safety_abort(2, None, 100.0, 50.0)
        assert since is None
        assert reason is None

    def test_ready_zero_not_yet_sustained_does_not_abort(self):
        since, reason = mod.evaluate_safety_abort(0, None, 100.0, 50.0)
        assert since == 100.0
        assert reason is None

    def test_ready_zero_sustained_past_threshold_aborts(self):
        since, reason = mod.evaluate_safety_abort(0, 100.0, 161.0, 50.0, ready_zero_abort_s=60.0)
        assert reason is not None
        assert "ready_replicas=0" in reason

    def test_ready_recovering_resets_the_zero_timer(self):
        since, reason = mod.evaluate_safety_abort(2, 100.0, 120.0, 50.0)
        assert since is None
        assert reason is None

    def test_memory_guard_triggers_abort(self):
        since, reason = mod.evaluate_safety_abort(2, None, 100.0, 231.0, memory_threshold_mib=230.0)
        assert reason is not None
        assert "memory" in reason

    def test_memory_exactly_at_threshold_triggers(self):
        _since, reason = mod.evaluate_safety_abort(2, None, 100.0, 230.0, memory_threshold_mib=230.0)
        assert reason is not None

    def test_memory_none_never_aborts(self):
        _since, reason = mod.evaluate_safety_abort(2, None, 100.0, None)
        assert reason is None


class TestParseHpaV2Status:
    def test_none_input_returns_none(self):
        assert mod.parse_hpa_v2_status(None) is None

    def test_parses_real_shaped_json(self):
        hpa_json = {
            "status": {
                "currentMetrics": [
                    {"resource": {"name": "cpu", "current": {"averageUtilization": 7}}, "type": "Resource"},
                    {"resource": {"name": "memory", "current": {"averageUtilization": 78}}, "type": "Resource"},
                ],
                "currentReplicas": 2,
                "desiredReplicas": 2,
            },
        }
        parsed = mod.parse_hpa_v2_status(hpa_json)
        assert parsed == {
            "cpu_current_pct": 7, "memory_current_pct": 78, "current_replicas": 2, "desired_replicas": 2,
        }

    def test_missing_fields_become_none_not_an_exception(self):
        parsed = mod.parse_hpa_v2_status({"status": {}})
        assert parsed == {
            "cpu_current_pct": None, "memory_current_pct": None, "current_replicas": None, "desired_replicas": None,
        }


class TestBuildSampleRow:
    def test_shape(self):
        obs_result = ObservationBuildResult(
            normalized=[0.5] * 11, raw={"cpu_per_replica": 1.0}, validity=_FakeValidity(True, "observed"),
            ready_replicas_used=2,
        )
        row = mod.build_sample_row(
            scenario="hpa_cpu", timestamp_utc="t", elapsed_in_scenario_s=30.4, obs_result=obs_result,
            replicas_desired=2, replicas_ready=2, hpa_status={"cpu_current_pct": 10, "memory_current_pct": 50,
            "current_replicas": 2, "desired_replicas": 2}, pod_restart_counts={"pod-a": 0}, pod_request_rates=None,
        )
        assert row["scenario"] == "hpa_cpu"
        assert row["normalized_observation"] == [0.5] * 11
        assert row["hpa_v2_present"] is True
        assert row["hpa_v2_cpu_current_pct"] == 10
        assert row["deployment_ready_replicas"] == 2
        assert row["pod_restart_counts"] == {"pod-a": 0}

    def test_hpa_absent_reports_none_fields(self):
        obs_result = ObservationBuildResult(
            normalized=[0.5] * 11, raw={}, validity=_FakeValidity(True, "observed"), ready_replicas_used=2,
        )
        row = mod.build_sample_row(
            scenario="fixed2_cpu", timestamp_utc="t", elapsed_in_scenario_s=0.0, obs_result=obs_result,
            replicas_desired=2, replicas_ready=2, hpa_status=None, pod_restart_counts={}, pod_request_rates=None,
        )
        assert row["hpa_v2_present"] is False
        assert row["hpa_v2_cpu_current_pct"] is None


@dataclass(frozen=True)
class _FakeValidity:
    sample_valid: bool
    latency_observation_state: str


# -- fakes for the ValidationRunner ------------------------------------------


class _FakeKubectl:
    def __init__(self, existing_hpas: Optional[List[str]] = None):
        self._existing_hpas = list(existing_hpas or ["demo-app-hpa-v2"])
        self.delete_hpa_calls = 0
        self.restore_hpa_calls = 0
        self.scale_calls: List[int] = []
        self._desired = 2
        self._ready = 2

    def get_replica_state(self):
        return ReplicaState(desired=self._desired, current=self._desired, ready=self._ready, available=self._ready)

    def scale(self, replicas: int) -> None:
        self.scale_calls.append(replicas)
        self._desired = replicas
        self._ready = replicas

    def wait_for_replica_state(self, target, timeout):
        self._ready = target
        return self.get_replica_state()

    def list_hpa_names(self):
        return list(self._existing_hpas)

    def delete_hpa(self):
        self.delete_hpa_calls += 1
        self._existing_hpas = [h for h in self._existing_hpas if h != "demo-app-hpa-v2"]

    def restore_hpa(self):
        self.restore_hpa_calls += 1
        if "demo-app-hpa-v2" not in self._existing_hpas:
            self._existing_hpas.append("demo-app-hpa-v2")


class _StuckAtZeroKubectl(_FakeKubectl):
    """A deployment that never reaches Ready -- scale()/wait_for_replica_state()
    are no-ops so `_ensure_2_ready()`'s own pre-scenario reset cannot mask
    the stuck condition this test exercises."""

    def __init__(self):
        super().__init__()
        self._ready = 0

    def scale(self, replicas: int) -> None:
        self.scale_calls.append(replicas)

    def wait_for_replica_state(self, target, timeout):
        return self.get_replica_state()


class _FakeSnapshot:
    def __init__(self, ready: int = 2, desired: int = 2):
        self.time_s = 0.0
        self.replicas = ReplicaState(desired=desired, current=desired, ready=ready, available=ready)
        from autoscaler.src.rl_env.types import RawMetricSnapshot

        self.metrics = RawMetricSnapshot(
            request_rate_rps=10.0, error_rate_rps=0.0, latency_p95_ms=50.0,
            requests_in_progress=1.0, cpu_cores=0.2, memory_bytes=40_000_000.0,
            target_up=True, query_error=False,
        )


class _FakeBackend:
    def __init__(self, kubectl: Optional[_FakeKubectl] = None):
        self._kubectl = kubectl or _FakeKubectl()
        self.set_desired_replicas_calls: List[int] = []

    def current_snapshot(self):
        return _FakeSnapshot(ready=self._kubectl._ready, desired=self._kubectl._desired)

    def set_desired_replicas(self, desired: int) -> None:
        self.set_desired_replicas_calls.append(desired)
        self._kubectl.scale(desired)


class _FakeOrchestrator:
    def __init__(self):
        self.started: List[str] = []
        self.stopped: List[str] = []
        self.force_cleanup_calls = 0

    def start_episode_traffic(self, profile_name, episode_id):
        self.started.append(profile_name)
        return object()

    def stop_episode_traffic(self, handle, episode_id):
        self.stopped.append(episode_id)

    def force_cleanup(self):
        self.force_cleanup_calls += 1


class _FakeCompletedProcess:
    def __init__(self, returncode=0, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


def _make_runner(tmp_path, monkeypatch, scenarios=None, kubectl=None):
    fake_kubectl = kubectl or _FakeKubectl()
    fake_backend = _FakeBackend(kubectl=fake_kubectl)
    fake_orch = _FakeOrchestrator()

    def fake_kubectl_run(*a, **k):
        if "demo-app-hpa-v2" in fake_kubectl.list_hpa_names():
            body = {
                "status": {
                    "currentMetrics": [
                        {"resource": {"name": "cpu", "current": {"averageUtilization": 10}}, "type": "Resource"},
                        {"resource": {"name": "memory", "current": {"averageUtilization": 50}}, "type": "Resource"},
                    ],
                    "currentReplicas": fake_kubectl._ready, "desiredReplicas": fake_kubectl._desired,
                },
            }
            return _FakeCompletedProcess(returncode=0, stdout=json.dumps(body))
        return _FakeCompletedProcess(returncode=1, stdout="")

    monkeypatch.setattr(mod, "get_pod_health", lambda *a, **k: [])
    monkeypatch.setattr(mod, "get_max_per_pod_memory_mib", lambda *a, **k: 50.0)
    monkeypatch.setattr(mod, "query_per_pod_request_rate", lambda *a, **k: None)

    # Reset the real controller lock file to CONTROLLER_NONE before each test
    # (these tests exercise the REAL acquire_controller/release_controller
    # functions against a FAKE kubectl, so the real default lock path is
    # used -- isolate it per-test via monkeypatch of the lock path default
    # would be cleaner, but acquire/release already take no path override
    # in this module's call sites, so we instead ensure a clean starting
    # state and restore it afterward via the module's own release calls).
    from autoscaler.src.rl_env import controller_exclusivity as ce

    ce.clear_lock()

    runner = mod.ValidationRunner(
        output_dir=tmp_path / "run",
        kubectl_run=fake_kubectl_run,
        sleep_fn=lambda s: None,
        clock=_clock(),
        scenarios=scenarios,
        backend=fake_backend,
        orchestrator=fake_orch,
    )
    return runner, fake_backend, fake_kubectl, fake_orch


def _clock():
    state = {"t": 0.0}

    def _tick():
        state["t"] += 1.0
        return state["t"]

    return _tick


class TestValidationRunnerScenarios:
    def test_idle_scenario_takes_no_lock_and_leaves_hpa_present(self, tmp_path, monkeypatch):
        scenarios = [{"name": "idle_baseline", "kind": "idle", "profile": None, "duration_s": 2.0}]
        runner, backend, kubectl, orch = _make_runner(tmp_path, monkeypatch, scenarios=scenarios)
        result = runner.run_scenario(scenarios[0])
        assert not result.aborted
        assert "demo-app-hpa-v2" in kubectl.list_hpa_names()
        assert read_lock() == CONTROLLER_NONE
        assert orch.started == []  # no traffic for idle

    def test_fixed2_scenario_removes_hpa_and_holds_ppo_lock_then_restores(self, tmp_path, monkeypatch):
        scenarios = [{"name": "fixed2_cpu", "kind": "fixed2", "profile": "constant_cpu", "duration_s": 2.0}]
        runner, backend, kubectl, orch = _make_runner(tmp_path, monkeypatch, scenarios=scenarios)
        result = runner.run_scenario(scenarios[0])
        assert not result.aborted
        assert kubectl.delete_hpa_calls == 1
        assert kubectl.restore_hpa_calls == 1  # restored in the scenario's own finally
        assert "demo-app-hpa-v2" in kubectl.list_hpa_names()  # back, after cleanup
        assert read_lock() == CONTROLLER_NONE  # released
        assert orch.started == ["constant_cpu"]

    def test_hpa_scenario_acquires_cpu_memory_hpa_lock(self, tmp_path, monkeypatch):
        scenarios = [{"name": "hpa_cpu", "kind": "hpa", "profile": "constant_cpu", "duration_s": 2.0}]
        runner, backend, kubectl, orch = _make_runner(tmp_path, monkeypatch, scenarios=scenarios)
        result = runner.run_scenario(scenarios[0])
        assert not result.aborted
        assert kubectl.restore_hpa_calls >= 1
        assert read_lock() == CONTROLLER_NONE  # released after scenario
        assert orch.started == ["constant_cpu"]

    def test_fixed2_never_aborts_on_restarts_only_on_ready_zero_or_memory(self, tmp_path, monkeypatch):
        """Restarts are the measurement in S2 -- confirmed by never passing
        restart counts into evaluate_safety_abort at all (only
        ready-replica count and max per-pod memory do)."""
        scenarios = [{"name": "fixed2_cpu", "kind": "fixed2", "profile": "constant_cpu", "duration_s": 2.0}]
        runner, backend, kubectl, orch = _make_runner(tmp_path, monkeypatch, scenarios=scenarios)
        monkeypatch.setattr(
            mod, "get_pod_health",
            lambda *a, **k: [_PodHealthStub("p1", restart_count=99)],
        )
        result = runner.run_scenario(scenarios[0])
        assert not result.aborted
        assert result.restart_deltas  # restarts ARE recorded, just never cause an abort

    def test_scenario_aborts_on_sustained_zero_ready(self, tmp_path, monkeypatch):
        """Simulates a deployment stuck at 0 Ready DESPITE the pre-scenario
        reset attempting to scale/wait it to 2 -- `scale()`/
        `wait_for_replica_state()` are no-ops here so `_ensure_2_ready()`
        cannot silently "fix" the stuck state before sampling begins."""
        scenarios = [{"name": "hpa_cpu", "kind": "hpa", "profile": "constant_cpu", "duration_s": 400.0}]
        kubectl = _StuckAtZeroKubectl()
        runner, backend, _kubectl, orch = _make_runner(tmp_path, monkeypatch, scenarios=scenarios, kubectl=kubectl)
        # clock advances 1.0s per call; sample_interval default 30s means many
        # samples before 60s elapses -- use a short runner-level sample interval.
        runner.sample_interval_s = 10.0
        result = runner.run_scenario(scenarios[0])
        assert result.aborted
        assert "ready_replicas=0" in result.abort_reason
        assert read_lock() == CONTROLLER_NONE  # still released even on abort

    def test_scenario_aborts_on_memory_guard(self, tmp_path, monkeypatch):
        scenarios = [{"name": "hpa_cpu", "kind": "hpa", "profile": "constant_cpu", "duration_s": 400.0}]
        runner, backend, kubectl, orch = _make_runner(tmp_path, monkeypatch, scenarios=scenarios)
        monkeypatch.setattr(mod, "get_max_per_pod_memory_mib", lambda *a, **k: 300.0)
        result = runner.run_scenario(scenarios[0])
        assert result.aborted
        assert "memory" in result.abort_reason

    def test_samples_written_to_jsonl(self, tmp_path, monkeypatch):
        scenarios = [{"name": "hpa_cpu", "kind": "hpa", "profile": "constant_cpu", "duration_s": 2.0}]
        runner, backend, kubectl, orch = _make_runner(tmp_path, monkeypatch, scenarios=scenarios)
        result = runner.run_scenario(scenarios[0])
        assert result.n_samples >= 1
        lines = runner.samples_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == result.n_samples
        row = json.loads(lines[0])
        assert row["scenario"] == "hpa_cpu"
        assert len(row["normalized_observation"]) == 11


class _PodHealthStub:
    def __init__(self, name, restart_count=0):
        self.name = name
        self.restart_count = restart_count


class TestValidationRunnerFullRun:
    def test_run_completes_and_writes_status_and_summary(self, tmp_path, monkeypatch):
        scenarios = [
            {"name": "idle_baseline", "kind": "idle", "profile": None, "duration_s": 1.0},
            {"name": "fixed2_cpu", "kind": "fixed2", "profile": "constant_cpu", "duration_s": 1.0},
        ]
        runner, backend, kubectl, orch = _make_runner(tmp_path, monkeypatch, scenarios=scenarios)
        runner.idle_settle_s = 0.0
        summary = runner.run()
        status = json.loads(runner.status_path.read_text(encoding="utf-8"))
        assert status["status"] == "COMPLETE"
        assert status["scenarios_completed"] == ["idle_baseline", "fixed2_cpu"]
        assert len(summary["results"]) == 2
        assert runner.summary_path.exists()
        # final cleanup always restores the HPA and 2 ready replicas, lock none
        assert "demo-app-hpa-v2" in kubectl.list_hpa_names()
        assert read_lock() == CONTROLLER_NONE

    def test_run_marks_failed_on_exception_but_still_cleans_up(self, tmp_path, monkeypatch):
        scenarios = [{"name": "hpa_cpu", "kind": "hpa", "profile": "constant_cpu", "duration_s": 1.0}]
        runner, backend, kubectl, orch = _make_runner(tmp_path, monkeypatch, scenarios=scenarios)
        runner.idle_settle_s = 0.0

        def _boom(*a, **k):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(runner, "run_scenario", _boom)
        summary = runner.run()
        status = json.loads(runner.status_path.read_text(encoding="utf-8"))
        assert status["status"] == "FAILED"
        assert "simulated failure" in status["reason"]
        assert summary["results"] == []
        # cleanup still ran
        assert "demo-app-hpa-v2" in kubectl.list_hpa_names()
        assert read_lock() == CONTROLLER_NONE

    def test_final_cleanup_restores_hpa_replicas_and_lock_even_if_a_step_raises(self, tmp_path, monkeypatch):
        runner, backend, kubectl, orch = _make_runner(tmp_path, monkeypatch, scenarios=[])
        kubectl.delete_hpa()  # simulate HPA absent going into cleanup
        acquire_lock_for_test = __import__(
            "autoscaler.src.rl_env.controller_exclusivity", fromlist=["acquire_controller"],
        )
        acquire_lock_for_test.acquire_controller(CONTROLLER_PPO, kubectl)
        runner._final_cleanup()
        assert "demo-app-hpa-v2" in kubectl.list_hpa_names()
        assert read_lock() == CONTROLLER_NONE
        assert orch.force_cleanup_calls == 1


def test_scenarios_constant_matches_task_spec():
    names = [s["name"] for s in mod.SCENARIOS]
    assert names == ["idle_baseline", "fixed2_cpu", "hpa_cpu", "hpa_memory", "hpa_mixed"]
    assert mod.SCENARIOS[0]["duration_s"] == 120.0
    for s in mod.SCENARIOS[1:]:
        assert s["duration_s"] == 300.0
    assert mod.SCENARIOS[1]["kind"] == "fixed2"
    for s in mod.SCENARIOS[2:]:
        assert s["kind"] == "hpa"
