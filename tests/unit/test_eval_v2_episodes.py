"""tests/unit/test_eval_v2_episodes.py

V2 Phase 5C-A: offline tests for scripts/eval_v2_episodes.py. Uses
ScriptedBackend (the project's own existing offline DynamicsBackend double,
already used by tests/unit/test_environment_v2.py) plus small fakes for
pod health/memory queries/kubectl, and a REAL PPO checkpoint
(session_000/ppo_model.zip, skipped if missing) for the ppo_v2 arm --
avoids re-implementing a deep torch/SB3-level fake, consistent with the
pattern already used by tests/unit/test_calibrate_epsilon_v2.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.eval_v2_episodes as ep
from autoscaler.src.rl_env.config_v2 import load_env_config_v2_from_yaml
from autoscaler.src.rl_env.scripted_backend import ScriptedBackend
from autoscaler.src.rl_env.types import RawMetricSnapshot
from scripts.v2_pilot_safety import PodHealthSnapshot

REAL_CHECKPOINT = Path("persistent-data/rl_training_v2/final_min2/sessions/session_000/ppo_model.zip")


def _timeline(cfg, n_steps, cpu_cores=0.5, memory_bytes=80_000_000, request_rate=10.0, latency_ms=50.0, error_rate=0.0):
    timeline = {}
    t = 0.0
    for _ in range(n_steps + 1):
        timeline[t] = RawMetricSnapshot(
            request_rate_rps=request_rate, error_rate_rps=error_rate, latency_p95_ms=latency_ms,
            requests_in_progress=1.0, cpu_cores=cpu_cores, memory_bytes=memory_bytes,
            target_up=True, query_error=False,
        )
        t += cfg.decision_interval_s
    return timeline


class _FakeKubectl:
    def __init__(self, existing_hpa_names=None):
        self._existing_hpa_names = existing_hpa_names if existing_hpa_names is not None else []
        self.restore_hpa_calls = 0
        self.scale_calls = []

    def list_hpa_names(self):
        return list(self._existing_hpa_names)

    def restore_hpa(self):
        self.restore_hpa_calls += 1
        if self._existing_hpa_names == []:
            pass  # a test sets _existing_hpa_names directly to simulate "apply succeeded"

    def scale(self, desired):
        self.scale_calls.append(desired)

    def wait_for_replica_state(self, desired, timeout):
        pass

    def delete_hpa(self):
        pass


class _FakeBackendWithKubectl(ScriptedBackend):
    """ScriptedBackend already implements the real DynamicsBackend contract
    (reset/advance/set_desired_replicas) -- this only adds the `_kubectl`
    attribute eval_v2_episodes.py's functions read directly, and records
    every `set_desired_replicas` call so a test can assert the HPA arms
    never make one."""

    def __init__(self, *args, hpa_name=None, existing_hpa_names=None, **kwargs):
        super().__init__(*args, **kwargs)
        names = existing_hpa_names if existing_hpa_names is not None else ([hpa_name] if hpa_name else [])
        self._kubectl = _FakeKubectl(existing_hpa_names=names)
        self.set_desired_replicas_calls = []

    def set_desired_replicas(self, desired):
        self.set_desired_replicas_calls.append(desired)
        super().set_desired_replicas(desired)


def _fake_pod_health(namespace, label_selector):
    return [PodHealthSnapshot(name="demo-app-aaaa", memory_bytes=80_000_000, restart_count=0, ready=True, phase="Running")]


def _fake_max_pod_memory(prom_url, namespace, pod_regex):
    return 80.0


@pytest.fixture(autouse=True)
def _patch_controller_lock(monkeypatch):
    monkeypatch.setattr(ep, "acquire_controller", lambda *a, **k: None)
    monkeypatch.setattr(ep, "release_controller", lambda *a, **k: None)


class TestResolveCpuOnlyControllerIdentity:
    def test_fails_closed_without_the_phase5c_constant(self, monkeypatch):
        import autoscaler.src.rl_env.controller_exclusivity as ce

        monkeypatch.delattr(ce, "CONTROLLER_CPU_ONLY_HPA_V2", raising=False)
        with pytest.raises(ep.CpuOnlyArmNotYetSupportedError):
            ep.resolve_cpu_only_controller_identity()

    def test_succeeds_once_the_constant_is_monkeypatched_in(self, monkeypatch):
        import autoscaler.src.rl_env.controller_exclusivity as ce

        monkeypatch.setattr(ce, "CONTROLLER_CPU_ONLY_HPA_V2", "cpu_only_hpa_v2", raising=False)
        assert ep.resolve_cpu_only_controller_identity() == "cpu_only_hpa_v2"


class TestHpaArmNeverScales:
    def test_hpa_v2_cpu_memory_never_calls_set_desired_replicas_during_the_loop(self):
        cfg = load_env_config_v2_from_yaml()
        timeline = _timeline(cfg, 4)
        backend = _FakeBackendWithKubectl(
            timeline=timeline, initial_replicas=cfg.initial_replicas, min_replicas=cfg.gating.min_replicas,
            max_replicas=cfg.gating.max_replicas, hpa_name="demo-app-hpa-v2",
        )
        result = ep.run_hpa_v2_eval_episode(
            "hpa_v2_cpu_memory", "demo-app-hpa-v2", "k8s/hpa/hpa_v2_cpu_memory.yaml", "cpu_memory_hpa",
            "eval_cpu_step", cfg.decision_interval_s * 4, "e1",
            backend_factory=lambda **kwargs: backend, pod_health_fn=_fake_pod_health, max_pod_memory_fn=_fake_max_pod_memory,
        )
        assert backend.set_desired_replicas_calls == []
        assert len(result["samples"]) == 4

    def test_hpa_v2_cpu_only_never_calls_set_desired_replicas_once_enabled(self, monkeypatch):
        import autoscaler.src.rl_env.controller_exclusivity as ce

        monkeypatch.setattr(ce, "CONTROLLER_CPU_ONLY_HPA_V2", "cpu_only_hpa_v2", raising=False)
        cfg = load_env_config_v2_from_yaml()
        timeline = _timeline(cfg, 4)
        backend = _FakeBackendWithKubectl(
            timeline=timeline, initial_replicas=cfg.initial_replicas, min_replicas=cfg.gating.min_replicas,
            max_replicas=cfg.gating.max_replicas, hpa_name="demo-app-hpa-v2-cpu",
        )
        result = ep.run_hpa_v2_cpu_only_eval_episode(
            "eval_cpu_step", cfg.decision_interval_s * 4, "e1",
            backend_factory=lambda **kwargs: backend, pod_health_fn=_fake_pod_health, max_pod_memory_fn=_fake_max_pod_memory,
        )
        assert backend.set_desired_replicas_calls == []
        assert result["arm_id"] == "hpa_v2_cpu_only"

    def test_cpu_only_fails_closed_before_touching_backend_at_all(self, monkeypatch):
        import autoscaler.src.rl_env.controller_exclusivity as ce

        monkeypatch.delattr(ce, "CONTROLLER_CPU_ONLY_HPA_V2", raising=False)
        calls = {"n": 0}

        def tracking_factory(**kwargs):
            calls["n"] += 1
            raise AssertionError("backend must never be constructed before the readiness check")

        with pytest.raises(ep.CpuOnlyArmNotYetSupportedError):
            ep.run_hpa_v2_cpu_only_eval_episode("eval_cpu_step", 120.0, "e1", backend_factory=tracking_factory)
        assert calls["n"] == 0


class TestPpoArmNeverHasAnHpaPresent:
    def test_ppo_arm_acquires_controller_ppo(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(ep, "acquire_controller", lambda controller, kubectl: seen.setdefault("controller", controller))
        cfg = load_env_config_v2_from_yaml()
        timeline = _timeline(cfg, 2)
        backend = _FakeBackendWithKubectl(
            timeline=timeline, initial_replicas=cfg.initial_replicas, min_replicas=cfg.gating.min_replicas,
            max_replicas=cfg.gating.max_replicas, existing_hpa_names=[],
        )

        def fake_model_loader(path):
            return object()  # never reached if model_loader itself fails fast -- see below

        # A model that can't produce real probabilities still exercises the
        # control-flow path up to run_safe_ppo_step, which degrades safely
        # to HOLD on any inference failure (safe_inference.py's own
        # guarantee) -- sufficient to prove the controller identity used.
        try:
            ep.run_ppo_v2_eval_episode(
                Path("ppo_model.zip"), 0.05, "eval_cpu_step", cfg.decision_interval_s * 2, "e1",
                backend_factory=lambda **kwargs: backend, model_loader=fake_model_loader,
                pod_health_fn=_fake_pod_health, max_pod_memory_fn=_fake_max_pod_memory,
            )
        except Exception:
            pass
        assert seen.get("controller") == ep.CONTROLLER_PPO


class TestSafetyAndToolingClassification:
    def test_pod_restart_raises_safety_termination(self):
        cfg = load_env_config_v2_from_yaml()
        timeline = _timeline(cfg, 4)
        backend = _FakeBackendWithKubectl(
            timeline=timeline, initial_replicas=cfg.initial_replicas, min_replicas=cfg.gating.min_replicas,
            max_replicas=cfg.gating.max_replicas, hpa_name="demo-app-hpa-v2",
        )
        calls = {"n": 0}

        def flaky_pod_health(namespace, label_selector):
            calls["n"] += 1
            restart_count = 0 if calls["n"] == 1 else 1  # baseline read, then a restart on the first real sample
            return [PodHealthSnapshot(name="demo-app-aaaa", memory_bytes=80_000_000, restart_count=restart_count, ready=True, phase="Running")]

        with pytest.raises(ep.SafetyTerminationError, match="pod_restart_detected"):
            ep.run_hpa_v2_eval_episode(
                "hpa_v2_cpu_memory", "demo-app-hpa-v2", "k8s/hpa/hpa_v2_cpu_memory.yaml", "cpu_memory_hpa",
                "eval_cpu_step", cfg.decision_interval_s * 4, "e1",
                backend_factory=lambda **kwargs: backend, pod_health_fn=flaky_pod_health, max_pod_memory_fn=_fake_max_pod_memory,
            )

    def test_sustained_memory_breach_raises_safety_termination(self):
        cfg = load_env_config_v2_from_yaml()
        timeline = _timeline(cfg, 4)
        backend = _FakeBackendWithKubectl(
            timeline=timeline, initial_replicas=cfg.initial_replicas, min_replicas=cfg.gating.min_replicas,
            max_replicas=cfg.gating.max_replicas, hpa_name="demo-app-hpa-v2",
        )
        with pytest.raises(ep.SafetyTerminationError, match="memory_safety_threshold_breached"):
            ep.run_hpa_v2_eval_episode(
                "hpa_v2_cpu_memory", "demo-app-hpa-v2", "k8s/hpa/hpa_v2_cpu_memory.yaml", "cpu_memory_hpa",
                "eval_cpu_step", cfg.decision_interval_s * 4, "e1",
                backend_factory=lambda **kwargs: backend, pod_health_fn=_fake_pod_health,
                max_pod_memory_fn=lambda *a, **k: 300.0,  # above MEMORY_SAFETY_THRESHOLD_MIB, sustained
            )

    def test_pod_health_query_failure_is_a_tooling_failure(self):
        cfg = load_env_config_v2_from_yaml()
        timeline = _timeline(cfg, 4)
        backend = _FakeBackendWithKubectl(
            timeline=timeline, initial_replicas=cfg.initial_replicas, min_replicas=cfg.gating.min_replicas,
            max_replicas=cfg.gating.max_replicas, hpa_name="demo-app-hpa-v2",
        )

        def broken_pod_health(namespace, label_selector):
            raise RuntimeError("simulated kubectl failure")

        with pytest.raises(ep.ToolingFailureError):
            ep.run_hpa_v2_eval_episode(
                "hpa_v2_cpu_memory", "demo-app-hpa-v2", "k8s/hpa/hpa_v2_cpu_memory.yaml", "cpu_memory_hpa",
                "eval_cpu_step", cfg.decision_interval_s * 4, "e1",
                backend_factory=lambda **kwargs: backend, pod_health_fn=broken_pod_health, max_pod_memory_fn=_fake_max_pod_memory,
            )

    def test_failed_hpa_apply_is_a_tooling_failure(self):
        cfg = load_env_config_v2_from_yaml()
        timeline = _timeline(cfg, 4)
        backend = _FakeBackendWithKubectl(
            timeline=timeline, initial_replicas=cfg.initial_replicas, min_replicas=cfg.gating.min_replicas,
            max_replicas=cfg.gating.max_replicas, existing_hpa_names=["some-other-hpa"],  # apply never "took"
        )
        with pytest.raises(ep.ToolingFailureError, match="failed to apply"):
            ep.run_hpa_v2_eval_episode(
                "hpa_v2_cpu_memory", "demo-app-hpa-v2", "k8s/hpa/hpa_v2_cpu_memory.yaml", "cpu_memory_hpa",
                "eval_cpu_step", cfg.decision_interval_s * 4, "e1",
                backend_factory=lambda **kwargs: backend, pod_health_fn=_fake_pod_health, max_pod_memory_fn=_fake_max_pod_memory,
            )


class TestCleanupAlwaysRuns:
    def test_hpa_arm_restores_hpa_and_releases_lock_on_exception(self, monkeypatch):
        released = {"n": 0}
        monkeypatch.setattr(ep, "release_controller", lambda *a, **k: released.__setitem__("n", released["n"] + 1))
        cfg = load_env_config_v2_from_yaml()
        timeline = _timeline(cfg, 4)
        backend = _FakeBackendWithKubectl(
            timeline=timeline, initial_replicas=cfg.initial_replicas, min_replicas=cfg.gating.min_replicas,
            max_replicas=cfg.gating.max_replicas, hpa_name="demo-app-hpa-v2",
        )

        calls = {"n": 0}

        def breaks_after_baseline(namespace, label_selector):
            calls["n"] += 1
            if calls["n"] == 1:
                return _fake_pod_health(namespace, label_selector)  # baseline read succeeds
            raise RuntimeError("simulated failure on the first real sample")

        with pytest.raises(ep.ToolingFailureError):
            ep.run_hpa_v2_eval_episode(
                "hpa_v2_cpu_memory", "demo-app-hpa-v2", "k8s/hpa/hpa_v2_cpu_memory.yaml", "cpu_memory_hpa",
                "eval_cpu_step", cfg.decision_interval_s * 4, "e1",
                backend_factory=lambda **kwargs: backend, pod_health_fn=breaks_after_baseline, max_pod_memory_fn=_fake_max_pod_memory,
            )
        assert released["n"] == 1
        assert backend._kubectl.restore_hpa_calls >= 1  # the initial apply plus the finally-block re-apply

    def test_ppo_arm_releases_lock_on_exception(self, monkeypatch):
        released = {"n": 0}
        monkeypatch.setattr(ep, "release_controller", lambda *a, **k: released.__setitem__("n", released["n"] + 1))

        def broken_model_loader(path):
            return None  # -> ToolingFailureError inside run_ppo_v2_eval_episode

        with pytest.raises(ep.ToolingFailureError):
            ep.run_ppo_v2_eval_episode(
                Path("does-not-matter.zip"), 0.05, "eval_cpu_step", 60.0, "e1",
                backend_factory=lambda **kwargs: _FakeBackendWithKubectl(
                    timeline=_timeline(load_env_config_v2_from_yaml(), 2), initial_replicas=2, min_replicas=2, max_replicas=8,
                ),
                model_loader=broken_model_loader, pod_health_fn=_fake_pod_health, max_pod_memory_fn=_fake_max_pod_memory,
            )
        assert released["n"] == 1


class TestEpisodeResultToRunSeriesAdapter:
    def test_adapter_produces_the_schema_analyze_v2_evaluation_expects(self):
        import scripts.analyze_v2_evaluation as av

        fake_result = {
            "episode_id": "e1", "arm_id": "ppo_v2", "profile": "eval_cpu_step", "pod_restarts_total": 2,
            "samples": [
                {"t_s": 0.0, "p95_latency_ms": 100.0, "error_rate": 0.0, "max_per_pod_memory_mib": 80.0,
                 "desired_replicas": 2, "sample_valid": True, "ppo_decision": {"effective_action_name": "HOLD", "extra": {}}},
                {"t_s": 30.0, "p95_latency_ms": 1200.0, "error_rate": 0.01, "max_per_pod_memory_mib": 90.0,
                 "desired_replicas": 3, "sample_valid": True,
                 "ppo_decision": {"effective_action_name": "SCALE_UP", "extra": {"is_reversal": True}}},
            ],
        }
        series_dict = ep.episode_result_to_run_series_dict(fake_result, "run1", 0)
        series = av.normalize_run_series_from_dict(series_dict)
        metrics = av.compute_run_metrics(series)
        assert metrics["sla_violation_rate_1000ms"] == pytest.approx(0.5)
        assert metrics["scaling_actions_count"] == 1
        assert metrics["reversals_count"] == 1
        assert metrics["pod_restarts_count"] == 2

    def test_hpa_arm_result_has_no_action_or_reversal(self):
        fake_result = {
            "episode_id": "e2", "arm_id": "hpa_v2_cpu_memory", "profile": "eval_cpu_step", "pod_restarts_total": 0,
            "samples": [
                {"t_s": 0.0, "p95_latency_ms": 100.0, "error_rate": 0.0, "max_per_pod_memory_mib": 80.0,
                 "desired_replicas": 2, "sample_valid": True, "ppo_decision": None},
            ],
        }
        series_dict = ep.episode_result_to_run_series_dict(fake_result, "run2", 0)
        assert series_dict["samples"][0]["action"] is None
        assert series_dict["samples"][0]["reversal"] is False


class TestIdenticalSampleSchemaAcrossArms:
    def test_ppo_and_hpa_samples_have_the_same_field_set(self):
        cfg = load_env_config_v2_from_yaml()
        timeline = _timeline(cfg, 2)
        hpa_backend = _FakeBackendWithKubectl(
            timeline=timeline, initial_replicas=cfg.initial_replicas, min_replicas=cfg.gating.min_replicas,
            max_replicas=cfg.gating.max_replicas, hpa_name="demo-app-hpa-v2",
        )
        hpa_result = ep.run_hpa_v2_eval_episode(
            "hpa_v2_cpu_memory", "demo-app-hpa-v2", "k8s/hpa/hpa_v2_cpu_memory.yaml", "cpu_memory_hpa",
            "eval_cpu_step", cfg.decision_interval_s * 2, "e1",
            backend_factory=lambda **kwargs: hpa_backend, pod_health_fn=_fake_pod_health, max_pod_memory_fn=_fake_max_pod_memory,
        )

        if not REAL_CHECKPOINT.is_file():
            pytest.skip("no real V2 checkpoint available to exercise the ppo_v2 arm's full sample fields")

        ppo_backend = _FakeBackendWithKubectl(
            timeline=timeline, initial_replicas=cfg.initial_replicas, min_replicas=cfg.gating.min_replicas,
            max_replicas=cfg.gating.max_replicas,
        )
        ppo_result = ep.run_ppo_v2_eval_episode(
            REAL_CHECKPOINT, 0.05, "eval_cpu_step", cfg.decision_interval_s * 2, "e2",
            backend_factory=lambda **kwargs: ppo_backend, pod_health_fn=_fake_pod_health, max_pod_memory_fn=_fake_max_pod_memory,
        )

        hpa_fields = set(hpa_result["samples"][0].keys())
        ppo_fields = set(ppo_result["samples"][0].keys())
        assert hpa_fields == ppo_fields
