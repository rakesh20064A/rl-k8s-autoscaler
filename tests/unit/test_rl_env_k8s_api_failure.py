"""tests/unit/test_rl_env_k8s_api_failure.py

Phase 16V: tests the environment.py fix that makes `k8s_api_failure`
(docs/rl_mdp_spec.md sec.13's table row "Kubernetes API patch call fails")
actually reachable and correctly recorded. Prior to this phase no backend
could ever raise from `set_desired_replicas()`, so this path was
unreachable/untested dead code (a real backend that CAN fail did not yet
exist). Uses a small scripted-backend variant that fails exactly once on
command, no live Kubernetes involved.
"""

from __future__ import annotations

from autoscaler.src.rl_env.config import load_env_config_from_yaml
from autoscaler.src.rl_env.environment import KubernetesAutoscalerEnv
from autoscaler.src.rl_env.scripted_backend import ScriptedBackend
from autoscaler.src.rl_env.types import OVERRIDE_K8S_API_FAILURE, RawMetricSnapshot

ACTION_UP = 2
ACTION_HOLD = 1
CFG = load_env_config_from_yaml()


def _valid_snapshot(**overrides):
    base = dict(
        request_rate_rps=5.0, error_rate_rps=0.0, latency_p95_ms=100.0,
        requests_in_progress=1.0, cpu_cores=0.2, memory_bytes=40_000_000.0,
    )
    base.update(overrides)
    return RawMetricSnapshot(**base)


class _FailOnceBackend(ScriptedBackend):
    """A ScriptedBackend whose set_desired_replicas() raises exactly once
    (simulating a transient kubectl/K8s-API failure), then behaves normally."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._should_fail = False

    def fail_next_mutation(self) -> None:
        self._should_fail = True

    def set_desired_replicas(self, desired: int) -> None:
        if self._should_fail:
            self._should_fail = False
            raise RuntimeError("simulated kubectl scale failure")
        super().set_desired_replicas(desired)


def _dense_timeline(times):
    return {t: _valid_snapshot() for t in times}


class TestK8sApiFailure:
    def test_failed_mutation_is_recorded_as_k8s_api_failure_not_propagated(self):
        backend = _FailOnceBackend(timeline=_dense_timeline([0.0, 30.0]), initial_replicas=1)
        env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=300.0, config=CFG)
        env.reset(seed=1)
        backend.fail_next_mutation()

        obs, reward, terminated, truncated, info = env.step(ACTION_UP)  # must not raise

        assert info["action_override_reason"] == OVERRIDE_K8S_API_FAILURE
        assert info["effective_replica_delta"] == 0
        assert info["bound_clamped"] is False
        assert info["pending_scale"] is False
        # The mutation never actually happened -- desired must be unchanged.
        assert info["deployment_desired_replicas_after"] == 1

    def test_reward_is_partial_and_action_side_zeroed_on_k8s_api_failure(self):
        backend = _FailOnceBackend(timeline=_dense_timeline([0.0, 30.0]), initial_replicas=1)
        env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=300.0, config=CFG)
        env.reset(seed=1)
        backend.fail_next_mutation()

        _, _, _, _, info = env.step(ACTION_UP)

        assert info["reward_validity"] == "partial"
        assert info["reward_components"]["action"] == 0.0
        assert info["reward_components"]["reversal"] == 0.0
        assert info["reward_components"]["boundary"] == 0.0
        assert info["reward_components"]["sla"] == 0.0
        assert info["reward_components"]["efficiency_bonus"] == 0.0

    def test_scale_tracking_state_not_updated_on_failed_mutation(self):
        """A failed mutation must not be treated as 'the last effective
        scaling action' for cooldown/reversal purposes."""
        backend = _FailOnceBackend(timeline=_dense_timeline([0.0, 30.0, 60.0]), initial_replicas=2)
        env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=300.0, config=CFG)
        env.reset(seed=1)
        backend.fail_next_mutation()

        _, _, _, _, info1 = env.step(ACTION_UP)  # fails -> no real scale-up recorded
        assert info1["action_override_reason"] == OVERRIDE_K8S_API_FAILURE

        # A genuine scale-up request now should proceed normally (no bogus
        # pending-scale gate left over from the failed attempt).
        _, _, _, _, info2 = env.step(ACTION_UP)
        assert info2["action_override_reason"] == "none"
        assert info2["effective_replica_delta"] == 1

    def test_normal_operation_unaffected_when_no_failure_occurs(self):
        backend = _FailOnceBackend(timeline=_dense_timeline([0.0, 30.0]), initial_replicas=1)
        env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=300.0, config=CFG)
        env.reset(seed=1)
        _, _, _, _, info = env.step(ACTION_UP)  # no fail_next_mutation() called
        assert info["action_override_reason"] == "none"
        assert info["effective_replica_delta"] == 1
