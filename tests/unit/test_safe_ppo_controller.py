"""tests/unit/test_safe_ppo_controller.py

V2 Phase 2 PART 5/9: end-to-end integration of safe_inference +
KubernetesAutoscalerEnv.step() + decision_record_v2, using ScriptedBackend
(TEST_ONLY_SYNTHETIC_METRICS, no real cluster) exactly as the project's own
existing environment-mechanics tests do. Proves:
  - a policy-inference failure never mutates replicas (requested_delta=0
    path through the UNCHANGED, frozen gating/environment code)
  - a successful inference step applies gating/actuation normally
  - the decision record is populated consistently with what env.step()
    actually did
"""

from __future__ import annotations

import numpy as np

from autoscaler.src.rl_env.config import load_env_config_from_yaml
from autoscaler.src.rl_env.environment import KubernetesAutoscalerEnv
from autoscaler.src.rl_env.safe_ppo_controller import run_safe_ppo_step
from autoscaler.src.rl_env.scripted_backend import ScriptedBackend
from autoscaler.src.rl_env.types import OVERRIDE_NONE, OVERRIDE_POLICY_INFERENCE_FAILURE, RawMetricSnapshot


def _make_env(n_steps: int = 10) -> KubernetesAutoscalerEnv:
    cfg = load_env_config_from_yaml()
    timeline = {}
    t = 0.0
    for _ in range(n_steps + 1):
        timeline[t] = RawMetricSnapshot(
            request_rate_rps=10.0, error_rate_rps=0.0, latency_p95_ms=50.0,
            requests_in_progress=1.0, cpu_cores=0.1, memory_bytes=10_000_000,
            target_up=True, query_error=False,
        )
        t += cfg.decision_interval_s
    backend = ScriptedBackend(timeline=timeline, initial_replicas=1)
    return KubernetesAutoscalerEnv(backend=backend, episode_duration_s=cfg.decision_interval_s * n_steps, config=cfg)


class TestPolicyInferenceFailureNeverMutatesReplicas:
    def test_none_model_produces_hold_and_no_replica_change(self):
        env = _make_env()
        obs, _info = env.reset()
        desired_before = env._last_snapshot.replicas.desired

        next_obs, reward, terminated, truncated, info, record = run_safe_ppo_step(
            env, None, obs, model_identifier="none-model", epsilon=0.05,
        )
        assert record.override_reason == OVERRIDE_POLICY_INFERENCE_FAILURE
        assert info["effective_replica_delta"] == 0
        assert info["deployment_desired_replicas_after"] == desired_before
        assert record.effective_action_index == 1  # HOLD

    def test_invalid_probabilities_produce_hold_and_no_replica_change(self, monkeypatch):
        import autoscaler.src.rl_env.safe_inference as mod

        monkeypatch.setattr(mod, "extract_action_probabilities", lambda model, obs: np.array([float("nan"), 0.5, 0.5]))
        env = _make_env()
        obs, _info = env.reset()

        next_obs, reward, terminated, truncated, info, record = run_safe_ppo_step(
            env, object(), obs, model_identifier="fake", epsilon=0.05,
        )
        assert record.override_reason == OVERRIDE_POLICY_INFERENCE_FAILURE
        assert info["effective_replica_delta"] == 0


class TestSuccessfulInferenceAppliesGatingNormally:
    def test_clear_scale_up_winner_increments_desired_replicas(self, monkeypatch):
        import autoscaler.src.rl_env.safe_inference as mod

        monkeypatch.setattr(mod, "extract_action_probabilities", lambda model, obs: np.array([0.05, 0.05, 0.9]))
        env = _make_env()
        obs, _info = env.reset()
        desired_before = env._last_snapshot.replicas.desired

        next_obs, reward, terminated, truncated, info, record = run_safe_ppo_step(
            env, object(), obs, model_identifier="fake", epsilon=0.05,
        )
        assert record.override_reason == OVERRIDE_NONE
        assert info["effective_replica_delta"] == 1
        assert info["deployment_desired_replicas_after"] == desired_before + 1
        assert record.effective_action_index == 2
        assert record.near_tie is False

    def test_exact_tie_holds_without_replica_change(self, monkeypatch):
        import autoscaler.src.rl_env.safe_inference as mod

        monkeypatch.setattr(mod, "extract_action_probabilities", lambda model, obs: np.array([0.5, 0.0, 0.5]))
        env = _make_env()
        obs, _info = env.reset()
        desired_before = env._last_snapshot.replicas.desired

        next_obs, reward, terminated, truncated, info, record = run_safe_ppo_step(
            env, object(), obs, model_identifier="fake", epsilon=0.05,
        )
        assert record.override_reason == OVERRIDE_NONE  # a genuine HOLD decision, not a failure
        assert record.near_tie is True
        assert info["deployment_desired_replicas_after"] == desired_before


class TestDecisionRecordConsistency:
    def test_record_model_identifier_passed_through(self, monkeypatch):
        import autoscaler.src.rl_env.safe_inference as mod

        monkeypatch.setattr(mod, "extract_action_probabilities", lambda model, obs: np.array([0.3, 0.4, 0.3]))
        env = _make_env()
        obs, _info = env.reset()
        _next_obs, _reward, _terminated, _truncated, _info, record = run_safe_ppo_step(
            env, object(), obs, model_identifier="models/ppo/ppo_model.zip", epsilon=0.05,
        )
        assert record.model_identifier == "models/ppo/ppo_model.zip"
        assert record.epsilon == 0.05
        assert len(record.observation_vector) == 11
