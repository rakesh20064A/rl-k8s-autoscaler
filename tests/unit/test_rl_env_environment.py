"""tests/unit/test_rl_env_environment.py

Phase 15B.1: integration/behavioral tests for
autoscaler/src/rl_env/environment.py -- the real Gymnasium environment.
Covers the Gymnasium/SB3 API surface, representative scale-up/scale-down
timelines, episode termination/truncation, and transition-info schema
completeness. All "time" is logical/deterministic -- no wall-clock sleeping.
"""

import math

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from autoscaler.src.rl_env.config import load_env_config_from_yaml
from autoscaler.src.rl_env.environment import KubernetesAutoscalerEnv
from autoscaler.src.rl_env.scripted_backend import ScriptedBackend
from autoscaler.src.rl_env.types import (
    ACTION_MAP,
    OVERRIDE_NONE,
    OVERRIDE_PENDING_SCALE_GATE,
    OVERRIDE_SCALE_DOWN_COOLDOWN,
)

ACTION_DOWN, ACTION_HOLD, ACTION_UP = 0, 1, 2
CFG = load_env_config_from_yaml()

REQUIRED_INFO_FIELDS = {
    "episode_id", "step_index", "decision_time_s", "raw_observation", "normalized_observation",
    "action_index", "semantic_action", "requested_replica_delta", "effective_replica_delta",
    "deployment_desired_replicas_before", "deployment_desired_replicas_after",
    "deployment_current_replicas_before", "deployment_current_replicas_after",
    "ready_replicas_before", "ready_replicas_after", "available_replicas_before", "available_replicas_after",
    "reward_total", "reward_components", "terminated", "truncated", "action_override_reason",
    "bound_clamped", "pending_scale", "sample_valid", "observation_age_seconds",
    "consecutive_invalid_steps", "reward_validity", "latency_observation_state",
}


def _valid_snapshot(**overrides):
    from autoscaler.src.rl_env.types import RawMetricSnapshot

    base = dict(
        request_rate_rps=5.0, error_rate_rps=0.0, latency_p95_ms=100.0,
        requests_in_progress=1.0, cpu_cores=0.2, memory_bytes=40_000_000.0,
    )
    base.update(overrides)
    return RawMetricSnapshot(**base)


def _dense_timeline(times):
    return {t: _valid_snapshot() for t in times}


class _AlwaysZeroReadyBackend:
    """Minimal DynamicsBackend stub used ONLY to exercise the environment's
    zero-Ready->120s termination branch -- ScriptedBackend's own min_replicas
    floor (>=1 in any realistic configuration) cannot otherwise produce a
    sustained ready==0 state, since that never occurs in this project's
    actual Kubernetes configuration either."""

    def __init__(self):
        self._time = 0.0
        from autoscaler.src.rl_env.types import BackendSnapshot, ReplicaState
        self._BackendSnapshot = BackendSnapshot
        self._ReplicaState = ReplicaState

    def reset(self):
        self._time = 0.0
        return self._snap()

    def _snap(self):
        return self._BackendSnapshot(
            time_s=self._time, metrics=_valid_snapshot(),
            replicas=self._ReplicaState(desired=1, current=1, ready=0, available=0),
        )

    def current_snapshot(self):
        return self._snap()

    def set_desired_replicas(self, desired):
        pass

    def advance(self, dt_seconds):
        self._time += dt_seconds
        return self._snap()


class TestGymnasiumApiSurface:
    def test_observation_and_action_spaces_match_frozen_contract(self):
        backend = ScriptedBackend(timeline=_dense_timeline([0.0, 30.0]), initial_replicas=1)
        env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=300.0, config=CFG)
        assert env.observation_space.shape == (11,)
        assert env.observation_space.dtype == np.float32
        assert env.action_space.n == 3

    def test_reset_returns_obs_and_info(self):
        backend = ScriptedBackend(timeline=_dense_timeline([0.0]), initial_replicas=1)
        env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=300.0, config=CFG)
        obs, info = env.reset(seed=42)
        assert obs.shape == (11,)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)

    def test_step_returns_standard_five_tuple(self):
        backend = ScriptedBackend(timeline=_dense_timeline([0.0, 30.0]), initial_replicas=1)
        env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=300.0, config=CFG)
        env.reset(seed=1)
        obs, reward, terminated, truncated, info = env.step(ACTION_HOLD)
        assert obs.shape == (11,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_sb3_check_env_passes(self):
        backend = ScriptedBackend(timeline=_dense_timeline([i * 30.0 for i in range(20)]), initial_replicas=1)
        env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=600.0, config=CFG)
        check_env(env, warn=True)  # raises on failure


class TestTransitionInfoSchema:
    def test_all_required_fields_present_on_reset_and_step(self):
        backend = ScriptedBackend(timeline=_dense_timeline([0.0, 30.0]), initial_replicas=1)
        env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=300.0, config=CFG)
        _, reset_info = env.reset(seed=1)
        assert REQUIRED_INFO_FIELDS.issubset(reset_info.keys())
        _, _, _, _, step_info = env.step(ACTION_HOLD)
        assert REQUIRED_INFO_FIELDS.issubset(step_info.keys())


class TestScaleUpTimeline:
    """Task sec.24's representative scale-up scenario."""

    def test_delayed_ready_then_resource_and_reward_use_correct_replica_fields(self):
        timeline = _dense_timeline([0.0, 30.0, 60.0])
        backend = ScriptedBackend(timeline=timeline, initial_replicas=1, scale_up_ready_delay_seconds=10.0)
        env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=300.0, config=CFG)
        env.reset(seed=1)

        obs, reward, terminated, truncated, info = env.step(ACTION_UP)
        assert info["effective_replica_delta"] == 1
        assert info["deployment_desired_replicas_before"] == 1
        assert info["deployment_desired_replicas_after"] == 2
        assert info["deployment_current_replicas_after"] == 2
        assert info["ready_replicas_before"] == 1
        # Readiness delay (10s) < decision interval (30s) -> realized by t=30.
        assert info["ready_replicas_after"] == 2
        # Resource reward term must reflect CURRENT (2), not some stale value.
        assert info["reward_components"]["resource"] == pytest.approx(1.0 / 7.0)

        # Next decision: pending gate no longer active (desired==ready==2).
        obs2, reward2, terminated2, truncated2, info2 = env.step(ACTION_HOLD)
        assert info2["pending_scale"] is False
        assert info2["action_override_reason"] == OVERRIDE_NONE

    def test_second_scale_up_request_before_ready_is_pending_gated(self):
        timeline = _dense_timeline([0.0, 30.0, 60.0, 90.0])
        # Slow readiness (40s) so it is NOT realized within one 30s step.
        backend = ScriptedBackend(timeline=timeline, initial_replicas=1, scale_up_ready_delay_seconds=40.0)
        env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=300.0, config=CFG)
        env.reset(seed=1)

        _, _, _, _, info1 = env.step(ACTION_UP)
        assert info1["ready_replicas_after"] == 1  # not yet realized at t=30
        assert info1["deployment_desired_replicas_after"] == 2

        _, _, _, _, info2 = env.step(ACTION_UP)  # requested again while pending
        assert info2["requested_replica_delta"] == 1
        assert info2["effective_replica_delta"] == 0
        assert info2["action_override_reason"] == OVERRIDE_PENDING_SCALE_GATE
        assert info2["ready_replicas_after"] == 2  # realized by t=60 (scheduled at t=40)


class TestScaleDownTimeline:
    """Task sec.25's representative scale-down scenario."""

    def test_scale_down_immediate_and_emergency_up_not_blocked(self):
        timeline = _dense_timeline([0.0, 30.0, 60.0])
        backend = ScriptedBackend(timeline=timeline, initial_replicas=2)
        env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=300.0, config=CFG)
        env.reset(seed=1)

        _, _, _, _, info1 = env.step(ACTION_DOWN)
        assert info1["effective_replica_delta"] == -1
        assert info1["deployment_desired_replicas_after"] == 1
        assert info1["deployment_current_replicas_after"] == 1
        assert info1["ready_replicas_after"] == 1
        assert info1["available_replicas_after"] == 1

        # 30s later: an urgent scale-up must NOT be blocked by the
        # scale-down cooldown (only 30s have elapsed, < 60s).
        _, _, _, _, info2 = env.step(ACTION_UP)
        assert info2["action_override_reason"] == OVERRIDE_NONE
        assert info2["effective_replica_delta"] == 1

    def test_another_scale_down_is_cooldown_blocked_within_60s(self):
        timeline = _dense_timeline([0.0, 30.0, 60.0])
        backend = ScriptedBackend(timeline=timeline, initial_replicas=3)
        env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=300.0, config=CFG)
        env.reset(seed=1)
        env.step(ACTION_DOWN)  # 3 -> 2 at t=30
        _, _, _, _, info2 = env.step(ACTION_DOWN)  # requested again 30s later
        assert info2["action_override_reason"] == OVERRIDE_SCALE_DOWN_COOLDOWN
        assert info2["effective_replica_delta"] == 0


class TestEpisodeTermination:
    def test_fourth_consecutive_invalid_step_terminates(self):
        # Valid at t=0 (reset) and t=120 (mid-sequence reset point); missing
        # everywhere else -> invalid.
        timeline = {0.0: _valid_snapshot(), 120.0: _valid_snapshot()}
        backend = ScriptedBackend(timeline=timeline, initial_replicas=1)
        env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=10_000.0, config=CFG)
        env.reset(seed=1)

        results = []
        for _ in range(8):
            _, _, terminated, truncated, info = env.step(ACTION_HOLD)
            results.append((terminated, info["consecutive_invalid_steps"]))

        # Steps at t=30,60,90 invalid (counts 1,2,3); t=120 valid (resets to 0);
        # t=150,180,210 invalid (1,2,3); t=240 invalid (4th) -> terminated.
        counts = [c for _, c in results]
        assert counts == [1, 2, 3, 0, 1, 2, 3, 4]
        assert results[6][0] is False  # not yet terminated at count 3
        assert results[7][0] is True  # terminated at the 4th consecutive invalid step

    def test_single_valid_step_resets_invalid_counter(self):
        timeline = {0.0: _valid_snapshot(), 30.0: _valid_snapshot(), 90.0: _valid_snapshot()}
        backend = ScriptedBackend(timeline=timeline, initial_replicas=1)
        env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=10_000.0, config=CFG)
        env.reset(seed=1)
        _, _, _, _, info1 = env.step(ACTION_HOLD)  # t=30 valid
        assert info1["consecutive_invalid_steps"] == 0
        _, _, _, _, info2 = env.step(ACTION_HOLD)  # t=60 missing -> invalid
        assert info2["consecutive_invalid_steps"] == 1
        _, _, _, _, info3 = env.step(ACTION_HOLD)  # t=90 valid -> resets
        assert info3["consecutive_invalid_steps"] == 0

    def test_zero_ready_for_over_120s_terminates(self):
        env = KubernetesAutoscalerEnv(backend=_AlwaysZeroReadyBackend(), episode_duration_s=10_000.0, config=CFG)
        env.reset(seed=1)
        terminated = False
        for _ in range(6):  # 6 * 30s = 180s > 120s
            _, _, terminated, _, _ = env.step(ACTION_HOLD)
            if terminated:
                break
        assert terminated is True


class TestTruncation:
    def test_truncates_when_episode_duration_reached(self):
        timeline = _dense_timeline([i * 30.0 for i in range(6)])
        backend = ScriptedBackend(timeline=timeline, initial_replicas=1)
        env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=90.0, config=CFG)
        env.reset(seed=1)
        truncated = False
        for _ in range(5):
            _, _, terminated, truncated, _ = env.step(ACTION_HOLD)
            if truncated:
                break
        assert truncated is True

    def test_not_truncated_before_duration(self):
        timeline = _dense_timeline([0.0, 30.0])
        backend = ScriptedBackend(timeline=timeline, initial_replicas=1)
        env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=300.0, config=CFG)
        env.reset(seed=1)
        _, _, _, truncated, _ = env.step(ACTION_HOLD)
        assert truncated is False


class TestNoFutureLeakage:
    def test_action_map_matches_frozen_encoding(self):
        assert ACTION_MAP == {0: -1, 1: 0, 2: 1}


class TestEpisodeResetControlAge:
    """Phase 15B.1A / CORR-30 sec.7-10: the logical reset-to-initial-replicas
    event is treated as an effective scaling action exactly
    `pre_episode_stabilization_s` (60s) before episode t=0."""

    def test_seconds_since_last_scale_is_stabilization_over_cap_at_t0(self):
        backend = ScriptedBackend(timeline=_dense_timeline([0.0]), initial_replicas=1)
        env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=300.0, config=CFG)
        obs, info = env.reset(seed=1)
        assert obs[10] == pytest.approx(60.0 / 300.0)

    def test_trend_features_are_exactly_zero_at_t0(self):
        backend = ScriptedBackend(timeline=_dense_timeline([0.0]), initial_replicas=1)
        env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=300.0, config=CFG)
        obs, info = env.reset(seed=1)
        assert obs[1] == 0.0  # cpu_per_replica_delta
        assert obs[5] == 0.0  # request_rate_delta
        assert obs[7] == 0.0  # p95_latency_delta

    def test_reset_does_not_leak_previous_episode_history(self):
        backend = ScriptedBackend(timeline=_dense_timeline([0.0, 30.0, 60.0]), initial_replicas=1)
        env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=300.0, config=CFG)
        env.reset(seed=1)
        env.step(ACTION_UP)  # perturb history/effective-action-time state
        env.step(ACTION_HOLD)
        obs, info = env.reset(seed=2)  # fresh episode
        assert obs[1] == 0.0
        assert obs[5] == 0.0
        assert obs[7] == 0.0
        assert obs[10] == pytest.approx(60.0 / 300.0)

    def test_scale_down_at_t0_resolves_as_bound_clamp_not_cooldown(self):
        """CORR-30 sec.10: even though seconds_since_last_scale=60s already
        satisfies the 60s cooldown, a SCALE_DOWN request at 1 replica (the
        min bound) must still resolve as replica_bound_clamp -- the bound
        check takes precedence."""
        backend = ScriptedBackend(timeline=_dense_timeline([0.0, 30.0]), initial_replicas=1)
        env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=300.0, config=CFG)
        env.reset(seed=1)
        _, _, _, _, info = env.step(ACTION_DOWN)
        assert info["action_override_reason"] == "replica_bound_clamp"
        assert info["bound_clamped"] is True


class TestValidityMatrixAtEnvironmentLevel:
    """CORR-30 Case C exercised through the real environment/backend path
    (not just the standalone reward.compute_reward tests)."""

    def test_valid_scale_up_then_next_step_telemetry_fails_preserves_action_penalty(self):
        # t=0 valid; t=30 valid (scale-up observed); t=60 MISSING (telemetry fails).
        timeline = _dense_timeline([0.0, 30.0])
        backend = ScriptedBackend(timeline=timeline, initial_replicas=1)
        env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=300.0, config=CFG)
        env.reset(seed=1)

        _, _, _, _, info1 = env.step(ACTION_UP)  # t=0 -> t=30, valid outcome
        assert info1["reward_validity"] == "full"
        assert math.isclose(info1["reward_components"]["action"], 0.1)

        _, _, _, _, info2 = env.step(ACTION_HOLD)  # t=30 -> t=60, MISSING timeline entry
        assert info2["sample_valid"] is False
        assert info2["reward_validity"] == "partial"
        # This step's own action (HOLD) incurs no action penalty regardless,
        # but the outcome-dependent SLA/efficiency must be zero, not fabricated.
        assert info2["reward_components"]["sla"] == 0.0
        assert info2["reward_components"]["efficiency_bonus"] == 0.0

    def test_effective_action_reward_not_erased_by_subsequent_invalid_outcome(self):
        """Direct environment-level Case C: the SAME step that issues a
        valid, in-bounds SCALE_UP request must show its action penalty in
        `info`, even when that very step's resulting observation is
        invalid (no scripted metrics at the destination time)."""
        timeline = {0.0: _dense_timeline([0.0])[0.0]}  # only t=0 scripted; t=30 missing
        backend = ScriptedBackend(timeline=timeline, initial_replicas=1)
        env = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=300.0, config=CFG)
        env.reset(seed=1)

        _, _, _, _, info = env.step(ACTION_UP)  # effective at t=0 (valid decision state)
        assert info["effective_replica_delta"] == 1
        assert info["sample_valid"] is False  # outcome at t=30 is missing/invalid
        assert info["reward_validity"] == "partial"
        assert math.isclose(info["reward_components"]["action"], 0.1)  # PRESERVED, not erased
        assert info["reward_components"]["sla"] == 0.0
