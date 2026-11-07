"""tests/unit/test_environment_v2.py

V2 Phase 2.2 PART 4/6/11: proves the ACTUAL environment integration -- not
merely the standalone reward_v2 function in isolation -- using the SAME
synthetic state/action with both the frozen V1 environment
(KubernetesAutoscalerEnv + reward.py) and the new V2 environment
(KubernetesAutoscalerEnvV2 + reward_v2.py), via ScriptedBackend (no real
cluster needed).
"""

from __future__ import annotations

import pytest

from autoscaler.src.rl_env.config import load_env_config_from_yaml
from autoscaler.src.rl_env.config_v2 import load_env_config_v2_from_yaml
from autoscaler.src.rl_env.environment import KubernetesAutoscalerEnv
from autoscaler.src.rl_env.environment_v2 import KubernetesAutoscalerEnvV2
from autoscaler.src.rl_env.scripted_backend import ScriptedBackend
from autoscaler.src.rl_env.types import RawMetricSnapshot

HOLD = 1


def _timeline(cfg, n_steps, cpu_cores, memory_bytes, request_rate=10.0, latency_ms=50.0, error_rate=0.0):
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


class TestGymnasiumCompatibility:
    """Caught live during V2 Phase 2.2 PART 10's dry-run test: SB3's
    `PPO(env=...)` outright rejects anything that is not a genuine
    `gymnasium.Env` instance -- composition alone was insufficient.
    `KubernetesAutoscalerEnvV2` must be a real `gym.Env` subclass."""

    def test_is_a_gymnasium_env_instance(self):
        import gymnasium as gym

        cfg = load_env_config_v2_from_yaml()
        timeline = _timeline(cfg, 2, 0.1, 20_000_000)
        backend = ScriptedBackend(timeline=timeline, initial_replicas=1)
        env = KubernetesAutoscalerEnvV2(backend=backend, episode_duration_s=cfg.decision_interval_s * 2, config=cfg)
        assert isinstance(env, gym.Env)

    def test_sb3_check_env_passes(self):
        from stable_baselines3.common.env_checker import check_env

        cfg = load_env_config_v2_from_yaml()
        timeline = _timeline(cfg, 20, 0.1, 20_000_000)
        backend = ScriptedBackend(timeline=timeline, initial_replicas=1)
        env = KubernetesAutoscalerEnvV2(backend=backend, episode_duration_s=600.0, config=cfg)
        check_env(env, warn=True)  # raises on failure

    def test_sb3_ppo_accepts_the_v2_environment(self):
        """Direct proof of the exact failure test_train_ppo_live_v2.py's
        live-training path caught: PPO(env=...) must not raise."""
        from stable_baselines3 import PPO

        cfg = load_env_config_v2_from_yaml()
        timeline = _timeline(cfg, 5, 0.1, 20_000_000)
        backend = ScriptedBackend(timeline=timeline, initial_replicas=1)
        env = KubernetesAutoscalerEnvV2(backend=backend, episode_duration_s=150.0, config=cfg)
        model = PPO(env=env, policy="MlpPolicy", n_steps=16, batch_size=8, verbose=0)  # must not raise
        assert model is not None


class TestV1EnvironmentUsesFrozenReward:
    def test_v1_env_imports_frozen_reward_module(self):
        import autoscaler.src.rl_env.environment as env_mod

        assert env_mod.compute_reward.__module__ == "autoscaler.src.rl_env.reward"

    def test_v1_env_default_config_is_frozen_loader(self):
        cfg = load_env_config_from_yaml()
        from autoscaler.src.rl_env.reward import RewardConfig

        assert isinstance(cfg.reward, RewardConfig)


class TestV2EnvironmentUsesReward2(object):
    def test_v2_env_imports_reward_v2_module(self):
        import autoscaler.src.rl_env.environment_v2 as env_v2_mod

        assert env_v2_mod.compute_reward_v2.__module__ == "autoscaler.src.rl_env.reward_v2"

    def test_v2_env_default_config_is_v2_loader(self):
        cfg = load_env_config_v2_from_yaml()
        from autoscaler.src.rl_env.reward_v2 import RewardConfigV2

        assert isinstance(cfg.reward, RewardConfigV2)


class TestSameSyntheticStatesBothEnvironments:
    """The task's required comparison matrix: healthy low-memory, high-
    memory healthy-latency, SLA violation, error state -- run through BOTH
    real environment integrations, not the standalone reward functions."""

    def _run_one_step(self, env_cls, cfg, cpu_cores, memory_bytes, request_rate=10.0, latency_ms=50.0, error_rate=0.0):
        timeline = _timeline(cfg, 2, cpu_cores, memory_bytes, request_rate, latency_ms, error_rate)
        backend = ScriptedBackend(timeline=timeline, initial_replicas=1)
        env = env_cls(backend=backend, episode_duration_s=cfg.decision_interval_s * 2, config=cfg)
        obs, _info = env.reset()
        _obs2, reward, _term, _trunc, info = env.step(HOLD)
        return reward, info

    def test_a_healthy_low_memory_state(self):
        v1_cfg = load_env_config_from_yaml()
        v2_cfg = load_env_config_v2_from_yaml()
        # cpu_cores=0.15 -> cpu_per_replica=1.5 (in-band); memory_bytes=20MB -> memory_per_replica~0.3 (low, safe)
        v1_reward, v1_info = self._run_one_step(KubernetesAutoscalerEnv, v1_cfg, 0.15, 20_000_000)
        v2_reward, v2_info = self._run_one_step(KubernetesAutoscalerEnvV2, v2_cfg, 0.15, 20_000_000)
        assert v1_info["reward_components"]["efficiency_bonus"] == pytest.approx(0.3)  # V1 earns its CPU-band bonus
        assert v2_info["reward_components"]["pressure"] == pytest.approx(0.0)  # V2: no unsafe pressure
        assert v2_info["reward_components"]["waste"] == pytest.approx(0.0)  # at min_replicas, never "waste"

    def test_b_high_memory_healthy_latency_state(self):
        """THE central multi-metric proof: the SAME high-memory state must
        clearly affect V2's reward (pressure term fires) while leaving
        V1's reward COMPLETELY UNCHANGED relative to an otherwise-identical
        low-memory state (V1 has no memory input at all)."""
        v1_cfg = load_env_config_from_yaml()
        v2_cfg = load_env_config_v2_from_yaml()
        # memory_bytes=200MB -> memory_per_replica ~2.98x request (well above V2's 1.2 safe ceiling)
        HIGH_MEM_BYTES = 200_000_000
        LOW_MEM_BYTES = 20_000_000

        v1_reward_high_mem, v1_info_high = self._run_one_step(KubernetesAutoscalerEnv, v1_cfg, 0.15, HIGH_MEM_BYTES)
        v1_reward_low_mem, v1_info_low = self._run_one_step(KubernetesAutoscalerEnv, v1_cfg, 0.15, LOW_MEM_BYTES)
        v2_reward_high_mem, v2_info_high = self._run_one_step(KubernetesAutoscalerEnvV2, v2_cfg, 0.15, HIGH_MEM_BYTES)
        v2_reward_low_mem, v2_info_low = self._run_one_step(KubernetesAutoscalerEnvV2, v2_cfg, 0.15, LOW_MEM_BYTES)

        # V1: reward is IDENTICAL regardless of memory level -- V1 cannot see memory at all.
        assert v1_reward_high_mem == pytest.approx(v1_reward_low_mem)
        assert v1_info_high["reward_components"] == v1_info_low["reward_components"]

        # V2: reward is CLEARLY WORSE under high memory pressure.
        assert v2_reward_high_mem < v2_reward_low_mem
        assert v2_info_high["reward_components"]["pressure"] > 0.0
        assert v2_info_low["reward_components"]["pressure"] == pytest.approx(0.0)

    def test_c_sla_violation_state(self):
        v1_cfg = load_env_config_from_yaml()
        v2_cfg = load_env_config_v2_from_yaml()
        v1_reward, v1_info = self._run_one_step(KubernetesAutoscalerEnv, v1_cfg, 0.15, 20_000_000, latency_ms=600.0)
        v2_reward, v2_info = self._run_one_step(KubernetesAutoscalerEnvV2, v2_cfg, 0.15, 20_000_000, latency_ms=600.0)
        assert v1_info["reward_components"]["sla"] > 0.0
        assert v2_info["reward_components"]["sla"] > 0.0
        assert v1_reward < 0
        assert v2_reward < 0

    def test_d_error_state(self):
        v1_cfg = load_env_config_from_yaml()
        v2_cfg = load_env_config_v2_from_yaml()
        # error_rate (the normalized fraction reward.py/reward_v2.py see)
        # is error_rate_rps/request_rate_rps -- request_rate=10.0, so an
        # error_rate_rps of 3.0 yields the intended 0.3 fraction.
        v1_reward, v1_info = self._run_one_step(
            KubernetesAutoscalerEnv, v1_cfg, 0.15, 20_000_000, error_rate=3.0,
        )
        v2_reward, v2_info = self._run_one_step(
            KubernetesAutoscalerEnvV2, v2_cfg, 0.15, 20_000_000, error_rate=3.0,
        )
        assert v1_info["reward_components"]["reliability"] == pytest.approx(5.0 * 0.3)  # V1 w_reliability=5.0
        assert v2_info["reward_components"]["reliability"] == pytest.approx(6.0 * 0.3)  # V2 w_reliability=6.0
        # Reliability dominates both: it is the largest single component in both reward designs.
        assert v1_info["reward_components"]["reliability"] >= max(
            v for k, v in v1_info["reward_components"].items() if k != "reliability"
        )
        assert v2_info["reward_components"]["reliability"] >= max(
            v for k, v in v2_info["reward_components"].items() if k != "reliability"
        )


class TestV2SlaErrorDominanceIntactInRealEnvironment:
    def test_severe_sla_violation_dominates_over_healthy_waste_free_state(self):
        v2_cfg = load_env_config_v2_from_yaml()
        helper = TestSameSyntheticStatesBothEnvironments()
        violation_reward, _ = helper._run_one_step(
            KubernetesAutoscalerEnvV2, v2_cfg, 0.1, 20_000_000, latency_ms=900.0,
        )
        healthy_reward, _ = helper._run_one_step(
            KubernetesAutoscalerEnvV2, v2_cfg, 0.1, 20_000_000, latency_ms=50.0,
        )
        assert violation_reward < healthy_reward

    def test_error_state_worse_than_pressure_state_in_v2(self):
        v2_cfg = load_env_config_v2_from_yaml()
        helper = TestSameSyntheticStatesBothEnvironments()
        error_reward, _ = helper._run_one_step(
            KubernetesAutoscalerEnvV2, v2_cfg, 0.1, 20_000_000, error_rate=8.0,  # error_rate_rps=8.0 / request_rate=10.0 -> fraction 0.8
        )
        pressure_reward, _ = helper._run_one_step(
            KubernetesAutoscalerEnvV2, v2_cfg, 0.1, 230_000_000,  # near memory hard limit
        )
        assert error_reward < pressure_reward  # reliability (w=6.0) > pressure (w=2.0)
