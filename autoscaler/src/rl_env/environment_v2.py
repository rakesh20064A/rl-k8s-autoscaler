"""autoscaler/src/rl_env/environment_v2.py

V2 Phase 2.2 PART 4: a SEPARATE V2 environment path. Audit finding this
responds to: `KubernetesAutoscalerEnv.step()` (environment.py) imports and
calls the FROZEN `reward.compute_reward()` directly, with `self.config`
defaulting to `config.load_env_config_from_yaml()` (the frozen V1 loader).
Simply constructing that class with `rl_v2.yaml`/`reward_v2.yaml` paths
would NOT wire in the new multi-metric reward -- `reward_v2.yaml`'s schema
is a genuinely different, redesigned formula (V2 Phase 2.1 PART 6) that the
frozen loader cannot even parse into a `RewardConfig`. There must be no
filename-default ambiguity: V1 training/eval code keeps using
`KubernetesAutoscalerEnv` + `config.load_env_config_from_yaml()` completely
unchanged; V2 code must explicitly use `KubernetesAutoscalerEnvV2` +
`config_v2.load_env_config_v2_from_yaml()`.

Design: composition over duplication. `KubernetesAutoscalerEnvV2` wraps an
internal, ordinary `KubernetesAutoscalerEnv` instance (reusing its FULLY
TESTED observation-building, gating, actuation, and termination logic
byte-for-byte -- that mechanics is numerically identical between V1 and V2,
only the reward formula differs) and RECOMPUTES the reward using
`reward_v2.compute_reward_v2()` from the inner env's own returned `info`
dict, which already exposes every raw quantity needed (including the new,
purely-additive `is_reversal` field -- see environment.py's
`_build_info()`). The inner env's own reward computation is never consulted
for the V2 instance's returned scalar reward -- it is fully overwritten/
discarded each step, never blended or averaged with the V2 reward.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from .backend import DynamicsBackend
from .config import EnvConfig
from .config_v2 import EnvConfigV2, load_env_config_v2_from_yaml
from .environment import KubernetesAutoscalerEnv
from .reward import RewardConfig
from .reward_v2 import compute_reward_v2
from .types import OVERRIDE_K8S_API_FAILURE, OVERRIDE_PROMETHEUS_UNAVAILABLE, OVERRIDE_STALE_OR_MISSING_METRIC

# The three override reasons that mean "the DECISION-time state itself was
# not trustworthy" (environment.py step()'s own `decision_state_valid`
# derivation: `s_t_valid and not k8s_api_failure`). Reconstructed here from
# the externally-visible `action_override_reason` because
# `decision_state_valid` itself is not separately exposed in `info` (it is
# fully determined by which of these three reasons, if any, is present).
_DECISION_INVALID_REASONS = frozenset(
    {OVERRIDE_STALE_OR_MISSING_METRIC, OVERRIDE_PROMETHEUS_UNAVAILABLE, OVERRIDE_K8S_API_FAILURE}
)


class KubernetesAutoscalerEnvV2(gym.Env):
    """V2 environment: identical observation/action/gating mechanics to
    `KubernetesAutoscalerEnv` (reused via composition, never duplicated),
    scored with the redesigned multi-metric V2 reward
    (`reward_v2.compute_reward_v2`) instead of the frozen V1
    `reward.compute_reward`. MUST be a genuine `gymnasium.Env` subclass --
    `stable_baselines3.PPO(env=...)` rejects anything else outright (this
    was caught live by tests/unit/test_train_ppo_live_v2.py during V2
    Phase 2.2, not assumed). The reward computation itself still uses
    composition (an internal `KubernetesAutoscalerEnv` instance, never
    subclassed) specifically so the "V1 reward is never consulted for
    scoring" guarantee stays structurally obvious in `step()` below -- only
    the Gymnasium `Env` base class itself is inherited, for SB3
    compatibility, not `KubernetesAutoscalerEnv`.
    """

    metadata: Dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        backend: DynamicsBackend,
        episode_duration_s: float,
        config: Optional[EnvConfigV2] = None,
    ) -> None:
        super().__init__()
        self.config_v2: EnvConfigV2 = config if config is not None else load_env_config_v2_from_yaml()

        # The inner V1-shaped EnvConfig's `reward` field is a STRUCTURALLY
        # REQUIRED placeholder ONLY -- `KubernetesAutoscalerEnv.step()`
        # does compute a reward with it internally, but that result is
        # UNCONDITIONALLY discarded by this wrapper's own `step()` below
        # (never read, never blended into the V2 reward). Using
        # `RewardConfig()`'s own frozen defaults here is deliberate: it
        # keeps this inert computation cheap and side-effect-free, and
        # makes the discard-dependency explicit to a reader (no risk of
        # this placeholder ever being mistaken for a real reward config,
        # since reward.py's RewardConfig and reward_v2.py's RewardConfigV2
        # are different, non-interchangeable types).
        inner_config = EnvConfig(
            decision_interval_s=self.config_v2.decision_interval_s,
            pre_episode_stabilization_s=self.config_v2.pre_episode_stabilization_s,
            initial_replicas=self.config_v2.initial_replicas,
            observation=self.config_v2.observation,
            gating=self.config_v2.gating,
            reward=RewardConfig(),
            max_consecutive_invalid_steps=self.config_v2.max_consecutive_invalid_steps,
            zero_ready_termination_seconds=self.config_v2.zero_ready_termination_seconds,
        )
        self._inner = KubernetesAutoscalerEnv(backend=backend, episode_duration_s=episode_duration_s, config=inner_config)
        self.observation_space = self._inner.observation_space
        self.action_space = self._inner.action_space

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        obs, info = self._inner.reset(seed=seed, options=options)
        info = dict(info)
        info["controller_generation"] = self.config_v2.controller_generation
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, _v1_reward_discarded, terminated, truncated, info = self._inner.step(action)

        raw = info["raw_observation"]
        override_reason = info["action_override_reason"]
        decision_state_valid = override_reason not in _DECISION_INVALID_REASONS

        result = compute_reward_v2(
            decision_state_valid=decision_state_valid,
            outcome_state_valid=info["sample_valid"],
            p95_latency_ms=raw.get("p95_latency_norm"),
            error_rate=raw.get("error_rate"),
            deployment_current_replicas=info["deployment_current_replicas_after"],
            cpu_per_replica=raw.get("cpu_per_replica"),
            memory_per_replica=raw.get("memory_per_replica"),
            requested_delta=info["requested_replica_delta"],
            effective_delta=info["effective_replica_delta"],
            override_reason=override_reason,
            is_reversal=info["is_reversal"],
            config=self.config_v2.reward,
        )

        info = dict(info)
        info["reward_total"] = result.total
        info["reward_components"] = dict(result.components)
        info["reward_validity"] = result.reward_validity
        info["controller_generation"] = self.config_v2.controller_generation
        return obs, float(result.total), terminated, truncated, info

    def render(self):
        return self._inner.render()

    def close(self) -> None:
        return self._inner.close()
