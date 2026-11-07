"""autoscaler/src/rl_env/environment.py

`KubernetesAutoscalerEnv` -- the real Gymnasium environment implementing
the frozen RL MDP contract v2 (docs/rl_mdp_spec.md, CORR-27/28), Phase
15B.1. Obtains all workload/replica state through an injected
`backend.DynamicsBackend` -- never hard-codes Kubernetes or Prometheus
operations. See autoscaler/src/rl_env/__init__.py for the scope limits of
this phase (mechanics only, no empirical dynamics model).

Step-order contract (docs/rl_mdp_spec.md sec.2, this phase's task sec.18):
at decision time t, the observation already returned represents information
available through t; `step(action)` determines the effective action from
that state (gating), applies it to the backend, advances the backend's
logical clock by one decision interval, and only then builds s_(t+1) and
the reward for the transition t -> t+30. No future information ever leaks
into either the action-gating decision or the observation.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .backend import DynamicsBackend
from .config import EnvConfig, load_env_config_from_yaml
from .gating import compute_effective_action
from .observation import HistoryState, build_observation
from .reward import compute_reward
from .types import ACTION_MAP, ACTION_NAMES, OVERRIDE_K8S_API_FAILURE, OVERRIDE_NONE

_OBS_LOW = np.array([0, -1, 0, 0, 0, -1, 0, -1, 0, 0, 0], dtype=np.float32)
_OBS_HIGH = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)

_episode_counter = itertools.count(1)


class KubernetesAutoscalerEnv(gym.Env):
    """See module docstring. `episode_duration_s` is the logical duration
    of one episode (mirrors "one full traffic-profile execution",
    docs/rl_mdp_spec.md sec.20) -- for Phase 15B.1's mechanics tests this is
    typically a small number of decision steps, not a real multi-minute
    profile."""

    metadata: Dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        backend: DynamicsBackend,
        episode_duration_s: float,
        config: Optional[EnvConfig] = None,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.episode_duration_s = episode_duration_s
        self.config: EnvConfig = config if config is not None else load_env_config_from_yaml()

        self.observation_space = spaces.Box(low=_OBS_LOW, high=_OBS_HIGH, shape=(11,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self._episode_id: str = ""
        self._time_s = 0.0
        self._step_index = 0
        self._consecutive_invalid_steps = 0
        self._last_effective_action_time_s: Optional[float] = None
        self._last_effective_direction: Optional[int] = None
        self._history = HistoryState.new(self.config.observation.ewma_span_steps)
        self._last_normalized_obs = None
        self._last_validity = None
        self._last_snapshot = None
        self._zero_ready_since_s: Optional[float] = None

    # -- internal helpers -------------------------------------------------

    def _seconds_since_last_effective(self, now_s: float) -> float:
        """`_last_effective_action_time_s` is never None after reset() --
        it is initialized to `-pre_episode_stabilization_s` (CORR-30 sec.7),
        so a controller that has never issued an in-episode scaling action
        correctly reports having "last scaled" exactly `stabilization_s`
        seconds ago (the logical reset-to-1-replica event), continuing to
        grow from there rather than reading a fixed cap indefinitely."""
        return now_s - self._last_effective_action_time_s

    def _build_info(
        self,
        *,
        obs_result,
        action_index: Optional[int],
        requested_delta: Optional[int],
        effective_delta: int,
        override_reason: str,
        bound_clamped: bool,
        pending_scale: bool,
        reward_result,
        terminated: bool,
        truncated: bool,
        snapshot_before,
        snapshot_after,
        is_reversal: bool = False,
    ) -> Dict[str, Any]:
        rb = snapshot_before.replicas
        ra = snapshot_after.replicas
        info: Dict[str, Any] = {
            "episode_id": self._episode_id,
            "step_index": self._step_index,
            "decision_time_s": snapshot_after.time_s,
            "raw_observation": dict(obs_result.raw),
            "normalized_observation": list(obs_result.normalized),
            "action_index": action_index,
            "semantic_action": ACTION_NAMES.get(action_index) if action_index is not None else None,
            "requested_replica_delta": requested_delta,
            "effective_replica_delta": effective_delta,
            "deployment_desired_replicas_before": rb.desired,
            "deployment_desired_replicas_after": ra.desired,
            "deployment_current_replicas_before": rb.current,
            "deployment_current_replicas_after": ra.current,
            "ready_replicas_before": rb.ready,
            "ready_replicas_after": ra.ready,
            "available_replicas_before": rb.available,
            "available_replicas_after": ra.available,
            "reward_total": reward_result.total if reward_result is not None else 0.0,
            "reward_components": dict(reward_result.components) if reward_result is not None else {},
            "terminated": terminated,
            "truncated": truncated,
            "action_override_reason": override_reason,
            "bound_clamped": bound_clamped,
            "pending_scale": pending_scale,
            # V2 Phase 2.2 PART 4: purely additive -- exposes the already-
            # computed decision-time reversal flag so a separate V2
            # environment/reward wrapper (environment_v2.py) can correctly
            # recompute action/reversal/boundary eligibility externally,
            # without reimplementing this step's direction-tracking logic.
            # Never read by this class's own (frozen) reward.compute_reward()
            # call, which already received `is_reversal` directly.
            "is_reversal": is_reversal,
            "sample_valid": obs_result.validity.sample_valid,
            "observation_age_seconds": 0.0,
            "consecutive_invalid_steps": self._consecutive_invalid_steps,
            "reward_validity": reward_result.reward_validity if reward_result is not None else "full",
            "latency_observation_state": obs_result.validity.latency_observation_state,
        }
        return info

    # -- Gymnasium API ------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._episode_id = f"episode_{next(_episode_counter)}"
        self._step_index = 0
        self._consecutive_invalid_steps = 0
        # CORR-30 sec.7: the logical reset-to-`initial_replicas` event is
        # treated as an effective scaling action that occurred exactly
        # `pre_episode_stabilization_s` seconds before episode t=0 -- NOT
        # "never scaled" (which would read as the 300s cap / normalized
        # 1.0). This deterministically pins the first observation's
        # seconds_since_last_scale_norm to stabilization_s/cap (60/300=0.2
        # under the frozen defaults) regardless of whatever control history
        # existed before this reset.
        self._last_effective_action_time_s = -self.config.pre_episode_stabilization_s
        self._last_effective_direction = None
        self._history = HistoryState.new(self.config.observation.ewma_span_steps)

        snapshot = self.backend.reset()
        self._time_s = snapshot.time_s
        self._zero_ready_since_s = snapshot.time_s if snapshot.replicas.ready == 0 else None

        obs_result = build_observation(
            metrics=snapshot.metrics,
            replicas=snapshot.replicas,
            seconds_since_last_effective_action=self._seconds_since_last_effective(self._time_s),
            history=self._history,
            previous_normalized=None,
            config=self.config.observation,
        )
        self._last_normalized_obs = obs_result.normalized
        self._last_validity = obs_result.validity
        self._last_snapshot = snapshot
        if not obs_result.validity.sample_valid:
            self._consecutive_invalid_steps = 1

        info = self._build_info(
            obs_result=obs_result, action_index=None, requested_delta=None, effective_delta=0,
            override_reason=OVERRIDE_NONE, bound_clamped=False, pending_scale=False,
            reward_result=None, terminated=False, truncated=False,
            snapshot_before=snapshot, snapshot_after=snapshot,
        )
        return np.array(obs_result.normalized, dtype=np.float32), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        requested_delta = ACTION_MAP[int(action)]
        snapshot_before = self._last_snapshot
        s_t_valid = self._last_validity.sample_valid
        desired_before = snapshot_before.replicas.desired
        ready_before = snapshot_before.replicas.ready

        seconds_since_last_effective = self._seconds_since_last_effective(self._time_s)

        if not s_t_valid:
            effective_delta = 0
            override_reason = self._last_validity.override_reason
            bound_clamped = False
            pending_scale = False
        else:
            gating_result = compute_effective_action(
                requested_delta, desired_before, ready_before,
                seconds_since_last_effective, self.config.gating,
            )
            effective_delta = gating_result.effective_delta
            override_reason = gating_result.reason
            bound_clamped = gating_result.bound_clamped
            pending_scale = gating_result.pending_scale

        is_reversal = False
        k8s_api_failure = False
        if effective_delta != 0:
            direction = 1 if effective_delta > 0 else -1
            if (
                self._last_effective_direction is not None
                and direction != self._last_effective_direction
                and seconds_since_last_effective <= self.config.reward.reversal_window_seconds
            ):
                is_reversal = True
            # Phase16V fix (conforms to the already-frozen
            # docs/rl_mdp_spec.md sec.13 table row "Kubernetes API patch
            # call fails -> Effective HOLD; no blind retry within the
            # step" -- previously unreachable/untested because no backend
            # prior to the live one could ever raise here). An actuation
            # failure must be recorded as a distinct k8s_api_failure event,
            # never silently absorbed: the requested mutation did not
            # happen, so this step's effective outcome reverts to HOLD and
            # none of the scale-tracking state is updated.
            try:
                self.backend.set_desired_replicas(desired_before + effective_delta)
            except Exception:  # noqa: BLE001 -- must become a recorded override, never propagate
                k8s_api_failure = True
                effective_delta = 0
                override_reason = OVERRIDE_K8S_API_FAILURE
                bound_clamped = False
                pending_scale = False
                is_reversal = False
            else:
                self._last_effective_action_time_s = self._time_s
                self._last_effective_direction = direction

        snapshot_after = self.backend.advance(self.config.decision_interval_s)
        self._time_s = snapshot_after.time_s
        self._step_index += 1

        new_seconds_since_last_effective = self._seconds_since_last_effective(self._time_s)
        obs_result = build_observation(
            metrics=snapshot_after.metrics,
            replicas=snapshot_after.replicas,
            seconds_since_last_effective_action=new_seconds_since_last_effective,
            history=self._history,
            previous_normalized=self._last_normalized_obs,
            config=self.config.observation,
        )

        if obs_result.validity.sample_valid:
            self._consecutive_invalid_steps = 0
        else:
            self._consecutive_invalid_steps += 1

        terminated = self._consecutive_invalid_steps >= self.config.max_consecutive_invalid_steps

        if snapshot_after.replicas.ready == 0:
            if self._zero_ready_since_s is None:
                self._zero_ready_since_s = self._time_s
            elif (self._time_s - self._zero_ready_since_s) > self.config.zero_ready_termination_seconds:
                terminated = True
        else:
            self._zero_ready_since_s = None

        truncated = self._time_s >= self.episode_duration_s

        # CORR-30: decision-state (s_t) and outcome-state (s_(t+1)) validity
        # are passed SEPARATELY -- an invalid outcome must never retroactively
        # erase an already-determined, valid policy action's action/reversal/
        # boundary consequences (docs/rl_mdp_spec.md sec.17-18).
        # Phase16V: k8s_api_failure is one of the four telemetry-failure
        # override reasons docs/rl_mdp_spec.md sec.18 lists as reward-
        # invalidating ("this step's action was not a real, fully-evaluated
        # decision") -- even though, unlike the other three, it reflects an
        # ACTUATION failure rather than an OBSERVATION failure. Applied here
        # per the frozen table's literal text (see
        # phase_16v_implementation_audit.md for the disclosed nuance: sec.18's
        # own rationale text assumes telemetry unavailability, which does not
        # perfectly fit an actuation-only failure -- implemented literally
        # rather than silently carving out an exception, per this phase's
        # instruction to report, not silently resolve, contract ambiguities).
        reward_result = compute_reward(
            decision_state_valid=s_t_valid and not k8s_api_failure,
            outcome_state_valid=obs_result.validity.sample_valid,
            p95_latency_ms=obs_result.raw.get("p95_latency_norm"),
            error_rate=obs_result.raw.get("error_rate"),
            deployment_current_replicas=snapshot_after.replicas.current,
            cpu_per_replica=obs_result.raw.get("cpu_per_replica"),
            requested_delta=requested_delta,
            effective_delta=effective_delta,
            override_reason=override_reason,
            is_reversal=is_reversal,
            config=self.config.reward,
        )

        self._last_normalized_obs = obs_result.normalized
        self._last_validity = obs_result.validity
        self._last_snapshot = snapshot_after

        info = self._build_info(
            obs_result=obs_result, action_index=int(action), requested_delta=requested_delta,
            effective_delta=effective_delta, override_reason=override_reason,
            bound_clamped=bound_clamped, pending_scale=pending_scale,
            reward_result=reward_result, terminated=terminated, truncated=truncated,
            snapshot_before=snapshot_before, snapshot_after=snapshot_after,
            is_reversal=is_reversal,
        )
        return (
            np.array(obs_result.normalized, dtype=np.float32),
            float(reward_result.total),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self):  # pragma: no cover - no rendering support
        return None

    def close(self) -> None:
        return None
