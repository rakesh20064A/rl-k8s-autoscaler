"""autoscaler/src/rl_env/safe_inference.py

V2 Phase 2 PART 5/6: wires the previously-declared-but-unused
`OVERRIDE_POLICY_INFERENCE_FAILURE` (types.py) into a real, reusable safe
inference path, WITHOUT modifying the frozen V1 evaluation script
(scripts/evaluate_ppo_vs_hpa.py, which calls `model.predict(obs,
deterministic=True)` directly, with no try/except around it -- see the V2
Phase 2 audit) or `environment.py`/`types.py` themselves.

Flow (per the task's required shape):
    observation -> validate finite/shape -> extract PPO probabilities
    -> safe action selector -> [caller applies gating/actuation]
    -> structured decision record

Every one of the six failure modes the task enumerates collapses to the
SAME safe outcome here: `effective_action_index = HOLD_ACTION_INDEX`,
`override_reason = OVERRIDE_POLICY_INFERENCE_FAILURE`, and -- because HOLD's
`ACTION_MAP` delta is 0 -- `KubernetesAutoscalerEnv`'s own gating
(gating.py's `requested_delta == 0` branch) and actuation path trivially
produce NO replica mutation when the caller passes this action through
unmodified. This module never calls `env.step()` itself (no Kubernetes/
environment dependency here) -- it only decides WHICH action index is safe
to pass to it.

Pure/offline where possible: `infer_safe_action()` takes an already-loaded
model (or None) and an observation array; no I/O. `load_ppo_model_safe()`
is the one function that touches the filesystem (loading the .zip
checkpoint), isolated so a model-load failure is caught at a single,
testable boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .action_selection import (
    DEFAULT_NEAR_TIE_EPSILON,
    HOLD_ACTION_INDEX,
    TIE_RULE_UNKNOWN_STRESS_FALLBACK,
    determine_decision_time_stress,
    extract_action_probabilities,
    select_safety_aware_action,
)
from .reward_v2 import RewardConfigV2, load_reward_config_v2_from_yaml
from .types import ACTION_NAMES, OVERRIDE_NONE, OVERRIDE_POLICY_INFERENCE_FAILURE

EXPECTED_OBSERVATION_SHAPE = (11,)

_DEFAULT_REWARD_CONFIG_CACHE: Optional[RewardConfigV2] = None


def _default_reward_config() -> RewardConfigV2:
    """Lazily loads (and caches for the process lifetime) the real
    `configs/reward_v2.yaml` -- only used when a caller does not pass its
    own `reward_config` to `infer_safe_action()`. Cached rather than
    re-read on every call, but never cached across a config CHANGE within
    the same process (this module is not used inside any long-running
    config-reload loop)."""
    global _DEFAULT_REWARD_CONFIG_CACHE
    if _DEFAULT_REWARD_CONFIG_CACHE is None:
        _DEFAULT_REWARD_CONFIG_CACHE = load_reward_config_v2_from_yaml()
    return _DEFAULT_REWARD_CONFIG_CACHE


@dataclass(frozen=True)
class SafeInferenceResult:
    """Structured outcome of one safe-inference attempt. `probabilities` and
    `is_tie` are None on failure (there is no valid distribution to report).
    `effective_action_index`/`effective_action_name` are ALWAYS populated --
    HOLD on any failure, the selector's own choice otherwise -- so a caller
    never needs a separate null-check before using the action.

    V2 Phase 4D-E / CORR-58 (additive): `tie_set`/`tie_rule_applied`/
    `stressed`/`stress_reasons` report the safety-aware tie-break's own
    reasoning -- `tie_rule_applied` is `None` only when `is_tie` is also
    falsy/None (no tie was ever evaluated, e.g. a hard failure before any
    probabilities existed)."""

    effective_action_index: int
    effective_action_name: str
    override_reason: str  # OVERRIDE_NONE on success, OVERRIDE_POLICY_INFERENCE_FAILURE on any failure
    is_tie: Optional[bool]
    probabilities: Optional[tuple]
    epsilon: float
    failure_detail: Optional[str]  # human-readable cause, only set on failure; never exposed as a replica mutation
    tie_set: Optional[Tuple[str, ...]] = None
    tie_rule_applied: Optional[str] = None
    stressed: Optional[bool] = None
    stress_reasons: Tuple[str, ...] = ()


def validate_observation(observation: Sequence[float]) -> None:
    """Raises ValueError if `observation` is not a length-11 vector of
    finite values. Never clips/repairs -- an invalid observation must
    surface as a failure, not a silently-corrected input."""
    arr = np.asarray(observation, dtype=float)
    if arr.shape != EXPECTED_OBSERVATION_SHAPE:
        raise ValueError(f"expected observation shape {EXPECTED_OBSERVATION_SHAPE}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"observation contains non-finite value(s): {arr.tolist()}")


def load_ppo_model_safe(model_path) -> Optional[object]:
    """Loads an SB3 PPO checkpoint, returning None (never raising) on any
    failure -- missing file, corrupt zip, incompatible version, etc. A
    caller passing None into `infer_safe_action()` gets the same safe HOLD/
    policy_inference_failure outcome as every other failure mode here,
    without ever needing its own separate model-load error handling."""
    try:
        from stable_baselines3 import PPO

        return PPO.load(model_path)
    except Exception:  # noqa: BLE001 -- any load failure must degrade to HOLD, never propagate
        return None


def infer_safe_action(
    model: Optional[object],
    observation: Sequence[float],
    epsilon: float = DEFAULT_NEAR_TIE_EPSILON,
    decision_time_raw_observation: Optional[Dict[str, Any]] = None,
    reward_config: Optional[RewardConfigV2] = None,
) -> SafeInferenceResult:
    """The single entry point covering every failure mode in the task's
    PART 6: model is None (load failure), invalid/NaN/Inf observation,
    probability-extraction failure, NaN/Inf or malformed probabilities.
    ALL of them produce `HOLD_ACTION_INDEX` /
    `OVERRIDE_POLICY_INFERENCE_FAILURE` -- never `SCALE_DOWN`, never a
    silent default that happens to coincide with SCALE_DOWN's index.

    V2 Phase 4D-E / CORR-58: uses the safety-aware tie-break
    (`select_safety_aware_action`) instead of the plain near-tie-HOLD rule.
    `decision_time_raw_observation` (a dict with `p95_latency_ms`,
    `error_rate`, `cpu_per_replica`, `memory_per_replica` keys -- same
    names `environment_v2.py`'s own `info["raw_observation"]` uses) is the
    RAW state that produced `observation` (the normalized vector), used
    only to determine STRESSED; omitting it (or passing one with all four
    values missing) falls back to R6 (HOLD) for an otherwise-R3-eligible
    tie, exactly like any other missing-telemetry case elsewhere in this
    module. `reward_config` defaults to the real, live
    `configs/reward_v2.yaml` (lazily loaded, cached) if not given --
    callers that already have a loaded `RewardConfigV2` should pass it to
    avoid a redundant file read per call."""
    if model is None:
        return _failure_result(epsilon, "model is None (load failure or not yet loaded)")

    try:
        validate_observation(observation)
    except ValueError as exc:
        return _failure_result(epsilon, f"invalid observation: {exc}")

    try:
        probabilities = extract_action_probabilities(model, np.asarray(observation, dtype=np.float32))
    except Exception as exc:  # noqa: BLE001 -- any extraction failure (device/shape/forward-pass error) -> HOLD
        return _failure_result(epsilon, f"probability extraction failed: {exc}")

    cfg = reward_config if reward_config is not None else _default_reward_config()
    raw = decision_time_raw_observation or {}
    stressed, stress_reasons = determine_decision_time_stress(
        p95_latency_ms=raw.get("p95_latency_ms"),
        error_rate=raw.get("error_rate"),
        cpu_per_replica=raw.get("cpu_per_replica"),
        memory_per_replica=raw.get("memory_per_replica"),
        sla_p95_target_ms=cfg.sla_p95_target_ms,
        cpu_safe_high=cfg.cpu_safe_high,
        memory_safe_high=cfg.mem_safe_high,
    )

    try:
        safe_result = select_safety_aware_action(
            probabilities, epsilon=epsilon, stressed=stressed, stress_reasons=stress_reasons,
        )
    except ValueError as exc:
        # Covers NaN/Inf probabilities, wrong-length vectors, negative
        # values -- select_safety_aware_action's own validation.
        return _failure_result(epsilon, f"invalid probability vector: {exc}")

    return SafeInferenceResult(
        effective_action_index=safe_result.action_index,
        effective_action_name=safe_result.action_name,
        override_reason=OVERRIDE_NONE,
        is_tie=safe_result.is_tie,
        probabilities=safe_result.probabilities,
        epsilon=epsilon,
        failure_detail=None,
        tie_set=safe_result.tie_set,
        tie_rule_applied=safe_result.tie_rule_applied,
        stressed=safe_result.stressed,
        stress_reasons=safe_result.stress_reasons,
    )


def _failure_result(epsilon: float, detail: str) -> SafeInferenceResult:
    return SafeInferenceResult(
        effective_action_index=HOLD_ACTION_INDEX,
        effective_action_name=ACTION_NAMES[HOLD_ACTION_INDEX],
        override_reason=OVERRIDE_POLICY_INFERENCE_FAILURE,
        is_tie=None,
        probabilities=None,
        epsilon=epsilon,
        failure_detail=detail,
        tie_set=None,
        tie_rule_applied=TIE_RULE_UNKNOWN_STRESS_FALLBACK,
        stressed=None,
        stress_reasons=(),
    )
