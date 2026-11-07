"""autoscaler/src/rl_env/safe_ppo_controller.py

V2 Phase 2 PART 5: the thin, reusable V2 controller step function that
`scripts/evaluate_ppo_vs_hpa_v2.py` (and, later, a V2 demo/UI path) calls
once per decision step. Composes three already-independently-tested V2
modules -- `safe_inference` (observation validation + safe action
selection, including policy-inference-failure handling), the existing
`KubernetesAutoscalerEnv.step()` (gating/actuation, UNCHANGED, frozen since
Phase 15B.1), and `decision_record_v2` (structured telemetry) -- into the
exact flow the task requires:

    observation -> validate finite/shape -> extract PPO probabilities
    -> safe action selector -> safety/gating -> scale action
    -> structured decision record

This module has NO Kubernetes/Prometheus dependency of its own -- it is
driven entirely by an injected `KubernetesAutoscalerEnv` (which may wrap
`ScriptedBackend` for offline tests or `LiveKubernetesBackend` for a real
cluster) and an injected model (or None, which safely degrades every step
to HOLD via `safe_inference.infer_safe_action`).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .decision_record_v2 import DecisionRecordV2, build_decision_record_v2
from .environment import KubernetesAutoscalerEnv
from .safe_inference import SafeInferenceResult, infer_safe_action
from .types import OVERRIDE_POLICY_INFERENCE_FAILURE


def run_safe_ppo_step(
    env: KubernetesAutoscalerEnv,
    model: Any,
    observation: np.ndarray,
    model_identifier: str,
    epsilon: float,
    timestamp_utc_fn=None,
) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any], DecisionRecordV2]:
    """Runs exactly one V2-safe decision step. Returns the same 5-tuple
    `env.step()` returns (obs, reward, terminated, truncated, info) PLUS the
    `DecisionRecordV2` for this step, so a caller can both drive the episode
    loop and persist/stream the structured telemetry without recomputing
    anything.

    On a policy-inference failure (model is None, invalid/NaN observation,
    probability-extraction failure, or a malformed/NaN probability vector),
    `inference_result.effective_action_index` is already `HOLD_ACTION_INDEX`
    (safe_inference.py's own guarantee) -- this function passes that index
    straight to `env.step()` UNCHANGED, so the environment's own gating
    trivially sees `requested_delta == 0` and performs NO replica mutation,
    regardless of whatever the policy failure was. The decision record's
    `override_reason` reports `policy_inference_failure` in this case
    (taking priority over the environment's own `none` gating outcome,
    since a requested-delta-of-zero HOLD is gating's correct response to
    EITHER a genuine HOLD action OR a safety fallback -- the record
    preserves which one actually happened)."""
    from datetime import datetime, timezone

    _now = timestamp_utc_fn or (lambda: datetime.now(timezone.utc).isoformat())

    inference_result: SafeInferenceResult = infer_safe_action(model, observation, epsilon=epsilon)
    action_index = inference_result.effective_action_index

    next_obs, reward, terminated, truncated, info = env.step(action_index)

    policy_failed = inference_result.override_reason == OVERRIDE_POLICY_INFERENCE_FAILURE
    effective_override_reason = (
        OVERRIDE_POLICY_INFERENCE_FAILURE if policy_failed else info["action_override_reason"]
    )

    raw = info.get("raw_observation", {}) or {}
    replicas_after_key = "deployment_desired_replicas_after"
    decision_record = build_decision_record_v2(
        timestamp_utc=_now(),
        observation_vector=list(observation.tolist() if hasattr(observation, "tolist") else observation),
        raw_cpu_cores=raw.get("cpu_per_replica"),
        raw_memory_bytes=raw.get("memory_per_replica"),
        raw_request_rate_rps=raw.get("request_rate_norm"),
        raw_p95_latency_ms=raw.get("p95_latency_norm"),
        current_replicas=info.get("deployment_current_replicas_after", 0),
        ready_replicas=info.get("ready_replicas_after", 0),
        desired_replicas=info.get(replicas_after_key, 0),
        inference_result=inference_result,
        effective_action_index=action_index if not policy_failed else action_index,
        effective_action_name=info.get("semantic_action") or inference_result.effective_action_name,
        effective_override_reason=effective_override_reason,
        model_identifier=model_identifier,
        extra={"failure_detail": inference_result.failure_detail} if policy_failed else {},
    )

    return next_obs, reward, terminated, truncated, info, decision_record
