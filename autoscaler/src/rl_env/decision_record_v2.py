"""autoscaler/src/rl_env/decision_record_v2.py

V2 Phase 2 PART 7: an ADDITIVE, V2-only structured decision record for one
PPO decision step. Does NOT read, write, or alter any frozen V1 ledger/
sample file (results/, task_outputs/phase_16u*, the
`results/evaluation/ledgers/*.jsonl` files scripts/evaluate_ppo_vs_hpa.py
appends to, or that script's own `_snapshot_dict()`/`_summarize_arm()`
shapes) -- this is a NEW, parallel record shape for the future V2 UI,
produced by the new V2 inference path (safe_inference.py) and a V2 eval/
demo script, never by the frozen V1 one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

CONTROLLER_IDENTIFIER_PPO_V2 = "ppo_v2_safe_inference"


@dataclass(frozen=True)
class DecisionRecordV2:
    """One decision step's complete, self-contained record. All fields the
    task's PART 7 lists are present; nothing here is derived by the reader
    from some other file -- this record stands alone."""

    timestamp_utc: str
    observation_vector: List[float]  # the normalized 11-element vector fed to the model

    # Raw (pre-normalization) important metrics, for human/UI readability --
    # NOT a substitute for observation_vector (the model's actual input).
    raw_cpu_cores: Optional[float]
    raw_memory_bytes: Optional[float]
    raw_request_rate_rps: Optional[float]
    raw_p95_latency_ms: Optional[float]

    current_replicas: int
    ready_replicas: int
    desired_replicas: int

    action_probabilities: Optional[tuple]  # None only on a policy_inference_failure step
    requested_action_index: Optional[int]  # the selector's own choice; None on failure
    requested_action_name: Optional[str]
    safe_selected_action_index: int  # ALWAYS populated -- HOLD on failure, selector's choice otherwise
    safe_selected_action_name: str
    effective_action_index: int  # after gating/actuation (set by the caller once known)
    effective_action_name: str
    override_reason: str  # one of autoscaler.src.rl_env.types.ALL_OVERRIDE_REASONS
    near_tie: Optional[bool]  # None on a policy_inference_failure step (no distribution to compare)
    epsilon: float

    model_identifier: str
    controller_identifier: str = CONTROLLER_IDENTIFIER_PPO_V2
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_decision_record_v2(
    *,
    timestamp_utc: str,
    observation_vector: List[float],
    raw_cpu_cores: Optional[float],
    raw_memory_bytes: Optional[float],
    raw_request_rate_rps: Optional[float],
    raw_p95_latency_ms: Optional[float],
    current_replicas: int,
    ready_replicas: int,
    desired_replicas: int,
    inference_result: "object",  # autoscaler.src.rl_env.safe_inference.SafeInferenceResult, typed loosely to avoid a hard import-order dependency
    effective_action_index: int,
    effective_action_name: str,
    effective_override_reason: str,
    model_identifier: str,
    controller_identifier: str = CONTROLLER_IDENTIFIER_PPO_V2,
    extra: Optional[Dict[str, Any]] = None,
) -> DecisionRecordV2:
    """Assembles a `DecisionRecordV2` from a `SafeInferenceResult`
    (safe_inference.py) plus the surrounding step's raw telemetry/replica
    state and the GATED effective outcome (which may differ from the
    selector's own `effective_action_index` on inference_result -- e.g. a
    clear SCALE_UP winner that gating then defers via
    `pending_scale_gate` -- hence these are passed separately rather than
    re-derived here)."""
    is_failure = inference_result.override_reason != "none"
    return DecisionRecordV2(
        timestamp_utc=timestamp_utc,
        observation_vector=list(observation_vector),
        raw_cpu_cores=raw_cpu_cores,
        raw_memory_bytes=raw_memory_bytes,
        raw_request_rate_rps=raw_request_rate_rps,
        raw_p95_latency_ms=raw_p95_latency_ms,
        current_replicas=current_replicas,
        ready_replicas=ready_replicas,
        desired_replicas=desired_replicas,
        action_probabilities=inference_result.probabilities,
        requested_action_index=None if is_failure else inference_result.effective_action_index,
        requested_action_name=None if is_failure else inference_result.effective_action_name,
        safe_selected_action_index=inference_result.effective_action_index,
        safe_selected_action_name=inference_result.effective_action_name,
        effective_action_index=effective_action_index,
        effective_action_name=effective_action_name,
        override_reason=effective_override_reason,
        near_tie=inference_result.is_tie,
        epsilon=inference_result.epsilon,
        model_identifier=model_identifier,
        controller_identifier=controller_identifier,
        extra=extra or {},
    )
