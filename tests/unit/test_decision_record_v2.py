"""tests/unit/test_decision_record_v2.py

V2 Phase 2 PART 7/9: proves the additive V2 decision telemetry record
carries every field the task requires and never touches any frozen V1
ledger/sample file.
"""

from __future__ import annotations

from autoscaler.src.rl_env.decision_record_v2 import (
    CONTROLLER_IDENTIFIER_PPO_V2,
    DecisionRecordV2,
    build_decision_record_v2,
)
from autoscaler.src.rl_env.safe_inference import SafeInferenceResult
from autoscaler.src.rl_env.types import OVERRIDE_NONE, OVERRIDE_POLICY_INFERENCE_FAILURE

REQUIRED_FIELDS = {
    "timestamp_utc", "observation_vector",
    "raw_cpu_cores", "raw_memory_bytes", "raw_request_rate_rps", "raw_p95_latency_ms",
    "current_replicas", "ready_replicas", "desired_replicas",
    "action_probabilities", "requested_action_index", "safe_selected_action_index",
    "effective_action_index", "override_reason", "near_tie", "epsilon",
    "model_identifier", "controller_identifier",
}


def _success_inference_result() -> SafeInferenceResult:
    return SafeInferenceResult(
        effective_action_index=2, effective_action_name="SCALE_UP_ONE",
        override_reason=OVERRIDE_NONE, is_tie=False, probabilities=(0.05, 0.05, 0.9),
        epsilon=0.05, failure_detail=None,
    )


def _failure_inference_result() -> SafeInferenceResult:
    return SafeInferenceResult(
        effective_action_index=1, effective_action_name="HOLD",
        override_reason=OVERRIDE_POLICY_INFERENCE_FAILURE, is_tie=None, probabilities=None,
        epsilon=0.05, failure_detail="simulated failure",
    )


class TestRequiredFieldsPresent:
    def test_all_task_required_fields_present(self):
        record = build_decision_record_v2(
            timestamp_utc="2026-10-11T00:00:00+00:00",
            observation_vector=[0.1] * 11,
            raw_cpu_cores=0.2, raw_memory_bytes=1_000_000, raw_request_rate_rps=5.0, raw_p95_latency_ms=42.0,
            current_replicas=2, ready_replicas=2, desired_replicas=2,
            inference_result=_success_inference_result(),
            effective_action_index=2, effective_action_name="SCALE_UP_ONE",
            effective_override_reason=OVERRIDE_NONE,
            model_identifier="models/ppo/ppo_model.zip",
        )
        d = record.to_dict()
        missing = REQUIRED_FIELDS - set(d.keys())
        assert missing == set()

    def test_controller_identifier_default(self):
        record = build_decision_record_v2(
            timestamp_utc="t", observation_vector=[0.0] * 11,
            raw_cpu_cores=None, raw_memory_bytes=None, raw_request_rate_rps=None, raw_p95_latency_ms=None,
            current_replicas=1, ready_replicas=1, desired_replicas=1,
            inference_result=_success_inference_result(),
            effective_action_index=2, effective_action_name="SCALE_UP_ONE",
            effective_override_reason=OVERRIDE_NONE,
            model_identifier="m",
        )
        assert record.controller_identifier == CONTROLLER_IDENTIFIER_PPO_V2


class TestSuccessVsFailureShape:
    def test_success_record_has_requested_action_and_probabilities(self):
        record = build_decision_record_v2(
            timestamp_utc="t", observation_vector=[0.0] * 11,
            raw_cpu_cores=None, raw_memory_bytes=None, raw_request_rate_rps=None, raw_p95_latency_ms=None,
            current_replicas=1, ready_replicas=1, desired_replicas=1,
            inference_result=_success_inference_result(),
            effective_action_index=2, effective_action_name="SCALE_UP_ONE",
            effective_override_reason=OVERRIDE_NONE,
            model_identifier="m",
        )
        assert record.requested_action_index == 2
        assert record.action_probabilities == (0.05, 0.05, 0.9)
        assert record.override_reason == OVERRIDE_NONE
        assert record.near_tie is False

    def test_failure_record_has_no_requested_action_or_probabilities(self):
        record = build_decision_record_v2(
            timestamp_utc="t", observation_vector=[0.0] * 11,
            raw_cpu_cores=None, raw_memory_bytes=None, raw_request_rate_rps=None, raw_p95_latency_ms=None,
            current_replicas=1, ready_replicas=1, desired_replicas=1,
            inference_result=_failure_inference_result(),
            effective_action_index=1, effective_action_name="HOLD",
            effective_override_reason=OVERRIDE_POLICY_INFERENCE_FAILURE,
            model_identifier="m",
        )
        assert record.requested_action_index is None
        assert record.action_probabilities is None
        assert record.override_reason == OVERRIDE_POLICY_INFERENCE_FAILURE
        assert record.safe_selected_action_index == 1
        assert record.near_tie is None

    def test_record_is_frozen_dataclass(self):
        record = build_decision_record_v2(
            timestamp_utc="t", observation_vector=[0.0] * 11,
            raw_cpu_cores=None, raw_memory_bytes=None, raw_request_rate_rps=None, raw_p95_latency_ms=None,
            current_replicas=1, ready_replicas=1, desired_replicas=1,
            inference_result=_success_inference_result(),
            effective_action_index=2, effective_action_name="SCALE_UP_ONE",
            effective_override_reason=OVERRIDE_NONE,
            model_identifier="m",
        )
        assert isinstance(record, DecisionRecordV2)
        with __import__("pytest").raises(Exception):
            record.override_reason = "tampered"  # frozen dataclass must reject mutation
