"""tests/unit/test_safety_aware_tie_break.py

V2 Phase 4D-E / CORR-58: the asymmetric safety-aware tie-break rule
(autoscaler/src/rl_env/action_selection.py's `select_safety_aware_action`/
`determine_decision_time_stress`), wired through `safe_inference.py` and
`safe_ppo_controller.py`. Inference-only -- training must never call any of
this (verified by the extended training-exclusion guard at the bottom).
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from autoscaler.src.rl_env.action_selection import (
    DOWN_ACTION_INDEX,
    HOLD_ACTION_INDEX,
    TIE_RULE_CLEAR_WINNER,
    TIE_RULE_DOWN_HOLD,
    TIE_RULE_DOWN_INVOLVES_UP,
    TIE_RULE_HOLD_UP_STRESS_DEPENDENT,
    TIE_RULE_UNKNOWN_STRESS_FALLBACK,
    UP_ACTION_INDEX,
    determine_decision_time_stress,
    select_safety_aware_action,
)
from autoscaler.src.rl_env.reward_v2 import RewardConfigV2, load_reward_config_v2_from_yaml
from autoscaler.src.rl_env.safe_inference import infer_safe_action

EPSILON = 0.05


class _StubModel:
    """Exposes exactly what safe_inference.extract_action_probabilities()
    needs to be monkeypatched around -- see `_patched_extractor` fixture."""


@pytest.fixture
def _patched_extractor(monkeypatch):
    probs_holder = {"probs": None}
    import autoscaler.src.rl_env.safe_inference as safe_inference_mod

    def _fake_extract(model, observation):
        return np.asarray(probs_holder["probs"], dtype=float)

    monkeypatch.setattr(safe_inference_mod, "extract_action_probabilities", _fake_extract)
    return probs_holder


# -- Rule matrix: tie combinations x stressed/not stressed -------------------


class TestClearWinners:
    @pytest.mark.parametrize("probs,expected_index", [
        ((0.8, 0.1, 0.1), DOWN_ACTION_INDEX),
        ((0.1, 0.8, 0.1), HOLD_ACTION_INDEX),
        ((0.1, 0.1, 0.8), UP_ACTION_INDEX),
    ])
    def test_clear_winner_for_each_action(self, probs, expected_index):
        result = select_safety_aware_action(probs, epsilon=EPSILON)
        assert result.action_index == expected_index
        assert result.tie_rule_applied == TIE_RULE_CLEAR_WINNER
        assert result.is_tie is False


class TestTieCombinations:
    def test_exact_three_way_tie(self):
        result = select_safety_aware_action((1 / 3, 1 / 3, 1 / 3), epsilon=EPSILON)
        assert result.action_index == HOLD_ACTION_INDEX
        assert result.tie_rule_applied == TIE_RULE_DOWN_INVOLVES_UP
        assert set(result.tie_set) == {"SCALE_DOWN_ONE", "HOLD", "SCALE_UP_ONE"}

    def test_down_hold_tie(self):
        result = select_safety_aware_action((0.5, 0.5, 0.0), epsilon=EPSILON)
        assert result.action_index == HOLD_ACTION_INDEX
        assert result.tie_rule_applied == TIE_RULE_DOWN_HOLD
        assert set(result.tie_set) == {"SCALE_DOWN_ONE", "HOLD"}

    def test_down_up_tie(self):
        result = select_safety_aware_action((0.5, 0.0, 0.5), epsilon=EPSILON)
        assert result.action_index == HOLD_ACTION_INDEX
        assert result.tie_rule_applied == TIE_RULE_DOWN_INVOLVES_UP
        assert set(result.tie_set) == {"SCALE_DOWN_ONE", "SCALE_UP_ONE"}

    def test_hold_up_tie_not_stressed(self):
        result = select_safety_aware_action((0.0, 0.5, 0.5), epsilon=EPSILON, stressed=False)
        assert result.action_index == HOLD_ACTION_INDEX
        assert result.tie_rule_applied == TIE_RULE_HOLD_UP_STRESS_DEPENDENT
        assert result.stressed is False

    def test_hold_up_tie_stressed(self):
        result = select_safety_aware_action(
            (0.0, 0.5, 0.5), epsilon=EPSILON, stressed=True, stress_reasons=("error_rate>0",),
        )
        assert result.action_index == UP_ACTION_INDEX
        assert result.tie_rule_applied == TIE_RULE_HOLD_UP_STRESS_DEPENDENT
        assert result.stressed is True
        assert result.stress_reasons == ("error_rate>0",)

    def test_hold_up_tie_unknown_stress_falls_back_to_r6(self):
        result = select_safety_aware_action((0.0, 0.5, 0.5), epsilon=EPSILON, stressed=None)
        assert result.action_index == HOLD_ACTION_INDEX
        assert result.tie_rule_applied == TIE_RULE_UNKNOWN_STRESS_FALLBACK


class TestEpsilonBoundaryCases:
    def test_exactly_epsilon_apart_is_a_tie(self):
        # max=0.50, epsilon=0.05 -> threshold is exactly 0.45; a candidate
        # AT the threshold (>=) counts as within epsilon -- a tie.
        result = select_safety_aware_action((0.45, 0.05, 0.50), epsilon=0.05)
        assert result.is_tie is True
        assert set(result.tie_set) == {"SCALE_DOWN_ONE", "SCALE_UP_ONE"}

    def test_just_inside_epsilon_is_a_tie(self):
        # max=0.5, epsilon=0.05 -> threshold 0.45; 0.46 is within.
        result = select_safety_aware_action((0.46, 0.04, 0.50), epsilon=0.05)
        assert result.is_tie is True

    def test_just_outside_epsilon_is_a_clear_winner(self):
        # max=0.50, epsilon=0.05 -> threshold 0.45; 0.44 is NOT within.
        result = select_safety_aware_action((0.44, 0.06, 0.50), epsilon=0.05)
        assert result.is_tie is False
        assert result.action_index == UP_ACTION_INDEX

    def test_epsilon_zero_only_exact_tie(self):
        result = select_safety_aware_action((0.5, 0.5, 0.0), epsilon=0.0)
        assert result.is_tie is True
        result2 = select_safety_aware_action((0.5000001, 0.4999999, 0.0), epsilon=0.0)
        assert result2.is_tie is False


# -- DOWN never wins a tie, property-style -----------------------------------


class TestDownNeverWinsATie:
    def test_down_never_selected_from_any_tie_random_vectors(self):
        rng = random.Random(42)
        for _ in range(2000):
            raw = [rng.random() for _ in range(3)]
            total = sum(raw)
            probs = tuple(v / total for v in raw)
            epsilon = rng.choice([0.0, 0.01, 0.05, 0.1, 0.2])
            stressed = rng.choice([True, False, None])
            result = select_safety_aware_action(probs, epsilon=epsilon, stressed=stressed)
            if result.is_tie:
                assert result.action_index != DOWN_ACTION_INDEX, (probs, epsilon, stressed, result)

    def test_down_never_selected_even_when_down_has_the_highest_raw_probability_in_a_tie(self):
        """DOWN tied at the TOP (with HOLD or UP) must still never win."""
        result = select_safety_aware_action((0.5, 0.5, 0.0), epsilon=0.05)
        assert result.action_index != DOWN_ACTION_INDEX
        result2 = select_safety_aware_action((0.5, 0.0, 0.5), epsilon=0.05)
        assert result2.action_index != DOWN_ACTION_INDEX


# -- Thresholds sourced from configs/reward_v2.yaml, not hardcoded -----------


class TestThresholdsFromRewardV2Yaml:
    def test_stress_determination_uses_live_config_thresholds(self):
        cfg = load_reward_config_v2_from_yaml()
        # p95 exactly at the live SLA target is NOT >= unless equal; use target itself.
        stressed, reasons = determine_decision_time_stress(
            p95_latency_ms=cfg.sla_p95_target_ms, error_rate=0.0, cpu_per_replica=0.1, memory_per_replica=0.1,
            sla_p95_target_ms=cfg.sla_p95_target_ms, cpu_safe_high=cfg.cpu_safe_high, memory_safe_high=cfg.mem_safe_high,
        )
        assert stressed is True
        assert "p95_latency_ms>=sla_p95_target_ms" in reasons

    def test_changing_the_loaded_config_changes_the_outcome(self):
        """Proves the thresholds are genuinely read from the config object,
        not a hardcoded constant -- a config with a much higher SLA target
        makes the SAME p95 value no longer stressed."""
        lenient_cfg = RewardConfigV2(sla_p95_target_ms=5000.0, cpu_safe_high=100.0, mem_safe_high=100.0)
        stressed, reasons = determine_decision_time_stress(
            p95_latency_ms=1200.0, error_rate=0.0, cpu_per_replica=0.1, memory_per_replica=0.1,
            sla_p95_target_ms=lenient_cfg.sla_p95_target_ms, cpu_safe_high=lenient_cfg.cpu_safe_high,
            memory_safe_high=lenient_cfg.mem_safe_high,
        )
        assert stressed is False
        assert reasons == ()

    def test_error_rate_and_pressure_thresholds_independently_trigger_stress(self):
        cfg = load_reward_config_v2_from_yaml()
        stressed, reasons = determine_decision_time_stress(
            p95_latency_ms=10.0, error_rate=0.01, cpu_per_replica=0.1, memory_per_replica=0.1,
            sla_p95_target_ms=cfg.sla_p95_target_ms, cpu_safe_high=cfg.cpu_safe_high, memory_safe_high=cfg.mem_safe_high,
        )
        assert stressed is True
        assert reasons == ("error_rate>0",)

        stressed2, reasons2 = determine_decision_time_stress(
            p95_latency_ms=10.0, error_rate=0.0, cpu_per_replica=cfg.cpu_safe_high + 0.1, memory_per_replica=0.1,
            sla_p95_target_ms=cfg.sla_p95_target_ms, cpu_safe_high=cfg.cpu_safe_high, memory_safe_high=cfg.mem_safe_high,
        )
        assert stressed2 is True
        assert reasons2 == ("cpu_per_replica>cpu_safe_high",)

    def test_all_four_missing_cannot_determine_stress(self):
        cfg = load_reward_config_v2_from_yaml()
        stressed, reasons = determine_decision_time_stress(
            p95_latency_ms=None, error_rate=None, cpu_per_replica=None, memory_per_replica=None,
            sla_p95_target_ms=cfg.sla_p95_target_ms, cpu_safe_high=cfg.cpu_safe_high, memory_safe_high=cfg.mem_safe_high,
        )
        assert stressed is None
        assert reasons == ()

    def test_one_present_value_is_enough_to_determine_not_stressed(self):
        cfg = load_reward_config_v2_from_yaml()
        stressed, reasons = determine_decision_time_stress(
            p95_latency_ms=10.0, error_rate=None, cpu_per_replica=None, memory_per_replica=None,
            sla_p95_target_ms=cfg.sla_p95_target_ms, cpu_safe_high=cfg.cpu_safe_high, memory_safe_high=cfg.mem_safe_high,
        )
        assert stressed is False
        assert reasons == ()


# -- infer_safe_action() end-to-end -------------------------------------------


class TestInferSafeActionWiring:
    def test_hold_up_tie_resolves_using_decision_time_raw_observation(self, _patched_extractor):
        _patched_extractor["probs"] = (0.0, 0.5, 0.5)
        cfg = load_reward_config_v2_from_yaml()
        result = infer_safe_action(
            model=_StubModel(), observation=np.zeros(11), epsilon=EPSILON,
            decision_time_raw_observation={"p95_latency_ms": cfg.sla_p95_target_ms + 500, "error_rate": 0.0,
                                            "cpu_per_replica": 0.1, "memory_per_replica": 0.1},
            reward_config=cfg,
        )
        assert result.effective_action_index == UP_ACTION_INDEX
        assert result.stressed is True
        assert result.tie_rule_applied == TIE_RULE_HOLD_UP_STRESS_DEPENDENT

    def test_hold_up_tie_without_raw_observation_falls_back_to_hold(self, _patched_extractor):
        _patched_extractor["probs"] = (0.0, 0.5, 0.5)
        result = infer_safe_action(model=_StubModel(), observation=np.zeros(11), epsilon=EPSILON)
        assert result.effective_action_index == HOLD_ACTION_INDEX
        assert result.tie_rule_applied == TIE_RULE_UNKNOWN_STRESS_FALLBACK

    def test_model_none_is_r6_hold(self):
        result = infer_safe_action(model=None, observation=np.zeros(11), epsilon=EPSILON)
        assert result.effective_action_index == HOLD_ACTION_INDEX
        assert result.tie_rule_applied == TIE_RULE_UNKNOWN_STRESS_FALLBACK
        assert result.probabilities is None

    def test_invalid_observation_is_r6_hold(self):
        result = infer_safe_action(model=_StubModel(), observation=[float("nan")] * 11, epsilon=EPSILON)
        assert result.effective_action_index == HOLD_ACTION_INDEX
        assert result.tie_rule_applied == TIE_RULE_UNKNOWN_STRESS_FALLBACK

    def test_decision_record_additive_fields_populated(self, _patched_extractor):
        from autoscaler.src.rl_env.decision_record_v2 import build_decision_record_v2

        _patched_extractor["probs"] = (0.0, 0.5, 0.5)
        result = infer_safe_action(
            model=_StubModel(), observation=np.zeros(11), epsilon=EPSILON,
            decision_time_raw_observation={"p95_latency_ms": 50.0, "error_rate": 0.0,
                                            "cpu_per_replica": 0.1, "memory_per_replica": 0.1},
        )
        record = build_decision_record_v2(
            timestamp_utc="t", observation_vector=[0.0] * 11,
            raw_cpu_cores=None, raw_memory_bytes=None, raw_request_rate_rps=None, raw_p95_latency_ms=None,
            current_replicas=2, ready_replicas=2, desired_replicas=2,
            inference_result=result, effective_action_index=result.effective_action_index,
            effective_action_name=result.effective_action_name, effective_override_reason=result.override_reason,
            model_identifier="stub",
        )
        assert record.tie_set == ("HOLD", "SCALE_UP_ONE")
        assert record.tie_rule_applied == TIE_RULE_HOLD_UP_STRESS_DEPENDENT
        assert record.stressed is False
        # raw_argmax_* fields still populated exactly as before (unchanged behavior).
        assert record.raw_argmax_action_name in ("HOLD", "SCALE_UP_ONE")


# -- Training exclusion (extended to cover the V2 training scripts too) -----


class TestTrainingNeverUsesSafetyAwareSelector:
    """Extends the existing structural guard
    (tests/unit/test_training_excludes_safe_tie_override.py) to explicitly
    confirm the NEW safety-aware functions are absent from every training
    entry point, V1 and V2 alike."""

    @pytest.mark.parametrize("path", [
        "scripts/train_ppo_live.py",
        "scripts/train_ppo_live_session.py",
        "scripts/train_ppo_final_v2.py",
        "scripts/train_ppo_live_v2.py",
        "autoscaler/src/rl_env/training_wrapper.py",
    ])
    def test_no_training_script_references_the_safety_aware_selector(self, path):
        source = open(path, encoding="utf-8").read()
        for forbidden in (
            "select_safety_aware_action", "determine_decision_time_stress",
            "select_safe_deterministic_action", "infer_safe_action", "safe_ppo_controller",
        ):
            assert forbidden not in source, f"{path} references {forbidden!r}"
