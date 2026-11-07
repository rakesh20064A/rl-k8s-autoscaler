"""tests/unit/test_safe_inference.py

V2 Phase 2 PART 6/9: wires OVERRIDE_POLICY_INFERENCE_FAILURE (previously
declared-but-unused, see types.py) into a real, tested safety path. Every
failure mode the task enumerates must degrade to HOLD, never SCALE_DOWN.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from autoscaler.src.rl_env.action_selection import DEFAULT_NEAR_TIE_EPSILON, HOLD_ACTION_INDEX
from autoscaler.src.rl_env.safe_inference import (
    infer_safe_action,
    load_ppo_model_safe,
    validate_observation,
)
from autoscaler.src.rl_env.types import OVERRIDE_NONE, OVERRIDE_POLICY_INFERENCE_FAILURE

VALID_OBS = [0.5] * 11


class _FakeModelReturning:
    """Stand-in for an SB3 model whose `.policy` would be used by
    `extract_action_probabilities` -- tests patch that function directly
    rather than build a real torch policy, so this class only needs to be a
    non-None object."""


def _patch_probs(monkeypatch, probs):
    import autoscaler.src.rl_env.safe_inference as mod

    monkeypatch.setattr(mod, "extract_action_probabilities", lambda model, obs: np.asarray(probs, dtype=float))


class TestValidateObservation:
    def test_valid_observation_passes(self):
        validate_observation(VALID_OBS)  # must not raise

    def test_wrong_length_rejected(self):
        with pytest.raises(ValueError):
            validate_observation([0.5, 0.5])

    def test_nan_rejected(self):
        bad = list(VALID_OBS)
        bad[3] = math.nan
        with pytest.raises(ValueError):
            validate_observation(bad)

    def test_inf_rejected(self):
        bad = list(VALID_OBS)
        bad[7] = math.inf
        with pytest.raises(ValueError):
            validate_observation(bad)


class TestModelLoadFailure:
    def test_missing_file_returns_none_not_raise(self, tmp_path):
        result = load_ppo_model_safe(tmp_path / "does_not_exist.zip")
        assert result is None

    def test_none_model_infers_hold_with_policy_inference_failure(self):
        result = infer_safe_action(None, VALID_OBS)
        assert result.effective_action_index == HOLD_ACTION_INDEX
        assert result.override_reason == OVERRIDE_POLICY_INFERENCE_FAILURE
        assert result.effective_action_name == "HOLD"
        assert result.probabilities is None
        assert result.failure_detail is not None


class TestInvalidObservationFailure:
    def test_nan_observation_yields_hold(self):
        bad = list(VALID_OBS)
        bad[0] = math.nan
        result = infer_safe_action(_FakeModelReturning(), bad)
        assert result.effective_action_index == HOLD_ACTION_INDEX
        assert result.override_reason == OVERRIDE_POLICY_INFERENCE_FAILURE

    def test_inf_observation_yields_hold(self):
        bad = list(VALID_OBS)
        bad[0] = math.inf
        result = infer_safe_action(_FakeModelReturning(), bad)
        assert result.effective_action_index == HOLD_ACTION_INDEX
        assert result.override_reason == OVERRIDE_POLICY_INFERENCE_FAILURE

    def test_wrong_shape_observation_yields_hold(self):
        result = infer_safe_action(_FakeModelReturning(), [0.1, 0.2])
        assert result.effective_action_index == HOLD_ACTION_INDEX
        assert result.override_reason == OVERRIDE_POLICY_INFERENCE_FAILURE


class TestProbabilityExtractionFailure:
    def test_extraction_exception_yields_hold(self, monkeypatch):
        import autoscaler.src.rl_env.safe_inference as mod

        def _raise(model, obs):
            raise RuntimeError("simulated forward-pass failure")

        monkeypatch.setattr(mod, "extract_action_probabilities", _raise)
        result = infer_safe_action(_FakeModelReturning(), VALID_OBS)
        assert result.effective_action_index == HOLD_ACTION_INDEX
        assert result.override_reason == OVERRIDE_POLICY_INFERENCE_FAILURE
        assert "simulated forward-pass failure" in result.failure_detail


class TestInvalidProbabilityFailure:
    def test_nan_probabilities_yield_hold(self, monkeypatch):
        _patch_probs(monkeypatch, [math.nan, 0.5, 0.5])
        result = infer_safe_action(_FakeModelReturning(), VALID_OBS)
        assert result.effective_action_index == HOLD_ACTION_INDEX
        assert result.override_reason == OVERRIDE_POLICY_INFERENCE_FAILURE

    def test_inf_probabilities_yield_hold(self, monkeypatch):
        _patch_probs(monkeypatch, [math.inf, 0.5, 0.5])
        result = infer_safe_action(_FakeModelReturning(), VALID_OBS)
        assert result.effective_action_index == HOLD_ACTION_INDEX
        assert result.override_reason == OVERRIDE_POLICY_INFERENCE_FAILURE

    def test_malformed_length_probabilities_yield_hold(self, monkeypatch):
        _patch_probs(monkeypatch, [0.5, 0.5])
        result = infer_safe_action(_FakeModelReturning(), VALID_OBS)
        assert result.effective_action_index == HOLD_ACTION_INDEX
        assert result.override_reason == OVERRIDE_POLICY_INFERENCE_FAILURE

    def test_negative_probabilities_yield_hold(self, monkeypatch):
        _patch_probs(monkeypatch, [-0.1, 0.6, 0.5])
        result = infer_safe_action(_FakeModelReturning(), VALID_OBS)
        assert result.effective_action_index == HOLD_ACTION_INDEX
        assert result.override_reason == OVERRIDE_POLICY_INFERENCE_FAILURE


class TestNeverSilentlyScaleDownOnFailure:
    """The precise regression this module exists to prevent: NONE of the
    failure modes above may resolve to SCALE_DOWN (index 0) merely because
    it is the lowest index of some fallback/argmax."""

    @pytest.mark.parametrize("bad_obs", [
        [math.nan] * 11,
        [0.1, 0.2],
        None,
    ])
    def test_various_failures_never_yield_scale_down(self, bad_obs, monkeypatch):
        if bad_obs is None:
            result = infer_safe_action(None, VALID_OBS)
        else:
            result = infer_safe_action(_FakeModelReturning(), bad_obs)
        assert result.effective_action_index != 0
        assert result.effective_action_index == HOLD_ACTION_INDEX


class TestClearWinnerAndTiePreservedOnSuccess:
    def test_clear_winner_preserved(self, monkeypatch):
        _patch_probs(monkeypatch, [0.05, 0.05, 0.9])
        result = infer_safe_action(_FakeModelReturning(), VALID_OBS)
        assert result.override_reason == OVERRIDE_NONE
        assert result.effective_action_name == "SCALE_UP_ONE"
        assert result.is_tie is False
        assert result.probabilities == (0.05, 0.05, 0.9)

    def test_exact_tie_resolves_to_hold(self, monkeypatch):
        _patch_probs(monkeypatch, [0.5, 0.0, 0.5])
        result = infer_safe_action(_FakeModelReturning(), VALID_OBS)
        assert result.override_reason == OVERRIDE_NONE  # a genuine HOLD decision, not a failure
        assert result.effective_action_index == HOLD_ACTION_INDEX
        assert result.is_tie is True

    def test_configurable_epsilon_changes_near_tie_outcome(self, monkeypatch):
        _patch_probs(monkeypatch, [0.0, 0.48, 0.52])
        wide = infer_safe_action(_FakeModelReturning(), VALID_OBS, epsilon=0.05)
        narrow = infer_safe_action(_FakeModelReturning(), VALID_OBS, epsilon=0.01)
        assert wide.is_tie is True
        assert narrow.is_tie is False
        assert narrow.effective_action_name == "SCALE_UP_ONE"

    def test_default_epsilon_matches_action_selection_default(self):
        result = infer_safe_action(_FakeModelReturning(), VALID_OBS)
        assert result.epsilon == DEFAULT_NEAR_TIE_EPSILON
