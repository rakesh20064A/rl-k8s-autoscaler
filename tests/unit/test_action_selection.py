"""Tests for autoscaler/src/rl_env/action_selection.py -- the V2 safe
deterministic action selector. Proves the near-tie-safe rule precisely,
with NO coupling to scripts/evaluate_ppo_vs_hpa.py or any frozen V1
evaluation path (this module is never imported there).
"""

from __future__ import annotations

import math

import pytest

from autoscaler.src.rl_env.action_selection import (
    DEFAULT_NEAR_TIE_EPSILON,
    HOLD_ACTION_INDEX,
    compute_margin_stats,
    select_safe_deterministic_action,
)
from autoscaler.src.rl_env.types import ACTION_MAP

SCALE_DOWN, HOLD, SCALE_UP = 0, 1, 2


def test_action_map_contract_unchanged():
    """Frozen contract guard: this module must never reorder it."""
    assert ACTION_MAP == {0: -1, 1: 0, 2: 1}
    assert HOLD_ACTION_INDEX == 1


class TestClearWinners:
    def test_unique_down_winner(self):
        r = select_safe_deterministic_action([0.9, 0.05, 0.05])
        assert r.action_index == SCALE_DOWN
        assert not r.is_tie

    def test_unique_hold_winner(self):
        r = select_safe_deterministic_action([0.05, 0.9, 0.05])
        assert r.action_index == HOLD
        assert not r.is_tie

    def test_unique_up_winner(self):
        r = select_safe_deterministic_action([0.05, 0.05, 0.9])
        assert r.action_index == SCALE_UP
        assert not r.is_tie


class TestExactTies:
    def test_exact_down_up_tie_resolves_to_hold(self):
        r = select_safe_deterministic_action([0.45, 0.10, 0.45])
        assert r.action_index == HOLD
        assert r.is_tie

    def test_exact_down_hold_tie_resolves_to_hold(self):
        r = select_safe_deterministic_action([0.45, 0.45, 0.10])
        assert r.action_index == HOLD
        assert r.is_tie

    def test_exact_hold_up_tie_resolves_to_hold(self):
        r = select_safe_deterministic_action([0.10, 0.45, 0.45])
        assert r.action_index == HOLD
        assert r.is_tie

    def test_all_three_equal_resolves_to_hold(self):
        r = select_safe_deterministic_action([1 / 3, 1 / 3, 1 / 3])
        assert r.action_index == HOLD
        assert r.is_tie

    def test_scale_down_never_wins_an_exact_tie_merely_from_index_order(self):
        """The precise regression this module exists to prevent: a plain
        argmax (SB3's own deterministic mode) would pick index 0
        (SCALE_DOWN) here. This selector must not."""
        r = select_safe_deterministic_action([0.5, 0.0, 0.5])
        assert r.action_index != SCALE_DOWN
        assert r.action_index == HOLD


class TestNearTies:
    @pytest.mark.parametrize("probs", [
        [0.50, 0.02, 0.48],   # DOWN/UP near-tie (within default epsilon 0.05)
        [0.50, 0.48, 0.02],   # DOWN/HOLD near-tie
        [0.02, 0.50, 0.48],   # HOLD/UP near-tie
        [0.35, 0.33, 0.32],   # three-way near-tie
    ])
    def test_near_tie_within_default_epsilon_resolves_to_hold(self, probs):
        r = select_safe_deterministic_action(probs, epsilon=DEFAULT_NEAR_TIE_EPSILON)
        assert r.action_index == HOLD
        assert r.is_tie

    def test_difference_just_outside_epsilon_yields_real_winner(self):
        epsilon = 0.05
        # max=0.52, runner-up=0.46 -> gap 0.06 > epsilon -> NOT a near-tie
        probs = [0.02, 0.46, 0.52]
        r = select_safe_deterministic_action(probs, epsilon=epsilon)
        assert r.action_index == SCALE_UP
        assert not r.is_tie

    def test_difference_just_inside_epsilon_is_a_near_tie(self):
        epsilon = 0.05
        # max=0.52, runner-up=0.48 -> gap 0.04 < epsilon -> near-tie -> HOLD
        probs = [0.0, 0.48, 0.52]
        r = select_safe_deterministic_action(probs, epsilon=epsilon)
        assert r.action_index == HOLD
        assert r.is_tie

    def test_epsilon_is_overridable(self):
        probs = [0.0, 0.48, 0.52]
        r = select_safe_deterministic_action(probs, epsilon=0.01)  # gap 0.04 > 0.01 now
        assert r.action_index == SCALE_UP
        assert not r.is_tie


class TestEpsilonZeroExactTieOnly:
    """V2 Phase 1.1: near_tie_epsilon=0.05 is PROVISIONAL and NOT finalized
    this phase -- these tests prove the selector itself correctly supports
    epsilon=0 (exact-tie-only behavior), independent of whatever default
    value is eventually chosen."""

    def test_epsilon_zero_exact_tie_resolves_to_hold(self):
        r = select_safe_deterministic_action([0.5, 0.0, 0.5], epsilon=0.0)
        assert r.action_index == HOLD
        assert r.is_tie

    def test_epsilon_zero_all_three_equal_resolves_to_hold(self):
        r = select_safe_deterministic_action([1 / 3, 1 / 3, 1 / 3], epsilon=0.0)
        assert r.action_index == HOLD
        assert r.is_tie

    def test_epsilon_zero_non_tie_yields_argmax_winner(self):
        # Any nonzero gap must be decisive at epsilon=0, even a tiny one.
        r = select_safe_deterministic_action([0.50, 0.499999, 0.000001], epsilon=0.0)
        assert r.action_index == SCALE_DOWN
        assert not r.is_tie

    def test_epsilon_zero_does_not_treat_near_tie_as_tie(self):
        # This same input IS a near-tie under the default epsilon, but must
        # NOT be treated as a tie at epsilon=0 -- proves epsilon=0 is a
        # strictly narrower (exact-tie-only), not merely smaller, rule.
        probs = [0.0, 0.48, 0.52]
        r = select_safe_deterministic_action(probs, epsilon=0.0)
        assert r.action_index == SCALE_UP
        assert not r.is_tie


class TestMalformedInputRejected:
    def test_nan_rejected(self):
        with pytest.raises(ValueError):
            select_safe_deterministic_action([math.nan, 0.5, 0.5])

    def test_inf_rejected(self):
        with pytest.raises(ValueError):
            select_safe_deterministic_action([math.inf, 0.5, 0.5])

    def test_wrong_length_rejected(self):
        with pytest.raises(ValueError):
            select_safe_deterministic_action([0.5, 0.5])

    def test_negative_probability_rejected(self):
        with pytest.raises(ValueError):
            select_safe_deterministic_action([-0.1, 0.6, 0.5])

    def test_negative_epsilon_rejected(self):
        with pytest.raises(ValueError):
            select_safe_deterministic_action([0.3, 0.3, 0.4], epsilon=-0.01)


class TestComputeMarginStats:
    """V2 Phase 2.1 PART 8: the per-validation-state record the post-
    training epsilon calibration procedure is built from."""

    def test_clear_winner_margin(self):
        stats = compute_margin_stats([0.05, 0.05, 0.9])
        assert stats.top1_probability == pytest.approx(0.9)
        assert stats.top2_probability == pytest.approx(0.05)
        assert stats.margin == pytest.approx(0.85)
        assert stats.argmax_action_index == SCALE_UP
        assert stats.argmax_action_name == "SCALE_UP_ONE"

    def test_exact_tie_has_zero_margin(self):
        stats = compute_margin_stats([0.5, 0.0, 0.5])
        assert stats.margin == pytest.approx(0.0)

    def test_near_tie_small_margin(self):
        stats = compute_margin_stats([0.0, 0.48, 0.52])
        assert stats.margin == pytest.approx(0.04)
        assert stats.argmax_action_index == SCALE_UP

    def test_wrong_length_rejected(self):
        with pytest.raises(ValueError):
            compute_margin_stats([0.5, 0.5])

    def test_nan_rejected(self):
        with pytest.raises(ValueError):
            compute_margin_stats([math.nan, 0.5, 0.5])

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            compute_margin_stats([-0.1, 0.6, 0.5])

    def test_does_not_itself_choose_an_epsilon(self):
        """compute_margin_stats is purely observational -- it has no
        epsilon parameter at all, confirming it cannot be mistaken for the
        selector itself."""
        import inspect

        sig = inspect.signature(compute_margin_stats)
        assert "epsilon" not in sig.parameters
