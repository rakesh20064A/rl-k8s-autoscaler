"""tests/unit/test_training_schedule_v2.py

V2 Phase 2.2 PART 8/11: deterministic, balanced training profile rotation.
"""

from __future__ import annotations

import pytest

from autoscaler.src.rl_env.training_contract_v2 import V2_TRAINING_PROFILE_ALLOWLIST
from autoscaler.src.rl_env.training_schedule_v2 import (
    build_training_profile_rotation,
    profile_occurrence_counts,
)


class TestDeterminism:
    def test_same_seed_same_schedule(self):
        a = build_training_profile_rotation(42, 20)
        b = build_training_profile_rotation(42, 20)
        assert a == b

    def test_different_seed_different_schedule(self):
        a = build_training_profile_rotation(1, 20)
        b = build_training_profile_rotation(2, 20)
        assert a != b

    def test_negative_n_episodes_rejected(self):
        with pytest.raises(ValueError):
            build_training_profile_rotation(42, -1)

    def test_zero_episodes_is_empty_schedule(self):
        assert build_training_profile_rotation(42, 0) == []


class TestNoProfileDominates:
    def test_every_profile_present_in_a_full_cycle(self):
        schedule = build_training_profile_rotation(42, 5)
        assert set(schedule) == set(V2_TRAINING_PROFILE_ALLOWLIST)

    def test_exactly_one_occurrence_per_profile_per_cycle(self):
        schedule = build_training_profile_rotation(42, 5)
        counts = profile_occurrence_counts(schedule)
        assert all(c == 1 for c in counts.values())

    def test_balanced_across_multiple_full_cycles(self):
        n = 5 * 10  # 10 full cycles
        schedule = build_training_profile_rotation(42, n)
        counts = profile_occurrence_counts(schedule)
        assert all(c == 10 for c in counts.values())  # exactly balanced, not merely "roughly"

    def test_partial_final_cycle_never_exceeds_imbalance_of_one(self):
        """n_episodes not a multiple of 5 -- the partial trailing cycle may
        give some profiles one extra occurrence, but never two."""
        schedule = build_training_profile_rotation(42, 13)  # 2 full cycles + 3
        counts = profile_occurrence_counts(schedule)
        assert max(counts.values()) - min(counts.values()) <= 1

    def test_within_cycle_order_varies_across_cycles(self):
        """The same fixed sequence repeating every cycle would make the
        workload trivially predictable -- confirm cycle 2 is not simply
        cycle 1 repeated."""
        schedule = build_training_profile_rotation(42, 10)
        cycle1, cycle2 = schedule[:5], schedule[5:10]
        assert cycle1 != cycle2


class TestAllRequiredWorkloadTypesPresent:
    def test_cpu_only_memory_only_mixed_ramp_burst_all_present(self):
        schedule = build_training_profile_rotation(42, 5)
        assert "constant_cpu" in schedule  # CPU-only
        assert "constant_memory" in schedule  # memory-only
        assert "constant_mixed" in schedule  # mixed
        assert "ramp_mixed" in schedule  # smooth changing load
        assert "burst_mixed" in schedule  # bursty changing load

    def test_schedule_only_contains_allowlisted_profiles(self):
        schedule = build_training_profile_rotation(42, 25)
        assert set(schedule).issubset(set(V2_TRAINING_PROFILE_ALLOWLIST))
