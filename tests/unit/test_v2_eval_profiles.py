"""tests/unit/test_v2_eval_profiles.py

V2 Phase 5A / CORR-60: offline checks on the 3 new held-out evaluation
traffic profiles (configs/traffic_profiles/v2_eval/). Verifies each parses
via the existing loadgen profile resolver, has a non-degenerate shape, an
exact 1200s duration, an unused-in-training seed, and -- the training
firewall -- that none of them is a member of
training_contract_v2.V2_TRAINING_PROFILE_ALLOWLIST, so
training_schedule_v2.build_training_profile_rotation() can never select
them, by construction.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autoscaler.src.rl_env.training_contract_v2 import V2_TRAINING_PROFILE_ALLOWLIST
from autoscaler.src.rl_env.training_schedule_v2 import build_training_profile_rotation
from loadgen.traffic_generator import parse_traffic_profile

EVAL_DIR = Path("configs/traffic_profiles/v2_eval")

EVAL_PROFILES = {
    "eval_cpu_step": {"seed": 9101, "resource_mode": "cpu", "peak_rps": 18.0},
    "eval_memory_ramp": {"seed": 9102, "resource_mode": "memory", "peak_rps": 18.0},
    "eval_mixed_burst": {"seed": 9103, "resource_mode": "mixed", "peak_rps": 20.0},
}

TRAINING_SEED = 42
MAX_PEAK_RPS = 21.0
MAX_INTENSITY = 3


def _load(name: str):
    return parse_traffic_profile(EVAL_DIR / f"{name}.yaml")


def _peak_rps(profile) -> float:
    total = sum(ph.duration_seconds for ph in profile.phases)
    t = 0.0
    peak = 0.0
    while t < total:
        rps, _, _ = profile.get_phase_and_target_rps(t)
        peak = max(peak, rps)
        t += 1.0
    return peak


class TestEvalDirectoryExists:
    def test_directory_exists(self):
        assert EVAL_DIR.is_dir()

    @pytest.mark.parametrize("name", list(EVAL_PROFILES))
    def test_file_exists(self, name):
        assert (EVAL_DIR / f"{name}.yaml").is_file()


class TestEachProfileParsesAndIsNonDegenerate:
    @pytest.mark.parametrize("name", list(EVAL_PROFILES))
    def test_parses_via_existing_resolver(self, name):
        profile = _load(name)
        assert profile.profile_name == name
        assert len(profile.phases) >= 1

    @pytest.mark.parametrize("name", list(EVAL_PROFILES))
    def test_duration_is_exactly_1200_seconds(self, name):
        profile = _load(name)
        total = sum(ph.duration_seconds for ph in profile.phases)
        assert total == 1200.0, f"{name}: expected exactly 1200s, got {total}"
        # 1200s / 30s control interval = exactly 40 decisions.
        assert total % 30 == 0

    @pytest.mark.parametrize("name,expected", [(n, c["resource_mode"]) for n, c in EVAL_PROFILES.items()])
    def test_resource_mode_matches_spec(self, name, expected):
        assert _load(name).resource_mode == expected

    @pytest.mark.parametrize("name,expected_seed", [(n, c["seed"]) for n, c in EVAL_PROFILES.items()])
    def test_seed_not_used_in_training(self, name, expected_seed):
        profile = _load(name)
        assert profile.seed == expected_seed
        assert profile.seed != TRAINING_SEED

    @pytest.mark.parametrize("name", list(EVAL_PROFILES))
    def test_intensity_within_training_envelope(self, name):
        assert _load(name).intensity <= MAX_INTENSITY

    @pytest.mark.parametrize("name,expected_peak", [(n, c["peak_rps"]) for n, c in EVAL_PROFILES.items()])
    def test_peak_rps_within_training_safety_envelope(self, name, expected_peak):
        profile = _load(name)
        peak = _peak_rps(profile)
        assert peak == expected_peak
        assert peak <= MAX_PEAK_RPS

    def test_memory_profiles_set_safe_in_flight_cap(self):
        for name in ("eval_memory_ramp", "eval_mixed_burst"):
            profile = _load(name)
            assert profile.memory_mb == 16
            assert profile.memory_safe_max_in_flight == 8


class TestTrainingFirewall:
    """Proves (not merely asserts by convention) that the live training
    schedule can never select any v2_eval profile: the schedule is built
    ONLY from V2_TRAINING_PROFILE_ALLOWLIST (training_schedule_v2.
    build_training_profile_rotation), which does not and structurally
    cannot contain these names."""

    @pytest.mark.parametrize("name", list(EVAL_PROFILES))
    def test_eval_profile_not_in_training_allowlist(self, name):
        assert name not in V2_TRAINING_PROFILE_ALLOWLIST

    def test_long_training_schedule_never_contains_an_eval_profile(self):
        # A schedule far longer than the amended 48-session campaign --
        # if the allowlist boundary ever leaked, this would catch it.
        schedule = build_training_profile_rotation(seed=42, n_episodes=10_000)
        assert set(schedule) == set(V2_TRAINING_PROFILE_ALLOWLIST)
        for name in EVAL_PROFILES:
            assert name not in schedule
