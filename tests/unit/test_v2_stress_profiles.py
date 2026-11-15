"""tests/unit/test_v2_stress_profiles.py

V2 Phase 6B / CORR-65: offline checks on the 3 new EXPLORATORY, POST-HOC
stress traffic profiles (configs/traffic_profiles/v2_stress/). Mirrors
tests/unit/test_v2_eval_profiles.py's pattern: each parses via the
existing loadgen profile resolver, has a non-degenerate shape, an exact
1200s duration, an unused-elsewhere seed, stays within the training
envelope (peak RPS/intensity/memory_mb/memory_safe_max_in_flight never
exceed what at least one live V2 training profile already covers), and
-- the training firewall -- is not a member of
training_contract_v2.V2_TRAINING_PROFILE_ALLOWLIST, so
training_schedule_v2.build_training_profile_rotation() can never select
it, by construction. Also checks no name collision with v2_eval/v2_demo.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autoscaler.src.rl_env.training_contract_v2 import V2_TRAINING_PROFILE_ALLOWLIST
from autoscaler.src.rl_env.training_schedule_v2 import build_training_profile_rotation
from loadgen.traffic_generator import parse_traffic_profile

STRESS_DIR = Path("configs/traffic_profiles/v2_stress")
EVAL_DIR = Path("configs/traffic_profiles/v2_eval")
DEMO_DIR = Path("configs/traffic_profiles/v2_demo")

STRESS_PROFILES = {
    "stress_cpu_burst": {"seed": 9201, "resource_mode": "cpu", "peak_rps": 70.0, "intensity": 4},
    "stress_memory_ramp": {"seed": 9202, "resource_mode": "memory", "peak_rps": 45.0, "intensity": 3},
    "stress_mixed_burst": {"seed": 9203, "resource_mode": "mixed", "peak_rps": 70.0, "intensity": 4},
}

TRAINING_SEED = 42
EVAL_SEEDS = (9101, 9102, 9103)
# Training envelope ceiling: configs/traffic_profiles/v2/burst_mixed.yaml
# (peak_rps=70, intensity=4, memory_mb=16, memory_safe_max_in_flight=8)
# and configs/traffic_profiles/v2/ramp_mixed.yaml (peak_rps=60).
MAX_PEAK_RPS = 70.0
MAX_INTENSITY = 4


def _load(name: str):
    return parse_traffic_profile(STRESS_DIR / f"{name}.yaml")


def _peak_rps(profile) -> float:
    total = sum(ph.duration_seconds for ph in profile.phases)
    t = 0.0
    peak = 0.0
    while t < total:
        rps, _, _ = profile.get_phase_and_target_rps(t)
        peak = max(peak, rps)
        t += 1.0
    return peak


class TestStressDirectoryExists:
    def test_directory_exists(self):
        assert STRESS_DIR.is_dir()

    @pytest.mark.parametrize("name", list(STRESS_PROFILES))
    def test_file_exists(self, name):
        assert (STRESS_DIR / f"{name}.yaml").is_file()


class TestEachProfileParsesAndIsNonDegenerate:
    @pytest.mark.parametrize("name", list(STRESS_PROFILES))
    def test_parses_via_existing_resolver(self, name):
        profile = _load(name)
        assert profile.profile_name == name
        assert len(profile.phases) >= 1

    @pytest.mark.parametrize("name", list(STRESS_PROFILES))
    def test_duration_is_exactly_1200_seconds(self, name):
        profile = _load(name)
        total = sum(ph.duration_seconds for ph in profile.phases)
        assert total == 1200.0, f"{name}: expected exactly 1200s, got {total}"
        assert total % 30 == 0

    @pytest.mark.parametrize("name,expected", [(n, c["resource_mode"]) for n, c in STRESS_PROFILES.items()])
    def test_resource_mode_matches_spec(self, name, expected):
        assert _load(name).resource_mode == expected

    @pytest.mark.parametrize("name,expected_seed", [(n, c["seed"]) for n, c in STRESS_PROFILES.items()])
    def test_seed_unused_elsewhere(self, name, expected_seed):
        profile = _load(name)
        assert profile.seed == expected_seed
        assert profile.seed != TRAINING_SEED
        assert profile.seed not in EVAL_SEEDS

    @pytest.mark.parametrize("name,expected_intensity", [(n, c["intensity"]) for n, c in STRESS_PROFILES.items()])
    def test_intensity_within_training_envelope(self, name, expected_intensity):
        profile = _load(name)
        assert profile.intensity == expected_intensity
        assert profile.intensity <= MAX_INTENSITY

    @pytest.mark.parametrize("name,expected_peak", [(n, c["peak_rps"]) for n, c in STRESS_PROFILES.items()])
    def test_peak_rps_within_training_envelope(self, name, expected_peak):
        profile = _load(name)
        peak = _peak_rps(profile)
        assert peak == expected_peak
        assert peak <= MAX_PEAK_RPS

    def test_memory_profiles_set_safe_in_flight_cap(self):
        for name in ("stress_memory_ramp", "stress_mixed_burst"):
            profile = _load(name)
            assert profile.memory_mb == 16
            assert profile.memory_safe_max_in_flight == 8

    def test_cpu_profile_has_no_memory_mode(self):
        profile = _load("stress_cpu_burst")
        assert profile.memory_mb == 0


class TestTrainingFirewall:
    """Proves (not merely asserts by convention) that the live training
    schedule can never select any v2_stress profile."""

    @pytest.mark.parametrize("name", list(STRESS_PROFILES))
    def test_stress_profile_not_in_training_allowlist(self, name):
        assert name not in V2_TRAINING_PROFILE_ALLOWLIST

    def test_long_training_schedule_never_contains_a_stress_profile(self):
        schedule = build_training_profile_rotation(seed=42, n_episodes=10_000)
        assert set(schedule) == set(V2_TRAINING_PROFILE_ALLOWLIST)
        for name in STRESS_PROFILES:
            assert name not in schedule


class TestNoNameCollisionWithEvalOrDemo:
    @pytest.mark.parametrize("name", list(STRESS_PROFILES))
    def test_name_not_reused_from_eval_dir(self, name):
        assert not (EVAL_DIR / f"{name}.yaml").exists()

    @pytest.mark.parametrize("name", list(STRESS_PROFILES))
    def test_name_not_reused_from_demo_dir(self, name):
        assert not (DEMO_DIR / f"{name}.yaml").exists()
