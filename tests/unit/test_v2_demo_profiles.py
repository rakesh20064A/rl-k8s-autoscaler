"""tests/unit/test_v2_demo_profiles.py

V2 Phase 5E: offline checks on the 3 new DEMO-ONLY traffic profiles
(configs/traffic_profiles/v2_demo/). A demo run is NOT scientific
evidence -- these profiles must be firewalled from BOTH the training
schedule (V2_TRAINING_PROFILE_ALLOWLIST) and the evaluation campaign
(scripts.evaluate_v2_campaign.EVAL_PROFILE_ALLOWLIST), by construction
(different directory, different allowlist, no import relationship).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autoscaler.src.rl_env.training_contract_v2 import V2_TRAINING_PROFILE_ALLOWLIST
from autoscaler.src.rl_env.training_schedule_v2 import build_training_profile_rotation
from loadgen.traffic_generator import parse_traffic_profile
from scripts.evaluate_v2_campaign import EVAL_PROFILE_ALLOWLIST

DEMO_DIR = Path("configs/traffic_profiles/v2_demo")

DEMO_PROFILES = {
    "demo_cpu_short": {"seed": 7001, "resource_mode": "cpu", "peak_rps": 18.0},
    "demo_memory_short": {"seed": 7002, "resource_mode": "memory", "peak_rps": 18.0},
    "demo_mixed_short": {"seed": 7003, "resource_mode": "mixed", "peak_rps": 18.0},
}

TRAINING_SEED = 42
MAX_PEAK_RPS = 21.0
TARGET_DURATION_S = 480.0


def _load(name: str):
    return parse_traffic_profile(DEMO_DIR / f"{name}.yaml")


def _peak_rps(profile) -> float:
    total = sum(ph.duration_seconds for ph in profile.phases)
    t = 0.0
    peak = 0.0
    while t < total:
        rps, _, _ = profile.get_phase_and_target_rps(t)
        peak = max(peak, rps)
        t += 1.0
    return peak


class TestDemoDirectoryExists:
    def test_directory_exists(self):
        assert DEMO_DIR.is_dir()

    @pytest.mark.parametrize("name", list(DEMO_PROFILES))
    def test_file_exists(self, name):
        assert (DEMO_DIR / f"{name}.yaml").is_file()


class TestEachProfileParsesAndIsShort:
    @pytest.mark.parametrize("name", list(DEMO_PROFILES))
    def test_parses_via_existing_resolver(self, name):
        profile = _load(name)
        assert profile.profile_name == name
        assert len(profile.phases) >= 1

    @pytest.mark.parametrize("name", list(DEMO_PROFILES))
    def test_duration_is_approximately_8_minutes(self, name):
        profile = _load(name)
        total = sum(ph.duration_seconds for ph in profile.phases)
        assert total == TARGET_DURATION_S
        assert total % 30 == 0  # exact multiple of the 30s decision interval

    @pytest.mark.parametrize("name,expected", [(n, c["resource_mode"]) for n, c in DEMO_PROFILES.items()])
    def test_resource_mode_matches_spec(self, name, expected):
        assert _load(name).resource_mode == expected

    @pytest.mark.parametrize("name,expected_seed", [(n, c["seed"]) for n, c in DEMO_PROFILES.items()])
    def test_seed_not_used_in_training(self, name, expected_seed):
        profile = _load(name)
        assert profile.seed == expected_seed
        assert profile.seed != TRAINING_SEED

    @pytest.mark.parametrize("name,expected_peak", [(n, c["peak_rps"]) for n, c in DEMO_PROFILES.items()])
    def test_peak_rps_within_safety_envelope(self, name, expected_peak):
        profile = _load(name)
        peak = _peak_rps(profile)
        assert peak == expected_peak
        assert peak <= MAX_PEAK_RPS

    def test_memory_profiles_set_safe_in_flight_cap(self):
        for name in ("demo_memory_short", "demo_mixed_short"):
            profile = _load(name)
            assert profile.memory_mb == 16
            assert profile.memory_safe_max_in_flight == 8

    def test_description_discloses_not_scientific_evidence(self):
        for name in DEMO_PROFILES:
            path = DEMO_DIR / f"{name}.yaml"
            content = path.read_text(encoding="utf-8")
            assert "not evaluation evidence" in content.lower() or "demo only" in content.lower()


class TestTrainingFirewall:
    @pytest.mark.parametrize("name", list(DEMO_PROFILES))
    def test_demo_profile_not_in_training_allowlist(self, name):
        assert name not in V2_TRAINING_PROFILE_ALLOWLIST

    def test_long_training_schedule_never_contains_a_demo_profile(self):
        schedule = build_training_profile_rotation(seed=42, n_episodes=10_000)
        for name in DEMO_PROFILES:
            assert name not in schedule


class TestEvaluationFirewall:
    @pytest.mark.parametrize("name", list(DEMO_PROFILES))
    def test_demo_profile_not_in_eval_allowlist(self, name):
        assert name not in EVAL_PROFILE_ALLOWLIST

    def test_eval_allowlist_profiles_not_in_demo_dir(self):
        for name in EVAL_PROFILE_ALLOWLIST:
            assert not (DEMO_DIR / f"{name}.yaml").exists()
