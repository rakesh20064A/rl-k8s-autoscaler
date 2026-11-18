"""tests/unit/test_v2_demo_profile_catalog.py

Offline checks on the demo UI's 36-profile `pattern x resource x load`
traffic-profile catalog (`scripts/generate_demo_profiles.py`,
`configs/traffic_profiles/v2_demo/demo_<pattern>_<resource>_<load>.yaml`).

Mirrors `tests/unit/test_v2_demo_profiles.py`'s firewall discipline for
the 3 original short demo profiles: every one of these 36 files must
also be firewalled from both the training schedule
(`V2_TRAINING_PROFILE_ALLOWLIST`) and the evaluation/stress profile sets
(`EVAL_PROFILE_ALLOWLIST` / `STRESS_PROFILE_ALLOWLIST`), and the demo
episode runner (`scripts/run_demo_episode_v2.py`) must accept every one
of them but nothing outside its own demo directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autoscaler.src.rl_env.training_contract_v2 import V2_TRAINING_PROFILE_ALLOWLIST
from autoscaler.src.rl_env.training_schedule_v2 import build_training_profile_rotation
from loadgen.traffic_generator import parse_traffic_profile
from scripts import generate_demo_profiles as gen
from scripts import run_demo_episode_v2 as runner
from scripts.evaluate_v2_campaign import EVAL_PROFILE_ALLOWLIST, STRESS_PROFILE_ALLOWLIST

DEMO_DIR = Path("configs/traffic_profiles/v2_demo")
TRAINING_SEED = 42


def _load(name: str):
    return parse_traffic_profile(DEMO_DIR / f"{name}.yaml")


def _peak_rps(profile) -> float:
    total = profile.total_duration
    t = 0.0
    peak = 0.0
    while t < total:
        rps, _, _ = profile.get_phase_and_target_rps(t)
        peak = max(peak, rps)
        t += 1.0
    return peak


class TestCatalogShape:
    def test_36_combinations(self):
        assert len(gen.ALL_COMBOS) == 36
        assert len(gen.ALL_PROFILE_NAMES) == 36
        assert len(set(gen.ALL_PROFILE_NAMES)) == 36  # no duplicate names

    def test_6_patterns_3_resources_2_loads(self):
        assert len(gen.PATTERNS) == 6
        assert len(gen.RESOURCES) == 3
        assert len(gen.LOADS) == 2

    def test_seeds_span_7101_to_7136_with_no_collisions(self):
        seeds = sorted(gen.seed_for(*combo) for combo in gen.ALL_COMBOS)
        assert seeds[0] == 7101
        assert seeds[-1] == 7136
        assert len(seeds) == len(set(seeds)) == 36

    def test_seed_assignment_order_is_pattern_outer_resource_middle_load_inner(self):
        # The first 6 seeds (one per pattern) should all use the first
        # resource/load pair (cpu, normal).
        for pattern in gen.PATTERNS:
            assert gen.seed_for(pattern, "cpu", "normal") == 7101 + gen.PATTERNS.index(pattern) * 6

    @pytest.mark.parametrize("name", gen.ALL_PROFILE_NAMES)
    def test_filename_round_trips_through_parse_profile_name(self, name):
        combo = gen.parse_profile_name(name)
        assert combo is not None
        assert gen.profile_slug(*combo) == name

    def test_parse_profile_name_rejects_legacy_short_profiles(self):
        for name in gen.LEGACY_DEMO_PROFILE_NAMES:
            assert gen.parse_profile_name(name) is None


class TestEachProfileFileExistsAndParses:
    @pytest.mark.parametrize("name", gen.ALL_PROFILE_NAMES)
    def test_file_exists(self, name):
        assert (DEMO_DIR / f"{name}.yaml").is_file()

    @pytest.mark.parametrize("name", gen.ALL_PROFILE_NAMES)
    def test_parses_via_existing_resolver(self, name):
        profile = _load(name)
        assert profile.profile_name == name
        assert len(profile.phases) >= 1

    @pytest.mark.parametrize("combo", gen.ALL_COMBOS, ids=lambda c: "_".join(c))
    def test_total_duration_is_exactly_480s(self, combo):
        profile = _load(gen.profile_slug(*combo))
        assert profile.total_duration == gen.TOTAL_DURATION_S
        assert profile.total_duration % gen.DECISION_INTERVAL_S == 0
        assert profile.total_duration / gen.DECISION_INTERVAL_S == gen.N_DECISIONS == 16

    @pytest.mark.parametrize("combo", gen.ALL_COMBOS, ids=lambda c: "_".join(c))
    def test_not_degenerate(self, combo):
        profile = _load(gen.profile_slug(*combo))
        assert _peak_rps(profile) > 0.0

    @pytest.mark.parametrize("combo", gen.ALL_COMBOS, ids=lambda c: "_".join(c))
    def test_peak_rps_within_load_level_limit(self, combo):
        pattern, resource, load = combo
        profile = _load(gen.profile_slug(*combo))
        peak = _peak_rps(profile)
        limit = gen._peak_limit_for(resource, load)
        assert 0.0 < peak <= limit

    @pytest.mark.parametrize("combo", gen.ALL_COMBOS, ids=lambda c: "_".join(c))
    def test_resource_mode_matches_catalog(self, combo):
        pattern, resource, load = combo
        profile = _load(gen.profile_slug(*combo))
        assert profile.resource_mode == resource

    @pytest.mark.parametrize("combo", gen.ALL_COMBOS, ids=lambda c: "_".join(c))
    def test_memory_and_mixed_profiles_set_safe_in_flight_cap(self, combo):
        pattern, resource, load = combo
        profile = _load(gen.profile_slug(*combo))
        if resource in ("memory", "mixed"):
            assert profile.memory_mb == 16
            assert profile.memory_safe_max_in_flight == 8
        else:
            assert profile.memory_mb == 0

    @pytest.mark.parametrize("combo", gen.ALL_COMBOS, ids=lambda c: "_".join(c))
    def test_seed_matches_catalog_and_not_the_training_seed(self, combo):
        profile = _load(gen.profile_slug(*combo))
        assert profile.seed == gen.seed_for(*combo)
        assert profile.seed != TRAINING_SEED

    def test_generator_check_mode_reports_no_problems(self):
        assert gen.check_all(DEMO_DIR) == []


class TestTrainingFirewall:
    @pytest.mark.parametrize("name", gen.ALL_PROFILE_NAMES)
    def test_demo_profile_not_in_training_allowlist(self, name):
        assert name not in V2_TRAINING_PROFILE_ALLOWLIST

    def test_long_training_schedule_never_contains_a_catalog_demo_profile(self):
        schedule = build_training_profile_rotation(seed=42, n_episodes=10_000)
        for name in gen.ALL_PROFILE_NAMES:
            assert name not in schedule


class TestEvaluationAndStressFirewall:
    @pytest.mark.parametrize("name", gen.ALL_PROFILE_NAMES)
    def test_demo_profile_not_in_eval_allowlist(self, name):
        assert name not in EVAL_PROFILE_ALLOWLIST

    @pytest.mark.parametrize("name", gen.ALL_PROFILE_NAMES)
    def test_demo_profile_not_in_stress_allowlist(self, name):
        assert name not in STRESS_PROFILE_ALLOWLIST

    def test_catalog_disjoint_from_training_eval_and_stress_allowlists(self):
        demo_set = set(gen.ALL_PROFILE_NAMES)
        assert demo_set.isdisjoint(set(V2_TRAINING_PROFILE_ALLOWLIST))
        assert demo_set.isdisjoint(set(EVAL_PROFILE_ALLOWLIST))
        assert demo_set.isdisjoint(set(STRESS_PROFILE_ALLOWLIST))

    def test_eval_and_stress_allowlist_profiles_not_in_demo_dir(self):
        for name in list(EVAL_PROFILE_ALLOWLIST) + list(STRESS_PROFILE_ALLOWLIST):
            assert not (DEMO_DIR / f"{name}.yaml").exists()


class TestDemoEpisodeRunnerAcceptsOnlyDemoProfiles:
    @pytest.mark.parametrize("name", gen.ALL_PROFILE_NAMES)
    def test_runner_resolves_each_catalog_profile_inside_demo_dir(self, name):
        path = runner.resolve_demo_profile_path(name)
        assert path == runner.DEMO_PROFILE_DIR / f"{name}.yaml"
        assert path.is_file()

    @pytest.mark.parametrize("name", list(EVAL_PROFILE_ALLOWLIST) + list(STRESS_PROFILE_ALLOWLIST))
    def test_runner_refuses_eval_and_stress_profile_names(self, name):
        with pytest.raises(runner.DemoProfileFirewallViolation):
            runner.resolve_demo_profile_path(name)

    @pytest.mark.parametrize("name", list(V2_TRAINING_PROFILE_ALLOWLIST))
    def test_runner_refuses_training_profile_names(self, name):
        with pytest.raises(runner.DemoProfileFirewallViolation):
            runner.resolve_demo_profile_path(name)
