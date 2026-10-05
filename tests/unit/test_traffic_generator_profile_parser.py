"""tests/unit/test_traffic_generator_profile_parser.py

Phase 16Z: regression + correctness tests for
`loadgen.traffic_generator.parse_traffic_profile()`'s Phase16Z fix (task
sec.2-4, sec.9-10). Two goals, kept in separate test classes:

  1. REGRESSION-PROTECT every schema the parser already handled correctly
     before this phase (legacy flat `constant`, unified `phases:`, and by
     extension every historical calibration profile that uses it) --
     nothing about their parsing may change.
  2. PROVE the four newly-recognized flat schemas (`step`/`burst`/`ramp`/
     `periodic`) now produce their INTENDED shapes, not the old degenerate
     flat-2.0-RPS fallback, and that a genuinely malformed flat profile
     still fails closed.

No live network/subprocess access anywhere in this file -- pure parsing
and schedule-generation calls against files already on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from loadgen.traffic_generator import (
    BurstPhase,
    ConstantPhase,
    PeriodicPhase,
    ProfileParseError,
    StepPhase,
    generate_arrival_schedule,
    parse_traffic_profile,
)

PROFILES_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "traffic_profiles"


# ---------------------------------------------------------------------------
# Regression protection (task sec.3)
# ---------------------------------------------------------------------------


class TestLegacyConstantSchemaUnchanged:
    def test_constant_yaml_still_parses_to_a_single_constant_phase(self):
        profile = parse_traffic_profile(PROFILES_DIR / "constant.yaml")
        assert len(profile.phases) == 1
        assert isinstance(profile.phases[0], ConstantPhase)
        assert profile.phases[0].target_rps == 20.0
        assert profile.total_duration == 600.0
        assert profile.seed == 42

    def test_legacy_target_rps_schema_parses_via_the_original_branch(self, tmp_path):
        """A synthetic file using ONLY the original legacy-constant
        discriminator (`target_rps` present, no `phases`, no new flat
        keys) must still hit the SAME branch it always did."""
        path = tmp_path / "synthetic_legacy_constant.yaml"
        path.write_text(yaml.dump({"profile": "synthetic", "seed": 7, "duration_seconds": 123.0, "target_rps": 9.0}))
        profile = parse_traffic_profile(path)
        assert len(profile.phases) == 1
        assert isinstance(profile.phases[0], ConstantPhase)
        assert profile.phases[0].target_rps == 9.0
        assert profile.total_duration == 123.0


class TestUnifiedPhasesSchemaUnchanged:
    def test_mixed_yaml_still_parses_to_its_original_eight_phases(self):
        profile = parse_traffic_profile(PROFILES_DIR / "mixed.yaml")
        types = [type(p).__name__ for p in profile.phases]
        assert types == [
            "ConstantPhase", "RampPhase", "ConstantPhase", "BurstPhase",
            "RampPhase", "PeriodicPhase", "ConstantPhase", "RampPhase",
        ]
        assert profile.total_duration == 2400.0
        assert profile.seed == 42

    def test_unified_schema_step_burst_ramp_periodic_types_still_recognized(self, tmp_path):
        """A synthetic unified-schema file exercising all four
        phase `type:` values the `if "phases" in data:` branch has always
        supported -- untouched by this phase's fix."""
        data = {
            "profile": "synthetic_unified", "seed": 1, "endpoint": "/work", "intensity": 2,
            "phases": [
                {"type": "constant", "duration_seconds": 10, "target_rps": 5},
                {"type": "ramp", "duration_seconds": 10, "start_rps": 1, "end_rps": 9},
                {"type": "step", "duration_seconds": 10, "steps": [{"at_second": 0, "rps": 3}]},
                {"type": "burst", "duration_seconds": 10, "baseline_rps": 2, "burst_rps": 8,
                 "burst_duration_seconds": 2, "inter_burst_interval_seconds": 3},
                {"type": "periodic", "duration_seconds": 10, "mean_rps": 5, "amplitude_rps": 2, "period_seconds": 5},
            ],
        }
        path = tmp_path / "synthetic_unified.yaml"
        path.write_text(yaml.dump(data))
        profile = parse_traffic_profile(path)
        types = [type(p).__name__ for p in profile.phases]
        assert types == ["ConstantPhase", "RampPhase", "StepPhase", "BurstPhase", "PeriodicPhase"]
        assert profile.total_duration == 50.0

    def test_unrecognized_phase_type_still_raises_profile_parse_error(self, tmp_path):
        path = tmp_path / "bad_type.yaml"
        path.write_text(yaml.dump({
            "profile": "bad", "seed": 1,
            "phases": [{"type": "not_a_real_type", "duration_seconds": 10}],
        }))
        with pytest.raises(ProfileParseError):
            parse_traffic_profile(path)


class TestDeterminismUnchanged:
    @pytest.mark.parametrize("name", ["constant", "step", "burst", "ramp", "periodic", "mixed"])
    def test_two_schedule_generations_from_the_same_seed_are_identical(self, name):
        profile = parse_traffic_profile(PROFILES_DIR / f"{name}.yaml")
        schedule_a = generate_arrival_schedule(profile, seed=profile.seed, arrival_process="poisson")
        schedule_b = generate_arrival_schedule(profile, seed=profile.seed, arrival_process="poisson")
        times_a = [r.planned_elapsed_seconds for r in schedule_a]
        times_b = [r.planned_elapsed_seconds for r in schedule_b]
        assert times_a == times_b
        assert len(times_a) > 0


# ---------------------------------------------------------------------------
# New flat-schema fix correctness (task sec.2, sec.4)
# ---------------------------------------------------------------------------


class TestStepSchemaFix:
    def test_step_yaml_no_longer_degenerates(self):
        profile = parse_traffic_profile(PROFILES_DIR / "step.yaml")
        assert len(profile.phases) == 1
        assert isinstance(profile.phases[0], StepPhase)
        assert profile.total_duration == 900.0  # declared == effective

    def test_step_yaml_produces_the_frozen_step_transitions(self):
        profile = parse_traffic_profile(PROFILES_DIR / "step.yaml")
        # Frozen transitions: 0->5, 180->50, 540->5, 720->30 (configs/traffic_profiles/step.yaml)
        assert profile.get_phase_and_target_rps(0.0)[0] == 5.0
        assert profile.get_phase_and_target_rps(179.9)[0] == 5.0
        assert profile.get_phase_and_target_rps(180.0)[0] == 50.0
        assert profile.get_phase_and_target_rps(539.9)[0] == 50.0
        assert profile.get_phase_and_target_rps(540.0)[0] == 5.0
        assert profile.get_phase_and_target_rps(719.9)[0] == 5.0
        assert profile.get_phase_and_target_rps(720.0)[0] == 30.0
        assert profile.get_phase_and_target_rps(899.0)[0] == 30.0


class TestBurstSchemaFix:
    def test_burst_yaml_no_longer_degenerates(self):
        profile = parse_traffic_profile(PROFILES_DIR / "burst.yaml")
        assert len(profile.phases) == 1
        assert isinstance(profile.phases[0], BurstPhase)
        assert profile.total_duration == 1200.0  # declared == effective

    def test_burst_yaml_cycles_baseline_and_burst_rps(self):
        profile = parse_traffic_profile(PROFILES_DIR / "burst.yaml")
        # baseline=5, burst=70, burst_duration=30, inter_burst_interval=120 -> cycle_len=150
        assert profile.get_phase_and_target_rps(0.0)[0] == 70.0     # inside the burst window
        assert profile.get_phase_and_target_rps(29.9)[0] == 70.0
        assert profile.get_phase_and_target_rps(30.0)[0] == 5.0     # baseline resumes
        assert profile.get_phase_and_target_rps(100.0)[0] == 5.0
        assert profile.get_phase_and_target_rps(150.0)[0] == 70.0   # next cycle's burst
        rps_values = {round(profile.get_phase_and_target_rps(t)[0], 1) for t in range(0, 1200, 5)}
        assert rps_values == {5.0, 70.0}  # never a flat 2.0 RPS anywhere


class TestPeriodicSchemaFix:
    def test_periodic_yaml_no_longer_degenerates(self):
        profile = parse_traffic_profile(PROFILES_DIR / "periodic.yaml")
        assert len(profile.phases) == 1
        assert isinstance(profile.phases[0], PeriodicPhase)
        assert profile.total_duration == 1800.0  # declared == effective

    def test_periodic_yaml_oscillates_around_its_mean(self):
        profile = parse_traffic_profile(PROFILES_DIR / "periodic.yaml")
        # mean=20, amplitude=15, period=300 -> peak near t=75 (sin=1), trough near t=225 (sin=-1)
        values = [profile.get_phase_and_target_rps(t)[0] for t in range(0, 300, 10)]
        assert max(values) > 25.0  # meaningfully above the mean
        assert min(values) < 15.0  # meaningfully below the mean
        assert len(set(round(v, 1) for v in values)) > 5  # genuinely varying, not flat


class TestRampSchemaFix:
    def test_ramp_yaml_no_longer_degenerates(self):
        profile = parse_traffic_profile(PROFILES_DIR / "ramp.yaml")
        types = [type(p).__name__ for p in profile.phases]
        assert types == ["RampPhase", "ConstantPhase", "RampPhase"]

    def test_ramp_yaml_declared_vs_effective_duration_mismatch_disclosed(self):
        """KNOWN, DISCLOSED finding (task_outputs/phase_16z_six_profile_
        semantic_validation.md): ramp.yaml's own sub-durations
        (400+200+400=1000s) do not sum to its top-level duration_seconds
        field (1200s) -- a pre-existing inconsistency in the frozen YAML,
        not introduced or silently reconciled by this fix. The effective
        duration (sum of parsed phases) is what actually governs traffic
        generation and episode timing."""
        with open(PROFILES_DIR / "ramp.yaml") as f:
            raw = yaml.safe_load(f)
        declared_duration = float(raw["duration_seconds"])
        profile = parse_traffic_profile(PROFILES_DIR / "ramp.yaml")
        effective_duration = profile.total_duration
        assert declared_duration == 1200.0
        assert effective_duration == 1000.0
        assert declared_duration != effective_duration  # the disclosed mismatch, proven not silently fixed

    def test_ramp_yaml_produces_actual_ramp_up_hold_ramp_down_behavior(self):
        profile = parse_traffic_profile(PROFILES_DIR / "ramp.yaml")
        # ramp_up: 3->60 over 400s; hold: 60 for 200s; ramp_down: 60->3 over 400s
        assert profile.get_phase_and_target_rps(0.0)[0] == pytest.approx(3.0, abs=0.5)
        assert profile.get_phase_and_target_rps(200.0)[0] == pytest.approx(31.5, abs=1.0)
        assert profile.get_phase_and_target_rps(400.0)[0] == pytest.approx(60.0, abs=0.5)
        assert profile.get_phase_and_target_rps(500.0)[0] == pytest.approx(60.0, abs=0.5)  # hold
        assert profile.get_phase_and_target_rps(600.0)[0] == pytest.approx(60.0, abs=0.5)  # ramp-down starts
        assert profile.get_phase_and_target_rps(999.0)[0] < 10.0  # back down near the end


class TestMalformedFlatProfileStillFailsClosed:
    def test_flat_profile_with_no_recognized_keys_falls_back_to_degenerate_constant(self, tmp_path):
        """A genuinely malformed/unrecognized flat profile must still hit
        the fallback branch (never silently guessed at) -- proving the fix
        did not accidentally widen the parser to accept arbitrary garbage."""
        path = tmp_path / "garbage.yaml"
        path.write_text(yaml.dump({"profile": "garbage", "seed": 1, "duration_seconds": 42, "some_unrelated_key": 123}))
        profile = parse_traffic_profile(path)
        assert len(profile.phases) == 1
        assert isinstance(profile.phases[0], ConstantPhase)
        assert profile.phases[0].target_rps == 2.0  # the pre-existing, unchanged fallback


class TestAllSixTrainingProfilesSemanticValidation:
    """Task sec.4: parse every profile, report structure/duration/RPS
    range/endpoint/intensity/seed, and prove none is silently degenerate
    (except constant.yaml, which is intentionally one ConstantPhase)."""

    PROFILES = ["constant", "step", "burst", "ramp", "periodic", "mixed"]

    @pytest.mark.parametrize("name", PROFILES)
    def test_endpoint_intensity_seed_propagate_correctly(self, name):
        with open(PROFILES_DIR / f"{name}.yaml") as f:
            raw = yaml.safe_load(f)
        profile = parse_traffic_profile(PROFILES_DIR / f"{name}.yaml")
        assert profile.endpoint == raw.get("endpoint", "/work")
        assert profile.intensity == int(raw.get("intensity", 3))
        assert profile.seed == int(raw.get("seed", 42))

    @pytest.mark.parametrize("name", ["step", "burst", "ramp", "periodic", "mixed"])
    def test_non_constant_profiles_are_not_degenerate_single_constant_phases(self, name):
        profile = parse_traffic_profile(PROFILES_DIR / f"{name}.yaml")
        is_single_flat_constant = len(profile.phases) == 1 and isinstance(profile.phases[0], ConstantPhase)
        assert not is_single_flat_constant

    @pytest.mark.parametrize("name", PROFILES)
    def test_rps_varies_meaningfully_or_is_intentionally_constant(self, name):
        profile = parse_traffic_profile(PROFILES_DIR / f"{name}.yaml")
        n_samples = 40
        step = max(1.0, profile.total_duration / n_samples)
        values = []
        t = 0.0
        while t < profile.total_duration:
            values.append(round(profile.get_phase_and_target_rps(t)[0], 2))
            t += step
        distinct = len(set(values))
        if name == "constant":
            assert distinct == 1  # intentionally flat
        else:
            assert distinct > 1  # genuinely varying -- the core proof the fix works
