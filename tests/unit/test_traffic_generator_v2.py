"""V2 additive coverage for loadgen/traffic_generator.py: resource_mode /
memory_mb parsing, validation, and forwarding into the dispatched request
URL. All existing (pre-V2) profiles omit these fields entirely and must
parse/behave exactly as before -- see test_traffic_generator.py's own
(unmodified, still-passing) suite for that baseline.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from unittest.mock import MagicMock

from loadgen.traffic_generator import (
    VALID_RESOURCE_MODES,
    PlannedRequest,
    ProfileParseError,
    TrafficGenerator,
    parse_traffic_profile,
)


def _write_profile(tmp_path: Path, name: str, extra: dict) -> Path:
    content = {
        "profile": name,
        "seed": 1,
        "endpoint": "/work",
        "intensity": 3,
        "phases": [{"name": "p1", "type": "constant", "duration_seconds": 5.0, "target_rps": 2.0}],
        **extra,
    }
    path = tmp_path / f"{name}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(content, f)
    return path


class TestProfileParsingBackwardCompatible:
    def test_profile_without_resource_fields_defaults_to_cpu_zero(self, tmp_path):
        path = _write_profile(tmp_path, "legacy", {})
        profile = parse_traffic_profile(path)
        assert profile.resource_mode == "cpu"
        assert profile.memory_mb == 0

    def test_existing_six_profiles_still_default(self):
        """Direct proof against the real, frozen V1 profile files -- none
        of them were modified by this phase."""
        for name in ("constant", "ramp", "burst", "step", "periodic", "mixed"):
            path = Path("configs/traffic_profiles") / f"{name}.yaml"
            profile = parse_traffic_profile(path)
            assert profile.resource_mode == "cpu"
            assert profile.memory_mb == 0


class TestProfileParsingV2Fields:
    def test_memory_resource_mode_parsed(self, tmp_path):
        path = _write_profile(tmp_path, "mem", {"resource_mode": "memory", "memory_mb": 32})
        profile = parse_traffic_profile(path)
        assert profile.resource_mode == "memory"
        assert profile.memory_mb == 32

    def test_mixed_resource_mode_parsed(self, tmp_path):
        path = _write_profile(tmp_path, "mix", {"resource_mode": "mixed", "memory_mb": 16})
        profile = parse_traffic_profile(path)
        assert profile.resource_mode == "mixed"

    def test_invalid_resource_mode_rejected(self, tmp_path):
        path = _write_profile(tmp_path, "bad", {"resource_mode": "disk"})
        with pytest.raises(ProfileParseError):
            parse_traffic_profile(path)

    def test_negative_memory_mb_rejected(self, tmp_path):
        path = _write_profile(tmp_path, "negmem", {"memory_mb": -5})
        with pytest.raises(ProfileParseError):
            parse_traffic_profile(path)

    def test_valid_resource_modes_constant(self):
        assert VALID_RESOURCE_MODES == ("cpu", "memory", "mixed")


class TestRequestUrlForwarding:
    @pytest.fixture()
    def generator(self, tmp_path):
        profile_path = _write_profile(tmp_path, "fwd", {"resource_mode": "mixed", "memory_mb": 8})
        gen = TrafficGenerator(
            experiment_id="exp_fwd",
            profile_path=str(profile_path.relative_to(tmp_path)),
            project_root=tmp_path,
        )
        gen.setup_directory_and_schedule()
        return gen

    def test_dispatched_url_contains_resource_mode_and_memory_mb(self, generator):
        captured = {}

        class _FakeClient:
            async def get(self, url, timeout=None):
                captured["url"] = url
                class _Resp:
                    status_code = 200
                return _Resp()

        req = PlannedRequest(0, 0.0, "p1", "constant", 2.0, 3)
        asyncio.run(
            generator._dispatch_single_request(_FakeClient(), req, 0.0, MagicMock())
        )
        assert "resource_mode=mixed" in captured["url"]
        assert "memory_mb=8" in captured["url"]
        assert "intensity=3" in captured["url"]  # per-request field still present, unchanged

    def test_legacy_profile_dispatch_url_defaults_preserved(self, tmp_path):
        """V2 Phase 1.1: a legacy profile (no resource_mode/memory_mb) must
        dispatch a URL byte-identical to V1 -- NO V2-only query parameters
        appended at all, not merely defaulted values. See the module-level
        correction note in _dispatch_single_request for why this changed
        from Phase 1's original (always-append) behavior."""
        profile_path = _write_profile(tmp_path, "legacy_fwd", {})
        gen = TrafficGenerator(
            experiment_id="exp_legacy_fwd",
            profile_path=str(profile_path.relative_to(tmp_path)),
            project_root=tmp_path,
        )
        gen.setup_directory_and_schedule()
        captured = {}

        class _FakeClient:
            async def get(self, url, timeout=None):
                captured["url"] = url
                class _Resp:
                    status_code = 200
                return _Resp()

        req = PlannedRequest(0, 0.0, "p1", "constant", 2.0, 3)
        asyncio.run(
            gen._dispatch_single_request(_FakeClient(), req, 0.0, MagicMock())
        )
        assert captured["url"] == f"{gen.target_url}/work?intensity=3"
        assert "resource_mode" not in captured["url"]
        assert "memory_mb" not in captured["url"]
        assert "memory_hold_ms" not in captured["url"]

    def test_mixed_profile_dispatch_url_includes_memory_hold_ms(self, generator):
        captured = {}

        class _FakeClient:
            async def get(self, url, timeout=None):
                captured["url"] = url
                class _Resp:
                    status_code = 200
                return _Resp()

        req = PlannedRequest(0, 0.0, "p1", "constant", 2.0, 3)
        asyncio.run(
            generator._dispatch_single_request(_FakeClient(), req, 0.0, MagicMock())
        )
        assert f"memory_hold_ms={generator.profile.memory_hold_ms}" in captured["url"]

    def test_resource_mode_only_profile_still_appends_v2_params(self, tmp_path):
        """A profile that sets resource_mode=memory but memory_mb=0 (edge
        case: no actual memory work requested) is still non-default and
        must still get the V2 params -- only a TRULY untouched legacy
        profile gets the bare V1 URL."""
        profile_path = _write_profile(tmp_path, "modeonly", {"resource_mode": "memory"})
        gen = TrafficGenerator(
            experiment_id="exp_modeonly",
            profile_path=str(profile_path.relative_to(tmp_path)),
            project_root=tmp_path,
        )
        gen.setup_directory_and_schedule()
        captured = {}

        class _FakeClient:
            async def get(self, url, timeout=None):
                captured["url"] = url
                class _Resp:
                    status_code = 200
                return _Resp()

        req = PlannedRequest(0, 0.0, "p1", "constant", 2.0, 3)
        asyncio.run(
            gen._dispatch_single_request(_FakeClient(), req, 0.0, MagicMock())
        )
        assert "resource_mode=memory" in captured["url"]


class TestMemoryHoldMsParsing:
    def test_memory_hold_ms_defaults(self, tmp_path):
        from loadgen.traffic_generator import DEFAULT_MEMORY_HOLD_MS
        path = _write_profile(tmp_path, "legacy", {})
        profile = parse_traffic_profile(path)
        assert profile.memory_hold_ms == DEFAULT_MEMORY_HOLD_MS

    def test_memory_hold_ms_parsed(self, tmp_path):
        path = _write_profile(tmp_path, "held", {"resource_mode": "memory", "memory_mb": 8, "memory_hold_ms": 1234})
        profile = parse_traffic_profile(path)
        assert profile.memory_hold_ms == 1234

    def test_negative_memory_hold_ms_rejected(self, tmp_path):
        path = _write_profile(tmp_path, "neghold", {"memory_hold_ms": -1})
        with pytest.raises(ProfileParseError):
            parse_traffic_profile(path)


class TestMemorySafeMaxInFlight:
    """V2 Phase 2.1 PART 3: a memory/mixed profile's own declared
    concurrency ceiling must only ever TIGHTEN the generator's general
    --max-in-flight bound, never loosen it, and must never affect a
    CPU-only profile at all."""

    def test_absent_by_default(self, tmp_path):
        path = _write_profile(tmp_path, "legacy", {})
        profile = parse_traffic_profile(path)
        assert profile.memory_safe_max_in_flight is None

    def test_parsed_when_present(self, tmp_path):
        path = _write_profile(
            tmp_path, "safe_mem", {"resource_mode": "memory", "memory_mb": 16, "memory_safe_max_in_flight": 10},
        )
        profile = parse_traffic_profile(path)
        assert profile.memory_safe_max_in_flight == 10

    def test_zero_rejected(self, tmp_path):
        path = _write_profile(tmp_path, "zero_safe", {"memory_safe_max_in_flight": 0})
        with pytest.raises(ProfileParseError):
            parse_traffic_profile(path)

    def test_negative_rejected(self, tmp_path):
        path = _write_profile(tmp_path, "neg_safe", {"memory_safe_max_in_flight": -3})
        with pytest.raises(ProfileParseError):
            parse_traffic_profile(path)

    def test_effective_max_in_flight_tightens_general_bound_for_memory_profile(self, tmp_path):
        profile_path = _write_profile(
            tmp_path, "tight_mem", {"resource_mode": "memory", "memory_mb": 16, "memory_safe_max_in_flight": 10},
        )
        gen = TrafficGenerator(
            experiment_id="exp1", profile_path=str(profile_path.relative_to(tmp_path)),
            project_root=tmp_path, max_in_flight=50,
        )
        assert gen.effective_max_in_flight == 10

    def test_effective_max_in_flight_never_loosens_general_bound(self, tmp_path):
        """A profile declaring a LARGER safe value than the general cap
        must not raise the effective ceiling above the general cap."""
        profile_path = _write_profile(
            tmp_path, "loose_mem", {"resource_mode": "memory", "memory_mb": 16, "memory_safe_max_in_flight": 100},
        )
        gen = TrafficGenerator(
            experiment_id="exp2", profile_path=str(profile_path.relative_to(tmp_path)),
            project_root=tmp_path, max_in_flight=50,
        )
        assert gen.effective_max_in_flight == 50

    def test_cpu_only_profile_unaffected_even_if_field_present(self, tmp_path):
        """memory_safe_max_in_flight only applies to resource_mode in
        (memory, mixed) -- a cpu-mode profile ignores it entirely."""
        profile_path = _write_profile(
            tmp_path, "cpu_with_field", {"resource_mode": "cpu", "memory_safe_max_in_flight": 5},
        )
        gen = TrafficGenerator(
            experiment_id="exp3", profile_path=str(profile_path.relative_to(tmp_path)),
            project_root=tmp_path, max_in_flight=50,
        )
        assert gen.effective_max_in_flight == 50

    def test_legacy_profile_effective_equals_general_bound(self, tmp_path):
        profile_path = _write_profile(tmp_path, "legacy_fwd2", {})
        gen = TrafficGenerator(
            experiment_id="exp4", profile_path=str(profile_path.relative_to(tmp_path)),
            project_root=tmp_path, max_in_flight=50,
        )
        assert gen.effective_max_in_flight == 50

    def test_mixed_profile_without_explicit_value_unaffected(self, tmp_path):
        profile_path = _write_profile(tmp_path, "mixed_no_cap", {"resource_mode": "mixed", "memory_mb": 16})
        gen = TrafficGenerator(
            experiment_id="exp5", profile_path=str(profile_path.relative_to(tmp_path)),
            project_root=tmp_path, max_in_flight=50,
        )
        assert gen.effective_max_in_flight == 50

    def test_real_v2_memory_profiles_declare_safe_concurrency(self):
        """Direct proof against the real, shipped V2 profile files.
        LOWERED from 10 to 8 in V2 Phase 2.2 PART 3 after live measurement
        through the actual loadgen showed cap=10 reached a peak of ~222.8Mi
        (87% of the 256Mi limit, only 13% headroom) -- too thin a margin.
        At cap=8, the live-measured peak was ~190.8Mi (~74.5%, ~25.5%
        headroom)."""
        for name in ("constant_memory", "constant_mixed", "ramp_mixed", "burst_mixed"):
            path = Path("configs/traffic_profiles/v2") / f"{name}.yaml"
            profile = parse_traffic_profile(path)
            assert profile.memory_safe_max_in_flight == 8
            assert profile.memory_safe_max_in_flight * profile.memory_mb < 256  # MiB, comfortably under pod limit


class TestScheduleCsvUnaffected:
    def test_schedule_csv_fieldnames_do_not_include_resource_fields(self):
        """resource_mode/memory_mb are whole-profile, not per-request --
        must never appear in the frozen schedule CSV schema."""
        from loadgen.traffic_generator import SCHEDULE_CSV_FIELDNAMES
        assert "resource_mode" not in SCHEDULE_CSV_FIELDNAMES
        assert "memory_mb" not in SCHEDULE_CSV_FIELDNAMES
