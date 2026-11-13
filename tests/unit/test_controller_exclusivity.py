"""tests/unit/test_controller_exclusivity.py

V2 Phase 2 PART 4/9: proves the new mutual-exclusion gate between the three
controllers (PPO, CPU-only HPA, CPU+memory HPA). All kubectl access is
faked -- no real cluster needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pytest

from autoscaler.src.rl_env.controller_exclusivity import (
    ALL_CONTROLLERS,
    CONTROLLER_CPU_HPA,
    CONTROLLER_CPU_MEMORY_HPA,
    CONTROLLER_CPU_ONLY_HPA_V2,
    CONTROLLER_NONE,
    CONTROLLER_PPO,
    HPA_NAME_BY_CONTROLLER,
    ControllerConflictError,
    acquire_controller,
    clear_lock,
    current_status,
    detect_conflicting_hpas,
    read_lock,
    release_controller,
)


@dataclass
class FakeHpaLister:
    names: List[str] = field(default_factory=list)

    def list_hpa_names(self) -> List[str]:
        return list(self.names)


@pytest.fixture()
def lock_path(tmp_path):
    return tmp_path / "controller" / "active_controller.json"


class TestReadWriteLock:
    def test_missing_lock_reads_as_none(self, lock_path):
        assert read_lock(lock_path) == CONTROLLER_NONE

    def test_acquire_then_read_reflects_controller(self, lock_path):
        acquire_controller(CONTROLLER_PPO, FakeHpaLister(), lock_path)
        assert read_lock(lock_path) == CONTROLLER_PPO

    def test_clear_lock_resets_to_none(self, lock_path):
        acquire_controller(CONTROLLER_PPO, FakeHpaLister(), lock_path)
        clear_lock(lock_path)
        assert read_lock(lock_path) == CONTROLLER_NONE

    def test_corrupt_lock_file_reads_as_none(self, lock_path):
        lock_path.parent.mkdir(parents=True)
        lock_path.write_text("not valid json {{{")
        assert read_lock(lock_path) == CONTROLLER_NONE


class TestDetectConflictingHpas:
    def test_ppo_conflicts_with_any_hpa(self):
        assert detect_conflicting_hpas(CONTROLLER_PPO, FakeHpaLister(["demo-app-hpa"])) == ["demo-app-hpa"]

    def test_ppo_no_conflict_when_no_hpa(self):
        assert detect_conflicting_hpas(CONTROLLER_PPO, FakeHpaLister([])) == []

    def test_cpu_hpa_conflicts_with_other_hpa_name_only(self):
        assert detect_conflicting_hpas(CONTROLLER_CPU_HPA, FakeHpaLister(["demo-app-hpa-v2"])) == ["demo-app-hpa-v2"]

    def test_cpu_hpa_no_conflict_with_its_own_name(self):
        assert detect_conflicting_hpas(CONTROLLER_CPU_HPA, FakeHpaLister(["demo-app-hpa"])) == []

    def test_cpu_memory_hpa_conflicts_with_cpu_hpa(self):
        assert detect_conflicting_hpas(CONTROLLER_CPU_MEMORY_HPA, FakeHpaLister(["demo-app-hpa"])) == ["demo-app-hpa"]

    def test_cpu_memory_hpa_no_conflict_with_its_own_name(self):
        assert detect_conflicting_hpas(CONTROLLER_CPU_MEMORY_HPA, FakeHpaLister(["demo-app-hpa-v2"])) == []


class TestAcquireControllerFailsClosed:
    def test_ppo_refused_when_hpa_present(self, lock_path):
        with pytest.raises(ControllerConflictError):
            acquire_controller(CONTROLLER_PPO, FakeHpaLister(["demo-app-hpa"]), lock_path)
        assert read_lock(lock_path) == CONTROLLER_NONE  # failed acquire must not write the lock

    def test_ppo_succeeds_when_no_hpa(self, lock_path):
        acquire_controller(CONTROLLER_PPO, FakeHpaLister([]), lock_path)
        assert read_lock(lock_path) == CONTROLLER_PPO

    def test_cpu_hpa_refused_when_ppo_already_locked(self, lock_path):
        acquire_controller(CONTROLLER_PPO, FakeHpaLister([]), lock_path)
        with pytest.raises(ControllerConflictError):
            acquire_controller(CONTROLLER_CPU_HPA, FakeHpaLister([]), lock_path)

    def test_cpu_hpa_refused_when_cpu_memory_hpa_object_present(self, lock_path):
        with pytest.raises(ControllerConflictError):
            acquire_controller(CONTROLLER_CPU_HPA, FakeHpaLister(["demo-app-hpa-v2"]), lock_path)

    def test_reacquiring_same_controller_is_idempotent_success(self, lock_path):
        acquire_controller(CONTROLLER_PPO, FakeHpaLister([]), lock_path)
        acquire_controller(CONTROLLER_PPO, FakeHpaLister([]), lock_path)  # must not raise
        assert read_lock(lock_path) == CONTROLLER_PPO

    def test_unknown_controller_rejected(self, lock_path):
        with pytest.raises(ValueError):
            acquire_controller("not_a_real_controller", FakeHpaLister([]), lock_path)

    def test_two_hpas_coexisting_blocks_ppo(self, lock_path):
        with pytest.raises(ControllerConflictError):
            acquire_controller(CONTROLLER_PPO, FakeHpaLister(["demo-app-hpa", "demo-app-hpa-v2"]), lock_path)


class TestReleaseController:
    def test_release_matching_controller_clears_lock(self, lock_path):
        acquire_controller(CONTROLLER_PPO, FakeHpaLister([]), lock_path)
        release_controller(CONTROLLER_PPO, lock_path)
        assert read_lock(lock_path) == CONTROLLER_NONE

    def test_release_non_matching_controller_is_a_noop(self, lock_path):
        acquire_controller(CONTROLLER_PPO, FakeHpaLister([]), lock_path)
        release_controller(CONTROLLER_CPU_HPA, lock_path)  # not the lock holder -- must not clear it
        assert read_lock(lock_path) == CONTROLLER_PPO

    def test_release_when_no_lock_held_is_a_noop(self, lock_path):
        release_controller(CONTROLLER_PPO, lock_path)  # must not raise
        assert read_lock(lock_path) == CONTROLLER_NONE


class TestCurrentStatus:
    def test_reports_both_signals(self, lock_path):
        acquire_controller(CONTROLLER_CPU_HPA, FakeHpaLister(["demo-app-hpa"]), lock_path)
        status = current_status(FakeHpaLister(["demo-app-hpa"]), lock_path)
        assert status.locked_controller == CONTROLLER_CPU_HPA
        assert status.active_hpa_names == ["demo-app-hpa"]


class TestCpuOnlyHpaV2Identity:
    """V2 Phase 5C / FREEZE_MANIFEST_AMENDMENT_002 / CORR-62: the third
    evaluation arm's controller identity, added post-training (additive
    only -- does not alter any pre-existing constant/behavior)."""

    def test_constant_value(self):
        assert CONTROLLER_CPU_ONLY_HPA_V2 == "cpu_only_hpa_v2"

    def test_registered_in_all_controllers(self):
        assert CONTROLLER_CPU_ONLY_HPA_V2 in ALL_CONTROLLERS
        assert len(ALL_CONTROLLERS) == 4

    def test_hpa_name_mapping(self):
        assert HPA_NAME_BY_CONTROLLER[CONTROLLER_CPU_ONLY_HPA_V2] == "demo-app-hpa-v2-cpu"

    def test_acquire_and_release_round_trip(self, lock_path):
        acquire_controller(CONTROLLER_CPU_ONLY_HPA_V2, FakeHpaLister(["demo-app-hpa-v2-cpu"]), lock_path)
        assert read_lock(lock_path) == CONTROLLER_CPU_ONLY_HPA_V2
        release_controller(CONTROLLER_CPU_ONLY_HPA_V2, lock_path)
        assert read_lock(lock_path) == CONTROLLER_NONE

    def test_ppo_blocked_while_cpu_only_hpa_v2_active(self, lock_path):
        with pytest.raises(ControllerConflictError):
            acquire_controller(CONTROLLER_PPO, FakeHpaLister(["demo-app-hpa-v2-cpu"]), lock_path)


class TestPpoNeverRunsWhileHpaActive:
    """End-to-end illustration of the task's own framing: 'PPO must never
    run while either HPA actively controls the deployment.'"""

    @pytest.mark.parametrize("hpa_name", ["demo-app-hpa", "demo-app-hpa-v2"])
    def test_ppo_blocked_by_either_hpa_variant(self, hpa_name, lock_path):
        with pytest.raises(ControllerConflictError):
            acquire_controller(CONTROLLER_PPO, FakeHpaLister([hpa_name]), lock_path)
