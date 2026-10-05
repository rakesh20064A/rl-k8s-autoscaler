from __future__ import annotations

import pytest

from demo_ui.models import AutoRunState, Controller, SingleRunState
from demo_ui.state_machine import AutoComparisonRun, InvalidTransition, SingleControllerRun


class TestSingleControllerRun:
    def test_initial_state_is_idle(self):
        run = SingleControllerRun(controller=Controller.HPA)
        assert run.state == SingleRunState.IDLE

    def test_full_linear_sequence(self):
        run = SingleControllerRun(controller=Controller.PPO)
        expected = [
            SingleRunState.PREPARING,
            SingleRunState.RUNNING_SELECTED_CONTROLLER,
            SingleRunState.RESETTING,
            SingleRunState.COMPLETE,
        ]
        for state in expected:
            assert run.advance() == state

    def test_cannot_advance_past_complete(self):
        run = SingleControllerRun(controller=Controller.HPA)
        for _ in range(4):
            run.advance()
        with pytest.raises(InvalidTransition):
            run.advance()

    def test_complete_sets_canonical_reset_snapshot(self):
        run = SingleControllerRun(controller=Controller.HPA)
        for _ in range(4):
            run.advance()
        assert run.reset_snapshot is not None
        assert run.reset_snapshot.replicas == 1
        assert run.reset_snapshot.traffic_rps == 0.0

    def test_stop_forces_resetting_from_any_non_terminal_state(self):
        run = SingleControllerRun(controller=Controller.PPO)
        run.advance()  # PREPARING
        run.advance()  # RUNNING_SELECTED_CONTROLLER
        run.stop()
        assert run.state == SingleRunState.RESETTING

    def test_stop_is_noop_once_complete(self):
        run = SingleControllerRun(controller=Controller.HPA)
        for _ in range(4):
            run.advance()
        run.stop()
        assert run.state == SingleRunState.COMPLETE

    def test_only_selected_controller_is_recorded(self):
        run = SingleControllerRun(controller=Controller.PPO)
        assert run.controller == Controller.PPO


class TestAutoComparisonRun:
    def test_full_sequence_up_to_validating(self):
        run = AutoComparisonRun()
        expected = [
            AutoRunState.PREPARING_HPA,
            AutoRunState.HPA_RUNNING,
            AutoRunState.HPA_COMPLETE,
            AutoRunState.RESETTING,
            AutoRunState.BASELINE_STABLE,
            AutoRunState.PREPARING_PPO,
            AutoRunState.PPO_RUNNING,
            AutoRunState.PPO_COMPLETE,
            AutoRunState.VALIDATING,
        ]
        for state in expected:
            assert run.advance() == state

    def test_baseline_stable_carries_canonical_reset_snapshot(self):
        run = AutoComparisonRun()
        for _ in range(5):  # up to and including RESETTING -> BASELINE_STABLE
            run.advance()
        assert run.state == AutoRunState.BASELINE_STABLE
        assert run.reset_snapshot.replicas == 1

    def test_validate_requires_validating_state(self):
        run = AutoComparisonRun()
        with pytest.raises(InvalidTransition):
            run.validate(True)

    def test_validate_rejects_schedule_mismatch(self):
        run = AutoComparisonRun()
        for _ in range(9):
            run.advance()
        assert run.state == AutoRunState.VALIDATING
        with pytest.raises(InvalidTransition):
            run.validate(False)
        assert run.state == AutoRunState.VALIDATING  # unchanged on rejection

    def test_validate_accepts_matching_schedule_and_reaches_comparison_ready(self):
        run = AutoComparisonRun()
        for _ in range(9):
            run.advance()
        run.validate(True)
        assert run.state == AutoRunState.COMPARISON_READY
        assert run.is_terminal()

    def test_advance_cannot_be_used_from_validating(self):
        run = AutoComparisonRun()
        for _ in range(9):
            run.advance()
        with pytest.raises(InvalidTransition):
            run.advance()

    def test_cannot_advance_past_comparison_ready(self):
        run = AutoComparisonRun()
        for _ in range(9):
            run.advance()
        run.validate(True)
        with pytest.raises(InvalidTransition):
            run.advance()
