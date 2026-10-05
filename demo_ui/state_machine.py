"""state_machine.py -- Pure, offline state machines for the experiment
control flow (sec.6-7). No I/O, no Streamlit, no Kubernetes/PPO/HPA calls --
transitions are advanced explicitly by the caller (UI event loop or a test),
so these classes are fully unit-testable without a running server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from demo_ui.models import AutoRunState, Controller, ResetSnapshot, SingleRunState

# Canonical reset target -- NEVER replicas=0 (sec.7).
CANONICAL_RESET = ResetSnapshot(traffic_rps=0.0, replicas=1, application_ready=True, metrics_stabilizing=False)


class InvalidTransition(Exception):
    pass


_SINGLE_RUN_ORDER: List[SingleRunState] = [
    SingleRunState.IDLE,
    SingleRunState.PREPARING,
    SingleRunState.RUNNING_SELECTED_CONTROLLER,
    SingleRunState.RESETTING,
    SingleRunState.COMPLETE,
]

_AUTO_RUN_ORDER: List[AutoRunState] = [
    AutoRunState.IDLE,
    AutoRunState.PREPARING_HPA,
    AutoRunState.HPA_RUNNING,
    AutoRunState.HPA_COMPLETE,
    AutoRunState.RESETTING,
    AutoRunState.BASELINE_STABLE,
    AutoRunState.PREPARING_PPO,
    AutoRunState.PPO_RUNNING,
    AutoRunState.PPO_COMPLETE,
    AutoRunState.VALIDATING,
    AutoRunState.COMPARISON_READY,
]


@dataclass
class SingleControllerRun:
    """Auto OFF: IDLE -> PREPARING -> RUNNING_SELECTED_CONTROLLER ->
    RESETTING -> COMPLETE. Only `controller` runs; linear, no branching."""

    controller: Controller
    state: SingleRunState = SingleRunState.IDLE
    reset_snapshot: Optional[ResetSnapshot] = None

    def advance(self) -> SingleRunState:
        idx = _SINGLE_RUN_ORDER.index(self.state)
        if idx == len(_SINGLE_RUN_ORDER) - 1:
            raise InvalidTransition(f"cannot advance past terminal state {self.state}")
        next_state = _SINGLE_RUN_ORDER[idx + 1]
        if next_state == SingleRunState.RESETTING:
            self.reset_snapshot = None  # cleared until reset completes
        if next_state == SingleRunState.COMPLETE:
            self.reset_snapshot = CANONICAL_RESET
        self.state = next_state
        return self.state

    def stop(self) -> SingleRunState:
        """STOP forces an immediate transition to RESETTING regardless of
        current state (the control panel's [STOP] button, sec.5), then the
        normal advance() path continues to COMPLETE."""
        if self.state in (SingleRunState.RESETTING, SingleRunState.COMPLETE):
            return self.state
        self.state = SingleRunState.RESETTING
        self.reset_snapshot = None
        return self.state

    def is_terminal(self) -> bool:
        return self.state == SingleRunState.COMPLETE


@dataclass
class AutoComparisonRun:
    """Auto ON: fair-replay flow (sec.6) -- HPA runs first to produce a
    deterministic traffic schedule, environment resets to the canonical
    baseline, then PPO replays the IDENTICAL schedule, then validation and
    comparison. Linear with one branch point handled by the caller: VALIDATING
    can only reach COMPARISON_READY if schedule-hash equality holds (checked
    externally and passed into `validate()`)."""

    state: AutoRunState = AutoRunState.IDLE
    hpa_schedule_hash: Optional[str] = None
    ppo_schedule_hash: Optional[str] = None
    reset_snapshot: Optional[ResetSnapshot] = None

    def advance(self) -> AutoRunState:
        idx = _AUTO_RUN_ORDER.index(self.state)
        if idx == len(_AUTO_RUN_ORDER) - 1:
            raise InvalidTransition(f"cannot advance past terminal state {self.state}")
        if self.state == AutoRunState.VALIDATING:
            raise InvalidTransition("VALIDATING must transition via validate(), not advance()")
        next_state = _AUTO_RUN_ORDER[idx + 1]
        if next_state == AutoRunState.RESETTING:
            self.reset_snapshot = None
        if next_state == AutoRunState.BASELINE_STABLE:
            self.reset_snapshot = CANONICAL_RESET
        self.state = next_state
        return self.state

    def validate(self, same_schedule_verified: bool) -> AutoRunState:
        if self.state != AutoRunState.VALIDATING:
            raise InvalidTransition(f"validate() only valid from VALIDATING, currently {self.state}")
        if not same_schedule_verified:
            raise InvalidTransition(
                "fair-replay invariant violated: HPA and PPO schedule hashes must match before COMPARISON_READY"
            )
        self.state = AutoRunState.COMPARISON_READY
        return self.state

    def is_terminal(self) -> bool:
        return self.state == AutoRunState.COMPARISON_READY
