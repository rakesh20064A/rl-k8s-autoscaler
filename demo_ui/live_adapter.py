"""live_adapter.py -- Future live ExperimentController structure (sec.18).

Implements the `ExperimentController` Protocol with a `dry_run` flag that
DEFAULTS TO TRUE and an explicit, separate safety gate
(`i_understand_this_will_mutate_the_live_cluster=True`) that must ALSO be
set before `start()` will do anything beyond validating/logging intent.

In this phase, this class NEVER executes a real mutation: `start()` always
records a `PlannedOperation` and returns without touching Kubernetes,
Prometheus, the traffic generator, or PPO -- whether or not the safety gate
is set. The gate exists so that a FUTURE phase can add real execution
behind it without changing this class's call signature or the pages that
call it today.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from demo_ui.models import Controller, TrafficProfile


class LiveMutationNotImplementedError(RuntimeError):
    """Raised if calling code ever tries to flip this adapter into executing
    a real mutation in this phase. There is intentionally no code path that
    performs a real kubectl/HPA/traffic/PPO action here yet."""


@dataclass
class PlannedOperation:
    """A structured record of what WOULD be done, for display/audit -- never
    executed in this phase."""
    step: str
    detail: str


@dataclass
class LiveExperimentController:
    """Future live Auto-ON adapter: HPA -> reset -> PPO -> compare (sec.18).

    Safety contract:
      - `dry_run` defaults to True and nothing in this phase ever sets it
        to False from within demo_ui itself.
      - Even if a caller constructs this with `dry_run=False`, `start()`
        still only plans operations -- real execution requires both
        `dry_run=False` AND `i_understand_this_will_mutate_the_live_cluster=True`,
        AND is further gated by `_EXECUTION_ENABLED` below, which this
        phase hard-codes to False. A future phase removes that hard-coded
        gate explicitly, as its own reviewable change.
    """

    dry_run: bool = True
    i_understand_this_will_mutate_the_live_cluster: bool = False

    _EXECUTION_ENABLED: bool = field(default=False, init=False, repr=False)

    planned_operations: List[PlannedOperation] = field(default_factory=list)
    is_running: bool = field(default=False, init=False)

    def start(self, controller: Controller, traffic_profile: TrafficProfile) -> None:
        if self.is_running:
            # Guards against duplicate launches from Streamlit reruns
            # (sec.19) -- a second start() while one is already "running"
            # (even in dry-run bookkeeping) is a no-op, not a second plan.
            return

        self.planned_operations = [
            PlannedOperation("verify_prerequisites", "would verify Prometheus/HPA/Deployment healthy, read-only"),
            PlannedOperation("start_hpa_phase", f"would run HPA controller against {traffic_profile.value} traffic"),
            PlannedOperation("canonical_reset", "would reset to traffic=0, replicas=1, application Ready"),
            PlannedOperation("start_ppo_phase", f"would replay IDENTICAL {traffic_profile.value} schedule under PPO"),
            PlannedOperation("validate_fair_replay", "would verify HPA/PPO schedule hashes match"),
            PlannedOperation("compute_comparison", "would run comparison_engine.compare() on real RunSummary data"),
        ]
        self.is_running = True

        if not self.dry_run and self.i_understand_this_will_mutate_the_live_cluster and self._EXECUTION_ENABLED:
            raise LiveMutationNotImplementedError(
                "Real execution is not implemented in this phase. This branch is unreachable "
                "because _EXECUTION_ENABLED is hard-coded False -- a future phase must change "
                "that as its own explicit, reviewable step."
            )
        # Otherwise: dry-run only. No kubectl/HPA/traffic/PPO call happens.

    def stop(self) -> None:
        self.is_running = False

    def reset(self) -> None:
        self.planned_operations = []
        self.is_running = False

    def describe_plan(self) -> List[str]:
        return [f"{op.step}: {op.detail}" for op in self.planned_operations]
