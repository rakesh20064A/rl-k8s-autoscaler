"""autoscaler/src/rl_env/reward.py

Frozen V2 reward formula (docs/rl_mdp_spec.md sec.15-19, CORR-27,
configs/reward.yaml reward_contract_version=2). Pure function -- no state,
no I/O. Coefficients match configs/reward.yaml exactly; `config.py`'s
`load_env_config_from_yaml()` cross-checks this module's defaults against
the YAML file so the two never silently drift apart.

Phase 15B.1A / CORR-30: `compute_reward()` takes two SEPARATE validity
flags -- `decision_state_valid` (was s_t, the state the action was chosen
from, valid?) and `outcome_state_valid` (is s_(t+1), the resulting state,
valid?) -- rather than one collapsed `transition_fully_valid` flag. This
corrects a Phase 15B.1 defect where an invalid OUTCOME observation would
retroactively erase an already-determined, valid policy action's
action/reversal/boundary consequences. Per docs/rl_mdp_spec.md sec.17,
R_action/R_reversal/R_boundary are functions of `override_reason`/
`effective_delta`/`is_reversal` alone -- all of which are fully determined
at decision time, before the outcome is ever observed -- so their
eligibility must depend only on `decision_state_valid`, never on
`outcome_state_valid`. R_sla/R_reliability/R_efficiency depend on outcome
data and are correctly zeroed when that data is unavailable. See CORR-30
for the full four-case (decision x outcome) validity matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .types import (
    OVERRIDE_NONE,
    OVERRIDE_PENDING_SCALE_GATE,
    OVERRIDE_REPLICA_BOUND_CLAMP,
    OVERRIDE_SCALE_DOWN_COOLDOWN,
)

REWARD_VALIDITY_FULL = "full"
REWARD_VALIDITY_PARTIAL = "partial"

# Override reasons for which a policy-requested non-HOLD action still
# incurs the small action/churn penalty (docs/rl_mdp_spec.md sec.17):
# the policy made a real, evaluable decision, whether or not it was
# realized. Telemetry-failure reasons and replica_bound_clamp are excluded
# (the latter is covered by the boundary penalty instead, never both).
_ACTION_PENALTY_ELIGIBLE_REASONS = frozenset(
    {OVERRIDE_NONE, OVERRIDE_PENDING_SCALE_GATE, OVERRIDE_SCALE_DOWN_COOLDOWN}
)


@dataclass(frozen=True)
class RewardConfig:
    w_sla: float = 1.0
    sla_p95_target_ms: float = 300.0
    sla_excess_clip: float = 4.0

    w_reliability: float = 5.0

    w_resource: float = 1.0
    min_replicas: int = 1
    max_replicas: int = 8

    w_action: float = 0.1
    w_reversal: float = 0.5
    reversal_window_seconds: float = 120.0
    w_boundary: float = 0.2

    w_efficiency: float = 0.3
    efficiency_cpu_low: float = 1.0
    efficiency_cpu_high: float = 3.0


@dataclass(frozen=True)
class RewardResult:
    total: float
    components: dict
    reward_validity: str


def _resource_component(deployment_current_replicas: Optional[int], config: RewardConfig) -> float:
    if deployment_current_replicas is None:
        return 0.0
    resource_fraction = (deployment_current_replicas - config.min_replicas) / (
        config.max_replicas - config.min_replicas
    )
    return config.w_resource * max(0.0, min(1.0, resource_fraction))


def _reliability_component(error_rate: Optional[float], config: RewardConfig) -> float:
    if error_rate is None:
        return 0.0
    return config.w_reliability * max(0.0, min(1.0, error_rate))


def _action_side_components(
    requested_delta: int, effective_delta: int, override_reason: str, is_reversal: bool, config: RewardConfig
) -> dict:
    """R_action / R_reversal / R_boundary depend ONLY on decision-time
    quantities (override_reason, effective_delta, is_reversal) -- never on
    whether the OUTCOME observation was later valid. Callers must only
    invoke this when `decision_state_valid` is True (docs/rl_mdp_spec.md
    sec.17-18, CORR-30)."""
    action_penalty_applies = requested_delta != 0 and override_reason in _ACTION_PENALTY_ELIGIBLE_REASONS
    reversal_applies = effective_delta != 0 and is_reversal
    boundary_applies = override_reason == OVERRIDE_REPLICA_BOUND_CLAMP
    return {
        "action": config.w_action if action_penalty_applies else 0.0,
        "reversal": config.w_reversal if reversal_applies else 0.0,
        "boundary": config.w_boundary if boundary_applies else 0.0,
    }


def compute_reward(
    *,
    decision_state_valid: bool,
    outcome_state_valid: bool,
    p95_latency_ms: Optional[float],
    error_rate: Optional[float],
    deployment_current_replicas: Optional[int],
    cpu_per_replica: Optional[float],
    requested_delta: int,
    effective_delta: int,
    override_reason: str,
    is_reversal: bool,
    config: RewardConfig = RewardConfig(),
) -> RewardResult:
    """Compute the V2 reward for one transition s_t -> s_(t+1), CORR-30's
    corrected four-case validity matrix:

      A) decision_state_valid=True,  outcome_state_valid=True  -> full reward,
         every component computed normally.
      B) decision_state_valid=False                            -> safety-HOLD:
         action/reversal/boundary/efficiency/SLA all zero; reliability/
         resource computed only if independently valid at the outcome.
      C) decision_state_valid=True,  outcome_state_valid=False -> the policy's
         action/reversal/boundary consequences (already fully determined at
         decision time) are PRESERVED; SLA/efficiency are zeroed (no valid
         outcome latency); reliability/resource computed only if
         independently valid.
      D) decision_state_valid=False, outcome_state_valid=False -> same as (B).

    reward_validity is "full" only for case (A); "partial" for (B)/(C)/(D).
    """
    if not decision_state_valid:
        # Cases B/D: the policy did not have a trustworthy state to act on;
        # this step's action was never a real, fully-evaluated decision.
        components = {
            "sla": 0.0,
            "reliability": _reliability_component(error_rate, config),
            "resource": _resource_component(deployment_current_replicas, config),
            "action": 0.0,
            "reversal": 0.0,
            "boundary": 0.0,
            "efficiency_bonus": 0.0,
        }
        total = (
            -components["sla"] - components["reliability"] - components["resource"]
            - components["action"] - components["reversal"] - components["boundary"]
            + components["efficiency_bonus"]
        )
        return RewardResult(total, components, REWARD_VALIDITY_PARTIAL)

    # decision_state_valid is True from here on (cases A or C) -- the
    # action/reversal/boundary consequences of THIS decision are always
    # computable, regardless of whether the outcome telemetry later failed.
    action_side = _action_side_components(requested_delta, effective_delta, override_reason, is_reversal, config)

    if not outcome_state_valid:
        # Case C: valid decision, invalid outcome. Preserve action-side
        # components; zero the outcome-dependent ones (no genuine p95/CPU
        # reading to evaluate SLA/efficiency against); reliability/resource
        # computed only if independently valid at the outcome.
        components = {
            "sla": 0.0,
            "reliability": _reliability_component(error_rate, config),
            "resource": _resource_component(deployment_current_replicas, config),
            **action_side,
            "efficiency_bonus": 0.0,
        }
        total = (
            -components["sla"] - components["reliability"] - components["resource"]
            - components["action"] - components["reversal"] - components["boundary"]
            + components["efficiency_bonus"]
        )
        return RewardResult(total, components, REWARD_VALIDITY_PARTIAL)

    # Case A: fully valid transition -- normal, complete reward.
    excess = max(0.0, (p95_latency_ms - config.sla_p95_target_ms) / config.sla_p95_target_ms)
    sla = config.w_sla * min(excess ** 2, config.sla_excess_clip)

    reliability = config.w_reliability * max(0.0, min(1.0, error_rate if error_rate is not None else 0.0))
    resource = _resource_component(deployment_current_replicas, config)

    efficiency_applies = (
        cpu_per_replica is not None
        and config.efficiency_cpu_low <= cpu_per_replica <= config.efficiency_cpu_high
        and p95_latency_ms < config.sla_p95_target_ms
        and (error_rate is None or error_rate == 0.0)
    )
    efficiency_bonus = config.w_efficiency if efficiency_applies else 0.0

    components = {
        "sla": sla,
        "reliability": reliability,
        "resource": resource,
        **action_side,
        "efficiency_bonus": efficiency_bonus,
    }
    total = (
        -components["sla"] - components["reliability"] - components["resource"]
        - components["action"] - components["reversal"] - components["boundary"]
        + components["efficiency_bonus"]
    )
    return RewardResult(total, components, REWARD_VALIDITY_FULL)
