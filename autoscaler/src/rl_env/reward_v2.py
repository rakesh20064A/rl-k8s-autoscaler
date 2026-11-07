"""autoscaler/src/rl_env/reward_v2.py

V2 Phase 2.1 PART 6: a NEW, parallel multi-metric reward function. Does NOT
modify or import from the frozen `reward.py`/`configs/reward.yaml` -- V1's
reward is used verbatim by any code still constructing `RewardConfig`
(reward.py) directly; this module is an entirely separate, additively-used
alternative, read from `configs/reward_v2.yaml`.

Motivation (see tests/unit/test_reward_v1_audit.py for the full numeric
evidence this design responds to):
  1. V1's resource term is a CONSTANT per-step cost proportional to replica
     count, paid regardless of whether those replicas are currently needed,
     while the SLA penalty only fires DURING an actual violation. Averaged
     over time (where healthy-SLA steps vastly outnumber violation steps
     under normal load), this structurally favors staying small -- fewer
     replicas is "free money" every single healthy step, SLA risk is a
     comparatively rare tax. V2 replaces the unconditional resource-count
     cost with a WASTE penalty that only applies when the policy is
     CURRENTLY over-provisioned relative to actual (healthy) load -- it is
     mathematically impossible to earn this penalty's absence by staying
     small during an SLA violation, because the condition requires SLA to
     already be healthy.
  2. V1's reward has NO memory input anywhere in its signature -- a pod can
     be arbitrarily close to its OOM limit with zero reward consequence.
     V2 adds an explicit, symmetric CPU-and-memory "unsafe pressure"
     penalty (not merely a vague "efficiency band"): pressure is measured
     the SAME way for both resources (distance from a safe ceiling toward
     the pod's actual hard limit, quadratic ramp), and the WORSE of the two
     drives the penalty (max, not sum) -- mirroring the multi-metric HPA
     v2's own "largest recommendation wins" semantics
     (k8s/hpa/hpa_v2_cpu_memory.yaml) for a consistent mental model across
     the project.

Formula (every term is a function of information available AT or BEFORE
decision time t -- see module docstring's "no future/unobservable
information" constraint, and docs/rl_mdp_spec.md sec.2's step-order
contract, unchanged by this module):

    R_v2 = - w_reliability * error_rate
           - w_sla         * clip(excess_sla^2, 0, 1)
           - w_pressure    * max(pressure_cpu, pressure_memory)
           - w_waste       * 1{waste_applies}
           - w_action      * 1{action_penalty_applies}
           - w_reversal    * 1{reversal_applies}
           - w_boundary    * 1{boundary_applies}

    excess_sla      = max(0, (p95_ms - sla_p95_target_ms) / sla_p95_target_ms)
    pressure_cpu    = clip((cpu_per_replica - cpu_safe_high) / (cpu_hard_limit - cpu_safe_high), 0, 1) ** 2
    pressure_memory = clip((mem_per_replica - mem_safe_high) / (mem_hard_limit - mem_safe_high), 0, 1) ** 2
    waste_applies   = current_replicas > min_replicas
                      AND cpu_per_replica < cpu_waste_threshold
                      AND mem_per_replica < mem_waste_threshold
                      AND p95_ms < sla_p95_target_ms
                      AND error_rate == 0

action/reversal/boundary terms are UNCHANGED in shape from V1 (same
override-reason-driven eligibility rules, `reward.py`'s
`_action_side_components()` logic re-expressed here to avoid importing from
the frozen module) -- no new evidence this phase suggested changing their
mechanics, only their relative weight versus the redesigned SLA/pressure/
waste terms.

Priority ordering realized through MAXIMUM possible per-step penalty
magnitude (task's required priority: reliability > SLA > safe resource
pressure > efficiency/waste > scaling churn):
    reliability   max = w_reliability * 1.0  = 6.0
    sla           max = w_sla * 1.0          = 4.0
    pressure      max = w_pressure * 1.0     = 2.0
    waste         max = w_waste              = 0.3
    reversal      max = w_reversal           = 0.5
    boundary      max = w_boundary           = 0.2
    action        max = w_action             = 0.1
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from .types import (
    OVERRIDE_NONE,
    OVERRIDE_PENDING_SCALE_GATE,
    OVERRIDE_REPLICA_BOUND_CLAMP,
    OVERRIDE_SCALE_DOWN_COOLDOWN,
)

REWARD_VALIDITY_FULL = "full"
REWARD_VALIDITY_PARTIAL = "partial"

_ACTION_PENALTY_ELIGIBLE_REASONS = frozenset(
    {OVERRIDE_NONE, OVERRIDE_PENDING_SCALE_GATE, OVERRIDE_SCALE_DOWN_COOLDOWN}
)


@dataclass(frozen=True)
class RewardConfigV2:
    w_reliability: float = 6.0

    w_sla: float = 4.0
    sla_p95_target_ms: float = 300.0

    w_pressure: float = 2.0
    # CPU safe ceiling / hard limit expressed as a multiple of the 100m CPU
    # request (same units as observation index 0, cpu_per_replica).
    # cpu_safe_high=3.0 matches V1's own efficiency_cpu_high (CORR-23,
    # application-verified SLA-safe at 6 RPS); cpu_hard_limit=5.0 is the
    # pod's ACTUAL CPU ceiling (500m limit / 100m request,
    # k8s/app/deployment.yaml) -- not an arbitrary number.
    cpu_safe_high: float = 3.0
    cpu_hard_limit: float = 5.0
    # Memory safe ceiling / hard limit expressed as a multiple of the 64Mi
    # memory request. mem_safe_high=1.2 matches the CALIBRATED_LIVE V2 HPA
    # memory target (k8s/hpa/hpa_v2_cpu_memory.yaml, V2 Phase 2.1 PART 4);
    # mem_hard_limit=4.0 is the pod's ACTUAL memory ceiling (256Mi limit /
    # 64Mi request).
    mem_safe_high: float = 1.2
    mem_hard_limit: float = 4.0

    w_waste: float = 0.3
    min_replicas: int = 1
    max_replicas: int = 8
    # Waste thresholds: well below the respective safe-pressure ceilings,
    # so waste and pressure can never both apply to the same reading.
    cpu_waste_threshold: float = 0.3
    mem_waste_threshold: float = 0.7

    w_action: float = 0.1
    w_reversal: float = 0.5
    reversal_window_seconds: float = 120.0
    w_boundary: float = 0.2


@dataclass(frozen=True)
class RewardResultV2:
    total: float
    components: dict
    reward_validity: str


DEFAULT_REWARD_V2_YAML = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "reward_v2.yaml"


def load_reward_config_v2_from_yaml(path: Path = DEFAULT_REWARD_V2_YAML) -> RewardConfigV2:
    """Parses configs/reward_v2.yaml into a RewardConfigV2. Deliberately
    SEPARATE from config.py's `load_env_config_from_yaml()` -- that loader
    is tied to the frozen V1 `RewardConfig` schema (reward.py) and must
    never be pointed at this file's genuinely different V2 schema."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    weights = cfg["weights"]
    pressure = cfg["pressure"]
    waste = cfg["waste"]
    return RewardConfigV2(
        w_reliability=float(weights["w_reliability"]),
        w_sla=float(weights["w_sla"]),
        sla_p95_target_ms=float(cfg["sla"]["p95_target_ms"]),
        w_pressure=float(weights["w_pressure"]),
        cpu_safe_high=float(pressure["cpu_safe_high"]),
        cpu_hard_limit=float(pressure["cpu_hard_limit"]),
        mem_safe_high=float(pressure["memory_safe_high"]),
        mem_hard_limit=float(pressure["memory_hard_limit"]),
        w_waste=float(weights["w_waste"]),
        cpu_waste_threshold=float(waste["cpu_waste_threshold"]),
        mem_waste_threshold=float(waste["memory_waste_threshold"]),
        w_action=float(weights["w_action"]),
        w_reversal=float(weights["w_reversal"]),
        reversal_window_seconds=float(cfg["reversal"]["reversal_window_seconds"]),
        w_boundary=float(weights["w_boundary"]),
    )


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _pressure(value: Optional[float], safe_high: float, hard_limit: float) -> float:
    if value is None:
        return 0.0
    return _clip01((value - safe_high) / (hard_limit - safe_high)) ** 2


def _action_side_components(
    requested_delta: int, effective_delta: int, override_reason: str, is_reversal: bool, config: RewardConfigV2,
) -> dict:
    action_penalty_applies = requested_delta != 0 and override_reason in _ACTION_PENALTY_ELIGIBLE_REASONS
    reversal_applies = effective_delta != 0 and is_reversal
    boundary_applies = override_reason == OVERRIDE_REPLICA_BOUND_CLAMP
    return {
        "action": config.w_action if action_penalty_applies else 0.0,
        "reversal": config.w_reversal if reversal_applies else 0.0,
        "boundary": config.w_boundary if boundary_applies else 0.0,
    }


def compute_reward_v2(
    *,
    decision_state_valid: bool,
    outcome_state_valid: bool,
    p95_latency_ms: Optional[float],
    error_rate: Optional[float],
    deployment_current_replicas: Optional[int],
    cpu_per_replica: Optional[float],
    memory_per_replica: Optional[float],
    requested_delta: int,
    effective_delta: int,
    override_reason: str,
    is_reversal: bool,
    config: RewardConfigV2 = RewardConfigV2(),
) -> RewardResultV2:
    """V2 multi-metric reward. Mirrors `reward.compute_reward()`'s four-case
    decision/outcome validity matrix EXACTLY (CORR-30) -- only the
    outcome-dependent component formulas themselves differ (SLA/pressure/
    waste replace SLA/resource/efficiency)."""
    if not decision_state_valid:
        components = {
            "reliability": config.w_reliability * _clip01(error_rate) if error_rate is not None else 0.0,
            "sla": 0.0, "pressure": 0.0, "waste": 0.0,
            "action": 0.0, "reversal": 0.0, "boundary": 0.0,
        }
        total = -sum(v for k, v in components.items())
        return RewardResultV2(total, components, REWARD_VALIDITY_PARTIAL)

    action_side = _action_side_components(requested_delta, effective_delta, override_reason, is_reversal, config)

    if not outcome_state_valid:
        components = {
            "reliability": config.w_reliability * _clip01(error_rate) if error_rate is not None else 0.0,
            "sla": 0.0, "pressure": 0.0, "waste": 0.0,
            **action_side,
        }
        total = -sum(v for k, v in components.items())
        return RewardResultV2(total, components, REWARD_VALIDITY_PARTIAL)

    reliability = config.w_reliability * _clip01(error_rate if error_rate is not None else 0.0)

    excess = max(0.0, (p95_latency_ms - config.sla_p95_target_ms) / config.sla_p95_target_ms)
    sla = config.w_sla * _clip01(excess ** 2)

    pressure_cpu = _pressure(cpu_per_replica, config.cpu_safe_high, config.cpu_hard_limit)
    pressure_mem = _pressure(memory_per_replica, config.mem_safe_high, config.mem_hard_limit)
    pressure = config.w_pressure * max(pressure_cpu, pressure_mem)

    waste_applies = (
        deployment_current_replicas is not None
        and deployment_current_replicas > config.min_replicas
        and cpu_per_replica is not None and cpu_per_replica < config.cpu_waste_threshold
        and memory_per_replica is not None and memory_per_replica < config.mem_waste_threshold
        and p95_latency_ms < config.sla_p95_target_ms
        and (error_rate is None or error_rate == 0.0)
    )
    waste = config.w_waste if waste_applies else 0.0

    components = {
        "reliability": reliability, "sla": sla, "pressure": pressure, "waste": waste,
        **action_side,
    }
    total = -sum(v for k, v in components.items())
    return RewardResultV2(total, components, REWARD_VALIDITY_FULL)
