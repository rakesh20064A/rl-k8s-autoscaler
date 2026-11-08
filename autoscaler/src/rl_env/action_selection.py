"""autoscaler/src/rl_env/action_selection.py

V2 safe deterministic action selection -- NOT part of the frozen V1
evaluation path (scripts/evaluate_ppo_vs_hpa.py is untouched by this
module and does not import it). Addresses a concrete bias in plain
`model.predict(obs, deterministic=True)`: Stable-Baselines3's deterministic
Categorical-distribution choice is `torch.argmax(probs)`, which resolves
exact (and, via float rounding, near-exact) ties to the LOWEST index.
Since `.types.ACTION_MAP` is frozen as {0: SCALE_DOWN, 1: HOLD, 2: SCALE_UP}
(index order NEVER changed by this module), that means an uncertain policy
can end up silently biased toward SCALE_DOWN for no reason other than its
index being 0 -- never a deliberate policy preference. This module makes
"uncertain -> HOLD" an explicit, auditable V2 safety policy instead.

Pure functions only: no I/O, no model mutation, no training. Reusable by a
future V2 evaluation runner, V2 demo inference, and V2 training-time
validation alike.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .types import ACTION_MAP, ACTION_NAMES

# Derived from the frozen ACTION_MAP, not hardcoded -- if that contract's
# index assignment ever changed (it must not), this still finds whichever
# index represents "no change" rather than silently assuming 1.
HOLD_ACTION_INDEX: int = next(idx for idx, delta in ACTION_MAP.items() if delta == 0)
DOWN_ACTION_INDEX: int = next(idx for idx, delta in ACTION_MAP.items() if delta < 0)
UP_ACTION_INDEX: int = next(idx for idx, delta in ACTION_MAP.items() if delta > 0)

EXPECTED_NUM_ACTIONS = len(ACTION_MAP)

# V2 PROVISIONAL default -- explicitly NOT a validated scientific value.
# A conservative starting point only (roughly "within a few percent of
# uniform noise" for a 3-way softmax); must be revisited against real V2
# policy probability distributions before any final V2 evaluation run.
DEFAULT_NEAR_TIE_EPSILON = 0.05


@dataclass(frozen=True)
class SafeActionResult:
    action_index: int
    action_name: str
    is_tie: bool
    probabilities: tuple
    epsilon: float


def select_safe_deterministic_action(
    probabilities: Sequence[float], epsilon: float = DEFAULT_NEAR_TIE_EPSILON,
) -> SafeActionResult:
    """Selects a discrete action from `probabilities` (length == len(ACTION_MAP),
    one probability per action index, matching `ACTION_MAP`'s own index
    order -- never reordered here) using the V2 near-tie-safe rule:

      - exactly one action within `epsilon` of the maximum probability
        -> that action wins (a "clear winner", including the ordinary
        case where it IS the maximum with no other candidate close to it)
      - two or more actions within `epsilon` of the maximum (this always
        includes an EXACT tie, since epsilon >= 0)
        -> HOLD, regardless of which indices are tied or their order

    Raises ValueError on malformed input (wrong length, non-finite,
    negative, or epsilon < 0) -- never silently coerces bad input into a
    default action."""
    if epsilon < 0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")

    probs = np.asarray(probabilities, dtype=float)
    if probs.shape != (EXPECTED_NUM_ACTIONS,):
        raise ValueError(f"expected exactly {EXPECTED_NUM_ACTIONS} probabilities, got shape {probs.shape}")
    if not np.all(np.isfinite(probs)):
        raise ValueError(f"probabilities must all be finite, got {probs.tolist()}")
    if np.any(probs < 0):
        raise ValueError(f"probabilities must be non-negative, got {probs.tolist()}")

    max_p = float(probs.max())
    within_epsilon = np.flatnonzero(probs >= max_p - epsilon)

    if len(within_epsilon) > 1:
        chosen = HOLD_ACTION_INDEX
        is_tie = True
    else:
        chosen = int(within_epsilon[0])
        is_tie = False

    return SafeActionResult(
        action_index=chosen,
        action_name=ACTION_NAMES[chosen],
        is_tie=is_tie,
        probabilities=tuple(probs.tolist()),
        epsilon=epsilon,
    )


# -- V2 Phase 4D-E / CORR-58: safety-aware asymmetric tie-breaking -----------
#
# INFERENCE ONLY -- never used by training (see
# tests/unit/test_training_excludes_safe_tie_override.py). Replaces the
# plain near-tie-HOLD rule above (`select_safe_deterministic_action`, left
# completely UNCHANGED for any existing caller/test that still wants the
# simpler rule) with an asymmetric policy: SCALE_DOWN must never WIN a tie,
# and a HOLD/SCALE_UP tie resolves toward SCALE_UP only when the decision-
# time state is observably STRESSED -- "scale up fast, scale down slow",
# mirroring conventional HPA behavior (fast scale-up, conservative
# scale-down) rather than a blind coin flip.

TIE_RULE_CLEAR_WINNER = "R1"
TIE_RULE_HOLD_UP_STRESS_DEPENDENT = "R3"
TIE_RULE_DOWN_INVOLVES_UP = "R4"
TIE_RULE_DOWN_HOLD = "R5"
TIE_RULE_UNKNOWN_STRESS_FALLBACK = "R6"


@dataclass(frozen=True)
class SafetyAwareActionResult:
    """Richer sibling of `SafeActionResult`, additive -- used only by the
    V2 safety-aware selector below, never replaces `SafeActionResult`
    itself (which stays exactly as-is for any existing caller)."""

    action_index: int
    action_name: str
    is_tie: bool
    probabilities: tuple
    epsilon: float
    tie_set: Tuple[str, ...]  # action NAMES within epsilon of the max, in ACTION_MAP index order
    tie_rule_applied: str  # one of TIE_RULE_* above
    stressed: Optional[bool]  # None when stress could not be determined (R6) or irrelevant (R1)
    stress_reasons: Tuple[str, ...]  # empty unless stressed is True


def determine_decision_time_stress(
    *,
    p95_latency_ms: Optional[float],
    error_rate: Optional[float],
    cpu_per_replica: Optional[float],
    memory_per_replica: Optional[float],
    sla_p95_target_ms: float,
    cpu_safe_high: float,
    memory_safe_high: float,
) -> Tuple[Optional[bool], Tuple[str, ...]]:
    """STRESSED (V2 Phase 4D-E): the decision-time raw state is stressed if
    ANY of p95_latency_ms >= sla_p95_target_ms, error_rate > 0,
    cpu_per_replica > cpu_safe_high, or memory_per_replica > memory_safe_high
    holds. A missing (None) individual value counts as NOT stressed for
    that term alone -- but if ALL FOUR values are missing, stress cannot be
    determined at all, and this returns `(None, ())` so the caller falls
    back to R6 (HOLD) rather than guessing "not stressed" by default.
    Thresholds are ALWAYS passed in by the caller (read from the live
    `configs/reward_v2.yaml` via `RewardConfigV2`), never hardcoded here."""
    if p95_latency_ms is None and error_rate is None and cpu_per_replica is None and memory_per_replica is None:
        return None, ()

    reasons: List[str] = []
    if p95_latency_ms is not None and p95_latency_ms >= sla_p95_target_ms:
        reasons.append("p95_latency_ms>=sla_p95_target_ms")
    if error_rate is not None and error_rate > 0:
        reasons.append("error_rate>0")
    if cpu_per_replica is not None and cpu_per_replica > cpu_safe_high:
        reasons.append("cpu_per_replica>cpu_safe_high")
    if memory_per_replica is not None and memory_per_replica > memory_safe_high:
        reasons.append("memory_per_replica>memory_safe_high")
    return bool(reasons), tuple(reasons)


def select_safety_aware_action(
    probabilities: Sequence[float],
    epsilon: float = DEFAULT_NEAR_TIE_EPSILON,
    stressed: Optional[bool] = None,
    stress_reasons: Tuple[str, ...] = (),
) -> SafetyAwareActionResult:
    """V2 Phase 4D-E / CORR-58: the asymmetric safety-aware tie-break rule.
    `stressed` is the caller's own `determine_decision_time_stress()`
    result (None if it could not be determined, e.g. all four raw inputs
    were missing). Same input validation as
    `select_safe_deterministic_action` (raises ValueError on malformed
    input -- never silently coerces).

    Rule (within-epsilon tie set, by ACTION_MAP index order [DOWN, HOLD, UP]):
      R1 exactly one action within epsilon of the max -> that action.
      R3 tie set == {HOLD, UP} -> UP if stressed else HOLD; if `stressed`
         is None (could not be determined), falls back to R6 instead of
         guessing.
      R4 tie set == {DOWN, UP} or {DOWN, HOLD, UP} -> HOLD.
      R5 tie set == {DOWN, HOLD} -> HOLD.
      R6 `stressed` could not be determined for an otherwise-R3-eligible
         tie -> HOLD (the same fail-safe default as an invalid/missing
         observation elsewhere in the V2 safe-inference path).
    DOWN never wins any tie (R2) -- structurally guaranteed: every tie set
    that includes DOWN resolves via R4/R5, both HOLD, never DOWN itself."""
    if epsilon < 0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")

    probs = np.asarray(probabilities, dtype=float)
    if probs.shape != (EXPECTED_NUM_ACTIONS,):
        raise ValueError(f"expected exactly {EXPECTED_NUM_ACTIONS} probabilities, got shape {probs.shape}")
    if not np.all(np.isfinite(probs)):
        raise ValueError(f"probabilities must all be finite, got {probs.tolist()}")
    if np.any(probs < 0):
        raise ValueError(f"probabilities must be non-negative, got {probs.tolist()}")

    max_p = float(probs.max())
    within_epsilon = set(np.flatnonzero(probs >= max_p - epsilon).tolist())
    tie_set_names = tuple(ACTION_NAMES[i] for i in sorted(within_epsilon))

    if len(within_epsilon) == 1:
        chosen = next(iter(within_epsilon))
        return SafetyAwareActionResult(
            action_index=chosen, action_name=ACTION_NAMES[chosen], is_tie=False,
            probabilities=tuple(probs.tolist()), epsilon=epsilon, tie_set=tie_set_names,
            tie_rule_applied=TIE_RULE_CLEAR_WINNER, stressed=None, stress_reasons=(),
        )

    if DOWN_ACTION_INDEX in within_epsilon:
        rule = TIE_RULE_DOWN_INVOLVES_UP if UP_ACTION_INDEX in within_epsilon else TIE_RULE_DOWN_HOLD
        return SafetyAwareActionResult(
            action_index=HOLD_ACTION_INDEX, action_name=ACTION_NAMES[HOLD_ACTION_INDEX], is_tie=True,
            probabilities=tuple(probs.tolist()), epsilon=epsilon, tie_set=tie_set_names,
            tie_rule_applied=rule, stressed=None, stress_reasons=(),
        )

    # Only {HOLD, UP} remains (the only other possible 2-way tie with 3 actions).
    if stressed is None:
        return SafetyAwareActionResult(
            action_index=HOLD_ACTION_INDEX, action_name=ACTION_NAMES[HOLD_ACTION_INDEX], is_tie=True,
            probabilities=tuple(probs.tolist()), epsilon=epsilon, tie_set=tie_set_names,
            tie_rule_applied=TIE_RULE_UNKNOWN_STRESS_FALLBACK, stressed=None, stress_reasons=(),
        )

    chosen = UP_ACTION_INDEX if stressed else HOLD_ACTION_INDEX
    return SafetyAwareActionResult(
        action_index=chosen, action_name=ACTION_NAMES[chosen], is_tie=True,
        probabilities=tuple(probs.tolist()), epsilon=epsilon, tie_set=tie_set_names,
        tie_rule_applied=TIE_RULE_HOLD_UP_STRESS_DEPENDENT, stressed=stressed,
        stress_reasons=stress_reasons if stressed else (),
    )


@dataclass(frozen=True)
class MarginStats:
    """One validation state's probability-margin statistics -- the exact
    per-state record V2 Phase 2.1 PART 8's post-training epsilon
    calibration procedure requires. `argmax_action_index` is the PLAIN
    argmax (epsilon=0 exact-tie-only choice), reported alongside the margin
    so the eventual calibration can see both 'how close was this call' and
    'which action would plain argmax have picked', without this module
    choosing a final epsilon itself."""

    top1_probability: float
    top2_probability: float
    margin: float  # top1 - top2; the quantity a chosen epsilon is compared against
    argmax_action_index: int
    argmax_action_name: str


def compute_margin_stats(probabilities: Sequence[float]) -> MarginStats:
    """V2 Phase 2.1 PART 8: NOT epsilon selection itself -- purely the
    observational step of the documented post-training calibration
    procedure (see action_selection.py's module docstring addendum below
    and the V2 Phase 2.1 report's `post_training_epsilon_calibration_plan`
    for the full procedure). Call this for each validation-set
    observation's probability vector (from `extract_action_probabilities()`
    against a REAL trained V2 model) to build the empirical margin
    distribution a future phase will choose epsilon from.

    Raises ValueError on the same malformed input
    `select_safe_deterministic_action` rejects (wrong length, non-finite,
    negative) -- never silently coerces bad input."""
    probs = np.asarray(probabilities, dtype=float)
    if probs.shape != (EXPECTED_NUM_ACTIONS,):
        raise ValueError(f"expected exactly {EXPECTED_NUM_ACTIONS} probabilities, got shape {probs.shape}")
    if not np.all(np.isfinite(probs)):
        raise ValueError(f"probabilities must all be finite, got {probs.tolist()}")
    if np.any(probs < 0):
        raise ValueError(f"probabilities must be non-negative, got {probs.tolist()}")

    order = np.argsort(probs)[::-1]
    top1_idx, top2_idx = int(order[0]), int(order[1])
    top1, top2 = float(probs[top1_idx]), float(probs[top2_idx])
    return MarginStats(
        top1_probability=top1,
        top2_probability=top2,
        margin=top1 - top2,
        argmax_action_index=top1_idx,
        argmax_action_name=ACTION_NAMES[top1_idx],
    )


def extract_action_probabilities(model, observation: np.ndarray) -> np.ndarray:
    """Reads the categorical action-probability vector an SB3 PPO policy
    assigns to `observation`, WITHOUT calling `.predict()` (so this never
    performs the policy's own argmax/sampling) and WITHOUT any gradient
    step or parameter mutation -- a pure forward pass, same in spirit to
    the read-only inference `scripts/evaluate_ppo_vs_hpa.py` already does
    via `model.predict()`, just exposing the full distribution instead of
    only its collapsed argmax choice.

    `model` is any object exposing SB3's `.policy.get_distribution()` /
    `.policy.device` contract (duck-typed deliberately, so this has no
    hard dependency on a specific stable_baselines3 import at module load
    time beyond what's already a project dependency)."""
    import torch
    from stable_baselines3.common.utils import obs_as_tensor

    obs_batch = np.asarray(observation, dtype=np.float32).reshape(1, -1)
    with torch.no_grad():
        obs_tensor = obs_as_tensor(obs_batch, model.policy.device)
        distribution = model.policy.get_distribution(obs_tensor)
        probs = distribution.distribution.probs.cpu().numpy()[0]
    return probs
