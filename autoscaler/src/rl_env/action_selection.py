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
from typing import Sequence

import numpy as np

from .types import ACTION_MAP, ACTION_NAMES

# Derived from the frozen ACTION_MAP, not hardcoded -- if that contract's
# index assignment ever changed (it must not), this still finds whichever
# index represents "no change" rather than silently assuming 1.
HOLD_ACTION_INDEX: int = next(idx for idx, delta in ACTION_MAP.items() if delta == 0)

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
