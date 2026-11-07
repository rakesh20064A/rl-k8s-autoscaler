"""autoscaler/src/rl_env/training_schedule_v2.py

V2 Phase 2.2 PART 8: the planned V2 training workload mix. Does NOT start
training -- defines a deterministic, seeded profile rotation ensuring the
agent experiences CPU-only, memory-only, mixed, smooth-changing (ramp), and
bursty-changing load, with no single profile dominating.

Design: each "cycle" is an independently shuffled permutation of all 5
V2_TRAINING_PROFILE_ALLOWLIST entries (autoscaler.src.rl_env.
training_contract_v2), using a `random.Random(seed)` instance that persists
across cycles (so cycle 2's shuffle is NOT a repeat of cycle 1's). This
guarantees every profile appears EXACTLY ONCE per 5-episode window aligned
to cycle boundaries -- a strictly stronger balance guarantee than "roughly
equal over many episodes" -- while still varying the WITHIN-cycle order
across cycles (never literally cycling through the same fixed sequence
every time, which risks the agent learning to exploit a predictable
sequence rather than the workload itself).

The final number of training timesteps is deliberately NOT decided here
(per the task's explicit instruction) -- `build_training_profile_rotation`
takes `n_episodes` as an input the caller supplies, not something this
module chooses.
"""

from __future__ import annotations

import random
from typing import List

from .training_contract_v2 import V2_TRAINING_PROFILE_ALLOWLIST


def build_training_profile_rotation(seed: int, n_episodes: int) -> List[str]:
    """Returns a length-`n_episodes` deterministic sequence of profile
    names drawn from V2_TRAINING_PROFILE_ALLOWLIST. Raises ValueError on a
    negative n_episodes -- never silently clamps to zero."""
    if n_episodes < 0:
        raise ValueError(f"n_episodes must be >= 0, got {n_episodes}")

    profiles = list(V2_TRAINING_PROFILE_ALLOWLIST)
    rng = random.Random(seed)
    schedule: List[str] = []
    while len(schedule) < n_episodes:
        cycle = list(profiles)
        rng.shuffle(cycle)
        schedule.extend(cycle)
    return schedule[:n_episodes]


def profile_occurrence_counts(schedule: List[str]) -> dict:
    """Small helper for reporting/auditing a built schedule's balance."""
    counts = {p: 0 for p in V2_TRAINING_PROFILE_ALLOWLIST}
    for p in schedule:
        counts[p] = counts.get(p, 0) + 1
    return counts
