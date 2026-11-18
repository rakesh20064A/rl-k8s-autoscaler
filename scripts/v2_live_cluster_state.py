#!/usr/bin/env python3
"""scripts/v2_live_cluster_state.py

One small, shared helper for a piece of live-cluster HPA logic that
`scripts/evaluate_v2_campaign.py`'s own `build_live_context().ppo_episode`
wrapper already implements inline for its live evaluation campaign:
before the `ppo_v2` arm may acquire the PPO controller,
`demo-app-hpa-v2` (the canonical IDLE state -- see that module's own
`restore_canonical_idle_state()`) must be deleted first, because the PPO
controller is a pure-Python controller that requires ZERO HPA objects
present (`controller_exclusivity.detect_conflicting_hpas()` treats ANY
HPA as conflicting for `CONTROLLER_PPO`).

`demo-app-hpa-v2` being present is NOT a conflict to refuse on -- it is
exactly the expected idle state -- it is simply cluster state that must
be torn down before PPO, never treated as a blocking precondition. This
module exists so `scripts/run_demo_episode_v2.py` (the demo UI's live
execution adapter target) can reuse the EXACT same one-liner instead of
re-implementing it, keeping a single source of truth for "how a PPO run
clears the idle HPA before it starts" across both live-cluster entry
points.
"""

from __future__ import annotations

import subprocess
from typing import Any, Dict, List

from autoscaler.src.rl_env.controller_exclusivity import V2_HPA_MANIFEST_PATH, V2_HPA_NAME


def remove_hpa_before_ppo_start() -> List[str]:
    """Idempotent: a no-op if no HPA object is present. Returns the list
    of HPA names that were present (and thus deleted) immediately before
    this call -- empty if the cluster was already clear."""
    from autoscaler.src.rl_env.live_backend import LiveKubernetesBackend

    backend = LiveKubernetesBackend(
        hpa_name=V2_HPA_NAME, hpa_manifest_path=V2_HPA_MANIFEST_PATH, min_replicas=2, initial_replicas=2,
    )
    existing = backend._kubectl.list_hpa_names()
    if existing:
        backend._kubectl.delete_hpa()
    return existing


# V2 Phase 9F: shared RECOVER logic -- used by both the demo UI (the
# Experiment page's RECOVER button, and automatically right after
# `experiment_store.reconcile()` finds a run newly gone stale/dead) and
# `scripts/demo_recover.py` (the terminal equivalent), so there is exactly
# one place that decides "is it actually safe to touch the live cluster
# right now, and if so, is there anything to fix."


def no_runner_process_alive(run_fn=subprocess.run) -> bool:
    """True only if NO `scripts/run_demo_episode_v2.py` process is
    currently running anywhere on this host (a cheap `pgrep -f`, the same
    mechanism that script's own training-protection check uses to look for
    OTHER scripts). Fails CLOSED: if the check itself cannot be performed
    (e.g. `pgrep` missing), this returns False -- never claim "no runner is
    alive" on doubt, since that claim is what gates a live-cluster mutation."""
    try:
        proc = run_fn(["pgrep", "-f", "run_demo_episode_v2.py"], capture_output=True, text=True, timeout=5)
    except Exception:  # noqa: BLE001
        return False
    return not proc.stdout.split()


def cluster_is_canonical_idle() -> bool:
    """True if the live cluster is already in the canonical idle state
    (controller lock is `none`, and the only HPA present, if any, is
    `demo-app-hpa-v2`) -- reuses `scripts/evaluate_v2_campaign.py`'s own
    `live_preflight_check()`, the single source of truth for what that
    state is, rather than re-deriving it here."""
    from scripts.evaluate_v2_campaign import live_preflight_check

    return bool(live_preflight_check().get("ok"))


def ensure_canonical_idle_state() -> Dict[str, Any]:
    """RECOVER: if (and only if) no demo runner process is alive anywhere
    on this host, checks whether the live cluster is already canonical
    idle and, if not, restores it (deletes any HPA that isn't
    `demo-app-hpa-v2`, re-applies it at 2 replicas, clears the controller
    lock). Never mutates the cluster while a runner might still be
    mid-cleanup itself -- that process's own `finally` is always given the
    chance to finish first. Never raises: a cluster/kubectl failure is
    reported in the returned dict, not thrown, so a UI page or recovery
    script can always show what happened instead of crashing over it."""
    if not no_runner_process_alive():
        return {"ran": False, "reason": "a run_demo_episode_v2.py process is still alive"}
    try:
        if cluster_is_canonical_idle():
            return {"ran": False, "reason": "cluster already in the canonical idle state"}
        from scripts.evaluate_v2_campaign import restore_canonical_idle_state

        result = restore_canonical_idle_state()
        return {"ran": True, **result}
    except Exception as exc:  # noqa: BLE001 -- never crash a RECOVER action over a cluster/kubectl failure
        return {"ran": False, "error": str(exc)}
