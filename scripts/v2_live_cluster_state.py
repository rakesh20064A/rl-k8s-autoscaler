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

from typing import List

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
