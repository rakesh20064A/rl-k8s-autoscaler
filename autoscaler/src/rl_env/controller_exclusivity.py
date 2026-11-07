"""autoscaler/src/rl_env/controller_exclusivity.py

V2 Phase 2: minimal safe mutual-exclusion between the three controllers that
can govern `demo-app`'s replica count -- the frozen CPU-only HPA
(`demo-app-hpa`, k8s/hpa/hpa.yaml), the new V2 CPU+memory HPA
(`demo-app-hpa-v2`, k8s/hpa/hpa_v2_cpu_memory.yaml), and PPO (which issues
`kubectl scale` directly via `LiveKubernetesBackend.set_desired_replicas()`,
never through an HPA object at all).

Audited gap this module closes (V2 Phase 2 PART 4): no file in this project
previously implemented a real mutex. Every existing script
(scripts/evaluate_ppo_vs_hpa.py, scripts/train_ppo_live.py,
scripts/run_demo_episode.py) independently calls
`KubectlRunner.hpa_exists()` as a one-time, uncoordinated precondition
check with inconsistent remediation (some refuse-and-exit, one
auto-deletes/restores) -- none of them can detect or prevent a SECOND
controller (in particular, a second simultaneously-applied HPA object)
coexisting with the one they believe is in sole control. This module does
NOT replace or redesign that existing orchestration; it adds one small,
explicit, fail-closed gate any of those call sites (or a new one) can use
before mutating cluster/experiment state.

Two complementary checks, both required to start a controller:
  1. A lock file (`persistent-data/controller/active_controller.json` by
     default) recording which controller this project's OWN tooling last
     started -- catches "I already started X, don't let me start Y from a
     second terminal" even before any HPA object exists.
  2. A live `kubectl get hpa` read (`KubectlRunner.list_hpa_names()`) --
     catches an HPA object that exists for ANY reason (started by this
     project's tooling, left over from a crashed run, or applied by hand)
     regardless of what the lock file claims. Fail-closed: if the two
     disagree, the live cluster read wins for HPA presence (the lock file is
     informational for cross-process coordination; it is never treated as a
     substitute for observing real cluster state before acting).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Protocol

CONTROLLER_NONE = "none"
CONTROLLER_PPO = "ppo"
CONTROLLER_CPU_HPA = "cpu_hpa"
CONTROLLER_CPU_MEMORY_HPA = "cpu_memory_hpa"

ALL_CONTROLLERS = (CONTROLLER_PPO, CONTROLLER_CPU_HPA, CONTROLLER_CPU_MEMORY_HPA)

# The exact HPA object names each HPA-based controller corresponds to (see
# k8s/hpa/hpa.yaml and k8s/hpa/hpa_v2_cpu_memory.yaml's metadata.name) --
# deliberately DIFFERENT names so applying one never silently replaces the
# other via same-name overwrite.
HPA_NAME_BY_CONTROLLER = {
    CONTROLLER_CPU_HPA: "demo-app-hpa",
    CONTROLLER_CPU_MEMORY_HPA: "demo-app-hpa-v2",
}

DEFAULT_LOCK_PATH = Path("persistent-data/controller/active_controller.json")


class ControllerConflictError(RuntimeError):
    """Raised when starting `requested_controller` would conflict with
    either the recorded lock or the live cluster's actual HPA state. Fail
    closed: raised BEFORE any experiment/traffic/training action begins."""


class HpaLister(Protocol):
    def list_hpa_names(self) -> List[str]: ...  # pragma: no cover - structural typing only


def _validate_controller(name: str) -> None:
    if name not in ALL_CONTROLLERS:
        raise ValueError(f"unknown controller {name!r}; must be one of {ALL_CONTROLLERS}")


def read_lock(lock_path: Path = DEFAULT_LOCK_PATH) -> str:
    """Returns the recorded active controller, or CONTROLLER_NONE if the
    lock file is missing or unreadable/corrupt (a corrupt lock file must
    never be treated as a hard failure -- the live `kubectl` read in
    `acquire_controller()` is the authoritative safety check regardless)."""
    try:
        with open(lock_path, encoding="utf-8") as f:
            data = json.load(f)
        controller = data.get("active_controller", CONTROLLER_NONE)
        return controller if controller in ALL_CONTROLLERS else CONTROLLER_NONE
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return CONTROLLER_NONE


def _write_lock(controller: str, lock_path: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"active_controller": controller, "updated_at_utc": datetime.now(timezone.utc).isoformat()}
    tmp_path = lock_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(lock_path)


def clear_lock(lock_path: Path = DEFAULT_LOCK_PATH) -> None:
    """Explicit cleanup path: unconditionally resets the lock to
    CONTROLLER_NONE. Intended for a deliberate operator/switch action (e.g.
    after confirming via `kubectl` that no controller is actually active),
    never called automatically as part of `acquire_controller()`."""
    _write_lock(CONTROLLER_NONE, lock_path)


def detect_conflicting_hpas(requested_controller: str, kubectl: HpaLister) -> List[str]:
    """Returns the names of any HPA objects present in the cluster that are
    NOT the one `requested_controller` itself owns (empty list if none).
    For `CONTROLLER_PPO`, ANY HPA object at all is conflicting (PPO must
    never run while either HPA actively controls the deployment)."""
    _validate_controller(requested_controller)
    active = kubectl.list_hpa_names()
    if requested_controller == CONTROLLER_PPO:
        return list(active)
    own_name = HPA_NAME_BY_CONTROLLER[requested_controller]
    return [name for name in active if name != own_name]


def acquire_controller(
    requested_controller: str, kubectl: HpaLister, lock_path: Path = DEFAULT_LOCK_PATH,
) -> None:
    """Fail-closed gate: call BEFORE starting any experiment/training/
    inference action that will mutate the Deployment's replica count or
    assume sole control of it. Raises `ControllerConflictError` (never
    silently proceeds, never auto-remediates) if:
      - the lock records a DIFFERENT controller as already active, OR
      - a live `kubectl get hpa` read finds a conflicting HPA object
        (see `detect_conflicting_hpas`).
    On success, records `requested_controller` as the active controller.
    Re-acquiring the SAME controller that is already recorded active is a
    no-op success (idempotent -- a controller re-checking its own lock
    mid-run is not a conflict)."""
    _validate_controller(requested_controller)

    current = read_lock(lock_path)
    if current != CONTROLLER_NONE and current != requested_controller:
        raise ControllerConflictError(
            f"cannot start controller {requested_controller!r}: controller {current!r} "
            f"is already recorded active (lock file {lock_path})"
        )

    conflicting = detect_conflicting_hpas(requested_controller, kubectl)
    if conflicting:
        raise ControllerConflictError(
            f"cannot start controller {requested_controller!r}: conflicting HPA object(s) "
            f"present in cluster: {conflicting}"
        )

    _write_lock(requested_controller, lock_path)


def release_controller(controller: str, lock_path: Path = DEFAULT_LOCK_PATH) -> None:
    """Explicit release: clears the lock ONLY if it currently records
    `controller` as active -- never clears a different controller's lock
    (a stale/mismatched release attempt is a silent no-op, not an error,
    since the caller releasing is not the one holding the lock and has no
    authority to clear it; use `clear_lock()` for a deliberate manual
    override)."""
    _validate_controller(controller)
    if read_lock(lock_path) == controller:
        clear_lock(lock_path)


@dataclass(frozen=True)
class ControllerStatus:
    locked_controller: str
    active_hpa_names: List[str]


def current_status(kubectl: HpaLister, lock_path: Path = DEFAULT_LOCK_PATH) -> ControllerStatus:
    """Read-only snapshot of both signals (lock file + live cluster HPA
    state) -- useful for a UI/CLI status display without attempting to
    acquire anything."""
    return ControllerStatus(locked_controller=read_lock(lock_path), active_hpa_names=list(kubectl.list_hpa_names()))
