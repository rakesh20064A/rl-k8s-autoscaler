#!/usr/bin/env python3
"""scripts/run_demo_episode.py

Isolated DEMO-ONLY execution adapter for the Streamlit Experiment page.

This is NOT part of the scientific pipeline: it writes no training
checkpoint, no training/evaluation ledger row, and no EVALUATION_ROOT
evidence file. It reuses the exact, already-tested live mechanics from
`scripts/evaluate_ppo_vs_hpa.py` (`run_ppo_episode` / `run_hpa_episode`)
against one of the six pre-registered, non-frozen TRAINING_PROFILES
(constant/ramp/burst/step/periodic/mixed) -- never the frozen final
holdout, which `traffic_orchestrator.resolve_profile_metadata()` rejects
by construction (`rl_live_common.assert_training_eligible`).

Usage:
    .venv/bin/python scripts/run_demo_episode.py \
        --controller {hpa,ppo} --profile {constant,ramp,burst,step,periodic,mixed} \
        --status-file /path/to/status.json [--model models/ppo/ppo_model.zip]

Writes a small JSON status document to --status-file at each phase
transition (running -> completed/error) so a caller (the demo UI) can
poll progress without parsing stdout.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autoscaler.src.rl_env.config import load_env_config_from_yaml  # noqa: E402
from autoscaler.src.rl_env.live_backend import LiveKubernetesBackend  # noqa: E402
from scripts import evaluate_ppo_vs_hpa as eval_mod  # noqa: E402
from scripts import traffic_orchestrator as to  # noqa: E402
from scripts.evaluate_ppo_vs_hpa import run_hpa_episode, run_ppo_episode  # noqa: E402

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ppo" / "ppo_model.zip"

# Demo-only traffic profiles: NEVER added to rl_live_common.TRAINING_PROFILES
# or any scientific firewall/allowlist. Resolved directly (bypassing
# `to.resolve_profile_metadata()`'s training-eligibility check), mirroring
# how `evaluate_ppo_vs_hpa.resolve_holdout_profile_metadata()` already
# builds a ProfileMetadata outside that same firewall for a different
# reason. No governing hash is pinned for these -- they are not scientific
# evidence, only a fast UI smoke aid.
DEMO_ONLY_PROFILES = {
    "demo_smoke_test": "configs/traffic_profiles/demo_smoke_test.yaml",
}
ALL_SELECTABLE_PROFILES = list(to.common.TRAINING_PROFILES) + list(DEMO_ONLY_PROFILES)


def _resolve_profile_metadata(profile_name: str) -> "to.ProfileMetadata":
    if profile_name in DEMO_ONLY_PROFILES:
        path = PROJECT_ROOT / DEMO_ONLY_PROFILES[profile_name]
        if not path.exists():
            raise FileNotFoundError(f"demo-only profile file missing: {path}")
        profile = to.parse_traffic_profile(path)
        return to.ProfileMetadata(
            name=profile_name, path=path, sha256=to.common.sha256_file(path),
            seed=profile.seed, duration_seconds=profile.total_duration, n_phases=len(profile.phases),
        )
    return to.resolve_profile_metadata(profile_name)


def _write_status(status_file: Optional[Path], payload: Dict[str, Any]) -> None:
    if status_file is None:
        return
    payload = {**payload, "updated_at": time.time()}
    tmp = status_file.with_suffix(status_file.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, default=str))
    tmp.replace(status_file)


def _log_stage(status_file: Optional[Path], stage: str, controller: str, profile_name: str, **extra: Any) -> None:
    """Timestamped, unbuffered stdout line (visible via the UI's log tail)
    PLUS a status-file update carrying the stage name (visible via the
    UI's status panel), so a stuck run's wait point is visible from the
    Experiment page, not just a terminal."""
    print(f"[{time.strftime('%H:%M:%S')}] STAGE={stage}", flush=True)
    _write_status(status_file, {"status": "running", "stage": stage, "controller": controller, "profile": profile_name, **extra})


class _BoundedRuntimeWatchdog:
    """Requirement: a demo episode must never hang indefinitely. Tracks the
    most recent stage reached; if total wall-clock exceeds `ceiling_s`
    (computed from the profile's own known, bounded cost centers -- NOT an
    arbitrarily enlarged timeout), it writes an ERROR status naming the
    stuck stage and force-kills whatever real child resources it was told
    about (the traffic-generator subprocess, the Docker relay container),
    so the main thread's blocked call fails/returns promptly instead of
    hanging forever."""

    def __init__(self, status_file: Optional[Path], controller: str, profile_name: str, ceiling_s: float):
        self._status_file = status_file
        self._controller = controller
        self._profile_name = profile_name
        self._ceiling_s = ceiling_s
        self._stage = "STARTING"
        self._generator_proc: Optional[subprocess.Popen] = None
        self._relay_container: Optional[str] = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def note_stage(self, stage: str) -> None:
        self._stage = stage

    def note_generator(self, proc: subprocess.Popen) -> None:
        self._generator_proc = proc

    def note_relay(self, container_name: str) -> None:
        self._relay_container = container_name

    def start(self) -> None:
        self._start_monotonic = time.monotonic()
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop_event.wait(5):
            if time.monotonic() - self._start_monotonic <= self._ceiling_s:
                continue
            stuck_stage = self._stage
            print(f"[{time.strftime('%H:%M:%S')}] STAGE=ERROR bounded runtime "
                  f"({self._ceiling_s:.0f}s) exceeded, stuck at {stuck_stage}", flush=True)
            _write_status(self._status_file, {
                "status": "error", "stage": f"TIMEOUT_AT_{stuck_stage}", "controller": self._controller,
                "profile": self._profile_name,
                "error": f"exceeded bounded runtime ({self._ceiling_s:.0f}s) while stuck at stage {stuck_stage}",
            })
            if self._generator_proc is not None and self._generator_proc.poll() is None:
                self._generator_proc.kill()
            if self._relay_container:
                subprocess.run(["docker", "rm", "-f", self._relay_container], capture_output=True, timeout=10)
            return


def _bounded_runtime_ceiling_s(profile_duration_s: float) -> float:
    """Sum of this pipeline's own documented, already-bounded cost centers
    (never an arbitrary inflation): ~60s pre-episode stabilization
    (live_backend.py), ~30s generic route/relay/generator setup margin, the
    profile's own real duration, ~30s drain, plus a 60s safety margin."""
    return 60.0 + 30.0 + profile_duration_s + 30.0 + 60.0


@contextlib.contextmanager
def _instrumented_traffic_lifecycle(status_file: Optional[Path], controller: str, profile_name: str, watchdog: "_BoundedRuntimeWatchdog"):
    """Monkeypatches (process-local, restored on exit; never touches the
    on-disk source of traffic_orchestrator.py/evaluate_ppo_vs_hpa.py) the
    traffic lifecycle calls that `run_hpa_episode`/`run_ppo_episode`
    (imported, unmodified, from the frozen scientific harness) make
    internally, purely to emit ROUTE_READY/TRAFFIC_STARTED/TRAFFIC_STOPPED/
    DRAINING/CLEANUP stage markers around them. Adds observability only --
    every wrapped call still does exactly what it always did, with the
    same arguments, same return value, same exceptions."""
    log = lambda stage, **extra: _log_stage(status_file, stage, controller, profile_name, **extra)  # noqa: E731

    orig_establish_route = to.V5TransportManager.establish_route
    orig_teardown_route = to.V5TransportManager.teardown_route
    orig_start_generator = to.start_generator
    orig_stop_generator = to.stop_generator
    orig_drain = to.drain_in_flight_requests

    def establish_route(self, run_id):
        t0 = time.monotonic()
        route = orig_establish_route(self, run_id)
        watchdog.note_stage("ROUTE_READY")
        watchdog.note_relay(route.container_name)
        log("ROUTE_READY", elapsed_s=round(time.monotonic() - t0, 1))
        return route

    def start_generator(*args, **kwargs):
        t0 = time.monotonic()
        handle = orig_start_generator(*args, **kwargs)
        watchdog.note_stage("TRAFFIC_STARTED")
        watchdog.note_generator(handle.process)
        log("TRAFFIC_STARTED", elapsed_s=round(time.monotonic() - t0, 1))
        return handle

    def stop_generator(*args, **kwargs):
        watchdog.note_stage("TRAFFIC_STOPPED")
        log("TRAFFIC_STOPPED")
        return orig_stop_generator(*args, **kwargs)

    def drain_in_flight_requests(*args, **kwargs):
        t0 = time.monotonic()
        watchdog.note_stage("DRAINING")
        log("DRAINING")
        result = orig_drain(*args, **kwargs)
        log("DRAIN_DONE", elapsed_s=round(time.monotonic() - t0, 1))
        return result

    def teardown_route(self, route):
        watchdog.note_stage("CLEANUP")
        log("CLEANUP")
        return orig_teardown_route(self, route)

    to.V5TransportManager.establish_route = establish_route
    to.V5TransportManager.teardown_route = teardown_route
    to.start_generator = start_generator
    to.stop_generator = stop_generator
    to.drain_in_flight_requests = drain_in_flight_requests
    try:
        yield
    finally:
        to.V5TransportManager.establish_route = orig_establish_route
        to.V5TransportManager.teardown_route = orig_teardown_route
        to.start_generator = orig_start_generator
        to.stop_generator = orig_stop_generator
        to.drain_in_flight_requests = orig_drain


@contextlib.contextmanager
def _instrumented_sample_collection(
    status_file: Optional[Path], controller: str, profile_name: str, profile_meta: "to.ProfileMetadata",
):
    """Monkeypatches (process-local, restored on exit; the on-disk source
    of evaluate_ppo_vs_hpa.py is never touched) that module's `_snapshot_dict`
    -- the one function `run_hpa_episode`/`run_ppo_episode` call every time
    they take a real decision-step measurement. Each call's result is
    forwarded UNCHANGED to its real caller (zero effect on scientific
    control logic, timing, or return values) and ALSO appended to a
    cumulative list written to the status file, so a real sample becomes
    visible to the polling UI within one 3s fragment tick of being
    produced -- never interpolated, never backfilled, exactly as many
    points as the real backend/evaluation loop has actually completed."""
    collected: list = []
    orig_snapshot_dict = eval_mod._snapshot_dict

    def snapshot_dict(*args, **kwargs):
        sample = orig_snapshot_dict(*args, **kwargs)
        collected.append(sample)
        _write_status(status_file, {
            "status": "running", "stage": "EPISODE_RUNNING", "controller": controller, "profile": profile_name,
            "workload_hash": profile_meta.sha256, "workload_seed": profile_meta.seed,
            "workload_duration_seconds": profile_meta.duration_seconds,
            "samples": list(collected),
        })
        return sample

    eval_mod._snapshot_dict = snapshot_dict
    try:
        yield collected
    finally:
        eval_mod._snapshot_dict = orig_snapshot_dict


def run_demo_hpa(profile_name: str, status_file: Optional[Path]) -> Dict[str, Any]:
    _log_stage(status_file, "STARTING", "HPA", profile_name)
    backend = LiveKubernetesBackend()
    if not backend._kubectl.hpa_exists():
        # Demo infra is expected to already have the canonical HPA applied;
        # restore it rather than fail the demo over a missing-but-trivial
        # precondition. Never touches CPU target/min/max -- applies the
        # project's own canonical manifest, unmodified.
        backend._kubectl.restore_hpa()

    env_config = load_env_config_from_yaml()
    profile_meta = _resolve_profile_metadata(profile_name)
    watchdog = _BoundedRuntimeWatchdog(status_file, "HPA", profile_name, _bounded_runtime_ceiling_s(profile_meta.duration_seconds))
    watchdog.start()
    try:
        with _instrumented_traffic_lifecycle(status_file, "HPA", profile_name, watchdog), \
             _instrumented_sample_collection(status_file, "HPA", profile_name, profile_meta):
            watchdog.note_stage("EPISODE_RUNNING")
            _log_stage(status_file, "EPISODE_RUNNING", "HPA", profile_name,
                       note="includes a fixed ~60s pre-episode stabilization wait (docs/rl_mdp_spec.md sec.20), before the workload itself runs",
                       workload_hash=profile_meta.sha256, workload_seed=profile_meta.seed,
                       workload_duration_seconds=profile_meta.duration_seconds)
            result = run_hpa_episode(0, 0, env_config, profile_meta)
    finally:
        watchdog.stop()
    # `samples` (per-decision-step replicas/cpu/latency) is KEPT here --
    # it's what lets the demo UI draw a real (not fabricated) chart for
    # this run once it completes. `decisions` (PPO's own per-step action
    # log) is dropped: not needed for the UI's replica/latency charts.
    return {k: v for k, v in result.items() if k != "decisions"}


def run_demo_ppo(profile_name: str, model_path: Path, status_file: Optional[Path]) -> Dict[str, Any]:
    from stable_baselines3 import PPO

    _log_stage(status_file, "STARTING", "PPO", profile_name)
    backend = LiveKubernetesBackend()
    deleted_hpa = False
    if backend._kubectl.hpa_exists():
        backend._kubectl.delete_hpa()
        deleted_hpa = True

    try:
        env_config = load_env_config_from_yaml()
        profile_meta = _resolve_profile_metadata(profile_name)
        model = PPO.load(model_path)  # inference only -- no .learn() call anywhere in this script
        watchdog = _BoundedRuntimeWatchdog(status_file, "PPO", profile_name, _bounded_runtime_ceiling_s(profile_meta.duration_seconds))
        watchdog.start()
        try:
            with _instrumented_traffic_lifecycle(status_file, "PPO", profile_name, watchdog), \
                 _instrumented_sample_collection(status_file, "PPO", profile_name, profile_meta):
                watchdog.note_stage("EPISODE_RUNNING")
                _log_stage(status_file, "EPISODE_RUNNING", "PPO", profile_name,
                           note="includes a fixed ~60s pre-episode stabilization wait (docs/rl_mdp_spec.md sec.20), before the workload itself runs",
                           workload_hash=profile_meta.sha256, workload_seed=profile_meta.seed,
                           workload_duration_seconds=profile_meta.duration_seconds)
                result = run_ppo_episode(0, 0, model, env_config, profile_meta)
        finally:
            watchdog.stop()
        return {k: v for k, v in result.items() if k != "decisions"}
    finally:
        # Always restore the canonical HPA (260% / min1 / max8, unmodified
        # manifest) once the PPO demo episode ends, succeeds or fails --
        # required so the cluster is immediately ready for another demo
        # run and so HPA's canonical presence is never left disturbed.
        if deleted_hpa:
            backend._kubectl.restore_hpa()
            _log_stage(status_file, "HPA_RESTORED", "PPO", profile_name)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", required=True, choices=["hpa", "ppo"])
    parser.add_argument("--profile", required=True, choices=ALL_SELECTABLE_PROFILES)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--status-file", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        if args.controller == "hpa":
            summary = run_demo_hpa(args.profile, args.status_file)
        else:
            summary = run_demo_ppo(args.profile, args.model, args.status_file)
    except Exception as exc:  # noqa: BLE001 -- must surface to the UI, never crash silently
        print(f"[{time.strftime('%H:%M:%S')}] STAGE=ERROR: {exc}", file=sys.stderr, flush=True)
        _write_status(args.status_file, {"status": "error", "stage": "ERROR", "error": str(exc)})
        return 1

    _log_stage(args.status_file, "COMPLETED", args.controller.upper(), args.profile)
    _write_status(args.status_file, {"status": "completed", "stage": "COMPLETED", "summary": summary})
    print(json.dumps(summary, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
