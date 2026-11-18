#!/usr/bin/env python3
"""scripts/run_demo_episode_v2.py

V2 Phase 5E: isolated DEMO-ONLY execution adapter for the Streamlit UI's
V2 Live view. Mirrors `scripts/run_demo_episode.py`'s status-file
streaming contract for V1 (atomic JSON write per stage transition), but
drives ONE V2 arm (`ppo_v2` or `hpa_v2_cpu_memory`) over ONE SHORT
`configs/traffic_profiles/v2_demo/` profile, using
`scripts/eval_v2_episodes.py`'s already-tested unified recorders
(imported, unmodified logic) -- this file adds only streaming, CLI, and
the training/evaluation protection check below.

NOT part of the scientific pipeline: writes no training checkpoint, no
training/evaluation ledger row, and no `persistent-data/evaluation_v2/`
campaign folder. Every sample this script's runs produce must be labelled
"DEMO -- not evaluation evidence" by any UI view that displays it.

TRAINING/EVALUATION PROTECTION -- refuses to start (clear message, NO
cluster mutation of any kind) if ANY of:
  - the controller lock is not "none" (`controller_exclusivity.read_lock()`)
  - a `train_ppo_final_v2.py` process is already running (`pgrep -f`)
  - an `evaluate_v2_campaign.py` process is already running (`pgrep -f`)
  - any HPA other than `demo-app-hpa-v2` is present

Usage:
    .venv/bin/python scripts/run_demo_episode_v2.py \
        --arm {ppo_v2,hpa_v2_cpu_memory} \
        --profile {demo_cpu_short,demo_memory_short,demo_mixed_short} \
        --samples-jsonl /path/to/samples.jsonl \
        --status-file /path/to/status.json \
        [--model path/to/ppo_model.zip] [--epsilon 0.05]
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autoscaler.src.rl_env.config_v2 import load_env_config_v2_from_yaml  # noqa: E402
from autoscaler.src.rl_env.controller_exclusivity import CONTROLLER_NONE, V2_HPA_NAME, clear_lock, read_lock  # noqa: E402
from collectors.metrics_collector import PrometheusClient  # noqa: E402
from loadgen.traffic_generator import parse_traffic_profile  # noqa: E402
import scripts.traffic_orchestrator as to  # noqa: E402
from scripts.eval_v2_episodes import (  # noqa: E402
    CpuOnlyArmNotYetSupportedError,
    EpisodeSample,
    SafetyTerminationError,
    ToolingFailureError,
    run_hpa_v2_cpu_memory_eval_episode,
    run_ppo_v2_eval_episode,
)
from scripts.evaluate_v2_campaign import restore_canonical_idle_state  # noqa: E402
from scripts.v2_pilot_safety import PROMETHEUS_URL  # noqa: E402
# V2 Phase 9E / the same CORR-63/CORR-64 fix scripts/evaluate_v2_campaign.py
# already made for its own live runs: `demo-app-hpa-v2` present is the
# canonical IDLE state, never a conflict to refuse on, and PPO (a pure
# Python controller) needs it deleted -- not treated as blocking -- before
# it can acquire the controller. `restore_canonical_idle_state` is REUSED
# (imported) directly from evaluate_v2_campaign.py rather than
# reimplemented, so there is exactly one source of truth for "what the
# canonical idle cluster state is and how to reach it" for every live V2
# entry point. The one piece evaluate_v2_campaign.py does not expose as a
# standalone function -- deleting the idle HPA immediately before a ppo_v2
# run starts -- is factored out into scripts/v2_live_cluster_state.py (a
# new, minimal shared module) instead of being copy-pasted here.
from scripts.generate_demo_profiles import (  # noqa: E402
    ALL_PROFILE_NAMES,
    LEGACY_DEMO_PROFILE_NAMES,
    LOAD_LABELS,
    PATTERN_LABELS,
    RESOURCE_LABELS,
    parse_profile_name,
)
from scripts.v2_live_cluster_state import remove_hpa_before_ppo_start  # noqa: E402

DEMO_PROFILE_DIR = PROJECT_ROOT / "configs" / "traffic_profiles" / "v2_demo"
# The 3 original short demo profiles (kept, unmodified, for backward
# compatibility) plus the 36-profile pattern x resource x load catalog
# (scripts/generate_demo_profiles.py) -- additive, never replacing the
# legacy entries. Every name here resolves ONLY inside DEMO_PROFILE_DIR,
# never a training/eval/stress profile directory.
# Task sec.7: a single, TEST-ONLY, short (120s) smoke profile, additive to
# (never replacing) the legacy/36-catalog allowlist above -- lets the
# mandatory real end-to-end live verification launch two real demo runs
# without waiting out one of the ~8-minute catalog profiles. Never
# surfaced in the UI's own Pattern/Resource/Load selects (those only ever
# construct a 36-catalog slug via scripts.generate_demo_profiles.
# profile_slug()) -- reachable only by passing this name directly to
# `RealExperimentControllerV2.start()`/this script's own `--profile`.
DEMO_SMOKE_TEST_PROFILE_NAMES = ("demo_smoke_bursts_cpu",)
DEMO_PROFILE_ALLOWLIST = LEGACY_DEMO_PROFILE_NAMES + ALL_PROFILE_NAMES + DEMO_SMOKE_TEST_PROFILE_NAMES
DEMO_ARMS = ("ppo_v2", "hpa_v2_cpu_memory")

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ppo_v2" / "ppo_model_v2_min2.zip"

# Task sec.1/sec.2: the demo UI's own traffic-provenance ledger -- kept
# SEPARATE from `to.TRAFFIC_PROVENANCE_LEDGER_PATH` (the live PPO TRAINING
# ledger) and from `scripts.evaluate_v2_campaign.EVAL_OUTPUT_ROOT`'s own
# evaluation ledger, exactly the same reasoning
# `evaluate_v2_campaign.build_live_context()` already documents for its
# own ledger_path override: a DEMO run's traffic provenance must never be
# mixed into the scientific training/evaluation record.
DEMO_TRAFFIC_LEDGER_PATH = PROJECT_ROOT / "persistent-data" / "demo_ui" / "traffic_provenance_ledger_demo_v2.jsonl"

# Task sec.2: the exact request-rate PromQL `autoscaler.src.rl_env.
# live_backend._query_dict()` already uses for every live episode's own
# `request_rate_rps` observation (namespace=autoscaler-app, route=/work --
# `to.APP_NS`/`to.WORKLOAD_ROUTE`, never re-typed by hand) -- the traffic
# guard below queries the SAME metric the episode itself will be reading
# every decision, so "confirmed" means exactly what the episode means by
# "the app is receiving traffic", not a different proxy signal.
DEMO_TRAFFIC_REQUEST_RATE_PROMQL = f'sum(rate(demo_app_requests_total{{namespace="{to.APP_NS}",route="{to.WORKLOAD_ROUTE}"}}[1m]))'

# Task sec.2: "confirm within 60-90s that requests are actually arriving."
# 90s (the upper end of that window) with a 5s poll interval -- generous
# enough that the generator's own warm-up (DNS/connection setup, the
# first few requests actually landing and being scraped by Prometheus's
# own ~15s-30s scrape interval) is never mistaken for a genuine failure,
# while still ending well within the spec's own ceiling.
TRAFFIC_CONFIRM_TIMEOUT_SECONDS = 90.0
TRAFFIC_CONFIRM_POLL_INTERVAL_SECONDS = 5.0
# A conservative floor, not "any traffic at all": every catalog/smoke
# profile's baseline rate is several req/s, so requiring >= 0.5 req/s
# confirmed via a real rate() query comfortably distinguishes "traffic is
# genuinely flowing" from float noise around zero.
TRAFFIC_CONFIRM_MIN_RPS = 0.5


class DemoProfileFirewallViolation(RuntimeError):
    pass


class DemoRefusedError(RuntimeError):
    """Raised when a training/evaluation-protection precondition fails --
    never a cluster mutation of any kind up to this point."""


class DemoStopRequested(RuntimeError):
    """Raised from the SIGTERM handler installed in `main()` so an
    explicit STOP request (`demo_ui.live_execution_adapter.
    RealExperimentControllerV2.stop()` sends SIGTERM to this whole
    process's process group) goes through the EXACT SAME `finally`
    cleanup path as a normal completion or an unexpected exception --
    never a bare process kill with no cluster cleanup left behind."""


def _raise_stop_requested(signum: int, frame: Any) -> None:  # noqa: ARG001
    raise DemoStopRequested(f"stop requested (signal {signum})")


def _restore_idle_state_safely() -> Dict[str, Any]:
    """Unconditional cleanup -- called from a `finally` block, so it must
    itself never raise and mask whatever triggered it (success, an
    unexpected exception, or `DemoStopRequested`). Mirrors
    `evaluate_v2_campaign._safe_cleanup()`'s own "a cleanup failure must
    never crash the caller" rule."""
    result: Dict[str, Any] = {}
    try:
        result.update(restore_canonical_idle_state())
    except Exception as exc:  # noqa: BLE001
        result["cleanup_error"] = str(exc)
    try:
        clear_lock()
    except Exception as exc:  # noqa: BLE001
        result.setdefault("cleanup_error", str(exc))
    return result


def resolve_demo_profile_path(profile_name: str) -> Path:
    if profile_name not in DEMO_PROFILE_ALLOWLIST:
        raise DemoProfileFirewallViolation(f"{profile_name!r} is not a v2_demo profile; must be one of {DEMO_PROFILE_ALLOWLIST}")
    path = DEMO_PROFILE_DIR / f"{profile_name}.yaml"
    if not path.exists():
        raise DemoProfileFirewallViolation(f"demo profile file missing on disk: {path}")
    return path


def resolve_demo_profile_metadata(profile_name: str) -> "to.ProfileMetadata":
    """Task sec.1: the DEMO equivalent of `scripts.evaluate_v2_campaign.
    resolve_eval_profile_metadata()` / `scripts.v2_traffic_orchestrator.
    resolve_v2_profile_metadata()` -- fails closed via
    `resolve_demo_profile_path()` (the ONE existing firewall for
    `DEMO_PROFILE_DIR`, unchanged), then computes the sha256/seed/duration/
    phase-count `to.ProfileMetadata` the generic `to.TrafficOrchestrator`
    machinery (and this run's own workload-identity fields, see
    `run_demo_episode_v2()`) need. Structurally unable to resolve a
    training or held-out-evaluation profile: different allowlist,
    different directory, no import of either of those registries."""
    path = resolve_demo_profile_path(profile_name)
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    profile = parse_traffic_profile(path)
    return to.ProfileMetadata(
        name=profile_name, path=path, sha256=actual_sha256, seed=profile.seed,
        duration_seconds=profile.total_duration, n_phases=len(profile.phases),
    )


class V2DemoTrafficOrchestrator(to.TrafficOrchestrator):
    """Task sec.1 -- THE root-cause fix: before this phase,
    `run_demo_episode_v2.py` never instantiated ANY traffic orchestrator
    at all. It logged "Traffic started: ..." as a purely cosmetic string
    and then ran `eval_v2_episodes.py`'s per-arm episode loop, which is
    observe-only (it reads Prometheus/pod-health/the Deployment every
    decision interval but never starts a generator process) -- so every
    decision of every demo run showed 0 req/s, because no traffic
    generator, relay route, or anything else was ever started. This class
    is the demo UI's own orchestrator, built with the EXACT SAME
    architecture `scripts.evaluate_v2_campaign.V2EvalTrafficOrchestrator`
    and `scripts.v2_traffic_orchestrator.V2TrafficOrchestrator` already
    use for the (working) evaluation/training traffic paths: every
    generic piece -- the V5 relay route lifecycle, generator-process
    lifecycle, drain, provenance -- is INHERITED UNCHANGED from
    `to.TrafficOrchestrator`; only `start_episode_traffic` is overridden,
    to resolve against `DEMO_PROFILE_ALLOWLIST`/`DEMO_PROFILE_DIR` via
    `resolve_demo_profile_metadata()` instead of any training/eval
    registry. Reuses the already-tested generic machinery rather than
    inventing a new traffic path."""

    def start_episode_traffic(self, profile_name: str, episode_id: str) -> "to.GeneratorHandle":
        profile = resolve_demo_profile_metadata(profile_name)
        run_id = to.resolve_next_available_run_id(episode_id, self.traffic_base_dir)
        route = self.ensure_route_established(run_id)
        if not to.verify_route_alive(route, self.http_get):
            raise to.RouteEstablishmentFailure(f"route {route.target_url} not alive before episode {episode_id}")
        handle = to.start_generator(profile, route, run_id, popen_factory=self.popen_factory, clock=self.clock)
        if not to.verify_generator_started(handle, sleep_fn=self.sleep_fn):
            to.stop_generator(handle)
            to.record_traffic_provenance(
                handle, episode_id=episode_id, session_index=self.session_index,
                exit_code=handle.process.poll(), reason="failed_to_start", ledger_path=self.ledger_path,
            )
            raise to.TrafficGeneratorFailure(f"generator failed to start for demo profile {profile_name!r}, episode {episode_id}")
        return handle


def _demo_http_get(url: str, timeout: float) -> "tuple[bool, str]":
    """Real HTTP GET for the relay route's own `/health` liveness check --
    identical implementation to `scripts.evaluate_v2_campaign.
    build_live_context()`'s own `_http_get` (itself reused verbatim from
    `scripts/train_ppo_final_v2.py`/`scripts/evaluate_ppo_vs_hpa.py`),
    reused here rather than reinvented."""
    import httpx

    try:
        resp = httpx.get(url, timeout=timeout)
        return resp.status_code == 200, f"HTTP {resp.status_code}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def build_demo_traffic_orchestrator() -> V2DemoTrafficOrchestrator:
    """Default `orchestrator_factory` for `run_demo_episode_v2()` --
    injectable so a unit test can substitute a fully offline fake (no
    subprocess, no Docker, no real network), exactly as
    `scripts.evaluate_v2_campaign._FakeOrchestrator` does for the
    evaluation campaign."""
    orchestrator = V2DemoTrafficOrchestrator(ledger_path=DEMO_TRAFFIC_LEDGER_PATH)
    orchestrator.http_get = _demo_http_get
    return orchestrator


def confirm_traffic_arriving(
    handle: "to.GeneratorHandle",
    prom_query_fn: Callable[[], "tuple[Optional[float], Optional[str]]"],
    timeout_s: float = TRAFFIC_CONFIRM_TIMEOUT_SECONDS,
    poll_interval_s: float = TRAFFIC_CONFIRM_POLL_INTERVAL_SECONDS,
    min_rps: float = TRAFFIC_CONFIRM_MIN_RPS,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_fn: Callable[[], float] = time.monotonic,
) -> Dict[str, Any]:
    """Task sec.2's traffic guard: confirms, after `start_episode_traffic()`
    has returned a handle, that requests are ACTUALLY arriving at the
    app -- the real request-rate Prometheus metric reads >= `min_rps` AND
    the generator process is still alive -- within `timeout_s`, before the
    run is allowed to proceed into the episode's decision loop. Checks the
    generator's own liveness FIRST on every poll (fails immediately,
    rather than waiting out the full timeout, if it already exited).
    Never raises -- always returns `{"ok": bool, ...}` so the caller
    decides how to fail the run (never lets a run proceed blind with zero
    confirmed traffic for its full duration)."""
    deadline = clock_fn() + timeout_s
    last_value: Optional[float] = None
    while True:
        exit_code = handle.process.poll()
        if exit_code is not None:
            return {
                "ok": False,
                "reason": f"traffic generator process exited early (code={exit_code}) before any request rate was confirmed",
            }
        value, err = prom_query_fn()
        if err is None and value is not None:
            last_value = value
            if value >= min_rps:
                return {"ok": True, "observed_rps": value}
        if clock_fn() >= deadline:
            return {
                "ok": False,
                "reason": (
                    f"no confirmed request rate >= {min_rps} req/s within {timeout_s:.0f}s "
                    f"(last observed: {last_value})"
                ),
            }
        sleep_fn(poll_interval_s)


def _pgrep(pattern: str, run: Callable[..., Any] = subprocess.run) -> List[str]:
    try:
        proc = run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=5)
    except Exception:  # noqa: BLE001
        return []
    return [p for p in proc.stdout.split() if p]


def check_training_protection(
    read_lock_fn: Callable[[], str] = read_lock,
    pgrep_fn: Callable[[str], List[str]] = _pgrep,
    list_hpa_names_fn: Optional[Callable[[], List[str]]] = None,
) -> Dict[str, Any]:
    """Pure, injectable precondition check -- NEVER mutates anything.
    `list_hpa_names_fn` has no default cluster dependency: a caller must
    inject it explicitly (production: a throwaway `LiveKubernetesBackend`'s
    `_kubectl.list_hpa_names`; tests: a fake), so this function itself has
    zero hidden live-cluster coupling and is fully unit-testable."""
    reasons: List[str] = []

    lock = read_lock_fn()
    if lock != CONTROLLER_NONE:
        reasons.append(f"controller lock is {lock!r}, not 'none'")

    trainer_pids = pgrep_fn("train_ppo_final_v2.py")
    if trainer_pids:
        reasons.append(f"train_ppo_final_v2.py is already running (pid(s): {trainer_pids})")

    eval_pids = pgrep_fn("evaluate_v2_campaign.py")
    if eval_pids:
        reasons.append(f"evaluate_v2_campaign.py is already running (pid(s): {eval_pids})")

    if list_hpa_names_fn is not None:
        hpa_names = list_hpa_names_fn()
        unexpected = [h for h in hpa_names if h != V2_HPA_NAME]
        if unexpected:
            reasons.append(f"unexpected HPA object(s) present: {unexpected}")

    return {"ok": len(reasons) == 0, "reasons": reasons}


def _real_list_hpa_names() -> List[str]:
    from autoscaler.src.rl_env.controller_exclusivity import V2_HPA_MANIFEST_PATH
    from autoscaler.src.rl_env.live_backend import LiveKubernetesBackend

    backend = LiveKubernetesBackend(hpa_name=V2_HPA_NAME, hpa_manifest_path=V2_HPA_MANIFEST_PATH, min_replicas=2, initial_replicas=2)
    return backend._kubectl.list_hpa_names()


def _log(message: str) -> None:
    """Task sec.1 (LIVE LOG, runner side): one timestamped, human-readable
    progress line to stdout, flushed immediately. `main()` runs this
    script with `-u` (unbuffered) AND `live_execution_adapter.
    RealExperimentControllerV2.start()` sets `PYTHONUNBUFFERED=1` on the
    child's env, but `flush=True` here is belt-and-suspenders -- it costs
    nothing and guarantees every line is visible to a tailing reader
    (demo_ui's "Recent log output" panel) the moment it is written, never
    batched until process exit. stdout (not stderr) is used throughout so
    every progress line lands in the SAME stream `start()` redirects to
    this run's `log_path` (`stderr=subprocess.STDOUT`) -- one ordered log,
    not two interleaved files."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _fmt_num(value: Any, spec: str = "{:.1f}") -> str:
    """Formats a possibly-`None`/non-numeric telemetry value for a log
    line -- never raises, falls back to `"?"` rather than crashing a demo
    run over a cosmetic logging format error."""
    try:
        return spec.format(value)
    except (TypeError, ValueError):
        return "?"


def _traffic_description(profile_name: str) -> str:
    """"Pattern / Resource / Load" (Title Case, task sec.1's exact
    separator) for a 36-catalog profile name; falls back to the raw
    profile name for a legacy `demo_*_short` profile, which has no such
    decomposition."""
    parsed = parse_profile_name(profile_name)
    if parsed is None:
        return profile_name
    pattern, resource, load = parsed
    return f"{PATTERN_LABELS[pattern]} / {RESOURCE_LABELS[resource]} / {LOAD_LABELS[load]}"


def _log_decision(
    sample_dict: Dict[str, Any], step_number: int, n_steps_total: Optional[int],
    before_replicas: Optional[int] = None,
) -> None:
    """One progress line per decision/step (task sec.1), branching on
    which arm produced the sample: `ppo_decision` is populated only for
    `ppo_v2` (a `DecisionRecordV2`, see `decision_record_v2.py`),
    `hpa_status` only for `hpa_v2_cpu_memory` -- never both, never
    neither, for a real sample from either recorder in `eval_v2_episodes.
    py`, so this is a reliable discriminator without needing `arm_id`.

    Task sec.4 (before->after replica log fix): `decision_record_v2.py`'s
    own `current_replicas` field is itself populated (inside a FROZEN
    training-contract helper, out of this phase's scope to edit) from
    `info["deployment_current_replicas_after"]` --
    the SAME post-step value `desired_replicas` also comes from -- so
    trusting `decision.get("current_replicas")` here always printed the
    post-action count twice (e.g. "3->3"), never the real pre-decision
    count. The fix lives entirely on THIS side: `run_demo_episode_v2()`'s
    `on_sample()` tracks each decision's own resulting replica count
    (`sample_dict["desired_replicas"]`, a real post-action value) and
    passes the PREVIOUS decision's resulting count in as `before_replicas`
    -- replicas only ever change via a decision, so "current going into
    decision N" is exactly "desired coming out of decision N-1" (or the
    episode's initial replica count, for decision 0)."""
    total_suffix = f"/{n_steps_total}" if n_steps_total else ""
    raw_obs = sample_dict.get("raw_observation") or {}
    cpu = raw_obs.get("cpu_per_replica")
    mem = raw_obs.get("memory_per_replica")
    p95 = sample_dict.get("p95_latency_ms")
    rps = sample_dict.get("request_rate_rps")
    decision = sample_dict.get("ppo_decision")
    if decision:
        action = decision.get("effective_action_name", "?")
        cur = before_replicas if before_replicas is not None else decision.get("current_replicas", "?")
        des = sample_dict.get("desired_replicas", decision.get("desired_replicas", "?"))
        tie = decision.get("tie_rule_applied") or "none"
        _log(
            f"Decision {step_number}{total_suffix}: {action} | replicas {cur}→{des} | "
            f"{_fmt_num(rps, '{:.0f}')} req/s | p95 {_fmt_num(p95, '{:.0f}')} ms | "
            f"CPU/replica {_fmt_num(cpu)} | mem/replica {_fmt_num(mem)} | tie rule: {tie}"
        )
    else:
        desired = sample_dict.get("desired_replicas", "?")
        current = sample_dict.get("ready_replicas", "?")
        cpu_pct = cpu * 100 if isinstance(cpu, (int, float)) else None
        mem_pct = mem * 100 if isinstance(mem, (int, float)) else None
        _log(
            f"Step {step_number}{total_suffix}: HPA desired {desired}, current {current} | "
            f"CPU {_fmt_num(cpu_pct, '{:.0f}')}%/260% | memory {_fmt_num(mem_pct, '{:.0f}')}%/120% | "
            f"p95 {_fmt_num(p95, '{:.0f}')} ms"
        )


def _write_status(status_file: Optional[Path], payload: Dict[str, Any]) -> None:
    if status_file is None:
        return
    payload = {**payload, "updated_at": time.time()}
    tmp = status_file.with_suffix(status_file.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, default=str))
    tmp.replace(status_file)


def _append_sample_jsonl(samples_jsonl: Optional[Path], sample: EpisodeSample) -> None:
    if samples_jsonl is None:
        return
    samples_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(samples_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(sample.to_dict(), default=str) + "\n")


def run_demo_episode_v2(
    arm: str,
    profile_name: str,
    status_file: Optional[Path] = None,
    samples_jsonl: Optional[Path] = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    epsilon: float = 0.05,
    protection_check_fn: Callable[[], Dict[str, Any]] = None,
    orchestrator_factory: Callable[[], Any] = build_demo_traffic_orchestrator,
    prom_query_fn: Optional[Callable[[], "tuple[Optional[float], Optional[str]]"]] = None,
    traffic_confirm_fn: Callable[..., Dict[str, Any]] = confirm_traffic_arriving,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Dict[str, Any]:
    """Orchestrates ONE demo episode: training-protection check, REAL
    traffic start + sec.2 traffic-arriving guard, then one of the two demo
    arms via `scripts/eval_v2_episodes.py`'s unified recorders, streaming
    each `EpisodeSample` to `samples_jsonl` the moment it is produced.
    Raises `DemoRefusedError` (no cluster change at all) if the protection
    check fails. `orchestrator_factory`/`prom_query_fn`/`traffic_confirm_fn`
    are injectable so a unit test can exercise the whole traffic-guard
    control flow with a fake generator/fake Prometheus reading -- no real
    subprocess, Docker, or cluster involved."""
    if arm not in DEMO_ARMS:
        raise ValueError(f"arm must be one of {DEMO_ARMS}, got {arm!r}")

    profile_meta = resolve_demo_profile_metadata(profile_name)
    episode_duration_s = profile_meta.duration_seconds
    traffic_desc = _traffic_description(profile_name)

    try:
        env_config_for_logging = load_env_config_v2_from_yaml()
        n_steps_total: Optional[int] = int(round(episode_duration_s / env_config_for_logging.decision_interval_s))
        initial_replicas_for_logging: Optional[int] = env_config_for_logging.initial_replicas
    except Exception:  # noqa: BLE001 -- purely a log-line nicety, never blocks a run
        n_steps_total = None
        initial_replicas_for_logging = None

    if prom_query_fn is None:
        _prom_client = PrometheusClient(base_url=PROMETHEUS_URL)
        prom_query_fn = lambda: _prom_client.query(DEMO_TRAFFIC_REQUEST_RATE_PROMQL)  # noqa: E731

    _log(f"Demo episode starting: arm={arm}, profile={profile_name!r} ({traffic_desc})")
    _write_status(status_file, {"status": "running", "stage": "PREFLIGHT", "arm": arm, "profile": profile_name})
    if protection_check_fn is None:
        protection_check_fn = lambda: check_training_protection(list_hpa_names_fn=_real_list_hpa_names)  # noqa: E731
    protection = protection_check_fn()
    if not protection["ok"]:
        message = "; ".join(protection["reasons"])
        _log(f"Preflight FAILED: {message}")
        _write_status(status_file, {
            "status": "refused", "stage": "REFUSED", "arm": arm, "profile": profile_name,
            "error": f"refusing to start: {message}",
        })
        raise DemoRefusedError(f"refusing to start demo episode: {message}")
    _log(f"Preflight OK: lock {read_lock()!r}, no conflicting process or unexpected HPA detected")

    if samples_jsonl is not None and samples_jsonl.exists():
        samples_jsonl.unlink()  # fresh stream for this run, never appends to a stale prior run's data

    if arm == "ppo_v2":
        # `demo-app-hpa-v2` being present here is the canonical IDLE state
        # the protection check above just confirmed (no OTHER HPA is
        # present) -- never a conflict to refuse on. PPO is a pure-Python
        # controller that requires ZERO HPA objects present to acquire the
        # controller lock (`acquire_controller(CONTROLLER_PPO, ...)` inside
        # `run_ppo_v2_eval_episode` fails closed on ANY HPA), so it must be
        # deleted here, immediately before the ppo_v2 arm starts -- same
        # pattern `evaluate_v2_campaign.build_live_context()`'s own
        # `ppo_episode` wrapper already uses for the live campaign
        # (CORR-63 lineage).
        removed = remove_hpa_before_ppo_start()
        if removed:
            _log(f"Removed idle HPA {removed}; controller lock → ppo")
        else:
            _log("No idle HPA to remove; controller lock → ppo")
    else:
        _log("Controller lock → hpa (CPU 260% / memory 120% targets)")

    _write_status(status_file, {"status": "running", "stage": "STARTING", "arm": arm, "profile": profile_name})

    def _to_status_sample(sample_dict: Dict[str, Any]) -> Dict[str, Any]:
        """`experiment_store.extract_series()` (demo_ui, generation-agnostic)
        requires a literal `"time_s"` key on every sample -- `EpisodeSample`'s
        own field is `t_s` (Phase 5C-A's schema, unrelated to the UI). This
        is the ONE place that renaming happens, so `eval_v2_episodes.py`'s
        schema stays exactly as designed and the UI's generic series
        extraction keeps working unmodified."""
        return {**sample_dict, "time_s": sample_dict["t_s"]}

    collected: List[Dict[str, Any]] = []
    # Task sec.4: tracks each decision's own resulting replica count so
    # the NEXT decision's log line can report the real pre-decision
    # ("before") count -- see `_log_decision()`'s own docstring for why
    # `decision_record_v2.py`'s `current_replicas` field cannot be trusted
    # for this. Seeded with the episode's configured initial replica count
    # (the real pre-decision-0 count), falling back to `None` (the log
    # line then falls back to whatever `decision.get("current_replicas")`
    # reports, never crashing over a logging nicety) if that config lookup
    # itself failed above.
    _prev_replicas: Dict[str, Optional[int]] = {"value": initial_replicas_for_logging}

    def on_sample(sample: EpisodeSample) -> None:
        sample_dict = sample.to_dict()
        _append_sample_jsonl(samples_jsonl, sample)
        collected.append(_to_status_sample(sample_dict))
        before_replicas = _prev_replicas["value"]
        _log_decision(sample_dict, len(collected), n_steps_total, before_replicas=before_replicas)
        after_replicas = sample_dict.get("desired_replicas")
        if after_replicas is not None:
            _prev_replicas["value"] = after_replicas
        _write_status(status_file, {
            "status": "running", "stage": "EPISODE_RUNNING", "arm": arm, "profile": profile_name,
            "decision_index": sample.decision_index, "note": "DEMO -- not evaluation evidence",
            "samples": list(collected),
        })

    def _partial_summary(episode_id_: str) -> Dict[str, Any]:
        return {
            "episode_id": episode_id_, "arm": arm, "profile": profile_name,
            "n_samples": len(collected), "demo_only": True, "partial": True,
            "workload_hash": profile_meta.sha256, "workload_seed": profile_meta.seed,
            "workload_duration_seconds": profile_meta.duration_seconds,
            "samples": list(collected),
        }

    # V2 Phase 9F: the run's terminal payload is built as soon as the
    # outcome (success/stop/error) is known, but NEVER written to the
    # status file until AFTER cleanup below has run -- see the ordering
    # note on `finally`. `None` here means "no outcome determined yet";
    # the broad `except Exception` further down fills it in for a truly
    # unanticipated failure that none of the specific `except` clauses
    # caught, so the status file is never left stuck on a mid-run stage
    # with nothing but a dead pid for reconciliation to go on.
    terminal_payload: Optional[Dict[str, Any]] = None
    episode_id = f"demo_v2_{arm}_{profile_name}_{int(time.time())}"
    orchestrator = orchestrator_factory()
    handle = None
    try:
        try:
            # Task sec.1/sec.2: start the REAL traffic generator (the V5
            # relay route + `python -m loadgen.traffic_generator`
            # subprocess, the SAME generic machinery the live evaluation/
            # training traffic paths already use), then CONFIRM (never
            # assume) that requests are actually arriving before the
            # episode's decision loop is allowed to start. Any failure
            # here -- route/generator start failure, or the sec.2 guard
            # timing out -- is raised as `ToolingFailureError`, caught by
            # the SAME handler below every other tooling failure uses, so
            # the run ends FAILED/cleaned-up exactly like any other
            # precondition failure, never left running blind with zero
            # traffic.
            try:
                handle = orchestrator.start_episode_traffic(profile_name, episode_id)
            except Exception as exc:  # noqa: BLE001 -- to.RouteEstablishmentFailure / to.TrafficGeneratorFailure / any transport failure
                raise ToolingFailureError(f"failed to start demo traffic: {exc}") from exc
            _log(f"Traffic generator started: {traffic_desc} ({episode_duration_s:.0f} s); confirming requests are arriving...")
            confirm = traffic_confirm_fn(handle, prom_query_fn, sleep_fn=sleep_fn)
            if not confirm.get("ok"):
                raise ToolingFailureError(f"traffic not reaching the app: {confirm.get('reason')}")
            _log(f"Traffic confirmed: {_fmt_num(confirm.get('observed_rps'), '{:.1f}')} req/s observed")

            try:
                if arm == "ppo_v2":
                    result = run_ppo_v2_eval_episode(
                        model_path, epsilon, profile_name, episode_duration_s, episode_id, on_sample=on_sample,
                    )
                else:
                    result = run_hpa_v2_cpu_memory_eval_episode(
                        profile_name, episode_duration_s, episode_id, on_sample=on_sample,
                    )
            finally:
                # Stops the generator/drains server-side in-flight
                # requests regardless of how the episode ended (success,
                # STOP, safety termination, ...) -- never masks the
                # episode's own outcome (caught, logged, never re-raised)
                # over a traffic-teardown failure; `orchestrator.
                # force_cleanup()` in the outer `finally` below is the
                # unconditional last-resort route teardown either way.
                try:
                    orchestrator.stop_episode_traffic(handle, episode_id)
                except Exception as exc:  # noqa: BLE001
                    _log(f"Warning: failed to stop the traffic generator cleanly: {exc}")
                try:
                    orchestrator.drain_server_side_requests()
                except Exception as exc:  # noqa: BLE001
                    _log(f"Warning: server-side request drain failed/timed out: {exc}")
        except DemoStopRequested as exc:
            _log(f"Run stopped: {exc}")
            terminal_payload = {
                "status": "stopped", "stage": "STOPPED", "arm": arm, "profile": profile_name,
                "error": str(exc), "summary": _partial_summary(episode_id),
            }
            raise
        except (SafetyTerminationError, ToolingFailureError, CpuOnlyArmNotYetSupportedError) as exc:
            _log(f"Run error ({type(exc).__name__}): {exc}")
            terminal_payload = {
                "status": "error", "stage": "ERROR", "arm": arm, "profile": profile_name,
                "error": str(exc), "summary": _partial_summary(episode_id),
            }
            raise

        summary = {
            "episode_id": result["episode_id"], "arm": arm, "profile": profile_name,
            "n_samples": len(result["samples"]), "demo_only": True, "partial": False,
            "workload_hash": profile_meta.sha256, "workload_seed": profile_meta.seed,
            "workload_duration_seconds": profile_meta.duration_seconds,
            "samples": [_to_status_sample(s) for s in result["samples"]],
        }
        _log(f"Episode complete: {len(result['samples'])} decisions recorded")
        terminal_payload = {"status": "completed", "stage": "COMPLETED", "arm": arm, "profile": profile_name, "summary": summary}
        # (a) write the run summary -- the full outcome is already on
        # disk and readable by RE-ATTACH/the UI, via a non-terminal
        # `stage` so reconciliation still treats the run as ACTIVE
        # (controls stay locked) until cleanup below actually reaches
        # the canonical idle cluster state -- never unlocking the UI
        # to start a new run before this one has finished tearing
        # down.
        _write_status(status_file, {"status": "running", "stage": "FINISHING", "arm": arm, "profile": profile_name, "summary": summary})
        return summary
    except Exception as exc:  # noqa: BLE001 -- catches anything the specific handlers above did not
        if terminal_payload is None:
            _log(f"Unexpected failure: {exc}")
            terminal_payload = {
                "status": "error", "stage": "ERROR", "arm": arm, "profile": profile_name,
                "error": f"unexpected failure: {exc}", "summary": _partial_summary(episode_id),
            }
        raise
    finally:
        # V2 Phase 9E/9F: UNCONDITIONAL cleanup -- on success, on any
        # exception, and on an explicit STOP request (DemoStopRequested,
        # raised from the SIGTERM/SIGINT handler `main()` installs, or
        # from the `atexit` safety net) alike. Always restores the
        # canonical idle cluster state (deletes any HPA object present
        # that isn't `demo-app-hpa-v2`, re-applies it at 2 replicas) and
        # releases the run lock, exactly as `evaluate_v2_campaign.py`'s own
        # live campaign does after every attempt
        # (`_safe_cleanup`/`restore_canonical_idle_state`). Task sec.1:
        # `orchestrator.force_cleanup()` is the traffic orchestrator's own
        # safety-net route teardown (never raises) -- covers the case
        # where `start_episode_traffic()` itself raised before `handle`
        # ever existed, or anything else left the relay route dangling.
        #
        # (b) cleanup happens here, BEFORE...
        _log("Traffic stopped; cleanup starting (restore idle HPA at 2 replicas, release controller lock)")
        orchestrator.force_cleanup()
        cleanup_result = _restore_idle_state_safely()
        _log(f"Cleanup result: {cleanup_result}")
        # (c) ...the terminal status is written LAST. This ordering (not
        # the original "write completed, then clean up") is deliberate:
        # the terminal status is the one signal reconciliation
        # (`experiment_store`) trusts as ground truth over a dead/stale
        # pid, so it must never be visible until the cluster is actually
        # back in its canonical idle state -- otherwise the UI could
        # unlock and let a new run start while cleanup from this one was
        # still in flight.
        final_payload = terminal_payload or {
            "status": "error", "stage": "ERROR", "arm": arm, "profile": profile_name,
            "error": "run ended without a determined outcome", "summary": _partial_summary(episode_id),
        }
        _log(f"Run finished: status={final_payload.get('status')}")
        if status_file is not None:
            _write_status(status_file, {**final_payload, "cleanup": cleanup_result})


def main(argv: Optional[List[str]] = None) -> int:
    # V2 Phase 9E/9F: `demo_ui.live_execution_adapter.RealExperimentControllerV2.
    # stop()` sends SIGTERM to this process's whole process group -- without
    # this handler that kills the process with NO cluster cleanup at all.
    # Converting it into `DemoStopRequested` routes a STOP exactly like any
    # other exception, through the same unconditional `finally` cleanup in
    # `run_demo_episode_v2()`. SIGINT (Ctrl-C, e.g. a developer running this
    # script directly from a terminal) is handled identically.
    signal.signal(signal.SIGTERM, _raise_stop_requested)
    signal.signal(signal.SIGINT, _raise_stop_requested)
    # Last-resort safety net for any normal interpreter shutdown path that
    # reaches here WITHOUT having gone through `run_demo_episode_v2()`'s own
    # `finally` (there should not be one, given the above, but this is
    # cheap and `_restore_idle_state_safely()`/`restore_canonical_idle_state()`
    # are both idempotent, so running it twice is harmless). Never fires on
    # SIGKILL -- nothing in a Python process can catch that.
    atexit.register(_restore_idle_state_safely)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=DEMO_ARMS)
    parser.add_argument("--profile", required=True, choices=DEMO_PROFILE_ALLOWLIST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--status-file", type=Path, default=None)
    parser.add_argument("--samples-jsonl", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        summary = run_demo_episode_v2(
            args.arm, args.profile, args.status_file, args.samples_jsonl, args.model, args.epsilon,
        )
    except Exception as exc:  # noqa: BLE001 -- must surface to the UI, never crash silently
        print(f"[{time.strftime('%H:%M:%S')}] STAGE=ERROR: {exc}", file=sys.stderr, flush=True)
        return 1

    # Task sec.4: never dump the full per-decision `samples` list (every
    # EpisodeSample, including the raw/normalized observation vectors) to
    # stdout -- this process's stdout is redirected verbatim into the
    # run's human-readable log file (`live_execution_adapter.py`'s
    # `stdout=log_handle`), and the full samples blob belongs ONLY in
    # `--samples-jsonl`/the status file's own JSON, never appended to that
    # log. This compact line (episode id/arm/profile/sample count) is the
    # ONE line printed after the per-decision progress lines, giving a
    # terminal-usable confirmation without repeating their content.
    compact_summary = {k: v for k, v in summary.items() if k != "samples"}
    compact_summary["n_samples"] = len(summary.get("samples") or [])
    print(json.dumps(compact_summary, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
