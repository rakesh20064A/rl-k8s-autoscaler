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
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autoscaler.src.rl_env.controller_exclusivity import CONTROLLER_NONE, V2_HPA_NAME, read_lock  # noqa: E402
from loadgen.traffic_generator import parse_traffic_profile  # noqa: E402
from scripts.eval_v2_episodes import (  # noqa: E402
    CpuOnlyArmNotYetSupportedError,
    EpisodeSample,
    SafetyTerminationError,
    ToolingFailureError,
    run_hpa_v2_cpu_memory_eval_episode,
    run_ppo_v2_eval_episode,
)

DEMO_PROFILE_DIR = PROJECT_ROOT / "configs" / "traffic_profiles" / "v2_demo"
DEMO_PROFILE_ALLOWLIST = ("demo_cpu_short", "demo_memory_short", "demo_mixed_short")
DEMO_ARMS = ("ppo_v2", "hpa_v2_cpu_memory")

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ppo_v2" / "ppo_model_v2_min2.zip"


class DemoProfileFirewallViolation(RuntimeError):
    pass


class DemoRefusedError(RuntimeError):
    """Raised when a training/evaluation-protection precondition fails --
    never a cluster mutation of any kind up to this point."""


def resolve_demo_profile_path(profile_name: str) -> Path:
    if profile_name not in DEMO_PROFILE_ALLOWLIST:
        raise DemoProfileFirewallViolation(f"{profile_name!r} is not a v2_demo profile; must be one of {DEMO_PROFILE_ALLOWLIST}")
    path = DEMO_PROFILE_DIR / f"{profile_name}.yaml"
    if not path.exists():
        raise DemoProfileFirewallViolation(f"demo profile file missing on disk: {path}")
    return path


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
) -> Dict[str, Any]:
    """Orchestrates ONE demo episode: training-protection check, then one
    of the two demo arms via `scripts/eval_v2_episodes.py`'s unified
    recorders, streaming each `EpisodeSample` to `samples_jsonl` the
    moment it is produced. Raises `DemoRefusedError` (no cluster change at
    all) if the protection check fails."""
    if arm not in DEMO_ARMS:
        raise ValueError(f"arm must be one of {DEMO_ARMS}, got {arm!r}")

    profile_path = resolve_demo_profile_path(profile_name)
    profile = parse_traffic_profile(profile_path)
    episode_duration_s = profile.total_duration

    _write_status(status_file, {"status": "running", "stage": "PREFLIGHT", "arm": arm, "profile": profile_name})
    if protection_check_fn is None:
        protection_check_fn = lambda: check_training_protection(list_hpa_names_fn=_real_list_hpa_names)  # noqa: E731
    protection = protection_check_fn()
    if not protection["ok"]:
        message = "; ".join(protection["reasons"])
        _write_status(status_file, {
            "status": "refused", "stage": "REFUSED", "arm": arm, "profile": profile_name,
            "error": f"refusing to start: {message}",
        })
        raise DemoRefusedError(f"refusing to start demo episode: {message}")

    if samples_jsonl is not None and samples_jsonl.exists():
        samples_jsonl.unlink()  # fresh stream for this run, never appends to a stale prior run's data

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

    def on_sample(sample: EpisodeSample) -> None:
        sample_dict = sample.to_dict()
        _append_sample_jsonl(samples_jsonl, sample)
        collected.append(_to_status_sample(sample_dict))
        _write_status(status_file, {
            "status": "running", "stage": "EPISODE_RUNNING", "arm": arm, "profile": profile_name,
            "decision_index": sample.decision_index, "note": "DEMO -- not evaluation evidence",
            "samples": list(collected),
        })

    episode_id = f"demo_v2_{arm}_{profile_name}_{int(time.time())}"
    try:
        if arm == "ppo_v2":
            result = run_ppo_v2_eval_episode(
                model_path, epsilon, profile_name, episode_duration_s, episode_id, on_sample=on_sample,
            )
        else:
            result = run_hpa_v2_cpu_memory_eval_episode(
                profile_name, episode_duration_s, episode_id, on_sample=on_sample,
            )
    except (SafetyTerminationError, ToolingFailureError, CpuOnlyArmNotYetSupportedError) as exc:
        _write_status(status_file, {"status": "error", "stage": "ERROR", "arm": arm, "profile": profile_name, "error": str(exc)})
        raise

    summary = {
        "episode_id": result["episode_id"], "arm": arm, "profile": profile_name,
        "n_samples": len(result["samples"]), "demo_only": True,
        "samples": [_to_status_sample(s) for s in result["samples"]],
    }
    _write_status(status_file, {"status": "completed", "stage": "COMPLETED", "arm": arm, "profile": profile_name, "summary": summary})
    return summary


def main(argv: Optional[List[str]] = None) -> int:
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

    print(json.dumps(summary, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
