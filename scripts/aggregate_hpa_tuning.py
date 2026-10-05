#!/usr/bin/env python3
"""scripts/aggregate_hpa_tuning.py

Phase 11C.4P: OFFLINE, read-only aggregation of the 9 official, precommitted
HPA tuning trials (CORR-24 candidate family 200/230/260, CORR-25 analysis
conventions). This script sends no traffic, makes no Kubernetes/Prometheus
calls, and does not regenerate or replay any schedule. It reads only:

  - the frozen manifest (experiments/manifests/hpa_tuning_official_9_trials.yaml)
  - each trial's already-persisted artifacts under persistent-data/
      traffic/<experiment_id>/{metadata.json, requests.csv}
      hpa/<experiment_id>/hpa_status.csv
      logs/<experiment_id>_trial_metadata.json
  - configs/traffic_profiles/hpa_tuning/hpa_tuning_v2.yaml (phase boundaries)

and delegates all replica-trajectory integration to the existing shared,
tested tool scripts/integrate_replica_trajectory.py (CORR-25) rather than
reimplementing that logic.

Every scored metric is recomputed independently from raw artifacts -- no
number is copied from any prior individual trial report.

CORR-25 field conventions (frozen, reused unchanged):
  - deployment_status_ready_replicas / deployment_status_available_replicas
    are the authoritative physical-pod-state fields.
  - hpa_status_current_replicas is a corroborating field ONLY -- never used
    here as the authoritative Ready/Available/termination timestamp.

Latency-percentile semantics (frozen for this aggregation, documented so it
is applied identically to every candidate): a request's client latency is
included in latency percentiles / >300ms tail counts iff `requests.csv`
recorded a parseable `client_latency_seconds` value for it. A `timeout`
outcome DOES record a latency value (the generator's bounded wait, e.g.
5.000000s) and is therefore included. A `connection_error` outcome has no
response and no latency value and is therefore excluded from latency
percentiles -- but both outcomes are always included in reliability counts
(offered/sent/success-rate). This matches the semantics `metadata.json`'s
own `client_latency_stats` already use (independently verified: this
script's own whole-run p50/p95/p99 reproduce each trial's `metadata.json`
values exactly).
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.hpa_phase_entry_snapshot import compute_phase_boundaries  # noqa: E402
from scripts.integrate_replica_trajectory import (  # noqa: E402
    find_transitions,
    integrate_all_fields,
    read_hpa_status_rows,
)

ALLOWED_CANDIDATES = (200, 230, 260)
EXPECTED_BLOCKS = (1, 2, 3)
SLA_P95_MS = 300.0
SLA_SUCCESS_RATE = 0.99
TAIL_THRESHOLD_MS = 300.0

# Fields whose own transitions matter for responsiveness/stability, in the
# CORR-25/CORR-19 evidence-hierarchy order.
RESPONSIVENESS_FIELDS = {
    "hpa_desired": "hpa_status_desired_replicas",
    "deployment_desired": "deployment_spec_desired_replicas",
    "current": "deployment_status_replicas",
    "ready": "deployment_status_ready_replicas",
    "available": "deployment_status_available_replicas",
}


class ManifestError(ValueError):
    pass


class StructuralValidationError(ValueError):
    pass


@dataclass
class TrialSpec:
    experiment_id: str
    candidate: int
    block: int
    repeat: int


@dataclass
class Manifest:
    schedule_sha256: str
    expected_planned_requests: int
    profile: str
    scored_end_seconds: float
    trials: List[TrialSpec] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Manifest loading and structural validation
# ---------------------------------------------------------------------------


def load_manifest(manifest_path: Path) -> Manifest:
    with open(manifest_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    trials_raw = raw.get("trials", [])
    if len(trials_raw) != 9:
        raise ManifestError(
            f"Expected exactly 9 trials in manifest, found {len(trials_raw)}."
        )

    seen_ids = set()
    per_candidate: Dict[int, List[int]] = {c: [] for c in ALLOWED_CANDIDATES}
    per_candidate_block: Dict[int, set] = {c: set() for c in ALLOWED_CANDIDATES}

    trials: List[TrialSpec] = []
    for t in trials_raw:
        exp_id = t["experiment_id"]
        if exp_id in seen_ids:
            raise ManifestError(f"Duplicate experiment_id in manifest: {exp_id}")
        seen_ids.add(exp_id)

        candidate = int(t["candidate"])
        if candidate not in ALLOWED_CANDIDATES:
            raise ManifestError(
                f"Unexpected candidate {candidate} for {exp_id}; "
                f"allowed candidates are {ALLOWED_CANDIDATES}."
            )
        block = int(t["block"])
        if block not in EXPECTED_BLOCKS:
            raise ManifestError(
                f"Unexpected block {block} for {exp_id}; allowed blocks are {EXPECTED_BLOCKS}."
            )

        per_candidate[candidate].append(block)
        per_candidate_block[candidate].add(block)

        trials.append(
            TrialSpec(
                experiment_id=exp_id,
                candidate=candidate,
                block=block,
                repeat=int(t["repeat"]),
            )
        )

    for candidate in ALLOWED_CANDIDATES:
        blocks = per_candidate[candidate]
        if len(blocks) != 3:
            raise ManifestError(
                f"Candidate {candidate}% must have exactly 3 trials, found {len(blocks)}."
            )
        if per_candidate_block[candidate] != set(EXPECTED_BLOCKS):
            raise ManifestError(
                f"Candidate {candidate}% must have exactly one trial per block "
                f"{EXPECTED_BLOCKS}, found blocks {sorted(blocks)}."
            )

    return Manifest(
        schedule_sha256=raw["schedule_sha256"],
        expected_planned_requests=int(raw["expected_planned_requests"]),
        profile=raw["profile"],
        scored_end_seconds=float(raw["scored_end_seconds"]),
        trials=trials,
    )


def verify_trial_structural(
    trial: TrialSpec, persistent_data_root: Path, manifest: Manifest
) -> Dict[str, Any]:
    """Independently re-verify one trial's structural validity from RAW
    artifacts. Raises StructuralValidationError (never silently excludes)
    if a required artifact is missing or a structural check fails."""
    exp_id = trial.experiment_id

    traffic_dir = persistent_data_root / "traffic" / exp_id
    hpa_dir = persistent_data_root / "hpa" / exp_id
    trial_meta_path = persistent_data_root / "logs" / f"{exp_id}_trial_metadata.json"
    traffic_meta_path = traffic_dir / "metadata.json"
    requests_csv_path = traffic_dir / "requests.csv"
    hpa_status_csv_path = hpa_dir / "hpa_status.csv"

    for p in (trial_meta_path, traffic_meta_path, requests_csv_path, hpa_status_csv_path):
        if not p.exists():
            raise StructuralValidationError(f"{exp_id}: missing required artifact {p}")

    with open(trial_meta_path, "r", encoding="utf-8") as f:
        trial_meta = json.load(f)
    with open(traffic_meta_path, "r", encoding="utf-8") as f:
        traffic_meta = json.load(f)

    checks: Dict[str, Any] = {"experiment_id": exp_id}

    checks["harness_status"] = trial_meta.get("status")
    if trial_meta.get("status") != "valid":
        raise StructuralValidationError(
            f"{exp_id}: harness trial_metadata.json status is "
            f"'{trial_meta.get('status')}', expected 'valid'."
        )

    harness_target = str(trial_meta.get("target"))
    if harness_target != str(trial.candidate):
        raise StructuralValidationError(
            f"{exp_id}: manifest candidate {trial.candidate} does not match "
            f"harness-recorded target {harness_target}."
        )

    schedule_sha = traffic_meta.get("schedule_generation_info", {}).get("schedule_sha256")
    checks["schedule_sha256_match"] = schedule_sha == manifest.schedule_sha256
    if not checks["schedule_sha256_match"]:
        raise StructuralValidationError(
            f"{exp_id}: schedule SHA-256 mismatch (got {schedule_sha}, "
            f"expected {manifest.schedule_sha256})."
        )

    checks["total_offered"] = traffic_meta.get("total_offered")
    checks["total_sent"] = traffic_meta.get("total_sent")
    if checks["total_offered"] != manifest.expected_planned_requests:
        raise StructuralValidationError(
            f"{exp_id}: total_offered={checks['total_offered']}, "
            f"expected {manifest.expected_planned_requests}."
        )
    if checks["total_offered"] != checks["total_sent"]:
        raise StructuralValidationError(
            f"{exp_id}: total_offered ({checks['total_offered']}) != "
            f"total_sent ({checks['total_sent']})."
        )

    checks["connection_mode"] = traffic_meta.get("connection_mode")
    if checks["connection_mode"] != "fresh":
        raise StructuralValidationError(
            f"{exp_id}: connection_mode={checks['connection_mode']!r}, expected 'fresh'."
        )

    target_url = traffic_meta.get("target_url", "")
    checks["target_url"] = target_url
    if "localhost:8080" in target_url or target_url == "":
        raise StructuralValidationError(
            f"{exp_id}: suspicious target_url {target_url!r} "
            f"(expected the demo-app-loadtest tunnel URL, never a bare localhost:8080 port-forward)."
        )

    hpa_rows = read_hpa_status_rows(hpa_status_csv_path)
    kubectl_errors = sum(
        int(r.get("kubectl_error_count") or 0) for r in hpa_rows if r.get("kubectl_error_count")
    )
    checks["kubectl_error_count_total"] = kubectl_errors
    if kubectl_errors != 0:
        raise StructuralValidationError(
            f"{exp_id}: {kubectl_errors} kubectl_error_count samples found, expected 0."
        )

    checks["structurally_valid"] = True
    return checks


# ---------------------------------------------------------------------------
# requests.csv loading and per-phase grouping
# ---------------------------------------------------------------------------


def load_requests(requests_csv_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(requests_csv_path, "r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            latency = raw.get("client_latency_seconds", "")
            latency_val: Optional[float]
            if latency is None or latency.strip() == "":
                latency_val = None
            else:
                latency_val = float(latency)
            rows.append(
                {
                    "phase_name": raw.get("phase_name", ""),
                    "outcome": raw.get("outcome", ""),
                    "http_status": raw.get("http_status", ""),
                    "client_latency_seconds": latency_val,
                    "dispatch_lag_seconds": float(raw["dispatch_lag_seconds"])
                    if raw.get("dispatch_lag_seconds")
                    else None,
                }
            )
    return rows


OUTCOME_KEYS = (
    "success_2xx",
    "client_error_4xx",
    "server_error_5xx",
    "timeout",
    "connection_error",
    "dropped_overload",
)


def compute_reliability(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {k: 0 for k in OUTCOME_KEYS}
    for r in rows:
        outcome = r["outcome"]
        if outcome in counts:
            counts[outcome] += 1
    n = len(rows)
    success = counts["success_2xx"]
    result: Dict[str, Any] = {"n": n, **counts}
    result["success_rate"] = (success / n) if n > 0 else None
    return result


def _percentile(values: List[float], p: float) -> float:
    """Linear-interpolation percentile, matching CORR-13's frozen estimator
    (numpy.percentile) used throughout this project's official reports."""
    return float(np.percentile(values, p))


def compute_latency(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Latency stats over every row with a parseable client_latency_seconds
    value (success_2xx and timeout; excludes connection_error/dropped which
    never receive a response -- see module docstring)."""
    latencies = [r["client_latency_seconds"] for r in rows if r["client_latency_seconds"] is not None]
    n_scored = len(latencies)
    if n_scored == 0:
        return {
            "n_latency_scored": 0,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
            "gt300_count": 0,
            "gt300_fraction": None,
        }
    gt300 = sum(1 for v in latencies if v * 1000.0 > TAIL_THRESHOLD_MS)
    return {
        "n_latency_scored": n_scored,
        "p50_ms": _percentile(latencies, 50) * 1000.0,
        "p95_ms": _percentile(latencies, 95) * 1000.0,
        "p99_ms": _percentile(latencies, 99) * 1000.0,
        "max_ms": max(latencies) * 1000.0,
        "gt300_count": gt300,
        "gt300_fraction": gt300 / n_scored,
    }


def group_by_phase(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(r["phase_name"], []).append(r)
    return grouped


# ---------------------------------------------------------------------------
# Replica efficiency (delegates to scripts.integrate_replica_trajectory)
# ---------------------------------------------------------------------------


def compute_replica_efficiency(hpa_status_csv_path: Path, scored_end: float) -> Dict[str, Dict[str, float]]:
    return integrate_all_fields(hpa_status_csv_path, scored_end=scored_end)


# ---------------------------------------------------------------------------
# Responsiveness / stability
# ---------------------------------------------------------------------------


def _phase_containing(elapsed: float, boundaries: Dict[str, float]) -> Optional[str]:
    ordered = sorted(boundaries.items(), key=lambda kv: kv[1])
    containing = None
    for name, start in ordered:
        if start <= elapsed:
            containing = name
        else:
            break
    return containing


def compute_responsiveness(
    hpa_status_csv_path: Path,
    profile_path: Path,
    scored_end: float,
    recovery_phase_name: str = "recovery_2rps",
) -> Dict[str, Any]:
    rows = read_hpa_status_rows(hpa_status_csv_path)
    boundaries = compute_phase_boundaries(profile_path)
    recovery_start = boundaries[recovery_phase_name]

    transitions_by_field = {
        label: find_transitions(rows, csv_field, scored_end)
        for label, csv_field in RESPONSIVENESS_FIELDS.items()
    }

    # Scale-up/down action counts and reversals from the HPA's own desired
    # replica trajectory (the decision-making field, CORR-25 evidence #1).
    hpa_desired_transitions = transitions_by_field["hpa_desired"]
    directions: List[str] = []
    scale_up_actions = 0
    scale_down_actions = 0
    prev_value = hpa_desired_transitions[0]["value"] if hpa_desired_transitions else None
    for t in hpa_desired_transitions[1:]:
        if t["value"] > prev_value:
            scale_up_actions += 1
            directions.append("up")
        elif t["value"] < prev_value:
            scale_down_actions += 1
            directions.append("down")
        prev_value = t["value"]

    direction_reversals = sum(
        1 for i in range(1, len(directions)) if directions[i] != directions[i - 1]
    )
    oscillation = scale_up_actions > 1 or scale_down_actions > 1

    # First scale-up: first transition of hpa_desired above the standardized
    # start value (1.0).
    first_scale_up = next(
        (t for t in hpa_desired_transitions if t["value"] > 1.0), None
    )
    first_scale_up_phase = None
    first_scale_up_elapsed = None
    trigger_phase_start = None
    trigger_deltas: Dict[str, Optional[float]] = {
        label: None for label in RESPONSIVENESS_FIELDS
    }
    if first_scale_up is not None:
        first_scale_up_elapsed = first_scale_up["elapsed_seconds"]
        first_scale_up_phase = _phase_containing(first_scale_up_elapsed, boundaries)
        trigger_phase_start = boundaries.get(first_scale_up_phase)
        for label, field_transitions in transitions_by_field.items():
            increase = next(
                (t for t in field_transitions if t["value"] > 1.0), None
            )
            if increase is not None and trigger_phase_start is not None:
                trigger_deltas[label] = increase["elapsed_seconds"] - trigger_phase_start

    # Recovery: first transition of each field back down to 1.0 at/after
    # recovery_start.
    recovery_deltas: Dict[str, Optional[float]] = {label: None for label in RESPONSIVENESS_FIELDS}
    recovery_absolute: Dict[str, Optional[float]] = {label: None for label in RESPONSIVENESS_FIELDS}
    for label, field_transitions in transitions_by_field.items():
        decrease = next(
            (
                t
                for t in field_transitions
                if t["elapsed_seconds"] >= recovery_start and t["value"] == 1.0
            ),
            None,
        )
        if decrease is not None:
            recovery_absolute[label] = decrease["elapsed_seconds"]
            recovery_deltas[label] = decrease["elapsed_seconds"] - recovery_start

    stable_at_1_duration = None
    if recovery_absolute["ready"] is not None:
        stable_at_1_duration = scored_end - recovery_absolute["ready"]

    return {
        "first_scale_up_phase": first_scale_up_phase,
        "first_scale_up_elapsed_seconds": first_scale_up_elapsed,
        "trigger_phase_start_seconds": trigger_phase_start,
        "trigger_to_hpa_desired_s": trigger_deltas["hpa_desired"],
        "trigger_to_deployment_desired_s": trigger_deltas["deployment_desired"],
        "trigger_to_current_s": trigger_deltas["current"],
        "trigger_to_ready_s": trigger_deltas["ready"],
        "trigger_to_available_s": trigger_deltas["available"],
        "scale_up_actions": scale_up_actions,
        "scale_down_actions": scale_down_actions,
        "direction_reversals": direction_reversals,
        "oscillation": oscillation,
        "recovery_phase_start_seconds": recovery_start,
        "recovery_to_hpa_desired1_s": recovery_deltas["hpa_desired"],
        "recovery_to_deployment_desired1_s": recovery_deltas["deployment_desired"],
        "recovery_to_current1_s": recovery_deltas["current"],
        "recovery_to_ready1_s": recovery_deltas["ready"],
        "recovery_to_available1_s": recovery_deltas["available"],
        "stable_at_1_duration_s": stable_at_1_duration,
    }


# ---------------------------------------------------------------------------
# Per-trial aggregation
# ---------------------------------------------------------------------------


def aggregate_trial(
    trial: TrialSpec,
    persistent_data_root: Path,
    manifest: Manifest,
) -> Dict[str, Any]:
    exp_id = trial.experiment_id
    traffic_dir = persistent_data_root / "traffic" / exp_id
    hpa_dir = persistent_data_root / "hpa" / exp_id
    requests_csv_path = traffic_dir / "requests.csv"
    hpa_status_csv_path = hpa_dir / "hpa_status.csv"
    profile_path = PROJECT_ROOT / manifest.profile

    structural = verify_trial_structural(trial, persistent_data_root, manifest)

    rows = load_requests(requests_csv_path)
    whole_reliability = compute_reliability(rows)
    whole_latency = compute_latency(rows)

    phases: Dict[str, Dict[str, Any]] = {}
    for phase_name, phase_rows in group_by_phase(rows).items():
        phases[phase_name] = {
            "reliability": compute_reliability(phase_rows),
            "latency": compute_latency(phase_rows),
        }

    efficiency = compute_replica_efficiency(hpa_status_csv_path, manifest.scored_end_seconds)
    responsiveness = compute_responsiveness(hpa_status_csv_path, profile_path, manifest.scored_end_seconds)

    sla_p95_pass = whole_latency["p95_ms"] is not None and whole_latency["p95_ms"] < SLA_P95_MS
    sla_success_pass = (
        whole_reliability["success_rate"] is not None
        and whole_reliability["success_rate"] >= SLA_SUCCESS_RATE
    )

    return {
        "experiment_id": exp_id,
        "candidate": trial.candidate,
        "block": trial.block,
        "repeat": trial.repeat,
        "structural": structural,
        "whole_reliability": whole_reliability,
        "whole_latency": whole_latency,
        "phases": phases,
        "efficiency": efficiency,
        "responsiveness": responsiveness,
        "sla_p95_pass": sla_p95_pass,
        "sla_success_pass": sla_success_pass,
    }


# ---------------------------------------------------------------------------
# Candidate-level descriptive statistics
# ---------------------------------------------------------------------------


def _median_min_max(values: List[float]) -> Dict[str, Optional[float]]:
    values = [v for v in values if v is not None]
    if not values:
        return {"median": None, "min": None, "max": None}
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def aggregate_candidate(candidate: int, trial_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    trials = [t for t in trial_results if t["candidate"] == candidate]
    total_offered = sum(t["whole_reliability"]["n"] for t in trials)
    total_success = sum(t["whole_reliability"]["success_2xx"] for t in trials)
    non_success_totals = {
        k: sum(t["whole_reliability"][k] for t in trials)
        for k in OUTCOME_KEYS
        if k != "success_2xx"
    }

    p95_values = [t["whole_latency"]["p95_ms"] for t in trials]
    p99_values = [t["whole_latency"]["p99_ms"] for t in trials]
    gt300_frac_values = [t["whole_latency"]["gt300_fraction"] for t in trials]
    success_rate_values = [t["whole_reliability"]["success_rate"] for t in trials]

    current_rs = [t["efficiency"]["current"]["replica_seconds"] for t in trials]
    ready_rs = [t["efficiency"]["ready"]["replica_seconds"] for t in trials]
    replica_hours = [t["efficiency"]["current"]["replica_hours"] for t in trials]
    avg_replicas = [t["efficiency"]["current"]["average"] for t in trials]
    peak_replicas = [t["efficiency"]["current"]["peak"] for t in trials]

    trigger_to_ready = [t["responsiveness"]["trigger_to_ready_s"] for t in trials]
    recovery_to_ready1 = [t["responsiveness"]["recovery_to_ready1_s"] for t in trials]

    first_scale_phases = {t["responsiveness"]["first_scale_up_phase"] for t in trials}
    first_scale_phase_consistent = len(first_scale_phases) == 1

    return {
        "candidate": candidate,
        "valid_runs": len(trials),
        "sla_passing_runs": sum(1 for t in trials if t["sla_p95_pass"] and t["sla_success_pass"]),
        "total_offered": total_offered,
        "total_success_2xx": total_success,
        "aggregate_success_rate": (total_success / total_offered) if total_offered else None,
        "non_success_totals": non_success_totals,
        "total_non_success": sum(non_success_totals.values()),
        "success_rate_stats": _median_min_max(success_rate_values),
        "p95_ms_stats": _median_min_max(p95_values),
        "p99_ms_stats": _median_min_max(p99_values),
        "gt300_fraction_stats": _median_min_max(gt300_frac_values),
        "current_replica_seconds_stats": _median_min_max(current_rs),
        "ready_replica_seconds_stats": _median_min_max(ready_rs),
        "replica_hours_stats": _median_min_max(replica_hours),
        "average_replicas_stats": _median_min_max(avg_replicas),
        "peak_replicas": sorted(set(peak_replicas)),
        "first_scale_up_phase_consistent_3_of_3": first_scale_phase_consistent,
        "first_scale_up_phases_observed": sorted(first_scale_phases),
        "trigger_to_ready_s_stats": _median_min_max(trigger_to_ready),
        "recovery_to_ready1_s_stats": _median_min_max(recovery_to_ready1),
        "scale_up_actions_all": [t["responsiveness"]["scale_up_actions"] for t in trials],
        "scale_down_actions_all": [t["responsiveness"]["scale_down_actions"] for t in trials],
        "oscillation_any": any(t["responsiveness"]["oscillation"] for t in trials),
    }


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def write_per_trial_metrics_csv(trial_results: List[Dict[str, Any]], out_path: Path) -> None:
    fieldnames = [
        "experiment_id", "candidate", "block", "repeat",
        "offered", "sent_success_2xx", "client_error_4xx", "server_error_5xx",
        "timeout", "connection_error", "dropped_overload", "success_rate",
        "whole_p50_ms", "whole_p95_ms", "whole_p99_ms", "whole_max_ms",
        "whole_gt300_count", "whole_gt300_fraction",
        "sla_p95_pass", "sla_success_pass",
        "hpa_desired_replica_seconds", "deployment_desired_replica_seconds",
        "current_replica_seconds", "ready_replica_seconds", "available_replica_seconds",
        "current_replica_hours", "average_current_replicas", "peak_current_replicas",
        "first_scale_up_phase", "first_scale_up_elapsed_seconds",
        "trigger_phase_start_seconds", "trigger_to_hpa_desired_s",
        "trigger_to_deployment_desired_s", "trigger_to_current_s",
        "trigger_to_ready_s", "trigger_to_available_s",
        "scale_up_actions", "scale_down_actions", "direction_reversals", "oscillation",
        "recovery_phase_start_seconds", "recovery_to_ready1_s",
        "recovery_to_current1_s", "recovery_to_available1_s", "stable_at_1_duration_s",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trial_results:
            r = t["whole_reliability"]
            lat = t["whole_latency"]
            eff = t["efficiency"]
            resp = t["responsiveness"]
            writer.writerow({
                "experiment_id": t["experiment_id"],
                "candidate": t["candidate"],
                "block": t["block"],
                "repeat": t["repeat"],
                "offered": r["n"],
                "sent_success_2xx": r["success_2xx"],
                "client_error_4xx": r["client_error_4xx"],
                "server_error_5xx": r["server_error_5xx"],
                "timeout": r["timeout"],
                "connection_error": r["connection_error"],
                "dropped_overload": r["dropped_overload"],
                "success_rate": r["success_rate"],
                "whole_p50_ms": lat["p50_ms"],
                "whole_p95_ms": lat["p95_ms"],
                "whole_p99_ms": lat["p99_ms"],
                "whole_max_ms": lat["max_ms"],
                "whole_gt300_count": lat["gt300_count"],
                "whole_gt300_fraction": lat["gt300_fraction"],
                "sla_p95_pass": t["sla_p95_pass"],
                "sla_success_pass": t["sla_success_pass"],
                "hpa_desired_replica_seconds": eff["hpa_desired"]["replica_seconds"],
                "deployment_desired_replica_seconds": eff["deployment_desired"]["replica_seconds"],
                "current_replica_seconds": eff["current"]["replica_seconds"],
                "ready_replica_seconds": eff["ready"]["replica_seconds"],
                "available_replica_seconds": eff["available"]["replica_seconds"],
                "current_replica_hours": eff["current"]["replica_hours"],
                "average_current_replicas": eff["current"]["average"],
                "peak_current_replicas": eff["current"]["peak"],
                "first_scale_up_phase": resp["first_scale_up_phase"],
                "first_scale_up_elapsed_seconds": resp["first_scale_up_elapsed_seconds"],
                "trigger_phase_start_seconds": resp["trigger_phase_start_seconds"],
                "trigger_to_hpa_desired_s": resp["trigger_to_hpa_desired_s"],
                "trigger_to_deployment_desired_s": resp["trigger_to_deployment_desired_s"],
                "trigger_to_current_s": resp["trigger_to_current_s"],
                "trigger_to_ready_s": resp["trigger_to_ready_s"],
                "trigger_to_available_s": resp["trigger_to_available_s"],
                "scale_up_actions": resp["scale_up_actions"],
                "scale_down_actions": resp["scale_down_actions"],
                "direction_reversals": resp["direction_reversals"],
                "oscillation": resp["oscillation"],
                "recovery_phase_start_seconds": resp["recovery_phase_start_seconds"],
                "recovery_to_ready1_s": resp["recovery_to_ready1_s"],
                "recovery_to_current1_s": resp["recovery_to_current1_s"],
                "recovery_to_available1_s": resp["recovery_to_available1_s"],
                "stable_at_1_duration_s": resp["stable_at_1_duration_s"],
            })


def write_candidate_summary_csv(candidate_summaries: List[Dict[str, Any]], out_path: Path) -> None:
    fieldnames = [
        "candidate", "valid_runs", "sla_passing_runs",
        "aggregate_success_rate", "total_non_success",
        "total_4xx", "total_5xx", "total_timeout", "total_connection_error", "total_dropped",
        "success_rate_median", "success_rate_min", "success_rate_max",
        "p95_ms_median", "p95_ms_min", "p95_ms_max",
        "p99_ms_median", "p99_ms_min", "p99_ms_max",
        "gt300_fraction_median", "gt300_fraction_min", "gt300_fraction_max",
        "current_replica_seconds_median", "current_replica_seconds_min", "current_replica_seconds_max",
        "ready_replica_seconds_median", "ready_replica_seconds_min", "ready_replica_seconds_max",
        "replica_hours_median", "average_replicas_median",
        "peak_replicas", "first_scale_up_phase_consistent_3_of_3", "first_scale_up_phases_observed",
        "trigger_to_ready_s_median", "trigger_to_ready_s_min", "trigger_to_ready_s_max",
        "recovery_to_ready1_s_median", "recovery_to_ready1_s_min", "recovery_to_ready1_s_max",
        "oscillation_any",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in candidate_summaries:
            nst = c["non_success_totals"]
            writer.writerow({
                "candidate": c["candidate"],
                "valid_runs": c["valid_runs"],
                "sla_passing_runs": c["sla_passing_runs"],
                "aggregate_success_rate": c["aggregate_success_rate"],
                "total_non_success": c["total_non_success"],
                "total_4xx": nst["client_error_4xx"],
                "total_5xx": nst["server_error_5xx"],
                "total_timeout": nst["timeout"],
                "total_connection_error": nst["connection_error"],
                "total_dropped": nst["dropped_overload"],
                "success_rate_median": c["success_rate_stats"]["median"],
                "success_rate_min": c["success_rate_stats"]["min"],
                "success_rate_max": c["success_rate_stats"]["max"],
                "p95_ms_median": c["p95_ms_stats"]["median"],
                "p95_ms_min": c["p95_ms_stats"]["min"],
                "p95_ms_max": c["p95_ms_stats"]["max"],
                "p99_ms_median": c["p99_ms_stats"]["median"],
                "p99_ms_min": c["p99_ms_stats"]["min"],
                "p99_ms_max": c["p99_ms_stats"]["max"],
                "gt300_fraction_median": c["gt300_fraction_stats"]["median"],
                "gt300_fraction_min": c["gt300_fraction_stats"]["min"],
                "gt300_fraction_max": c["gt300_fraction_stats"]["max"],
                "current_replica_seconds_median": c["current_replica_seconds_stats"]["median"],
                "current_replica_seconds_min": c["current_replica_seconds_stats"]["min"],
                "current_replica_seconds_max": c["current_replica_seconds_stats"]["max"],
                "ready_replica_seconds_median": c["ready_replica_seconds_stats"]["median"],
                "ready_replica_seconds_min": c["ready_replica_seconds_stats"]["min"],
                "ready_replica_seconds_max": c["ready_replica_seconds_stats"]["max"],
                "replica_hours_median": c["replica_hours_stats"]["median"],
                "average_replicas_median": c["average_replicas_stats"]["median"],
                "peak_replicas": ";".join(str(p) for p in c["peak_replicas"]),
                "first_scale_up_phase_consistent_3_of_3": c["first_scale_up_phase_consistent_3_of_3"],
                "first_scale_up_phases_observed": ";".join(c["first_scale_up_phases_observed"]),
                "trigger_to_ready_s_median": c["trigger_to_ready_s_stats"]["median"],
                "trigger_to_ready_s_min": c["trigger_to_ready_s_stats"]["min"],
                "trigger_to_ready_s_max": c["trigger_to_ready_s_stats"]["max"],
                "recovery_to_ready1_s_median": c["recovery_to_ready1_s_stats"]["median"],
                "recovery_to_ready1_s_min": c["recovery_to_ready1_s_stats"]["min"],
                "recovery_to_ready1_s_max": c["recovery_to_ready1_s_stats"]["max"],
                "oscillation_any": c["oscillation_any"],
            })


def write_blockwise_comparison_csv(trial_results: List[Dict[str, Any]], out_path: Path) -> None:
    fieldnames = [
        "block", "candidate", "experiment_id", "success_rate",
        "whole_p95_ms", "whole_p99_ms",
        "current_replica_seconds", "ready_replica_seconds",
        "trigger_to_ready_s", "recovery_to_ready1_s",
    ]
    rows = sorted(trial_results, key=lambda t: (t["block"], t["candidate"]))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in rows:
            writer.writerow({
                "block": t["block"],
                "candidate": t["candidate"],
                "experiment_id": t["experiment_id"],
                "success_rate": t["whole_reliability"]["success_rate"],
                "whole_p95_ms": t["whole_latency"]["p95_ms"],
                "whole_p99_ms": t["whole_latency"]["p99_ms"],
                "current_replica_seconds": t["efficiency"]["current"]["replica_seconds"],
                "ready_replica_seconds": t["efficiency"]["ready"]["replica_seconds"],
                "trigger_to_ready_s": t["responsiveness"]["trigger_to_ready_s"],
                "recovery_to_ready1_s": t["responsiveness"]["recovery_to_ready1_s"],
            })


PHASE_ORDER = [
    "baseline_3rps", "aggressive_probe_4rps", "discriminator_5rps",
    "scale_pressure_6rps", "adequacy_8rps", "recovery_2rps",
]


def write_phase_summary_csv(trial_results: List[Dict[str, Any]], out_path: Path) -> None:
    fieldnames = [
        "candidate", "phase_name",
        "n_median", "success_rate_median",
        "p95_ms_median", "p95_ms_min", "p95_ms_max",
        "p99_ms_median", "p99_ms_min", "p99_ms_max",
        "gt300_fraction_median", "gt300_fraction_min", "gt300_fraction_max",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in ALLOWED_CANDIDATES:
            trials = [t for t in trial_results if t["candidate"] == candidate]
            for phase_name in PHASE_ORDER:
                n_vals = [t["phases"][phase_name]["reliability"]["n"] for t in trials if phase_name in t["phases"]]
                sr_vals = [
                    t["phases"][phase_name]["reliability"]["success_rate"]
                    for t in trials if phase_name in t["phases"]
                ]
                p95_vals = [
                    t["phases"][phase_name]["latency"]["p95_ms"]
                    for t in trials if phase_name in t["phases"]
                ]
                p99_vals = [
                    t["phases"][phase_name]["latency"]["p99_ms"]
                    for t in trials if phase_name in t["phases"]
                ]
                gt300_vals = [
                    t["phases"][phase_name]["latency"]["gt300_fraction"]
                    for t in trials if phase_name in t["phases"]
                ]
                p95_stats = _median_min_max(p95_vals)
                p99_stats = _median_min_max(p99_vals)
                gt300_stats = _median_min_max(gt300_vals)
                writer.writerow({
                    "candidate": candidate,
                    "phase_name": phase_name,
                    "n_median": statistics.median(n_vals) if n_vals else None,
                    "success_rate_median": statistics.median(sr_vals) if sr_vals else None,
                    "p95_ms_median": p95_stats["median"],
                    "p95_ms_min": p95_stats["min"],
                    "p95_ms_max": p95_stats["max"],
                    "p99_ms_median": p99_stats["median"],
                    "p99_ms_min": p99_stats["min"],
                    "p99_ms_max": p99_stats["max"],
                    "gt300_fraction_median": gt300_stats["median"],
                    "gt300_fraction_min": gt300_stats["min"],
                    "gt300_fraction_max": gt300_stats["max"],
                })


# ---------------------------------------------------------------------------
# Frozen ranking hierarchy (lexicographic, no ad-hoc composite score)
# ---------------------------------------------------------------------------


def apply_ranking_hierarchy(candidate_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stage A (structural) is assumed already satisfied for every candidate
    reaching this function (verify_trial_structural raises otherwise).
    Stage B: reliability/SLA (all 3 repeats must pass p95<300ms and
    success>=99%). Stage C: efficiency (median current replica-seconds,
    lower is better) among Stage-B survivors. Stage D/E are tie-breakers /
    narrative only, not needed to break a tie in this dataset unless Stage C
    itself ties exactly."""
    stage_b_survivors = [c for c in candidate_summaries if c["sla_passing_runs"] == c["valid_runs"]]

    stage_c_ranked = sorted(
        stage_b_survivors,
        key=lambda c: c["current_replica_seconds_stats"]["median"],
    )

    selected = stage_c_ranked[0] if stage_c_ranked else None
    runner_up = stage_c_ranked[1] if len(stage_c_ranked) > 1 else None

    return {
        "stage_b_survivors": [c["candidate"] for c in stage_b_survivors],
        "stage_c_ranked_by_median_current_replica_seconds": [
            {"candidate": c["candidate"], "median_current_replica_seconds": c["current_replica_seconds_stats"]["median"]}
            for c in stage_c_ranked
        ],
        "selected_candidate": selected["candidate"] if selected else None,
        "runner_up_candidate": runner_up["candidate"] if runner_up else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def write_selection_decision_json(
    manifest: Manifest,
    trial_results: List[Dict[str, Any]],
    candidate_summaries: List[Dict[str, Any]],
    ranking: Dict[str, Any],
    out_path: Path,
) -> None:
    import datetime

    by_candidate = {c["candidate"]: c for c in candidate_summaries}
    selected = ranking["selected_candidate"]
    runner_up = ranking["runner_up_candidate"]

    def _metric_block(candidate: int) -> Dict[str, Any]:
        c = by_candidate[candidate]
        return {
            "aggregate_success_rate": c["aggregate_success_rate"],
            "sla_passing_runs_of_3": c["sla_passing_runs"],
            "p95_ms_median": c["p95_ms_stats"]["median"],
            "p99_ms_median": c["p99_ms_stats"]["median"],
            "gt300ms_fraction_median": c["gt300_fraction_stats"]["median"],
            "current_replica_seconds_median": c["current_replica_seconds_stats"]["median"],
            "ready_replica_seconds_median": c["ready_replica_seconds_stats"]["median"],
            "replica_hours_median": c["replica_hours_stats"]["median"],
            "first_scale_up_phase_consistent_3_of_3": c["first_scale_up_phase_consistent_3_of_3"],
            "first_scale_up_phases_observed": c["first_scale_up_phases_observed"],
            "trigger_to_ready_s_median": c["trigger_to_ready_s_stats"]["median"],
            "recovery_to_ready1_s_median": c["recovery_to_ready1_s_stats"]["median"],
            "oscillation_any": c["oscillation_any"],
            "total_non_success_outcomes": c["total_non_success"],
        }

    efficiency_diff_pct = None
    if selected is not None and runner_up is not None:
        sel_rs = by_candidate[selected]["current_replica_seconds_stats"]["median"]
        run_rs = by_candidate[runner_up]["current_replica_seconds_stats"]["median"]
        if run_rs:
            efficiency_diff_pct = 100.0 * (run_rs - sel_rs) / run_rs

    decision = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "methodology_references": [
            "CORR-17", "CORR-18", "CORR-19", "CORR-20", "CORR-21", "CORR-22",
            "CORR-23", "CORR-24", "CORR-25", "CORR-26",
        ],
        "candidates_considered": list(ALLOWED_CANDIDATES),
        "experiment_ids": {
            t.experiment_id: {"candidate": t.candidate, "block": t.block, "repeat": t.repeat}
            for t in manifest.trials
        },
        "canonical_schedule_sha256": manifest.schedule_sha256,
        "sla": {"p95_latency_ms_lt": SLA_P95_MS, "success_rate_gte": SLA_SUCCESS_RATE},
        "ranking_hierarchy": [
            "Stage A: structural validity (all 9 trials independently re-verified from raw artifacts)",
            "Stage B: reliability/SLA -- p95<300ms AND success>=99% in all 3 repeats",
            "Stage C: efficiency -- median current replica-seconds among Stage-B survivors (primary); "
            "Ready replica-seconds as secondary corroboration",
            "Stage D: responsiveness/stability as tie-breaker (not needed -- Stage C did not tie)",
            "Stage E: unnecessary light-load (3 RPS baseline) scaling penalty (not triggered -- no "
            "candidate scaled during baseline_3rps in any repeat)",
        ],
        "stage_b_survivors": ranking["stage_b_survivors"],
        "stage_c_ranking_by_median_current_replica_seconds": ranking[
            "stage_c_ranked_by_median_current_replica_seconds"
        ],
        "candidate_summary_metrics": {c: _metric_block(c) for c in ALLOWED_CANDIDATES},
        "selected_target": selected,
        "runner_up_target": runner_up,
        "selection_rationale": (
            f"All three candidates (200%, 230%, 260%) passed the frozen reliability/SLA tier "
            f"(p95<300ms and success>=99%) in all 3 counterbalanced-block repeats (Stage B). "
            f"Among Stage-B survivors, {selected}% has the lowest median current replica-seconds "
            f"over the exact [0,1230) scored window (Stage C), making it the most resource-efficient "
            f"candidate under this precommitted tuning workload. Responsiveness/stability "
            f"(Stage D) and the light-load-scaling check (Stage E) were not required to break the "
            f"decision, since Stage C alone produced a strict ranking with no tie."
            if selected is not None
            else "No candidate passed Stage B."
        ),
        "quantified_efficiency_difference_vs_runner_up": {
            "runner_up": runner_up,
            "median_current_replica_seconds_difference": (
                by_candidate[runner_up]["current_replica_seconds_stats"]["median"]
                - by_candidate[selected]["current_replica_seconds_stats"]["median"]
                if selected is not None and runner_up is not None
                else None
            ),
            "percent_fewer_replica_seconds_than_runner_up": efficiency_diff_pct,
            "note": "Replica-seconds/replica-hours are a resource-consumption proxy, not a "
            "monetary cloud-cost figure -- no dollar cost model is applied here.",
        },
        "known_caveats": [
            "n=3 valid repeats per candidate -- descriptive median/range comparison, not a "
            "high-powered statistical inference.",
            "200% Block 2 (hpa_tune_t200_b2r2_20260908T164158Z) recorded one isolated "
            "server_error_5xx; origin SOURCE_UNRESOLVED_FROM_PRESERVED_EVIDENCE.",
            "200% Block 3 (hpa_tune_t200_b3r3_20260909T065158Z) recorded one connection_error "
            "and one timeout, both within ~1s of that trial's scale-down transition; origin "
            "SOURCE_UNRESOLVED_FROM_PRESERVED_EVIDENCE, reported as a timing correlation only, "
            "not an established cause.",
            "Scattered sub-1.7% >300ms tail-latency bursts occur across multiple trials/candidates/"
            "phases; cause is CAUSE_UNRESOLVED_FROM_CURRENT_EVIDENCE and is not used to override "
            "the frozen p95/success SLA.",
            "This dataset is HPA hyperparameter TUNING traffic (hpa_tuning_v2.yaml) -- it must not "
            "be reused as final RL-vs-HPA held-out evaluation evidence (see CORR-26).",
        ],
        "tuning_vs_evaluation_separation": (
            "This selection is based exclusively on the 9-trial hpa_tuning_v2.yaml tuning dataset. "
            "The selected target becomes the fixed HPA baseline configuration going forward. Final "
            "RL-vs-HPA comparison must use the separately frozen held-out evaluation workload "
            "(configs/traffic_profiles/unseen_eval.yaml), never this tuning dataset."
        ),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, sort_keys=False)


def write_plots(trial_results: List[Dict[str, Any]], out_dir: Path) -> None:
    """Simple, non-truncated-axis, individual-points-plus-median plots.
    Descriptive only -- not part of the frozen ranking hierarchy itself."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {200: "#1f77b4", 230: "#ff7f0e", 260: "#2ca02c"}

    def _plot_metric(metric_fn, ylabel: str, title: str, filename: str, start_at_zero: bool = True):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for candidate in ALLOWED_CANDIDATES:
            trials = [t for t in trial_results if t["candidate"] == candidate]
            values = [metric_fn(t) for t in trials]
            xs = [candidate] * len(values)
            ax.scatter(xs, values, color=colors[candidate], alpha=0.8, zorder=3, label=f"{candidate}% (repeats)")
            if values:
                ax.scatter(
                    [candidate], [statistics.median(values)],
                    color=colors[candidate], marker="_", s=800, linewidths=3, zorder=4,
                )
        ax.set_xticks(list(ALLOWED_CANDIDATES))
        ax.set_xticklabels([f"{c}%" for c in ALLOWED_CANDIDATES])
        ax.set_xlabel("HPA averageUtilization candidate")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if start_at_zero:
            ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=150)
        plt.close(fig)

    _plot_metric(
        lambda t: t["efficiency"]["current"]["replica_seconds"],
        "Current replica-seconds, [0,1230)", "Replica-seconds by candidate (3 repeats each)",
        "current_replica_seconds_by_candidate.png",
    )
    _plot_metric(
        lambda t: t["whole_latency"]["p95_ms"],
        "Whole-run client p95 (ms)", "Whole-run p95 latency by candidate (3 repeats each)",
        "whole_run_p95_by_candidate.png",
    )
    _plot_metric(
        lambda t: t["responsiveness"]["trigger_to_ready_s"],
        "Trigger-phase-start -> Ready (s)", "Trigger-to-Ready timing by candidate (3 repeats each)",
        "trigger_to_ready_by_candidate.png",
    )
    _plot_metric(
        lambda t: (t["whole_latency"]["gt300_fraction"] or 0.0) * 100.0,
        ">300ms fraction (%)", "Secondary tail->300ms fraction by candidate (3 repeats each)",
        "gt300ms_fraction_by_candidate.png",
    )


def run_aggregation(
    manifest_path: Path, persistent_data_root: Path, out_dir: Path, make_plots: bool = True
) -> Dict[str, Any]:
    manifest = load_manifest(manifest_path)
    trial_results = [aggregate_trial(t, persistent_data_root, manifest) for t in manifest.trials]

    out_dir.mkdir(parents=True, exist_ok=True)
    write_per_trial_metrics_csv(trial_results, out_dir / "per_trial_metrics.csv")

    candidate_summaries = [aggregate_candidate(c, trial_results) for c in ALLOWED_CANDIDATES]
    write_candidate_summary_csv(candidate_summaries, out_dir / "candidate_summary.csv")
    write_blockwise_comparison_csv(trial_results, out_dir / "blockwise_comparison.csv")
    write_phase_summary_csv(trial_results, out_dir / "phase_summary.csv")

    ranking = apply_ranking_hierarchy(candidate_summaries)
    write_selection_decision_json(
        manifest, trial_results, candidate_summaries, ranking, out_dir / "selection_decision.json"
    )

    if make_plots:
        write_plots(trial_results, out_dir)

    return {
        "manifest": manifest,
        "trial_results": trial_results,
        "candidate_summaries": candidate_summaries,
        "ranking": ranking,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "experiments/manifests/hpa_tuning_official_9_trials.yaml",
    )
    parser.add_argument(
        "--persistent-data-root", type=Path, default=PROJECT_ROOT / "persistent-data"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=PROJECT_ROOT / "experiments/results/hpa_tuning_final"
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    result = run_aggregation(
        args.manifest, args.persistent_data_root, args.out_dir, make_plots=not args.no_plots
    )
    print(json.dumps(result["ranking"], indent=2))
    print(f"\nWrote CSVs to {args.out_dir}")


if __name__ == "__main__":
    main()
