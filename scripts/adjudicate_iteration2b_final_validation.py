#!/usr/bin/env python3
"""Phase 16M: one-shot, offline scientific adjudication of the Iteration-2B
I2B_FINAL_UNSEEN_VALIDATION holdout against the frozen Phase16K predictions.

This script NEVER: generates traffic, mutates Kubernetes/HPA, fits or
refits any model, tunes any threshold, or changes support. It only reads
already-collected, already-frozen artifacts and computes the preregistered
scoring formula exactly as historically established (Phase16D/16E/16F),
extended (not reinvented) to the Iteration-2B fitting dataset and 16-cell
final-validation matrix.

Deterministic: reruns of this script against the same frozen inputs
produce byte-identical outputs.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_dynamics_calibration_slot as harness  # noqa: E402

TASK_OUT = Path("/Users/rakeshdandamudi/Capstone/task_outputs")
VAL_ROOT = PROJECT_ROOT / "persistent-data/dynamics_validation_iteration2b/final_unseen"
LEDGER_PATH = VAL_ROOT / "iteration2b_final_validation_ledger.json"
SEQUENCE_PATH = PROJECT_ROOT / "configs/traffic_profiles/rl_dynamics_calibration_iteration2b/iteration2b_final_validation_sequence.json"
MATRIX_PATH = TASK_OUT / "phase_16j3_iteration2b_final_validation_matrix.csv"
PREREGISTRATION_PATH = TASK_OUT / "phase_16k_final_validation_prediction_preregistration.csv"
FITTING_DATASET_PATH = TASK_OUT / "phase_16k_iteration2b_model_fitting_dataset.csv"
SUPPORT_CSV_PATH = TASK_OUT / "phase_16k_effective_direct_support.csv"
MODELS_DIR = PROJECT_ROOT / "persistent-data/models/dynamics_iteration2b"

TARGETS = ["request_rate_rps", "cpu_cores", "memory_bytes", "requests_in_progress",
           "latency_p95_seconds", "error_rate_rps"]
REGRESSION_TARGETS = ["request_rate_rps", "cpu_cores", "memory_bytes", "requests_in_progress",
                       "latency_p95_seconds"]
EPSILON = {"request_rate_rps": 0.01, "cpu_cores": 0.001, "latency_p95_seconds": 0.001,
           "memory_bytes": 1048576.0, "requests_in_progress": 0.01, "error_rate_rps": 0.01}

FULL_ACTION_SPACE_RANGE = (1, 8)


# ---------------------------------------------------------------------------
# Section 6: support predicate (reconstructed unchanged from the frozen
# Phase16K effective-direct-support freeze -- no new support geometry).
# ---------------------------------------------------------------------------
def load_support_intervals() -> Dict[int, Tuple[float, float]]:
    intervals = {}
    with open(SUPPORT_CSV_PATH) as f:
        for row in csv.DictReader(f):
            intervals[int(row["replicas"])] = (float(row["lower_offered_rps"]), float(row["upper_offered_rps"]))
    return intervals


SUPPORT_INTERVALS = load_support_intervals()


def support_supported(replicas: int, offered_rps: float) -> bool:
    if replicas < FULL_ACTION_SPACE_RANGE[0] or replicas > FULL_ACTION_SPACE_RANGE[1]:
        return False
    if replicas in SUPPORT_INTERVALS:
        lo, hi = SUPPORT_INTERVALS[replicas]
        return lo <= offered_rps <= hi
    return False


def support_guard_check() -> List[Dict[str, Any]]:
    guards = [(5, 40.0), (7, 56.0)]
    rows = []
    for rep, rps in guards:
        supported = support_supported(rep, rps)
        rows.append({
            "replicas": rep, "offered_rps": rps,
            "support_interval": SUPPORT_INTERVALS.get(rep),
            "result": "IN_SUPPORT" if supported else "DYNAMICS_OUT_OF_SUPPORT",
            "guard_pass": (not supported),
        })
    return rows


# ---------------------------------------------------------------------------
# Section 3: official-attempt inventory
# ---------------------------------------------------------------------------
def load_official_attempt_inventory() -> List[Dict[str, Any]]:
    ledger = json.loads(LEDGER_PATH.read_text())
    seq = json.loads(SEQUENCE_PATH.read_text())
    seq_by_idx = {e["campaign_sequence_index"]: e for e in seq}
    rows = []
    for idx in range(1, 33):
        slot = ledger["slots"][str(idx)]
        assert slot["state"] == "VALID", f"index {idx} not VALID: {slot['state']}"
        entry = seq_by_idx[idx]
        official_id = slot["valid_attempt_id"]
        assert official_id in slot["all_attempt_ids"]
        md = json.loads((VAL_ROOT / official_id / "metadata.json").read_text())
        rows.append({
            "validation_sequence_index": idx,
            "planned_slot_id": entry["planned_slot_id"],
            "cell_id": entry["cell_or_transition_id"],
            "repeat_number": entry["repeat_number"],
            "replicas": entry["replicas"],
            "offered_rps": entry["offered_rps"],
            "official_attempt_id": official_id,
            "official_attempt_number": md["execution_attempt_number"],
            "structural_status": md["execution_structural_validity"],
            "route_status": md["route_integrity"]["classification"],
        })
    return rows


def load_nonofficial_attempts(official_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ledger = json.loads(LEDGER_PATH.read_text())
    rows = []
    for idx in range(1, 33):
        slot = ledger["slots"][str(idx)]
        official_id = slot["valid_attempt_id"]
        for aid in slot["all_attempt_ids"]:
            if aid != official_id:
                rows.append({"validation_sequence_index": idx, "attempt_id": aid,
                             "reason": "STRUCTURALLY_INVALID_PRESERVED", "scientific_weight": 0})
    rows.append({"validation_sequence_index": 30, "attempt_id": "i2bval_r8_V1_repeat2_attempt1_20260926T144119Z",
                 "reason": "UNCLASSIFIED_CRASH_BEFORE_LEDGER_WRITE (pre-ledger, never in all_attempt_ids)",
                 "scientific_weight": 0})
    return rows


# ---------------------------------------------------------------------------
# Section 8-9: raw evidence + decision-extraction reproducibility
# ---------------------------------------------------------------------------
def decision_extraction_reproducibility(official_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    manifest = harness.load_manifest()
    decision_timestamps = manifest["steady_state_timing"]["decision_aligned_extraction_timestamps_relative_to_traffic_onset"]
    rows = []
    for r in official_rows:
        attempt_dir = VAL_ROOT / r["official_attempt_id"]
        stored = json.loads((attempt_dir / "analysis/decision_aligned_observations.json").read_text())
        metrics_rows = harness.load_metrics_csv(attempt_dir / "metrics/metrics.csv")
        readiness_rows = harness.load_pod_readiness_history(attempt_dir / "cluster/pod_readiness_history.csv")
        recomputed = harness.extract_decision_aligned_v2(
            metrics_rows, readiness_rows, decision_timestamps,
            staleness_threshold=30.0, intended_replicas=int(r["replicas"]),
        )
        rows.append({
            "validation_sequence_index": r["validation_sequence_index"],
            "attempt_id": r["official_attempt_id"],
            "recomputed_equals_stored": recomputed == stored,
        })
    return rows


def flatten_decision_points(official_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Section 8 output: one row per (attempt, decision timestamp, target)."""
    rows = []
    for r in official_rows:
        attempt_dir = VAL_ROOT / r["official_attempt_id"]
        stored = json.loads((attempt_dir / "analysis/decision_aligned_observations.json").read_text())
        for dp in stored["decision_points"]:
            observable = dp["decision_point_observable"]
            for t in TARGETS:
                val = dp.get(t)
                present = val is not None
                scorable = observable and present and dp.get("steady_state_fit_eligible", False)
                exclusion = "NONE" if scorable else (
                    "DECISION_POINT_NOT_OBSERVABLE" if not observable else
                    ("TARGET_VALUE_MISSING" if not present else dp.get("exclusion_reason", "NOT_ELIGIBLE"))
                )
                rows.append({
                    "validation_sequence_index": r["validation_sequence_index"],
                    "attempt_id": r["official_attempt_id"], "cell_id": r["cell_id"],
                    "repeat_number": r["repeat_number"], "replicas": r["replicas"], "offered_rps": r["offered_rps"],
                    "decision_point_seconds": dp["target_decision_timestamp_s"],
                    "observable": observable, "target": t,
                    "target_value": val, "target_present": present, "target_scorable": scorable,
                    "exclusion_reason": exclusion,
                })
    return rows


# ---------------------------------------------------------------------------
# Section 11: prediction reproducibility (recompute from frozen model params)
# ---------------------------------------------------------------------------
def degree_basis(offered_rps: float, replicas: int, degree: int) -> List[float]:
    if degree == 1:
        return [1.0, offered_rps, replicas]
    return [1.0, offered_rps, replicas, offered_rps * replicas, offered_rps ** 2, replicas ** 2]


def interp_predict(anchors_by_rep: Dict[int, Dict[float, float]], replicas: int, offered_rps: float) -> Optional[float]:
    if replicas not in anchors_by_rep or len(anchors_by_rep[replicas]) == 0:
        return None
    levels = sorted(anchors_by_rep[replicas].items())
    exact = [v for r, v in levels if abs(r - offered_rps) < 1e-9]
    if exact:
        return exact[0]
    lower = [(r, v) for r, v in levels if r < offered_rps]
    upper = [(r, v) for r, v in levels if r > offered_rps]
    if not lower or not upper:
        return None
    lo_r, lo_v = lower[-1]
    hi_r, hi_v = upper[0]
    frac = (offered_rps - lo_r) / (hi_r - lo_r)
    return lo_v + frac * (hi_v - lo_v)


def load_model_predict_fn(target: str):
    params = json.loads((MODELS_DIR / target / "model_parameters.json").read_text())
    family = params["family"]
    if family == "piecewise_linear_regression_surface":
        coeffs = params["coefficients"]
        degree = params["degree"]
        return lambda rep, rps: float(sum(c * b for c, b in zip(coeffs, degree_basis(rps, rep, degree))))
    if family == "piecewise_linear_regression_logspace":
        coeffs = params["coefficients"]
        degree = params["degree"]
        def predict(rep, rps):
            v = sum(c * b for c, b in zip(coeffs, degree_basis(rps, rep, degree)))
            return math.expm1(v)
        return predict
    if family in ("monotone_interpolation_lookup", "threshold_regime_model"):
        anchors_by_cell = params["anchors_by_cell"]
        cell_coords = params["cell_coordinates"]
        anchors_by_rep: Dict[int, Dict[float, float]] = defaultdict(dict)
        for cell, val in anchors_by_cell.items():
            rep, rps = cell_coords[cell]
            anchors_by_rep[rep][rps] = val
        return lambda rep, rps: interp_predict(anchors_by_rep, rep, rps)
    raise ValueError(f"unrecognized model family for reproducibility: {family}")


def prediction_reproducibility_audit(matrix_rows: List[Dict[str, Any]], preregistration_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    predict_fns = {t: load_model_predict_fn(t) for t in TARGETS}
    prereg_by_key = {(int(r["replicas"]), r["label"]): r for r in preregistration_rows}
    rows = []
    for m in matrix_rows:
        key = (int(m["replicas"]), m["label"])
        prereg = prereg_by_key[key]
        for t in TARGETS:
            recomputed = predict_fns[t](int(m["replicas"]), float(m["offered_rps"]))
            preregistered = float(prereg[f"{t}_pred"])
            diff = abs(recomputed - preregistered) if recomputed is not None else None
            rows.append({
                "replicas": m["replicas"], "label": m["label"], "target": t,
                "preregistered_prediction": preregistered, "recomputed_prediction": recomputed,
                "absolute_difference": diff,
                "match_within_tolerance": (diff is not None and diff < 1e-6),
            })
    return rows


# ---------------------------------------------------------------------------
# Section 13-14: normalization + cell scoring
# ---------------------------------------------------------------------------
def fit_space_value(x: float, target: str) -> float:
    return math.log1p(x) if target == "latency_p95_seconds" else x


def compute_pooled_fitting_sd(target: str) -> Optional[float]:
    """Recomputed from the 223-row Iteration-2B FITTING dataset (never from
    final-validation outcomes) -- identical formula to Phase16F's
    frozen_pooled_fitting_sd(), extended to the expanded fitting dataset."""
    fit_tr = math.log1p if target == "latency_p95_seconds" else None
    repeat_vals: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    with open(FITTING_DATASET_PATH) as f:
        for row in csv.DictReader(f):
            v = float(row[target])
            if fit_tr:
                v = fit_tr(v)
            repeat_vals[row["cell_id"]][row["repeat_number"]].append(v)
    repeat_means = {c: {k: sum(v) / len(v) for k, v in reps.items()} for c, reps in repeat_vals.items()}
    cell_var_dof = []
    for c, reps in repeat_means.items():
        vals = list(reps.values())
        n = len(vals)
        if n >= 2:
            m = sum(vals) / n
            var = sum((x - m) ** 2 for x in vals) / (n - 1)
            cell_var_dof.append((n - 1, var))
    den = sum(w for w, _ in cell_var_dof)
    return math.sqrt(sum(w * v for w, v in cell_var_dof) / den) if den > 0 else None


def within_cell_repeat_sd(pooled_points: List[Tuple[str, float]], target: str) -> Optional[float]:
    fit_tr = math.log1p if target == "latency_p95_seconds" else (lambda x: x)
    per_repeat = defaultdict(list)
    for rp, v in pooled_points:
        per_repeat[rp].append(fit_tr(v))
    repeat_means = [sum(v) / len(v) for v in per_repeat.values() if v]
    if len(repeat_means) < 2:
        return None
    m = sum(repeat_means) / len(repeat_means)
    var = sum((x - m) ** 2 for x in repeat_means) / (len(repeat_means) - 1)
    return math.sqrt(var)


def compute_cell_scores(decision_point_rows: List[Dict[str, Any]], matrix_rows: List[Dict[str, Any]],
                          preregistration_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[float]]]:
    prereg_by_key = {(int(r["replicas"]), r["label"]): r for r in preregistration_rows}
    cell_key_by_id = {}
    for m in matrix_rows:
        cell_id = f"i2bval_r{m['replicas']}_{m['label']}"
        cell_key_by_id[cell_id] = (int(m["replicas"]), m["label"])

    by_cell_target: Dict[str, Dict[str, List[Tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in decision_point_rows:
        if row["target_scorable"]:
            by_cell_target[row["cell_id"]][row["target"]].append((str(row["repeat_number"]), float(row["target_value"])))

    pooled_sd = {t: compute_pooled_fitting_sd(t) for t in REGRESSION_TARGETS}

    rows = []
    for cell_id, key in sorted(cell_key_by_id.items()):
        rep, label = key
        prereg = prereg_by_key[key]
        for t in TARGETS:
            pts = by_cell_target[cell_id][t]
            n1 = sum(1 for rp, v in pts if rp == "1")
            n2 = sum(1 for rp, v in pts if rp == "2")
            n_total = len(pts)
            observed_mean = (sum(v for _, v in pts) / n_total) if n_total > 0 else None
            pred = float(prereg[f"{t}_pred"])
            if t not in REGRESSION_TARGETS:
                # error_rate_rps handled separately (Section 17), still recorded here for completeness
                rows.append({
                    "target": t, "cell_id": cell_id, "replicas": rep, "label": label,
                    "offered_rps": prereg["offered_rps"], "predicted_cell_mean": pred,
                    "observed_cell_mean": observed_mean, "repeat1_scorable_points": n1,
                    "repeat2_scorable_points": n2, "total_scorable_points": n_total,
                    "within_cell_repeat_sd": None, "pooled_fitting_sd": None, "epsilon": EPSILON[t],
                    "s_eff": None, "absolute_error": (abs(pred - observed_mean) if observed_mean is not None else None),
                    "normalized_error": None,
                })
                continue
            if observed_mean is None:
                rows.append({
                    "target": t, "cell_id": cell_id, "replicas": rep, "label": label,
                    "offered_rps": prereg["offered_rps"], "predicted_cell_mean": pred,
                    "observed_cell_mean": None, "repeat1_scorable_points": n1,
                    "repeat2_scorable_points": n2, "total_scorable_points": 0,
                    "within_cell_repeat_sd": None, "pooled_fitting_sd": pooled_sd[t], "epsilon": EPSILON[t],
                    "s_eff": None, "absolute_error": None, "normalized_error": None,
                })
                continue
            wsd = within_cell_repeat_sd(pts, t)
            candidates = []
            if wsd is not None:
                candidates.append(wsd)
            if pooled_sd[t] is not None:
                candidates.append(0.5 * pooled_sd[t])
            candidates.append(EPSILON[t])
            s_eff = max(candidates)
            fp = fit_space_value(pred, t)
            fo = fit_space_value(observed_mean, t)
            ne = abs(fp - fo) / s_eff
            abs_err = abs(pred - observed_mean)
            rows.append({
                "target": t, "cell_id": cell_id, "replicas": rep, "label": label,
                "offered_rps": prereg["offered_rps"], "predicted_cell_mean": pred,
                "observed_cell_mean": observed_mean, "repeat1_scorable_points": n1,
                "repeat2_scorable_points": n2, "total_scorable_points": n_total,
                "within_cell_repeat_sd": wsd, "pooled_fitting_sd": pooled_sd[t], "epsilon": EPSILON[t],
                "s_eff": s_eff, "absolute_error": abs_err, "normalized_error": ne,
            })
    return rows, pooled_sd


# ---------------------------------------------------------------------------
# Section 15: evidence adequacy gate
# ---------------------------------------------------------------------------
def evidence_adequacy(cell_score_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for t in TARGETS:
        target_rows = [r for r in cell_score_rows if r["target"] == t]
        scorable_cells = [r for r in target_rows if r["total_scorable_points"] > 0]
        n_scorable = len(scorable_cells)
        replicas_covered = sorted(set(int(r["replicas"]) for r in scorable_cells))
        replica_coverage_ok = all(rep in replicas_covered for rep in range(1, 9))
        adequate = (n_scorable >= 12) and replica_coverage_ok
        rows.append({
            "target": t, "n_scorable_cells": n_scorable, "n_total_cells": len(target_rows),
            "replicas_covered": replicas_covered, "all_8_replicas_covered": replica_coverage_ok,
            "evidence_adequate": adequate,
            "result": "ADEQUATE" if adequate else "ITERATION2B_FINAL_VALIDATION_EVIDENCE_INSUFFICIENT",
        })
    return rows


# ---------------------------------------------------------------------------
# Section 16: regression gates
# ---------------------------------------------------------------------------
def regression_gates(cell_score_rows: List[Dict[str, Any]], evidence_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    evidence_by_target = {r["target"]: r for r in evidence_rows}
    rows = []
    for t in REGRESSION_TARGETS:
        scorable = [r for r in cell_score_rows if r["target"] == t and r["normalized_error"] is not None]
        if not evidence_by_target[t]["evidence_adequate"]:
            rows.append({"target": t, "status": "INSUFFICIENT_EVIDENCE", "n_scorable_cells": len(scorable),
                         "median_normalized_error": None, "fraction_at_or_below_2_0": None,
                         "max_normalized_error": None, "gate_median_le_2_0": None,
                         "gate_fraction_ge_0_90": None, "gate_max_le_4_0": None})
            continue
        nes = sorted(r["normalized_error"] for r in scorable)
        n = len(nes)
        median = nes[n // 2] if n % 2 == 1 else (nes[n // 2 - 1] + nes[n // 2]) / 2
        frac_le_2 = sum(1 for x in nes if x <= 2.0) / n
        worst = max(nes)
        gate1 = median <= 2.0
        gate2 = frac_le_2 >= 0.90
        gate3 = worst <= 4.0
        passed = gate1 and gate2 and gate3
        rows.append({
            "target": t, "status": "PASS" if passed else "FAIL", "n_scorable_cells": n,
            "median_normalized_error": median, "fraction_at_or_below_2_0": frac_le_2,
            "max_normalized_error": worst, "gate_median_le_2_0": gate1,
            "gate_fraction_ge_0_90": gate2, "gate_max_le_4_0": gate3,
        })
    return rows


# ---------------------------------------------------------------------------
# Section 17: error-rate target adjudication
# ---------------------------------------------------------------------------
def error_target_adjudication(decision_point_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scorable_error_obs = [float(r["target_value"]) for r in decision_point_rows
                           if r["target"] == "error_rate_rps" and r["target_scorable"]]
    n_total = len(scorable_error_obs)
    n_nonzero = sum(1 for v in scorable_error_obs if v > 0.0)
    if n_nonzero == 0:
        return {
            "n_scorable_observations": n_total, "n_nonzero_observations": 0,
            "label": "ZERO_ONLY_WITHIN_OBSERVED_CALIBRATION_DOMAIN",
            "balanced_accuracy": "NOT_ESTIMABLE", "nonzero_class_recall": "NOT_ESTIMABLE",
            "status": "NOT_ESTIMABLE",
        }
    # Frozen preregistered gate (only reached if nonzero observations exist).
    # All preregistered error_rate_rps predictions are 0.0 (see the prediction
    # file), so the model's predicted class is always "zero" -- tp (true
    # positive: predicted nonzero, observed nonzero) is therefore always 0.
    tp = 0
    tn = sum(1 for v in scorable_error_obs if v == 0.0)  # predicted 0, observed 0
    fn = n_nonzero  # predicted 0, observed nonzero
    fp = 0
    balanced_accuracy = 0.5 * ((tp / (tp + fn) if (tp + fn) else 0.0) + (tn / (tn + fp) if (tn + fp) else 0.0))
    nonzero_recall = tp / n_nonzero if n_nonzero else 0.0
    return {
        "n_scorable_observations": n_total, "n_nonzero_observations": n_nonzero,
        "label": "NONZERO_OBSERVED", "balanced_accuracy": balanced_accuracy,
        "nonzero_class_recall": nonzero_recall,
        "status": "PASS" if (balanced_accuracy >= 0.80 and nonzero_recall >= 0.80) else "FAIL",
    }


# ---------------------------------------------------------------------------
# Section 19: joint/physical consistency
# ---------------------------------------------------------------------------
def joint_consistency_audit(decision_point_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    by_key: Dict[Tuple, Dict[str, float]] = defaultdict(dict)
    for r in decision_point_rows:
        if r["target_scorable"]:
            key = (r["validation_sequence_index"], r["decision_point_seconds"])
            by_key[key][r["target"]] = float(r["target_value"])
    for key, vals in sorted(by_key.items()):
        idx, dp_sec = key
        checks = {}
        if "error_rate_rps" in vals:
            checks["error_rate_nonnegative"] = vals["error_rate_rps"] >= 0.0
            if "request_rate_rps" in vals:
                checks["error_rate_le_request_rate_plus_tolerance"] = vals["error_rate_rps"] <= vals["request_rate_rps"] + 1e-6
        for t in ("cpu_cores", "memory_bytes", "requests_in_progress", "latency_p95_seconds", "request_rate_rps"):
            if t in vals:
                checks[f"{t}_nonnegative"] = vals[t] >= 0.0
        rows.append({
            "validation_sequence_index": idx, "decision_point_seconds": dp_sec,
            "checks": json.dumps(checks), "all_pass": all(checks.values()),
        })
    return rows


def main():
    result: Dict[str, Any] = {}

    result["support_guard"] = support_guard_check()
    guard_ok = all(g["guard_pass"] for g in result["support_guard"])
    result["support_guard_all_pass"] = guard_ok
    if not guard_ok:
        result["verdict"] = "ITERATION2B_SUPPORT_GUARD_FAILED"
        return result

    official_rows = load_official_attempt_inventory()
    result["official_attempt_inventory"] = official_rows
    result["nonofficial_attempts"] = load_nonofficial_attempts(official_rows)

    result["decision_extraction_reproducibility"] = decision_extraction_reproducibility(official_rows)
    extraction_mismatches = [r for r in result["decision_extraction_reproducibility"] if not r["recomputed_equals_stored"]]
    if extraction_mismatches:
        result["verdict"] = "EVIDENCE_REPRODUCIBILITY_DEFECT"
        result["extraction_mismatches"] = extraction_mismatches
        return result

    decision_point_rows = flatten_decision_points(official_rows)
    result["decision_point_rows"] = decision_point_rows

    matrix_rows = list(csv.DictReader(open(MATRIX_PATH)))
    preregistration_rows = list(csv.DictReader(open(PREREGISTRATION_PATH)))
    result["prediction_reproducibility"] = prediction_reproducibility_audit(matrix_rows, preregistration_rows)
    pred_mismatches = [r for r in result["prediction_reproducibility"] if not r["match_within_tolerance"]]
    if pred_mismatches:
        result["verdict"] = "PREDICTION_REPRODUCIBILITY_DEFECT"
        result["prediction_mismatches"] = pred_mismatches
        return result

    cell_score_rows, pooled_sd = compute_cell_scores(decision_point_rows, matrix_rows, preregistration_rows)
    result["cell_score_rows"] = cell_score_rows
    result["pooled_fitting_sd"] = pooled_sd

    evidence_rows = evidence_adequacy(cell_score_rows)
    result["evidence_adequacy"] = evidence_rows

    regression_rows = regression_gates(cell_score_rows, evidence_rows)
    result["regression_gates"] = regression_rows

    error_result = error_target_adjudication(decision_point_rows)
    result["error_target"] = error_result

    joint_rows = joint_consistency_audit(decision_point_rows)
    result["joint_consistency"] = joint_rows
    joint_all_pass = all(r["all_pass"] for r in joint_rows)
    result["joint_consistency_all_pass"] = joint_all_pass

    any_insufficient = any(r["status"] == "INSUFFICIENT_EVIDENCE" for r in regression_rows)
    any_fail = any(r["status"] == "FAIL" for r in regression_rows) or error_result["status"] == "FAIL"
    all_pass = all(r["status"] == "PASS" for r in regression_rows) and error_result["status"] in ("PASS", "NOT_ESTIMABLE")

    if any_insufficient:
        result["verdict"] = "ITERATION2B_FINAL_VALIDATION_EVIDENCE_INSUFFICIENT"
    elif any_fail or not joint_all_pass:
        result["verdict"] = "ITERATION2B_FINAL_UNSEEN_VALIDATION_FAILED"
    elif all_pass:
        result["verdict"] = "ITERATION2B_FINAL_UNSEEN_VALIDATION_PASSED"
    else:
        result["verdict"] = "ITERATION2B_FINAL_UNSEEN_VALIDATION_AMBIGUOUS"

    return result


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: v for k, v in r.items() if k not in (
        "official_attempt_inventory", "nonofficial_attempts", "decision_point_rows",
        "cell_score_rows", "prediction_reproducibility", "decision_extraction_reproducibility",
        "joint_consistency",
    )}, indent=2, default=str))
