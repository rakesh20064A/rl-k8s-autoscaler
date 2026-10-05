#!/usr/bin/env python3
"""scripts/adjudicate_iteration3_final_validation.py

Phase 16S: one-shot OFFLINE scientific scoring/adjudication of the
completed Iteration-3 final-unseen-validation campaign (32/32 official
VALID attempts, Phase16Q). Deterministic, reads only already-frozen
artifacts -- never fits/refits, never sends traffic, never touches
Kubernetes. Reuses the exact frozen s_eff/normalized-error formula and
regression/error-target gates from Phase16D/16D.1 (identical to
scripts/adjudicate_iteration2b_final_validation.py's own implementation,
verified there and reused unchanged here), and the exact frozen
prediction-reconstruction logic from
scripts/fit_iteration3_dynamics_models.py (reconstruct_predict_from_params).
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASK_OUTPUTS = Path("/Users/rakeshdandamudi/Capstone/task_outputs")

LEDGER_PATH = PROJECT_ROOT / "persistent-data/dynamics_validation_iteration3/final_unseen/iteration3_final_validation_ledger.json"
ATTEMPTS_ROOT = PROJECT_ROOT / "persistent-data/dynamics_validation_iteration3/final_unseen"
DEV_DATASET_PATH = TASK_OUTPUTS / "phase_16n_iteration3_development_dataset.csv"
FRESH_MATRIX_PATH = TASK_OUTPUTS / "phase_16n_iteration3_fresh_validation_matrix.csv"
PREREG_PATH = TASK_OUTPUTS / "phase_16o_iteration3_final_validation_prediction_preregistration.csv"
SUPPORT_PATH = TASK_OUTPUTS / "phase_16k_effective_direct_support.csv"

import sys  # noqa: E402
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.fit_iteration3_dynamics_models import reconstruct_predict_from_params  # noqa: E402

TARGETS = ["request_rate_rps", "cpu_cores", "latency_p95_seconds", "memory_bytes", "requests_in_progress", "error_rate_rps"]
REGRESSION_TARGETS = ["request_rate_rps", "cpu_cores", "latency_p95_seconds", "memory_bytes", "requests_in_progress"]
LOGSPACE_TARGETS = {"latency_p95_seconds"}

EPSILON = {"request_rate_rps": 0.01, "cpu_cores": 0.001, "latency_p95_seconds": 0.001,
           "memory_bytes": 1048576.0, "requests_in_progress": 0.01, "error_rate_rps": 0.01}

MODELS_DIR = PROJECT_ROOT / "persistent-data/models/dynamics_iteration3"


def fit_space(target: str, x: float) -> float:
    return math.log1p(x) if target in LOGSPACE_TARGETS else x


# ---------------------------------------------------------------------------
# Section 1: integrity gate
# ---------------------------------------------------------------------------
def integrity_gate() -> Dict[str, Any]:
    ledger = json.loads(LEDGER_PATH.read_text())
    slots = ledger["slots"]
    checks = {
        "n_slots_is_32": len(slots) == 32,
        "all_32_valid": all(v["state"] == "VALID" for v in slots.values()),
        "each_slot_exactly_one_attempt": all(len(v["all_attempt_ids"]) == 1 for v in slots.values()),
        "fitting_weight_zero": ledger["provenance"]["fitting_weight"] == 0,
    }
    return {"ledger": ledger, "checks": checks, "all_pass": all(checks.values())}


def official_attempts(ledger: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for idx, slot in ledger["slots"].items():
        attempt_id = slot["valid_attempt_id"]
        meta = json.loads((ATTEMPTS_ROOT / attempt_id / "metadata.json").read_text())
        rows.append({
            "sequence_index": int(idx), "attempt_id": attempt_id,
            "cell_id": meta["cell_or_transition_id"], "repeat_number": meta["repeat_number"],
            "replicas": meta["replicas"], "offered_rps": meta["offered_rps"],
        })
    return sorted(rows, key=lambda r: r["sequence_index"])


# ---------------------------------------------------------------------------
# Section 2: flatten decision points -> model_usable filter (Phase16D/16M-frozen)
# ---------------------------------------------------------------------------
def flatten_decision_points(attempts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for r in attempts:
        stored = json.loads((ATTEMPTS_ROOT / r["attempt_id"] / "analysis/decision_aligned_observations.json").read_text())
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
                    "sequence_index": r["sequence_index"], "attempt_id": r["attempt_id"], "cell_id": r["cell_id"],
                    "repeat_number": r["repeat_number"], "replicas": r["replicas"], "offered_rps": r["offered_rps"],
                    "decision_point_seconds": dp["target_decision_timestamp_s"],
                    "observable": observable, "target": t, "target_value": val,
                    "target_present": present, "target_scorable": scorable, "exclusion_reason": exclusion,
                })
    return rows


# ---------------------------------------------------------------------------
# Section 3-4: normalization (pooled SD from FROZEN 312-row dev dataset only)
# ---------------------------------------------------------------------------
def compute_pooled_fitting_sd(target: str) -> Optional[float]:
    repeat_vals: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    with open(DEV_DATASET_PATH) as f:
        for row in csv.DictReader(f):
            v = fit_space(target, float(row[target]))
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
    per_repeat = defaultdict(list)
    for rp, v in pooled_points:
        per_repeat[rp].append(fit_space(target, v))
    repeat_means = [sum(v) / len(v) for v in per_repeat.values() if v]
    if len(repeat_means) < 2:
        return None
    m = sum(repeat_means) / len(repeat_means)
    var = sum((x - m) ** 2 for x in repeat_means) / (len(repeat_means) - 1)
    return math.sqrt(var)


def s_eff(target: str, wsd: Optional[float], pooled_sd: Optional[float]) -> float:
    candidates = []
    if wsd is not None:
        candidates.append(wsd)
    if pooled_sd is not None:
        candidates.append(0.5 * pooled_sd)
    candidates.append(EPSILON[target])
    return max(candidates)


# ---------------------------------------------------------------------------
# Section 4: cell scoring
# ---------------------------------------------------------------------------
def load_model_predict_fns() -> Dict[str, Any]:
    fns = {}
    for t in TARGETS:
        params = json.loads((MODELS_DIR / t / "model_parameters.json").read_text())
        fns[t] = reconstruct_predict_from_params(params)
    return fns


def load_preregistration() -> Dict[Tuple[str, str], float]:
    out = {}
    with open(PREREG_PATH) as f:
        for row in csv.DictReader(f):
            out[(row["cell_id"], row["target"])] = float(row["prediction"])
    return out


def compute_cell_scores(dp_rows: List[Dict[str, Any]], matrix_rows: List[Dict[str, Any]],
                          prereg: Dict[Tuple[str, str], float]) -> List[Dict[str, Any]]:
    by_cell_target: Dict[str, Dict[str, List[Tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in dp_rows:
        if row["target_scorable"]:
            by_cell_target[row["cell_id"]][row["target"]].append((str(row["repeat_number"]), float(row["target_value"])))

    pooled_sd = {t: compute_pooled_fitting_sd(t) for t in REGRESSION_TARGETS}

    rows = []
    for m in sorted(matrix_rows, key=lambda r: (int(r["replicas"]), r["label"])):
        cell_id = m["cell_id"]
        for t in TARGETS:
            pts = by_cell_target[cell_id][t]
            n1 = sum(1 for rp, v in pts if rp == "1")
            n2 = sum(1 for rp, v in pts if rp == "2")
            n_total = len(pts)
            observed_mean = (sum(v for _, v in pts) / n_total) if n_total > 0 else None
            pred = prereg[(cell_id, t)]
            if t not in REGRESSION_TARGETS:
                rows.append({
                    "target": t, "cell_id": cell_id, "replicas": m["replicas"], "offered_rps": m["offered_rps"],
                    "predicted_cell_mean": pred, "observed_cell_mean": observed_mean,
                    "repeat1_scorable_points": n1, "repeat2_scorable_points": n2, "total_scorable_points": n_total,
                    "within_cell_repeat_sd": None, "pooled_fitting_sd": None, "epsilon": EPSILON[t],
                    "s_eff": None, "absolute_error": (abs(pred - observed_mean) if observed_mean is not None else None),
                    "normalized_error": None,
                })
                continue
            if observed_mean is None:
                rows.append({
                    "target": t, "cell_id": cell_id, "replicas": m["replicas"], "offered_rps": m["offered_rps"],
                    "predicted_cell_mean": pred, "observed_cell_mean": None,
                    "repeat1_scorable_points": n1, "repeat2_scorable_points": n2, "total_scorable_points": 0,
                    "within_cell_repeat_sd": None, "pooled_fitting_sd": pooled_sd[t], "epsilon": EPSILON[t],
                    "s_eff": None, "absolute_error": None, "normalized_error": None,
                })
                continue
            wsd = within_cell_repeat_sd(pts, t)
            se = s_eff(t, wsd, pooled_sd[t])
            fp, fo = fit_space(t, pred), fit_space(t, observed_mean)
            ne = abs(fp - fo) / se
            rows.append({
                "target": t, "cell_id": cell_id, "replicas": m["replicas"], "offered_rps": m["offered_rps"],
                "predicted_cell_mean": pred, "observed_cell_mean": observed_mean,
                "repeat1_scorable_points": n1, "repeat2_scorable_points": n2, "total_scorable_points": n_total,
                "within_cell_repeat_sd": wsd, "pooled_fitting_sd": pooled_sd[t], "epsilon": EPSILON[t],
                "s_eff": se, "absolute_error": abs(pred - observed_mean), "normalized_error": ne,
            })
    return rows


# ---------------------------------------------------------------------------
# Section 5-6: evidence adequacy, regression gates, error target, verdict
# ---------------------------------------------------------------------------
def evidence_adequacy(cell_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    for t in TARGETS:
        rows = [r for r in cell_rows if r["target"] == t]
        scorable = [r for r in rows if r["observed_cell_mean"] is not None]
        replicas_covered = {int(r["replicas"]) for r in scorable}
        adequate = len(scorable) >= 12 and all(rep in replicas_covered for rep in range(1, 9))
        out[t] = {"n_scorable_cells": len(scorable), "n_total_cells": len(rows),
                   "replicas_covered": sorted(replicas_covered), "adequate": adequate}
    return out


def regression_gates(cell_rows: List[Dict[str, Any]], adequacy: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for t in REGRESSION_TARGETS:
        if not adequacy[t]["adequate"]:
            out[t] = {"status": "INSUFFICIENT_EVIDENCE"}
            continue
        nes = sorted(r["normalized_error"] for r in cell_rows if r["target"] == t and r["normalized_error"] is not None)
        n = len(nes)
        median = nes[n // 2] if n % 2 == 1 else (nes[n // 2 - 1] + nes[n // 2]) / 2
        frac_le_2 = sum(1 for x in nes if x <= 2.0) / n
        worst = max(nes)
        passed = median <= 2.0 and frac_le_2 >= 0.90 and worst <= 4.0
        out[t] = {"status": "PASS" if passed else "FAIL", "median": median, "fraction_le_2": frac_le_2, "max": worst, "n": n}
    return out


def error_target_adjudication(dp_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scorable = [r for r in dp_rows if r["target"] == "error_rate_rps" and r["target_scorable"]]
    n = len(scorable)
    nonzero = sum(1 for r in scorable if float(r["target_value"]) > 0)
    if n == 0 or nonzero == 0:
        return {"status": "NOT_ESTIMABLE", "n_scorable_observations": n, "n_nonzero_observations": nonzero}
    return {"status": "ESTIMABLE_BUT_UNSCORED_SEE_REPORT", "n_scorable_observations": n, "n_nonzero_observations": nonzero}


def main() -> Dict[str, Any]:
    integrity = integrity_gate()
    assert integrity["all_pass"], f"INTEGRITY GATE FAILED: {integrity['checks']}"

    attempts = official_attempts(integrity["ledger"])
    assert len(attempts) == 32
    dp_rows = flatten_decision_points(attempts)
    assert len(dp_rows) == 32 * 4 * 6

    matrix_rows = list(csv.DictReader(open(FRESH_MATRIX_PATH)))
    assert len(matrix_rows) == 16
    prereg = load_preregistration()
    assert len(prereg) == 96

    cell_rows = compute_cell_scores(dp_rows, matrix_rows, prereg)
    adequacy = evidence_adequacy(cell_rows)
    gates = regression_gates(cell_rows, adequacy)
    error_adj = error_target_adjudication(dp_rows)

    targets_failed = [t for t, g in gates.items() if g["status"] == "FAIL"]
    targets_insufficient = [t for t, g in gates.items() if g["status"] == "INSUFFICIENT_EVIDENCE"]
    targets_passed = [t for t, g in gates.items() if g["status"] == "PASS"]

    if targets_failed:
        verdict = "ITERATION3_FINAL_UNSEEN_VALIDATION_FAILED"
        canonical = "EMPIRICAL_DYNAMICS_MODEL_VALIDATION_FAILED"
    elif targets_insufficient:
        verdict = "ITERATION3_FINAL_UNSEEN_VALIDATION_INSUFFICIENT_EVIDENCE"
        canonical = "EMPIRICAL_DYNAMICS_MODEL_VALIDATION_INSUFFICIENT_EVIDENCE"
    else:
        verdict = "ITERATION3_FINAL_UNSEEN_VALIDATION_PASSED"
        canonical = "EMPIRICAL_DYNAMICS_MODEL_VALIDATION_PASSED"

    return {
        "integrity": integrity, "attempts": attempts, "dp_rows": dp_rows, "cell_rows": cell_rows,
        "adequacy": adequacy, "gates": gates, "error_adjudication": error_adj,
        "targets_passed": targets_passed, "targets_failed": targets_failed, "targets_insufficient": targets_insufficient,
        "verdict": verdict, "canonical_failure_state": canonical if targets_failed or targets_insufficient else None,
    }


if __name__ == "__main__":
    result = main()
    print(json.dumps({"verdict": result["verdict"], "gates": result["gates"], "error_adjudication": result["error_adjudication"]}, indent=2, default=str))
