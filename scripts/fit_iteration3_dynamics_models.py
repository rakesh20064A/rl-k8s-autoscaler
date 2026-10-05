#!/usr/bin/env python3
"""scripts/fit_iteration3_dynamics_models.py

Phase 16O: offline, deterministic candidate-level CV coverage gate,
model-selection, final refit, joint-consistency audit, and fresh-holdout
prediction preregistration for the three Iteration-3 redevelopment targets
(cpu_cores, requests_in_progress, latency_p95_seconds).

Reuses the exact frozen Phase16D/16D.1/16D.2 candidate implementations,
metric formulas, s_eff normalization, lexicographic selection procedure,
and error<=request_rate joint-consistency check, applied to the new
312-row Iteration-3 development dataset and its frozen repeat-based CV
folds.

This script never opens an HTTP client, never touches subprocess or
Kubernetes, and never fits/refits the three preserved targets
(request_rate_rps, memory_bytes, error_rate_rps) -- those are carried
forward byte-identically from persistent-data/models/dynamics_iteration2b/.
"""
from __future__ import annotations

import csv
import datetime
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.tree import DecisionTreeRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASK_OUTPUTS = Path("/Users/rakeshdandamudi/Capstone/task_outputs")

DEV_DATASET_PATH = TASK_OUTPUTS / "phase_16n_iteration3_development_dataset.csv"
FOLD_ASSIGNMENT_PATH = TASK_OUTPUTS / "phase_16n_iteration3_cv_fold_assignment.csv"
FRESH_MATRIX_PATH = TASK_OUTPUTS / "phase_16n_iteration3_fresh_validation_matrix.csv"
SUPPORT_PATH = TASK_OUTPUTS / "phase_16k_effective_direct_support.csv"
CANDIDATE_REGISTRY_PATH = TASK_OUTPUTS / "phase_16d_candidate_model_registry.csv"

IT2B_MODELS_DIR = PROJECT_ROOT / "persistent-data/models/dynamics_iteration2b"
IT3_MODELS_DIR = PROJECT_ROOT / "persistent-data/models/dynamics_iteration3"

RANDOM_STATE = 1310581619

TARGETS_REDEVELOP = ["cpu_cores", "requests_in_progress", "latency_p95_seconds"]
TARGETS_FROZEN = ["request_rate_rps", "memory_bytes", "error_rate_rps"]
ALL_TARGETS = ["request_rate_rps", "cpu_cores", "latency_p95_seconds", "memory_bytes",
               "requests_in_progress", "error_rate_rps"]

LOGSPACE_TARGETS = {"latency_p95_seconds"}

EPSILON = {
    "request_rate_rps": 0.01,
    "cpu_cores": 0.001,
    "latency_p95_seconds": 0.001,   # log1p space
    "memory_bytes": 1048576.0,
    "requests_in_progress": 0.01,
    "error_rate_rps": 0.01,
}

CANDIDATE_FAMILIES_IN_SCOPE = {
    "cpu_cores": ["monotone_interpolation_lookup", "piecewise_linear_regression_surface", "tree_based_regressor"],
    "requests_in_progress": ["monotone_interpolation_lookup", "piecewise_linear_regression_surface",
                              "load_regime_conditioned_empirical_distribution"],
    "latency_p95_seconds": ["monotone_interpolation_lookup_logspace", "piecewise_linear_regression_logspace",
                             "tree_based_regressor"],
}

PRIMARY_METRIC = {"cpu_cores": "MAE_cores", "requests_in_progress": "MAE_count", "latency_p95_seconds": "MAE_log1p_space"}
SECONDARY_METRIC = {"cpu_cores": "MAE_per_replica_utilization_fraction", "requests_in_progress": None,
                     "latency_p95_seconds": None}

COMPLEXITY_RANK = {
    "per_replica_empirical_baseline": 1,
    "bounded_lookup_interpolation": 2,
    "monotone_interpolation_lookup": 3,
    "monotone_interpolation_lookup_logspace": 4,
    "threshold_regime_model": 5,
    "load_regime_conditioned_empirical_distribution": 6,
    "piecewise_linear_regression_surface": 7,
    "piecewise_linear_regression_logspace": 8,
    "tree_based_regressor": 9,
    "tree_based_regressor_or_classifier": 9,
}

MONOTONIC_ENFORCED_TARGETS = {"cpu_cores", "latency_p95_seconds", "requests_in_progress", "error_rate_rps", "memory_bytes"}


def fit_space(target: str, x: float) -> float:
    return math.log1p(x) if target in LOGSPACE_TARGETS else x


def inv_fit_space(target: str, x: float) -> float:
    return math.expm1(x) if target in LOGSPACE_TARGETS else x


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dev_rows() -> List[Dict[str, Any]]:
    with open(DEV_DATASET_PATH) as f:
        return list(csv.DictReader(f))


def load_fold_assignment() -> Dict[Tuple[str, str], str]:
    """Returns {(attempt_id, decision_time_seconds): fold}."""
    out = {}
    with open(FOLD_ASSIGNMENT_PATH) as f:
        for r in csv.DictReader(f):
            out[(r["attempt_id"], r["decision_time_seconds"])] = r["fold"]
    return out


def load_support() -> Dict[int, Tuple[float, float]]:
    out = {}
    with open(SUPPORT_PATH) as f:
        for r in csv.DictReader(f):
            out[int(r["replicas"])] = (float(r["lower_offered_rps"]), float(r["upper_offered_rps"]))
    return out


def load_fresh_matrix() -> List[Dict[str, Any]]:
    with open(FRESH_MATRIX_PATH) as f:
        return list(csv.DictReader(f))


def supported(support: Dict[int, Tuple[float, float]], replicas: int, offered_rps: float) -> bool:
    if replicas not in support:
        return False
    lo, hi = support[replicas]
    return lo <= offered_rps <= hi


def build_rows_with_fold(dev_rows: List[Dict[str, Any]], fold_map: Dict[Tuple[str, str], str]) -> List[Dict[str, Any]]:
    rows = []
    for r in dev_rows:
        key = (r["attempt_id"], r["decision_time_seconds"])
        fold = fold_map.get(key)
        assert fold is not None, f"row not found in frozen fold assignment: {key}"
        row = dict(r)
        row["fold"] = fold
        row["replicas"] = int(r["replicas"])
        row["offered_rps"] = float(r["offered_rps"])
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Section 3: candidate model implementations (Phase16D.1-frozen semantics)
# ---------------------------------------------------------------------------
def degree_basis(offered_rps: float, replicas: int, degree: int) -> List[float]:
    if degree == 1:
        return [1.0, offered_rps, replicas]
    return [1.0, offered_rps, replicas, offered_rps * replicas, offered_rps ** 2, replicas ** 2]


class CandidateFitFailure(Exception):
    """Raised when a candidate cannot be fit at all on the given training rows
    (e.g. zero training rows) -- distinct from a per-query coverage failure."""


class LookupCandidate:
    """monotone_interpolation_lookup[_logspace] and (without PAVA)
    load_regime_conditioned_empirical_distribution. One anchor per tested
    (replicas, offered_rps) fitting level; PAVA (isotonic, increasing)
    optionally applied per replica; 1-D linear interpolation between
    anchors; universal cross-replica interpolation layer when a queried
    replica count has zero of its own anchors."""

    def __init__(self, target: str, logspace: bool, apply_pava: bool):
        self.target = target
        self.logspace = logspace
        self.apply_pava = apply_pava
        self.anchors_by_replicas: Dict[int, Dict[float, float]] = {}
        self.anchors_by_cell: Dict[str, float] = {}
        self.cell_coordinates: Dict[str, Tuple[int, float]] = {}

    def fit(self, rows: List[Dict[str, Any]]) -> None:
        # One anchor per tested (replicas, offered_rps) cell -- grouped by
        # cell_id first (matching the frozen anchors_by_cell/cell_coordinates
        # serialization format exactly), then PAVA-adjusted per replica.
        by_cell: Dict[str, List[float]] = defaultdict(list)
        cell_coord: Dict[str, Tuple[int, float]] = {}
        for r in rows:
            v = float(r[self.target])
            if self.logspace:
                v = math.log1p(v)
            by_cell[r["cell_id"]].append(v)
            cell_coord[r["cell_id"]] = (r["replicas"], r["offered_rps"])
        cell_mean = {cell: sum(vals) / len(vals) for cell, vals in by_cell.items()}

        by_rep_cells: Dict[int, List[str]] = defaultdict(list)
        for cell, (rep, _rps) in cell_coord.items():
            by_rep_cells[rep].append(cell)

        anchors_by_replicas: Dict[int, Dict[float, float]] = {}
        anchors_by_cell: Dict[str, float] = {}
        for rep, cells in by_rep_cells.items():
            cells_sorted = sorted(cells, key=lambda c: cell_coord[c][1])
            means = [cell_mean[c] for c in cells_sorted]
            if self.apply_pava and len(means) >= 2:
                x = np.arange(len(means), dtype=float)
                iso = IsotonicRegression(increasing=True)
                means = list(iso.fit_transform(x, np.array(means)))
            table = {}
            for c, m in zip(cells_sorted, means):
                table[cell_coord[c][1]] = m
                anchors_by_cell[c] = m
            anchors_by_replicas[rep] = table

        self.anchors_by_replicas = anchors_by_replicas
        self.anchors_by_cell = anchors_by_cell
        self.cell_coordinates = cell_coord

    def _predict_at_fitted_replica(self, rep: int, rps: float) -> Optional[float]:
        table = self.anchors_by_replicas.get(rep)
        if not table:
            return None
        levels = sorted(table.items())
        for lr, lv in levels:
            if abs(lr - rps) < 1e-9:
                return lv
        lower = [(r_, v_) for r_, v_ in levels if r_ < rps]
        upper = [(r_, v_) for r_, v_ in levels if r_ > rps]
        if not lower or not upper:
            return None
        lo_r, lo_v = lower[-1]
        hi_r, hi_v = upper[0]
        frac = (rps - lo_r) / (hi_r - lo_r)
        return lo_v + frac * (hi_v - lo_v)

    def predict(self, replicas: int, offered_rps: float) -> Optional[float]:
        val = cross_replica_predict(
            replicas, offered_rps, list(self.anchors_by_replicas.keys()),
            lambda rep, rps: self._predict_at_fitted_replica(rep, rps),
        )
        if val is None:
            return None
        return math.expm1(val) if self.logspace else val

    def serialize(self) -> Dict[str, Any]:
        family = "monotone_interpolation_lookup_logspace" if self.logspace else (
            "load_regime_conditioned_empirical_distribution" if not self.apply_pava else "monotone_interpolation_lookup")
        return {
            "family": family,
            "logspace": self.logspace,
            "anchors_by_cell": dict(self.anchors_by_cell),
            "cell_coordinates": {cell: [rep, rps] for cell, (rep, rps) in self.cell_coordinates.items()},
        }


class SurfaceCandidate:
    """piecewise_linear_regression_surface[_logspace]: numpy.linalg.lstsq
    ordinary least squares over a joint 2-D polynomial basis, pooling ALL
    training rows into one fit (never per-replica segmented)."""

    def __init__(self, target: str, degree: int, logspace: bool):
        self.target = target
        self.degree = degree
        self.logspace = logspace
        self.coefficients: Optional[np.ndarray] = None
        self.rank: Optional[int] = None
        self.fitted_replicas: List[int] = []

    def fit(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            raise CandidateFitFailure("zero training rows")
        X = np.array([degree_basis(r["offered_rps"], r["replicas"], self.degree) for r in rows], dtype=np.float64)
        y = np.array([math.log1p(float(r[self.target])) if self.logspace else float(r[self.target]) for r in rows],
                     dtype=np.float64)
        coeffs, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        self.coefficients = coeffs
        self.rank = int(rank)
        self.fitted_replicas = sorted({r["replicas"] for r in rows})

    def predict(self, replicas: int, offered_rps: float) -> Optional[float]:
        def evaluate(rep: int, rps: float) -> float:
            row = np.array(degree_basis(rps, rep, self.degree), dtype=np.float64)
            return float(row @ self.coefficients)
        val = cross_replica_predict(replicas, offered_rps, self.fitted_replicas, evaluate)
        if val is None:
            return None
        return math.expm1(val) if self.logspace else val

    def serialize(self) -> Dict[str, Any]:
        family = "piecewise_linear_regression_logspace" if self.logspace else "piecewise_linear_regression_surface"
        return {"family": family, "logspace": self.logspace, "degree": self.degree,
                "coefficients": list(self.coefficients), "rank": self.rank, "fitted_replicas": self.fitted_replicas}


class TreeCandidate:
    """tree_based_regressor: sklearn.tree.DecisionTreeRegressor, fully
    pinned arguments per Phase16D.1's estimator_arguments block."""

    def __init__(self, target: str, max_depth: int, logspace: bool = False):
        self.target = target
        self.max_depth = max_depth
        self.logspace = logspace
        self.tree: Optional[DecisionTreeRegressor] = None
        self.fitted_replicas: List[int] = []

    def fit(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            raise CandidateFitFailure("zero training rows")
        X = np.array([[r["offered_rps"], r["replicas"]] for r in rows], dtype=np.float64)
        y = np.array([math.log1p(float(r[self.target])) if self.logspace else float(r[self.target]) for r in rows],
                     dtype=np.float64)
        tree = DecisionTreeRegressor(
            criterion="squared_error", splitter="best", max_depth=self.max_depth,
            min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
            max_features=None, random_state=RANDOM_STATE, max_leaf_nodes=None,
            min_impurity_decrease=0.0, ccp_alpha=0.0,
        )
        tree.fit(X, y)
        self.tree = tree
        self.fitted_replicas = sorted({r["replicas"] for r in rows})

    def predict(self, replicas: int, offered_rps: float) -> Optional[float]:
        def evaluate(rep: int, rps: float) -> float:
            return float(self.tree.predict(np.array([[rps, rep]], dtype=np.float64))[0])
        val = cross_replica_predict(replicas, offered_rps, self.fitted_replicas, evaluate)
        if val is None:
            return None
        return math.expm1(val) if self.logspace else val

    def serialize(self) -> Dict[str, Any]:
        t = self.tree.tree_
        return {
            "family": "tree_based_regressor",
            "logspace": self.logspace,
            "max_depth": self.max_depth,
            "random_state": RANDOM_STATE,
            "fitted_replicas": self.fitted_replicas,
            "sklearn_tree_children_left": [int(x) for x in t.children_left],
            "sklearn_tree_children_right": [int(x) for x in t.children_right],
            "sklearn_tree_feature": [int(x) for x in t.feature],
            "sklearn_tree_threshold": [float(x) for x in t.threshold],
            "sklearn_tree_value": [float(x[0][0]) for x in t.value],
        }


def cross_replica_predict(replicas: int, offered_rps: float, fitted_replicas: List[int],
                            evaluate: Callable[[int, float], Optional[float]]) -> Optional[float]:
    """Universal cross-replica interpolation layer (Phase16D.1): if `replicas`
    itself has a fitted representation, evaluate directly there. Otherwise
    bracket to the nearest fitted replica counts below/above, evaluate the
    SAME family at each bracket using the equivalent per-replica-scaled rps,
    and linearly interpolate the two resulting values in replica-count
    space. Returns None (coverage failure) if replicas cannot be bracketed."""
    if replicas in fitted_replicas:
        return evaluate(replicas, offered_rps)
    lower = [r for r in fitted_replicas if r < replicas]
    upper = [r for r in fitted_replicas if r > replicas]
    if not lower or not upper:
        return None
    lo_r = max(lower)
    hi_r = min(upper)
    per_replica_rps = offered_rps / replicas
    lo_val = evaluate(lo_r, per_replica_rps * lo_r)
    hi_val = evaluate(hi_r, per_replica_rps * hi_r)
    if lo_val is None or hi_val is None:
        return None
    weight_upper = (replicas - lo_r) / (hi_r - lo_r)
    return (1 - weight_upper) * lo_val + weight_upper * hi_val


def build_candidate(target: str, family: str, hyperparam: Any) -> Any:
    logspace = target in LOGSPACE_TARGETS
    if family in ("monotone_interpolation_lookup", "monotone_interpolation_lookup_logspace"):
        return LookupCandidate(target, logspace=logspace, apply_pava=True)
    if family == "load_regime_conditioned_empirical_distribution":
        return LookupCandidate(target, logspace=logspace, apply_pava=False)
    if family in ("piecewise_linear_regression_surface", "piecewise_linear_regression_logspace"):
        return SurfaceCandidate(target, degree=hyperparam, logspace=logspace)
    if family == "tree_based_regressor":
        return TreeCandidate(target, max_depth=hyperparam, logspace=logspace)
    raise ValueError(f"unknown family {family}")


def candidate_hyperparam_grid(family: str) -> List[Any]:
    if family in ("piecewise_linear_regression_surface", "piecewise_linear_regression_logspace"):
        return [1, 2]
    if family == "tree_based_regressor":
        return [2, 3, 4]
    return [None]  # lookup/regime families: no free hyperparameters


# ---------------------------------------------------------------------------
# Section 4.1: candidate-independent common CV evaluability mask
# (Phase16D.2 common_evaluability_formula, recomputed fresh for the 312-row
# Iteration-3 development dataset -- never assumed from Phase16K's 136-row
# count)
# ---------------------------------------------------------------------------
def common_cv_evaluable_instances(rows: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Returns the list of (cell_id, fold) pairs where validation_rows(cell,fold)>=1
    AND training_rows(cell,fold)>=1. Depends only on row availability (cell_id,
    fold), never on target or candidate family -- identical across all 6 targets
    since every one of the 312 rows has all 6 target columns populated (verified
    in Phase16N)."""
    cells = sorted({r["cell_id"] for r in rows})
    folds = ["fold1", "fold2", "fold3"]
    by_cell_fold = defaultdict(list)
    for r in rows:
        by_cell_fold[(r["cell_id"], r["fold"])].append(r)
    instances = []
    for cell in cells:
        for fold in folds:
            validation_rows = len(by_cell_fold.get((cell, fold), []))
            training_rows = sum(len(by_cell_fold.get((cell, f), [])) for f in folds if f != fold)
            if validation_rows >= 1 and training_rows >= 1:
                instances.append((cell, fold))
    return instances


def cell_coordinate(rows: List[Dict[str, Any]], cell_id: str) -> Tuple[int, float]:
    for r in rows:
        if r["cell_id"] == cell_id:
            return r["replicas"], r["offered_rps"]
    raise KeyError(cell_id)


# ---------------------------------------------------------------------------
# Section 5-6: physical constraint / monotonicity check domains, s_eff
# ---------------------------------------------------------------------------
def physical_and_monotonicity_grid(support: Dict[int, Tuple[float, float]],
                                     rows: List[Dict[str, Any]],
                                     fresh_matrix: List[Dict[str, Any]]) -> Dict[int, List[float]]:
    """For each of the 8 replica counts: 25 evenly-spaced offered_rps points
    across [lo,hi] (endpoints inclusive), PLUS the exact fitting anchors for
    that replica (distinct offered_rps values in the development dataset),
    PLUS the (identity-only) Iteration-3 fresh validation coordinates for
    that replica -- reusing the frozen Phase16D.1 grid construction, extended
    to all 8 replicas since Iteration-2/2B added direct empirical anchors at
    every replica count (no longer only the original 5)."""
    anchors_by_rep: Dict[int, set] = defaultdict(set)
    for r in rows:
        anchors_by_rep[r["replicas"]].add(r["offered_rps"])
    fresh_by_rep: Dict[int, set] = defaultdict(set)
    for r in fresh_matrix:
        fresh_by_rep[int(r["replicas"])].add(float(r["offered_rps"]))

    grid = {}
    for rep, (lo, hi) in support.items():
        evenly = [lo + i * (hi - lo) / 24 for i in range(25)]
        points = sorted(set(evenly) | anchors_by_rep.get(rep, set()) | fresh_by_rep.get(rep, set()))
        grid[rep] = points
    return grid


def check_monotonicity(predict_fn: Callable[[int, float], Optional[float]], grid: Dict[int, List[float]],
                        target: str) -> List[Dict[str, Any]]:
    """Adjacent pair (i,i+1) violates monotonicity iff predicted[i+1] < predicted[i] - tolerance.
    Per Phase16D.1's monotonicity_check_domain: for latency_p95_seconds this
    check and its tolerance (epsilon[latency_p95_seconds], defined in log1p
    units) are BOTH evaluated in log1p space -- predict_fn returns raw
    (post-expm1) values, so each prediction is converted back to fit_space
    before comparison, matching the frozen "non-decreasing in log1p space"
    wording exactly rather than silently comparing log1p-scale tolerance
    against raw-scale values."""
    violations = []
    tol = EPSILON[target]
    for rep, points in grid.items():
        preds = [predict_fn(rep, p) for p in points]
        for i in range(len(preds) - 1):
            a, b = preds[i], preds[i + 1]
            if a is None or b is None:
                continue
            a_space, b_space = fit_space(target, a), fit_space(target, b)
            if b_space < a_space - tol:
                violations.append({"replicas": rep, "offered_rps_lower": points[i], "offered_rps_upper": points[i + 1],
                                    "predicted_lower": a, "predicted_upper": b})
    return violations


def check_physical_constraints(predict_fn: Callable[[int, float], Optional[float]], grid: Dict[int, List[float]]
                                 ) -> List[Dict[str, Any]]:
    """Reject if prediction < 0 anywhere in the check domain (tolerance -1e-9)."""
    violations = []
    for rep, points in grid.items():
        for p in points:
            val = predict_fn(rep, p)
            if val is None:
                continue
            if val < -1e-9:
                violations.append({"replicas": rep, "offered_rps": p, "predicted": val})
    return violations


def compute_pooled_fitting_sd(rows: List[Dict[str, Any]], target: str) -> Optional[float]:
    """Phase16D.1 pooled_sd_formula: pooled variance across every cell (for
    this target) with >=2 usable repeats, in the target's fitting/transform
    space; repeat-means computed once per cell across ALL repeats (not
    per-fold)."""
    by_cell_repeat: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = fit_space(target, float(r[target]))
        by_cell_repeat[r["cell_id"]][r["repeat_number"]].append(v)
    cell_var_dof = []
    for cell, by_repeat in by_cell_repeat.items():
        repeat_means = [sum(vals) / len(vals) for vals in by_repeat.values()]
        n = len(repeat_means)
        if n >= 2:
            m = sum(repeat_means) / n
            var = sum((x - m) ** 2 for x in repeat_means) / (n - 1)
            cell_var_dof.append((n - 1, var))
    den = sum(w for w, _ in cell_var_dof)
    if den == 0:
        return None
    return math.sqrt(sum(w * v for w, v in cell_var_dof) / den)


def within_cell_repeat_sd(pooled_points: List[Tuple[str, float]], target: str) -> Optional[float]:
    per_repeat = defaultdict(list)
    for repeat_number, v in pooled_points:
        per_repeat[repeat_number].append(fit_space(target, v))
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
# Section 4.2-4.4: per-fold fitting, candidate prediction coverage
# ---------------------------------------------------------------------------
def evaluate_candidate_cv(target: str, family: str, hyperparam: Any, rows: List[Dict[str, Any]],
                            common_instances: List[Tuple[str, str]]) -> Dict[str, Any]:
    """Fits `family`/`hyperparam` within each of the 3 folds (training on the
    other 2 folds) and scores it against every (cell,fold) in the frozen
    common-evaluable set. Returns coverage info + per-instance normalized
    errors, never silently narrowing the mask."""
    folds = ["fold1", "fold2", "fold3"]
    rows_by_fold = defaultdict(list)
    for r in rows:
        rows_by_fold[r["fold"]].append(r)

    pooled_sd = compute_pooled_fitting_sd(rows, target)
    rows_by_cell = defaultdict(list)
    rows_by_cell_fold = defaultdict(list)
    for r in rows:
        rows_by_cell[r["cell_id"]].append((r["repeat_number"], float(r[target])))
        rows_by_cell_fold[(r["cell_id"], r["fold"])].append(float(r[target]))

    per_instance = []
    n_required = len(common_instances)
    n_covered = 0

    for fold in folds:
        train_rows = [r for f in folds if f != fold for r in rows_by_fold[f]]
        try:
            model = build_candidate(target, family, hyperparam)
            model.fit(train_rows)
        except CandidateFitFailure:
            model = None
        for cell_id, held_fold in common_instances:
            if held_fold != fold:
                continue
            replicas, offered_rps = cell_coordinate(rows, cell_id)
            pred = model.predict(replicas, offered_rps) if model is not None else None
            if pred is None:
                per_instance.append({"cell_id": cell_id, "fold": fold, "covered": False,
                                      "reason": "CV_PREDICTION_COVERAGE_FAILURE"})
                continue
            n_covered += 1
            val_values = rows_by_cell_fold[(cell_id, fold)]
            observed_mean = sum(val_values) / len(val_values)
            wsd = within_cell_repeat_sd(rows_by_cell[cell_id], target)
            se = s_eff(target, wsd, pooled_sd)
            fo = fit_space(target, observed_mean)
            fp = fit_space(target, pred)
            ne = abs(fp - fo) / se
            per_instance.append({
                "cell_id": cell_id, "fold": fold, "covered": True, "replicas": replicas, "offered_rps": offered_rps,
                "predicted": pred, "observed": observed_mean, "s_eff": se, "normalized_error": ne,
                "abs_error": abs(pred - observed_mean),
            })

    coverage_fraction = (n_covered / n_required) if n_required else 1.0
    return {
        "target": target, "family": family, "hyperparam": hyperparam,
        "n_required": n_required, "n_covered": n_covered, "coverage_fraction": coverage_fraction,
        "full_coverage": n_covered == n_required,
        "per_instance": per_instance,
    }


# ---------------------------------------------------------------------------
# Section 5, 7: hard gates + lexicographic selection
# ---------------------------------------------------------------------------
def aggregate_metric(target: str, family: str, per_instance: List[Dict[str, Any]]) -> Dict[str, float]:
    """Cross-cell aggregation: every (cell,fold) instance = one independent
    evaluation instance, pooled directly (never fold-averaged first)."""
    scored = [x for x in per_instance if x.get("covered")]
    nes = [x["normalized_error"] for x in scored]
    abs_errs = [x["abs_error"] for x in scored]
    return {
        "median_normalized_error": float(np.median(nes)) if nes else None,
        "worst_normalized_error": max(nes) if nes else None,
        "fraction_le_2": (sum(1 for x in nes if x <= 2.0) / len(nes)) if nes else None,
        "MAE": (sum(abs_errs) / len(abs_errs)) if abs_errs else None,
    }


def mae_per_replica_utilization(per_instance: List[Dict[str, Any]]) -> Optional[float]:
    scored = [x for x in per_instance if x.get("covered")]
    if not scored:
        return None
    vals = [abs(x["predicted"] / x["replicas"] - x["observed"] / x["replicas"]) for x in scored]
    return sum(vals) / len(vals)


def score_candidate(target: str, family: str, hyperparam: Any, rows: List[Dict[str, Any]],
                     common_instances: List[Tuple[str, str]], support: Dict[int, Tuple[float, float]],
                     fresh_matrix: List[Dict[str, Any]]) -> Dict[str, Any]:
    cv_result = evaluate_candidate_cv(target, family, hyperparam, rows, common_instances)
    result: Dict[str, Any] = {
        "target": target, "family": family, "hyperparam": hyperparam,
        "coverage_fraction": cv_result["coverage_fraction"], "full_coverage": cv_result["full_coverage"],
        "n_required": cv_result["n_required"], "n_covered": cv_result["n_covered"],
    }
    if not cv_result["full_coverage"]:
        result["status"] = "CV_PREDICTION_COVERAGE_FAILURE"
        return result

    metrics = aggregate_metric(target, family, cv_result["per_instance"])
    result["median_normalized_error"] = metrics["median_normalized_error"]
    result["worst_normalized_error"] = metrics["worst_normalized_error"]
    result["fraction_le_2"] = metrics["fraction_le_2"]
    result["MAE"] = metrics["MAE"]
    if SECONDARY_METRIC.get(target) == "MAE_per_replica_utilization_fraction":
        result["secondary_metric_value"] = mae_per_replica_utilization(cv_result["per_instance"])
    else:
        result["secondary_metric_value"] = None

    # Hard gates: refit on ALL rows for this fold-independent full-data check
    # (monotonicity/physical constraints are evaluated on the model that would
    # actually be deployed -- i.e. the full-data refit -- per Phase16D.1's own
    # "Step 1" framing, which is a property of the FINAL candidate, not of any
    # one fold's partial fit).
    full_model = build_candidate(target, family, hyperparam)
    try:
        full_model.fit(rows)
    except CandidateFitFailure:
        result["status"] = "FIT_FAILURE"
        return result
    grid = physical_and_monotonicity_grid(support, rows, fresh_matrix)
    phys_violations = check_physical_constraints(full_model.predict, grid)
    mono_violations = ([] if target not in MONOTONIC_ENFORCED_TARGETS
                        else check_monotonicity(full_model.predict, grid, target))
    result["physical_violations"] = phys_violations
    result["monotonicity_violations"] = mono_violations
    if phys_violations:
        result["status"] = "REJECTED_PHYSICAL_CONSTRAINT"
        return result
    if mono_violations:
        result["status"] = "REJECTED_MONOTONICITY"
        return result

    result["status"] = "CANDIDATE_VALID"
    result["complexity_rank"] = COMPLEXITY_RANK[family]
    return result


def select_model(target: str, candidates_scored: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [c for c in candidates_scored if c["status"] == "CANDIDATE_VALID"]
    if not valid:
        return {"target": target, "status": "NO_VALID_CANDIDATE", "selected": None}

    def sort_key(c):
        # Step2: primary metric (PRIMARY_METRIC[target], always this target's own MAE-family metric).
        # Step3: secondary metric, if defined for this target (else contributes nothing -- constant 0.0
        # for every candidate, so it never discriminates when undefined, matching "if defined" in the
        # frozen selection rule).
        # Step4: worst-cell (max) normalized error. Step5: complexity rank. Step6: alphabetical family name.
        secondary = c["secondary_metric_value"] if c["secondary_metric_value"] is not None else 0.0
        return (c["MAE"], secondary, c["worst_normalized_error"], c["complexity_rank"], c["family"])

    ranked = sorted(valid, key=sort_key)
    selected = ranked[0]
    return {"target": target, "status": "SELECTED", "selected": selected, "ranking": ranked}


# ---------------------------------------------------------------------------
# Section 8: final refit on ALL 312 development rows
# ---------------------------------------------------------------------------
def final_refit(target: str, family: str, hyperparam: Any, rows: List[Dict[str, Any]]) -> Any:
    model = build_candidate(target, family, hyperparam)
    model.fit(rows)
    return model


def reconstruct_predict_from_params(params: Dict[str, Any]) -> Callable[[int, float], Optional[float]]:
    """Rebuilds a predict(replicas, offered_rps) function purely from a
    serialized model_parameters.json dict -- never re-fits. Handles every
    family this project's frozen model artifacts actually use:
    piecewise_linear_regression_surface[_logspace] ({family, degree,
    coefficients, rank}), monotone_interpolation_lookup[_logspace] /
    load_regime_conditioned_empirical_distribution / threshold_regime_model
    ({family, anchors_by_cell: {cell_id: value}, cell_coordinates: {cell_id:
    [replicas, offered_rps]}} -- the exact frozen historical format,
    confirmed against the real Iteration-2B artifacts), and tree_based_regressor
    (serialized sklearn tree arrays)."""
    family = params["family"]
    # Prefer the explicit `logspace` field (written by this script's own
    # serialize() methods, which are self-describing and target-independent).
    # Historical Iteration-2B artifacts predate this field but are always
    # self-describing via their family-name suffix instead (no historical
    # artifact ever used a family whose name is logspace-ambiguous, e.g.
    # tree_based_regressor was never selected for latency_p95_seconds in
    # Iteration-2B) -- so the suffix heuristic is a safe fallback for them.
    logspace = params["logspace"] if "logspace" in params else family.endswith("logspace")

    if family in ("piecewise_linear_regression_surface", "piecewise_linear_regression_logspace"):
        coeffs = np.array(params["coefficients"], dtype=np.float64)
        degree = params["degree"]
        fitted_replicas = params.get("fitted_replicas") or list(range(1, 9))

        def evaluate(rep: int, rps: float) -> float:
            row = np.array(degree_basis(rps, rep, degree), dtype=np.float64)
            return float(row @ coeffs)

        def predict(rep: int, rps: float) -> Optional[float]:
            val = cross_replica_predict(rep, rps, fitted_replicas, evaluate)
            if val is None:
                return None
            return math.expm1(val) if logspace else val
        return predict

    if family in ("monotone_interpolation_lookup", "monotone_interpolation_lookup_logspace",
                  "load_regime_conditioned_empirical_distribution", "threshold_regime_model"):
        anchors_by_cell = params["anchors_by_cell"]
        cell_coords = params["cell_coordinates"]
        anchors_by_rep: Dict[int, Dict[float, float]] = defaultdict(dict)
        for cell, val in anchors_by_cell.items():
            rep, rps = cell_coords[cell]
            anchors_by_rep[int(rep)][float(rps)] = val
        fitted_replicas = sorted(anchors_by_rep.keys())

        def lookup_at(rep: int, rps: float) -> Optional[float]:
            table = anchors_by_rep.get(rep)
            if not table:
                return None
            levels = sorted(table.items())
            for lr, lv in levels:
                if abs(lr - rps) < 1e-9:
                    return lv
            lower = [(r_, v_) for r_, v_ in levels if r_ < rps]
            upper = [(r_, v_) for r_, v_ in levels if r_ > rps]
            if not lower or not upper:
                return None
            lo_r, lo_v = lower[-1]
            hi_r, hi_v = upper[0]
            frac = (rps - lo_r) / (hi_r - lo_r)
            return lo_v + frac * (hi_v - lo_v)

        def predict(rep: int, rps: float) -> Optional[float]:
            val = cross_replica_predict(rep, rps, fitted_replicas, lookup_at)
            if val is None:
                return None
            return math.expm1(val) if logspace else val
        return predict

    if family == "tree_based_regressor":
        children_left = params["sklearn_tree_children_left"]
        children_right = params["sklearn_tree_children_right"]
        feature = params["sklearn_tree_feature"]
        threshold = params["sklearn_tree_threshold"]
        value = params["sklearn_tree_value"]
        fitted_replicas = params.get("fitted_replicas") or list(range(1, 9))

        def tree_eval(rps: float, rep: int) -> float:
            node = 0
            x = [rps, rep]
            while children_left[node] != -1:
                f = feature[node]
                node = children_left[node] if x[f] <= threshold[node] else children_right[node]
            return value[node]

        def predict(rep: int, rps: float) -> Optional[float]:
            val = cross_replica_predict(rep, rps, fitted_replicas, lambda r_, p_: tree_eval(p_, r_))
            if val is None:
                return None
            return math.expm1(val) if logspace else val
        return predict

    raise ValueError(f"unsupported family for reconstruction: {family}")


def load_frozen_model_predict_fn(target: str) -> Tuple[Callable[[int, float], Optional[float]], str, Dict[str, Any]]:
    """Loads a preserved Iteration-2B model (request_rate_rps / memory_bytes /
    error_rate_rps) and reconstructs its predict function purely from the
    serialized parameters -- never re-fit."""
    path = IT2B_MODELS_DIR / target / "model_parameters.json"
    params = json.loads(path.read_text())
    return reconstruct_predict_from_params(params), params["family"], params


# ---------------------------------------------------------------------------
# Section 9: joint consistency (error_rate_rps <= request_rate_rps + tol)
# ---------------------------------------------------------------------------
def joint_consistency_audit(support: Dict[int, Tuple[float, float]], rows: List[Dict[str, Any]],
                              fresh_matrix: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rr_predict, _, _ = load_frozen_model_predict_fn("request_rate_rps")
    er_predict, _, _ = load_frozen_model_predict_fn("error_rate_rps")
    grid = physical_and_monotonicity_grid(support, rows, fresh_matrix)
    tol = EPSILON["request_rate_rps"]
    audit_rows = []
    for rep, points in grid.items():
        for p in points:
            rr = rr_predict(rep, p)
            er = er_predict(rep, p)
            if rr is None or er is None:
                continue
            upper_bound = rr + tol
            violation = (er < -1e-9) or (er > upper_bound + 1e-9)
            audit_rows.append({
                "replicas": rep, "offered_rps": p, "predicted_request_rate_rps": rr,
                "predicted_error_rate_rps": er, "upper_bound": upper_bound, "violation": violation,
            })
    return audit_rows


# ---------------------------------------------------------------------------
# Section 10: prediction preregistration for the 16 fresh Iteration-3 cells
# ---------------------------------------------------------------------------
def preregister_predictions(support: Dict[int, Tuple[float, float]], fresh_matrix: List[Dict[str, Any]],
                              final_models: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows_out = []
    for cell in fresh_matrix:
        replicas = int(cell["replicas"])
        offered_rps = float(cell["offered_rps"])
        lo, hi = support[replicas]
        support_status = "SUPPORTED" if lo < offered_rps < hi else "OUT_OF_SUPPORT_OR_BOUNDARY"
        for target in ALL_TARGETS:
            entry = final_models[target]
            pred = entry["predict"](replicas, offered_rps) if support_status == "SUPPORTED" else None
            rows_out.append({
                "cell_id": cell["cell_id"], "replicas": replicas, "offered_rps": offered_rps, "target": target,
                "prediction": pred, "family": entry["family"], "hyperparameter": entry.get("hyperparameter"),
                "model_artifact_sha256": entry["sha256"], "support_status": support_status,
                "transform_space": "log1p" if target in LOGSPACE_TARGETS else "raw",
                "model_status": entry["status"],
            })
    return rows_out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def sha256_of(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


PRE_FIT_FREEZE_INPUTS = {
    "task_outputs/phase_16n_iteration3_development_dataset.csv": "f9c7a972d2eedac50eb69e218d38df97947d7d8b1f41b600560a1329a6895e54",
    "task_outputs/phase_16n_iteration3_cv_fold_assignment.csv": "bd766fb2d5161089e3dd19754c41ad75eaa2c92e8c14d8cc5c9d1d7bc0ff4995",
    "task_outputs/phase_16n_iteration3_fresh_validation_matrix.csv": "dfdc9f8698c131c7096c3e779dc20077fa1c10f9cab21561f646bba8d9bb637b",
    "task_outputs/phase_16n_iteration3_final_validation_sequence.csv": "d97b2b3be8e182e2c299d4abaec91e54fdbbc9639c1ab070f2def3084a01d7de",
    "rl-k8s-autoscaler/configs/traffic_profiles/rl_dynamics_calibration_iteration3/iteration3_final_validation_sequence.json":
        "5c2556160bd4cfaca487c40fe3faca114fb19d5cc9b86480e202fa43a83f4070",
    "rl-k8s-autoscaler/configs/traffic_profiles/rl_dynamics_calibration_iteration3/iteration3_final_validation_schedule_registry.json":
        "e3a952e9be006702f01b8a698987224fa204ec5c1ef135ebb8bc5d3222a70c4b",
    "task_outputs/phase_16k_effective_direct_support.csv": "8f25d33a7842ee81c4064522f4cfbefb7ddd576d405712491c2a6c2b1e5c2310",
    "task_outputs/phase_16m_locked_scientific_verdict.json": "0bc7a5ab9adeef903fab4e61fde7e34d7c540f560864a5f170677c0243d94aa9",
    "rl-k8s-autoscaler/persistent-data/models/dynamics_iteration2b/request_rate_rps/model_parameters.json":
        "43c2303c8a716cd5b0a06fbb79c3e24866944bc3cb5ba9ec57623551a5d2d430",
    "rl-k8s-autoscaler/persistent-data/models/dynamics_iteration2b/memory_bytes/model_parameters.json":
        "55e32d1a2dee03ae8bdd1a83d9e5650f203f6c6fdda967de0508fc614c6a0597",
    "rl-k8s-autoscaler/persistent-data/models/dynamics_iteration2b/error_rate_rps/model_parameters.json":
        "3ec5d09cd3840116eed5c4ec4ba40eb2835fc90529fc6a443d33b7f1661bbc90",
    "task_outputs/phase_16d_candidate_model_registry.csv": "7fb2742aef71e9f8c13770b48810b86ea73d3bb96f948eeee9ba7cc8ee6e3dd5",
}


def run_predevelopment_freeze_audit() -> Tuple[bool, List[Dict[str, Any]]]:
    rows = []
    all_ok = True
    for rel_path, expected in PRE_FIT_FREEZE_INPUTS.items():
        if rel_path.startswith("rl-k8s-autoscaler/"):
            p = PROJECT_ROOT.parent / rel_path
        else:
            p = PROJECT_ROOT.parent / rel_path
        actual = sha256_of(p)
        ok = actual == expected
        all_ok = all_ok and ok
        rows.append({"path": rel_path, "expected_sha256": expected, "actual_sha256": actual, "match": ok})
    return all_ok, rows


def main() -> Dict[str, Any]:
    freeze_ok, freeze_rows = run_predevelopment_freeze_audit()
    write_csv(TASK_OUTPUTS / "phase_16o_predevelopment_freeze_audit.csv", freeze_rows,
              ["path", "expected_sha256", "actual_sha256", "match"])
    assert freeze_ok, "PHASE16O_FROZEN_INPUT_DRIFT_DETECTED"

    dev_rows_raw = load_dev_rows()
    fold_map = load_fold_assignment()
    rows = build_rows_with_fold(dev_rows_raw, fold_map)
    assert len(rows) == 312
    support = load_support()
    fresh_matrix = load_fresh_matrix()
    assert len(fresh_matrix) == 16

    common_instances = common_cv_evaluable_instances(rows)
    write_csv(TASK_OUTPUTS / "phase_16o_candidate_common_cv_evaluation_mask.csv",
              [{"cell_id": c, "fold": f} for c, f in common_instances], ["cell_id", "fold"])

    registry_rows = []
    coverage_rows = []
    cv_result_rows = []
    physical_rows = []
    selection_rows = []
    final_models: Dict[str, Any] = {}
    selection_verdicts = {}

    coverage_failure_found = False
    for target in TARGETS_REDEVELOP:
        scored = []
        for family in CANDIDATE_FAMILIES_IN_SCOPE[target]:
            for hp in candidate_hyperparam_grid(family):
                registry_rows.append({"target": target, "family": family, "hyperparameter": hp,
                                       "complexity_rank": COMPLEXITY_RANK[family]})
                cv_res = evaluate_candidate_cv(target, family, hp, rows, common_instances)
                coverage_rows.append({
                    "target": target, "family": family, "hyperparameter": hp,
                    "n_required": cv_res["n_required"], "n_covered": cv_res["n_covered"],
                    "coverage_fraction": cv_res["coverage_fraction"], "full_coverage": cv_res["full_coverage"],
                })
                for inst in cv_res["per_instance"]:
                    cv_result_rows.append({
                        "target": target, "family": family, "hyperparameter": hp, "cell_id": inst["cell_id"],
                        "fold": inst["fold"], "covered": inst["covered"],
                        "replicas": inst.get("replicas"), "offered_rps": inst.get("offered_rps"),
                        "predicted": inst.get("predicted"), "observed": inst.get("observed"),
                        "s_eff": inst.get("s_eff"), "normalized_error": inst.get("normalized_error"),
                    })
                if not cv_res["full_coverage"]:
                    coverage_failure_found = True
                    scored.append({"target": target, "family": family, "hyperparam": hp,
                                    "status": "CV_PREDICTION_COVERAGE_FAILURE", "coverage_fraction": cv_res["coverage_fraction"],
                                    "n_required": cv_res["n_required"], "n_covered": cv_res["n_covered"]})
                    continue
                s = score_candidate(target, family, hp, rows, common_instances, support, fresh_matrix)
                scored.append(s)
                for pv in s.get("physical_violations", []):
                    physical_rows.append({"target": target, "family": family, "hyperparameter": hp,
                                           "check": "PHYSICAL_CONSTRAINT", **pv})
                for mv in s.get("monotonicity_violations", []):
                    physical_rows.append({"target": target, "family": family, "hyperparameter": hp,
                                           "check": "MONOTONICITY", **mv})

        for s in scored:
            selection_rows.append({
                "target": s["target"], "family": s["family"], "hyperparameter": s.get("hyperparam"),
                "status": s["status"], "coverage_fraction": s.get("coverage_fraction"),
                "MAE": s.get("MAE"), "secondary_metric": s.get("secondary_metric_value"),
                "worst_normalized_error": s.get("worst_normalized_error"),
                "median_normalized_error": s.get("median_normalized_error"),
                "complexity_rank": s.get("complexity_rank"),
            })
        sel = select_model(target, scored)
        selection_verdicts[target] = sel
        if sel["status"] != "SELECTED":
            write_csv(TASK_OUTPUTS / "phase_16o_candidate_registry_and_grid_audit.csv", registry_rows,
                      ["target", "family", "hyperparameter", "complexity_rank"])
            write_csv(TASK_OUTPUTS / "phase_16o_candidate_prediction_coverage_audit.csv", coverage_rows,
                      ["target", "family", "hyperparameter", "n_required", "n_covered", "coverage_fraction", "full_coverage"])
            write_csv(TASK_OUTPUTS / "phase_16o_cv_fold_evaluation_results.csv", cv_result_rows,
                      ["target", "family", "hyperparameter", "cell_id", "fold", "covered", "replicas", "offered_rps",
                       "predicted", "observed", "s_eff", "normalized_error"])
            write_csv(TASK_OUTPUTS / "phase_16o_candidate_selection_summary.csv", selection_rows,
                      ["target", "family", "hyperparameter", "status", "coverage_fraction", "MAE", "secondary_metric",
                       "worst_normalized_error", "median_normalized_error", "complexity_rank"])
            return {"verdict": "PHASE16O_ITERATION3_MODEL_SELECTION_FAILED", "target": target}

        chosen = sel["selected"]
        model = final_refit(target, chosen["family"], chosen["hyperparam"], rows)
        n_survivors = sum(1 for c in scored if c["status"] == "CANDIDATE_VALID")
        final_models[target] = {
            "predict": model.predict, "family": chosen["family"], "hyperparameter": chosen["hyperparam"],
            "serialized": model.serialize(), "status": "REDEVELOPED",
            "selection": chosen, "n_survivors": n_survivors,
        }

    write_csv(TASK_OUTPUTS / "phase_16o_candidate_registry_and_grid_audit.csv", registry_rows,
              ["target", "family", "hyperparameter", "complexity_rank"])
    write_csv(TASK_OUTPUTS / "phase_16o_candidate_prediction_coverage_audit.csv", coverage_rows,
              ["target", "family", "hyperparameter", "n_required", "n_covered", "coverage_fraction", "full_coverage"])
    write_csv(TASK_OUTPUTS / "phase_16o_cv_fold_evaluation_results.csv", cv_result_rows,
              ["target", "family", "hyperparameter", "cell_id", "fold", "covered", "replicas", "offered_rps",
               "predicted", "observed", "s_eff", "normalized_error"])
    write_csv(TASK_OUTPUTS / "phase_16o_candidate_physical_validity_audit.csv", physical_rows,
              ["target", "family", "hyperparameter", "check", "replicas", "offered_rps", "offered_rps_lower",
               "offered_rps_upper", "predicted", "predicted_lower", "predicted_upper"])
    write_csv(TASK_OUTPUTS / "phase_16o_candidate_selection_summary.csv", selection_rows,
              ["target", "family", "hyperparameter", "status", "coverage_fraction", "MAE", "secondary_metric",
               "worst_normalized_error", "median_normalized_error", "complexity_rank"])

    # Frozen targets: carried forward byte-identical -- ALL provenance files
    # copied verbatim (model_parameters.json, model_metadata.json,
    # model_spec.yaml, training_row_identities.csv), never regenerated.
    PROVENANCE_FILES = ["model_parameters.json", "model_metadata.json", "model_spec.yaml",
                         "training_row_identities.csv"]
    frozen_carryforward_rows = []
    for target in TARGETS_FROZEN:
        predict_fn, family, params = load_frozen_model_predict_fn(target)
        final_models[target] = {"predict": predict_fn, "family": family, "hyperparameter": None,
                                 "serialized": params, "status": "FROZEN_CARRIED_FORWARD"}
        (IT3_MODELS_DIR / target).mkdir(parents=True, exist_ok=True)
        for fname in PROVENANCE_FILES:
            src = IT2B_MODELS_DIR / target / fname
            dst = IT3_MODELS_DIR / target / fname
            dst.write_bytes(src.read_bytes())
            assert sha256_of(src) == sha256_of(dst)
            frozen_carryforward_rows.append({
                "target": target, "file": fname,
                "source_path": str(src.relative_to(PROJECT_ROOT.parent)),
                "dest_path": str(dst.relative_to(PROJECT_ROOT.parent)),
                "source_sha256": sha256_of(src), "dest_sha256": sha256_of(dst),
                "byte_identical": sha256_of(src) == sha256_of(dst),
            })
    write_csv(TASK_OUTPUTS / "phase_16o_frozen_target_carryforward_audit.csv", frozen_carryforward_rows,
              ["target", "file", "source_path", "dest_path", "source_sha256", "dest_sha256", "byte_identical"])

    # Write redeveloped model artifacts
    common_cv_mask_hash = sha256_of(TASK_OUTPUTS / "phase_16o_candidate_common_cv_evaluation_mask.csv")
    fitting_driver_hash = sha256_of(Path(__file__).resolve())
    n_cells = len({r["cell_id"] for r in rows})
    library_versions = {
        "python": "3.10.14", "numpy": np.__version__,
        "sklearn": __import__("sklearn").__version__,
    }

    refit_provenance_rows = []
    model_hash_rows = []
    for target in TARGETS_REDEVELOP:
        entry = final_models[target]
        dst = IT3_MODELS_DIR / target / "model_parameters.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(entry["serialized"], indent=2, default=lambda o: float(o)))
        entry["sha256"] = sha256_of(dst)
        refit_provenance_rows.append({
            "target": target, "family": entry["family"], "hyperparameter": entry["hyperparameter"],
            "n_training_rows": 312, "model_path": str(dst.relative_to(PROJECT_ROOT.parent)),
            "sha256": entry["sha256"],
        })
        model_hash_rows.append({"target": target, "status": "REDEVELOPED", "family": entry["family"],
                                 "path": str(dst.relative_to(PROJECT_ROOT.parent)), "sha256": entry["sha256"]})

        chosen = entry["selection"]
        secondary_name = SECONDARY_METRIC.get(target)
        metadata = {
            "target": target, "selected_family": entry["family"],
            "hyperparameters": {"value": entry["hyperparameter"]},
            "random_seed": RANDOM_STATE if entry["family"] == "tree_based_regressor" else "N/A -- deterministic family",
            "training_dataset_hash": "f9c7a972d2eedac50eb69e218d38df97947d7d8b1f41b600560a1329a6895e54",
            "training_row_count": 312, "training_cell_count": n_cells,
            "common_cv_mask_hash": common_cv_mask_hash, "fitting_driver_hash": fitting_driver_hash,
            "candidate_registry_hash": "7fb2742aef71e9f8c13770b48810b86ea73d3bb96f948eeee9ba7cc8ee6e3dd5",
            "target_transform": "log1p" if target in LOGSPACE_TARGETS else "raw (no transform)",
            "library_versions": library_versions,
            "fit_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model_parameter_hash": entry["sha256"],
            "internal_selection_metrics": {
                "primary_metric_name": PRIMARY_METRIC[target], "primary_metric_value": chosen["MAE"],
                "secondary_metric_name": secondary_name, "secondary_metric_value": chosen["secondary_metric_value"],
                "worst_normalized_error": chosen["worst_normalized_error"], "n_survivors": entry["n_survivors"],
            },
            "internal_acceptance_status": "PASS",
            "final_validation_status": "NOT_YET_RUN",
            "iteration": "ITERATION3",
        }
        (IT3_MODELS_DIR / target / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

        spec = {
            "target": target, "selected_family": entry["family"],
            "hyperparameters": {"value": entry["hyperparameter"]},
            "cross_replica_strategy": "DIRECT_INTEGER_REPLICA_EVIDENCE_1_THROUGH_8 -- no cross-replica bridging wrapper used (all 8 replica counts have direct Iteration-2/2B/Phase16M anchors)",
            "support_guard_dependency": "Phase16K effective_direct_support evaluated before any prediction",
            "target_transform": "log1p" if target in LOGSPACE_TARGETS else "raw (no transform)",
            "iteration": "ITERATION3",
        }
        with open(IT3_MODELS_DIR / target / "model_spec.yaml", "w") as f:
            import yaml as _yaml
            _yaml.safe_dump(spec, f, sort_keys=False)

        with open(IT3_MODELS_DIR / target / "training_row_identities.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["cell_id", "repeat_number", "attempt_id", "decision_time_seconds", "source_dataset"])
            w.writeheader()
            for r in rows:
                w.writerow({"cell_id": r["cell_id"], "repeat_number": r["repeat_number"], "attempt_id": r["attempt_id"],
                            "decision_time_seconds": r["decision_time_seconds"], "source_dataset": r["source_dataset"]})
    for target in TARGETS_FROZEN:
        entry = final_models[target]
        entry["sha256"] = sha256_of(IT3_MODELS_DIR / target / "model_parameters.json")
        model_hash_rows.append({"target": target, "status": "FROZEN_CARRIED_FORWARD", "family": entry["family"],
                                 "path": str((IT3_MODELS_DIR / target / "model_parameters.json").relative_to(PROJECT_ROOT.parent)),
                                 "sha256": entry["sha256"]})

    write_csv(TASK_OUTPUTS / "phase_16o_final_refit_provenance.csv", refit_provenance_rows,
              ["target", "family", "hyperparameter", "n_training_rows", "model_path", "sha256"])
    write_csv(TASK_OUTPUTS / "phase_16o_iteration3_model_hash_manifest.csv", model_hash_rows,
              ["target", "status", "family", "path", "sha256"])

    selected_models_json = {}
    for target in ALL_TARGETS:
        entry = final_models[target]
        selected_models_json[target] = {
            "status": entry["status"], "family": entry["family"], "hyperparameter": entry.get("hyperparameter"),
            "sha256": entry["sha256"],
        }
    (TASK_OUTPUTS / "phase_16o_selected_target_models.json").write_text(json.dumps(selected_models_json, indent=2))

    # Joint consistency
    joint_rows = joint_consistency_audit(support, rows, fresh_matrix)
    write_csv(TASK_OUTPUTS / "phase_16o_joint_consistency_audit.csv", joint_rows,
              ["replicas", "offered_rps", "predicted_request_rate_rps", "predicted_error_rate_rps", "upper_bound", "violation"])
    n_joint_violations = sum(1 for r in joint_rows if r["violation"])
    if n_joint_violations > 0:
        return {"verdict": "PHASE16O_ITERATION3_JOINT_CONSISTENCY_FAILED", "n_violations": n_joint_violations}

    # Prediction preregistration
    prereg_rows = preregister_predictions(support, fresh_matrix, final_models)
    write_csv(TASK_OUTPUTS / "phase_16o_iteration3_final_validation_prediction_preregistration.csv", prereg_rows,
              ["cell_id", "replicas", "offered_rps", "target", "prediction", "family", "hyperparameter",
               "model_artifact_sha256", "support_status", "transform_space", "model_status"])
    assert len(prereg_rows) == 96

    # Reproducibility: recompute all 96 from frozen serialized artifacts on disk
    reload_models = {}
    for target in ALL_TARGETS:
        predict_fn, family, params = load_frozen_model_predict_fn_iteration3(target)
        reload_models[target] = predict_fn
    repro_rows = []
    n_mismatch = 0
    for row in prereg_rows:
        rep, rps, target = row["replicas"], row["offered_rps"], row["target"]
        if row["support_status"] != "SUPPORTED":
            recomputed = None
        else:
            recomputed = reload_models[target](rep, rps)
        original = row["prediction"]
        if original is None and recomputed is None:
            match = True
        elif original is None or recomputed is None:
            match = False
        else:
            match = math.isclose(original, recomputed, rel_tol=1e-9, abs_tol=1e-9)
        if not match:
            n_mismatch += 1
        repro_rows.append({"cell_id": row["cell_id"], "target": target, "original_prediction": original,
                            "recomputed_prediction": recomputed, "match": match})
    write_csv(TASK_OUTPUTS / "phase_16o_prediction_reproducibility_audit.csv", repro_rows,
              ["cell_id", "target", "original_prediction", "recomputed_prediction", "match"])

    return {
        "verdict": "PHASE16O_ITERATION3_MODEL_REDEVELOPMENT_AND_PREREGISTRATION_COMPLETE",
        "selection_verdicts": {t: selection_verdicts[t]["selected"]["family"] for t in TARGETS_REDEVELOP},
        "any_cv_prediction_coverage_failure": coverage_failure_found,
        "n_joint_violations": n_joint_violations,
        "n_prereg_rows": len(prereg_rows),
        "n_reproducibility_mismatches": n_mismatch,
    }


def load_frozen_model_predict_fn_iteration3(target: str) -> Tuple[Callable[[int, float], Optional[float]], str, Dict[str, Any]]:
    """Same reconstruction logic as load_frozen_model_predict_fn(), but reads
    from the NEW persistent-data/models/dynamics_iteration3/ tree (used for
    the post-write independent reproducibility check)."""
    path = IT3_MODELS_DIR / target / "model_parameters.json"
    params = json.loads(path.read_text())
    return reconstruct_predict_from_params(params), params["family"], params


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2, default=str))
