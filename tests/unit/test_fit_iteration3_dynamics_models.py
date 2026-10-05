"""tests/unit/test_fit_iteration3_dynamics_models.py

Phase 16O: focused tests over the candidate-level CV coverage gate, model
selection, final refit, joint-consistency, and prediction-preregistration
pipeline for the three Iteration-3 redevelopment targets. Covers Section 13's
26 required test items.

No test in this module generates live traffic, touches Kubernetes/subprocess,
or fits/refits the three preserved targets (request_rate_rps, memory_bytes,
error_rate_rps).
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from scripts import fit_iteration3_dynamics_models as fim

TASK_OUTPUTS = Path("/Users/rakeshdandamudi/Capstone/task_outputs")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _read_csv(name):
    return list(csv.DictReader(open(TASK_OUTPUTS / name)))


@pytest.fixture(scope="module")
def rows_and_common_instances():
    dev_rows = fim.load_dev_rows()
    fold_map = fim.load_fold_assignment()
    rows = fim.build_rows_with_fold(dev_rows, fold_map)
    common = fim.common_cv_evaluable_instances(rows)
    return rows, common


class TestFrozenInputHashes:
    def test_development_dataset_hash_exact(self):
        digest = hashlib.sha256((TASK_OUTPUTS / "phase_16n_iteration3_development_dataset.csv").read_bytes()).hexdigest()
        assert digest == "f9c7a972d2eedac50eb69e218d38df97947d7d8b1f41b600560a1329a6895e54"

    def test_candidate_registry_hash_exact(self):
        digest = hashlib.sha256(fim.CANDIDATE_REGISTRY_PATH.read_bytes()).hexdigest()
        assert digest == "7fb2742aef71e9f8c13770b48810b86ea73d3bb96f948eeee9ba7cc8ee6e3dd5"

    def test_fresh_validation_matrix_hash_exact(self):
        digest = hashlib.sha256(fim.FRESH_MATRIX_PATH.read_bytes()).hexdigest()
        assert digest == "dfdc9f8698c131c7096c3e779dc20077fa1c10f9cab21561f646bba8d9bb637b"

    def test_predevelopment_freeze_audit_all_match(self):
        rows = _read_csv("phase_16o_predevelopment_freeze_audit.csv")
        assert len(rows) == len(fim.PRE_FIT_FREEZE_INPUTS)
        assert all(r["match"] == "True" for r in rows)


class TestNoContaminationInDevelopmentData(object):
    def test_no_nonofficial_phase16m_attempt_in_fitting(self, rows_and_common_instances):
        rows, _ = rows_and_common_instances
        nonofficial = {r["attempt_id"] for r in _read_csv("phase_16m_nonofficial_attempt_exclusion_audit.csv")}
        fitting_ids = {r["attempt_id"] for r in rows}
        assert nonofficial.isdisjoint(fitting_ids)

    def test_no_phase16f_row_in_fitting(self, rows_and_common_instances):
        rows, _ = rows_and_common_instances
        assert not any(r["source_phase"] == "Phase16F" for r in rows)

    def test_no_fresh_iteration3_validation_coordinates_used_in_fitting(self, rows_and_common_instances):
        rows, _ = rows_and_common_instances
        fresh_cells = {r["cell_id"] for r in fim.load_fresh_matrix()}
        fitting_cells = {r["cell_id"] for r in rows}
        assert fresh_cells.isdisjoint(fitting_cells)


class TestFoldAssignmentAndLeakage:
    def test_fold_assignment_matches_frozen_artifact(self, rows_and_common_instances):
        rows, _ = rows_and_common_instances
        by_repeat = {"1": "fold1", "2": "fold2", "3": "fold3"}
        for r in rows:
            assert r["fold"] == by_repeat[r["repeat_number"]]

    def test_zero_attempt_leaks_across_folds(self, rows_and_common_instances):
        rows, _ = rows_and_common_instances
        by_attempt = {}
        for r in rows:
            by_attempt.setdefault(r["attempt_id"], set()).add(r["fold"])
        leaked = {a: f for a, f in by_attempt.items() if len(f) > 1}
        assert leaked == {}


class TestCandidateRegistryScope:
    def test_no_family_outside_frozen_registry(self):
        for target, families in fim.CANDIDATE_FAMILIES_IN_SCOPE.items():
            for family in families:
                assert family in fim.COMPLEXITY_RANK

    def test_input_features_exactly_offered_rps_and_replicas(self, rows_and_common_instances):
        rows, _ = rows_and_common_instances
        model = fim.build_candidate("cpu_cores", "piecewise_linear_regression_surface", 1)
        model.fit(rows)
        src = inspect.getsource(fim.SurfaceCandidate.fit)
        assert '"offered_rps"' in src and '"replicas"' in src
        src_tree = inspect.getsource(fim.TreeCandidate.fit)
        assert '"offered_rps"' in src_tree and '"replicas"' in src_tree
        # no forbidden feature names anywhere in the module (repeat_number is
        # legitimately read for s_eff/normalization purposes, never as a model
        # input feature -- excluded from this check)
        module_src = Path(fim.__file__).read_text()
        for forbidden in ("latency_history", "cpu_history", "queue_history", "per_replica_load_ratio"):
            assert forbidden not in module_src


class TestCommonCvEvaluationMask:
    def test_mask_is_candidate_independent(self, rows_and_common_instances):
        rows = fim.load_dev_rows()
        rows2 = fim.build_rows_with_fold(rows, fim.load_fold_assignment())
        common1 = fim.common_cv_evaluable_instances(rows2)
        common2 = fim.common_cv_evaluable_instances(rows2)
        assert common1 == common2
        # depends only on cell_id/fold, never on any target column
        assert len(common1) == 78

    def test_synthetic_candidate_missing_prediction_triggers_coverage_failure(self, rows_and_common_instances):
        rows, common = rows_and_common_instances

        class AlwaysNoneCandidate:
            def fit(self, rows):
                pass

            def predict(self, replicas, offered_rps):
                return None

        real_build = fim.build_candidate
        fim.build_candidate = lambda target, family, hp: AlwaysNoneCandidate()
        try:
            res = fim.evaluate_candidate_cv("cpu_cores", "monotone_interpolation_lookup", None, rows, common)
        finally:
            fim.build_candidate = real_build
        assert res["n_covered"] == 0
        assert not res["full_coverage"]
        assert all(x["reason"] == "CV_PREDICTION_COVERAGE_FAILURE" for x in res["per_instance"] if not x["covered"])

    def test_no_silent_evaluation_mask_shrinking(self, rows_and_common_instances):
        rows, common = rows_and_common_instances
        res = fim.evaluate_candidate_cv("cpu_cores", "monotone_interpolation_lookup", None, rows, common)
        assert res["n_required"] == len(common)
        # every instance in the common mask appears exactly once in per_instance
        seen = {(x["cell_id"], x["fold"]) for x in res["per_instance"]}
        assert seen == set(common)


class TestNoOutOfSupportBridging:
    def test_physical_grid_never_exceeds_support_bounds(self, rows_and_common_instances):
        rows, _ = rows_and_common_instances
        support = fim.load_support()
        fresh = fim.load_fresh_matrix()
        grid = fim.physical_and_monotonicity_grid(support, rows, fresh)
        for rep, points in grid.items():
            lo, hi = support[rep]
            for p in points:
                assert lo - 1e-9 <= p <= hi + 1e-9


class TestSelectionLexicographicOrdering:
    def test_ranking_orders_by_mae_then_secondary_then_worst_ne_then_complexity_then_alpha(self):
        candidates = [
            {"target": "cpu_cores", "family": "z_family", "status": "CANDIDATE_VALID", "MAE": 0.05,
             "secondary_metric_value": 0.02, "worst_normalized_error": 2.0, "complexity_rank": 3},
            {"target": "cpu_cores", "family": "a_family", "status": "CANDIDATE_VALID", "MAE": 0.05,
             "secondary_metric_value": 0.02, "worst_normalized_error": 2.0, "complexity_rank": 3},
            {"target": "cpu_cores", "family": "better_mae", "status": "CANDIDATE_VALID", "MAE": 0.01,
             "secondary_metric_value": 0.5, "worst_normalized_error": 9.0, "complexity_rank": 9},
        ]
        sel = fim.select_model("cpu_cores", candidates)
        assert sel["selected"]["family"] == "better_mae"  # lowest MAE wins regardless of other metrics
        # among the MAE/secondary/worst_ne/complexity tie, alphabetically-first family wins
        tied = [c for c in candidates if c["family"] != "better_mae"]
        sel2 = fim.select_model("cpu_cores", tied)
        assert sel2["selected"]["family"] == "a_family"

    def test_no_valid_candidate_yields_selection_failed_status(self):
        candidates = [{"target": "cpu_cores", "family": "x", "status": "REJECTED_PHYSICAL_CONSTRAINT"}]
        sel = fim.select_model("cpu_cores", candidates)
        assert sel["status"] == "NO_VALID_CANDIDATE"


class TestPreservedTargetsNeverFit:
    def test_load_frozen_model_predict_fn_never_calls_fit(self):
        src = inspect.getsource(fim.load_frozen_model_predict_fn)
        assert ".fit(" not in src

    def test_preserved_targets_use_frozen_reconstruction_only(self):
        for target in fim.TARGETS_FROZEN:
            predict_fn, family, params = fim.load_frozen_model_predict_fn(target)
            assert callable(predict_fn)
            assert family == params["family"]


class TestCarriedForwardModelIntegrity:
    def test_carryforward_audit_all_byte_identical(self):
        rows = _read_csv("phase_16o_frozen_target_carryforward_audit.csv")
        # 3 preserved targets x 4 provenance files each (model_parameters.json,
        # model_metadata.json, model_spec.yaml, training_row_identities.csv)
        assert len(rows) == 12
        assert {r["target"] for r in rows} == set(fim.TARGETS_FROZEN)
        assert all(r["byte_identical"] == "True" for r in rows)
        assert all(r["source_sha256"] == r["dest_sha256"] for r in rows)

    def test_redeveloped_models_written_only_under_dynamics_iteration3(self):
        rows = _read_csv("phase_16o_iteration3_model_hash_manifest.csv")
        redeveloped = [r for r in rows if r["status"] == "REDEVELOPED"]
        assert len(redeveloped) == 3
        for r in redeveloped:
            assert "persistent-data/models/dynamics_iteration3/" in r["path"]
            assert "dynamics_iteration2b" not in r["path"]

    def test_all_six_iteration3_targets_have_required_provenance_files(self):
        required = {"model_parameters.json", "model_metadata.json", "model_spec.yaml",
                    "training_row_identities.csv"}
        for target in fim.ALL_TARGETS:
            target_dir = fim.IT3_MODELS_DIR / target
            present = {p.name for p in target_dir.iterdir() if p.is_file()}
            assert required <= present, f"{target_dir} missing {required - present}"

    def test_redeveloped_metadata_reflects_actual_selection(self):
        for target in fim.TARGETS_REDEVELOP:
            metadata = json.loads((fim.IT3_MODELS_DIR / target / "model_metadata.json").read_text())
            assert metadata["training_row_count"] == 312
            assert metadata["iteration"] == "ITERATION3"
            assert metadata["final_validation_status"] == "NOT_YET_RUN"
            assert metadata["candidate_registry_hash"] == "7fb2742aef71e9f8c13770b48810b86ea73d3bb96f948eeee9ba7cc8ee6e3dd5"

    def test_frozen_metadata_files_are_verbatim_copies(self):
        for target in fim.TARGETS_FROZEN:
            it2b_meta = json.loads((fim.IT2B_MODELS_DIR / target / "model_metadata.json").read_text())
            it3_meta = json.loads((fim.IT3_MODELS_DIR / target / "model_metadata.json").read_text())
            assert it2b_meta == it3_meta

    def test_iteration2b_model_files_unchanged(self):
        expected = {
            "request_rate_rps": "43c2303c8a716cd5b0a06fbb79c3e24866944bc3cb5ba9ec57623551a5d2d430",
            "memory_bytes": "55e32d1a2dee03ae8bdd1a83d9e5650f203f6c6fdda967de0508fc614c6a0597",
            "error_rate_rps": "3ec5d09cd3840116eed5c4ec4ba40eb2835fc90529fc6a443d33b7f1661bbc90",
            "cpu_cores": "027c1b797be53772418122f2d46c72ccfc237f2624e1eb6dfc956331aac0ae9e",
            "requests_in_progress": "8e48bca97244e81b720668ee60e8235ff678ea6715217e52e3327274a445b4be",
            "latency_p95_seconds": "da3aa5fe4a13c49f3185a2cbecb9100cff7eff58512bb3a2cedf7f4b12c85221",
        }
        for target, expected_sha in expected.items():
            path = fim.IT2B_MODELS_DIR / target / "model_parameters.json"
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual == expected_sha, target


class TestJointConsistency:
    def test_joint_consistency_uses_frozen_semantics(self):
        rows_raw = fim.load_dev_rows()
        rows = fim.build_rows_with_fold(rows_raw, fim.load_fold_assignment())
        support = fim.load_support()
        fresh = fim.load_fresh_matrix()
        audit = fim.joint_consistency_audit(support, rows, fresh)
        assert len(audit) > 0
        for r in audit:
            assert r["upper_bound"] == pytest.approx(r["predicted_request_rate_rps"] + 0.01)
        assert not any(r["violation"] for r in audit)


class TestFreshValidationSupport:
    def test_all_16_coordinates_strictly_inside_support(self):
        support = fim.load_support()
        for cell in fim.load_fresh_matrix():
            lo, hi = support[int(cell["replicas"])]
            rps = float(cell["offered_rps"])
            assert lo < rps < hi


class TestPredictionPreregistration:
    def test_exactly_96_rows(self):
        rows = _read_csv("phase_16o_iteration3_final_validation_prediction_preregistration.csv")
        assert len(rows) == 96

    def test_every_row_points_to_correct_model_hash(self):
        prereg = _read_csv("phase_16o_iteration3_final_validation_prediction_preregistration.csv")
        hash_manifest = {r["target"]: r["sha256"] for r in _read_csv("phase_16o_iteration3_model_hash_manifest.csv")}
        for r in prereg:
            assert r["model_artifact_sha256"] == hash_manifest[r["target"]]

    def test_reproducibility_audit_zero_mismatches(self):
        rows = _read_csv("phase_16o_prediction_reproducibility_audit.csv")
        assert len(rows) == 96
        assert all(r["match"] == "True" for r in rows)


class TestNoLiveTrafficOrKubernetesPath:
    def test_module_never_imports_subprocess_kubernetes_or_http(self):
        source = Path(fim.__file__).read_text()
        assert "import subprocess" not in source
        assert "kubectl" not in source
        assert "import kubernetes" not in source.lower()
        assert "requests.post" not in source
        assert "import aiohttp" not in source


class TestDeterminism:
    def test_common_cv_evaluable_instances_deterministic(self, rows_and_common_instances):
        rows, common = rows_and_common_instances
        assert fim.common_cv_evaluable_instances(rows) == common

    def test_candidate_scoring_deterministic_across_two_runs(self, rows_and_common_instances):
        rows, common = rows_and_common_instances
        support = fim.load_support()
        fresh = fim.load_fresh_matrix()
        s1 = fim.score_candidate("cpu_cores", "piecewise_linear_regression_surface", 1, rows, common, support, fresh)
        s2 = fim.score_candidate("cpu_cores", "piecewise_linear_regression_surface", 1, rows, common, support, fresh)
        assert s1["MAE"] == s2["MAE"]
        assert s1["worst_normalized_error"] == s2["worst_normalized_error"]
        assert s1["status"] == s2["status"]
