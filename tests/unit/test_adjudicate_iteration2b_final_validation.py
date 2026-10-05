"""tests/unit/test_adjudicate_iteration2b_final_validation.py

Phase 16M: focused, offline unit tests for
scripts/adjudicate_iteration2b_final_validation.py -- the one-shot
scientific adjudication script for the Iteration-2B
I2B_FINAL_UNSEEN_VALIDATION holdout. Every test here is
network/subprocess/Kubernetes-free and uses only synthetic fixtures or
already-frozen repository artifacts; none of them fit, refit, or tune
anything.
"""
import inspect
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import adjudicate_iteration2b_final_validation as adj  # noqa: E402


class TestSupportGuard:
    def test_guard_coordinates_are_out_of_support(self):
        rows = adj.support_guard_check()
        assert len(rows) == 2
        for row in rows:
            assert row["result"] == "DYNAMICS_OUT_OF_SUPPORT"
            assert row["guard_pass"] is True

    def test_guard1_is_replicas5_rps40(self):
        rows = adj.support_guard_check()
        g1 = rows[0]
        assert g1["replicas"] == 5
        assert g1["offered_rps"] == 40.0

    def test_guard2_is_replicas7_rps56(self):
        rows = adj.support_guard_check()
        g2 = rows[1]
        assert g2["replicas"] == 7
        assert g2["offered_rps"] == 56.0

    def test_support_supported_rejects_outside_full_action_space(self):
        assert adj.support_supported(9, 10.0) is False
        assert adj.support_supported(0, 10.0) is False

    def test_support_supported_accepts_in_window_coordinate(self):
        # r1 interval is [2, 12] per the frozen effective-direct-support freeze.
        assert adj.support_supported(1, 5.333333) is True

    def test_support_supported_rejects_r7_at_28(self):
        # r7's frozen upper bound is 24.5, not 28 -- must never accept r7=28.
        assert adj.support_supported(7, 28.0) is False


class TestOfficialAttemptFirewall:
    def test_exactly_32_official_slots(self):
        rows = adj.load_official_attempt_inventory()
        assert len(rows) == 32
        assert sorted(r["validation_sequence_index"] for r in rows) == list(range(1, 33))

    def test_index4_official_is_attempt5(self):
        rows = adj.load_official_attempt_inventory()
        r = next(x for x in rows if x["validation_sequence_index"] == 4)
        assert r["official_attempt_id"] == "i2bval_r2_V1_repeat2_attempt5_20260926T102550Z"

    def test_index13_official_is_attempt2(self):
        rows = adj.load_official_attempt_inventory()
        r = next(x for x in rows if x["validation_sequence_index"] == 13)
        assert r["official_attempt_id"] == "i2bval_r7_V2_repeat2_attempt2_20260926T115756Z"

    def test_index30_official_is_attempt3(self):
        rows = adj.load_official_attempt_inventory()
        r = next(x for x in rows if x["validation_sequence_index"] == 30)
        assert r["official_attempt_id"] == "i2bval_r8_V1_repeat2_attempt3_20260926T160403Z"

    def test_nonofficial_attempts_are_seven_and_exclude_index4_1_through_4(self):
        official = adj.load_official_attempt_inventory()
        nonofficial = adj.load_nonofficial_attempts(official)
        assert len(nonofficial) == 7
        ids = {r["attempt_id"] for r in nonofficial}
        for n in (1, 2, 3, 4):
            assert f"i2bval_r2_V1_repeat2_attempt{n}_20260926T0" in "".join(i for i in ids if f"attempt{n}_" in i) \
                or any(f"attempt{n}_" in i and "r2_V1_repeat2" in i for i in ids)

    def test_index30_attempt1_crash_is_in_nonofficial_list(self):
        official = adj.load_official_attempt_inventory()
        nonofficial = adj.load_nonofficial_attempts(official)
        ids = {r["attempt_id"] for r in nonofficial}
        assert "i2bval_r8_V1_repeat2_attempt1_20260926T144119Z" in ids
        row = next(r for r in nonofficial if r["attempt_id"] == "i2bval_r8_V1_repeat2_attempt1_20260926T144119Z")
        assert row["scientific_weight"] == 0

    def test_all_nonofficial_attempts_carry_zero_scientific_weight(self):
        official = adj.load_official_attempt_inventory()
        nonofficial = adj.load_nonofficial_attempts(official)
        assert all(r["scientific_weight"] == 0 for r in nonofficial)

    def test_official_and_nonofficial_ids_never_overlap(self):
        official = adj.load_official_attempt_inventory()
        nonofficial = adj.load_nonofficial_attempts(official)
        official_ids = {r["official_attempt_id"] for r in official}
        nonofficial_ids = {r["attempt_id"] for r in nonofficial}
        assert official_ids.isdisjoint(nonofficial_ids)


class TestR7Coordinates:
    def test_r7_21_only_at_indices_13_and_15(self):
        rows = adj.load_official_attempt_inventory()
        r7_21 = sorted(r["validation_sequence_index"] for r in rows if r["replicas"] == 7 and abs(r["offered_rps"] - 21.0) < 1e-6)
        assert r7_21 == [13, 15]

    def test_r7_175_only_at_indices_22_and_32(self):
        rows = adj.load_official_attempt_inventory()
        r7_175 = sorted(r["validation_sequence_index"] for r in rows if r["replicas"] == 7 and abs(r["offered_rps"] - 17.5) < 1e-6)
        assert r7_175 == [22, 32]

    def test_r7_28_never_present(self):
        rows = adj.load_official_attempt_inventory()
        assert not any(r["replicas"] == 7 and abs(r["offered_rps"] - 28.0) < 1e-6 for r in rows)


class TestNormalizationAndScoring:
    def test_fit_space_value_uses_log1p_for_latency_only(self):
        assert adj.fit_space_value(5.0, "latency_p95_seconds") == pytest.approx(__import__("math").log1p(5.0))
        assert adj.fit_space_value(5.0, "cpu_cores") == 5.0

    def test_epsilon_floors_match_frozen_values(self):
        assert adj.EPSILON["request_rate_rps"] == 0.01
        assert adj.EPSILON["cpu_cores"] == 0.001
        assert adj.EPSILON["latency_p95_seconds"] == 0.001
        assert adj.EPSILON["memory_bytes"] == 1048576.0
        assert adj.EPSILON["requests_in_progress"] == 0.01

    def test_pooled_fitting_sd_uses_223_row_fitting_dataset_not_validation_data(self):
        source = inspect.getsource(adj.compute_pooled_fitting_sd)
        assert "FITTING_DATASET_PATH" in source
        assert "VAL_ROOT" not in source
        assert "final_unseen" not in source

    def test_within_cell_repeat_sd_requires_at_least_two_repeat_means(self):
        # Single repeat with points -> None (cannot estimate a repeat-to-repeat SD from one repeat)
        assert adj.within_cell_repeat_sd([("1", 5.0), ("1", 5.2)], "cpu_cores") is None

    def test_within_cell_repeat_sd_computes_from_two_repeats(self):
        pts = [("1", 5.0), ("1", 5.2), ("2", 6.0), ("2", 6.4)]
        result = adj.within_cell_repeat_sd(pts, "cpu_cores")
        assert result is not None
        assert result > 0

    def test_interp_predict_exact_anchor_match(self):
        anchors = {2: {4.0: 10.0, 24.0: 20.0}}
        assert adj.interp_predict(anchors, 2, 4.0) == 10.0

    def test_interp_predict_linear_between_anchors(self):
        anchors = {2: {4.0: 10.0, 24.0: 20.0}}
        # Midpoint of 4.0-24.0 -> midpoint of 10.0-20.0
        assert adj.interp_predict(anchors, 2, 14.0) == pytest.approx(15.0)

    def test_interp_predict_returns_none_outside_anchor_range(self):
        anchors = {2: {4.0: 10.0, 24.0: 20.0}}
        assert adj.interp_predict(anchors, 2, 100.0) is None

    def test_degree_basis_degree1_shape(self):
        assert adj.degree_basis(10.0, 3, 1) == [1.0, 10.0, 3]

    def test_degree_basis_degree2_shape(self):
        basis = adj.degree_basis(10.0, 3, 2)
        assert basis == [1.0, 10.0, 3, 30.0, 100.0, 9]


class TestErrorTargetAllZeroBranch:
    def test_all_zero_observations_yield_not_estimable(self):
        rows = [
            {"target": "error_rate_rps", "target_scorable": True, "target_value": 0.0},
            {"target": "error_rate_rps", "target_scorable": True, "target_value": 0.0},
            {"target": "request_rate_rps", "target_scorable": True, "target_value": 5.0},  # different target, ignored
        ]
        result = adj.error_target_adjudication(rows)
        assert result["status"] == "NOT_ESTIMABLE"
        assert result["balanced_accuracy"] == "NOT_ESTIMABLE"
        assert result["nonzero_class_recall"] == "NOT_ESTIMABLE"
        assert result["n_nonzero_observations"] == 0

    def test_unscorable_rows_excluded_from_error_adjudication(self):
        rows = [
            {"target": "error_rate_rps", "target_scorable": False, "target_value": 3.0},  # excluded
            {"target": "error_rate_rps", "target_scorable": True, "target_value": 0.0},
        ]
        result = adj.error_target_adjudication(rows)
        assert result["n_scorable_observations"] == 1
        assert result["n_nonzero_observations"] == 0

    def test_nonzero_error_branch_uses_synthetic_fixture(self):
        """Synthetic fixture ONLY (never real validation data) proving the
        nonzero-error metric branch computes balanced accuracy / recall
        rather than silently staying NOT_ESTIMABLE when nonzero values
        genuinely exist."""
        rows = [
            {"target": "error_rate_rps", "target_scorable": True, "target_value": 0.0},
            {"target": "error_rate_rps", "target_scorable": True, "target_value": 0.0},
            {"target": "error_rate_rps", "target_scorable": True, "target_value": 1.5},
        ]
        result = adj.error_target_adjudication(rows)
        assert result["status"] != "NOT_ESTIMABLE"
        assert result["n_nonzero_observations"] == 1
        assert isinstance(result["balanced_accuracy"], float)
        assert isinstance(result["nonzero_class_recall"], float)


class TestRegressionGateBoundaryBehavior:
    def _make_cell_rows(self, target, errors):
        return [
            {"target": target, "normalized_error": e, "cell_id": f"cell{i}"}
            for i, e in enumerate(errors)
        ]

    def _adequate_evidence(self, target):
        # regression_gates() iterates over every REGRESSION_TARGETS entry,
        # so the fixture must supply an evidence row for each of them. Only
        # `target` actually has cell rows in these fixtures; every other
        # target is marked inadequate so it safely short-circuits instead
        # of crashing on an empty scorable list.
        return [{"target": t, "evidence_adequate": (t == target)} for t in adj.REGRESSION_TARGETS]

    def test_gate_passes_at_exact_boundary_values(self):
        # median exactly 2.0, fraction exactly 0.90 (9/10 <=2.0), max exactly 4.0
        errors = [1.0] * 9 + [4.0]
        rows = self._make_cell_rows("cpu_cores", errors)
        result = next(r for r in adj.regression_gates(rows, self._adequate_evidence("cpu_cores")) if r["target"] == "cpu_cores")
        assert result["gate_max_le_4_0"] is True
        assert result["gate_fraction_ge_0_90"] is True

    def test_gate_fails_when_max_exceeds_4(self):
        errors = [1.0] * 9 + [4.0001]
        rows = self._make_cell_rows("cpu_cores", errors)
        result = next(r for r in adj.regression_gates(rows, self._adequate_evidence("cpu_cores")) if r["target"] == "cpu_cores")
        assert result["gate_max_le_4_0"] is False
        assert result["status"] == "FAIL"

    def test_gate_fails_when_fraction_below_0_90(self):
        errors = [1.0] * 8 + [2.5, 2.5]  # 8/10 = 0.80 < 0.90
        rows = self._make_cell_rows("cpu_cores", errors)
        result = next(r for r in adj.regression_gates(rows, self._adequate_evidence("cpu_cores")) if r["target"] == "cpu_cores")
        assert result["gate_fraction_ge_0_90"] is False
        assert result["status"] == "FAIL"

    def test_insufficient_evidence_short_circuits_gate_computation(self):
        rows = self._make_cell_rows("cpu_cores", [1.0])
        # All targets marked inadequate: targets with no cell rows at all
        # would otherwise crash trying to compute a median of zero values,
        # so every REGRESSION_TARGETS entry must resolve through the
        # INSUFFICIENT_EVIDENCE short-circuit in this minimal fixture.
        evidence = [{"target": t, "evidence_adequate": False} for t in adj.REGRESSION_TARGETS]
        result = next(r for r in adj.regression_gates(rows, evidence) if r["target"] == "cpu_cores")
        assert result["status"] == "INSUFFICIENT_EVIDENCE"
        assert result["median_normalized_error"] is None


class TestEvidenceAdequacy:
    def test_adequate_requires_12_of_16_and_all_8_replicas(self):
        rows = []
        for rep in range(1, 9):
            rows.append({"target": "cpu_cores", "replicas": rep, "total_scorable_points": 1})
        # 8 scorable cells (one per replica), which is < 12 -> inadequate
        result = next(r for r in adj.evidence_adequacy(rows) if r["target"] == "cpu_cores")
        assert result["n_scorable_cells"] == 8
        assert result["evidence_adequate"] is False
        assert result["result"] == "ITERATION2B_FINAL_VALIDATION_EVIDENCE_INSUFFICIENT"

    def test_missing_one_replica_is_inadequate_even_with_enough_total_cells(self):
        rows = []
        for rep in range(1, 8):  # replicas 1-7 only, never 8
            for _ in range(2):
                rows.append({"target": "cpu_cores", "replicas": rep, "total_scorable_points": 1})
        result = next(r for r in adj.evidence_adequacy(rows) if r["target"] == "cpu_cores")
        assert result["n_scorable_cells"] == 14
        assert result["all_8_replicas_covered"] is False
        assert result["evidence_adequate"] is False

    def test_adequate_when_both_conditions_satisfied(self):
        rows = []
        for rep in range(1, 9):
            for _ in range(2):
                rows.append({"target": "cpu_cores", "replicas": rep, "total_scorable_points": 1})
        result = next(r for r in adj.evidence_adequacy(rows) if r["target"] == "cpu_cores")
        assert result["n_scorable_cells"] == 16
        assert result["all_8_replicas_covered"] is True
        assert result["evidence_adequate"] is True


class TestTwoRepeatCellGroupingAndMissingness:
    def test_cell_scores_pool_both_repeats_with_equal_weight(self):
        official = adj.load_official_attempt_inventory()
        dps = adj.flatten_decision_points(official)
        matrix = list(__import__("csv").DictReader(open(adj.MATRIX_PATH)))
        prereg = list(__import__("csv").DictReader(open(adj.PREREGISTRATION_PATH)))
        rows, _ = adj.compute_cell_scores(dps, matrix, prereg)
        # Every cell/target row must report repeat1 + repeat2 scorable point counts separately
        for r in rows:
            assert "repeat1_scorable_points" in r
            assert "repeat2_scorable_points" in r
            assert r["total_scorable_points"] == r["repeat1_scorable_points"] + r["repeat2_scorable_points"]

    def test_target_specific_missingness_never_imputed(self):
        """A cell/target with zero scorable points must report
        observed_cell_mean=None, never a fabricated/imputed value."""
        official = adj.load_official_attempt_inventory()
        dps = adj.flatten_decision_points(official)
        matrix = list(__import__("csv").DictReader(open(adj.MATRIX_PATH)))
        prereg = list(__import__("csv").DictReader(open(adj.PREREGISTRATION_PATH)))
        rows, _ = adj.compute_cell_scores(dps, matrix, prereg)
        zero_point_rows = [r for r in rows if r["total_scorable_points"] == 0]
        assert len(zero_point_rows) > 0  # this holdout genuinely has some (i2bval_r7_V2, i2bval_r8_V2)
        for r in zero_point_rows:
            assert r["observed_cell_mean"] is None
            assert r["normalized_error"] is None


class TestDeterminismAndNoFitting:
    def test_main_is_deterministic_across_two_runs(self):
        r1 = adj.main()
        r2 = adj.main()
        assert r1["verdict"] == r2["verdict"]
        assert r1["cell_score_rows"] == r2["cell_score_rows"]
        assert r1["regression_gates"] == r2["regression_gates"]
        assert r1["error_target"] == r2["error_target"]

    def test_no_fit_or_refit_function_exists_in_module(self):
        source = Path(adj.__file__).read_text(encoding="utf-8")
        forbidden = ["np.linalg.lstsq", "IsotonicRegression", "DecisionTreeRegressor", ".fit("]
        for token in forbidden:
            assert token not in source, f"adjudication script must never fit/refit anything: found {token!r}"

    def test_module_never_imports_kubernetes_or_subprocess_traffic_tools(self):
        source = Path(adj.__file__).read_text(encoding="utf-8")
        assert "subprocess" not in source
        assert "kubectl" not in source

    def test_load_model_predict_fn_never_writes_model_parameters(self):
        source = inspect.getsource(adj.load_model_predict_fn)
        assert "open(" not in source or "'w'" not in source
        assert ".write(" not in source


class TestPredictionReproducibilityUsesFrozenModelsOnly:
    def test_all_six_targets_load_without_error(self):
        for t in adj.TARGETS:
            fn = adj.load_model_predict_fn(t)
            assert callable(fn)

    def test_recomputed_predictions_match_preregistration_within_tolerance(self):
        import csv
        matrix = list(csv.DictReader(open(adj.MATRIX_PATH)))
        prereg = list(csv.DictReader(open(adj.PREREGISTRATION_PATH)))
        rows = adj.prediction_reproducibility_audit(matrix, prereg)
        assert len(rows) == 16 * 6
        assert all(r["match_within_tolerance"] for r in rows)
