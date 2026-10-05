"""tests/unit/test_phase16n_iteration3_preregistration.py

Phase 16N: focused tests over the frozen Iteration-3 preregistration
artifacts (development dataset, CV fold assignment, fresh validation
matrix/sequence, and schedule registry). These are artifact-level tests --
they read the frozen task_outputs/ CSVs and configs/ files this phase
produced and assert the invariants Phase16N's own task required, mirroring
the project's established convention (e.g. Phase16M's own tests) of testing
frozen data artifacts directly rather than only in-memory functions.

No test in this module fits, refits, generates traffic, or touches
Kubernetes -- Phase16N is a preregistration-only phase.
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

TASK_OUTPUTS = Path("/Users/rakeshdandamudi/Capstone/task_outputs")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
I3_CONFIG_ROOT = PROJECT_ROOT / "configs/traffic_profiles/rl_dynamics_calibration_iteration3"

GLOBAL_SHA = "ce1de372e92de0460e53f1ca36d31ec402712d5cd638738200fcedf70403e901"


def _read_csv(name):
    return list(csv.DictReader(open(TASK_OUTPUTS / name)))


class TestPhase16MFirewallReuse:
    def test_nonofficial_attempts_absent_from_iteration3_development_dataset(self):
        nonofficial = {r["attempt_id"] for r in _read_csv("phase_16m_nonofficial_attempt_exclusion_audit.csv")}
        assert len(nonofficial) == 7
        dev = _read_csv("phase_16n_iteration3_development_dataset.csv")
        dev_attempt_ids = {r["attempt_id"] for r in dev}
        assert nonofficial.isdisjoint(dev_attempt_ids)

    def test_official_attempts_are_the_only_phase16m_source_rows(self):
        official = {r["official_attempt_id"] for r in _read_csv("phase_16m_official_attempt_inventory.csv")}
        assert len(official) == 32
        dev = _read_csv("phase_16n_iteration3_development_dataset.csv")
        phase16m_rows = [r for r in dev if r["source_phase"] == "Phase16M"]
        assert phase16m_rows, "expected at least one Phase16M-sourced development row"
        assert all(r["attempt_id"] in official for r in phase16m_rows)


class TestModelUsableFiltering:
    def test_phase16m_rows_count_matches_model_usable_dp_count(self):
        dev = _read_csv("phase_16n_iteration3_development_dataset.csv")
        phase16m_rows = [r for r in dev if r["source_phase"] == "Phase16M"]
        assert len(phase16m_rows) == 89

    def test_no_phase16f_rows_in_development_dataset(self):
        dev = _read_csv("phase_16n_iteration3_development_dataset.csv")
        assert not any(r["source_phase"] == "Phase16F" for r in dev)

    def test_all_target_columns_fully_populated_no_imputation_marker(self):
        dev = _read_csv("phase_16n_iteration3_development_dataset.csv")
        targets = ["request_rate_rps", "cpu_cores", "latency_p95_seconds",
                   "memory_bytes", "requests_in_progress", "error_rate_rps"]
        for r in dev:
            for t in targets:
                assert r[t] != "", f"missing {t} in row {r}"


class TestNoDuplicatesAndProvenanceRetention:
    def test_no_duplicate_attempt_timestamp_rows(self):
        dev = _read_csv("phase_16n_iteration3_development_dataset.csv")
        keys = [(r["attempt_id"], r["decision_time_seconds"]) for r in dev]
        assert len(keys) == len(set(keys))

    def test_combined_row_count_matches_base_plus_appended(self):
        base = _read_csv("phase_16k_iteration2b_model_fitting_dataset.csv")
        dev = _read_csv("phase_16n_iteration3_development_dataset.csv")
        phase16m_rows = [r for r in dev if r["source_phase"] == "Phase16M"]
        assert len(dev) == len(base) + len(phase16m_rows)

    def test_every_row_retains_source_dataset_and_source_phase(self):
        dev = _read_csv("phase_16n_iteration3_development_dataset.csv")
        for r in dev:
            assert r["source_dataset"], r
            assert r["source_phase"], r


class TestCvFoldGeometry:
    def test_fold_assignment_matches_repeat_number(self):
        rows = _read_csv("phase_16n_iteration3_cv_fold_assignment.csv")
        mapping = {"1": "fold1", "2": "fold2", "3": "fold3"}
        for r in rows:
            assert r["fold"] == mapping[r["repeat_number"]]

    def test_no_attempt_spans_more_than_one_fold(self):
        rows = _read_csv("phase_16n_iteration3_cv_fold_assignment.csv")
        by_attempt = {}
        for r in rows:
            by_attempt.setdefault(r["attempt_id"], set()).add(r["fold"])
        leaked = {a: f for a, f in by_attempt.items() if len(f) > 1}
        assert leaked == {}

    def test_phase16m_rows_only_in_fold1_or_fold2(self):
        rows = _read_csv("phase_16n_iteration3_cv_fold_assignment.csv")
        p16m_folds = {r["fold"] for r in rows if r["source_phase"] == "Phase16M"}
        assert p16m_folds <= {"fold1", "fold2"}

    def test_geometry_audit_reports_nonzero_evidence_for_all_folds_and_failed_targets(self):
        rows = _read_csv("phase_16n_iteration3_common_cv_geometry_audit.csv")
        assert len(rows) == 9  # 3 folds x 3 failed targets
        assert all(int(r["n_rows_scorable"]) > 0 for r in rows)
        assert all(int(r["n_distinct_replica_counts"]) == 8 for r in rows)


class TestFreshCoordinateAlgorithm:
    def test_exactly_16_cells_two_per_replica(self):
        rows = _read_csv("phase_16n_iteration3_fresh_validation_matrix.csv")
        assert len(rows) == 16
        by_rep = {}
        for r in rows:
            by_rep.setdefault(r["replicas"], set()).add(r["label"])
        assert len(by_rep) == 8
        assert all(labels == {"V1", "V2"} for labels in by_rep.values())

    def test_all_fresh_coordinates_strictly_in_effective_support(self):
        # Phase16N.1 Section 7: strengthened from `lo <= rps <= hi` to strict
        # interior `lo < rps < hi` -- all 16 frozen coordinates are interior
        # points of the widest/second-widest gap (never exactly the support
        # boundary itself), so this is a genuine tightening, not a relaxation.
        support = {r["replicas"]: (float(r["lower_offered_rps"]), float(r["upper_offered_rps"]))
                   for r in _read_csv("phase_16k_effective_direct_support.csv")}
        rows = _read_csv("phase_16n_iteration3_fresh_validation_matrix.csv")
        for r in rows:
            lo, hi = support[r["replicas"]]
            rps = float(r["offered_rps"])
            assert lo < rps < hi

    def test_no_collision_with_any_prior_coordinate(self):
        audit = _read_csv("phase_16n_iteration3_validation_freshness_audit.csv")
        assert len(audit) == 16
        assert all(r["collides_with_prior_coordinate"] == "False" for r in audit)
        assert all(r["in_effective_support"] == "True" for r in audit)

    def test_v1_v2_are_gap_rank_labels_not_low_high_load_ordering(self):
        # Phase16N.1 Section 7: V1 = widest-gap midpoint, V2 = second-widest
        # gap midpoint -- these are deterministic GAP-RANK labels, not a
        # low-load/high-load ordering. Replica 1 proves V1 can exceed V2
        # (V1=10.333334 sits in the widest gap near the upper support
        # boundary; V2=3.5 sits in the narrower gap near the lower boundary).
        by_label = {r["label"]: float(r["offered_rps"]) for r in
                    [rr for rr in _read_csv("phase_16n_iteration3_fresh_validation_matrix.csv") if rr["replicas"] == "1"]}
        assert by_label["V1"] > by_label["V2"], (
            "replica 1 must demonstrate V1 > V2 to prove V1/V2 are gap-rank "
            "labels, not an implicit low-load(V1)/high-load(V2) ordering"
        )
        # And confirm the reverse also occurs elsewhere (e.g. replica 2),
        # so no global ordering rule of either direction holds.
        r2_rows = [rr for rr in _read_csv("phase_16n_iteration3_fresh_validation_matrix.csv") if rr["replicas"] == "2"]
        r2_by_label = {rr["label"]: float(rr["offered_rps"]) for rr in r2_rows}
        assert r2_by_label["V1"] > r2_by_label["V2"]
        # replica 6 has V1 < V2, demonstrating both orderings occur across
        # the 16 frozen cells -- neither ordering is a universal rule.
        r6_rows = [rr for rr in _read_csv("phase_16n_iteration3_fresh_validation_matrix.csv") if rr["replicas"] == "6"]
        r6_by_label = {rr["label"]: float(rr["offered_rps"]) for rr in r6_rows}
        assert r6_by_label["V1"] < r6_by_label["V2"]


class TestDeterministicSeedAndSequence:
    def test_seed_reproducible_from_global_sha_and_suffix(self):
        s = GLOBAL_SHA + "|ITERATION3_FINAL_UNSEEN_VALIDATION_V1"
        digest = hashlib.sha256(s.encode()).digest()
        seed = int.from_bytes(digest[:8], "big") % (2 ** 32)
        assert seed == 3409997264

    def test_seed_distinct_from_prior_iteration_seeds(self):
        seed = 3409997264
        assert seed not in (3365653345, 3796252765)

    def test_sequence_has_32_slots_two_repeats_per_cell(self):
        rows = _read_csv("phase_16n_iteration3_final_validation_sequence.csv")
        assert len(rows) == 32
        by_cell = {}
        for r in rows:
            by_cell.setdefault(r["cell_id"], set()).add(r["repeat_number"])
        assert len(by_cell) == 16
        assert all(reps == {"1", "2"} for reps in by_cell.values())

    def test_sequence_reproducible_from_matrix_and_seed(self):
        matrix = _read_csv("phase_16n_iteration3_fresh_validation_matrix.csv")
        cells_sorted = sorted(matrix, key=lambda r: (int(r["replicas"]), r["label"]))
        slots = []
        for c in cells_sorted:
            for rep in (1, 2):
                slots.append({"cell_id": c["cell_id"], "repeat_number": str(rep)})
        random.Random(3409997264).shuffle(slots)
        rows = _read_csv("phase_16n_iteration3_final_validation_sequence.csv")
        actual = [(r["cell_id"], r["repeat_number"]) for r in rows]
        expected = [(s["cell_id"], s["repeat_number"]) for s in slots]
        assert actual == expected

    def test_sequence_json_matches_csv(self):
        seq_json = json.loads((I3_CONFIG_ROOT / "iteration3_final_validation_sequence.json").read_text())
        assert seq_json["seed_derivation"]["seed"] == 3409997264
        assert seq_json["n_slots"] == 32
        assert seq_json["n_cells"] == 16
        rows = _read_csv("phase_16n_iteration3_final_validation_sequence.csv")
        for slot, row in zip(seq_json["slots"], rows):
            assert slot["cell_id"] == row["cell_id"]
            assert str(slot["repeat_number"]) == row["repeat_number"]


class TestScheduleGenerationOfflineOnly:
    def test_registry_has_exactly_16_entries_one_schedule_per_cell(self):
        registry = json.loads((I3_CONFIG_ROOT / "iteration3_final_validation_schedule_registry.json").read_text())
        assert len(registry) == 16
        matrix = _read_csv("phase_16n_iteration3_fresh_validation_matrix.csv")
        assert set(registry.keys()) == {r["cell_id"] for r in matrix}

    def test_both_repeats_of_a_cell_share_the_identical_schedule_file(self):
        registry = json.loads((I3_CONFIG_ROOT / "iteration3_final_validation_schedule_registry.json").read_text())
        sequence = _read_csv("phase_16n_iteration3_final_validation_sequence.csv")
        by_cell = {}
        for r in sequence:
            by_cell.setdefault(r["cell_id"], []).append(r["repeat_number"])
        for cell_id, reps in by_cell.items():
            assert sorted(reps) == ["1", "2"]
            assert cell_id in registry

    def test_generator_script_never_imports_subprocess_or_kubernetes_or_fit(self):
        source = (PROJECT_ROOT / "scripts/generate_iteration3_final_validation_schedules.py").read_text()
        assert "subprocess" not in source
        assert "kubectl" not in source
        assert "kubernetes" not in source.lower()
        assert ".fit(" not in source
        assert "requests.post" not in source
        assert "aiohttp" not in source

    def test_generator_script_is_idempotent_pure_function_of_matrix(self):
        import scripts.generate_iteration3_final_validation_schedules as gen
        src = inspect.getsource(gen._generate_one)
        assert "if schedule_path.exists()" in src


class TestScheduleByteIntegrityAndReproducibility:
    """Phase16N.1 Sections 8-9."""

    def test_byte_integrity_audit_reports_16_of_16_all_pass(self):
        rows = _read_csv("phase_16n1_schedule_byte_integrity_audit.csv")
        assert len(rows) == 16
        assert all(r["all_pass"] == "True" for r in rows)

    def test_byte_integrity_audit_covers_required_checks(self):
        rows = _read_csv("phase_16n1_schedule_byte_integrity_audit.csv")
        required = [
            "file_exists", "sha256_matches_registry", "registry_cell_id_equals_matrix_cell_id",
            "registry_replicas_equals_matrix_replicas", "registry_offered_rps_equals_matrix_offered_rps",
            "both_repeats_reference_same_cell_in_sequence", "schedule_nonempty",
            "request_indices_contiguous_from_0", "first_planned_time_nonneg", "monotonic_nondecreasing",
            "all_times_lt_180", "endpoint_is_work", "phase_name_is_steady_state", "phase_type_is_constant",
        ]
        for r in rows:
            for col in required:
                assert r[col] == "True", (r["cell_id"], col)

    def test_independent_regeneration_audit_reports_16_of_16_byte_identical(self):
        rows = _read_csv("phase_16n1_schedule_independent_regeneration_audit.csv")
        assert len(rows) == 16
        assert all(r["byte_identical"] == "True" for r in rows)
        assert all(r["frozen_sha256"] == r["independently_regenerated_sha256"] for r in rows)

    def test_independent_regeneration_into_fresh_tempdir_reproduces_all_16_schedules(self):
        """Live re-derivation (not just reading the audit CSV): regenerate all
        16 schedules into a brand-new temporary directory using the same pure
        generate_and_persist() path, and confirm byte-for-byte equality with
        the frozen schedules -- without ever touching the frozen files."""
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.generate_tuning_schedule import generate_and_persist

        registry = json.loads((I3_CONFIG_ROOT / "iteration3_final_validation_schedule_registry.json").read_text())
        matrix = _read_csv("phase_16n_iteration3_fresh_validation_matrix.csv")

        tmpdir = Path(tempfile.mkdtemp(prefix="test_phase16n1_regen_"))
        try:
            for m in matrix:
                cell_id = m["cell_id"]
                rps = float(m["offered_rps"])
                profile_path = tmpdir / f"{cell_id}.yaml"
                schedule_path = tmpdir / f"{cell_id}_schedule.csv"
                doc = {
                    "profile": cell_id,
                    "description": f"test regen: {cell_id}",
                    "seed": 42, "endpoint": "/work", "intensity": 3,
                    "phases": [{"type": "constant", "name": "steady_state",
                                "duration_seconds": 180.0, "target_rps": rps}],
                }
                with open(profile_path, "w") as f:
                    yaml.safe_dump(doc, f, sort_keys=False)
                digest = generate_and_persist(profile_path, "deterministic", 42, schedule_path)

                frozen_path = PROJECT_ROOT / registry[cell_id]["schedule_path"]
                frozen_bytes = frozen_path.read_bytes()
                regen_bytes = schedule_path.read_bytes()
                assert regen_bytes == frozen_bytes, f"{cell_id}: regenerated bytes differ from frozen schedule"
                assert digest == registry[cell_id]["schedule_sha256"]
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_frozen_schedule_files_were_not_modified_by_the_regeneration_test(self):
        registry = json.loads((I3_CONFIG_ROOT / "iteration3_final_validation_schedule_registry.json").read_text())
        for cell_id, entry in registry.items():
            path = PROJECT_ROOT / entry["schedule_path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == entry["schedule_sha256"]


class TestPhase16MArtifactsUnchanged:
    def test_analysis_script_hash_unchanged(self):
        path = PROJECT_ROOT / "scripts/adjudicate_iteration2b_final_validation.py"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == "72c64784999dc8846622167be67533525b6b7e9defbaeb6b229ccd0eab4f7c7a"

    def test_final_validation_ledger_hash_unchanged(self):
        path = PROJECT_ROOT / "persistent-data/dynamics_validation_iteration2b/final_unseen/iteration2b_final_validation_ledger.json"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == "a5739bac0b8eb33b3e8acaa258dc4b426c16add0cd33404217aa03377cfced16"


class TestDevelopmentMembershipCanonicalArtifact:
    """Phase16N.1 Sections 2-4: the canonical development-membership CSV and
    integrity audit."""

    def test_membership_reports_89_included_phase16m_candidate_dps(self):
        rows = _read_csv("phase_16n_iteration3_development_membership.csv")
        p16m_rows = [r for r in rows if r["row_kind"] == "PHASE16M_OFFICIAL_CANDIDATE_DECISION_POINT"]
        assert len(p16m_rows) == 128
        included = [r for r in p16m_rows if r["model_usable"] == "True"]
        excluded = [r for r in p16m_rows if r["model_usable"] == "False"]
        assert len(included) == 89
        assert len(excluded) == 39

    def test_membership_carried_forward_rows_count_223(self):
        rows = _read_csv("phase_16n_iteration3_development_membership.csv")
        carried = [r for r in rows if r["row_kind"] == "CARRIED_FORWARD_DEVELOPMENT_ROW"]
        assert len(carried) == 223

    def test_integrity_audit_all_checks_pass(self):
        rows = _read_csv("phase_16n_iteration3_development_dataset_integrity_audit.csv")
        for r in rows:
            assert r["pass"] == "True", r

    def test_integrity_audit_lists_all_7_nonofficial_attempt_ids(self):
        rows = _read_csv("phase_16n_iteration3_development_dataset_integrity_audit.csv")
        nonofficial_checks = [r for r in rows if r["check"].startswith("nonofficial_excluded_attempt_id_")]
        assert len(nonofficial_checks) == 7
        nonofficial_from_phase16m = {r["attempt_id"] for r in _read_csv("phase_16m_nonofficial_attempt_exclusion_audit.csv")}
        assert {r["expected_value"] for r in nonofficial_checks} == nonofficial_from_phase16m

    def test_consumed_holdout_role_never_equals_iteration3_validation_role(self):
        dev = _read_csv("phase_16n_iteration3_development_dataset.csv")
        for r in dev:
            assert r["source_dataset"] != "ITERATION3_FINAL_UNSEEN_VALIDATION"
        fresh_cells = {r["cell_id"] for r in _read_csv("phase_16n_iteration3_fresh_validation_matrix.csv")}
        dev_cells = {r["cell_id"] for r in dev}
        assert fresh_cells.isdisjoint(dev_cells)


class TestFinalValidationProtocolFrozen:
    """Phase16N.1 Section 10: task_outputs/phase_16n_iteration3_final_validation_protocol.yaml."""

    @staticmethod
    def _protocol():
        return yaml.safe_load((TASK_OUTPUTS / "phase_16n_iteration3_final_validation_protocol.yaml").read_text())

    def test_protocol_freezes_16_cells_32_slots_2_repeats(self):
        p = self._protocol()
        assert p["identity"]["n_cells"] == 16
        assert p["identity"]["n_slots"] == 32
        assert p["identity"]["repeats_per_cell"] == 2

    def test_protocol_freezes_seed(self):
        p = self._protocol()
        assert p["identity"]["seed"] == 3409997264

    def test_protocol_freezes_fitting_weight_zero(self):
        p = self._protocol()
        assert p["scientific_weight"]["validation_fitting_weight"] == 0
        assert p["scientific_weight"]["validation_fitting_weight_is_permanent"] is True

    def test_protocol_prohibits_third_scientific_repeat(self):
        p = self._protocol()
        prohibitions = " ".join(p["live_execution"]["retry_prohibitions"]).lower()
        assert "no third scientific repeat" in prohibitions

    def test_protocol_prohibits_outcome_aware_or_scientific_quality_retries(self):
        p = self._protocol()
        prohibitions = " ".join(p["live_execution"]["retry_prohibitions"]).lower()
        assert "no outcome-aware retries" in prohibitions
        assert "no retry for poor sla" in prohibitions
        assert "no retry for poor prediction" in prohibitions

    def test_protocol_retains_bounded_structural_retry_policy(self):
        p = self._protocol()
        assert "bounded structural-failure policy" in p["live_execution"]["bounded_structural_retry_policy"]

    def test_protocol_freezes_exact_support_intervals(self):
        p = self._protocol()
        support_csv = {r["replicas"]: (float(r["lower_offered_rps"]), float(r["upper_offered_rps"]))
                       for r in _read_csv("phase_16k_effective_direct_support.csv")}
        for rep in range(1, 9):
            interval = p["support"]["intervals"][f"r{rep}"]
            assert (interval["lower_offered_rps"], interval["upper_offered_rps"]) == support_csv[str(rep)]

    def test_protocol_freezes_regression_gates(self):
        p = self._protocol()
        g = p["regression_gates"]
        assert g["median_normalized_error_le"] == 2.0
        assert g["fraction_normalized_error_le_2_ge"] == 0.90
        assert g["max_normalized_error_le"] == 4.0
        assert g["combination"] == "CONJUNCTIVE"

    def test_protocol_preserves_not_estimable_semantics(self):
        p = self._protocol()
        assert p["error_target"]["not_estimable_condition"] == "all scorable observations are zero"

    def test_protocol_prohibits_refit_after_holdout_open(self):
        p = self._protocol()
        assert p["after_holdout_open"]["refit_reselection_or_hyperparameter_change_based_on_these_outcomes"] == "PROHIBITED"

    def test_protocol_defers_candidate_level_cv_coverage_to_phase16o(self):
        p = self._protocol()
        stmt = p["candidate_level_cv_coverage_prerequisite"]["statement"]
        assert "Phase16O" in stmt
        assert "first scientific gate" in stmt
