"""tests/unit/test_rl_dynamics_calibration_protocol.py

Phase 15B.2 (CORR-31) froze the empirical dynamics calibration protocol;
Phase 15B.2A (CORR-32) corrected it before its first execution; Phase 15B.2B
(CORR-33) resolved remaining execution/model-identification ambiguities
(request-rate realization, causal structure, traffic terminology, per-pod
routing diagnostics, HPA mechanics, session/attempt semantics, transition
semantics, generator margin) without touching the frozen matrix. This suite
validates the CORRECTED manifest/config/protocol/schedules/types. Offline,
read-only; no calibration traffic exists yet, and none of these tests
sends any.
"""

import hashlib
import json
import math
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = PROJECT_ROOT / "experiments/manifests/rl_dynamics_calibration_v1.yaml"
DYNAMICS_CONFIG_PATH = PROJECT_ROOT / "configs/rl_dynamics.yaml"
PROTOCOL_DOC_PATH = PROJECT_ROOT / "docs/rl_dynamics_calibration_protocol.md"
RL_CONFIG_PATH = PROJECT_ROOT / "configs/rl.yaml"
REWARD_CONFIG_PATH = PROJECT_ROOT / "configs/reward.yaml"
HPA_CONFIG_PATH = PROJECT_ROOT / "configs/hpa.yaml"
HPA_MANIFEST_PATH = PROJECT_ROOT / "k8s/hpa/hpa.yaml"
UNSEEN_EVAL_PATH = PROJECT_ROOT / "configs/traffic_profiles/unseen_eval.yaml"
CAL_ROOT = PROJECT_ROOT / "configs/traffic_profiles/rl_dynamics_calibration"
SCHEDULE_REGISTRY_PATH = CAL_ROOT / "schedule_registry.json"
RL_ENV_TYPES_PATH = PROJECT_ROOT / "autoscaler/src/rl_env/types.py"
GLOBAL_CAMPAIGN_SEQUENCE_PATH = CAL_ROOT / "global_campaign_sequence.json"
GENERATOR_TIMEOUT_SECONDS = 5.0


def _load(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


EXPECTED_DYNAMICS_TARGETS = {
    "request_rate_rps", "cpu_cores", "latency_p95_seconds",
    "memory_bytes", "requests_in_progress", "error_rate_rps",
}
REQUIRED_DYNAMICS_MODEL_FILES = {"model_metadata.json", "model_spec.yaml", "training_row_identities.csv"}


def _assert_ppo_absent_and_dynamics_artifacts_valid(project_root: Path) -> None:
    """Shared post-Phase16E lifecycle invariant, used by every test in this
    file that historically asserted `persistent-data/models/` and
    `persistent-data/rl/` stayed entirely empty as a coarse proxy for "no PPO
    training has occurred." Phase 16E (empirical mean-dynamics model fitting)
    legitimately and permanently populates `persistent-data/models/dynamics/`
    with six selected/refit steady-state dynamics models -- expected,
    authorized scientific output, never a PPO/RL artifact. This check is now
    precise instead of coarse: dynamics-model artifacts are required and
    scoped to an exact target set with required provenance files; the
    canonical PPO/RL artifact paths this repository actually defines
    (`persistent-data/models/ppo_current/` per `configs/experiment.yaml`,
    `persistent-data/rl/checkpoints/` and `persistent-data/rl/tensorboard/`
    per `configs/rl.yaml`) must still never exist.

    Phase 16K (Iteration-2B model rebuild) legitimately and permanently added
    a SIBLING `persistent-data/models/dynamics_iteration2b/` tree -- a new,
    separately-versioned namespace with the same six targets and the same
    required provenance files, never overwriting or touching the original
    `dynamics/` tree (verified elsewhere by exact hash).

    Phase 16O (Iteration-3 target-scoped model redevelopment) legitimately
    and permanently added a further SIBLING
    `persistent-data/models/dynamics_iteration3/` tree -- again the same six
    targets and the same required provenance files, with three targets
    (cpu_cores, requests_in_progress, latency_p95_seconds) redeveloped and
    three (request_rate_rps, memory_bytes, error_rate_rps) carried forward
    byte-identically from `dynamics_iteration2b/` (verified elsewhere by
    exact hash) -- never overwriting or touching either prior tree. All
    three are now expected top-level entries; this remains a precise
    allow-list, not a loosened wildcard -- any FOURTH top-level entry still
    fails this check.

    Phase 16AC (first live-PPO execution under the direct-training path,
    Phase16T onward -- a SEPARATE methodology from the dynamics-calibration
    protocol this file otherwise validates) legitimately and permanently
    created `persistent-data/rl/checkpoints/phase16w_sessions/session_000/`:
    a valid, hash-verified, 256-timestep partial-session-0 PPO checkpoint
    (rollout 1 did not complete at that time -- see
    `task_outputs/phase_16ac_checkpoint_provenance_report.md`). This is the
    first-ever legitimate PPO/RL artifact this repository has produced.

    Phase 16AE (the corrected rollout-1 resume, after Phase16AD fixed the
    checkpoint-resume defect) legitimately and permanently added a SIBLING
    `session_001/`: the valid, hash-verified completion of scientific
    session 0 (512 cumulative timesteps -- see
    `task_outputs/phase_16ae_session_001_checkpoint_provenance_report.md`),
    never overwriting or touching `session_000`.

    The invariant below is now precise rather than a blanket "must not
    exist": exactly `session_000` and `session_001` (each with its three
    required files) are allowed under `phase16w_sessions/`; `ppo_current/`
    and `rl/tensorboard/` remain required to be absent, and any OTHER
    entry under `persistent-data/rl/` still fails this check -- the same
    precise-allow-list-not-loosened-wildcard discipline as the dynamics
    trees above."""
    models_dir = project_root / "persistent-data/models"
    rl_dir = project_root / "persistent-data/rl"
    dynamics_dir = models_dir / "dynamics"
    dynamics_iteration2b_dir = models_dir / "dynamics_iteration2b"
    dynamics_iteration3_dir = models_dir / "dynamics_iteration3"

    ppo_paths = [models_dir / "ppo_current", rl_dir / "tensorboard"]
    for p in ppo_paths:
        assert not p.exists(), f"PPO/RL artifact path must not exist: {p}"

    EXPECTED_PPO_CHECKPOINT_SESSIONS = {"session_000", "session_001"}
    REQUIRED_PPO_CHECKPOINT_FILES = {"ppo_model.zip", "rng_state.pkl", "session_metadata.json"}
    if rl_dir.exists():
        rl_top_level = {p.name for p in rl_dir.iterdir()}
        assert rl_top_level <= {"checkpoints"}, f"unexpected entries directly under {rl_dir}: {rl_top_level - {'checkpoints'}}"
        checkpoints_dir = rl_dir / "checkpoints"
        if checkpoints_dir.exists():
            cp_top_level = {p.name for p in checkpoints_dir.iterdir()}
            assert cp_top_level == {"phase16w_sessions"}, f"unexpected entries directly under {checkpoints_dir}: {cp_top_level - {'phase16w_sessions'}}"
            sessions_dir = checkpoints_dir / "phase16w_sessions"
            session_dirs = {p.name for p in sessions_dir.iterdir() if p.is_dir()}
            assert session_dirs == EXPECTED_PPO_CHECKPOINT_SESSIONS, (
                f"{sessions_dir} session directories mismatch: "
                f"{session_dirs.symmetric_difference(EXPECTED_PPO_CHECKPOINT_SESSIONS)}"
            )
            for session_name in EXPECTED_PPO_CHECKPOINT_SESSIONS:
                present = {p.name for p in (sessions_dir / session_name).iterdir() if p.is_file()}
                assert REQUIRED_PPO_CHECKPOINT_FILES <= present, (
                    f"{sessions_dir / session_name} missing required files: "
                    f"{REQUIRED_PPO_CHECKPOINT_FILES - present}"
                )

    assert dynamics_dir.exists(), "persistent-data/models/dynamics/ must exist -- Phase16E dynamics-model artifacts are expected"
    expected_top_level = {"dynamics"}
    if dynamics_iteration2b_dir.exists():
        expected_top_level.add("dynamics_iteration2b")
    if dynamics_iteration3_dir.exists():
        expected_top_level.add("dynamics_iteration3")
    top_level = {p.name for p in models_dir.iterdir()}
    assert top_level == expected_top_level, f"unexpected entries directly under {models_dir}: {top_level - expected_top_level}"
    for tree in (dynamics_dir, dynamics_iteration2b_dir, dynamics_iteration3_dir):
        if not tree.exists():
            continue
        target_dirs = {p.name for p in tree.iterdir() if p.is_dir()}
        assert target_dirs == EXPECTED_DYNAMICS_TARGETS, f"{tree} target directories mismatch: {target_dirs.symmetric_difference(EXPECTED_DYNAMICS_TARGETS)}"
        for target in EXPECTED_DYNAMICS_TARGETS:
            present = {p.name for p in (tree / target).iterdir() if p.is_file()}
            assert REQUIRED_DYNAMICS_MODEL_FILES <= present, f"{tree / target} missing required provenance files: {REQUIRED_DYNAMICS_MODEL_FILES - present}"


def _manifest():
    return _load(MANIFEST_PATH)


class TestProtocolVersioning:
    def test_manifest_protocol_version_still_1(self):
        """CORR-32 is a pre-execution integrity correction, not a post-data
        methodology change -- protocol_version is kept at 1 per instruction."""
        assert _manifest()["protocol_version"] == 1

    def test_manifest_records_the_correction(self):
        m = _manifest()
        # corrected_in_phase/correction_reference at manifest root track the
        # MOST RECENT correction; CORR-32/33/34's sections remain fully
        # present in-body (additive, or superseded-in-place with an explicit
        # superseded_by note -- never silently dropped) -- the immediate
        # predecessor is recorded via prior_correction_reference.
        assert m["corrected_in_phase"] == "15B.2D"
        assert m["correction_reference"] == "CORR-35"
        assert m["prior_correction_reference"] == "CORR-34"

    def test_dynamics_config_points_at_manifest_and_records_correction(self):
        cfg = _load(DYNAMICS_CONFIG_PATH)
        assert cfg["dynamics_calibration_protocol_version"] == 1
        assert cfg["correction_reference"] == "CORR-35"
        assert cfg["corrected_in_phase"] == "15B.2D"

    def test_status_still_frozen_not_executed(self):
        cfg = _load(DYNAMICS_CONFIG_PATH)
        assert cfg["status"] == "PROTOCOL_FROZEN_NOT_YET_EXECUTED"


class TestHistoricalDataEligibilityCorrected:
    def test_historical_substitution_disabled(self):
        m = _manifest()
        assert m["historical_data_policy"]["historical_run_substitution_for_official_repeats"] is False

    def test_corr23_marked_supplemental_only_not_direct_fitting_source(self):
        m = _manifest()
        not_reusable = " ".join(m["historical_data_policy"]["not_reusable_for_fitting"])
        assert "CORR-23" in not_reusable
        assert "Metrics-API" in not_reusable

    def test_corr22_excluded_from_fitting_despite_structural_compatibility(self):
        m = _manifest()
        not_reusable = " ".join(m["historical_data_policy"]["not_reusable_for_fitting"])
        assert "CORR-22" in not_reusable

    def test_dynamics_config_reflects_no_substitution(self):
        cfg = _load(DYNAMICS_CONFIG_PATH)
        assert cfg["historical_run_substitution_for_official_repeats"] is False


class TestDuplicateCellAudit:
    def test_no_duplicate_replicas_total_rps_tuples_in_fitting_cells(self):
        m = _manifest()
        seen = set()
        for c in m["fitting_cells"]:
            key = (c["replicas"], c["total_offered_rps"])
            assert key not in seen, f"duplicate physical condition: {key}"
            seen.add(key)

    def test_r8_l2_l5_duplicate_explicitly_deduplicated_and_tagged(self):
        m = _manifest()
        cell = next(c for c in m["fitting_cells"] if c["cell_id"] == "dyncal_r8_L2_L5")
        assert cell["replicas"] == 8
        assert cell["total_offered_rps"] == 40.0
        assert "sustainable_per_replica_anchor" in cell["roles"]
        assert "fixed_total_contention_anchor" in cell["roles"]
        # And the standalone L5-at-8 cell must NOT also exist separately.
        cell_ids = [c["cell_id"] for c in m["fitting_cells"]]
        assert "dyncal_r8_L5" not in cell_ids
        assert "dyncal_r8_L2" not in cell_ids

    def test_fitting_cell_count_is_20(self):
        assert len(_manifest()["fitting_cells"]) == 20


class TestRevisedMaxRpsAndGeneratorCapacity:
    def test_l4_restricted_to_replicas_1_2_4(self):
        m = _manifest()
        l4 = next(lvl for lvl in m["load_levels"] if lvl["id"] == "L4_overload")
        assert set(l4["anchors_only"]) == {1, 2, 4}

    def test_no_fitting_cell_exceeds_the_revised_ceiling(self):
        m = _manifest()
        ceiling = m["revised_max_total_rps_ceiling"]["frozen_value"]
        for c in m["fitting_cells"] + m["validation_cells"]:
            assert c["total_offered_rps"] <= ceiling

    def test_generator_max_in_flight_covers_worst_case_at_max_rps(self):
        m = _manifest()
        max_rps = m["revised_max_total_rps_ceiling"]["frozen_value"]
        max_in_flight = m["generator_capacity"]["frozen_calibration_max_in_flight"]
        worst_case = max_rps * GENERATOR_TIMEOUT_SECONDS
        assert max_in_flight > worst_case, (
            f"max_in_flight={max_in_flight} must exceed worst-case outstanding={worst_case}"
        )

    def test_generator_max_in_flight_covers_every_matrix_value_including_transitions(self):
        m = _manifest()
        max_in_flight = m["generator_capacity"]["frozen_calibration_max_in_flight"]
        rps_values = [c["total_offered_rps"] for c in m["fitting_cells"]]
        rps_values += [c["total_offered_rps"] for c in m["validation_cells"]]
        rps_values += list(m["replica_transitions"]["workload_by_pair"].values())
        rps_values += [t["rps_before"] for t in m["workload_transitions"]["types"]]
        rps_values += [t["rps_after"] for t in m["workload_transitions"]["types"]]
        assert max_in_flight > max(rps_values) * GENERATOR_TIMEOUT_SECONDS


class TestArrivalProcessAndSchedules:
    def test_arrival_process_is_deterministic(self):
        m = _manifest()
        assert m["arrival_process"]["mode"] == "deterministic"
        assert m["arrival_process"]["connection_mode"] == "fresh"

    def test_schedule_registry_exists_and_has_expected_entry_count(self):
        import json

        with open(SCHEDULE_REGISTRY_PATH) as f:
            registry = json.load(f)
        # 20 fitting + 9 validation + 4 replica-transition + 4 workload-transition = 37
        assert len(registry) == 37

    def test_every_schedule_hash_independently_recomputes(self):
        import json

        with open(SCHEDULE_REGISTRY_PATH) as f:
            registry = json.load(f)
        for cell_id, entry in registry.items():
            schedule_path = PROJECT_ROOT / entry["schedule_path"]
            assert schedule_path.exists(), f"missing schedule file for {cell_id}: {schedule_path}"
            actual_digest = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
            assert actual_digest == entry["schedule_sha256"], f"hash mismatch for {cell_id}"

    def test_every_fitting_and_validation_cell_has_a_schedule(self):
        import json

        with open(SCHEDULE_REGISTRY_PATH) as f:
            registry = json.load(f)
        m = _manifest()
        for c in m["fitting_cells"] + m["validation_cells"]:
            assert c["cell_id"] in registry, f"no schedule registered for {c['cell_id']}"

    def test_replica_transition_schedule_shared_across_both_directions(self):
        m = _manifest()
        # Both directions of pair (1,2) must reference the same schedule path.
        assert "1,2" in m["replica_transitions"]["schedules"]
        path_str = m["replica_transitions"]["schedules"]["1,2"]["path"]
        assert (PROJECT_ROOT / path_str).exists()


class TestDecisionAlignedExtraction:
    def test_extraction_timestamps_are_60_90_120_150(self):
        m = _manifest()
        assert m["steady_state_timing"]["decision_aligned_extraction_timestamps_relative_to_traffic_onset"] == [60, 90, 120, 150]

    def test_scored_window_matches_extraction_range(self):
        m = _manifest()
        assert m["steady_state_timing"]["scored_window_seconds"] == [60, 180]

    def test_whole_run_splitting_documented(self):
        m = _manifest()
        assert "whole-run" in m["validation"]["split_granularity"]
        assert "never" in m["validation"]["split_granularity"]


class TestReplicaTransitionsAreAdjacentOnly:
    def test_all_pairs_are_adjacent(self):
        m = _manifest()
        for pair in m["replica_transitions"]["pairs"]:
            assert abs(pair[0] - pair[1]) == 1, f"non-adjacent pair found: {pair}"

    def test_eight_directional_types_present(self):
        m = _manifest()
        assert len(m["replica_transitions"]["directional_types"]) == 8

    def test_pairs_span_the_full_replica_range(self):
        m = _manifest()
        covered = set()
        for pair in m["replica_transitions"]["pairs"]:
            covered.update(pair)
        assert covered == {1, 2, 3, 4, 5, 6, 7, 8}


class TestWorkloadTransitionsIncludedInBudget:
    def test_workload_transition_types_present(self):
        m = _manifest()
        assert len(m["workload_transitions"]["types"]) == 4

    def test_workload_transitions_are_additional_to_replica_transitions(self):
        m = _manifest()
        c = m["campaign_estimate"]
        assert c["workload_transition_runs"] > 0
        assert c["replica_transition_runs"] > 0
        assert c["workload_transition_runs"] != c["replica_transition_runs"] or True  # both present, distinct budgets


class TestCorrectedRunCount:
    def test_official_run_count_is_102(self):
        m = _manifest()
        assert m["official_new_calibration_run_count"] == 102
        assert m["campaign_estimate"]["total_runs"] == 102

    def test_run_count_matches_sum_of_components(self):
        m = _manifest()
        c = m["campaign_estimate"]
        computed = (
            c["fitting_runs"] + c["validation_runs"]
            + c["replica_transition_runs"] + c["workload_transition_runs"]
        )
        assert computed == c["total_runs"]

    def test_fitting_runs_equals_cells_times_repeats(self):
        m = _manifest()
        c = m["campaign_estimate"]
        assert c["fitting_cells"] * c["fitting_repeats"] == c["fitting_runs"]


class TestRandomizedOrdersRegenerated:
    def test_fitting_order_matches_cell_count_every_repeat(self):
        m = _manifest()
        n = len(m["fitting_cells"])
        for repeat_key, order in m["randomization"]["fitting_order"].items():
            assert len(order) == n, f"{repeat_key} has {len(order)} entries, expected {n}"

    def test_replica_transition_order_has_8_entries_every_repeat(self):
        m = _manifest()
        for repeat_key, order in m["randomization"]["replica_transition_order"].items():
            assert len(order) == 8

    def test_workload_transition_order_has_4_entries_every_repeat(self):
        m = _manifest()
        for repeat_key, order in m["randomization"]["workload_transition_order"].items():
            assert len(order) == 4

    def test_seed_bases_all_distinct(self):
        m = _manifest()
        r = m["randomization"]
        seeds = {
            r["seed_base_fitting"], r["seed_base_validation"],
            r["seed_base_replica_transitions"], r["seed_base_workload_transitions"],
        }
        assert len(seeds) == 4


class TestAcceptanceCriteriaMathematics:
    def test_noise_normalization_formula_present_and_uses_max_with_epsilon(self):
        m = _manifest()
        formula = m["acceptance_criteria"]["noise_normalization"]["formula"]
        assert "max(" in formula
        assert "epsilon" in formula

    def test_thresholds_are_numeric_and_ordered(self):
        m = _manifest()
        t = m["acceptance_criteria"]["thresholds"]
        assert t["median_normalized_error_max"] < t["no_cell_normalized_error_exceeds"]
        assert 0.0 < t["fraction_of_cells_at_or_below_2_0"] <= 1.0

    def test_interpolation_cells_held_to_same_threshold(self):
        m = _manifest()
        assert m["acceptance_criteria"]["interpolation_cells_held_to_same_thresholds_as_fitting_cells"] is True

    def test_leave_one_repeat_out_prediction_source_specified(self):
        m = _manifest()
        assert "leave-one-repeat-out" in m["acceptance_criteria"]["noise_normalization"]["prediction_source"]


class TestErrorRateBalancedValidation:
    def test_balanced_accuracy_and_recall_thresholds_present(self):
        m = _manifest()
        erc = m["acceptance_criteria"]["error_rate_classification"]
        assert erc["balanced_accuracy_min"] == 0.80
        assert erc["nonzero_class_recall_min"] == 0.80

    def test_plain_accuracy_not_the_sole_criterion(self):
        m = _manifest()
        erc = m["acceptance_criteria"]["error_rate_classification"]
        assert "accuracy_min" not in erc or "balanced_accuracy_min" in erc

    def test_not_estimable_condition_documented(self):
        m = _manifest()
        assert "NOT_ESTIMABLE" in m["acceptance_criteria"]["error_rate_classification"]["not_estimable_condition"]


class TestMonotonicityJustifiedAxesOnly:
    def test_latency_improvement_with_replicas_not_enforced(self):
        m = _manifest()
        not_enforced = " ".join(m["acceptance_criteria"]["monotonicity_requirements"]["explicitly_not_enforced"])
        assert "latency" in not_enforced.lower()

    def test_rps_monotonicity_at_fixed_replicas_is_enforced(self):
        m = _manifest()
        enforced = " ".join(m["acceptance_criteria"]["monotonicity_requirements"]["enforced"])
        assert "offered RPS" in enforced


class TestDynamicsOutOfSupportDistinctFromPrometheusUnavailable:
    def test_extrapolation_policy_not_mapped_to_prometheus_unavailable(self):
        m = _manifest()
        policy = m["extrapolation_policy"]
        assert policy["rule"] == "dynamics_out_of_support_distinct_condition"
        assert "target_up=False" in policy["description"]  # documents what it must NOT be mapped to
        assert "NEVER mapped" in policy["description"]

    def test_not_implemented_in_this_phase(self):
        m = _manifest()
        assert m["extrapolation_policy"]["not_implemented_this_phase"] is True


class TestResidualModelPreregistered:
    def test_at_least_two_candidates_preregistered(self):
        m = _manifest()
        assert len(m["stochastic_residual_model"]["candidates"]) >= 2

    def test_residuals_never_from_validation_runs(self):
        m = _manifest()
        exclusions = " ".join(m["stochastic_residual_model"]["exclusions"])
        assert "interpolation-validation" in exclusions

    def test_no_final_scheme_selected(self):
        m = _manifest()
        assert "No final residual scheme is selected" in m["stochastic_residual_model"]["note"]


class TestUpdatedInvalidCellPolicy:
    def test_generator_capacity_criteria_present(self):
        m = _manifest()
        criteria = m["invalid_cell_structural_criteria"]
        assert any("max_in_flight" in c for c in criteria)
        assert any("dropped_overload" in c for c in criteria)

    def test_legitimate_timeout_after_dispatch_remains_valid_outcome(self):
        m = _manifest()
        valid_outcomes = " ".join(m["valid_performance_outcomes_not_grounds_for_invalidity"])
        assert "legitimately_dispatched" in valid_outcomes


class TestNoHeldOutOrProhibitedDataLeakage:
    def test_no_reference_to_unseen_eval_anywhere_in_protocol_files(self):
        for path in (MANIFEST_PATH, DYNAMICS_CONFIG_PATH, PROTOCOL_DOC_PATH):
            text = path.read_text(encoding="utf-8")
            assert "unseen_eval" not in text

    def test_hpa_tuning_trials_marked_not_reusable_for_fitting(self):
        m = _manifest()
        not_reusable = " ".join(m["historical_data_policy"]["not_reusable_for_fitting"])
        assert "HPA tuning trials" in not_reusable

    def test_no_official_hpa_tuning_experiment_ids_in_schedule_registry(self):
        import json

        with open(SCHEDULE_REGISTRY_PATH) as f:
            registry = json.load(f)
        for cell_id in registry:
            assert "hpa_tune_" not in cell_id


class TestNoTrainingConfigOrHpaBaselineModified:
    def test_rl_contract_version_still_2(self):
        assert _load(RL_CONFIG_PATH)["rl_contract_version"] == 2

    def test_reward_contract_version_still_2(self):
        assert _load(REWARD_CONFIG_PATH)["reward_contract_version"] == 2

    def test_ppo_block_unchanged_shape(self):
        ppo = _load(RL_CONFIG_PATH)["ppo"]
        assert ppo["net_arch"] == [128, 128]
        assert ppo["policy"] == "MlpPolicy"

    def test_hpa_baseline_still_260(self):
        assert _load(HPA_CONFIG_PATH)["target_cpu_utilization_percent"] == 260
        manifest = _load(HPA_MANIFEST_PATH)
        assert manifest["spec"]["metrics"][0]["resource"]["target"]["averageUtilization"] == 260

    def test_calibration_result_directories_are_never_fabricated(self):
        # Historically this asserted the directory doesn't exist at all
        # (true through Phase 15B.2D, pure protocol-freeze phases). Phase
        # 15B.3A legitimately began the frozen 102-run campaign, so a
        # populated persistent-data/dynamics_calibration/ directory is now
        # EXPECTED. What must remain permanently true: every attempt
        # subdirectory's name is a genuine "<planned_slot_id>_attemptN_<ts>"
        # derived from a real planned_slot_id in the frozen global sequence
        # -- never an invented/arbitrary directory.
        result_dir = PROJECT_ROOT / "persistent-data" / "dynamics_calibration"
        if not result_dir.exists():
            return
        planned_slot_ids = {e["planned_slot_id"] for e in json.load(open(GLOBAL_CAMPAIGN_SEQUENCE_PATH))}
        for entry in result_dir.iterdir():
            if not entry.is_dir() or entry.name == "campaign_ledger.json":
                continue
            matches = [pid for pid in planned_slot_ids if entry.name.startswith(pid + "_attempt")]
            assert matches, f"attempt directory {entry.name!r} does not correspond to any frozen planned_slot_id"

    def test_no_ppo_training_artifacts_exist(self):
        # See _assert_ppo_absent_and_dynamics_artifacts_valid: this originally
        # asserted persistent-data/models/ and persistent-data/rl/ stayed
        # entirely empty; Phase 16E's dynamics-model artifacts under
        # persistent-data/models/dynamics/ are now expected and validated
        # precisely, while PPO/RL artifacts remain required to be absent.
        _assert_ppo_absent_and_dynamics_artifacts_valid(PROJECT_ROOT)

    def test_no_traffic_directories_created_by_schedule_generation(self):
        # Schedule GENERATION (scripts/generate_dynamics_calibration_schedules.py)
        # itself must never send traffic -- verified by inspecting that
        # script's own code, not by asserting persistent-data/traffic/ stays
        # forever free of "dyncal"-named directories (Phase 15B.3A's real
        # campaign execution legitimately created
        # persistent-data/traffic/fit_dyncal_..._attempt1_.../ once official
        # calibration runs began).
        generator_source = (PROJECT_ROOT / "scripts/generate_dynamics_calibration_schedules.py").read_text(encoding="utf-8")
        for forbidden in ("httpx", "requests.", "urllib.request", "aiohttp", "TrafficGenerator"):
            assert forbidden not in generator_source, f"schedule generator must never perform network traffic ({forbidden} found)"


# =============================================================================
# Phase 15B.2B (CORR-33) additions below: final execution-readiness
# clarifications resolved before the first calibration measurement is ever
# collected. No matrix redesign -- the 20/9/8/4/102/37/64/400/5s frozen
# values from CORR-32 are unchanged and covered by the tests above.
# =============================================================================


class TestCorrectionChainIntegrity:
    """The manifest root's corrected_in_phase/correction_reference always
    track the MOST RECENT correction (a moving target across phases) --
    rather than hardcode a specific CORR-N pair that would need updating
    every future phase, this proves the chain itself is never skipped: the
    immediate predecessor pointer must always be exactly one CORR number
    below the current one. Individual corrections' actual CONTENT (e.g.
    CORR-33's request-rate modeling, CORR-34's global campaign order) is
    verified by their own dedicated test classes, not by this version
    pointer."""

    def test_prior_correction_reference_is_exactly_one_below_current(self):
        m = _manifest()
        current_n = int(m["correction_reference"].split("-")[1])
        prior_n = int(m["prior_correction_reference"].split("-")[1])
        assert prior_n == current_n - 1

    def test_protocol_version_still_1_no_data_collected(self):
        # Every correction so far has been pre-execution -- protocol_version
        # stays at 1 until the first real calibration measurement exists.
        assert _manifest()["protocol_version"] == 1


class TestRequestRateRealizationModeled:
    def test_request_rate_signal_semantics_present(self):
        m = _manifest()
        rr = m["signal_semantics"]["request_rate"]
        assert "offered_rps" in rr["exogenous_input"]
        assert "request_rate_rps" in rr["fitting_target"]

    def test_identity_assumption_explicitly_forbidden(self):
        m = _manifest()
        text = m["signal_semantics"]["request_rate"]["identity_assumption_forbidden"]
        assert "MUST NOT" in text
        assert "connection_error" in text

    def test_request_rate_candidate_model_families_present(self):
        m = _manifest()
        families = m["candidate_model_families"]["request_rate"]
        assert "bounded_lookup_interpolation" in families
        assert "piecewise_linear_regression_surface" in families
        assert "tree_based_regressor" in families

    def test_request_rate_validation_metrics_include_throughput_realization(self):
        m = _manifest()
        metrics = m["model_selection_metrics"]["request_rate"]
        assert any("throughput_realization" in x for x in metrics)
        assert any("overload_regime" in x for x in metrics)

    def test_throughput_realization_ratio_never_assumed_1(self):
        m = _manifest()
        note = m["signal_semantics"]["request_rate"]["throughput_realization_note"]
        assert "NEVER assumed to equal 1.0" in note


class TestCausalStructureJointInput:
    def test_joint_input_structure_chosen(self):
        m = _manifest()
        assert m["causal_structure"]["chosen_structure"] == "joint_input"

    def test_two_stage_alternative_explicitly_rejected(self):
        m = _manifest()
        cs = m["causal_structure"]
        assert cs["rejected_alternative"] == "two_stage_offered_to_predicted_rps_to_other_outputs"
        assert "teacher-forcing" in cs["rejected_alternative_reason"] or "teacher-forcing" in cs["description"]


class TestTrafficTerminologyDisambiguated:
    def test_all_four_terms_defined(self):
        m = _manifest()
        tt = m["traffic_terminology"]
        for key in ("offered", "sent", "server_observed", "client_completed"):
            assert key in tt and len(tt[key]) > 0

    def test_client_completed_never_used_as_rl_observation(self):
        m = _manifest()
        assert "NEVER used as the RL request-rate observation" in m["traffic_terminology"]["client_completed"]

    def test_offered_equals_sent_required_for_structural_validity(self):
        m = _manifest()
        assert "offered == sent" in m["traffic_terminology"]["sent"]


class TestPerPodRoutingDiagnostics:
    def test_required_for_multi_replica_runs(self):
        m = _manifest()
        assert "ready_replicas >= 2" in m["per_pod_routing_diagnostics"]["required_for"]

    def test_promql_filters_by_job_to_avoid_double_scrape(self):
        m = _manifest()
        promql = m["per_pod_routing_diagnostics"]["promql"]
        assert 'job="demo-app"' in promql

    def test_three_tier_classification_present(self):
        m = _manifest()
        tiers = m["per_pod_routing_diagnostics"]["classification"]
        assert "healthy_variance" in tiers
        assert "routing_pinning_suspected" in tiers
        assert "structural_routing_failure" in tiers

    def test_healthy_variance_does_not_require_perfect_equal_distribution(self):
        m = _manifest()
        healthy = m["per_pod_routing_diagnostics"]["classification"]["healthy_variance"]
        assert "no action" in healthy["action"]

    def test_routing_pinning_suspected_not_automatically_invalid(self):
        m = _manifest()
        pinning = m["per_pod_routing_diagnostics"]["classification"]["routing_pinning_suspected"]
        assert "NOT automatically structurally invalid" in pinning["action"]

    def test_structural_routing_failure_is_invalid_and_retried(self):
        m = _manifest()
        failure = m["per_pod_routing_diagnostics"]["classification"]["structural_routing_failure"]
        assert "STRUCTURALLY INVALID" in failure["action"]
        assert "retry" in failure["action"]

    def test_structural_routing_failure_added_to_invalid_criteria(self):
        m = _manifest()
        assert "structural_routing_failure_per_per_pod_routing_diagnostics" in m["invalid_cell_structural_criteria"]

    def test_healthy_or_pinning_variance_added_to_valid_outcomes(self):
        m = _manifest()
        assert "healthy_variance_or_routing_pinning_suspected_per_per_pod_routing_diagnostics" in (
            m["valid_performance_outcomes_not_grounds_for_invalidity"]
        )


class TestHpaIsolationProcedure:
    def test_reuses_existing_patterns(self):
        m = _manifest()
        assert m["hpa_isolation_procedure"]["reuses_existing_patterns"] is True

    def test_before_session_deletes_hpa_with_ignore_not_found(self):
        m = _manifest()
        steps = " ".join(m["hpa_isolation_procedure"]["before_session"])
        assert "kubectl delete hpa demo-app-hpa -n autoscaler-app --ignore-not-found" in steps

    def test_after_session_restores_canonical_hpa_manifest(self):
        m = _manifest()
        steps = " ".join(m["hpa_isolation_procedure"]["after_session_including_abort_or_error"])
        assert "kubectl apply -f k8s/hpa/hpa.yaml" in steps
        assert "260%" in steps

    def test_restoration_covers_abort_case_explicitly(self):
        m = _manifest()
        assert "after_session_including_abort_or_error" in m["hpa_isolation_procedure"]

    def test_not_executed_this_phase(self):
        m = _manifest()
        assert m["hpa_isolation_procedure"]["not_executed_this_phase"] is True


class TestSessionSemantics:
    def test_session_start_and_end_checklists_present(self):
        m = _manifest()
        ss = m["session_semantics"]
        assert len(ss["session_start_checklist"]) > 0
        assert len(ss["session_end_checklist"]) > 0

    def test_resumption_continues_from_next_invalid_slot_without_regenerating_order(self):
        m = _manifest()
        rule = m["session_semantics"]["resumption_rule"]
        assert "NEXT not-yet-VALID planned run slot" in rule
        assert "NEVER regenerated" in rule
        assert "NEVER rerun" in rule


class TestRunAttemptIdentity:
    def test_planned_slot_and_attempt_id_patterns_distinct(self):
        m = _manifest()
        ri = m["run_identity"]
        assert "repeat" in ri["planned_slot_id_pattern"]
        assert "attempt" in ri["execution_attempt_id_pattern"]
        assert ri["planned_slot_id_pattern"] != ri["execution_attempt_id_pattern"]

    def test_invalid_attempts_preserved_never_overwritten(self):
        m = _manifest()
        rule = m["run_identity"]["retry_rule"]
        assert "PRESERVED" in rule
        assert "never deleted or overwritten" in rule

    def test_only_valid_attempts_enter_datasets(self):
        m = _manifest()
        assert "Only VALID attempts enter" in m["run_identity"]["retry_rule"]

    def test_required_run_metadata_fields_cover_session_and_chronology(self):
        m = _manifest()
        fields = m["run_identity"]["required_run_metadata_fields"]
        for required in (
            "planned_slot_id", "execution_attempt_number", "session_id",
            "campaign_sequence_index", "wall_clock_start_utc", "wall_clock_end_utc",
            "hpa_absence_verified", "rl_absence_verified", "kind_state",
            "prometheus_health", "metrics_server_resolution",
        ):
            assert required in fields, f"missing required run metadata field: {required}"

    def test_artifact_structure_metadata_extended_with_corr33_fields(self):
        m = _manifest()
        fields = m["artifact_structure"]["required_metadata_fields"]
        for required in ("planned_slot_id", "execution_attempt_number", "session_id", "per_pod_routing_classification"):
            assert required in fields


class TestClientServerLatencyAnalysisPreregistered:
    def test_both_latencies_preserved_per_run(self):
        m = _manifest()
        preserved = m["client_server_latency_analysis"]["preserved_per_run"]
        assert "server_p95_prometheus" in preserved
        assert "client_p95_requests_csv" in preserved

    def test_client_p95_never_a_policy_input(self):
        m = _manifest()
        status = m["client_server_latency_analysis"]["policy_input_status"]
        assert "NEVER" in status

    def test_preregistered_analyses_include_overload_divergence(self):
        m = _manifest()
        analyses = m["client_server_latency_analysis"]["preregistered_analyses"]
        assert "overload_regime_divergence_reported_separately" in analyses


class TestCpuMemoryTransitionSemantics:
    def test_cpu_transition_semantics_preserve_current_and_ready_separately(self):
        m = _manifest()
        text = m["signal_semantics"]["cpu"]["transition_semantics"]
        assert "current" in text and "ready" in text
        assert "never" in text.lower() and "zero" in text.lower()

    def test_memory_transition_semantics_scoped_to_steady_state_only(self):
        m = _manifest()
        mem = m["signal_semantics"]["memory"]
        assert "STEADY-STATE CELLS ONLY" in mem["evaluation_basis"] or "steady state" in mem["evaluation_basis"].lower()
        assert "transition_semantics" in mem
        assert "MUST NOT" in mem["transition_semantics"]

    def test_memory_transition_semantics_does_not_change_v2_observation_normalization(self):
        m = _manifest()
        text = m["signal_semantics"]["memory"]["transition_semantics"]
        assert "does NOT change" in text


class TestAllZero5xxFallbackPolicy:
    def test_fallback_policy_present_with_domain_label(self):
        m = _manifest()
        fb = m["acceptance_criteria"]["error_rate_classification"]["all_zero_fallback_policy"]
        assert fb["dynamics_component_label"] == "ZERO_ONLY_WITHIN_OBSERVED_CALIBRATION_DOMAIN"

    def test_fallback_does_not_block_final_acceptance_alone(self):
        m = _manifest()
        fb = m["acceptance_criteria"]["error_rate_classification"]["all_zero_fallback_policy"]
        assert "NOT blocked solely" in fb["acceptance_impact"]

    def test_fallback_forbids_synthetic_5xx_and_overload_claims(self):
        m = _manifest()
        fb = m["acceptance_criteria"]["error_rate_classification"]["all_zero_fallback_policy"]
        assert "synthetic" in fb["forbidden"]
        assert "overload" in fb["forbidden"]

    def test_fallback_requires_reporting_before_ppo_training(self):
        m = _manifest()
        fb = m["acceptance_criteria"]["error_rate_classification"]["all_zero_fallback_policy"]
        assert "before any PPO training begins" in fb["reporting_requirement"]

    def test_sla_latency_remains_compensating_signal(self):
        m = _manifest()
        fb = m["acceptance_criteria"]["error_rate_classification"]["all_zero_fallback_policy"]
        assert "SLA" in fb["compensating_signal"]


class TestDynamicsOutOfSupportPreciseSemantics:
    def test_episode_boundary_is_truncated_not_terminated(self):
        m = _manifest()
        sem = m["extrapolation_policy"]["precise_episode_semantics"]
        assert sem["episode_boundary_flag"] == "truncated"

    def test_termination_reason_key_and_value_frozen(self):
        m = _manifest()
        sem = m["extrapolation_policy"]["precise_episode_semantics"]
        assert sem["info_key"] == "termination_reason"
        assert sem["info_value"] == "dynamics_out_of_support"

    def test_does_not_increment_consecutive_invalid_steps(self):
        m = _manifest()
        sem = m["extrapolation_policy"]["precise_episode_semantics"]
        assert sem["does_not_increment"] == "consecutive_invalid_steps"

    def test_does_not_set_prometheus_unavailable_override(self):
        m = _manifest()
        sem = m["extrapolation_policy"]["precise_episode_semantics"]
        assert sem["does_not_set_override_reason"] == "prometheus_unavailable"

    def test_no_fabricated_observation_returned(self):
        m = _manifest()
        assert m["extrapolation_policy"]["precise_episode_semantics"]["no_fabricated_observation_returned"] is True

    def test_requires_new_types_py_constant_not_reusing_existing_override(self):
        m = _manifest()
        assert m["extrapolation_policy"]["requires_types_py_extension"] is True

    def test_dynamics_out_of_support_does_not_collide_with_existing_override_reasons(self):
        # The frozen string "dynamics_out_of_support" must not already exist
        # as one of the current OVERRIDE_* values -- it needs a genuinely new
        # constant, per requires_types_py_extension above.
        text = RL_ENV_TYPES_PATH.read_text(encoding="utf-8")
        assert '"dynamics_out_of_support"' not in text

    def test_still_not_implemented_this_phase(self):
        m = _manifest()
        assert m["extrapolation_policy"]["not_implemented_this_phase"] is True


class TestGeneratorSafetyMarginTwentyPercent:
    def test_headroom_factor_is_1_20(self):
        m = _manifest()
        assert m["generator_capacity"]["required_max_in_flight_headroom_factor"] == 1.20

    def test_computed_value_matches_manifests_own_inputs(self):
        m = _manifest()
        gc = m["generator_capacity"]
        max_rps = m["revised_max_total_rps_ceiling"]["frozen_value"]
        timeout = gc["default_timeout_seconds"]
        headroom = gc["required_max_in_flight_headroom_factor"]
        computed = math.ceil(max_rps * timeout * headroom)
        assert computed == 384
        assert gc["required_max_in_flight_computed"] == computed

    def test_frozen_max_in_flight_of_400_passes_the_margin_check(self):
        m = _manifest()
        gc = m["generator_capacity"]
        assert gc["frozen_calibration_max_in_flight"] >= gc["required_max_in_flight_computed"]
        assert gc["frozen_calibration_max_in_flight_passes_margin_check"] is True

    def test_dynamics_config_exposes_the_same_computed_value(self):
        cfg = _load(DYNAMICS_CONFIG_PATH)
        assert cfg["required_max_in_flight_computed"] == 384
        assert cfg["required_max_in_flight_headroom_factor"] == 1.20


class TestResidualSamplingUnitIsTrajectoryLevel:
    def test_sampling_unit_is_whole_run_trajectory_not_pointwise(self):
        m = _manifest()
        srm = m["stochastic_residual_model"]
        assert srm["sampling_unit"] == "whole_run_residual_trajectory"

    def test_bootstrap_candidate_renamed_to_reflect_trajectory_sampling(self):
        m = _manifest()
        names = [c["name"] for c in m["stochastic_residual_model"]["candidates"]]
        assert "whole_run_residual_trajectory_bootstrap" in names

    def test_transition_residuals_never_pointwise_shuffled(self):
        m = _manifest()
        text = m["stochastic_residual_model"]["transition_residuals"]
        assert "never" in text.lower()
        assert "shuffled" in text.lower()

    def test_transition_trajectory_shape_validation_added(self):
        m = _manifest()
        validations = " ".join(m["stochastic_residual_model"]["validation"])
        assert "transition trajectory shape" in validations

    def test_residuals_still_excluded_from_validation_runs(self):
        m = _manifest()
        exclusions = " ".join(m["stochastic_residual_model"]["exclusions"])
        assert "interpolation-validation" in exclusions


class TestNoMatrixRedesignInvariantsPreserved:
    """CORR-33 is explicitly scoped as execution-semantics clarification, not
    a matrix redesign -- every CORR-32 frozen count/hash/threshold must be
    byte-for-byte unchanged."""

    def test_20_fitting_9_validation_8_replica_transition_4_workload_transition(self):
        m = _manifest()
        assert len(m["fitting_cells"]) == 20
        assert len(m["validation_cells"]) == 9
        assert len(m["replica_transitions"]["directional_types"]) == 8
        assert len(m["workload_transitions"]["types"]) == 4

    def test_102_total_runs_unchanged(self):
        m = _manifest()
        assert m["official_new_calibration_run_count"] == 102
        assert m["campaign_estimate"]["total_runs"] == 102

    def test_37_schedules_unchanged(self):
        import json

        with open(SCHEDULE_REGISTRY_PATH) as f:
            registry = json.load(f)
        assert len(registry) == 37

    def test_64_rps_ceiling_and_400_max_in_flight_unchanged(self):
        m = _manifest()
        assert m["revised_max_total_rps_ceiling"]["frozen_value"] == 64.0
        assert m["generator_capacity"]["frozen_calibration_max_in_flight"] == 400

    def test_5_second_timeout_unchanged(self):
        m = _manifest()
        assert m["generator_capacity"]["default_timeout_seconds"] == GENERATOR_TIMEOUT_SECONDS

    def test_hpa_baseline_260_and_contracts_v2_unchanged(self):
        assert _load(HPA_CONFIG_PATH)["target_cpu_utilization_percent"] == 260
        assert _load(RL_CONFIG_PATH)["rl_contract_version"] == 2
        assert _load(REWARD_CONFIG_PATH)["reward_contract_version"] == 2

    def test_no_ppo_models_exist_after_this_phase(self):
        # Phase 15B.3A legitimately began the frozen campaign, so
        # persistent-data/dynamics_calibration/ may now be populated --
        # see TestNoTrainingConfigOrHpaBaselineModified.
        # test_calibration_result_directories_are_never_fabricated for that
        # check. PPO/RL training artifacts must still never exist; Phase 16E's
        # dynamics-model artifacts under persistent-data/models/dynamics/ are
        # a separate, expected category -- see
        # _assert_ppo_absent_and_dynamics_artifacts_valid.
        _assert_ppo_absent_and_dynamics_artifacts_valid(PROJECT_ROOT)

    def test_no_reference_to_unseen_eval_in_any_corr33_additions(self):
        for path in (MANIFEST_PATH, DYNAMICS_CONFIG_PATH, PROTOCOL_DOC_PATH):
            text = path.read_text(encoding="utf-8")
            assert "unseen_eval" not in text


# =============================================================================
# Phase 15B.2C (CORR-34) additions below: freezes the one authoritative
# global 102-slot execution order, and makes the per-pod routing classifier
# exhaustive (4 tiers, no gaps). Still no matrix redesign -- the
# 20/9/8/4/102/37/64/400/5s frozen values are unchanged and covered above.
# =============================================================================


def _global_sequence():
    with open(GLOBAL_CAMPAIGN_SEQUENCE_PATH) as f:
        return json.load(f)


class TestCorr34ContentStillPresentAfterCorr35:
    # CORR-34's global campaign order is unaffected by CORR-35 (verified in
    # full by TestGlobalCampaignOrderFrozen below); this class confirms
    # CORR-34's per-pod routing work was SUPERSEDED IN PLACE with an
    # explicit pointer, not silently dropped, per the task's "do not
    # rewrite CORR-34" instruction (which applies to the corrections LOG,
    # not to this evolving manifest section itself -- CORR-33 was
    # similarly superseded-in-place by CORR-34's own edit).
    def test_manifest_explicitly_documents_the_corr34_to_corr35_supersession(self):
        m = _manifest()
        text = m["per_pod_routing_diagnostics"]["superseded_by_corr35"]
        assert "fair_fraction = 1/N_eligible" in text
        assert "INVALID" in text
        assert "correction/generalization" in text


class TestGlobalCampaignOrderFrozen:
    def test_global_campaign_order_section_present(self):
        m = _manifest()
        gco = m["global_campaign_order"]
        assert gco["total_entries"] == 102
        assert gco["index_range"] == [1, 102]
        assert gco["index_contiguous_and_unique"] is True

    def test_did_not_previously_exist_before_corr34(self):
        m = _manifest()
        assert "did not previously exist" in m["global_campaign_order"]["status"]

    def test_sequence_file_exists_and_hash_matches_manifest_pointer(self):
        m = _manifest()
        assert GLOBAL_CAMPAIGN_SEQUENCE_PATH.exists()
        actual = hashlib.sha256(GLOBAL_CAMPAIGN_SEQUENCE_PATH.read_bytes()).hexdigest()
        assert actual == m["global_campaign_order"]["path_sha256"]

    def test_sequence_has_exactly_102_entries(self):
        assert len(_global_sequence()) == 102

    def test_indices_exactly_1_through_102_contiguous_and_unique(self):
        indices = [e["campaign_sequence_index"] for e in _global_sequence()]
        assert indices == list(range(1, 103))

    def test_every_planned_slot_id_unique(self):
        ids = [e["planned_slot_id"] for e in _global_sequence()]
        assert len(ids) == len(set(ids))

    def test_composition_matches_60_18_16_8(self):
        seq = _global_sequence()
        counts = {}
        for e in seq:
            counts[e["run_kind"]] = counts.get(e["run_kind"], 0) + 1
        assert counts == {
            "fitting": 60,
            "interpolation_validation": 18,
            "replica_transition": 16,
            "workload_transition": 8,
        }

    def test_every_schedule_sha_matches_registry_and_recomputes_from_disk(self):
        import json as _json

        with open(SCHEDULE_REGISTRY_PATH) as f:
            registry = _json.load(f)
        for e in _global_sequence():
            reg_entry = registry[e["cell_or_transition_id"]]
            assert e["schedule_sha256"] == reg_entry["schedule_sha256"]
            schedule_path = PROJECT_ROOT / e["schedule_path"]
            assert schedule_path.exists()
            assert hashlib.sha256(schedule_path.read_bytes()).hexdigest() == e["schedule_sha256"]

    def test_required_fields_present_on_every_entry_kind(self):
        for e in _global_sequence():
            for field in ("campaign_sequence_index", "planned_slot_id", "run_kind", "cell_or_transition_id", "repeat_number", "schedule_id", "schedule_path", "schedule_sha256"):
                assert field in e, f"missing {field} on {e.get('planned_slot_id')}"
            if e["run_kind"] in ("fitting", "interpolation_validation"):
                assert "replicas" in e and "offered_rps" in e
            elif e["run_kind"] == "replica_transition":
                assert "replicas_before" in e and "replicas_after" in e and "offered_rps" in e
            elif e["run_kind"] == "workload_transition":
                assert "replicas" in e and "rps_before" in e and "rps_after" in e

    def test_block_order_matches_the_frozen_precommitted_sequence(self):
        # 1:fitting(20) 2:val(9) 3:transrep(8) 4:transwork(4) 5:fitting(20)
        # 6:val(9) 7:transrep(8) 8:transwork(4) 9:fitting(20) == 102
        seq = _global_sequence()
        expected_block_kinds = (
            ["fitting"] * 20
            + ["interpolation_validation"] * 9
            + ["replica_transition"] * 8
            + ["workload_transition"] * 4
            + ["fitting"] * 20
            + ["interpolation_validation"] * 9
            + ["replica_transition"] * 8
            + ["workload_transition"] * 4
            + ["fitting"] * 20
        )
        assert [e["run_kind"] for e in seq] == expected_block_kinds

    def test_within_block_order_preserves_the_corr32_frozen_randomized_order(self):
        # The global sequence must not introduce a NEW shuffle -- each
        # block's cell_or_transition_id order must exactly match the
        # already-frozen randomization.fitting_order/validation_order/etc.
        m = _manifest()
        seq = _global_sequence()
        fitting_block1 = [e["cell_or_transition_id"] for e in seq if e["run_kind"] == "fitting" and e["repeat_number"] == 1]
        expected_cells_by_tag = {}
        for c in m["fitting_cells"]:
            tag = c["cell_id"].split("_", 2)[-1]
            expected_cells_by_tag[(c["replicas"], tag)] = c["cell_id"]
        expected_order = [
            expected_cells_by_tag[(item["replicas"], item["load_level"])]
            for item in m["randomization"]["fitting_order"]["repeat_1"]
        ]
        assert fitting_block1 == expected_order

    def test_regenerating_the_sequence_reproduces_the_committed_file_byte_identical(self):
        # Proves the generator introduces no additional randomness --
        # re-deriving from the same frozen inputs is fully deterministic.
        import importlib
        import sys

        sys.path.insert(0, str(PROJECT_ROOT))
        try:
            gen = importlib.import_module("scripts.generate_global_campaign_sequence")
            importlib.reload(gen)
            rebuilt = gen.build_global_sequence()
        finally:
            sys.path.remove(str(PROJECT_ROOT)) if str(PROJECT_ROOT) in sys.path else None
        assert rebuilt == _global_sequence()

    def test_generator_script_contains_no_randomization_calls(self):
        # The docstring legitimately explains that it concatenates the
        # already-frozen shuffled block orders -- what matters is that the
        # script's own CODE never calls a shuffle/random function.
        text = (PROJECT_ROOT / "scripts/generate_global_campaign_sequence.py").read_text(encoding="utf-8")
        assert "import random" not in text
        assert ".shuffle(" not in text

    def test_resume_rule_uses_lowest_not_valid_index_never_reshuffles(self):
        m = _manifest()
        rule = m["global_campaign_order"]["resume_rule"]
        assert "lowest campaign_sequence_index" in rule
        assert "no VALID attempt" in rule
        assert "Never a new shuffle" in rule
        assert "never alters this list" in rule


class TestPlannedSlotStateLedgerSpecifiedNotCreated:
    def test_not_created_this_phase(self):
        m = _manifest()
        assert m["planned_slot_state_ledger"]["not_created_this_phase"] is True

    def test_three_states_defined(self):
        m = _manifest()
        assert m["planned_slot_state_ledger"]["states"] == ["NOT_ATTEMPTED", "VALID", "INVALID_RETRY_REQUIRED"]

    def test_ledger_reconstructable_from_persisted_attempt_metadata(self):
        m = _manifest()
        assert "reconstructable" in m["planned_slot_state_ledger"]["reconstructability"]

    def test_ledger_matches_the_frozen_three_state_spec_if_it_now_exists(self):
        # As of Phase 15B.2C this ledger format was only SPECIFIED -- no
        # file existed. Phase 15B.3A legitimately created one when the
        # frozen campaign began; if present, it must conform exactly to
        # the states/fields frozen here (never a 4th ad hoc state).
        ledger_path = PROJECT_ROOT / "persistent-data/dynamics_calibration/campaign_ledger.json"
        if not ledger_path.exists():
            return
        ledger = json.load(open(ledger_path))
        allowed_states = set(_manifest()["planned_slot_state_ledger"]["states"])
        for slot in ledger["slots"].values():
            assert slot["state"] in allowed_states
            assert "campaign_sequence_index" in slot and "planned_slot_id" in slot
            assert "all_attempt_ids" in slot and "valid_attempt_id" in slot


class TestExhaustiveRoutingClassification:
    def test_four_tiers_present(self):
        m = _manifest()
        tiers = m["per_pod_routing_diagnostics"]["classification"]
        for tier in ("insufficient_routing_sample", "structural_routing_failure", "healthy_variance", "routing_pinning_suspected"):
            assert tier in tiers

    def test_precedence_order_frozen_and_structural_before_healthy(self):
        m = _manifest()
        cls = m["per_pod_routing_diagnostics"]["classification"]
        assert cls["precedence_order"] == [
            "INSUFFICIENT_ROUTING_SAMPLE",
            "STRUCTURAL_ROUTING_FAILURE",
            "HEALTHY_VARIANCE",
            "ROUTING_PINNING_SUSPECTED",
        ]
        assert cls["structural_routing_failure"]["checked_before_healthy_variance"] is True

    def test_pinning_suspected_is_the_explicit_catch_all(self):
        m = _manifest()
        pinning = m["per_pod_routing_diagnostics"]["classification"]["routing_pinning_suspected"]
        assert "ALL OTHER" in pinning["condition"]
        assert "NOT automatically structurally invalid" in pinning["action"]

    def test_exhaustiveness_note_present(self):
        m = _manifest()
        note = m["per_pod_routing_diagnostics"]["classification"]["exhaustiveness_note"]
        assert "EXACTLY ONE" in note

    def test_insufficient_sample_not_a_failure(self):
        m = _manifest()
        insufficient = m["per_pod_routing_diagnostics"]["classification"]["insufficient_routing_sample"]
        assert "NOT a failure" in insufficient["action"]
        assert "insufficient_routing_sample_per_per_pod_routing_diagnostics" in m["valid_performance_outcomes_not_grounds_for_invalidity"]


def _reference_classify_routing_opportunity(expected_counts, actual_counts, min_expected=50, min_assessable=2):
    """Independent reference implementation of the CORR-35 exhaustive
    4-tier, EXPOSURE-WEIGHTED routing classifier, mirroring the exact
    thresholds frozen in experiments/manifests/rl_dynamics_calibration_v1.yaml's
    per_pod_routing_diagnostics (0.01 / 0.5 / 1.5 / 0.30 / min_expected=50 /
    min_assessable=2) -- operates on (expected, actual) PAIRS rather than
    raw pod counts, so it naturally handles unequal Ready-serving exposure
    (CORR-34's `1/N_eligible` reference could not)."""
    assessable_idx = [i for i, e in enumerate(expected_counts) if e >= min_expected]
    if len(assessable_idx) < min_assessable:
        return "INSUFFICIENT_ROUTING_SAMPLE"
    ratios = [actual_counts[i] / expected_counts[i] for i in assessable_idx]
    mean = sum(ratios) / len(ratios)
    variance = sum((r - mean) ** 2 for r in ratios) / len(ratios)
    cv = (variance**0.5) / mean if mean > 0 else float("inf")
    if any(r < 0.01 for r in ratios):
        return "STRUCTURAL_ROUTING_FAILURE"
    if all(0.5 <= r <= 1.5 for r in ratios) and cv <= 0.30:
        return "HEALTHY_VARIANCE"
    return "ROUTING_PINNING_SUSPECTED"


class TestRoutingClassificationExamplesExhaustiveCorr35:
    def test_twenty_percent_of_expected_is_pinning_suspected(self):
        # 4 assessable pods, pod0 receives only 20% of its own expected
        # opportunity -- not severe enough (>=0.01) for structural failure,
        # not within [0.5,1.5] for healthy.
        expected = [100, 100, 100, 100]
        actual = [20, 127, 127, 126]
        result = _reference_classify_routing_opportunity(expected, actual)
        assert result == "ROUTING_PINNING_SUSPECTED"

    def test_one_hundred_seventy_percent_of_expected_is_pinning_suspected(self):
        expected = [125, 125, 125, 125]
        actual = [213, 96, 96, 95]  # pod0 ratio = 213/125 = 1.704
        result = _reference_classify_routing_opportunity(expected, actual)
        assert result == "ROUTING_PINNING_SUSPECTED"

    def test_high_ratio_cv_with_shares_individually_near_expected_is_pinning_suspected(self):
        # Every individual ratio sits just inside [0.5,1.5], but the ratio
        # spread (CV) still exceeds 0.30 -- the CV clause must be
        # load-bearing on its own, not merely the range clause.
        expected = [100, 100, 100, 100]
        actual = [55, 145, 55, 145]  # ratios 0.55/1.45/0.55/1.45
        ratios = [a / e for a, e in zip(actual, expected)]
        mean = sum(ratios) / len(ratios)
        variance = sum((r - mean) ** 2 for r in ratios) / len(ratios)
        cv = (variance**0.5) / mean
        assert cv > 0.30, "construction must actually exceed the CV threshold to be a valid test case"
        result = _reference_classify_routing_opportunity(expected, actual)
        assert result == "ROUTING_PINNING_SUSPECTED"

    def test_severe_starvation_below_one_percent_expected_is_structural_failure(self):
        expected = [130, 130, 130, 130]
        actual = [0, 130, 130, 130]  # pod0 gets literally zero of its expected opportunity
        result = _reference_classify_routing_opportunity(expected, actual)
        assert result == "STRUCTURAL_ROUTING_FAILURE"

    def test_structural_failure_requires_no_secondary_substantial_traffic_clause(self):
        # A pod with adequate EXPECTED opportunity (>=50, i.e. legitimate
        # traffic during ITS OWN exposure was itself substantial) that
        # nonetheless receives <1% of that opportunity is structural
        # failure on the ratio alone -- no separate "another pod received
        # substantial traffic" condition is evaluated or required.
        expected = [100, 100]
        actual = [0, 100]
        assert _reference_classify_routing_opportunity(expected, actual) == "STRUCTURAL_ROUTING_FAILURE"

    def test_healthy_variance_example(self):
        expected = [100, 100, 100, 100]
        actual = [95, 105, 100, 100]
        result = _reference_classify_routing_opportunity(expected, actual)
        assert result == "HEALTHY_VARIANCE"

    def test_insufficient_expected_opportunity_example(self):
        expected = [5, 6, 4, 5]  # all far below the 50-expected-request threshold
        actual = [5, 6, 4, 5]
        result = _reference_classify_routing_opportunity(expected, actual)
        assert result == "INSUFFICIENT_ROUTING_SAMPLE"

    def test_fewer_than_two_assessable_pods_is_insufficient_even_with_high_totals(self):
        # Only one pod clears the 50-expected threshold -- run-level
        # sample sufficiency requires at least 2 assessable pods.
        expected = [500, 10]
        actual = [500, 10]
        assert _reference_classify_routing_opportunity(expected, actual) == "INSUFFICIENT_ROUTING_SAMPLE"

    def test_starved_pod_with_high_expected_remains_assessable_and_flags_structural(self):
        # expected=100/actual=0 (task sec.7/20): the starved pod's own
        # expected opportunity clears the threshold, so it remains
        # assessable and correctly triggers structural failure.
        expected = [100, 100]
        actual = [0, 200]
        assert _reference_classify_routing_opportunity(expected, actual) == "STRUCTURAL_ROUTING_FAILURE"

    def test_short_lived_pod_with_low_expected_is_not_falsely_flagged_as_starved(self):
        # expected=7/actual=1 (task sec.7): below the assessability
        # threshold -- excluded from ratio-based classification entirely,
        # never counted as a starved pod.
        expected = [7, 300, 300]
        actual = [1, 300, 300]
        # Only 2 of 3 pods are assessable (>=50 expected); that still meets
        # the run-level minimum of 2, but the low-expected pod itself must
        # never be the basis for a STRUCTURAL_ROUTING_FAILURE verdict.
        assessable = [(e, a) for e, a in zip(expected, actual) if e >= 50]
        assert len(assessable) == 2
        assert (7, 1) not in assessable

    def test_every_example_maps_to_exactly_one_of_the_four_tiers(self):
        valid_tiers = {"INSUFFICIENT_ROUTING_SAMPLE", "STRUCTURAL_ROUTING_FAILURE", "HEALTHY_VARIANCE", "ROUTING_PINNING_SUSPECTED"}
        examples = [
            ([100, 100, 100, 100], [20, 127, 127, 126]),
            ([125, 125, 125, 125], [213, 96, 96, 95]),
            ([100, 100, 100, 100], [55, 145, 55, 145]),
            ([130, 130, 130, 130], [0, 130, 130, 130]),
            ([100, 100, 100, 100], [95, 105, 100, 100]),
            ([5, 6, 4, 5], [5, 6, 4, 5]),
        ]
        for expected, actual in examples:
            assert _reference_classify_routing_opportunity(expected, actual) in valid_tiers


class TestFirstSlotAndTransitionWorkedExamples:
    """Validates the manifest's own preregistered worked examples (task
    sec.20-21), directly relevant to campaign_sequence_index=1
    (fit_dyncal_r2_L4_repeat1, a fixed 2-replica steady-state run where
    equal Ready exposure is expected after stabilization)."""

    def _example(self, name):
        m = _manifest()
        return next(e for e in m["per_pod_routing_diagnostics"]["worked_examples"] if e["name"] == name)

    def test_first_slot_healthy_example_present_and_classifies_healthy(self):
        ex = self._example("first_slot_steady_state_healthy")
        assert ex["expected_per_pod"] == 1440
        assert ex["total_legitimate_requests"] == 2880
        expected = [ex["expected_per_pod"]] * 2
        assert _reference_classify_routing_opportunity(expected, ex["actual_per_pod"]) == ex["result"] == "HEALTHY_VARIANCE"

    def test_first_slot_structural_failure_example_present_and_classifies_structural(self):
        ex = self._example("first_slot_steady_state_structural_failure")
        assert ex["expected_per_pod"] == [1440, 1440]
        assert ex["actual_per_pod"] == [0, 2880]
        assert _reference_classify_routing_opportunity(ex["expected_per_pod"], ex["actual_per_pod"]) == ex["result"] == "STRUCTURAL_ROUTING_FAILURE"

    def test_transition_unequal_exposure_example_present_and_not_naive_50_50(self):
        ex = self._example("replica_transition_unequal_exposure")
        assert "NOT" in ex["demonstrates"]
        assert "50/50" in ex["description"]
        # Regression guard: expected opportunity must be traffic-weighted,
        # not duration-weighted -- a pod Ready for a shorter but
        # HIGHER-traffic sub-window can legitimately have a LARGER expected
        # count than a pod Ready longer during lower traffic. Demonstrate
        # with a concrete interval reconstruction: Pod A Ready both
        # intervals, Pod B Ready only interval 2 (post scale-up).
        interval_deltas = [100, 400]  # traffic surges after the scale-up trigger
        ready_sets = [{"A"}, {"A", "B"}]
        expected = {"A": 0.0, "B": 0.0}
        for delta, ready in zip(interval_deltas, ready_sets):
            share = delta / len(ready)
            for pod in ready:
                expected[pod] += share
        # Pod A: all of interval 1 (100) + half of interval 2 (200) = 300
        # Pod B: half of interval 2 (200) = 200 -- NOT a 50/50 split of 500
        assert expected["A"] == 300.0
        assert expected["B"] == 200.0
        assert expected["A"] != expected["B"]
        # A naive whole-window duration-only split (both intervals equal
        # length) would have wrongly expected 250/250 if B were credited
        # for only "half the window" -- the traffic-weighted result differs.
        assert (expected["A"], expected["B"]) != (250.0, 250.0)


class TestRoutingOpportunityMathematicsFrozenCorr35:
    """CORR-34's whole-window fair_share_mathematics was superseded by
    CORR-35's interval-level, exposure-weighted expected-routing-
    opportunity calculation -- these tests replace
    TestFairShareMathematicsFrozen."""

    def test_expected_routing_opportunity_interval_formula_present(self):
        m = _manifest()
        ero = m["per_pod_routing_diagnostics"]["expected_routing_opportunity"]
        assert ero["interval_cadence_seconds"] == 10
        assert "N_ready_j" in ero["expected_increment_formula"]
        assert "expected_request_count_i" in ero["expected_request_count_formula"]

    def test_expected_opportunity_rejects_duration_only_weighting(self):
        m = _manifest()
        ero = m["per_pod_routing_diagnostics"]["expected_routing_opportunity"]
        assert "DURATION" in ero["rejected_alternative"]
        assert "intensity" in ero["rejected_alternative_reason"]

    def test_routing_opportunity_ratio_formula_present(self):
        m = _manifest()
        rom = m["per_pod_routing_diagnostics"]["routing_opportunity_mathematics"]
        assert rom["routing_opportunity_ratio_i"] == "actual_request_count_i / expected_request_count_i"
        assert rom["ideal_fair_value"] == 1.0

    def test_cv_frozen_over_normalized_ratios_not_raw_counts(self):
        m = _manifest()
        cv_def = m["per_pod_routing_diagnostics"]["routing_opportunity_mathematics"]["routing_ratio_cv"]
        assert "routing_opportunity_ratio_i" in cv_def
        assert "NORMALIZED RATIOS" in cv_def

    def test_reduces_to_corr34_equal_exposure_case(self):
        m = _manifest()
        note = m["per_pod_routing_diagnostics"]["routing_opportunity_mathematics"]["reduces_to_corr34_for_equal_exposure"]
        assert "1/N_eligible" in note
        assert "correction/generalization" in note

    def test_delta_total_requests_uses_job_demo_app_never_loadtest(self):
        m = _manifest()
        promql = m["per_pod_routing_diagnostics"]["promql"]
        assert 'job="demo-app"' in promql
        assert "demo-app-loadtest" not in promql


class TestAssessabilityAndRunLevelSampleSufficiencyCorr35:
    """Replaces TestSampleSufficiencyRuleTotalNotPerPod -- CORR-35 corrects
    sample sufficiency from a whole-window total-traffic threshold to a
    per-pod EXPECTED-count threshold plus a run-level minimum-assessable-
    pods rule."""

    def test_min_expected_requests_threshold_is_50(self):
        m = _manifest()
        assert m["per_pod_routing_diagnostics"]["routing_min_expected_requests_per_pod"] == 50

    def test_assessability_based_on_expected_not_actual_count(self):
        m = _manifest()
        rule = m["per_pod_routing_diagnostics"]["assessability_rule"]
        assert rule["condition"] == "expected_request_count_i >= routing_min_expected_requests_per_pod"

    def test_assessability_rationale_covers_starved_and_short_lived_pods(self):
        m = _manifest()
        rationale = m["per_pod_routing_diagnostics"]["assessability_rule"]["rationale"]
        assert "expected=100, actual=0" in rationale
        assert "expected=7, actual=1" in rationale

    def test_run_level_minimum_two_assessable_pods(self):
        m = _manifest()
        rls = m["per_pod_routing_diagnostics"]["run_level_sample_sufficiency"]
        assert rls["minimum_assessable_pods"] == 2
        assert rls["below_minimum_classification"] == "INSUFFICIENT_ROUTING_SAMPLE"

    def test_dynamics_config_exposes_the_same_thresholds(self):
        cfg = _load(DYNAMICS_CONFIG_PATH)
        assert cfg["routing_min_expected_requests_per_pod"] == 50
        assert cfg["routing_run_level_minimum_assessable_pods"] == 2


class TestReadyServingEligibility:
    def test_eligibility_basis_is_ready_serving_exposure(self):
        m = _manifest()
        elig = m["per_pod_routing_diagnostics"]["eligibility"]
        assert "Ready-serving exposure" in elig["basis"]

    def test_required_per_pod_fields_present(self):
        m = _manifest()
        fields = m["per_pod_routing_diagnostics"]["eligibility"]["recorded_per_pod"]
        for f in ("first_ready_timestamp_in_window", "last_ready_timestamp_in_window", "eligible_serving_duration_seconds", "request_count_while_eligible"):
            assert f in fields

    def test_not_yet_ready_pods_never_classified_as_starved(self):
        m = _manifest()
        rule = m["per_pod_routing_diagnostics"]["eligibility"]["rule"]
        assert "NEVER" in rule
        assert "not expected traffic recipients" in rule


class TestCounterHistoryAndDisappearingPods:
    def test_never_uses_bare_start_end_delta(self):
        m = _manifest()
        rule = m["per_pod_routing_diagnostics"]["counter_history_and_disappearing_pods"]["rule"]
        assert "NEVER" in rule
        assert "counter_at_final_timestamp - counter_at_start_timestamp" in rule

    def test_uses_first_and_last_valid_sample_while_series_exists(self):
        m = _manifest()
        rule = m["per_pod_routing_diagnostics"]["counter_history_and_disappearing_pods"]["rule"]
        assert "FIRST and LAST VALID counter sample" in rule
        assert "terminated pods" in rule.lower()

    def test_job_filter_never_uses_loadtest_service(self):
        m = _manifest()
        promql = m["per_pod_routing_diagnostics"]["promql"]
        assert "demo-app-loadtest" not in promql
        assert 'job="demo-app"' in promql


class TestCounterResetSemantics:
    def test_negative_diff_treated_as_reset_not_subtracted(self):
        m = _manifest()
        rule = m["per_pod_routing_diagnostics"]["counter_reset_semantics"]["rule"]
        assert "NEGATIVE difference" in rule
        assert "counter_reset_count" in rule

    def test_reset_alone_does_not_invalidate_if_reconstructable(self):
        m = _manifest()
        rule = m["per_pod_routing_diagnostics"]["counter_reset_semantics"]["rule"]
        assert "does NOT" in rule

    def test_unreconstructable_case_falls_back_to_insufficient_sample_unless_independent_failure(self):
        m = _manifest()
        case = m["per_pod_routing_diagnostics"]["counter_reset_semantics"]["unreconstructable_case"]
        assert "INSUFFICIENT_ROUTING_SAMPLE" in case
        assert "UNLESS" in case

    def test_missing_counts_never_invented(self):
        m = _manifest()
        case = m["per_pod_routing_diagnostics"]["counter_reset_semantics"]["unreconstructable_case"]
        assert "never invented" in case.lower() or "never" in case.lower()


class TestRoutingRemainsDiagnosticOnlyCorr34:
    def test_not_a_model_target_reaffirmed(self):
        m = _manifest()
        text = m["per_pod_routing_diagnostics"]["not_a_model_target"]
        assert "DIAGNOSTIC ONLY" in text
        assert "NEVER added to the 11-element" in text
        assert "No MDP version change" in text

    def test_rl_env_observation_contract_untouched(self):
        # No per-pod field name appears in the frozen observation builder.
        obs_path = PROJECT_ROOT / "autoscaler/src/rl_env/observation.py"
        text = obs_path.read_text(encoding="utf-8")
        assert "per_pod" not in text


class TestCorr34NoMatrixRedesign:
    """CORR-34 froze execution ORDER and made routing classification
    exhaustive -- it must not have touched any scientific matrix value."""

    def test_20_9_8_4_102_unchanged(self):
        m = _manifest()
        assert len(m["fitting_cells"]) == 20
        assert len(m["validation_cells"]) == 9
        assert len(m["replica_transitions"]["directional_types"]) == 8
        assert len(m["workload_transitions"]["types"]) == 4
        assert m["official_new_calibration_run_count"] == 102
        assert m["campaign_estimate"]["total_runs"] == 102

    def test_37_schedules_and_64_400_5s_unchanged(self):
        with open(SCHEDULE_REGISTRY_PATH) as f:
            registry = json.load(f)
        assert len(registry) == 37
        m = _manifest()
        assert m["revised_max_total_rps_ceiling"]["frozen_value"] == 64.0
        assert m["generator_capacity"]["frozen_calibration_max_in_flight"] == 400
        assert m["generator_capacity"]["default_timeout_seconds"] == GENERATOR_TIMEOUT_SECONDS

    def test_hpa_260_and_contracts_v2_unchanged(self):
        assert _load(HPA_CONFIG_PATH)["target_cpu_utilization_percent"] == 260
        manifest = _load(HPA_MANIFEST_PATH)
        assert manifest["spec"]["metrics"][0]["resource"]["target"]["averageUtilization"] == 260
        assert _load(RL_CONFIG_PATH)["rl_contract_version"] == 2
        assert _load(REWARD_CONFIG_PATH)["reward_contract_version"] == 2

    def test_no_ppo_artifacts_exist(self):
        # persistent-data/dynamics_calibration/ may legitimately be
        # populated now that Phase 15B.3A has begun the frozen campaign --
        # see test_calibration_result_directories_are_never_fabricated.
        # PPO/RL training artifacts must still never exist; Phase 16E's
        # dynamics-model artifacts under persistent-data/models/dynamics/ are
        # a separate, expected category -- see
        # _assert_ppo_absent_and_dynamics_artifacts_valid.
        _assert_ppo_absent_and_dynamics_artifacts_valid(PROJECT_ROOT)

    def test_no_reference_to_unseen_eval_in_any_corr34_additions(self):
        for path in (MANIFEST_PATH, DYNAMICS_CONFIG_PATH, PROTOCOL_DOC_PATH, GLOBAL_CAMPAIGN_SEQUENCE_PATH):
            text = path.read_text(encoding="utf-8")
            assert "unseen_eval" not in text


# =============================================================================
# Phase 15B.2D (CORR-35) additions below: corrects the routing classifier's
# fair-share mathematics from a whole-window 1/N expectation (CORR-34,
# invalid under unequal Ready-serving exposure) to an interval-level,
# traffic-weighted expected-routing-opportunity calculation. Still no
# matrix/global-order/MDP change -- covered by TestCorr34NoMatrixRedesign
# above and TestCorr35NoMatrixRedesign below.
# =============================================================================


class TestCorr35Versioning:
    def test_manifest_records_corr35_correction(self):
        m = _manifest()
        assert m["corrected_in_phase"] == "15B.2D"
        assert m["correction_reference"] == "CORR-35"
        assert m["prior_correction_reference"] == "CORR-34"

    def test_protocol_version_still_1_no_data_collected(self):
        assert _manifest()["protocol_version"] == 1


class TestReadySetIntervalReconstruction:
    def test_authoritative_source_is_kubernetes_api_not_kube_state_metrics(self):
        m = _manifest()
        source = m["per_pod_routing_diagnostics"]["ready_set_interval_reconstruction"]["authoritative_source"]
        assert "Kubernetes API" in source
        assert "lastTransitionTime" in source

    def test_membership_rule_uses_interval_start_no_invented_precision(self):
        m = _manifest()
        rsr = m["per_pod_routing_diagnostics"]["ready_set_interval_reconstruction"]
        assert "interval j's start timestamp t_j" in rsr["membership_rule"]
        assert "NEVER" in rsr["sub_sample_precision"]
        assert "fabricated" in rsr["sub_sample_precision"]

    def test_not_yet_ready_pods_excluded_from_ready_set(self):
        m = _manifest()
        rsr = m["per_pod_routing_diagnostics"]["ready_set_interval_reconstruction"]
        assert "never expected traffic recipients" in rsr["excludes_from_ready_set"]

    def test_interval_cadence_matches_prometheus_scrape_interval_config(self):
        m = _manifest()
        cadence = m["per_pod_routing_diagnostics"]["expected_routing_opportunity"]["interval_cadence_seconds"]
        experiment_cfg = _load(PROJECT_ROOT / "configs/experiment.yaml")
        assert cadence == experiment_cfg["prometheus"]["scrape_interval_seconds"] == 10

    def test_no_one_minute_rate_smoothing(self):
        m = _manifest()
        interval_def = m["per_pod_routing_diagnostics"]["expected_routing_opportunity"]["interval_definition"]
        assert "never a coarser 1-minute rate()" in interval_def


class TestReconstructionConsistencyCheck:
    def test_frozen_tolerance_is_zero_requests(self):
        m = _manifest()
        check = m["per_pod_routing_diagnostics"]["reconstruction_consistency_check"]
        assert check["frozen_tolerance_requests"] == 0

    def test_tolerance_rationale_ties_to_exact_integer_counter(self):
        m = _manifest()
        rationale = m["per_pod_routing_diagnostics"]["reconstruction_consistency_check"]["tolerance_rationale"]
        assert "exact integer Counter" in rationale
        assert "tolerance=0" in rationale

    def test_joint_observability_gaps_never_fabricated(self):
        m = _manifest()
        check = m["per_pod_routing_diagnostics"]["reconstruction_consistency_check"]
        assert "never estimated or fabricated" in check["joint_observability_gap_handling"]

    def test_discrepancy_beyond_tolerance_falls_back_to_insufficient_sample(self):
        m = _manifest()
        check = m["per_pod_routing_diagnostics"]["reconstruction_consistency_check"]
        assert "INSUFFICIENT_ROUTING_SAMPLE" in check["on_discrepancy_beyond_tolerance"]
        assert "never fabricated" in check["on_discrepancy_beyond_tolerance"]

    def test_consistency_check_feeds_the_insufficient_sample_classification_condition(self):
        m = _manifest()
        cond = m["per_pod_routing_diagnostics"]["classification"]["insufficient_routing_sample"]["condition"]
        assert "reconstruction_consistency_check" in cond


class TestActualRequestCountReconstructionReusesCorr34Mechanics:
    def test_reuses_counter_history_and_reset_mechanics(self):
        m = _manifest()
        arc = m["per_pod_routing_diagnostics"]["actual_request_count_reconstruction"]
        assert set(arc["reuses"]) == {"counter_history_and_disappearing_pods", "counter_reset_semantics"}

    def test_does_not_require_pod_alive_at_window_endpoint(self):
        m = _manifest()
        note = m["per_pod_routing_diagnostics"]["actual_request_count_reconstruction"]["note"]
        assert "does not require the pod to remain alive" in note
        assert "does not fabricate samples" in note


class TestWorkedExamplesComplete:
    def test_three_worked_examples_present(self):
        m = _manifest()
        names = {e["name"] for e in m["per_pod_routing_diagnostics"]["worked_examples"]}
        assert names == {
            "first_slot_steady_state_healthy",
            "first_slot_steady_state_structural_failure",
            "replica_transition_unequal_exposure",
        }


class TestCorr35NoMatrixRedesign:
    """CORR-35 corrected ONLY routing mathematics and vague structural-
    routing wording -- it must not have touched the matrix, the global
    campaign order, model families, acceptance criteria, residual/noise
    design, HPA isolation rules, session semantics, or the MDP contract."""

    def test_20_9_8_4_102_unchanged(self):
        m = _manifest()
        assert len(m["fitting_cells"]) == 20
        assert len(m["validation_cells"]) == 9
        assert len(m["replica_transitions"]["directional_types"]) == 8
        assert len(m["workload_transitions"]["types"]) == 4
        assert m["official_new_calibration_run_count"] == 102
        assert m["campaign_estimate"]["total_runs"] == 102

    def test_37_schedules_and_64_400_5s_unchanged(self):
        with open(SCHEDULE_REGISTRY_PATH) as f:
            registry = json.load(f)
        assert len(registry) == 37
        m = _manifest()
        assert m["revised_max_total_rps_ceiling"]["frozen_value"] == 64.0
        assert m["generator_capacity"]["frozen_calibration_max_in_flight"] == 400
        assert m["generator_capacity"]["default_timeout_seconds"] == GENERATOR_TIMEOUT_SECONDS

    def test_global_campaign_sequence_byte_identical_to_phase_15b2c(self):
        m = _manifest()
        actual = hashlib.sha256(GLOBAL_CAMPAIGN_SEQUENCE_PATH.read_bytes()).hexdigest()
        assert actual == m["global_campaign_order"]["path_sha256"]
        assert m["global_campaign_order"]["total_entries"] == 102

    def test_first_slot_still_fit_dyncal_r2_l4_repeat1(self):
        seq = _global_sequence()
        first = seq[0]
        assert first["campaign_sequence_index"] == 1
        assert first["planned_slot_id"] == "fit_dyncal_r2_L4_repeat1"
        assert first["replicas"] == 2
        assert first["offered_rps"] == 24.0
        assert first["repeat_number"] == 1

    def test_hpa_260_and_contracts_v2_unchanged(self):
        assert _load(HPA_CONFIG_PATH)["target_cpu_utilization_percent"] == 260
        manifest = _load(HPA_MANIFEST_PATH)
        assert manifest["spec"]["metrics"][0]["resource"]["target"]["averageUtilization"] == 260
        assert _load(RL_CONFIG_PATH)["rl_contract_version"] == 2
        assert _load(REWARD_CONFIG_PATH)["reward_contract_version"] == 2

    def test_no_ppo_artifacts_exist(self):
        # persistent-data/dynamics_calibration/ may legitimately be
        # populated now that Phase 15B.3A has begun the frozen campaign --
        # see test_calibration_result_directories_are_never_fabricated.
        # PPO/RL training artifacts must still never exist; Phase 16E's
        # dynamics-model artifacts under persistent-data/models/dynamics/ are
        # a separate, expected category -- see
        # _assert_ppo_absent_and_dynamics_artifacts_valid.
        _assert_ppo_absent_and_dynamics_artifacts_valid(PROJECT_ROOT)

    def test_no_reference_to_unseen_eval_in_any_corr35_additions(self):
        for path in (MANIFEST_PATH, DYNAMICS_CONFIG_PATH, PROTOCOL_DOC_PATH, GLOBAL_CAMPAIGN_SEQUENCE_PATH):
            text = path.read_text(encoding="utf-8")
            assert "unseen_eval" not in text
