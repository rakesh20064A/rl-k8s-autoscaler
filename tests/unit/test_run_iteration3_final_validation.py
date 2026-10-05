"""tests/unit/test_run_iteration3_final_validation.py

Phase 16P: focused tests over the new Iteration-3 final-validation pre-live
closure wrapper and its initialized ledger. No test in this module sends
live traffic, mutates Kubernetes/HPA, fits/refits any model, scores any
prediction, or runs PPO.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_iteration3_final_validation as w
from scripts.dynamics_calibration_lib import load_ledger, LEDGER_STATE_VALID, LEDGER_STATE_NOT_ATTEMPTED

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestAllSlotsResolveExactly:
    def test_all_32_sequence_slots_resolve(self):
        for i in range(1, 33):
            info = w.resolve_slot(i)
            assert info["sequence_index"] == i
            assert info["cell_id"].startswith("i3val_r")
            assert info["repeat_number"] in (1, 2)
            assert info["replicas"] in range(1, 9)
            assert info["schedule_path"]
            assert len(info["schedule_sha256"]) == 64

    def test_out_of_range_index_raises(self):
        with pytest.raises(ValueError):
            w.resolve_slot(0)
        with pytest.raises(ValueError):
            w.resolve_slot(33)

    def test_each_of_16_cells_appears_exactly_twice_with_repeats_1_and_2(self):
        by_cell = {}
        for i in range(1, 33):
            info = w.resolve_slot(i)
            by_cell.setdefault(info["cell_id"], set()).add(info["repeat_number"])
        assert len(by_cell) == 16
        assert all(reps == {1, 2} for reps in by_cell.values())


class TestScheduleHashesMatchFrozenRegistry:
    def test_every_slot_schedule_hash_matches_registry(self):
        registry = w.load_i3val_schedule_registry()
        for i in range(1, 33):
            info = w.resolve_slot(i)
            assert info["schedule_sha256"] == registry[info["cell_id"]]["schedule_sha256"]

    def test_schedule_bytes_match_frozen_hash(self):
        for i in range(1, 33):
            info = w.resolve_slot(i)
            path = PROJECT_ROOT / info["schedule_path"]
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual == info["schedule_sha256"]


class TestPinnedHashes:
    def test_fresh_matrix_hash_pinned(self):
        path = Path("/Users/rakeshdandamudi/Capstone/task_outputs/phase_16n_iteration3_fresh_validation_matrix.csv")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == w.EXPECTED_FRESH_MATRIX_SHA256

    def test_sequence_csv_hash_pinned(self):
        path = Path("/Users/rakeshdandamudi/Capstone/task_outputs/phase_16n_iteration3_final_validation_sequence.csv")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == w.EXPECTED_SEQUENCE_CSV_SHA256

    def test_sequence_projection_hash_pinned_and_reproducible(self):
        assert hashlib.sha256(w.I3VAL_SEQUENCE_PROJECTION_PATH.read_bytes()).hexdigest() == w.EXPECTED_SEQUENCE_PROJECTION_SHA256
        sequence = w.load_verified_i3val_sequence()
        reproduced = json.dumps(sequence, indent=2) + "\n"
        assert hashlib.sha256(reproduced.encode()).hexdigest() == w.EXPECTED_SEQUENCE_PROJECTION_SHA256

    def test_schedule_registry_hash_pinned(self):
        assert hashlib.sha256(w.I3VAL_SCHEDULE_REGISTRY_PATH.read_bytes()).hexdigest() == w.EXPECTED_SCHEDULE_REGISTRY_SHA256

    def test_protocol_hash_pinned(self):
        path = Path("/Users/rakeshdandamudi/Capstone/task_outputs/phase_16n_iteration3_final_validation_protocol.yaml")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == w.EXPECTED_PROTOCOL_SHA256

    def test_prediction_preregistration_hash_pinned_and_matches_external_audit(self):
        path = Path("/Users/rakeshdandamudi/Capstone/task_outputs/phase_16o_iteration3_final_validation_prediction_preregistration.csv")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == w.EXPECTED_PREDICTION_PREREGISTRATION_SHA256
        assert actual == "525bd55ad0d9c609329607a7fb8c8522a20a2a04bb91a0ccc775814952d57d8f"

    def test_verify_all_governing_hashes_passes(self):
        report = w.verify_all_governing_hashes()
        assert all(v["match"] for v in report.values())


class TestCorr52Corr54SourceUnchanged:
    def test_corr52_traffic_generator_hash_unchanged(self):
        """CORR-52's original purpose was to catch any UNDISCLOSED edit to
        this file during/shortly after the Iteration-3 live campaign
        (Phase16Q), before Phase16S's adjudication locked its results. That
        campaign is now complete and its evidence permanently closed
        (Phase16S/16T). Phase16Z (2026-09-30) made a disclosed, authorized,
        scientifically-scoped fix to `parse_traffic_profile()` -- adding
        NEW `elif` branches recognizing four previously-unhandled flat
        training-profile schemas -- for RL live-training traffic, a
        capability that did not exist during Iteration-3. This is provably
        behavior-preserving for every calibration-era profile: those all
        use the unified `phases:` schema, whose own parsing branch
        (`if "phases" in data:`) is untouched by this fix, and Phase16Z's
        own regression tests
        (tests/unit/test_traffic_generator_profile_parser.py) directly
        verify this. The hash below is intentionally updated to the new,
        disclosed value -- not silently, and not to hide a change -- see
        task_outputs/phase_16z_parser_correction_report.md."""
        path = PROJECT_ROOT / "loadgen/traffic_generator.py"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == "8c90012a5a9676409f9f9a9ffe67c0e6c00af8decc2fb151fe6481091455f80c"

    def test_corr54_runner_hash_unchanged(self):
        path = PROJECT_ROOT / "scripts/run_dynamics_calibration_slot.py"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == "c97bb95098f5347a3563a08bf14c5d59d1d6aa2af263004e9aa2361ececb9f5d"


class TestInitialLedger:
    def test_ledger_has_exactly_32_planned_slots(self):
        ledger = load_ledger(w.I3VAL_LEDGER_PATH)
        assert ledger is not None
        assert len(ledger["slots"]) == 32

    def test_ledger_reflects_completed_32_of_32_live_campaign(self):
        # The Iteration-3 live-collection campaign (Phase16Q/16Q.1/16Q.2)
        # has genuinely completed: all 32 frozen sequence slots resolved to
        # VALID on their first physical attempt, with zero retries. This
        # test intentionally asserts the POST-campaign state rather than
        # the pre-campaign "0 attempts" baseline it originally checked --
        # the ledger is a live, append-only scientific record, not a fixed
        # fixture, and this suite must describe its actual current state
        # honestly rather than assume it is still untouched.
        ledger = load_ledger(w.I3VAL_LEDGER_PATH)
        assert all(s["state"] == LEDGER_STATE_VALID for s in ledger["slots"].values())
        assert all(len(s["all_attempt_ids"]) == 1 for s in ledger["slots"].values()), (
            "every slot must have resolved on exactly one physical attempt -- zero retries"
        )
        assert all(s["valid_attempt_id"] is not None for s in ledger["slots"].values())

    def test_ledger_provenance_block_present_and_correct(self):
        ledger = load_ledger(w.I3VAL_LEDGER_PATH)
        prov = ledger["provenance"]
        assert prov["fitting_weight"] == 0
        assert prov["fitting_weight_is_permanent"] is True
        assert prov["n_cells"] == 16
        assert prov["n_repeats_per_cell"] == 2
        assert prov["n_slots"] == 32
        assert prov["third_scientific_repeat_permitted"] is False
        assert prov["execution_protocol_sha256"] == w.EXPECTED_PROTOCOL_SHA256
        assert prov["prediction_preregistration_sha256"] == w.EXPECTED_PREDICTION_PREREGISTRATION_SHA256
        assert prov["model_hashes"]["cpu_cores"] == "e27a910b5c8d078cdac36e20461599fa4a18b0045c8ca6681b120d95605d5543"

    def test_iteration2b_ledger_untouched(self):
        from scripts import run_iteration2b_final_validation_slot as it2bval
        digest = hashlib.sha256(it2bval.I2BVAL_LEDGER_PATH.read_bytes()).hexdigest()
        assert digest == "a5739bac0b8eb33b3e8acaa258dc4b426c16add0cd33404217aa03377cfced16"

    def test_iteration3_root_ledger_sequence_registry_disjoint_from_prior_campaigns(self):
        assert w.I3VAL_PERSISTENT_ROOT not in w._OTHER_ROOTS
        assert w.I3VAL_LEDGER_PATH not in w._OTHER_LEDGERS
        assert w.I3VAL_SEQUENCE_PROJECTION_PATH not in w._OTHER_SEQUENCES
        assert w.I3VAL_SCHEDULE_REGISTRY_PATH not in w._OTHER_REGISTRIES


class TestDryRunDoesNotMutateLedger:
    def test_dry_run_all_does_not_change_ledger_hash(self):
        before = hashlib.sha256(w.I3VAL_LEDGER_PATH.read_bytes()).hexdigest()
        rc = w.main(["--dry-run-all"])
        assert rc == 0
        after = hashlib.sha256(w.I3VAL_LEDGER_PATH.read_bytes()).hexdigest()
        assert before == after

    def test_single_slot_dry_run_does_not_change_ledger_hash(self):
        before = hashlib.sha256(w.I3VAL_LEDGER_PATH.read_bytes()).hexdigest()
        rc = w.main(["--validation-sequence-index", "5", "--dry-run"])
        assert rc == 0
        after = hashlib.sha256(w.I3VAL_LEDGER_PATH.read_bytes()).hexdigest()
        assert before == after

    def test_dry_run_creates_no_attempt_directories(self):
        before = {p.name for p in w.I3VAL_PERSISTENT_ROOT.iterdir()}
        w.main(["--validation-sequence-index", "10", "--dry-run"])
        after = {p.name for p in w.I3VAL_PERSISTENT_ROOT.iterdir()}
        assert before == after


class TestPhysicalAttemptNumbering:
    def test_untouched_slot_first_attempt_is_1(self, tmp_path):
        n = w.resolve_next_physical_attempt_number(tmp_path, "i3val_r2_V1_repeat1")
        assert n == 1

    def test_never_resets_after_a_consumed_attempt(self, tmp_path):
        (tmp_path / "i3val_r2_V1_repeat1_attempt1_20260101T000000Z").mkdir()
        assert w.resolve_next_physical_attempt_number(tmp_path, "i3val_r2_V1_repeat1") == 2

    def test_pre_traffic_tooling_failure_directory_still_consumes_a_number(self, tmp_path):
        # A preserved pre-traffic-tooling-failure directory (no ledger row)
        # still physically exists on disk and must still consume its number.
        (tmp_path / "i3val_r8_V1_repeat2_attempt1_20260101T000000Z").mkdir()
        assert w.resolve_next_physical_attempt_number(tmp_path, "i3val_r8_V1_repeat2") == 2

    def test_numbering_is_per_slot_not_global(self, tmp_path):
        (tmp_path / "i3val_r2_V1_repeat1_attempt1_20260101T000000Z").mkdir()
        assert w.resolve_next_physical_attempt_number(tmp_path, "i3val_r7_V1_repeat2") == 1


class TestStrictSequenceOrder:
    def test_index_1_permitted_execute_when_ledger_absent(self, tmp_path, monkeypatch):
        # An absent ledger path means "nothing has ever been attempted" --
        # index 1 is always permitted in that case, regardless of the real
        # campaign ledger's current (now fully-completed) state.
        monkeypatch.setattr(w, "I3VAL_LEDGER_PATH", tmp_path / "does_not_exist.json")
        assert w._earlier_indices_all_valid(1) is True

    def test_index_2_not_permitted_when_index_1_not_valid(self, tmp_path, monkeypatch):
        # Isolated mock ledger (not the real campaign ledger, which has now
        # genuinely completed all 32 slots VALID) so this test's assertion
        # remains meaningful regardless of live-campaign progress.
        ledger_path = tmp_path / "fake_ledger.json"
        ledger_path.write_text(json.dumps({"slots": {"1": {"state": LEDGER_STATE_NOT_ATTEMPTED}}}))
        monkeypatch.setattr(w, "I3VAL_LEDGER_PATH", ledger_path)
        assert w._earlier_indices_all_valid(2) is False

    def test_index_n_permitted_once_all_earlier_are_valid(self, tmp_path, monkeypatch):
        ledger_path = tmp_path / "fake_ledger.json"
        ledger = {"slots": {str(i): {"state": LEDGER_STATE_VALID} for i in range(1, 5)}}
        ledger_path.write_text(json.dumps(ledger))
        monkeypatch.setattr(w, "I3VAL_LEDGER_PATH", ledger_path)
        assert w._earlier_indices_all_valid(5) is True

    def test_index_n_not_permitted_when_one_earlier_is_missing(self, tmp_path, monkeypatch):
        ledger_path = tmp_path / "fake_ledger.json"
        ledger = {"slots": {"1": {"state": LEDGER_STATE_VALID}, "2": {"state": LEDGER_STATE_NOT_ATTEMPTED},
                            "3": {"state": LEDGER_STATE_VALID}, "4": {"state": LEDGER_STATE_VALID}}}
        ledger_path.write_text(json.dumps(ledger))
        monkeypatch.setattr(w, "I3VAL_LEDGER_PATH", ledger_path)
        assert w._earlier_indices_all_valid(5) is False


class TestNoThirdRepeatNoFittingWeight:
    def test_no_cell_has_more_than_2_repeats_in_frozen_sequence(self):
        sequence = w.load_verified_i3val_sequence()
        by_cell = {}
        for e in sequence:
            by_cell.setdefault(e["cell_or_transition_id"], []).append(e["repeat_number"])
        assert all(len(v) == 2 for v in by_cell.values())

    def test_fitting_weight_is_always_zero_in_ledger_provenance(self):
        ledger = load_ledger(w.I3VAL_LEDGER_PATH)
        assert ledger["provenance"]["fitting_weight"] == 0


class TestWrapperIsAThinDelegationLayerOnly:
    def test_no_prediction_scoring_code_in_wrapper(self):
        source = Path(w.__file__).read_text()
        for forbidden in ("normalized_error", "s_eff", "balanced_accuracy", "median_normalized_error", "acceptance_gate"):
            assert forbidden not in source

    def test_no_model_fitting_code_in_wrapper(self):
        source = Path(w.__file__).read_text()
        assert ".fit(" not in source
        assert "DecisionTreeRegressor" not in source
        assert "IsotonicRegression" not in source
        assert "lstsq" not in source

    def test_no_ppo_path_in_wrapper(self):
        import re
        source = Path(w.__file__).read_text()
        assert re.search(r"\bppo\b", source, re.IGNORECASE) is None
        assert "stable_baselines" not in source.lower()

    def test_wrapper_never_implements_its_own_traffic_send(self):
        source = Path(w.__file__).read_text()
        assert "aiohttp.ClientSession" not in source
        assert "requests.post" not in source
        # it must delegate, not duplicate
        assert "historical.main(" in source

    def test_wrapper_never_reads_final_validation_outcome_metrics(self):
        source = Path(w.__file__).read_text()
        for forbidden in ("decision_aligned_observations", "metrics.csv", "pod_readiness_history"):
            assert forbidden not in source


class TestScientificReturnCodeContractSourceLevel:
    """The exit-code/ledger contract is PRE_EXISTING_FROZEN (CORR-54) inside
    scripts/run_dynamics_calibration_slot.py; these tests verify, at the
    source level, that the wrapper does not alter or intercept it."""

    def test_wrapper_returns_historical_main_exit_code_unmodified(self):
        source = Path(w.__file__).read_text()
        assert "return historical.main(" in source

    def test_corr54_exit_code_3_constant_unchanged(self):
        from scripts import run_dynamics_calibration_slot as historical
        assert historical.EXIT_PRE_TRAFFIC_TOOLING_FAILURE == 3

    def test_corr54_pretraffictoolingfailure_never_writes_ledger(self):
        import inspect as _inspect
        source = _inspect.getsource(historical_main_module().main)
        # The PreTrafficToolingFailure except-block must return before any
        # ACTUAL atomic_write_ledger CALL reachable from that branch (a
        # comment merely mentioning the function name, e.g. "atomic_write_ledger()
        # is never called for this attempt", does not count as a call).
        idx = source.index("except PreTrafficToolingFailure")
        return_idx = source.index("return EXIT_PRE_TRAFFIC_TOOLING_FAILURE")
        block = source[idx:return_idx]
        code_lines = [ln for ln in block.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        assert not any("atomic_write_ledger(" in ln for ln in code_lines)


def historical_main_module():
    from scripts import run_dynamics_calibration_slot as historical
    return historical
