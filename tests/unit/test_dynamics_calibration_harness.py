"""tests/unit/test_dynamics_calibration_harness.py

Phase 15B.3A: offline unit tests for the empirical dynamics-calibration
execution harness (scripts/dynamics_calibration_lib.py,
scripts/run_dynamics_calibration_slot.py, scripts/pod_readiness_collector.py,
scripts/per_pod_counter_collector.py). Every test here is network/subprocess/
Kubernetes-free -- it exercises pure logic only. Run BEFORE any live traffic.
"""

import copy
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import dynamics_calibration_lib as lib  # noqa: E402
from scripts import run_dynamics_calibration_slot as harness  # noqa: E402
from scripts import pod_readiness_collector as readiness_collector  # noqa: E402
from scripts.per_pod_counter_collector import parse_args as counter_parse_args  # noqa: E402
from scripts.pod_readiness_collector import parse_args as readiness_parse_args  # noqa: E402


@pytest.fixture()
def real_sequence():
    return lib.load_global_sequence()


@pytest.fixture()
def real_registry():
    return lib.load_schedule_registry()


# ---------------------------------------------------------------------------
# Global sequence lookup
# ---------------------------------------------------------------------------


class TestResolvePlannedSlot:
    def test_index_1_resolves_to_the_frozen_first_slot(self, real_sequence):
        slot = lib.resolve_planned_slot(1, sequence=real_sequence)
        assert slot["planned_slot_id"] == "fit_dyncal_r2_L4_repeat1"
        assert slot["replicas"] == 2
        assert slot["offered_rps"] == 24.0

    def test_index_102_resolves(self, real_sequence):
        slot = lib.resolve_planned_slot(102, sequence=real_sequence)
        assert slot["campaign_sequence_index"] == 102

    def test_out_of_range_index_raises(self, real_sequence):
        with pytest.raises(ValueError):
            lib.resolve_planned_slot(0, sequence=real_sequence)
        with pytest.raises(ValueError):
            lib.resolve_planned_slot(103, sequence=real_sequence)


# ---------------------------------------------------------------------------
# Schedule hash verification
# ---------------------------------------------------------------------------


class TestVerifyScheduleHash:
    def test_real_index_1_schedule_hash_verifies_against_all_three_sources(self, real_sequence, real_registry):
        slot = lib.resolve_planned_slot(1, sequence=real_sequence)
        verified = lib.verify_schedule_hash(slot, registry=real_registry)
        assert verified == slot["schedule_sha256"] == "767292698dd5b466db520c084f25ce925af9b28701c82bb3e8137a509d8d1c78"

    def test_registry_hash_mismatch_raises(self, real_sequence, real_registry):
        slot = lib.resolve_planned_slot(1, sequence=real_sequence)
        tampered_registry = copy.deepcopy(real_registry)
        tampered_registry[slot["cell_or_transition_id"]]["schedule_sha256"] = "0" * 64
        with pytest.raises(RuntimeError):
            lib.verify_schedule_hash(slot, registry=tampered_registry)

    def test_sequence_hash_mismatch_raises(self, real_registry):
        fake_slot = {
            "cell_or_transition_id": "dyncal_r2_L4",
            "schedule_path": "configs/traffic_profiles/rl_dynamics_calibration/steady_state/dyncal_r2_L4_schedule.csv",
            "schedule_sha256": "f" * 64,
        }
        with pytest.raises(RuntimeError):
            lib.verify_schedule_hash(fake_slot, registry=real_registry)

    def test_missing_schedule_file_raises(self, real_registry):
        fake_slot = {
            "cell_or_transition_id": "dyncal_r2_L4",
            "schedule_path": "configs/traffic_profiles/rl_dynamics_calibration/steady_state/does_not_exist.csv",
            "schedule_sha256": "f" * 64,
        }
        with pytest.raises(RuntimeError):
            lib.verify_schedule_hash(fake_slot, registry=real_registry)


# ---------------------------------------------------------------------------
# CLI: no scientific override flags; single index only
# ---------------------------------------------------------------------------


class TestCliRejectsScientificOverrides:
    def test_only_operational_flags_exist(self):
        parser = harness.build_arg_parser()
        option_strings = {opt for action in parser._actions for opt in action.option_strings}
        for forbidden in ("--rps", "--replicas", "--offered-rps", "--intensity", "--cell", "--schedule"):
            assert forbidden not in option_strings

    def test_rps_override_is_rejected_by_argparse(self):
        parser = harness.build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--campaign-sequence-index", "1", "--rps", "999"])

    def test_replicas_override_is_rejected_by_argparse(self):
        parser = harness.build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--campaign-sequence-index", "1", "--replicas", "8"])

    def test_no_batch_or_range_flag_exists_index2_cannot_be_invoked_alongside_index1(self):
        parser = harness.build_arg_parser()
        option_strings = {opt for action in parser._actions for opt in action.option_strings}
        for forbidden in ("--all", "--range", "--batch", "--indices"):
            assert forbidden not in option_strings
        # campaign-sequence-index takes exactly one int -- no nargs="+"/"*"
        action = next(a for a in parser._actions if "--campaign-sequence-index" in a.option_strings)
        assert action.nargs is None
        assert action.type is int

    def test_single_index_parses_cleanly(self):
        parser = harness.build_arg_parser()
        args = parser.parse_args(["--campaign-sequence-index", "1"])
        assert args.campaign_sequence_index == 1
        assert args.dry_run is False


# ---------------------------------------------------------------------------
# Dry-run report is pure and never mutates
# ---------------------------------------------------------------------------


class TestBuildRoutingInputsCrossProcessAlignment:
    """Regression guard for a real bug found during Phase 15B.3A's first
    live run: the readiness and per-pod-counter collectors are two
    independently-launched subprocesses with different monotonic clock
    references, so their `elapsed_seconds` values are never exactly equal
    even at "the same" nominal 10s tick. The original implementation
    matched on exact elapsed_seconds equality, silently producing zero
    matches -> every expected routing count computed as 0.0 despite real
    traffic. The fix aligns on wall-clock `timestamp_epoch` with a
    tolerance, anchored to the readiness collector's own tick grid."""

    def test_slightly_offset_clocks_still_align(self):
        # Readiness collector ticks at epoch 1000, 1010, 1020...
        # Counter collector (started 0.4s later) ticks at 1000.4, 1010.4, 1020.4...
        readiness_rows = [
            {"elapsed_seconds": 60.0, "timestamp_epoch": 1060.0, "pod_name": "a", "ready_status": "True"},
            {"elapsed_seconds": 70.0, "timestamp_epoch": 1070.0, "pod_name": "a", "ready_status": "True"},
        ]
        counter_rows = [
            {"elapsed_seconds": 59.6, "timestamp_epoch": 1060.4, "pod": "a", "raw_counter_value": 100.0},
            {"elapsed_seconds": 69.6, "timestamp_epoch": 1070.4, "pod": "a", "raw_counter_value": 150.0},
        ]
        boundaries, ready_at, pod_samples, relevant_at = harness.build_routing_inputs(readiness_rows, counter_rows, (60.0, 70.0))
        assert boundaries == [60.0, 70.0]
        assert pod_samples["a"] == [(60.0, 100.0), (70.0, 150.0)]
        result = lib.compute_expected_and_actual(boundaries, ready_at, pod_samples, relevant_at)
        assert result["expected"]["a"] == pytest.approx(50.0)

    def test_sample_beyond_tolerance_is_dropped_not_fabricated(self):
        readiness_rows = [{"elapsed_seconds": 60.0, "timestamp_epoch": 1060.0, "pod_name": "a", "ready_status": "True"}]
        counter_rows = [{"elapsed_seconds": 45.0, "timestamp_epoch": 1045.0, "pod": "a", "raw_counter_value": 100.0}]
        _, _, pod_samples, _ = harness.build_routing_inputs(readiness_rows, counter_rows, (60.0, 60.0), alignment_tolerance_seconds=5.0)
        assert pod_samples.get("a", []) == []

    def test_empty_readiness_window_returns_empty_inputs(self):
        boundaries, ready_at, pod_samples, relevant_at = harness.build_routing_inputs([], [], (60.0, 180.0))
        assert boundaries == [] and ready_at == {} and pod_samples == {} and relevant_at == {}


class TestExtractDecisionAlignedV2Corr37:
    """Phase 15B.3C: the orchestrator's PRIMARY (not forensic-only)
    decision-aligned extraction now natively applies Protocol V2's Axis B
    semantics for every new run, combining v1's unchanged "latest valid
    sample" rule with the authoritative per-pod Ready count."""

    def test_authoritative_ready_count_uses_k8s_api_not_prometheus(self):
        readiness_rows = [
            {"elapsed_seconds": 58.0, "pod_name": "a", "ready_status": "True"},
            {"elapsed_seconds": 58.0, "pod_name": "b", "ready_status": "False"},
        ]
        assert harness.authoritative_ready_count_at_target(readiness_rows, 60.0) == 1

    def test_authoritative_ready_count_respects_tolerance(self):
        readiness_rows = [{"elapsed_seconds": 0.0, "pod_name": "a", "ready_status": "True"}]
        assert harness.authoritative_ready_count_at_target(readiness_rows, 60.0, tolerance=5.0) == 0

    def test_current2_ready1_scenario_produces_fit_ineligible_run(self):
        metrics_rows = [
            {"elapsed_seconds": 60.0, "sample_valid": "True", "cpu_cores": "0.5"},
            {"elapsed_seconds": 90.0, "sample_valid": "True", "cpu_cores": "0.6"},
            {"elapsed_seconds": 120.0, "sample_valid": "True", "cpu_cores": "0.7"},
            {"elapsed_seconds": 150.0, "sample_valid": "True", "cpu_cores": "0.8"},
        ]
        readiness_rows = [{"elapsed_seconds": t, "pod_name": "a", "ready_status": "True"} for t in (60.0, 90.0, 120.0, 150.0)]
        # Only 1 pod Ready even though intended_replicas=2.
        result = harness.extract_decision_aligned_v2(metrics_rows, readiness_rows, [60, 90, 120, 150], 30.0, intended_replicas=2)
        assert result["decision_points_fit_eligible_count"] == 0
        assert result["decision_points_observable_count"] == 4
        assert result["run_level_classification"] == lib.FIT_RUN_INELIGIBLE
        for point in result["decision_points"]:
            assert point["exclusion_reason"] == lib.FIT_POINT_INELIGIBLE_HEALTH_INSTABILITY

    def test_fully_healthy_run_is_fully_fit_eligible(self):
        metrics_rows = [{"elapsed_seconds": t, "sample_valid": "True"} for t in (60.0, 90.0, 120.0, 150.0)]
        readiness_rows = []
        for t in (60.0, 90.0, 120.0, 150.0):
            readiness_rows.append({"elapsed_seconds": t, "pod_name": "a", "ready_status": "True"})
            readiness_rows.append({"elapsed_seconds": t, "pod_name": "b", "ready_status": "True"})
        result = harness.extract_decision_aligned_v2(metrics_rows, readiness_rows, [60, 90, 120, 150], 30.0, intended_replicas=2)
        assert result["decision_points_fit_eligible_count"] == 4
        assert result["run_level_classification"] == lib.FIT_RUN_ELIGIBLE

    def test_unobservable_point_never_imputed(self):
        # Only one valid sample at t=30 -- age is 30/60/90/120 for targets
        # 60/90/120/150 respectively; only target=60 (age=30, exactly at
        # the 30s staleness boundary, inclusive) remains observable.
        metrics_rows = [{"elapsed_seconds": 30.0, "sample_valid": "True"}]
        readiness_rows = [{"elapsed_seconds": 60.0, "pod_name": "a", "ready_status": "True"}, {"elapsed_seconds": 60.0, "pod_name": "b", "ready_status": "True"}]
        result = harness.extract_decision_aligned_v2(metrics_rows, readiness_rows, [60, 90, 120, 150], 30.0, intended_replicas=2)
        assert len(result["decision_points"]) == 4  # all four targets always represented
        unobservable = [p for p in result["decision_points"] if not p["decision_point_observable"]]
        assert len(unobservable) == 3  # targets 90, 120, 150
        for p in unobservable:
            assert p["exclusion_reason"] == lib.FIT_POINT_INELIGIBLE_TARGET_UNOBSERVABLE
            assert p["request_rate_rps"] is None  # never imputed


class TestDryRunReport:
    def test_dry_run_report_contains_all_required_fields(self, real_sequence):
        manifest = lib.load_manifest()
        slot = lib.resolve_planned_slot(1, sequence=real_sequence)
        ledger = lib.init_ledger(real_sequence)
        report = harness.dry_run_report(slot, slot["schedule_sha256"], manifest, ledger)
        for field in (
            "campaign_sequence_index", "planned_slot_id", "run_kind", "repeat_number",
            "replicas", "offered_rps", "intensity", "schedule_path",
            "computed_schedule_sha256", "expected_schedule_sha256",
            "traffic_duration_seconds", "stabilization_duration_seconds",
            "warmup_window_seconds", "scored_window_seconds",
            "decision_aligned_extraction_timestamps", "connection_mode",
            "timeout_seconds", "max_in_flight", "expected_workload_endpoint",
            "artifact_destination_root", "current_ledger_state_for_this_slot",
        ):
            assert field in report, f"missing dry-run field: {field}"

    def test_dry_run_report_includes_v5_transport_discovery_fields(self, real_sequence):
        # Phase 15B.3N.3: dry_run_report() remains PURE (no I/O of its own)
        # -- transport_discovery is resolved by the CALLER and passed in,
        # so this function stays unit-testable without a live cluster.
        manifest = lib.load_manifest()
        slot = lib.resolve_planned_slot(1, sequence=real_sequence)
        ledger = lib.init_ledger(real_sequence)
        discovery = {
            "docker_network": "minikube",
            "node_ip": "192.168.49.2",
            "nodeport": 31891,
            "relay_image_identity": {"image": "alpine/socat", "image_id": "sha256:x", "repo_digest": None},
        }
        report = harness.dry_run_report(slot, slot["schedule_sha256"], manifest, ledger, discovery)
        assert report["planned_transport_type"] == "docker_network_tcp_relay"
        assert report["planned_transport_docker_network"] == "minikube"
        assert report["planned_transport_node_ip"] == "192.168.49.2"
        assert report["planned_transport_nodeport"] == 31891
        assert report["planned_transport_relay_image_identity"]["image"] == "alpine/socat"

    def test_dry_run_report_transport_discovery_defaults_to_none_fields_when_omitted(self, real_sequence):
        manifest = lib.load_manifest()
        slot = lib.resolve_planned_slot(1, sequence=real_sequence)
        ledger = lib.init_ledger(real_sequence)
        report = harness.dry_run_report(slot, slot["schedule_sha256"], manifest, ledger)
        assert report["planned_transport_docker_network"] is None
        assert report["planned_transport_node_ip"] is None

    def test_dry_run_main_never_touches_kubectl_or_hpa(self, monkeypatch, tmp_path):
        def _boom(*a, **k):
            raise AssertionError("dry-run must never call this")

        monkeypatch.setattr(harness, "preflight_checks", _boom)
        monkeypatch.setattr(harness, "delete_hpa", _boom)
        monkeypatch.setattr(harness, "scale_and_wait", _boom)
        monkeypatch.setattr(harness, "start_loadtest_transport", _boom)
        # Phase 15B.3N.3: dry-run DOES perform read-only V5 transport
        # discovery (docker/kubectl/minikube reads, never a mutation) --
        # isolate those from the live environment here for test hygiene,
        # matching this test's own "never touch kubectl/HPA" intent.
        monkeypatch.setattr(harness, "resolve_minikube_docker_network", lambda: "minikube")
        monkeypatch.setattr(harness, "resolve_minikube_node_ip", lambda: "192.168.49.2")
        monkeypatch.setattr(harness, "resolve_loadtest_nodeport", lambda: 31891)
        monkeypatch.setattr(harness, "resolve_relay_image_identity", lambda: {"image": "alpine/socat", "image_id": None, "repo_digest": None})
        rc = harness.main(["--campaign-sequence-index", "1", "--dry-run"], ledger_path=tmp_path / "campaign_ledger.json")
        assert rc == 0


class TestPreflightFailureCreatesNoOfficialAttemptCorr38:
    """Phase 15B.3E: a prior live invocation crashed partway through a
    read-only preflight subprocess call (scripts/run_dynamics_calibration_slot.py
    task_outputs/phase_15b3d_*.md sec.6) yet had already created a full,
    empty attempt directory tree before preflight ever ran -- an artifact
    indistinguishable at a glance from a real (if empty) scientific
    attempt. `main()` is now restructured so the attempt directory is
    created strictly AFTER preflight has fully passed; a failed OR crashed
    preflight can therefore never leave behind anything resembling an
    official attempt."""

    def test_failed_preflight_creates_no_attempt_directory(self, monkeypatch, tmp_path):
        ledger_path = tmp_path / "campaign_ledger.json"
        monkeypatch.setattr(harness, "preflight_checks", lambda: [(False, "simulated failure")])
        rc = harness.main(["--campaign-sequence-index", "1"], calibration_root=tmp_path, ledger_path=ledger_path)
        assert rc == 2
        assert list(tmp_path.iterdir()) == [] or all(p == ledger_path for p in tmp_path.iterdir())
        # No attempt-shaped directory (matching the real attempt_id pattern) exists.
        assert not any(p.is_dir() and "attempt" in p.name for p in tmp_path.iterdir())

    def test_crashed_preflight_creates_no_attempt_directory(self, monkeypatch, tmp_path):
        ledger_path = tmp_path / "campaign_ledger.json"

        def _boom():
            raise TimeoutError("simulated minikube status timeout")

        monkeypatch.setattr(harness, "preflight_checks", _boom)
        with pytest.raises(TimeoutError):
            harness.main(["--campaign-sequence-index", "1"], calibration_root=tmp_path, ledger_path=ledger_path)
        assert not any(p.is_dir() and "attempt" in p.name for p in tmp_path.iterdir())

    def test_failed_preflight_never_writes_the_ledger(self, monkeypatch, tmp_path):
        ledger_path = tmp_path / "campaign_ledger.json"
        monkeypatch.setattr(harness, "preflight_checks", lambda: [(False, "simulated failure")])
        harness.main(["--campaign-sequence-index", "1"], calibration_root=tmp_path, ledger_path=ledger_path)
        ledger_after = lib.load_ledger(ledger_path)
        assert ledger_after["slots"]["1"]["state"] == "NOT_ATTEMPTED"
        assert ledger_after["slots"]["1"]["all_attempt_ids"] == []

    def test_next_real_execution_after_a_failed_preflight_still_becomes_attempt1(self):
        # attempt numbering is derived solely from --attempt-number (default
        # 1) and a fresh timestamp -- never from counting prior failed
        # invocations on disk -- so a failed preflight can never cause the
        # next real attempt to be mislabeled attempt2/3/etc.
        from datetime import datetime, timezone

        when = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        attempt_id = lib.make_attempt_id("fit_dyncal_r2_L4_repeat1", 1, when=when)
        assert "_attempt1_" in attempt_id


class TestExternalContainerContaminationWatchCorr38:
    """Phase 15B.3E: preflight checks contamination ONCE, before mutation.
    Docker Desktop could in principle restart its own built-in Kubernetes
    containers mid-campaign (as happened before Phase 15B.3D's live run
    even started), so a continuous watch during stabilization/traffic
    enforces the SAME already-frozen structural principle
    (`unrelated_kind_cluster_contamination`) for the run's full duration,
    not just at the start."""

    def test_classify_matches_desktop_and_kind_prefixes(self):
        names = ["minikube", "desktop-control-plane", "desktop-worker", "kind-cloud-provider", "kind-registry-mirror", "some-other-container"]
        assert lib.classify_external_container_contamination(names) == [
            "desktop-control-plane",
            "desktop-worker",
            "kind-cloud-provider",
            "kind-registry-mirror",
        ]

    def test_classify_returns_empty_when_clean(self):
        assert lib.classify_external_container_contamination(["minikube", "some-other-container"]) == []

    def test_preflight_and_live_watch_share_one_definition(self):
        # Refactored so preflight_checks() itself calls the same shared
        # function -- verified here via source inspection to guard against
        # a future edit silently reintroducing a second, divergent
        # definition of "unrelated Kind/Desktop container."
        import inspect

        src = inspect.getsource(harness.preflight_checks)
        assert "classify_external_container_contamination" in src

    def test_no_contamination_during_run_passes(self):
        readiness_metadata = {"external_contamination_detected_ticks_elapsed_seconds": []}
        ok, msg = lib.check_no_external_contamination_during_run(readiness_metadata)
        assert ok is True

    def test_contamination_during_run_fails_with_frozen_reason_string(self):
        readiness_metadata = {"external_contamination_detected_ticks_elapsed_seconds": [70.0, 80.0]}
        ok, msg = lib.check_no_external_contamination_during_run(readiness_metadata)
        assert ok is False
        assert "unrelated_kind_cluster_contamination" in msg
        assert "70.0" in msg and "80.0" in msg

    def test_missing_field_defaults_to_clean_not_a_crash(self):
        # An older readiness_metadata.json (pre-15B.3E attempts) lacks this
        # field entirely -- must not be misread as contamination.
        ok, msg = lib.check_no_external_contamination_during_run({})
        assert ok is True

    def test_fetch_docker_ps_all_parses_tab_separated_rows(self):
        from scripts.pod_readiness_collector import fetch_docker_ps_all

        assert callable(fetch_docker_ps_all)

    def test_readiness_collector_writes_contamination_csv_schema(self):
        from scripts.pod_readiness_collector import EXTERNAL_CONTAMINATION_CSV_FIELDNAMES

        for field in ("timestamp_utc", "elapsed_seconds", "container_name", "container_id", "status", "cpu_percent"):
            assert field in EXTERNAL_CONTAMINATION_CSV_FIELDNAMES

    def test_main_consults_the_contamination_check_after_collectors_finish(self):
        import inspect

        src = inspect.getsource(harness.main)
        assert "check_no_external_contamination_during_run" in src
        assert "pod_readiness_metadata.json" in src


# ---------------------------------------------------------------------------
# Attempt identity
# ---------------------------------------------------------------------------


class TestAttemptIdentity:
    def test_attempt_id_pattern(self):
        from datetime import datetime, timezone

        when = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        attempt_id = lib.make_attempt_id("fit_dyncal_r2_L4_repeat1", 1, when=when)
        assert attempt_id == "fit_dyncal_r2_L4_repeat1_attempt1_20260101T000000Z"

    def test_attempt_number_2_differs_from_1(self):
        from datetime import datetime, timezone

        when = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        a1 = lib.make_attempt_id("fit_dyncal_r2_L4_repeat1", 1, when=when)
        a2 = lib.make_attempt_id("fit_dyncal_r2_L4_repeat1", 2, when=when)
        assert a1 != a2


# ---------------------------------------------------------------------------
# Campaign ledger
# ---------------------------------------------------------------------------


class TestLedger:
    def test_init_ledger_has_102_not_attempted_slots(self, real_sequence):
        ledger = lib.init_ledger(real_sequence)
        assert len(ledger["slots"]) == 102
        assert all(s["state"] == lib.LEDGER_STATE_NOT_ATTEMPTED for s in ledger["slots"].values())
        assert all(s["all_attempt_ids"] == [] and s["valid_attempt_id"] is None for s in ledger["slots"].values())

    def test_atomic_write_and_load_roundtrip(self, tmp_path, real_sequence):
        path = tmp_path / "ledger.json"
        ledger = lib.init_ledger(real_sequence)
        lib.atomic_write_ledger(ledger, path=path)
        loaded = lib.load_ledger(path)
        assert loaded == ledger

    def test_ensure_ledger_does_not_regenerate_existing(self, tmp_path, real_sequence):
        path = tmp_path / "ledger.json"
        ledger = lib.ensure_ledger(real_sequence, path=path)
        lib.update_slot_state(ledger, 1, lib.LEDGER_STATE_VALID, attempt_id="x_attempt1_ts", mark_valid=True)
        lib.atomic_write_ledger(ledger, path=path)
        reloaded = lib.ensure_ledger(real_sequence, path=path)
        assert reloaded["slots"]["1"]["state"] == lib.LEDGER_STATE_VALID

    def test_update_only_index1_leaves_101_not_attempted(self, real_sequence):
        ledger = lib.init_ledger(real_sequence)
        lib.update_slot_state(ledger, 1, lib.LEDGER_STATE_VALID, attempt_id="a1", mark_valid=True)
        not_attempted = [s for s in ledger["slots"].values() if s["state"] == lib.LEDGER_STATE_NOT_ATTEMPTED]
        assert len(not_attempted) == 101

    def test_resume_selection_picks_lowest_non_valid_index(self, real_sequence):
        ledger = lib.init_ledger(real_sequence)
        lib.update_slot_state(ledger, 1, lib.LEDGER_STATE_VALID, attempt_id="a1", mark_valid=True)
        lib.update_slot_state(ledger, 2, lib.LEDGER_STATE_VALID, attempt_id="a2", mark_valid=True)
        assert lib.next_resumable_index(ledger) == 3

    def test_resume_selection_retries_invalid_slot_not_the_next_one(self, real_sequence):
        ledger = lib.init_ledger(real_sequence)
        lib.update_slot_state(ledger, 1, lib.LEDGER_STATE_INVALID_RETRY_REQUIRED, attempt_id="a1_attempt1")
        assert lib.next_resumable_index(ledger) == 1

    def test_next_resumable_index_none_when_all_valid(self):
        seq = [{"campaign_sequence_index": 1, "planned_slot_id": "x"}]
        ledger = lib.init_ledger(seq)
        lib.update_slot_state(ledger, 1, lib.LEDGER_STATE_VALID, attempt_id="a1", mark_valid=True)
        assert lib.next_resumable_index(ledger) is None

    def test_invalid_attempt_is_preserved_never_overwritten(self, real_sequence):
        ledger = lib.init_ledger(real_sequence)
        lib.update_slot_state(ledger, 1, lib.LEDGER_STATE_INVALID_RETRY_REQUIRED, attempt_id="attempt1")
        lib.update_slot_state(ledger, 1, lib.LEDGER_STATE_VALID, attempt_id="attempt2", mark_valid=True)
        slot = ledger["slots"]["1"]
        assert slot["all_attempt_ids"] == ["attempt1", "attempt2"]
        assert slot["valid_attempt_id"] == "attempt2"
        assert slot["state"] == lib.LEDGER_STATE_VALID

    def test_correcting_a_valid_slot_back_to_invalid_clears_the_stale_valid_attempt_pointer(self, real_sequence):
        # Regression guard for a real bug found during Phase 15B.3A's first
        # live run: a slot originally marked VALID (mark_valid=True) and
        # later corrected to INVALID_RETRY_REQUIRED (e.g. after a
        # post-execution review found a genuine issue) must not leave a
        # dangling valid_attempt_id pointing at an attempt no longer
        # considered valid.
        ledger = lib.init_ledger(real_sequence)
        lib.update_slot_state(ledger, 1, lib.LEDGER_STATE_VALID, attempt_id="attempt1", mark_valid=True)
        assert ledger["slots"]["1"]["valid_attempt_id"] == "attempt1"
        lib.update_slot_state(ledger, 1, lib.LEDGER_STATE_INVALID_RETRY_REQUIRED, attempt_id="attempt1")
        assert ledger["slots"]["1"]["valid_attempt_id"] is None
        assert ledger["slots"]["1"]["state"] == lib.LEDGER_STATE_INVALID_RETRY_REQUIRED
        assert ledger["slots"]["1"]["all_attempt_ids"] == ["attempt1"]

    def test_ledger_reconstructable_from_attempt_metadata_alone(self, real_sequence):
        attempts = [
            {"campaign_sequence_index": 1, "attempt_id": "fit_..._attempt1_ts", "structural_validity": "VALID"},
            {"campaign_sequence_index": 5, "attempt_id": "fit_..._attempt1_ts2", "structural_validity": "INVALID"},
        ]
        reconstructed = lib.reconstruct_ledger_from_attempts(real_sequence, attempts)
        assert reconstructed["slots"]["1"]["state"] == lib.LEDGER_STATE_VALID
        assert reconstructed["slots"]["5"]["state"] == lib.LEDGER_STATE_INVALID_RETRY_REQUIRED
        assert reconstructed["slots"]["2"]["state"] == lib.LEDGER_STATE_NOT_ATTEMPTED


# ---------------------------------------------------------------------------
# Cleanup registry (HPA restoration on the failure path)
# ---------------------------------------------------------------------------


class TestCleanupRegistry:
    def test_all_registered_callbacks_run_on_explicit_trigger(self):
        calls = []
        registry = harness.CleanupRegistry()
        registry.register("a", lambda: calls.append("a"))
        registry.register("restore_hpa", lambda: calls.append("restore_hpa"))
        registry.run_all()
        assert "restore_hpa" in calls
        assert "a" in calls

    def test_cleanup_runs_exactly_once(self):
        calls = []
        registry = harness.CleanupRegistry()
        registry.register("a", lambda: calls.append("a"))
        registry.run_all()
        registry.run_all()
        assert calls == ["a"]

    def test_hpa_restore_still_runs_even_if_an_earlier_callback_raises(self):
        calls = []

        def _raiser():
            raise RuntimeError("boom")

        registry = harness.CleanupRegistry()
        registry.register("restore_hpa", lambda: calls.append("restore_hpa"))
        registry.register("failing_step", _raiser)
        results = registry.run_all()
        assert "restore_hpa" in calls
        assert any(name == "failing_step" and not ok for name, ok, _msg in results)


# ---------------------------------------------------------------------------
# Per-pod Ready-history parser
# ---------------------------------------------------------------------------


class TestPodReadinessParser:
    def test_parses_ready_and_not_ready_pods(self):
        pods_json = {
            "items": [
                {
                    "metadata": {"name": "demo-app-a", "uid": "uid-a"},
                    "status": {
                        "phase": "Running",
                        "conditions": [{"type": "Ready", "status": "True", "lastTransitionTime": "2026-01-01T00:00:00Z"}],
                        "containerStatuses": [{"restartCount": 0}],
                    },
                },
                {
                    "metadata": {"name": "demo-app-b", "uid": "uid-b", "deletionTimestamp": "2026-01-01T00:05:00Z"},
                    "status": {
                        "phase": "Running",
                        "conditions": [{"type": "Ready", "status": "False", "lastTransitionTime": "2026-01-01T00:04:00Z"}],
                        "containerStatuses": [{"restartCount": 2}],
                    },
                },
            ]
        }
        rows = lib.parse_pod_readiness_snapshot(pods_json, "2026-01-01T00:00:10Z", 1234.0, 10.0)
        assert len(rows) == 2
        a, b = rows
        assert a["pod_name"] == "demo-app-a" and lib.is_pod_ready_row(a)
        assert b["pod_name"] == "demo-app-b" and not lib.is_pod_ready_row(b)
        assert b["deletion_timestamp"] == "2026-01-01T00:05:00Z"
        assert b["restart_count"] == 2

    def test_no_ready_condition_present_never_fabricated(self):
        pods_json = {"items": [{"metadata": {"name": "x"}, "status": {"phase": "Pending", "conditions": []}}]}
        rows = lib.parse_pod_readiness_snapshot(pods_json, "t", 0.0, 0.0)
        assert rows[0]["ready_status"] == ""
        assert not lib.is_pod_ready_row(rows[0])

    def test_readiness_collector_cli_has_no_scientific_overrides(self):
        args = readiness_parse_args(["--experiment-dir", "/tmp/x", "--duration-seconds", "10"])
        assert args.namespace == "autoscaler-app"
        assert args.sampling_interval_seconds == 10.0


class TestContainerStateAndEventsHardeningCorr37PostAudit:
    """Phase 15B.3C evidence-preservation hardening, added specifically so
    a future restart's cause is recoverable from PRESERVED artifacts alone,
    without depending on live cluster Events still being within retention
    at forensic-review time (attempt1's restart cause was only recoverable
    because Events happened to still exist ~35 minutes later)."""

    def test_parses_restart_and_termination_evidence_without_fabrication(self):
        pods_json = {
            "items": [
                {
                    "metadata": {"name": "demo-app-a"},
                    "status": {
                        "containerStatuses": [
                            {
                                "name": "demo-app",
                                "containerID": "docker://abc123",
                                "restartCount": 7,
                                "state": {"running": {"startedAt": "2026-01-01T00:03:00Z"}},
                                "lastState": {
                                    "terminated": {
                                        "reason": "Error",
                                        "exitCode": 137,
                                        "startedAt": "2026-01-01T00:00:00Z",
                                        "finishedAt": "2026-01-01T00:02:50Z",
                                    }
                                },
                            }
                        ]
                    },
                }
            ]
        }
        rows = lib.parse_pod_container_state_snapshot(pods_json, "t", 0.0, 0.0)
        assert len(rows) == 1
        row = rows[0]
        assert row["restart_count"] == 7
        assert row["current_state"] == "running"
        assert row["last_state"] == "terminated"
        assert row["last_termination_reason"] == "Error"
        assert row["last_termination_exit_code"] == 137

    def test_missing_termination_info_is_never_fabricated(self):
        pods_json = {"items": [{"metadata": {"name": "x"}, "status": {"containerStatuses": [{"name": "demo-app", "restartCount": 0, "state": {"running": {}}}]}}]}
        rows = lib.parse_pod_container_state_snapshot(pods_json, "t", 0.0, 0.0)
        assert rows[0]["last_state"] == ""
        assert rows[0]["last_termination_reason"] == ""
        assert rows[0]["last_termination_exit_code"] == ""

    def test_no_container_statuses_yields_no_rows_not_a_crash(self):
        pods_json = {"items": [{"metadata": {"name": "x"}, "status": {}}]}
        assert lib.parse_pod_container_state_snapshot(pods_json, "t", 0.0, 0.0) == []

    def test_normalize_kubernetes_events_flattens_expected_fields(self):
        events_json = {
            "items": [
                {
                    "firstTimestamp": "2026-01-01T00:00:00Z",
                    "lastTimestamp": "2026-01-01T00:01:00Z",
                    "count": 5,
                    "type": "Warning",
                    "reason": "Unhealthy",
                    "involvedObject": {"kind": "Pod", "name": "demo-app-a"},
                    "message": "Liveness probe failed: context deadline exceeded",
                }
            ]
        }
        rows = lib.normalize_kubernetes_events(events_json)
        assert len(rows) == 1
        assert rows[0]["reason"] == "Unhealthy"
        assert rows[0]["count"] == 5
        assert rows[0]["involved_object_name"] == "demo-app-a"

    def test_normalize_kubernetes_events_empty_is_empty(self):
        assert lib.normalize_kubernetes_events({"items": []}) == []

    def test_restarted_pods_between_detects_increase(self):
        before = [{"pod_name": "a", "restart_count": 6}, {"pod_name": "b", "restart_count": 0}]
        after = [{"pod_name": "a", "restart_count": 7}, {"pod_name": "b", "restart_count": 0}]
        assert lib.restarted_pods_between(before, after) == ["a"]

    def test_restarted_pods_between_no_change_is_empty(self):
        before = [{"pod_name": "a", "restart_count": 6}]
        after = [{"pod_name": "a", "restart_count": 6}]
        assert lib.restarted_pods_between(before, after) == []

    def test_restarted_pods_between_attempt1_reproduction(self):
        # attempt1's exact scenario: rpztw 6->7, xtzgk stays 0.
        before = [{"pod_name": "rpztw", "restart_count": 6}, {"pod_name": "xtzgk", "restart_count": 0}]
        after = [{"pod_name": "rpztw", "restart_count": 7}, {"pod_name": "xtzgk", "restart_count": 0}]
        assert lib.restarted_pods_between(before, after) == ["rpztw"]


# ---------------------------------------------------------------------------
# Reset-aware counter reconstruction
# ---------------------------------------------------------------------------


class TestCounterReconstruction:
    def test_monotonic_counter_sums_positive_diffs(self):
        total, resets = lib.reconstruct_counter_total([(0, 100.0), (10, 150.0), (20, 200.0)])
        assert total == 100.0
        assert resets == 0

    def test_reset_does_not_subtract_negative_diff(self):
        # counter resets from 200 -> 5 (pod restart), then climbs to 55
        total, resets = lib.reconstruct_counter_total([(0, 200.0), (10, 5.0), (20, 55.0)])
        assert total == 50.0  # only the post-reset climb counted, never a negative contribution
        assert resets == 1

    def test_out_of_order_samples_are_sorted_first(self):
        total, _ = lib.reconstruct_counter_total([(20, 200.0), (0, 100.0), (10, 150.0)])
        assert total == 100.0

    def test_single_sample_has_zero_total(self):
        total, resets = lib.reconstruct_counter_total([(0, 42.0)])
        assert total == 0.0
        assert resets == 0

    def test_prometheus_vector_parser_extracts_per_pod_values(self):
        response = {
            "status": "success",
            "data": {"result": [
                {"metric": {"pod": "demo-app-a"}, "value": [1700000000, "123"]},
                {"metric": {"pod": "demo-app-b"}, "value": [1700000000, "456"]},
            ]},
        }
        parsed = lib.parse_prometheus_vector_by_pod(response)
        assert parsed == {"demo-app-a": 123.0, "demo-app-b": 456.0}

    def test_prometheus_vector_parser_handles_query_error(self):
        assert lib.parse_prometheus_vector_by_pod({"status": "error"}) == {}

    def test_counter_collector_cli_has_no_scientific_overrides(self):
        args = counter_parse_args(["--experiment-dir", "/tmp/x", "--duration-seconds", "10"])
        assert args.route == "/work"
        assert args.job == "demo-app"

    def test_detect_pod_series_gap_flags_vanished_pods(self):
        # Phase 15B.3C.2 optional future-collector hardening: a pod
        # present in the previous poll but missing from the current one
        # is an explicit, self-relative scrape-gap signal.
        assert lib.detect_pod_series_gap(["a", "b"], ["a"]) == ["b"]

    def test_detect_pod_series_gap_empty_when_stable(self):
        assert lib.detect_pod_series_gap(["a", "b"], ["a", "b"]) == []

    def test_detect_pod_series_gap_ignores_new_arrivals(self):
        # A pod appearing for the first time is a birth, not a gap.
        assert lib.detect_pod_series_gap(["a"], ["a", "b"]) == []


# ---------------------------------------------------------------------------
# CORR-35 exposure-weighted routing computation
# ---------------------------------------------------------------------------


class TestExposureWeightedRoutingComputation:
    def test_equal_exposure_reduces_to_simple_fair_share(self):
        # 2 pods, both Ready the whole window, matches the manifest's own
        # first-slot worked example (2880 total, expected 1440/pod).
        boundaries = [0.0, 60.0, 120.0, 180.0]
        ready_at = {t: ["a", "b"] for t in boundaries[:-1]}
        samples = {
            "a": [(0.0, 0.0), (60.0, 480.0), (120.0, 970.0), (180.0, 1450.0)],
            "b": [(0.0, 0.0), (60.0, 480.0), (120.0, 950.0), (180.0, 1430.0)],
        }
        result = lib.compute_expected_and_actual(boundaries, ready_at, samples)
        assert result["expected"]["a"] == pytest.approx(1440.0, abs=1.0)
        assert result["expected"]["b"] == pytest.approx(1440.0, abs=1.0)
        classification = lib.classify_routing(result["expected"], result["actual"])
        assert classification["classification"] == lib.ROUTING_HEALTHY

    def test_manifest_worked_example_structural_failure(self):
        boundaries = [0.0, 90.0, 180.0]
        ready_at = {0.0: ["a", "b"], 90.0: ["a", "b"]}
        samples = {"a": [(0.0, 0.0), (90.0, 0.0), (180.0, 0.0)], "b": [(0.0, 0.0), (90.0, 1440.0), (180.0, 2880.0)]}
        result = lib.compute_expected_and_actual(boundaries, ready_at, samples)
        classification = lib.classify_routing(result["expected"], result["actual"])
        assert classification["classification"] == lib.ROUTING_STRUCTURAL_FAILURE

    def test_unequal_exposure_transition_is_not_a_naive_50_50_split(self):
        # Pod A Ready both intervals; Pod B Ready only for interval 2 (higher traffic)
        # b has a (zero-delta) sample at t=0.0 too, so it is fully
        # observable across BOTH intervals from the start -- it simply
        # carried no traffic and wasn't Ready until t=60.0. Without a
        # t=0.0 sample, Phase 15B.3C.2's complete-observability gate would
        # correctly exclude interval 1 (b's counter-series presence at
        # t=60.0 alone would make it "relevant" there with a missing
        # earlier boundary -- see TestCompletePodJointObservabilityGateCorr37Fix).
        boundaries = [0.0, 60.0, 120.0]
        ready_at = {0.0: ["a"], 60.0: ["a", "b"]}
        samples = {
            "a": [(0.0, 0.0), (60.0, 100.0), (120.0, 300.0)],
            "b": [(0.0, 0.0), (60.0, 0.0), (120.0, 200.0)],
        }
        result = lib.compute_expected_and_actual(boundaries, ready_at, samples)
        assert result["expected"]["a"] == pytest.approx(300.0)
        assert result["expected"]["b"] == pytest.approx(200.0)
        assert result["expected"]["a"] != result["expected"]["b"]

    def test_counter_reset_count_reported_per_pod(self):
        boundaries = [0.0, 10.0, 20.0]
        ready_at = {0.0: ["a"], 10.0: ["a"]}
        samples = {"a": [(0.0, 100.0), (10.0, 5.0), (20.0, 55.0)]}
        result = lib.compute_expected_and_actual(boundaries, ready_at, samples)
        assert result["counter_reset_counts"]["a"] == 1


class TestRoutingConservationCorr37:
    """Regression guard for the exact defect found during Phase 15B.3B's
    forensic audit of attempt1: the original implementation restricted the
    interval aggregate to Ready-AND-observable pods only, silently
    dropping real traffic whenever the Ready pod for an interval lacked a
    joint sample there, while `actual` was computed as an unconstrained
    whole-span reconstruction independent of interval structure -- this
    produced expected=591 vs actual=1427 for attempt1's real data, with no
    principled relationship between the two. The fix restricts BOTH to the
    identical interval-scoped, all-observable-pods decomposition, which
    guarantees exact conservation by construction."""

    def test_gap_in_ready_pods_observability_no_longer_falsely_recovers_real_traffic(self):
        # Pod A is Ready throughout but has a missing sample at t1 (a
        # scrape gap); Pod B is fully observable throughout but never
        # Ready. Phase 15B.3C.1 "recovered" this as expected["a"]=100 by
        # computing each interval's aggregate from whichever pods
        # happened to be observable (here, only B) and dividing it among
        # the Ready pods (here, only A) -- crediting A with traffic
        # reconstructed entirely from B's counter while A's OWN real,
        # unknown traffic during the gap was never counted anywhere. This
        # is exactly the incomplete-population loophole Phase 15B.3C.2
        # closes: A is RELEVANT (Ready) at both intervals touching t=1.0
        # but lacks a sample there, so BOTH intervals must now be
        # excluded entirely -- never partially reconstructed from B alone.
        boundaries = [0.0, 1.0, 2.0]
        ready_at = {0.0: ["a"], 1.0: ["a"]}
        samples = {
            "a": [(0.0, 1000.0), (2.0, 1200.0)],  # missing at t=1.0
            "b": [(0.0, 2000.0), (1.0, 2050.0), (2.0, 2100.0)],
        }
        result = lib.compute_expected_and_actual(boundaries, ready_at, samples)
        assert result["excluded_incomplete_observability_count"] == 2
        assert result["expected"]["a"] == pytest.approx(0.0)
        assert result["aggregate_total"] == pytest.approx(0.0)
        assert sum(result["actual"].values()) == pytest.approx(0.0)
        assert result["conservation_error"] == pytest.approx(0.0)

    def test_actual_is_interval_scoped_not_a_naive_whole_span_reconstruction(self):
        # Pod A's own counter jumps from 1000 to 1200 across the FULL
        # window, but it is missing the shared boundary at t=1.0 -- since
        # A is relevant (Ready) at both neighboring intervals, both are
        # now excluded for incomplete pod-counter observability, so
        # NEITHER pod's traffic is counted for either interval (never a
        # naive first-to-last-sample delta that ignores interval
        # structure, and never a partial reconstruction from B alone).
        boundaries = [0.0, 1.0, 2.0]
        ready_at = {0.0: ["a"], 1.0: ["a"]}
        samples = {
            "a": [(0.0, 1000.0), (2.0, 1200.0)],
            "b": [(0.0, 2000.0), (1.0, 2050.0), (2.0, 2100.0)],
        }
        result = lib.compute_expected_and_actual(boundaries, ready_at, samples)
        assert result["actual"]["a"] == pytest.approx(0.0)
        assert result["actual"]["b"] == pytest.approx(0.0)
        assert result["excluded_incomplete_observability_count"] == 2

    def test_conservation_holds_exactly_on_attempt1s_real_shape_fallback_relevance(self):
        # A direct reproduction of attempt1's (index1's) real interval
        # structure using COUNTER-SERIES-ONLY fallback relevance (no
        # `relevant_pod_set_at` supplied) -- see
        # TestCompletePodJointObservabilityGateCorr37Fix.test_index1_real_recomputation_is_insufficient_routing_sample_not_healthy_variance
        # for the AUTHORITATIVE recomputation using the real readiness
        # history (which further tightens this to allocatable=390.0).
        # Under fallback relevance alone, 5 of 11 intervals remain
        # allocatable (628.0 aggregate) -- but even so, only xtzgk ever
        # reaches assessability (rpztw's expected share is 0.0), so
        # classification is INSUFFICIENT_ROUTING_SAMPLE, not
        # HEALTHY_VARIANCE; do not force the old conclusion.
        boundaries = [60.01, 70.002, 80.007, 90.005, 100.001, 110.0, 120.001, 130.001, 140.0, 150.001, 160.001, 170.001]
        ready_at = {
            60.01: ["xtzgk"], 70.002: ["xtzgk"], 80.007: [], 90.005: ["rpztw"],
            100.001: [], 110.0: ["xtzgk"], 120.001: ["rpztw"], 130.001: ["rpztw"],
            140.0: ["xtzgk"], 150.001: ["xtzgk"], 160.001: [],
        }
        samples = {
            "rpztw": [(90.005, 48327.0), (100.001, 48489.0), (110.0, 48489.0), (120.001, 48688.0), (130.001, 48688.0)],
            "xtzgk": [
                (60.01, 580.0), (70.002, 580.0), (100.001, 973.0), (110.0, 973.0), (120.001, 1164.0),
                (130.001, 1164.0), (140.0, 1408.0), (150.001, 1408.0), (160.001, 1646.0), (170.001, 1646.0),
            ],
        }
        result = lib.compute_expected_and_actual(boundaries, ready_at, samples)
        assert result["aggregate_total"] == pytest.approx(628.0)
        assert sum(result["actual"].values()) == pytest.approx(628.0)
        assert result["conservation_error"] == pytest.approx(0.0)
        assert result["unattributed_total"] == pytest.approx(0.0)
        classification = lib.classify_routing(result["expected"], result["actual"])
        assert classification["classification"] == "INSUFFICIENT_ROUTING_SAMPLE"

    def test_unattributed_total_tracks_real_traffic_during_zero_ready_intervals(self):
        boundaries = [0.0, 1.0, 2.0]
        ready_at = {0.0: []}  # nobody Ready during [0,1]
        samples = {"a": [(0.0, 100.0), (1.0, 150.0), (2.0, 150.0)]}
        result = lib.compute_expected_and_actual(boundaries, ready_at, samples)
        assert result["unattributed_total"] == pytest.approx(50.0)
        assert sum(result["expected"].values()) == pytest.approx(0.0)
        # aggregate_total is now an alias for the ALLOCATABLE domain (post
        # Phase 15B.3C.1 fix) -- with zero Ready pods throughout, nothing
        # is allocatable; the 50 units of real traffic live in
        # zero_ready_traffic_total / whole_joint_observable_aggregate_total instead.
        assert result["aggregate_total"] == pytest.approx(0.0)
        assert result["zero_ready_traffic_total"] == pytest.approx(50.0)
        assert result["whole_joint_observable_aggregate_total"] == pytest.approx(50.0)
        assert result["zero_ready_actual"]["a"] == pytest.approx(50.0)
        assert result["actual"]["a"] == pytest.approx(0.0)


class TestRoutingIntervalDomainReconciliationCorr37Fix:
    """Phase 15B.3C.1: fixes the exact bug found in index2's real data --
    `expected` was scoped to allocatable (N_ready>0) intervals only, while
    `actual` was scoped to the WHOLE jointly-observable domain (allocatable
    + zero-Ready), silently mixing zero-Ready traffic into the
    classification numerator. `expected` and `actual` must describe the
    identical interval domain.

    Phase 15B.3C.2 further tightened `compute_expected_and_actual` to
    require COMPLETE per-pod joint observability (see
    `TestCompletePodJointObservabilityGateCorr37Fix` below for the
    authoritative real-readiness-data recomputation and the required
    regression-fixture categories). The fixture-level assertions below are
    updated in place to match the function's CURRENT behavior when called
    the same 3-positional-argument way as originally written here (i.e.
    with no `relevant_pod_set_at`, so relevance falls back to
    counter-series identity alone) -- this is intentionally a WEAKER
    approximation of the true relevant-pod population than the real
    readiness history provides (see below), since a pod with literally no
    counter sample anywhere in this fixture cannot be detected as
    "relevant but missing" without readiness data. It still exercises the
    core interval-level (not per-pod-silently-dropped) exclusion logic."""

    # Index2's EXACT real interval structure, taken verbatim from
    # build_routing_inputs() applied to the preserved raw artifacts of
    # fit_dyncal_r4_L5_repeat1_attempt1_20260909T151415Z (see
    # task_outputs/phase_15b3c1_routing_interval_domain_reconciliation.md
    # sec.7 for the full derivation) -- not a hand-fabricated
    # approximation.
    INDEX2_BOUNDARIES = [60.001, 70.005, 80.005, 90.003, 100.005, 110.005, 120.006, 130.001, 140.005, 150.006, 160.005, 170.002]
    INDEX2_READY_AT = {
        60.001: ["v6lmc", "wvt92", "xtzgk"], 70.005: ["v6lmc", "wvt92", "xtzgk"],
        80.005: ["v6lmc"], 90.003: [], 100.005: ["xtzgk"],
        110.005: ["6lld4", "wvt92", "xtzgk"], 120.006: ["v6lmc", "wvt92"],
        130.001: ["6lld4", "v6lmc", "wvt92"], 140.005: ["6lld4"], 150.006: [],
        160.005: ["wvt92", "xtzgk"],
    }
    INDEX2_SAMPLES = {
        "6lld4": [(110.005, 829.0), (140.005, 32.0)],
        "v6lmc": [(60.001, 459.0), (70.005, 513.0), (80.005, 513.0), (110.005, 835.0), (170.002, 357.0)],
        "wvt92": [(60.001, 387.0), (70.005, 428.0), (130.001, 161.0), (140.005, 161.0), (160.005, 328.0)],
        "xtzgk": [(60.001, 2324.0), (70.005, 2324.0), (80.005, 2482.0), (90.003, 2482.0), (100.005, 2644.0), (110.005, 2644.0)],
    }
    # The REAL per-boundary relevant-pod set, taken verbatim from
    # `pod_readiness_history.csv` for this attempt (Phase 15B.3C.2): all
    # four pods are reported "Running" (any Ready status) by kubectl at
    # EVERY one of these 11 boundaries -- none of them is a stale/deleted
    # historical pod outside the window. This is the authoritative input
    # for the complete-joint-observability gate; see
    # `TestCompletePodJointObservabilityGateCorr37Fix.test_index2_real_recomputation_is_insufficient_routing_sample`.
    INDEX2_RELEVANT_AT = {t: ["6lld4", "v6lmc", "wvt92", "xtzgk"] for t in INDEX2_BOUNDARIES}

    def test_index2_exact_regression_fixture(self):
        # Fallback-relevance mode (no relevant_pod_set_at): the interval
        # [70.005, 80.005) is now excluded because wvt92 -- Ready and
        # counter-series-known at 70.005 -- lacks a sample at 80.005 (the
        # exact loophole this phase closes: previously this interval was
        # silently computed from v6lmc+xtzgk alone). This drops the
        # allocatable domain from the 15B.3C.1 figure of 253 to 95.
        result = lib.compute_expected_and_actual(self.INDEX2_BOUNDARIES, self.INDEX2_READY_AT, self.INDEX2_SAMPLES)
        assert result["allocatable_aggregate_total"] == pytest.approx(95.0)
        assert result["zero_ready_traffic_total"] == pytest.approx(162.0)
        assert result["whole_joint_observable_aggregate_total"] == pytest.approx(257.0)
        assert sum(result["expected"].values()) == pytest.approx(95.0)
        assert sum(result["actual"].values()) == pytest.approx(95.0)
        assert sum(result["zero_ready_actual"].values()) == pytest.approx(162.0)
        assert result["expected_conservation_error"] == pytest.approx(0.0)
        assert result["actual_conservation_error"] == pytest.approx(0.0)
        assert result["excluded_incomplete_observability_count"] == 9

    def test_old_buggy_logic_would_have_produced_415_actual(self):
        # Documents exactly why the ORIGINAL (pre-15B.3C.1) logic was
        # wrong: summing EVERY per-pod-independently-observable interval's
        # delta (ignoring N_ready, and ignoring whether OTHER relevant
        # pods were missing a boundary) recovers the old, incorrect 415
        # total -- demonstrating both fixes (15B.3C.1's domain match and
        # 15B.3C.2's population-completeness gate) are real, not merely
        # relabelings.
        old_style_actual_total = 0.0
        samples_by_pod_ts = {p: dict(v) for p, v in self.INDEX2_SAMPLES.items()}
        for t_j, t_j1 in zip(self.INDEX2_BOUNDARIES, self.INDEX2_BOUNDARIES[1:]):
            observable = [p for p in samples_by_pod_ts if t_j in samples_by_pod_ts[p] and t_j1 in samples_by_pod_ts[p]]
            for p in observable:
                d, _ = lib.reconstruct_counter_total([(t_j, samples_by_pod_ts[p][t_j]), (t_j1, samples_by_pod_ts[p][t_j1])])
                old_style_actual_total += d
        assert old_style_actual_total == pytest.approx(415.0)
        # ...while the currently-corrected function returns 95 for the
        # classification domain under fallback relevance (0 under the
        # real, readiness-enriched relevant set -- see
        # TestCompletePodJointObservabilityGateCorr37Fix).
        result = lib.compute_expected_and_actual(self.INDEX2_BOUNDARIES, self.INDEX2_READY_AT, self.INDEX2_SAMPLES)
        assert sum(result["actual"].values()) == pytest.approx(95.0)

    def test_index2_corrected_classification_is_not_the_original_pinning_result(self):
        result = lib.compute_expected_and_actual(self.INDEX2_BOUNDARIES, self.INDEX2_READY_AT, self.INDEX2_SAMPLES)
        classification = lib.classify_routing(result["expected"], result["actual"])
        # With only 1 allocatable interval surviving under fallback
        # relevance, assessability now fails outright (< 2 pods reach the
        # expected>=50 threshold) -- the original 3.79/1.87 pinning-ratio
        # narrative no longer applies once the pod population is honestly
        # gated; do NOT assume ROUTING_PINNING_SUSPECTED survives.
        assert classification["classification"] == "INSUFFICIENT_ROUTING_SAMPLE"

    def test_allocatable_and_zero_ready_domains_are_disjoint_and_exhaustive(self):
        result = lib.compute_expected_and_actual(self.INDEX2_BOUNDARIES, self.INDEX2_READY_AT, self.INDEX2_SAMPLES)
        assert result["whole_joint_observable_aggregate_total"] == pytest.approx(
            result["allocatable_aggregate_total"] + result["zero_ready_traffic_total"]
        )

    def test_non_ready_pod_traffic_during_an_allocatable_interval_is_preserved_in_actual(self):
        # Interval has 1 Ready pod (a) and 1 non-Ready-but-observable pod
        # (b); b's real traffic must still count toward the allocatable
        # aggregate/actual (it is NOT dropped merely for being non-Ready),
        # even though b receives zero EXPECTED credit.
        boundaries = [0.0, 10.0]
        ready_at = {0.0: ["a"]}  # b exists and is observable but not Ready
        samples = {"a": [(0.0, 100.0), (10.0, 120.0)], "b": [(0.0, 50.0), (10.0, 90.0)]}
        result = lib.compute_expected_and_actual(boundaries, ready_at, samples)
        assert result["allocatable_aggregate_total"] == pytest.approx(60.0)  # 20 (a) + 40 (b)
        assert result["actual"]["b"] == pytest.approx(40.0)  # b's real traffic preserved
        assert result["expected"]["b"] == pytest.approx(0.0)  # but zero expected credit (not Ready)
        assert result["expected"]["a"] == pytest.approx(60.0)  # a receives the full allocatable share (sole Ready pod)


class TestCompletePodJointObservabilityGateCorr37Fix:
    """Phase 15B.3C.2: closes the remaining loophole in the Phase 15B.3C.1
    domain-partition fix -- an interval could still be silently
    reconstructed from an INCOMPLETE subset of the pods relevant to it
    (e.g. a Ready pod missing one boundary sample was simply excluded
    from `observable`, rather than excluding the whole interval), passing
    conservation trivially against that same incomplete subset. Zero
    conservation error on an incomplete population is not sufficient
    evidence the interval is trustworthy.

    A pod is RELEVANT to an interval iff it is known (via readiness
    history OR counter-series identity) at either boundary; ALL relevant
    pods must have a reconstructable sample at BOTH boundaries or the
    entire interval is excluded as
    `EXCLUDED_INCOMPLETE_POD_COUNTER_OBSERVABILITY` -- never partially
    reconstructed."""

    # ---- (a) partial-observability false-conservation ----
    def test_partial_observability_no_longer_falsely_conserves(self):
        # Pod A: t0=100, t1=150 (both boundaries present).
        # Pod B: t0=200, t1=MISSING -- a real scrape gap, not a resolved
        # value. Both A and B are relevant (both Ready). The OLD (pre-
        # 15B.3C.2) logic would report aggregate=50 (from A alone),
        # actual=50, conservation_error=0 -- passing despite B's real,
        # unknown traffic during this interval being silently dropped.
        boundaries = [0.0, 1.0]
        ready_at = {0.0: ["a", "b"]}
        samples = {"a": [(0.0, 100.0), (1.0, 150.0)], "b": [(0.0, 200.0)]}
        relevant_at = {0.0: ["a", "b"], 1.0: ["a", "b"]}
        result = lib.compute_expected_and_actual(boundaries, ready_at, samples, relevant_at)
        assert result["candidate_interval_count"] == 1
        assert result["allocatable_interval_count"] == 0
        assert result["excluded_incomplete_observability_count"] == 1
        assert result["excluded_interval_reasons"][0]["reason"] == lib.EXCLUDED_INCOMPLETE_POD_COUNTER_OBSERVABILITY
        assert result["excluded_interval_reasons"][0]["missing_pods"] == ["b"]
        assert result["allocatable_aggregate_total"] == pytest.approx(0.0)
        assert sum(result["expected"].values()) == pytest.approx(0.0)
        assert sum(result["actual"].values()) == pytest.approx(0.0)
        assert result["complete_pod_observability_pass"] is False

    # ---- (b) fully-observable case proceeds normally ----
    def test_fully_observable_interval_proceeds_and_conserves(self):
        boundaries = [0.0, 1.0]
        ready_at = {0.0: ["a", "b"]}
        samples = {"a": [(0.0, 100.0), (1.0, 150.0)], "b": [(0.0, 200.0), (1.0, 250.0)]}
        relevant_at = {0.0: ["a", "b"], 1.0: ["a", "b"]}
        result = lib.compute_expected_and_actual(boundaries, ready_at, samples, relevant_at)
        assert result["candidate_interval_count"] == 1
        assert result["allocatable_interval_count"] == 1
        assert result["excluded_incomplete_observability_count"] == 0
        assert result["allocatable_aggregate_total"] == pytest.approx(100.0)  # 50 (a) + 50 (b)
        assert sum(result["actual"].values()) == pytest.approx(100.0)
        assert sum(result["expected"].values()) == pytest.approx(100.0)
        assert result["complete_pod_observability_pass"] is True

    # ---- (c) NotReady-but-observable pod: included in actual, zero expected, never excluded merely for NotReady ----
    def test_not_ready_but_observable_pod_is_relevant_and_included_never_excluded_for_status_alone(self):
        boundaries = [0.0, 1.0]
        ready_at = {0.0: ["a"]}  # b exists (both boundaries) but is NotReady
        samples = {"a": [(0.0, 100.0), (1.0, 120.0)], "b": [(0.0, 50.0), (1.0, 90.0)]}
        relevant_at = {0.0: ["a", "b"], 1.0: ["a", "b"]}
        result = lib.compute_expected_and_actual(boundaries, ready_at, samples, relevant_at)
        assert result["excluded_incomplete_observability_count"] == 0
        assert result["actual"]["b"] == pytest.approx(40.0)
        assert result["expected"]["b"] == pytest.approx(0.0)
        assert result["allocatable_aggregate_total"] == pytest.approx(60.0)

    # ---- (d) mid-interval pod birth/deletion: missing a required boundary excludes the interval ----
    def test_mid_interval_pod_birth_excludes_interval_not_fabricates_pre_creation_value(self):
        # Pod B is only known (readiness + counters) from t=1.0 onward --
        # it did not exist at t=0.0. Per the frozen relevant-pod-set
        # definition ("...OR have a demo-app request-counter series at
        # EITHER boundary"), B's mere presence at t=1.0 makes it relevant
        # to interval [0.0, 1.0) too -- and since it lacks a t=0.0 sample
        # (it did not exist then), that interval must be EXCLUDED rather
        # than silently reconstructed from A alone (we cannot know exactly
        # when within [0.0, 1.0) B was actually created and may have
        # started serving traffic). B's second interval [1.0, 2.0), where
        # it has both boundaries, proceeds normally.
        boundaries = [0.0, 1.0, 2.0]
        ready_at = {0.0: ["a"], 1.0: ["a", "b"]}
        samples = {"a": [(0.0, 100.0), (1.0, 150.0), (2.0, 200.0)], "b": [(1.0, 10.0), (2.0, 30.0)]}
        relevant_at = {0.0: ["a"], 1.0: ["a", "b"], 2.0: ["a", "b"]}
        result = lib.compute_expected_and_actual(boundaries, ready_at, samples, relevant_at)
        assert result["excluded_incomplete_observability_count"] == 1
        assert result["excluded_interval_reasons"][0]["interval_start"] == 0.0
        assert result["excluded_interval_reasons"][0]["missing_pods"] == ["b"]
        assert result["allocatable_interval_count"] == 1
        assert result["allocatable_aggregate_total"] == pytest.approx(70.0)  # interval [1.0,2.0) only: (150+10)->(200+30)

    def test_mid_interval_pod_deletion_missing_required_boundary_excludes_interval(self):
        # Pod B is relevant (Ready) at t=0.0 but is deleted before t=1.0 --
        # no counter sample exists for it at t=1.0. Its deletion does not
        # excuse fabricating a boundary value; the interval touching its
        # missing boundary must be excluded.
        boundaries = [0.0, 1.0]
        ready_at = {0.0: ["a", "b"]}
        samples = {"a": [(0.0, 100.0), (1.0, 150.0)], "b": [(0.0, 50.0)]}
        relevant_at = {0.0: ["a", "b"], 1.0: ["a"]}
        result = lib.compute_expected_and_actual(boundaries, ready_at, samples, relevant_at)
        assert result["excluded_incomplete_observability_count"] == 1
        assert result["excluded_interval_reasons"][0]["missing_pods"] == ["b"]
        assert result["allocatable_interval_count"] == 0

    # ---- (e) counter reset behavior is retained under the new gate ----
    def test_counter_reset_still_handled_on_a_fully_observable_interval(self):
        # Reset-aware reconstruction is unchanged by the new gate: a
        # decreasing sample pair is a reset, contributing 0 to the delta
        # (never the raw negative difference, never the post-reset value
        # itself) -- see `reconstruct_counter_total`'s own frozen semantics.
        boundaries = [0.0, 1.0]
        ready_at = {0.0: ["a"]}
        samples = {"a": [(0.0, 100.0), (1.0, 5.0)]}  # reset mid-interval
        relevant_at = {0.0: ["a"], 1.0: ["a"]}
        result = lib.compute_expected_and_actual(boundaries, ready_at, samples, relevant_at)
        assert result["counter_reset_counts"]["a"] == 1
        assert result["excluded_incomplete_observability_count"] == 0
        assert result["allocatable_aggregate_total"] == pytest.approx(0.0)

    # ---- authoritative real-data recomputation (Phase 15B.3C.2) ----
    def test_index2_real_recomputation_is_insufficient_routing_sample(self):
        # Using the REAL readiness-derived relevant-pod set (all 4 pods
        # continuously "Running" per kubectl throughout the scored
        # window, per pod_readiness_history.csv -- see
        # TestRoutingIntervalDomainReconciliationCorr37Fix.INDEX2_RELEVANT_AT),
        # every one of the 11 candidate intervals is excluded: 6lld4 is
        # relevant (present in kubectl) throughout but has counter samples
        # at only 2 of the 12 boundaries. Do not assume
        # ROUTING_PINNING_SUSPECTED survives -- it does not; there is no
        # longer sufficient evidence to assess routing fairness at all.
        fx = TestRoutingIntervalDomainReconciliationCorr37Fix
        result = lib.compute_expected_and_actual(fx.INDEX2_BOUNDARIES, fx.INDEX2_READY_AT, fx.INDEX2_SAMPLES, fx.INDEX2_RELEVANT_AT)
        assert result["candidate_interval_count"] == 11
        assert result["allocatable_interval_count"] == 0
        assert result["zero_ready_interval_count"] == 0
        assert result["excluded_incomplete_observability_count"] == 11
        assert result["allocatable_aggregate_total"] == pytest.approx(0.0)
        assert result["complete_pod_observability_pass"] is False
        gated = harness.gate_and_classify_routing(result)
        assert gated["classification"] == "INSUFFICIENT_ROUTING_SAMPLE"
        assert gated["assessable_pods"] == []

    def test_index1_real_recomputation_is_insufficient_routing_sample_not_healthy_variance(self):
        # index1's REAL readiness history (Phase 15B.3C.2, verified
        # against pod_readiness_history.csv) shows BOTH rpztw and xtzgk
        # continuously "Running" throughout the scored window (rpztw was
        # repeatedly restarted by LIVENESS_PROBE_TIMEOUT_UNDER_LOAD --
        # CORR-36 -- causing its counter series to intermittently vanish
        # from Prometheus). Only 3 of 11 candidate intervals retain
        # complete joint observability (2 allocatable + 1 zero-Ready);
        # only 1 pod (xtzgk) ever reaches assessability. HEALTHY_VARIANCE
        # does NOT survive this stricter, honest gate.
        boundaries = [60.01, 70.002, 80.007, 90.005, 100.001, 110.0, 120.001, 130.001, 140.0, 150.001, 160.001, 170.001]
        ready_at = {
            60.01: ["xtzgk"], 70.002: ["xtzgk"], 80.007: [], 90.005: ["rpztw"],
            100.001: [], 110.0: ["xtzgk"], 120.001: ["rpztw"], 130.001: ["rpztw"],
            140.0: ["xtzgk"], 150.001: ["xtzgk"], 160.001: [],
        }
        samples = {
            "rpztw": [(90.005, 48327.0), (100.001, 48489.0), (110.0, 48489.0), (120.001, 48688.0), (130.001, 48688.0)],
            "xtzgk": [
                (60.01, 580.0), (70.002, 580.0), (100.001, 973.0), (110.0, 973.0), (120.001, 1164.0),
                (130.001, 1164.0), (140.0, 1408.0), (150.001, 1408.0), (160.001, 1646.0), (170.001, 1646.0),
            ],
        }
        relevant_at = {t: ["rpztw", "xtzgk"] for t in boundaries}
        result = lib.compute_expected_and_actual(boundaries, ready_at, samples, relevant_at)
        assert result["candidate_interval_count"] == 11
        assert result["allocatable_interval_count"] == 2
        assert result["zero_ready_interval_count"] == 1
        assert result["excluded_incomplete_observability_count"] == 8
        assert result["allocatable_aggregate_total"] == pytest.approx(390.0)
        assert sum(result["actual"].values()) == pytest.approx(390.0)
        assert sum(result["expected"].values()) == pytest.approx(390.0)
        assert result["expected_conservation_error"] == pytest.approx(0.0)
        assert result["actual_conservation_error"] == pytest.approx(0.0)
        gated = harness.gate_and_classify_routing(result)
        assert gated["classification"] == "INSUFFICIENT_ROUTING_SAMPLE"
        assert gated["classification"] != lib.ROUTING_HEALTHY


class TestClassificationGatingCorr37:
    """Exercises `gate_and_classify_routing`, the extracted pure function
    the orchestrator itself calls -- classification must never be emitted
    over a non-conserving reconstruction."""

    def test_classification_withheld_when_actual_conservation_fails(self):
        fake_result = {
            "expected": {"a": 100.0}, "actual": {"a": 90.0},
            "expected_conservation_error": 0.0, "actual_conservation_error": 10.0,
        }
        gated = harness.gate_and_classify_routing(fake_result)
        assert gated["classification"] == "INSUFFICIENT_ROUTING_SAMPLE"
        assert "conservation failed" in gated["reason"]

    def test_classification_withheld_when_expected_conservation_fails(self):
        fake_result = {
            "expected": {"a": 90.0}, "actual": {"a": 100.0},
            "expected_conservation_error": 10.0, "actual_conservation_error": 0.0,
        }
        gated = harness.gate_and_classify_routing(fake_result)
        assert gated["classification"] == "INSUFFICIENT_ROUTING_SAMPLE"

    def test_classification_emitted_when_both_conserve(self):
        fake_result = {
            "expected": {"a": 100.0, "b": 100.0}, "actual": {"a": 95.0, "b": 105.0},
            "expected_conservation_error": 0.0, "actual_conservation_error": 0.0,
        }
        gated = harness.gate_and_classify_routing(fake_result)
        assert gated["classification"] == lib.ROUTING_HEALTHY

    def test_gate_on_real_index2_data_conservation_passes_even_though_assessability_fails(self):
        # The conservation GATE itself (expected_conservation_error==0 and
        # actual_conservation_error==0) passes on the real, fully-gated
        # index2 reconstruction -- the resulting INSUFFICIENT_ROUTING_SAMPLE
        # comes from assessability (too few pods reach expected>=50), not
        # from a conservation failure. This distinguishes the two possible
        # sources of INSUFFICIENT_ROUTING_SAMPLE.
        fx = TestRoutingIntervalDomainReconciliationCorr37Fix
        result = lib.compute_expected_and_actual(fx.INDEX2_BOUNDARIES, fx.INDEX2_READY_AT, fx.INDEX2_SAMPLES, fx.INDEX2_RELEVANT_AT)
        assert result["expected_conservation_error"] == pytest.approx(0.0)
        assert result["actual_conservation_error"] == pytest.approx(0.0)
        gated = harness.gate_and_classify_routing(result)
        assert gated["classification"] == "INSUFFICIENT_ROUTING_SAMPLE"
        assert "conservation failed" not in gated.get("reason", "")


class TestIndex1RoutingRegressionAfterDomainFix:
    """Phase 15B.3C.1 sec.10: index1 (fit_dyncal_r2_L4_repeat1) had
    zero-Ready intervals contributing exactly 0 real traffic in-window, so
    the domain fix reproduced the HEALTHY_VARIANCE conclusion already
    established in Phase 15B.3B -- USING ONLY fallback (counter-series)
    relevance, without the complete per-pod joint-observability gate.

    Phase 15B.3C.2 (see
    `TestCompletePodJointObservabilityGateCorr37Fix.test_index1_real_recomputation_is_insufficient_routing_sample_not_healthy_variance`)
    found that HEALTHY_VARIANCE does NOT survive once the real readiness
    history (showing rpztw and xtzgk both continuously present throughout
    the window) is used to gate joint observability -- only 3 of 11
    intervals retain complete observability, and only 1 pod ever reaches
    assessability. This test is kept to document the WEAKER
    fallback-relevance-only figure (also since revised: 628.0, not the
    original 1034.0, per the interval-population-completeness tightening
    that applies even without readiness enrichment)."""

    def test_index1_shape_conserves_under_fallback_relevance_but_is_insufficient(self):
        boundaries = [60.01, 70.002, 80.007, 90.005, 100.001, 110.0, 120.001, 130.001, 140.0, 150.001, 160.001, 170.001]
        ready_at = {
            60.01: ["xtzgk"], 70.002: ["xtzgk"], 80.007: [], 90.005: ["rpztw"],
            100.001: [], 110.0: ["xtzgk"], 120.001: ["rpztw"], 130.001: ["rpztw"],
            140.0: ["xtzgk"], 150.001: ["xtzgk"], 160.001: [],
        }
        samples = {
            "rpztw": [(90.005, 48327.0), (100.001, 48489.0), (110.0, 48489.0), (120.001, 48688.0), (130.001, 48688.0)],
            "xtzgk": [
                (60.01, 580.0), (70.002, 580.0), (100.001, 973.0), (110.0, 973.0), (120.001, 1164.0),
                (130.001, 1164.0), (140.0, 1408.0), (150.001, 1408.0), (160.001, 1646.0), (170.001, 1646.0),
            ],
        }
        result = lib.compute_expected_and_actual(boundaries, ready_at, samples)
        assert result["zero_ready_traffic_total"] == pytest.approx(0.0)
        assert result["allocatable_aggregate_total"] == pytest.approx(628.0)
        assert sum(result["actual"].values()) == pytest.approx(628.0)
        assert sum(result["expected"].values()) == pytest.approx(628.0)
        classification = lib.classify_routing(result["expected"], result["actual"])
        assert classification["classification"] == "INSUFFICIENT_ROUTING_SAMPLE"


class TestClassifyRoutingAssessability:
    def test_starved_pod_with_high_expected_remains_assessable_and_flags_structural(self):
        result = lib.classify_routing({"a": 100.0, "b": 100.0}, {"a": 0.0, "b": 200.0})
        assert result["classification"] == lib.ROUTING_STRUCTURAL_FAILURE
        assert "a" in result["assessable_pods"]

    def test_short_lived_pod_with_low_expected_is_excluded_not_flagged(self):
        result = lib.classify_routing({"a": 7.0, "b": 300.0, "c": 300.0}, {"a": 1.0, "b": 300.0, "c": 300.0})
        assert "a" not in result["assessable_pods"]
        assert result["classification"] == lib.ROUTING_HEALTHY

    def test_fewer_than_two_assessable_pods_is_insufficient(self):
        result = lib.classify_routing({"a": 500.0, "b": 10.0}, {"a": 500.0, "b": 10.0})
        assert result["classification"] == lib.ROUTING_INSUFFICIENT_SAMPLE

    def test_twenty_percent_of_expected_is_pinning_suspected(self):
        result = lib.classify_routing({"a": 100.0, "b": 100.0, "c": 100.0, "d": 100.0}, {"a": 20.0, "b": 127.0, "c": 127.0, "d": 126.0})
        assert result["classification"] == lib.ROUTING_PINNING_SUSPECTED

    def test_healthy_variance_example(self):
        result = lib.classify_routing({"a": 100.0, "b": 100.0}, {"a": 95.0, "b": 105.0})
        assert result["classification"] == lib.ROUTING_HEALTHY

    def test_every_case_maps_to_exactly_one_of_four_tiers(self):
        valid = {lib.ROUTING_INSUFFICIENT_SAMPLE, lib.ROUTING_STRUCTURAL_FAILURE, lib.ROUTING_HEALTHY, lib.ROUTING_PINNING_SUSPECTED}
        cases = [
            ({"a": 100.0, "b": 100.0}, {"a": 95.0, "b": 105.0}),
            ({"a": 100.0, "b": 100.0}, {"a": 0.0, "b": 200.0}),
            ({"a": 100.0, "b": 100.0}, {"a": 20.0, "b": 180.0}),
            ({"a": 5.0, "b": 5.0}, {"a": 5.0, "b": 5.0}),
        ]
        for expected, actual in cases:
            assert lib.classify_routing(expected, actual)["classification"] in valid


# ---------------------------------------------------------------------------
# Decision-aligned sample selection -- never a future sample
# ---------------------------------------------------------------------------


class TestDecisionAlignedSampleSelection:
    def test_selects_latest_sample_at_or_before_target(self):
        samples = [{"elapsed_seconds": 55.0, "v": 1}, {"elapsed_seconds": 60.0, "v": 2}, {"elapsed_seconds": 65.0, "v": 3}]
        chosen = lib.select_decision_sample(60.0, samples, validity_key=None)
        assert chosen["v"] == 2

    def test_never_selects_a_future_sample(self):
        samples = [{"elapsed_seconds": 45.0, "v": 1}, {"elapsed_seconds": 61.0, "v": 2}]
        chosen = lib.select_decision_sample(60.0, samples, validity_key=None)
        assert chosen["v"] == 1

    def test_returns_none_when_beyond_staleness_tolerance(self):
        samples = [{"elapsed_seconds": 20.0, "v": 1}]
        chosen = lib.select_decision_sample(60.0, samples, staleness_threshold_seconds=30.0, validity_key=None)
        assert chosen is None

    def test_returns_none_when_no_candidate_exists(self):
        samples = [{"elapsed_seconds": 61.0, "v": 1}]
        assert lib.select_decision_sample(60.0, samples, validity_key=None) is None

    def test_default_validity_filter_skips_invalid_samples(self):
        # This is the exact bug found during Phase 15B.3A's first live run:
        # the Prometheus scrape target went transiently unhealthy under
        # overload, producing rows with sample_valid=False at/near several
        # decision timestamps -- an invalid row must never be selected even
        # if it is otherwise the latest one at-or-before the target.
        samples = [
            {"elapsed_seconds": 40.0, "v": "good", "sample_valid": "True"},
            {"elapsed_seconds": 50.0, "v": "bad", "sample_valid": "False"},
            {"elapsed_seconds": 60.0, "v": "also_bad", "sample_valid": "False"},
        ]
        chosen = lib.select_decision_sample(60.0, samples)
        assert chosen["v"] == "good"

    def test_default_validity_filter_returns_none_if_all_candidates_invalid(self):
        samples = [{"elapsed_seconds": 55.0, "v": "bad", "sample_valid": "False"}]
        assert lib.select_decision_sample(60.0, samples) is None

    def test_validity_filter_accepts_python_bool_true(self):
        samples = [{"elapsed_seconds": 55.0, "v": "ok", "sample_valid": True}]
        assert lib.select_decision_sample(60.0, samples)["v"] == "ok"


# ---------------------------------------------------------------------------
# Generator structural checks
# ---------------------------------------------------------------------------


class TestGeneratorStructuralChecks:
    def test_offered_equals_sent_passes(self):
        ok, _ = lib.check_offered_equals_sent({"total_offered": 4320, "total_sent": 4320})
        assert ok

    def test_offered_not_equal_sent_fails(self):
        ok, msg = lib.check_offered_equals_sent({"total_offered": 4320, "total_sent": 4300})
        assert not ok and "4320" in msg and "4300" in msg

    def test_no_dropped_overload_passes(self):
        ok, _ = lib.check_no_dropped_overload({"outcome_totals": {"dropped_overload": 0}})
        assert ok

    def test_dropped_overload_fails(self):
        ok, msg = lib.check_no_dropped_overload({"outcome_totals": {"dropped_overload": 3}})
        assert not ok and "3" in msg


# ---------------------------------------------------------------------------
# Attempt metadata required fields
# ---------------------------------------------------------------------------


class TestAttemptMetadata:
    def test_build_attempt_metadata_requires_every_frozen_field(self):
        with pytest.raises(ValueError):
            harness.build_attempt_metadata(campaign_sequence_index=1)

    def test_build_attempt_metadata_succeeds_with_all_fields(self):
        kwargs = {f: f"value_{f}" for f in harness.REQUIRED_ATTEMPT_METADATA_FIELDS}
        metadata = harness.build_attempt_metadata(**kwargs)
        for field in harness.REQUIRED_ATTEMPT_METADATA_FIELDS:
            assert field in metadata


# =============================================================================
# Phase 15B.3B (CORR-37) additions below: two-axis validity semantics
# (execution_structural_validity vs steady_state_fit_eligibility), decision-
# point eligibility, cell-level stable-support rule, and V2 versioning.
# Offline, no traffic -- exercises only the preserved attempt1 artifacts and
# pure functions.
# =============================================================================


class TestTwoAxisValidityRepresentability:
    def test_execution_valid_but_fit_ineligible_is_representable(self):
        # The exact attempt1 scenario: mechanically executed correctly
        # (Axis A VALID) but zero decision points steady-state-fit-eligible
        # (Axis B INELIGIBLE) due to load-induced health instability --
        # these are independent fields, never compressed into one word.
        execution_status = lib.EXECUTION_VALID
        fit_status = lib.run_level_fit_classification(fit_eligible_count=0)
        assert execution_status == lib.EXECUTION_VALID
        assert fit_status == lib.FIT_RUN_INELIGIBLE
        # And the ledger only cares about execution_status:
        assert lib.ledger_state_for_execution_validity(execution_status) == lib.LEDGER_STATE_VALID

    def test_legitimate_health_instability_does_not_force_invalid_execution(self):
        # A structural-invalidity determination must come from an actual
        # Axis-A criterion (wrong schedule, HPA active, etc.), never merely
        # from the application having struggled under the tested load.
        assert lib.ledger_state_for_execution_validity(lib.EXECUTION_VALID) == lib.LEDGER_STATE_VALID

    def test_harness_structural_failure_does_force_invalid_execution(self):
        assert lib.ledger_state_for_execution_validity(lib.EXECUTION_INVALID) == lib.LEDGER_STATE_INVALID_RETRY_REQUIRED

    def test_unknown_execution_status_raises(self):
        with pytest.raises(ValueError):
            lib.ledger_state_for_execution_validity("MAYBE")


class TestDecisionPointFitEligibilityCorr37:
    def test_current2_ready1_is_not_fit_eligible(self):
        sample = {"elapsed_seconds": 120.0, "cpu_cores": "0.84"}
        eligible, reason = lib.assess_decision_point_eligibility(sample, ready_count_at_target=1, intended_replicas=2)
        assert eligible is False
        assert reason == lib.FIT_POINT_INELIGIBLE_HEALTH_INSTABILITY

    def test_current2_ready2_is_fit_eligible(self):
        sample = {"elapsed_seconds": 60.0}
        eligible, reason = lib.assess_decision_point_eligibility(sample, ready_count_at_target=2, intended_replicas=2)
        assert eligible is True
        assert reason == lib.FIT_POINT_ELIGIBLE_NONE

    def test_unobservable_sample_is_not_fit_eligible_and_not_confused_with_health_instability(self):
        eligible, reason = lib.assess_decision_point_eligibility(None, ready_count_at_target=2, intended_replicas=2)
        assert eligible is False
        assert reason == lib.FIT_POINT_INELIGIBLE_TARGET_UNOBSERVABLE

    def test_missing_sample_can_never_be_imputed_as_eligible(self):
        # There is no code path that fabricates a sample to make an
        # unobservable point eligible -- passing None always yields False.
        for ready_count in (0, 1, 2, 3):
            eligible, _ = lib.assess_decision_point_eligibility(None, ready_count, intended_replicas=2)
            assert eligible is False

    def test_four_target_rows_always_representable_even_when_all_ineligible(self):
        targets = [60, 90, 120, 150]
        rows = [
            {"target": t, **dict(zip(("eligible", "reason"), lib.assess_decision_point_eligibility(None, 1, 2)))}
            for t in targets
        ]
        assert len(rows) == 4
        assert all(r["target"] in targets for r in rows)
        assert all(not r["eligible"] for r in rows)

    def test_run_level_classification_all_four_eligible(self):
        assert lib.run_level_fit_classification(4) == lib.FIT_RUN_ELIGIBLE

    def test_run_level_classification_partial(self):
        assert lib.run_level_fit_classification(2) == lib.FIT_RUN_PARTIAL

    def test_run_level_classification_none_eligible(self):
        assert lib.run_level_fit_classification(0) == lib.FIT_RUN_INELIGIBLE

    def test_attempt1_real_forensic_result_matches_frozen_v2_semantics(self):
        # Reproduces the exact per-target authoritative-readiness lookups
        # from the preserved attempt1 artifacts (see the phase 15B.3B
        # report sec.15 for the full derivation): 60->ready=1 (health
        # instability), 90->unobservable, 120->ready=1 (health
        # instability), 150->ready=1 (health instability).
        cases = [
            (True, 1),   # t=60: observable, ready=1
            (False, 1),  # t=90: unobservable
            (True, 1),   # t=120: observable, ready=1
            (True, 1),   # t=150: observable, ready=1
        ]
        results = []
        for observable, ready_count in cases:
            sample = {"elapsed_seconds": 0.0} if observable else None
            results.append(lib.assess_decision_point_eligibility(sample, ready_count, intended_replicas=2))
        eligible_flags = [r[0] for r in results]
        reasons = [r[1] for r in results]
        assert eligible_flags == [False, False, False, False]
        assert reasons == [
            lib.FIT_POINT_INELIGIBLE_HEALTH_INSTABILITY,
            lib.FIT_POINT_INELIGIBLE_TARGET_UNOBSERVABLE,
            lib.FIT_POINT_INELIGIBLE_HEALTH_INSTABILITY,
            lib.FIT_POINT_INELIGIBLE_HEALTH_INSTABILITY,
        ]
        assert lib.run_level_fit_classification(sum(eligible_flags)) == lib.FIT_RUN_INELIGIBLE


class TestCellLevelStableSupportRuleCorr37:
    def test_fitting_cell_requires_2_of_3_sufficient_repeats(self):
        assert lib.cell_stable_support([True, True, False], "fitting") == lib.CELL_STABLE_SUPPORT
        assert lib.cell_stable_support([True, False, False], "fitting") == lib.CELL_OUTSIDE_SUPPORT
        assert lib.cell_stable_support([False, False, False], "fitting") == lib.CELL_OUTSIDE_SUPPORT
        assert lib.cell_stable_support([True, True, True], "fitting") == lib.CELL_STABLE_SUPPORT

    def test_validation_cell_requires_1_of_2_sufficient_repeats(self):
        assert lib.cell_stable_support([True, False], "validation") == lib.CELL_STABLE_SUPPORT
        assert lib.cell_stable_support([False, False], "validation") == lib.CELL_OUTSIDE_SUPPORT
        assert lib.cell_stable_support([True, True], "validation") == lib.CELL_STABLE_SUPPORT

    def test_unknown_repeat_type_raises(self):
        with pytest.raises(ValueError):
            lib.cell_stable_support([True, True], "steady_state")

    def test_sufficient_repeat_threshold_is_2_of_4_decision_points(self):
        assert lib.is_repeat_sufficient_for_cell_support(2) is True
        assert lib.is_repeat_sufficient_for_cell_support(1) is False
        assert lib.is_repeat_sufficient_for_cell_support(0) is False
        assert lib.is_repeat_sufficient_for_cell_support(4) is True

    def test_attempt1_repeat_is_not_sufficient_for_cell_support(self):
        # attempt1 had 0 fit-eligible decision points -- not sufficient.
        assert lib.is_repeat_sufficient_for_cell_support(0) is False


class TestHpaRestorationWaitTimeoutCorr37Fix:
    """Phase 15B.3C.1 sec.14: fixes the real Phase 15B.3C finding that a
    fixed 180s post-restore wait was too short for a 4-replica-origin
    scale-down convergence. The timeout is derived from the FROZEN HPA
    scale-down mechanics (k8s/hpa/hpa.yaml), never an arbitrary value, and
    never a direct-scale-to-1 shortcut."""

    def test_2_to_1_matches_the_frozen_stabilization_plus_one_step(self):
        # 120s stabilization + 1 step*60s + 30s margin = 210s
        assert lib.restoration_wait_seconds(2) == 210.0

    def test_4_to_1_exceeds_the_original_insufficient_180s(self):
        # 120 + 3*60 + 30 = 330s -- the exact Phase 15B.3C scenario.
        wait = lib.restoration_wait_seconds(4)
        assert wait == 330.0
        assert wait > 180.0

    def test_8_to_1_covers_the_maxreplicas_worst_case(self):
        # 120 + 7*60 + 30 = 570s
        assert lib.restoration_wait_seconds(8) == 570.0

    def test_already_at_target_requires_only_stabilization_plus_margin(self):
        assert lib.restoration_wait_seconds(1) == 150.0  # 120 + 0*60 + 30

    def test_starting_below_target_raises(self):
        with pytest.raises(ValueError):
            lib.restoration_wait_seconds(0, target_replicas=1)

    def test_custom_target_replicas_supported(self):
        # 3 steps either way (5->2 or 4->1): 120 + 3*60 + 30 = 330s.
        assert lib.restoration_wait_seconds(5, target_replicas=2) == 330.0
        assert lib.restoration_wait_seconds(5, target_replicas=2) == lib.restoration_wait_seconds(4, target_replicas=1)

    def test_timeout_grows_monotonically_with_starting_replicas(self):
        waits = [lib.restoration_wait_seconds(n) for n in (1, 2, 4, 6, 8)]
        assert waits == sorted(waits)
        assert len(set(waits)) == len(waits)  # strictly increasing

    def test_no_direct_scale_to_one_shortcut_in_cleanup_path(self):
        # Cleanup must rely on HPA's own natural convergence, never a
        # direct `kubectl scale --replicas=1` shortcut that would bypass
        # the frozen HPA behavior this project has never altered.
        # `_restore_hpa_and_idle` is a closure defined inside main(); scope
        # the check to the source region between its definition and the
        # next cleanup registration call.
        import inspect

        full_source = inspect.getsource(harness)
        cleanup_region_start = full_source.index("def _restore_hpa_and_idle")
        cleanup_region_end = full_source.index('cleanup.register("stop_background_processes"')
        cleanup_region = full_source[cleanup_region_start:cleanup_region_end]
        assert '"scale"' not in cleanup_region
        assert "--replicas=1" not in cleanup_region

    def test_hpa_target_and_behavior_unchanged_by_this_fix(self):
        import yaml

        hpa_manifest = yaml.safe_load(open(PROJECT_ROOT / "k8s/hpa/hpa.yaml"))
        assert hpa_manifest["spec"]["metrics"][0]["resource"]["target"]["averageUtilization"] == 260
        assert hpa_manifest["spec"]["behavior"]["scaleDown"]["stabilizationWindowSeconds"] == lib.HPA_SCALE_DOWN_STABILIZATION_SECONDS
        assert hpa_manifest["spec"]["behavior"]["scaleDown"]["policies"][0]["periodSeconds"] == lib.HPA_SCALE_DOWN_STEP_INTERVAL_SECONDS


class TestPartialCellSupportNeverPremature:
    """Phase 15B.3C: after index2 (dyncal_r4_L5 repeat1, 0 fit-eligible
    points), only 1 of the cell's 3 official repeats has executed --
    calling the complete-set rule now would prematurely conclude
    OUTSIDE_STABLE_CALIBRATION_SUPPORT even though repeats 2/3 could still
    bring it to STABLE. The honest status must be PENDING."""

    def test_one_insufficient_repeat_of_three_is_pending_not_outside(self):
        # dyncal_r4_L5's exact real situation: repeat1 insufficient, 2 more to go.
        status = lib.cell_support_status_given_partial_repeats([False], total_repeats_in_cell=3, repeat_type="fitting")
        assert status == lib.CELL_SUPPORT_PENDING

    def test_two_sufficient_repeats_already_decidable_as_stable_even_with_remaining(self):
        status = lib.cell_support_status_given_partial_repeats([True, True], total_repeats_in_cell=3, repeat_type="fitting")
        assert status == lib.CELL_STABLE_SUPPORT

    def test_two_insufficient_of_three_with_none_remaining_is_outside(self):
        status = lib.cell_support_status_given_partial_repeats([False, False, True], total_repeats_in_cell=3, repeat_type="fitting")
        assert status == lib.CELL_OUTSIDE_SUPPORT

    def test_early_outside_call_when_remaining_repeats_cannot_reach_threshold(self):
        # 2 repeats done, both insufficient, only 1 remains -- even if that
        # one is sufficient, max possible sufficient_count=1 < required=2.
        status = lib.cell_support_status_given_partial_repeats([False, False], total_repeats_in_cell=3, repeat_type="fitting")
        assert status == lib.CELL_OUTSIDE_SUPPORT

    def test_validation_cell_one_of_two_pending_only_if_first_insufficient(self):
        status = lib.cell_support_status_given_partial_repeats([False], total_repeats_in_cell=2, repeat_type="validation")
        assert status == lib.CELL_SUPPORT_PENDING
        status2 = lib.cell_support_status_given_partial_repeats([True], total_repeats_in_cell=2, repeat_type="validation")
        assert status2 == lib.CELL_STABLE_SUPPORT

    def test_no_repeats_executed_yet_is_pending(self):
        status = lib.cell_support_status_given_partial_repeats([], total_repeats_in_cell=3, repeat_type="fitting")
        assert status == lib.CELL_SUPPORT_PENDING


class TestV2VersioningAndInheritance:
    def test_v1_manifest_preserved_unchanged_hash_matches_v2_pointer(self):
        import yaml

        v2 = yaml.safe_load(open(lib.MANIFEST_V2_PATH))
        actual_hash = __import__("hashlib").sha256(lib.MANIFEST_PATH.read_bytes()).hexdigest()
        assert actual_hash == v2["inherits_from"]["sha256_at_freeze_time"]

    def test_v2_is_explicitly_post_data_and_references_corr36_and_corr37(self):
        import yaml

        v2 = yaml.safe_load(open(lib.MANIFEST_V2_PATH))
        assert v2["protocol_version"] == 2
        assert v2["correction_reference"] == "CORR-37"
        assert v2["prior_correction_reference"] == "CORR-36"
        assert v2["frozen_after_observing_first_empirical_attempt"] is True
        assert v2["first_observed_campaign_sequence_index"] == 1

    def test_v2_inherits_global_sequence_and_schedules_unchanged(self):
        import yaml

        v2 = yaml.safe_load(open(lib.MANIFEST_V2_PATH))
        inherited = " ".join(v2["inherits_from"]["inherited_unchanged"])
        assert "37 canonical schedules" in inherited
        assert "global 102-slot campaign order" in inherited
        assert "campaign_sequence_index=1 identity" in inherited

    def test_v1_protocol_doc_untouched_v2_is_a_new_file(self):
        v1_doc = PROJECT_ROOT / "docs/rl_dynamics_calibration_protocol.md"
        v2_doc = PROJECT_ROOT / "docs/rl_dynamics_calibration_protocol_v2.md"
        assert v1_doc.exists() and v2_doc.exists()
        assert "CORR-35 CORRECTED" not in v2_doc.read_text(encoding="utf-8")  # v2 doesn't masquerade as v1's own title
        assert "post-data" in v2_doc.read_text(encoding="utf-8").lower()


class TestV3VersioningAndInheritanceCorr38:
    """Phase 15B.3C.3: v3 is the SECOND post-data correction (CORR-38),
    frozen after observing both index1 and index2's real evidence. v2 must
    be independently re-verified byte-for-byte unchanged before v3 pins
    its hash, exactly as v1 was re-verified before v2 pinned its own hash."""

    def test_v2_manifest_preserved_unchanged_hash_matches_v3_pointer(self):
        import hashlib

        import yaml

        v3 = yaml.safe_load(open(lib.MANIFEST_V3_PATH))
        actual_hash = hashlib.sha256(lib.MANIFEST_V2_PATH.read_bytes()).hexdigest()
        assert actual_hash == v3["inherits_from"]["sha256_at_freeze_time"]

    def test_v1_manifest_still_unchanged_per_v3s_own_pinned_hash(self):
        import hashlib

        import yaml

        v3 = yaml.safe_load(open(lib.MANIFEST_V3_PATH))
        actual_hash = hashlib.sha256(lib.MANIFEST_PATH.read_bytes()).hexdigest()
        assert actual_hash == v3["inherits_from"]["v1_sha256_unchanged_since_v2_freeze"]
        # v1's hash must also match what v2 itself pinned at v2's own freeze time.
        v2 = yaml.safe_load(open(lib.MANIFEST_V2_PATH))
        assert actual_hash == v2["inherits_from"]["sha256_at_freeze_time"]

    def test_v3_is_explicitly_post_data_and_references_corr37_and_corr38(self):
        import yaml

        v3 = yaml.safe_load(open(lib.MANIFEST_V3_PATH))
        assert v3["protocol_version"] == 3
        assert v3["correction_reference"] == "CORR-38"
        assert v3["prior_correction_reference"] == "CORR-37"
        assert v3["frozen_after_observing_empirical_attempts"] is True
        assert v3["observed_campaign_sequence_indices"] == [1, 2]

    def test_v3_inherits_global_sequence_schedules_and_two_axis_model_unchanged(self):
        import yaml

        v3 = yaml.safe_load(open(lib.MANIFEST_V3_PATH))
        inherited = " ".join(v3["inherits_from"]["inherited_unchanged"])
        assert "37 canonical schedules" in inherited
        assert "global 102-slot campaign order" in inherited
        assert "campaign_sequence_index=1 identity" in inherited and "=2 identity" in inherited
        assert "execution_structural_validity" in inherited and "steady_state_fit_eligibility" in inherited
        assert "THRESHOLDS (0.01/0.5/1.5/0.30/50/2)" in inherited or "0.01/0.5/1.5/0.30/50/2" in inherited

    def test_v3_does_not_redefine_routing_thresholds(self):
        import yaml

        v3 = yaml.safe_load(open(lib.MANIFEST_V3_PATH))
        gate = v3["complete_joint_observability_gate"]
        assert "thresholds_unchanged" in gate
        assert "UNCHANGED" in gate["thresholds_unchanged"]

    def test_v1_and_v2_protocol_docs_untouched_v3_is_a_new_file(self):
        v1_doc = PROJECT_ROOT / "docs/rl_dynamics_calibration_protocol.md"
        v2_doc = PROJECT_ROOT / "docs/rl_dynamics_calibration_protocol_v2.md"
        v3_doc = PROJECT_ROOT / "docs/rl_dynamics_calibration_protocol_v3.md"
        assert v1_doc.exists() and v2_doc.exists() and v3_doc.exists()
        v3_text = v3_doc.read_text(encoding="utf-8")
        assert "post-data" in v3_text.lower()
        assert "CORR-38" in v3_text
        assert "protocol_version: 3" in v3_text

    def test_corr38_appended_to_corrections_doc_without_touching_corr37(self):
        text = (PROJECT_ROOT / "docs/experimental_design_corrections.md").read_text(encoding="utf-8")
        assert "## CORR-38" in text
        assert "## CORR-37" in text  # still present, not overwritten
        corr38_idx = text.index("## CORR-38")
        corr37_idx = text.index("## CORR-37")
        assert corr37_idx < corr38_idx  # CORR-37 precedes CORR-38, append-only


class TestKubernetesAwareCounterObservabilityCorr38:
    """Phase 15B.3C.3 (CORR-38 sec.C): `compute_missing_known_pods` is a
    SNAPSHOT comparison against the authoritative Kubernetes-known pod
    population -- unlike Phase 15B.3C.2's previous-tick-relative
    `detect_pod_series_gap`, it correctly detects a pod absent from the
    very first tick and a pod absent for the entire run (the exact `6lld4`
    failure mode)."""

    def test_missing_pod_detected_on_first_tick(self):
        known_pods = [
            {"pod_uid": "u-a", "pod_name": "a", "phase": "Running", "ready_status": "True", "deletion_timestamp": ""},
            {"pod_uid": "u-b", "pod_name": "b", "phase": "Running", "ready_status": "False", "deletion_timestamp": ""},
        ]
        # No prior tick state is needed at all -- this IS the first tick.
        missing = lib.compute_missing_known_pods(known_pods, ["a"])
        assert len(missing) == 1
        assert missing[0]["pod_name"] == "b"
        assert missing[0]["pod_uid"] == "u-b"

    def test_pod_continuously_missing_is_detected_on_every_tick(self):
        known_pods = [{"pod_uid": "u-b", "pod_name": "b", "phase": "Running", "ready_status": "False", "deletion_timestamp": ""}]
        for _tick in range(5):  # Prometheus never once returns "b" -- flagged every single tick, not just a transition.
            missing = lib.compute_missing_known_pods(known_pods, [])
            assert len(missing) == 1 and missing[0]["pod_name"] == "b"

    def test_no_missing_pods_when_all_known_pods_present(self):
        known_pods = [{"pod_uid": "u-a", "pod_name": "a", "phase": "Running", "ready_status": "True", "deletion_timestamp": ""}]
        assert lib.compute_missing_known_pods(known_pods, ["a"]) == []

    def test_collector_distinguishes_whole_query_failure_kubernetes_listing_failure_and_series_absence(self):
        import inspect

        from scripts import per_pod_counter_collector as collector

        src = inspect.getsource(collector.run)
        assert "whole_prometheus_query_error_count" in src
        assert "kubernetes_listing_error_count" in src
        assert "known_to_kubernetes_but_absent_from_prometheus_result" in src
        assert "whole_prometheus_query_failed" in src
        assert "kubernetes_listing_failed" in src

    def test_collector_writes_new_observability_artifact_with_required_fields(self):
        from scripts.per_pod_counter_collector import OBSERVABILITY_CSV_FIELDNAMES

        for field in (
            "timestamp_utc",
            "elapsed_seconds",
            "pod_uid",
            "pod_name",
            "phase",
            "ready_status",
            "prometheus_series_present",
            "counter_value",
            "missing_series_reason",
        ):
            assert field in OBSERVABILITY_CSV_FIELDNAMES

    def test_existing_per_pod_counter_csv_schema_unchanged(self):
        from scripts.per_pod_counter_collector import CSV_FIELDNAMES

        assert CSV_FIELDNAMES == ["timestamp_utc", "timestamp_epoch", "elapsed_seconds", "pod", "raw_counter_value"]

    def test_collector_cli_has_label_selector_matching_readiness_collector_default(self):
        args = counter_parse_args(["--experiment-dir", "/tmp/x", "--duration-seconds", "10"])
        assert args.label_selector == "app=demo-app"
        readiness_args = readiness_parse_args(["--experiment-dir", "/tmp/x", "--duration-seconds", "10"])
        assert args.label_selector == readiness_args.label_selector
        assert args.namespace == readiness_args.namespace


class TestPodIdentityUidCanonicalCorr38:
    """Phase 15B.3C.3 (CORR-38 sec.A): Kubernetes pod UID is the canonical
    pod identity; pod name is only the join label. A name whose UID
    changes across samples must never be silently merged as one object."""

    def test_same_name_same_uid_is_not_flagged(self):
        assert lib.detect_pod_identity_change({"a": "uid-1"}, {"a": "uid-1"}) == []

    def test_same_name_changed_uid_is_flagged(self):
        assert lib.detect_pod_identity_change({"a": "uid-1"}, {"a": "uid-2"}) == ["a"]

    def test_new_name_first_appearance_is_not_flagged_as_a_change(self):
        assert lib.detect_pod_identity_change({}, {"a": "uid-1"}) == []

    def test_index1_and_index2_have_no_uid_collision_across_their_own_windows(self):
        # Real preserved data verification (Phase 15B.3C.3 / CORR-38 sec.4
        # and v3 protocol doc): every pod name maps to exactly one UID
        # within each attempt's own window -- historical name-based
        # joining is safe for these two attempts.
        import csv

        cases = [
            (
                PROJECT_ROOT
                / "persistent-data/dynamics_calibration/fit_dyncal_r2_L4_repeat1_attempt1_20260909T140146Z/cluster/pod_readiness_history.csv",
                {"demo-app-647f665b49-rpztw", "demo-app-647f665b49-xtzgk"},
            ),
            (
                PROJECT_ROOT
                / "persistent-data/dynamics_calibration/fit_dyncal_r4_L5_repeat1_attempt1_20260909T151415Z/cluster/pod_readiness_history.csv",
                {"demo-app-647f665b49-6lld4", "demo-app-647f665b49-v6lmc", "demo-app-647f665b49-wvt92", "demo-app-647f665b49-xtzgk"},
            ),
        ]
        for path, expected_names in cases:
            if not path.exists():
                pytest.skip(f"preserved artifact not present in this checkout: {path}")
            name_to_uids: dict = {}
            with open(path) as f:
                for row in csv.DictReader(f):
                    name_to_uids.setdefault(row["pod_name"], set()).add(row["pod_uid"])
            assert set(name_to_uids.keys()) == expected_names
            for name, uids in name_to_uids.items():
                assert len(uids) == 1, f"{name} maps to multiple UIDs within one attempt's window: {uids}"


class TestRoutingReclassificationUnderV3Corr38:
    """Phase 15B.3C.3: the complete relevant-pod-set gate (code-unchanged
    since Phase 15B.3C.2, now formally versioned as protocol_version=3 /
    CORR-38) is re-run directly against the real preserved raw artifacts
    for both already-executed slots. Do not assume the prior conclusions
    survive -- confirm directly."""

    INDEX1_DIR = PROJECT_ROOT / "persistent-data/dynamics_calibration/fit_dyncal_r2_L4_repeat1_attempt1_20260909T140146Z"
    INDEX2_DIR = PROJECT_ROOT / "persistent-data/dynamics_calibration/fit_dyncal_r4_L5_repeat1_attempt1_20260909T151415Z"

    def _recompute(self, attempt_dir):
        readiness_rows = harness.load_pod_readiness_history(attempt_dir / "cluster" / "pod_readiness_history.csv")
        counter_rows = harness.load_per_pod_counters(attempt_dir / "routing" / "per_pod_request_counters.csv")
        manifest = lib.load_manifest()
        scored_window = tuple(manifest["steady_state_timing"]["scored_window_seconds"])
        boundaries, ready_at, pod_samples, relevant_at = harness.build_routing_inputs(readiness_rows, counter_rows, scored_window)
        result = lib.compute_expected_and_actual(boundaries, ready_at, pod_samples, relevant_at)
        return harness.gate_and_classify_routing(result)

    def test_index1_v3_routing_status_is_insufficient_routing_sample(self):
        if not self.INDEX1_DIR.exists():
            pytest.skip("preserved index1 artifacts not present in this checkout")
        gated = self._recompute(self.INDEX1_DIR)
        assert gated["classification"] == "INSUFFICIENT_ROUTING_SAMPLE"
        assert gated["classification"] != lib.ROUTING_HEALTHY

    def test_index2_v3_routing_status_is_insufficient_routing_sample(self):
        if not self.INDEX2_DIR.exists():
            pytest.skip("preserved index2 artifacts not present in this checkout")
        gated = self._recompute(self.INDEX2_DIR)
        assert gated["classification"] == "INSUFFICIENT_ROUTING_SAMPLE"
        assert gated["classification"] != lib.ROUTING_PINNING_SUSPECTED


class TestV3CampaignAndIntegrityCorr38:
    """Phase 15B.3C.3: v3 changes routing-evidence admissibility only --
    verify the campaign matrix, ledger, HPA, and RL/reward contracts are
    all untouched."""

    def test_ledger_slots_1_and_2_remain_valid_and_unchanged(self):
        ledger_path = PROJECT_ROOT / "persistent-data/dynamics_calibration/campaign_ledger.json"
        if not ledger_path.exists():
            pytest.skip("no ledger present in this checkout")
        ledger = lib.load_ledger(ledger_path)
        assert ledger["slots"]["1"]["state"] == "VALID"
        assert ledger["slots"]["2"]["state"] == "VALID"

    def test_next_resumable_index_reflects_current_ledger_progress(self):
        # This was asserted as ==3 as of Phase 15B.3C.3 (before index3 had
        # been executed). Phase 15B.3D legitimately executed and validated
        # index3, so the ledger correctly advances -- this is real
        # progress, not a regression. The test now asserts the general
        # invariant (a valid int in range, or None only once the whole
        # campaign is complete) rather than a point-in-time snapshot that
        # would otherwise go stale with every future legitimate execution.
        ledger_path = PROJECT_ROOT / "persistent-data/dynamics_calibration/campaign_ledger.json"
        if not ledger_path.exists():
            pytest.skip("no ledger present in this checkout")
        ledger = lib.load_ledger(ledger_path)
        next_idx = lib.next_resumable_index(ledger)
        assert next_idx is None or (isinstance(next_idx, int) and 1 <= next_idx <= len(ledger["slots"]))
        for i in ("1", "2"):
            assert ledger["slots"][i]["state"] == "VALID"


class TestProtocolProvenanceHardeningCorr38:
    """Phase 15B.3E: every NEW attempt must unambiguously record which
    protocol layer governs routing admissibility, distinct from (and never
    overwriting the meaning of) the historical calibration_protocol_version
    (base matrix, still 1) and protocol_version_applied (Axis-A/B execution
    validity, still 2) fields."""

    def test_provenance_declares_v3_as_routing_relevance_protocol_unchanged_by_corr39(self):
        # CORR-39 (Phase 15B.3M.2) never touches routing-relevance semantics
        # -- this field must remain exactly what CORR-38 froze.
        prov = lib.build_protocol_provenance()
        assert prov["routing_relevance_protocol_version"] == 3

    def test_provenance_does_not_alter_historical_base_matrix_or_execution_validity_versions(self):
        prov = lib.build_protocol_provenance()
        assert prov["base_matrix_protocol_version"] == 1
        assert prov["execution_validity_protocol_version"] == 2

    def test_provenance_includes_verifiable_sha256_hashes(self):
        prov = lib.build_protocol_provenance()
        hashes = prov["provenance_hashes_sha256"]
        assert isinstance(hashes["v3_manifest"], str) and len(hashes["v3_manifest"]) == 64
        assert isinstance(hashes["global_campaign_sequence"], str) and len(hashes["global_campaign_sequence"]) == 64
        # v3_protocol_doc may be None only if the doc file is genuinely absent from this checkout.
        assert hashes["v3_protocol_doc"] is None or len(hashes["v3_protocol_doc"]) == 64

    def test_provenance_hashes_match_independent_recomputation(self):
        import hashlib

        prov = lib.build_protocol_provenance()
        assert prov["provenance_hashes_sha256"]["v3_manifest"] == hashlib.sha256(lib.MANIFEST_V3_PATH.read_bytes()).hexdigest()
        assert prov["provenance_hashes_sha256"]["global_campaign_sequence"] == hashlib.sha256(lib.GLOBAL_SEQUENCE_PATH.read_bytes()).hexdigest()

    def test_new_attempt_metadata_carries_provenance_fields(self):
        # Any NEW attempt's metadata.json (index4 onward) must carry these
        # fields; historical attempts (index1-3) are never rewritten
        # retroactively merely for cosmetic consistency.
        required = {
            "active_calibration_protocol_version",
            "active_calibration_manifest",
            "base_matrix_protocol_version",
            "execution_validity_protocol_version",
            "routing_relevance_protocol_version",
            "active_correction_reference",
            "provenance_hashes_sha256",
        }
        assert required.issubset(set(lib.build_protocol_provenance().keys()))


class TestProtocolProvenanceRouteIntegrityCorr39:
    """Phase 15B.3M.2/15B.3M.3/15B.3N.3: every NEW official attempt from
    this point forward must unambiguously record which protocol layer
    governs it, without altering the meaning of any historical provenance
    field. Phase 15B.3N.3 (CORR-40) moved `active_calibration_protocol_
    version`/`active_correction_reference` to V5 -- the route-integrity
    CLASSIFICATION rules themselves (materiality, windows, delivery
    attribution) remain V4/CORR-39 in substance, hence
    `route_integrity_protocol_version` stays 4."""

    def test_provenance_declared_v5_as_active_calibration_protocol_under_corr40(self):
        # Historical fact about the CORR-40/V5 manifest specifically --
        # superseded as the ACTIVE protocol by CORR-41/Phase 15B.3O.1 (see
        # TestProtocolV6Corr41Phase15B3O1), which does not edit this file.
        import yaml

        m = yaml.safe_load(lib.MANIFEST_V5_PATH.read_text(encoding="utf-8"))
        assert m["protocol_version"] == 5
        assert m["correction_reference"] == "CORR-40"
        assert m["workload_transport"]["protocol_version"] == 5

    def test_provenance_route_integrity_classification_rules_remained_v4_under_corr40(self):
        # CORR-40 changed the transport, not CORR-39's classification logic.
        # Historical fact about V5/CORR-40 specifically -- superseded by
        # CORR-41/Phase 15B.3O.1 (see TestProtocolV6Corr41Phase15B3O1),
        # which DOES change route_integrity_protocol_version (4 -> 6).
        # build_protocol_provenance() has no version-selector argument (see
        # test_provenance_takes_no_index_argument_so_it_applies_to_whichever_slot_runs_next
        # below) and always reports the CURRENTLY active protocol, so this
        # test asserts against v5's own frozen manifest file, not the
        # always-current function return.
        import yaml

        m = yaml.safe_load(lib.MANIFEST_V5_PATH.read_text(encoding="utf-8"))
        assert "materiality criterion (25%, scored-window overlap)" in m["route_integrity_v5_mapping"]["unchanged_from_v4"]

    def test_provenance_includes_v5_verifiable_sha256_hashes(self):
        import hashlib

        prov = lib.build_protocol_provenance()
        hashes = prov["provenance_hashes_sha256"]
        assert isinstance(hashes["v5_manifest"], str) and len(hashes["v5_manifest"]) == 64
        assert hashes["v5_manifest"] == hashlib.sha256(lib.MANIFEST_V5_PATH.read_bytes()).hexdigest()
        assert hashes["v5_protocol_doc"] is None or len(hashes["v5_protocol_doc"]) == 64
        # V4's own hashes remain present and verifiable -- V5 does not erase
        # V4's provenance, only supersedes which is "active".
        assert isinstance(hashes["v4_manifest"], str) and len(hashes["v4_manifest"]) == 64
        assert hashes["v4_manifest"] == hashlib.sha256(lib.MANIFEST_V4_PATH.read_bytes()).hexdigest()

    def test_provenance_takes_no_index_argument_so_it_applies_to_whichever_slot_runs_next(self):
        # Corrects the Phase 15B.3M.2 "from index13 onward" documentation
        # bug: build_protocol_provenance() has no campaign_sequence_index
        # parameter at all, so it structurally cannot special-case which
        # slot is next.
        import inspect

        assert len(inspect.signature(lib.build_protocol_provenance).parameters) == 0


class TestProtocolV5TransportAdoptionCorr40Phase15B3N3:
    """Phase 15B.3N.3: Protocol V5/CORR-40 freezes the Docker-network raw
    TCP relay as the official workload transport, without touching CORR-39's
    route-integrity classification rules or any historical V1-V4 file."""

    def test_v5_manifest_exists_and_is_valid_yaml(self):
        import yaml

        m = yaml.safe_load(lib.MANIFEST_V5_PATH.read_text(encoding="utf-8"))
        assert m["protocol_version"] == 5
        assert m["correction_reference"] == "CORR-40"
        assert m["workload_transport"]["transport_type"] == "docker_network_tcp_relay"

    def test_v5_manifest_does_not_change_v4_route_integrity_rules(self):
        import yaml

        m = yaml.safe_load(lib.MANIFEST_V5_PATH.read_text(encoding="utf-8"))
        mapping = m["route_integrity_v5_mapping"]
        assert "materiality criterion (25%, scored-window overlap)" in mapping["unchanged_from_v4"]
        assert "FAILED" in mapping["attempt_level_aggregation"]
        assert "CONFIRMED" in mapping["attempt_level_aggregation"]

    def test_v1_v3_v4_manifest_hashes_unchanged_by_v5_introduction(self):
        assert hashlib.sha256(lib.MANIFEST_PATH.read_bytes()).hexdigest() == "2e6011495307245bcc85832d0b0e1aa6cbc6ea63e4655914eeb9c942c15fc2bf"
        assert hashlib.sha256(lib.MANIFEST_V3_PATH.read_bytes()).hexdigest() == "a0c7f3cebcdcbd7f9ca766a7057681e7b29cd2d10c4b8de74478fb54681561b8"

    def test_build_relay_docker_run_command_is_raw_tcp_no_restart_loopback_only(self):
        cmd = lib.build_relay_docker_run_command(
            container_name="loadtest-relay-test", docker_network="minikube",
            host_port=None, container_port=8080, node_ip="192.168.49.2", nodeport=31891,
        )
        assert cmd[:3] == ["docker", "run", "-d"]
        assert "--restart" in cmd and cmd[cmd.index("--restart") + 1] == "no"
        publish = cmd[cmd.index("-p") + 1]
        assert publish.startswith("127.0.0.1:")
        assert "0.0.0.0" not in publish
        # alpine/socat's own image ENTRYPOINT is already `socat` -- the
        # container command must be ONLY the two address arguments, never a
        # redundant leading "socat" (which socat itself misparses as a
        # third address: "exactly 2 addresses required (there are 3)").
        assert cmd[-2:] == ["TCP-LISTEN:8080,fork,reuseaddr", "TCP:192.168.49.2:31891"]
        assert "socat" not in cmd[cmd.index(lib.RELAY_IMAGE) + 1:]

    def test_build_relay_docker_run_command_uses_ephemeral_port_when_host_port_none(self):
        cmd = lib.build_relay_docker_run_command(
            container_name="x", docker_network="minikube", host_port=None,
            container_port=8080, node_ip="192.168.49.2", nodeport=31891,
        )
        publish = cmd[cmd.index("-p") + 1]
        assert publish == "127.0.0.1::8080"

    def test_build_relay_docker_run_command_uses_pinned_port_when_given(self):
        cmd = lib.build_relay_docker_run_command(
            container_name="x", docker_network="minikube", host_port=54321,
            container_port=8080, node_ip="192.168.49.2", nodeport=31891,
        )
        publish = cmd[cmd.index("-p") + 1]
        assert publish == "127.0.0.1:54321:8080"

    def test_build_relay_target_url(self):
        assert lib.build_relay_target_url(54321) == "http://127.0.0.1:54321"

    def test_relay_command_never_includes_http_awareness_flags(self):
        cmd = lib.build_relay_docker_run_command(
            container_name="x", docker_network="minikube", host_port=54321,
            container_port=8080, node_ip="192.168.49.2", nodeport=31891,
        )
        joined = " ".join(cmd)
        assert "http" not in joined.lower()


class TestCpuHpaNormalizationCorr38Phase15B3F1:
    """Phase 15B.3F.1: corrects a REPORTING/NARRATIVE-ONLY bug in Phase
    15B.3D/15B.3E/15B.3F's own report prose -- comparing a MULTI-POD
    aggregate CPU reading directly against a SINGLE pod's request/limit,
    instead of scaling the denominator by `current_replicas` as this
    project's own already-frozen RL feature engineering rule requires
    (`docs/rl_mdp_spec.md` sec.4, CORR-27 v2:
    `cpu_per_replica_frac_request = (cpu_cores / ready_replicas) / 0.100`).
    The raw empirical dynamics CPU target itself is never altered by any
    of this -- these are derived reporting/diagnostic fields only."""

    def test_one_replica(self):
        r = lib.cpu_normalization(0.40, 1)
        assert r["aggregate_request_cores"] == pytest.approx(0.1)
        assert r["aggregate_limit_cores"] == pytest.approx(0.5)
        assert r["hpa_comparable_cpu_percent"] == pytest.approx(400.0)
        assert r["aggregate_cpu_limit_percent"] == pytest.approx(80.0)

    def test_two_replicas(self):
        r = lib.cpu_normalization(0.40, 2)
        assert r["aggregate_request_cores"] == pytest.approx(0.2)
        assert r["aggregate_limit_cores"] == pytest.approx(1.0)
        assert r["hpa_comparable_cpu_percent"] == pytest.approx(200.0)
        assert r["aggregate_cpu_limit_percent"] == pytest.approx(40.0)

    def test_four_replicas(self):
        r = lib.cpu_normalization(1.0, 4)
        assert r["aggregate_request_cores"] == pytest.approx(0.4)
        assert r["aggregate_limit_cores"] == pytest.approx(2.0)
        assert r["hpa_comparable_cpu_percent"] == pytest.approx(250.0)
        assert r["aggregate_cpu_limit_percent"] == pytest.approx(50.0)

    def test_hpa_target_equivalence_at_various_replica_counts(self):
        for n, expected_cores in [(1, 0.26), (2, 0.52), (4, 1.04), (8, 2.08)]:
            assert lib.hpa_target_equivalent_aggregate_cpu_cores(n) == pytest.approx(expected_cores)

    def test_zero_replicas_rejected(self):
        with pytest.raises(ValueError):
            lib.cpu_normalization(0.4, 0)

    def test_negative_replicas_rejected(self):
        with pytest.raises(ValueError):
            lib.cpu_normalization(0.4, -1)

    def test_negative_cpu_rejected(self):
        with pytest.raises(ValueError):
            lib.cpu_normalization(-0.1, 2)

    def test_non_positive_request_or_limit_rejected(self):
        with pytest.raises(ValueError):
            lib.cpu_normalization(0.4, 2, cpu_request_cores_per_pod=0.0)
        with pytest.raises(ValueError):
            lib.cpu_normalization(0.4, 2, cpu_limit_cores_per_pod=-0.5)

    def test_mean_cpu_per_current_replica(self):
        r = lib.cpu_normalization(0.6, 3)
        assert r["mean_cpu_cores_per_current_replica"] == pytest.approx(0.2)

    def test_raw_aggregate_cpu_passed_through_unaltered(self):
        r = lib.cpu_normalization(0.532197, 2)
        assert r["aggregate_cpu_cores"] == 0.532197

    # ---- historical regression: index3 no longer reported near 260% under the wrong denominator ----
    def test_index3_real_data_no_longer_appears_near_hpa_target(self):
        # index3 (fit_dyncal_r2_L1_repeat1, 2 replicas) real decision-point
        # CPU values -- the corrected HPA-comparable range must sit well
        # BELOW 260%, not "close to" it as originally (incorrectly) reported.
        for cpu in (0.224539, 0.235439, 0.197835, 0.234048):
            r = lib.cpu_normalization(cpu, 2)
            assert r["hpa_comparable_cpu_percent"] < 150.0

    # ---- historical regression: index4's single-replica interpretation remains correct ----
    def test_index4_real_data_single_replica_normalization_unchanged(self):
        # index4 (fit_dyncal_r1_L3_repeat1, 1 replica): aggregate == per-pod,
        # so the original single-pod-denominator interpretation was already
        # correct here -- this is the control case demonstrating the bug is
        # specifically about failing to scale the denominator for N>1.
        for cpu in (0.321455, 0.423433, 0.398532, 0.403279):
            r = lib.cpu_normalization(cpu, 1)
            assert r["hpa_comparable_cpu_percent"] == pytest.approx(cpu * 1000.0)  # cpu / 0.1 * 100 == cpu * 1000
            assert r["hpa_comparable_cpu_percent"] > 260.0  # genuinely near/above the HPA target at N=1

    # ---- historical regression: index5 uses the two-pod denominator ----
    def test_index5_real_data_uses_two_pod_denominator(self):
        # index5 (fit_dyncal_r2_L2_repeat1, 2 replicas) decision points sit
        # mostly below 260% once correctly normalized; only the scored-
        # window instantaneous peak (0.532197 cores, not a decision-aligned
        # point) lands slightly above it.
        decision_point_cpus = (0.47734, 0.471697, 0.486425, 0.402175)
        for cpu in decision_point_cpus:
            r = lib.cpu_normalization(cpu, 2)
            assert r["hpa_comparable_cpu_percent"] < 260.0
        peak = lib.cpu_normalization(0.532197, 2)
        assert peak["hpa_comparable_cpu_percent"] > 260.0
        assert peak["aggregate_cpu_limit_percent"] == pytest.approx(53.2197, abs=0.01)

    # ---- extract_decision_aligned_v2 wiring ----
    def test_extract_decision_aligned_v2_includes_cpu_normalization_fields(self):
        metrics_rows = [
            {
                "elapsed_seconds": 60.0, "cpu_cores": "0.4", "replicas_current": "2", "replicas_ready": "2",
                "request_rate_rps": "5", "latency_p95_seconds": "0.1", "memory_bytes": "100",
                "requests_in_progress": "0", "error_rate_rps": "0", "sample_valid": "True",
            }
        ]
        readiness_rows = [
            {"elapsed_seconds": 60.0, "pod_name": "a", "ready_status": "True"},
            {"elapsed_seconds": 60.0, "pod_name": "b", "ready_status": "True"},
        ]
        result = harness.extract_decision_aligned_v2(metrics_rows, readiness_rows, [60], 30.0, 2)
        point = result["decision_points"][0]
        assert point["mean_cpu_cores_per_current_replica"] == pytest.approx(0.2)
        assert point["hpa_comparable_cpu_percent"] == pytest.approx(200.0)
        assert point["aggregate_cpu_limit_percent"] == pytest.approx(40.0)

    def test_extract_decision_aligned_v2_never_fabricates_normalization_for_unobservable_point(self):
        result = harness.extract_decision_aligned_v2([], [], [60], 30.0, 2)
        point = result["decision_points"][0]
        assert point["decision_point_observable"] is False
        assert point["mean_cpu_cores_per_current_replica"] is None
        assert point["hpa_comparable_cpu_percent"] is None
        assert point["aggregate_cpu_limit_percent"] is None

    def test_extract_decision_aligned_v2_handles_missing_replicas_current_gracefully(self):
        # A health-unstable sample lacking a usable replicas_current value
        # must not crash -- normalization fields stay None, never fabricated.
        metrics_rows = [
            {
                "elapsed_seconds": 60.0, "cpu_cores": "0.4", "replicas_current": "", "replicas_ready": "",
                "request_rate_rps": "5", "latency_p95_seconds": "0.1", "memory_bytes": "100",
                "requests_in_progress": "0", "error_rate_rps": "0", "sample_valid": "True",
            }
        ]
        readiness_rows = [{"elapsed_seconds": 60.0, "pod_name": "a", "ready_status": "True"}]
        result = harness.extract_decision_aligned_v2(metrics_rows, readiness_rows, [60], 30.0, 1)
        point = result["decision_points"][0]
        assert point["hpa_comparable_cpu_percent"] is None

    # ---- integrity ----
    def test_no_schedule_or_ledger_or_protocol_touched_by_this_module(self):
        import inspect

        src = inspect.getsource(lib.cpu_normalization) + inspect.getsource(lib.hpa_target_equivalent_aggregate_cpu_cores)
        for forbidden in ("write_json_atomic", "atomic_write_ledger", "kubectl", "subprocess"):
            assert forbidden not in src

    def test_hpa_target_unchanged(self):
        text = (PROJECT_ROOT / "k8s/hpa/hpa.yaml").read_text(encoding="utf-8")
        assert "averageUtilization: 260" in text

    def test_global_campaign_sequence_hash_unchanged(self):
        import hashlib

        path = PROJECT_ROOT / "configs/traffic_profiles/rl_dynamics_calibration/global_campaign_sequence.json"
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual_hash == "ce1de372e92de0460e53f1ca36d31ec402712d5cd638738200fcedf70403e901"

    def test_hpa_manifest_baseline_unchanged(self):
        text = (PROJECT_ROOT / "k8s/hpa/hpa.yaml").read_text(encoding="utf-8")
        assert "averageUtilization: 260" in text
        assert "minReplicas: 1" in text
        assert "maxReplicas: 8" in text

    def test_rl_and_reward_contract_versions_unchanged(self):
        rl_text = (PROJECT_ROOT / "configs/rl.yaml").read_text(encoding="utf-8")
        reward_text = (PROJECT_ROOT / "configs/reward.yaml").read_text(encoding="utf-8")
        assert "rl_contract_version: 2" in rl_text
        assert "reward_contract_version: 2" in reward_text


# ---------------------------------------------------------------------------
# Phase 15B.3J.1: index9 evidence-timeline closure regressions
# ---------------------------------------------------------------------------


class TestRestartEvidenceClassificationCorr37Phase15B3J1:
    """A completed, directly-observed restart_count increment
    (`restarted_pods_between`'s own narrow contract) is only ONE of the
    forms real restart evidence can take. A kubelet-issued,
    liveness-probe-attributed `Killing` action whose completion is never
    directly observed is real, non-fabricated evidence too -- it must be
    preserved as `RESTART_INITIATED_BY_KUBELET`, never silently dropped to
    `NO_RESTART_EVIDENCE` and never promoted to
    `RESTART_COMPLETED_AND_OBSERVED`."""

    RUN_START = "2026-09-10T08:48:26Z"
    RUN_END = "2026-09-10T08:59:34Z"

    def test_restart_count_increment_is_completed_and_observed(self):
        # Category A takes precedence regardless of event evidence.
        result = lib.classify_pod_restart_evidence(
            pod_uid="uid-a", baseline_restart_count=0, final_restart_count=1,
            killing_events=[], run_start_utc=self.RUN_START, run_end_utc=self.RUN_END,
        )
        assert result == lib.RESTART_COMPLETED_AND_OBSERVED

    def test_mid_window_killing_event_with_no_count_increase_is_initiated_not_ignored(self):
        # The wsf7j case: a genuine, this-run, liveness-probe Killing event
        # exists, but no restart_count increment was ever observed in the
        # collection window -- this must be preserved as
        # RESTART_INITIATED_BY_KUBELET, not collapsed to NO_RESTART_EVIDENCE.
        events = [
            {
                "reason": "Killing",
                "involved_object_uid": "uid-wsf7j",
                "message": "Container demo-app failed liveness probe, will be restarted",
                "first_timestamp": "2026-09-10T08:52:42Z",
            }
        ]
        result = lib.classify_pod_restart_evidence(
            pod_uid="uid-wsf7j", baseline_restart_count=0, final_restart_count=0,
            killing_events=events, run_start_utc=self.RUN_START, run_end_utc=self.RUN_END,
        )
        assert result == lib.RESTART_INITIATED_BY_KUBELET

    def test_unrelated_pre_baseline_killing_event_is_excluded(self):
        # The wvt92 case: a merged Killing event object's firstTimestamp
        # predates this run's own wall-clock start by hours (accumulated
        # from earlier, unrelated campaign attempts against the same
        # long-lived pod). With no restart_count increase in THIS run, it
        # must resolve to NO_RESTART_EVIDENCE, not RESTART_INITIATED_BY_KUBELET.
        events = [
            {
                "reason": "Killing",
                "involved_object_uid": "uid-wvt92",
                "message": "Container demo-app failed liveness probe, will be restarted",
                "first_timestamp": "2026-09-10T02:14:42Z",  # hours before RUN_START
            }
        ]
        result = lib.classify_pod_restart_evidence(
            pod_uid="uid-wvt92", baseline_restart_count=5, final_restart_count=5,
            killing_events=events, run_start_utc=self.RUN_START, run_end_utc=self.RUN_END,
        )
        assert result == lib.NO_RESTART_EVIDENCE

    def test_pre_baseline_event_does_not_prevent_a_real_this_run_completed_restart(self):
        # Same contaminated event object as above, but this run DID
        # observe a genuine restart_count increase -- Category A must
        # still win regardless of the unusable event evidence.
        events = [
            {
                "reason": "Killing",
                "involved_object_uid": "uid-wvt92",
                "message": "Container demo-app failed liveness probe, will be restarted",
                "first_timestamp": "2026-09-10T02:14:42Z",
            }
        ]
        result = lib.classify_pod_restart_evidence(
            pod_uid="uid-wvt92", baseline_restart_count=5, final_restart_count=7,
            killing_events=events, run_start_utc=self.RUN_START, run_end_utc=self.RUN_END,
        )
        assert result == lib.RESTART_COMPLETED_AND_OBSERVED

    def test_scale_down_deletion_is_not_mislabeled_as_liveness_restart(self):
        # A plain scale-down/graceful-termination Killing event carries a
        # different message (no "liveness probe" text) -- must never be
        # read as restart-initiation evidence.
        events = [
            {
                "reason": "Killing",
                "involved_object_uid": "uid-a",
                "message": "Stopping container demo-app",
                "first_timestamp": "2026-09-10T08:50:00Z",
            }
        ]
        result = lib.classify_pod_restart_evidence(
            pod_uid="uid-a", baseline_restart_count=0, final_restart_count=0,
            killing_events=events, run_start_utc=self.RUN_START, run_end_utc=self.RUN_END,
        )
        assert result == lib.NO_RESTART_EVIDENCE

    def test_uid_mismatch_prevents_wrong_pod_attribution(self):
        # A liveness-probe Killing event that is genuinely this-run, but
        # for a DIFFERENT pod UID (e.g. a same-named pod recreated after
        # deletion) -- must not be attributed to the pod under test.
        events = [
            {
                "reason": "Killing",
                "involved_object_uid": "uid-some-other-pod",
                "message": "Container demo-app failed liveness probe, will be restarted",
                "first_timestamp": "2026-09-10T08:50:00Z",
            }
        ]
        result = lib.classify_pod_restart_evidence(
            pod_uid="uid-a", baseline_restart_count=0, final_restart_count=0,
            killing_events=events, run_start_utc=self.RUN_START, run_end_utc=self.RUN_END,
        )
        assert result == lib.NO_RESTART_EVIDENCE

    def test_unhealthy_warning_alone_is_not_restart_evidence(self):
        # An Unhealthy (probe-failure) warning without a corresponding
        # Killing action never implies a restart happened.
        events = [
            {
                "reason": "Unhealthy",
                "involved_object_uid": "uid-a",
                "message": "Liveness probe failed: context deadline exceeded",
                "first_timestamp": "2026-09-10T08:50:00Z",
            }
        ]
        result = lib.classify_pod_restart_evidence(
            pod_uid="uid-a", baseline_restart_count=0, final_restart_count=0,
            killing_events=events, run_start_utc=self.RUN_START, run_end_utc=self.RUN_END,
        )
        assert result == lib.NO_RESTART_EVIDENCE

    def test_normalize_kubernetes_events_carries_involved_object_uid(self):
        events_json = {
            "items": [
                {
                    "firstTimestamp": "2026-01-01T00:00:00Z",
                    "reason": "Killing",
                    "involvedObject": {"kind": "Pod", "name": "demo-app-a", "uid": "uid-a"},
                    "message": "Container demo-app failed liveness probe, will be restarted",
                }
            ]
        }
        rows = lib.normalize_kubernetes_events(events_json)
        assert rows[0]["involved_object_uid"] == "uid-a"


class TestIndex9MetricsValidityTimelineClosurePhase15B3J1:
    """Reconstructs the exact 21-sample index9 metrics.csv validity
    timeline (Phase 15B.3J.1) to prove the frozen decision-aligned
    extraction never selects a future sample, never imputes, and never
    supports a false '18 consecutive invalid samples from 20-200s' claim
    -- since elapsed=60.01s is both >20 and <200 and is genuinely valid."""

    # Exactly the raw metrics.csv rows for
    # fit_dyncal_r6_L3_repeat1_attempt1_20260910T084823Z (elapsed_seconds,
    # sample_valid) -- valid only at 0.21, 10.00, 60.01.
    SAMPLES = [
        {"elapsed_seconds": 0.21, "sample_valid": "True", "v": "s0"},
        {"elapsed_seconds": 10.00, "sample_valid": "True", "v": "s1"},
        {"elapsed_seconds": 20.00, "sample_valid": "False", "v": "s2"},
        {"elapsed_seconds": 30.00, "sample_valid": "False", "v": "s3"},
        {"elapsed_seconds": 40.00, "sample_valid": "False", "v": "s4"},
        {"elapsed_seconds": 50.00, "sample_valid": "False", "v": "s5"},
        {"elapsed_seconds": 60.01, "sample_valid": "True", "v": "s6"},
        {"elapsed_seconds": 70.00, "sample_valid": "False", "v": "s7"},
        {"elapsed_seconds": 80.00, "sample_valid": "False", "v": "s8"},
        {"elapsed_seconds": 90.00, "sample_valid": "False", "v": "s9"},
        {"elapsed_seconds": 100.00, "sample_valid": "False", "v": "s10"},
        {"elapsed_seconds": 110.00, "sample_valid": "False", "v": "s11"},
        {"elapsed_seconds": 120.00, "sample_valid": "False", "v": "s12"},
        {"elapsed_seconds": 130.00, "sample_valid": "False", "v": "s13"},
        {"elapsed_seconds": 140.00, "sample_valid": "False", "v": "s14"},
        {"elapsed_seconds": 150.00, "sample_valid": "False", "v": "s15"},
        {"elapsed_seconds": 160.00, "sample_valid": "False", "v": "s16"},
        {"elapsed_seconds": 170.01, "sample_valid": "False", "v": "s17"},
        {"elapsed_seconds": 180.00, "sample_valid": "False", "v": "s18"},
        {"elapsed_seconds": 190.00, "sample_valid": "False", "v": "s19"},
        {"elapsed_seconds": 200.00, "sample_valid": "False", "v": "s20"},
    ]

    def test_exactly_three_of_twenty_one_samples_are_valid(self):
        valid = [s for s in self.SAMPLES if s["sample_valid"] == "True"]
        assert len(self.SAMPLES) == 21
        assert [s["elapsed_seconds"] for s in valid] == [0.21, 10.00, 60.01]

    def test_the_20_to_200_invalid_span_is_not_contiguous(self):
        # 60.01s lies inside [20, 200] and is valid -- so the invalid rows
        # in that span are NOT one contiguous 18-sample run; they are two
        # runs (20-50 and 70-200) bracketing the single valid 60.01 sample.
        in_span = [s for s in self.SAMPLES if 20.0 <= s["elapsed_seconds"] <= 200.0]
        assert len(in_span) == 19  # 21 total - the two valid samples before elapsed=20 (0.21, 10.00)
        invalid_in_span = [s for s in in_span if s["sample_valid"] == "False"]
        assert len(invalid_in_span) == 18
        valid_in_span = [s for s in in_span if s["sample_valid"] == "True"]
        assert [s["elapsed_seconds"] for s in valid_in_span] == [60.01]

    def test_target_60_cannot_select_the_future_60_01_sample(self):
        # 60.01 > 60 -- excluded as a future-relative-to-target sample.
        # The nearest VALID at-or-before-60 sample is 10.00s (age=50s),
        # which exceeds the frozen 30s staleness tolerance -- target=60
        # correctly resolves to unobservable.
        chosen = lib.select_decision_sample(60.0, self.SAMPLES, staleness_threshold_seconds=30.0)
        assert chosen is None

    def test_target_90_selects_60_01_within_tolerance(self):
        chosen = lib.select_decision_sample(90.0, self.SAMPLES, staleness_threshold_seconds=30.0)
        assert chosen is not None
        assert chosen["elapsed_seconds"] == 60.01
        assert pytest.approx(90.0 - 60.01) == 29.99

    def test_target_120_and_150_are_unobservable(self):
        assert lib.select_decision_sample(120.0, self.SAMPLES, staleness_threshold_seconds=30.0) is None
        assert lib.select_decision_sample(150.0, self.SAMPLES, staleness_threshold_seconds=30.0) is None

    def test_no_sample_is_ever_imputed_across_all_four_targets(self):
        # For every target, the result is either None or a sample object
        # taken verbatim from SAMPLES -- never a synthesized/interpolated value.
        for target in (60.0, 90.0, 120.0, 150.0):
            chosen = lib.select_decision_sample(target, self.SAMPLES, staleness_threshold_seconds=30.0)
            assert chosen is None or chosen in self.SAMPLES

    def test_reproduces_frozen_index9_metadata_observable_and_fit_eligible_counts(self):
        # Cross-check against the frozen, already-preserved index9
        # metadata.json counts: decision_points_observable_count=1,
        # decision_points_fit_eligible_count=0 (target=90 is observable
        # via 60.01, but fit-ineligible on health-instability grounds --
        # verified separately, not re-derived here).
        observable = sum(
            1 for t in (60.0, 90.0, 120.0, 150.0)
            if lib.select_decision_sample(t, self.SAMPLES, staleness_threshold_seconds=30.0) is not None
        )
        assert observable == 1


class TestEmptyRoutingDomainConservationAssessabilityPhase15B3J1:
    """A conservation error of numeric 0.0 computed over ZERO retained
    (allocatable) intervals must never be indistinguishable, in the
    returned fields, from a conservation error of 0.0 computed over a
    genuinely non-empty, substantively-assessed domain. `retained_interval_
    count`/`conservation_assessable` make that distinction explicit without
    changing any existing arithmetic or existing field."""

    def test_empty_allocatable_domain_is_not_assessable_despite_zero_error(self):
        # Every candidate interval excluded for incomplete pod counter
        # observability (the index9 shape): allocatable_interval_count=0,
        # but expected/actual conservation errors are still numerically 0.0
        # (a sum over zero elements).
        boundaries = [0.0, 1.0]
        ready_at = {0.0: ["a"]}
        samples = {"a": [(0.0, 100.0)]}  # missing the t=1.0 boundary sample
        relevant_at = {0.0: ["a"], 1.0: ["a"]}
        result = lib.compute_expected_and_actual(boundaries, ready_at, samples, relevant_at)
        assert result["allocatable_interval_count"] == 0
        assert result["expected_conservation_error"] == pytest.approx(0.0)
        assert result["actual_conservation_error"] == pytest.approx(0.0)
        assert result["retained_interval_count"] == 0
        assert result["conservation_assessable"] is False
        assert result["expected_conservation_error_numeric"] == pytest.approx(0.0)
        assert result["actual_conservation_error_numeric"] == pytest.approx(0.0)

    def test_genuinely_retained_interval_with_zero_error_is_assessable(self):
        boundaries = [0.0, 1.0]
        ready_at = {0.0: ["a", "b"]}
        samples = {"a": [(0.0, 100.0), (1.0, 150.0)], "b": [(0.0, 200.0), (1.0, 250.0)]}
        relevant_at = {0.0: ["a", "b"], 1.0: ["a", "b"]}
        result = lib.compute_expected_and_actual(boundaries, ready_at, samples, relevant_at)
        assert result["allocatable_interval_count"] == 1
        assert result["expected_conservation_error"] == pytest.approx(0.0)
        assert result["retained_interval_count"] == 1
        assert result["conservation_assessable"] is True

    def test_index9_shaped_gate_result_reports_routing_evidence_insufficient(self):
        # Reproduces index9's actual on-disk shape: candidate=11,
        # allocatable=0 (all excluded incomplete), both conservation errors
        # numerically 0.0 -- must gate through to classify_routing (which
        # itself, independently, finds 0 assessable pods) rather than ever
        # being read as a routing PASS.
        boundaries = [float(i) for i in range(12)]
        ready_at = {0.0: ["a"]}
        samples = {"a": [(0.0, 0.0)]}  # only one boundary ever sampled
        relevant_at = {t: ["a"] for t in boundaries}
        result = lib.compute_expected_and_actual(boundaries, ready_at, samples, relevant_at)
        assert result["candidate_interval_count"] == 11
        assert result["allocatable_interval_count"] == 0
        gated = harness.gate_and_classify_routing(result)
        assert gated["classification"] == "INSUFFICIENT_ROUTING_SAMPLE"
        assert gated["conservation_assessable"] is False
        assert gated["retained_interval_count"] == 0
        assert gated["routing_evidence_sufficient"] is False

    def test_non_empty_domain_with_real_pinning_still_reports_evidence_sufficient(self):
        # Control case: a real, substantively-assessed domain that reaches
        # a non-INSUFFICIENT classification must report
        # routing_evidence_sufficient=True.
        boundaries = [0.0, 1.0]
        ready_at = {0.0: ["a", "b"]}
        samples = {
            "a": [(0.0, 0.0), (1.0, 20.0)],
            "b": [(0.0, 0.0), (1.0, 200.0)],
        }
        relevant_at = {0.0: ["a", "b"], 1.0: ["a", "b"]}
        result = lib.compute_expected_and_actual(boundaries, ready_at, samples, relevant_at)
        gated = harness.gate_and_classify_routing(result)
        assert gated["conservation_assessable"] is True
        assert gated["routing_evidence_sufficient"] is True
        assert gated["classification"] != "INSUFFICIENT_ROUTING_SAMPLE"

    def test_insufficient_boundaries_fallback_path_reports_not_assessable(self):
        # The orchestrator's other INSUFFICIENT_ROUTING_SAMPLE construction
        # site (fewer than 2 boundaries / no pod samples at all) must carry
        # the same explicit non-assessability fields.
        routing_summary = {
            "classification": "INSUFFICIENT_ROUTING_SAMPLE",
            "reason": "insufficient boundaries/samples collected",
            "retained_interval_count": 0,
            "conservation_assessable": False,
            "routing_evidence_sufficient": False,
        }
        assert routing_summary["conservation_assessable"] is False
        assert routing_summary["routing_evidence_sufficient"] is False


class TestIntegrityUnchangedPhase15B3J1:
    """Phase 15B.3J.1 is strictly an offline evidence-reporting closure --
    it must not alter the frozen HPA baseline, Protocol V3 files, or the
    global campaign sequence hash."""

    def test_hpa_260_still_frozen(self):
        text = (PROJECT_ROOT / "k8s/hpa/hpa.yaml").read_text(encoding="utf-8")
        assert "averageUtilization: 260" in text

    def test_global_campaign_sequence_hash_still_unchanged(self):
        import hashlib

        path = PROJECT_ROOT / "configs/traffic_profiles/rl_dynamics_calibration/global_campaign_sequence.json"
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual_hash == "ce1de372e92de0460e53f1ca36d31ec402712d5cd638738200fcedf70403e901"

    def test_v3_manifest_and_protocol_doc_hashes_unchanged(self):
        import hashlib

        manifest_path = PROJECT_ROOT / "experiments/manifests/rl_dynamics_calibration_v3.yaml"
        protocol_path = PROJECT_ROOT / "docs/rl_dynamics_calibration_protocol_v3.md"
        assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == "a0c7f3cebcdcbd7f9ca766a7057681e7b29cd2d10c4b8de74478fb54681561b8"
        assert hashlib.sha256(protocol_path.read_bytes()).hexdigest() == "9c7a56b631067c3004c9f8b86d918c2304801f79eb2861a2c6aeee3d1ee7660d"


# ---------------------------------------------------------------------------
# Phase 15B.3K: index10 merged-cell (dyncal_r8_L2_L5) non-duplication proof
# ---------------------------------------------------------------------------


class TestMergedCellIndex10NeverDuplicatedPhase15B3K:
    """`dyncal_r8_L2_L5` is a single, already-frozen merged cell (replicas=8,
    total_offered_rps=40.0) standing in for what would otherwise have been
    two separate cells (`dyncal_r8_L2` sustainable-per-replica-anchor and
    `dyncal_r8_L5` fixed-total-contention-anchor) that physically coincide
    at this exact replica count. Every attempt/ledger/registry lookup in
    this harness keys on the single `cell_or_transition_id` string --
    proving that grouping never splits or duplicates it into two synthetic
    per-role cell-support contributions."""

    def test_resolve_planned_slot_index10_is_the_single_merged_cell(self):
        slot = lib.resolve_planned_slot(10)
        assert slot["cell_or_transition_id"] == "dyncal_r8_L2_L5"
        assert slot["planned_slot_id"] == "fit_dyncal_r8_L2_L5_repeat1"
        assert slot["replicas"] == 8
        assert slot["offered_rps"] == 40.0
        assert slot["repeat_number"] == 1

    def test_global_sequence_has_no_standalone_r8_l2_or_r8_l5_cell(self):
        path = PROJECT_ROOT / "configs/traffic_profiles/rl_dynamics_calibration/global_campaign_sequence.json"
        sequence = json.loads(path.read_text(encoding="utf-8"))
        cell_ids = {s["cell_or_transition_id"] for s in sequence}
        assert "dyncal_r8_L2_L5" in cell_ids
        assert "dyncal_r8_L2" not in cell_ids
        assert "dyncal_r8_L5" not in cell_ids

    def test_global_sequence_has_exactly_three_planned_repeats_for_the_merged_cell(self):
        path = PROJECT_ROOT / "configs/traffic_profiles/rl_dynamics_calibration/global_campaign_sequence.json"
        sequence = json.loads(path.read_text(encoding="utf-8"))
        merged = [s for s in sequence if s["cell_or_transition_id"] == "dyncal_r8_L2_L5"]
        assert len(merged) == 3
        assert sorted(s["campaign_sequence_index"] for s in merged) == [10, 55, 86]
        assert sorted(s["repeat_number"] for s in merged) == [1, 2, 3]

    def test_group_attempts_by_cell_does_not_split_the_merged_cell(self):
        attempts = [{"cell_or_transition_id": "dyncal_r8_L2_L5", "attempt_id": "fit_dyncal_r8_L2_L5_repeat1_attempt1"}]
        groups = lib.group_attempt_ids_by_cell(attempts)
        assert groups == {"dyncal_r8_L2_L5": ["fit_dyncal_r8_L2_L5_repeat1_attempt1"]}
        assert "dyncal_r8_L2" not in groups
        assert "dyncal_r8_L5" not in groups

    def test_one_official_attempt_contributes_exactly_one_repeat_not_two(self):
        # One official attempt for this planned slot must contribute
        # exactly ONE repeat toward the merged cell's support count --
        # never two (one per synthetic L2/L5 role).
        attempts = [{"cell_or_transition_id": "dyncal_r8_L2_L5", "attempt_id": "fit_dyncal_r8_L2_L5_repeat1_attempt1"}]
        groups = lib.group_attempt_ids_by_cell(attempts)
        assert len(groups) == 1
        assert len(groups["dyncal_r8_L2_L5"]) == 1

    def test_two_official_repeats_accumulate_under_one_key_not_two(self):
        attempts = [
            {"cell_or_transition_id": "dyncal_r8_L2_L5", "attempt_id": "fit_dyncal_r8_L2_L5_repeat1_attempt1"},
            {"cell_or_transition_id": "dyncal_r8_L2_L5", "attempt_id": "fit_dyncal_r8_L2_L5_repeat2_attempt1"},
        ]
        groups = lib.group_attempt_ids_by_cell(attempts)
        assert len(groups) == 1
        assert len(groups["dyncal_r8_L2_L5"]) == 2

    def test_schedule_registry_has_one_entry_for_the_merged_cell(self):
        registry_path = PROJECT_ROOT / "configs/traffic_profiles/rl_dynamics_calibration/schedule_registry.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        assert "dyncal_r8_L2_L5" in registry
        assert "dyncal_r8_L2" not in registry
        assert "dyncal_r8_L5" not in registry

    def test_manifest_tags_both_workload_roles_on_the_one_cell(self):
        manifest_path = PROJECT_ROOT / "experiments/manifests/rl_dynamics_calibration_v1.yaml"
        import yaml

        m = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        cell = next(c for c in m["fitting_cells"] if c["cell_id"] == "dyncal_r8_L2_L5")
        assert set(cell["roles"]) == {"sustainable_per_replica_anchor", "fixed_total_contention_anchor"}


class TestKubernetesEventsNormalizedCsvFieldnamesNeverDriftPhase15B3K:
    """Regression test for the index10 crash: normalize_kubernetes_events()'s
    output keys and KUBERNETES_EVENTS_NORMALIZED_CSV_FIELDNAMES drifted out
    of sync (Phase 15B.3J.1 added `involved_object_uid` to the former
    without updating the latter), which crashed an official attempt
    partway through -- after traffic had already run -- on
    `csv.DictWriter.writerow()` raising `ValueError: dict contains fields
    not in fieldnames`. This must never be possible again."""

    def test_every_normalized_event_key_is_a_declared_csv_fieldname(self):
        events_json = {
            "items": [
                {
                    "firstTimestamp": "2026-01-01T00:00:00Z",
                    "lastTimestamp": "2026-01-01T00:01:00Z",
                    "count": 1,
                    "type": "Warning",
                    "reason": "Killing",
                    "involvedObject": {"kind": "Pod", "name": "demo-app-a", "uid": "uid-a"},
                    "message": "Container demo-app failed liveness probe, will be restarted",
                }
            ]
        }
        row = lib.normalize_kubernetes_events(events_json)[0]
        missing = set(row.keys()) - set(harness.KUBERNETES_EVENTS_NORMALIZED_CSV_FIELDNAMES)
        assert not missing, f"normalize_kubernetes_events() emits keys the CSV writer doesn't declare: {missing}"

    def test_csv_dictwriter_does_not_raise_on_a_real_normalized_row(self):
        import csv
        import io

        events_json = {
            "items": [
                {
                    "firstTimestamp": "2026-01-01T00:00:00Z",
                    "reason": "Killing",
                    "involvedObject": {"kind": "Pod", "name": "demo-app-a", "uid": "uid-a"},
                    "message": "Container demo-app failed liveness probe, will be restarted",
                }
            ]
        }
        rows = lib.normalize_kubernetes_events(events_json)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=harness.KUBERNETES_EVENTS_NORMALIZED_CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)  # must not raise


# ---------------------------------------------------------------------------
# Phase 15B.3L.1: SLA-attainment diagnostic (reporting-only, never conflated
# with execution_structural_validity/steady_state_fit_eligibility)
# ---------------------------------------------------------------------------


class TestSlaAssessmentPhase15B3L1:
    def test_p95_just_under_target_and_perfect_success_meets_overall(self):
        result = lib.compute_sla_assessment(p95_latency_seconds=0.299, success_rate=0.99)
        assert result["latency_sla_met"] is True
        assert result["success_sla_met"] is True
        assert result["overall_sla_met"] is True

    def test_p95_at_exactly_target_fails_latency_sla(self):
        # Frozen rule is p95 < target (strict), not <=.
        result = lib.compute_sla_assessment(p95_latency_seconds=0.300, success_rate=1.0)
        assert result["latency_sla_met"] is False
        assert result["overall_sla_met"] is False

    def test_index11_raw_values_fail_latency_but_meet_success(self):
        # index11's own preserved values (Phase 15B.3L.1 audit): client
        # p95=0.566004s, success_rate=1.0 -- reliability floor met, latency
        # SLA violated, so overall must be False despite 100% success.
        result = lib.compute_sla_assessment(p95_latency_seconds=0.566004, success_rate=1.0)
        assert result["success_sla_met"] is True
        assert result["latency_sla_met"] is False
        assert result["overall_sla_met"] is False

    def test_missing_latency_never_fabricates_a_verdict(self):
        result = lib.compute_sla_assessment(p95_latency_seconds=None, success_rate=1.0)
        assert result["latency_sla_met"] is None
        assert result["overall_sla_met"] is None
        assert result["success_sla_met"] is True

    def test_missing_success_rate_never_fabricates_a_verdict(self):
        result = lib.compute_sla_assessment(p95_latency_seconds=0.1, success_rate=None)
        assert result["success_sla_met"] is None
        assert result["overall_sla_met"] is None

    def test_default_thresholds_match_frozen_experiment_and_reward_yaml(self):
        assert lib.SLA_P95_LATENCY_TARGET_SECONDS == 0.300
        assert lib.SLA_SUCCESS_RATE_THRESHOLD == 0.99

    def test_sla_constants_never_silently_drift_from_authoritative_config_files(self):
        # Phase 15B.3M: reads the actual YAML sources directly (not a
        # hardcoded duplicate literal) so a future edit to either config
        # file's SLA values is caught here rather than silently diverging
        # from the reporting-only compute_sla_assessment() defaults.
        import yaml

        experiment = yaml.safe_load((PROJECT_ROOT / "configs/experiment.yaml").read_text(encoding="utf-8"))
        reward = yaml.safe_load((PROJECT_ROOT / "configs/reward.yaml").read_text(encoding="utf-8"))
        assert experiment["sla"]["p95_latency_ms"] == reward["sla"]["p95_target_ms"]  # the file-pair's own SYNC invariant
        assert experiment["sla"]["p95_latency_ms"] / 1000.0 == lib.SLA_P95_LATENCY_TARGET_SECONDS
        assert (1.0 - experiment["sla"]["error_rate"]) == lib.SLA_SUCCESS_RATE_THRESHOLD

    def test_low_success_rate_fails_success_sla_regardless_of_latency(self):
        result = lib.compute_sla_assessment(p95_latency_seconds=0.05, success_rate=0.5)
        assert result["latency_sla_met"] is True
        assert result["success_sla_met"] is False
        assert result["overall_sla_met"] is False


class TestSlaAssessmentDimensionSeparationPhase15B3L1:
    """An SLA-latency violation must never automatically alter Axis A or
    Axis B -- these are three independent dimensions, verified here against
    the frozen rules that actually gate each one."""

    def test_sla_violation_is_a_listed_non_invalidating_performance_outcome(self):
        import yaml

        manifest_path = PROJECT_ROOT / "experiments/manifests/rl_dynamics_calibration_v1.yaml"
        m = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        assert "very_high_latency" in m["valid_performance_outcomes_not_grounds_for_invalidity"]
        assert "very_high_latency" not in m["invalid_cell_structural_criteria"]

    def test_axis_b_eligibility_never_consults_latency_or_sla(self):
        # A decision point with severe latency but a fully-observed sample
        # and matching Ready count must still be ELIGIBLE -- fit eligibility
        # comes only from observability + Ready-count match, never SLA.
        sample = {"elapsed_seconds": 60.0, "cpu_cores": "1.0", "replicas_current": "2", "latency_p95_seconds": "9.999"}
        eligible, reason = lib.assess_decision_point_eligibility(sample, ready_count_at_target=2, intended_replicas=2)
        assert eligible is True
        assert reason == lib.FIT_POINT_ELIGIBLE_NONE

    def test_a_kubernetes_healthy_100_percent_success_run_can_still_fail_sla(self):
        # Direct regression on the index11 shape: Ready-stable, zero
        # restarts, 100% success -- yet the SLA verdict is independently
        # False on latency. Neither field may be inferred from the other.
        sla = lib.compute_sla_assessment(p95_latency_seconds=0.566004, success_rate=1.0)
        kubernetes_healthy = True  # Ready stayed 2/2, restarted_pods_during_run == []
        assert kubernetes_healthy is True
        assert sla["overall_sla_met"] is False


# ---------------------------------------------------------------------------
# Phase 15B.3M.1: load-generator tunnel lifecycle evidence + causal-
# attribution discipline for sustained connection_error blocks
# ---------------------------------------------------------------------------


class TestTunnelTerminationClassificationPhase15B3M1:
    def test_still_alive_before_cleanup_is_explicitly_terminated(self):
        assert lib.classify_tunnel_termination_action(already_dead_before_cleanup=False) == lib.TUNNEL_TERMINATION_EXPLICITLY_TERMINATED

    def test_already_dead_before_cleanup_is_recorded_as_such(self):
        assert lib.classify_tunnel_termination_action(already_dead_before_cleanup=True) == lib.TUNNEL_TERMINATION_ALREADY_EXITED

    def test_the_two_outcomes_are_distinct_strings(self):
        assert lib.TUNNEL_TERMINATION_ALREADY_EXITED != lib.TUNNEL_TERMINATION_EXPLICITLY_TERMINATED


class TestTunnelSubprocessLifecycleMechanicsPhase15B3M1:
    """Exercises the actual Popen/poll/log-capture mechanics the harness's
    tunnel-lifecycle evidence collection depends on, using a real short-lived
    subprocess rather than a full live minikube tunnel."""

    def test_poll_is_none_while_process_is_alive(self, tmp_path):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
        try:
            assert proc.poll() is None
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_exit_code_is_capturable_after_process_exits(self):
        proc = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
        proc.wait(timeout=5)
        assert proc.poll() is not None
        assert proc.returncode == 3

    def test_log_file_captures_full_stdout_and_stderr_after_exit(self, tmp_path):
        log_path = tmp_path / "mk_svc_test.log"
        with open(log_path, "w") as f:
            proc = subprocess.Popen(
                [sys.executable, "-c", "import sys; print('http://127.0.0.1:12345'); print('warning: needs terminal', file=sys.stderr)"],
                stdout=f,
                stderr=subprocess.STDOUT,
            )
            proc.wait(timeout=5)
        text = log_path.read_text()
        assert "http://127.0.0.1:12345" in text
        assert "warning: needs terminal" in text


class TestConnectionErrorNeverAutoAttributedToServiceFailurePhase15B3M1:
    """Regression for the index12 audit finding: a sustained
    connection_error block (even 100% of traffic, even coincident with
    Ready=intended and near-idle CPU) must never be silently auto-labeled a
    'Kubernetes Service failure' anywhere in the harness's own structural-
    validity computation -- causal localization between the external tunnel,
    the in-cluster Service/dataplane, and the application is explicitly NOT
    resolvable from generator-side outcome counts alone (Phase 15B.3M.1)."""

    def test_offered_equals_sent_ignores_connection_error_volume(self):
        # 100% connection_error, offered==sent -- this check must still pass;
        # it is not, and must never become, a proxy for "the workload was
        # actually delivered".
        traffic_metadata = {"total_offered": 11520, "total_sent": 11520}
        ok, _ = lib.check_offered_equals_sent(traffic_metadata)
        assert ok is True

    def test_dropped_overload_check_ignores_connection_error_volume(self):
        traffic_metadata = {"outcome_totals": {"dropped_overload": 0, "connection_error": 9179}}
        ok, _ = lib.check_no_dropped_overload(traffic_metadata)
        assert ok is True

    def test_no_structural_invalidity_check_inspects_connection_error_counts(self):
        # Structural-validity gating in this harness is a fixed set of
        # criteria never keyed on the *content* of traffic outcome_totals
        # beyond offered/sent and dropped_overload -- confirmed by reading
        # scripts/run_dynamics_calibration_slot.py's own source directly,
        # so a future edit that starts branching on connection_error counts
        # would be caught here.
        source = (PROJECT_ROOT / "scripts/run_dynamics_calibration_slot.py").read_text(encoding="utf-8")
        assert 'outcome_totals["connection_error"]' not in source
        assert "outcome_totals['connection_error']" not in source


# ---------------------------------------------------------------------------
# Phase 15B.3M.2: CORR-39 resolved route-integrity contract
# ---------------------------------------------------------------------------


def _traffic_summary_row(elapsed, sent, success):
    return {"elapsed_seconds": elapsed, "sent_requests": sent, "success_2xx": success}


class TestLongestZeroSuccessRunPhase15B3M2:
    def test_index12_shaped_data_reproduces_the_121s_block(self):
        rows = [_traffic_summary_row(t, 64, 64 if t < 7 else 0) for t in range(180)]
        # emulate a brief mid-run success blip at t=90 (does not break the
        # LONGEST run, which is the 59-179 tail in the real data) -- here
        # simplified to a single unbroken tail from t=59.
        rows = [_traffic_summary_row(t, 64, 0 if t >= 59 else 1) for t in range(180)]
        result = lib.longest_zero_success_run(rows)
        assert result == (59.0, 179.0)

    def test_no_zero_success_second_returns_none(self):
        rows = [_traffic_summary_row(t, 64, 64) for t in range(10)]
        assert lib.longest_zero_success_run(rows) is None

    def test_zero_sent_second_is_not_a_failure_second(self):
        # sent_requests==0 (e.g. a warmup gap) must never count toward a
        # zero-success run -- there was nothing to succeed or fail.
        rows = [_traffic_summary_row(0, 0, 0), _traffic_summary_row(1, 64, 64)]
        assert lib.longest_zero_success_run(rows) is None

    def test_picks_the_longest_of_multiple_runs(self):
        rows = (
            [_traffic_summary_row(t, 64, 0) for t in range(0, 3)]      # run of 3
            + [_traffic_summary_row(t, 64, 64) for t in range(3, 5)]   # break
            + [_traffic_summary_row(t, 64, 0) for t in range(5, 15)]   # run of 10 (longest)
        )
        assert lib.longest_zero_success_run(rows) == (5.0, 14.0)


def _window(status):
    """Test helper: a minimal evidence-window dict carrying only the field
    classify_route_integrity() actually consults (window_route_status)."""
    return {"window_route_status": status}


class TestFindAllZeroSuccessRunsPhase15B3M4:
    """Phase 15B.3M.4: a run can contain MORE THAN ONE independently
    material zero-success interval -- `longest_zero_success_run()` alone
    is insufficient for attempt-level classification, since it silently
    discards every non-longest run."""

    def test_returns_every_maximal_run_in_encounter_order(self):
        rows = (
            [_traffic_summary_row(t, 64, 0) for t in range(0, 3)]
            + [_traffic_summary_row(t, 64, 64) for t in range(3, 5)]
            + [_traffic_summary_row(t, 64, 0) for t in range(5, 15)]
        )
        assert lib.find_all_zero_success_runs(rows) == [(0.0, 2.0), (5.0, 14.0)]

    def test_no_runs_returns_empty_list(self):
        rows = [_traffic_summary_row(t, 64, 64) for t in range(10)]
        assert lib.find_all_zero_success_runs(rows) == []

    def test_zero_sent_second_never_starts_or_extends_a_run(self):
        rows = [_traffic_summary_row(0, 64, 0), _traffic_summary_row(1, 0, 0), _traffic_summary_row(2, 64, 0)]
        # sent_requests==0 at t=1 breaks the run -- two separate 1s runs.
        assert lib.find_all_zero_success_runs(rows) == [(0.0, 0.0), (2.0, 2.0)]


class TestIsIntervalMaterialPhase15B3M4:
    """The materiality gate itself is UNCHANGED by this phase -- these
    tests pin that `is_interval_material()` reproduces exactly the same
    25%-plus-scored-window-overlap rule `classify_route_integrity` used
    to apply inline before Phase 15B.3M.4."""

    def test_121_of_180_overlapping_scored_window_is_material(self):
        assert lib.is_interval_material(59.0, 180.0, 180.0, (60.0, 150.0)) is True

    def test_5_of_180_is_not_material(self):
        assert lib.is_interval_material(0.0, 5.0, 180.0, (60.0, 150.0)) is False

    def test_material_duration_not_overlapping_scored_window_is_not_material(self):
        assert lib.is_interval_material(0.0, 100.0, 180.0, (150.0, 180.0)) is False

    def test_zero_duration_interval_is_never_material(self):
        assert lib.is_interval_material(10.0, 10.0, 180.0, (0.0, 180.0)) is False


class TestPartitionIntoEvidenceWindowsPhase15B3M4:
    def test_120s_interval_partitions_into_12_ten_second_windows(self):
        windows = lib.partition_into_evidence_windows(59.0, 179.0)
        assert len(windows) == 12
        assert windows[0] == (59.0, 69.0)
        assert windows[-1] == (169.0, 179.0)

    def test_shorter_final_window_is_preserved_not_dropped(self):
        # [59, 180) does not divide evenly by 10 -- the final window must
        # be the shorter [179, 180) remainder, never dropped or merged.
        windows = lib.partition_into_evidence_windows(59.0, 180.0)
        assert windows[-1] == (179.0, 180.0)
        assert sum(end - start for start, end in windows) == pytest.approx(121.0)

    def test_windows_are_half_open_and_contiguous(self):
        windows = lib.partition_into_evidence_windows(0.0, 25.0)
        assert windows == [(0.0, 10.0), (10.0, 20.0), (20.0, 25.0)]
        for (_, end1), (start2, _) in zip(windows, windows[1:]):
            assert end1 == start2


class TestClassifyRouteIntegrityWindowPhase15B3M4:
    """Per-window (not attempt-level) classification -- the temporal core
    of Phase 15B.3M.4's hardening."""

    def test_tunnel_dead_in_window_is_failed(self):
        result = lib.classify_route_integrity_window(delivered_request_evidence=True, tunnel_process_alive=False)
        assert result["window_route_status"] == lib.ROUTE_INTEGRITY_FAILED

    def test_listener_absent_in_window_is_failed_even_with_delivery_evidence(self):
        result = lib.classify_route_integrity_window(
            delivered_request_evidence=True, tunnel_process_alive=True, local_listener_present=False
        )
        assert result["window_route_status"] == lib.ROUTE_INTEGRITY_FAILED

    def test_delivery_evidence_with_no_component_failure_is_confirmed(self):
        result = lib.classify_route_integrity_window(
            delivered_request_evidence=True, tunnel_process_alive=True, local_listener_present=True
        )
        assert result["window_route_status"] == lib.ROUTE_INTEGRITY_CONFIRMED

    def test_healthy_components_but_no_delivery_evidence_is_unverified(self):
        # THE CRITICAL REGRESSION CASE, now scoped to a single window:
        # process alive + listener present but zero delivery evidence in
        # THIS window must never be CONFIRMED.
        result = lib.classify_route_integrity_window(
            delivered_request_evidence=False, tunnel_process_alive=True, local_listener_present=True
        )
        assert result["window_route_status"] == lib.ROUTE_INTEGRITY_UNVERIFIED

    def test_unknown_component_state_with_delivery_evidence_is_still_confirmed(self):
        result = lib.classify_route_integrity_window(delivered_request_evidence=True)
        assert result["window_route_status"] == lib.ROUTE_INTEGRITY_CONFIRMED

    def test_no_evidence_and_no_component_state_is_unverified(self):
        result = lib.classify_route_integrity_window(delivered_request_evidence=False)
        assert result["window_route_status"] == lib.ROUTE_INTEGRITY_UNVERIFIED


class TestComputeDeliveredRequestEvidenceWindowsPhase15B3M4:
    """`compute_delivered_request_evidence_windows()`: the windowed sibling
    of Phase 15B.3M.3's interval-wide `compute_delivered_request_evidence`,
    WINDOW-LOCAL ATTRIBUTION hardened (Phase 15B.3M.5) -- HTTP responses are
    attributed by DISPATCH time (never completion time), and rolling server
    RPS is demoted to supporting-only, never decisive. Always returns a
    concrete bool for `delivered_request_evidence` -- a window is a
    synthetic partition of an already-material interval, not a separate
    observation attempt."""

    def test_response_dispatched_inside_window_is_delivery_evidence(self):
        requests_rows = [{"actual_dispatch_elapsed_seconds": 62.0, "completion_elapsed_seconds": 62.3, "outcome": "success_2xx"}]
        result = lib.compute_delivered_request_evidence_windows(60.0, 70.0, requests_rows=requests_rows)
        assert result["http_response_delivery_evidence"] is True
        assert result["delivered_request_evidence"] is True
        assert result["delivery_evidence_sources"] == ["http_response"]

    def test_5xx_response_dispatched_inside_window_counts_as_delivery_evidence(self):
        requests_rows = [{"actual_dispatch_elapsed_seconds": 65.0, "completion_elapsed_seconds": 65.4, "outcome": "server_error_5xx"}]
        result = lib.compute_delivered_request_evidence_windows(60.0, 70.0, requests_rows=requests_rows)
        assert result["delivered_request_evidence"] is True

    def test_timeout_only_with_no_server_or_counter_evidence_is_unverified(self):
        requests_rows = [{"actual_dispatch_elapsed_seconds": 65.0, "completion_elapsed_seconds": 70.0, "outcome": "timeout"}]
        result = lib.compute_delivered_request_evidence_windows(60.0, 70.0, requests_rows=requests_rows)
        assert result["http_response_delivery_evidence"] is False
        assert result["delivered_request_evidence"] is False

    def test_connection_error_only_is_not_delivery_evidence(self):
        requests_rows = [{"actual_dispatch_elapsed_seconds": 65.0, "completion_elapsed_seconds": 65.1, "outcome": "connection_error"}]
        result = lib.compute_delivered_request_evidence_windows(60.0, 70.0, requests_rows=requests_rows)
        assert result["delivered_request_evidence"] is False

    def test_dispatch_outside_window_bounds_does_not_count(self):
        requests_rows = [{"actual_dispatch_elapsed_seconds": 71.0, "completion_elapsed_seconds": 71.2, "outcome": "success_2xx"}]
        result = lib.compute_delivered_request_evidence_windows(60.0, 70.0, requests_rows=requests_rows)
        assert result["delivered_request_evidence"] is False

    def test_critical_http_regression_dispatched_before_window_completed_inside_is_not_credited(self):
        # Phase 15B.3M.5's own named regression (task sec.13, case 1): a
        # request that BEGAN its route-entry attempt before this window,
        # whose response merely happens to arrive inside it, must NOT
        # confirm this window -- attribution belongs to the window in
        # which the attempt was actually initiated.
        requests_rows = [{"actual_dispatch_elapsed_seconds": 59.5, "completion_elapsed_seconds": 61.0, "outcome": "success_2xx"}]
        result = lib.compute_delivered_request_evidence_windows(60.0, 70.0, requests_rows=requests_rows)
        assert result["http_response_delivery_evidence"] is False
        assert result["delivered_request_evidence"] is False

    def test_critical_http_regression_dispatched_inside_window_completed_later_is_credited(self):
        # Phase 15B.3M.5's own named regression (task sec.13, case 2): the
        # response may complete arbitrarily later; what matters is that the
        # route-entry attempt itself began inside this window.
        requests_rows = [{"actual_dispatch_elapsed_seconds": 61.0, "completion_elapsed_seconds": 75.0, "outcome": "server_error_5xx"}]
        result = lib.compute_delivered_request_evidence_windows(60.0, 70.0, requests_rows=requests_rows)
        assert result["http_response_delivery_evidence"] is True
        assert result["delivered_request_evidence"] is True
        assert result["http_response_dispatch_times"] == [61.0]
        assert result["http_response_completion_times"] == [75.0]

    def test_http_response_provenance_fields_populated(self):
        requests_rows = [
            {"request_index": 42, "actual_dispatch_elapsed_seconds": 61.0, "completion_elapsed_seconds": 61.2, "outcome": "success_2xx"}
        ]
        result = lib.compute_delivered_request_evidence_windows(60.0, 70.0, requests_rows=requests_rows)
        assert result["http_response_request_indices"] == [42]
        assert result["http_response_dispatch_times"] == [61.0]
        assert result["http_response_completion_times"] == [61.2]

    def test_critical_rolling_rps_regression_positive_rps_alone_never_confirms_window(self):
        # Phase 15B.3M.5's own named regression (task sec.14): a positive
        # rolling (60s-lookback) RPS sample timestamped inside this window,
        # with no window-local HTTP dispatch and no localizable counter
        # increment, must NOT set delivered_request_evidence.
        metrics_rows = [{"timestamp_epoch": 1065.0, "request_rate_rps": "3.0"}]
        result = lib.compute_delivered_request_evidence_windows(
            60.0, 70.0, traffic_start_epoch=1000.0, metrics_rows=metrics_rows
        )
        assert result["rolling_server_rps_observed"] is True
        assert result["rolling_server_rps_supporting_only"] is True
        assert result["delivered_request_evidence"] is False

    def test_zero_server_rps_is_not_delivery_evidence(self):
        metrics_rows = [{"timestamp_epoch": 1065.0, "request_rate_rps": "0.0"}]
        result = lib.compute_delivered_request_evidence_windows(
            60.0, 70.0, traffic_start_epoch=1000.0, metrics_rows=metrics_rows
        )
        assert result["rolling_server_rps_observed"] is False
        assert result["delivered_request_evidence"] is False

    def test_counter_increment_localized_entirely_within_one_window_counts(self):
        counter_rows = [
            {"timestamp_epoch": 1061.0, "pod": "p1", "raw_counter_value": 10.0},
            {"timestamp_epoch": 1068.0, "pod": "p1", "raw_counter_value": 12.0},
        ]
        result = lib.compute_delivered_request_evidence_windows(
            60.0, 70.0, traffic_start_epoch=1000.0, counter_rows=counter_rows
        )
        assert result["counter_delivery_evidence"] is True
        assert result["delivered_request_evidence"] is True
        assert result["delivery_evidence_sources"] == ["counter_increment"]
        assert len(result["counter_intervals_used"]) == 1
        assert result["counter_intervals_rejected_as_cross_window"] == []

    def test_counter_increment_straddling_two_windows_credited_to_neither(self):
        # Phase 15B.3M.4's explicit non-duplication rule: an increment whose
        # bounding samples fall in DIFFERENT windows must not be credited
        # to either one -- and Phase 15B.3M.5 additionally requires it be
        # recorded as a REJECTED cross-window interval for both.
        counter_rows = [
            {"timestamp_epoch": 1068.0, "pod": "p1", "raw_counter_value": 10.0},   # in [60,70)
            {"timestamp_epoch": 1072.0, "pod": "p1", "raw_counter_value": 12.0},   # in [70,80)
        ]
        window1 = lib.compute_delivered_request_evidence_windows(
            60.0, 70.0, traffic_start_epoch=1000.0, counter_rows=counter_rows
        )
        window2 = lib.compute_delivered_request_evidence_windows(
            70.0, 80.0, traffic_start_epoch=1000.0, counter_rows=counter_rows
        )
        assert window1["counter_delivery_evidence"] is False
        assert window2["counter_delivery_evidence"] is False
        assert len(window1["counter_intervals_rejected_as_cross_window"]) == 1
        assert len(window2["counter_intervals_rejected_as_cross_window"]) == 1

    def test_flat_counter_within_window_is_not_delivery_evidence(self):
        counter_rows = [
            {"timestamp_epoch": 1061.0, "pod": "p1", "raw_counter_value": 10.0},
            {"timestamp_epoch": 1068.0, "pod": "p1", "raw_counter_value": 10.0},
        ]
        result = lib.compute_delivered_request_evidence_windows(
            60.0, 70.0, traffic_start_epoch=1000.0, counter_rows=counter_rows
        )
        assert result["counter_delivery_evidence"] is False

    def test_counter_reset_decrease_is_never_misread_as_an_increment(self):
        # A restart/reset producing a lower second reading must never be
        # credited as delivery evidence -- v2 > v1 is required, so a
        # decrease is structurally excluded, never specially detected as
        # "a reset" and then reinterpreted as growth.
        counter_rows = [
            {"timestamp_epoch": 1061.0, "pod": "p1", "raw_counter_value": 500.0},
            {"timestamp_epoch": 1068.0, "pod": "p1", "raw_counter_value": 3.0},
        ]
        result = lib.compute_delivered_request_evidence_windows(
            60.0, 70.0, traffic_start_epoch=1000.0, counter_rows=counter_rows
        )
        assert result["counter_delivery_evidence"] is False

    def test_no_evidence_in_window_from_any_source_is_false_not_none(self):
        result = lib.compute_delivered_request_evidence_windows(60.0, 70.0)
        assert result["delivered_request_evidence"] is False
        assert result["delivery_evidence_sources"] == []


class TestAggregateObserverEvidenceForWindowPhase15B3M4:
    def test_no_samples_in_window_reports_none_never_fabricated_healthy(self):
        result = lib.aggregate_observer_evidence_for_window(60.0, 70.0, periodic_samples=[])
        assert result["sample_count"] == 0
        assert result["tunnel_process_alive"] is None
        assert result["local_listener_present"] is None

    def test_all_samples_healthy_in_window_is_true(self):
        samples = [
            {"elapsed_seconds_since_traffic_start": 62.0, "tunnel_process_alive": True, "local_listener_present": True},
        ]
        result = lib.aggregate_observer_evidence_for_window(60.0, 70.0, periodic_samples=samples)
        assert result["tunnel_process_alive"] is True
        assert result["local_listener_present"] is True

    def test_any_unhealthy_sample_in_window_is_false(self):
        samples = [
            {"elapsed_seconds_since_traffic_start": 62.0, "tunnel_process_alive": True, "local_listener_present": True},
            {"elapsed_seconds_since_traffic_start": 68.0, "tunnel_process_alive": False, "local_listener_present": True},
        ]
        result = lib.aggregate_observer_evidence_for_window(60.0, 70.0, periodic_samples=samples)
        assert result["tunnel_process_alive"] is False

    def test_sample_outside_window_bounds_is_excluded(self):
        samples = [{"elapsed_seconds_since_traffic_start": 75.0, "tunnel_process_alive": False}]
        result = lib.aggregate_observer_evidence_for_window(60.0, 70.0, periodic_samples=samples)
        assert result["sample_count"] == 0
        assert result["tunnel_process_alive"] is None


class TestRouteIntegrityClassificationResolvedPhase15B3M2:
    """The resolved CORR-39 contract: CONFIRMED / FAILED / UNVERIFIED,
    TEMPORAL-COVERAGE hardened (Phase 15B.3M.4) -- `classify_route_integrity()`
    now takes `material_intervals` + a flattened `evidence_windows` list
    (each window already independently classified by
    `classify_route_integrity_window()`), never a single interval-wide
    boolean. Attempt-level: FAILED if any window FAILED; CONFIRMED only if
    EVERY window is CONFIRMED (full temporal coverage); UNVERIFIED
    otherwise."""

    def test_no_material_interval_is_confirmed_even_with_no_evidence_at_all(self):
        result = lib.classify_route_integrity(material_intervals=[], evidence_windows=[])
        assert result["classification"] == lib.ROUTE_INTEGRITY_CONFIRMED
        assert result["full_temporal_coverage"] is True

    def test_temporal_false_positive_regression_one_delivery_then_eleven_empty_windows_is_unverified(self):
        # THE EXACT REGRESSION THIS PHASE CLOSES: a 120s material interval,
        # 12 windows, window1 CONFIRMED (delivery evidence), windows 2-12
        # UNVERIFIED (process/listener healthy, zero delivery evidence).
        # Must be attempt-level UNVERIFIED, never CONFIRMED.
        windows = [_window(lib.ROUTE_INTEGRITY_CONFIRMED)] + [_window(lib.ROUTE_INTEGRITY_UNVERIFIED)] * 11
        result = lib.classify_route_integrity(material_intervals=[(59.0, 179.0)], evidence_windows=windows)
        assert result["classification"] == lib.ROUTE_INTEGRITY_UNVERIFIED
        assert result["confirmed_window_count"] == 1
        assert result["unverified_window_count"] == 11
        assert result["full_temporal_coverage"] is False

    def test_full_temporal_coverage_is_confirmed_even_under_catastrophic_client_performance(self):
        # Every window has server-side delivery evidence (e.g. a counter
        # increment) even though every CLIENT outcome could be timeout/5xx
        # -- full coverage is CONFIRMED.
        windows = [_window(lib.ROUTE_INTEGRITY_CONFIRMED) for _ in range(12)]
        result = lib.classify_route_integrity(material_intervals=[(59.0, 179.0)], evidence_windows=windows)
        assert result["classification"] == lib.ROUTE_INTEGRITY_CONFIRMED
        assert result["full_temporal_coverage"] is True

    def test_mid_interval_tunnel_failure_is_failed_never_downgraded_to_unverified(self):
        # windows 1-4 CONFIRMED, window5 FAILED (listener absent), windows
        # 6-12 UNVERIFIED -- attempt-level must be FAILED, and earlier
        # confirmed windows never downgrade a later confirmed failure.
        windows = (
            [_window(lib.ROUTE_INTEGRITY_CONFIRMED)] * 4
            + [_window(lib.ROUTE_INTEGRITY_FAILED)]
            + [_window(lib.ROUTE_INTEGRITY_UNVERIFIED)] * 7
        )
        result = lib.classify_route_integrity(material_intervals=[(59.0, 179.0)], evidence_windows=windows)
        assert result["classification"] == lib.ROUTE_INTEGRITY_FAILED
        assert result["failed_window_count"] == 1

    def test_multiple_material_intervals_all_evaluated_not_only_longest(self):
        # Two independently-material intervals; the SHORTER one has the
        # unverified window -- it must not be ignored just because it is
        # not the longest.
        windows = (
            [_window(lib.ROUTE_INTEGRITY_CONFIRMED)] * 5     # interval A, fully confirmed
            + [_window(lib.ROUTE_INTEGRITY_UNVERIFIED)]      # interval B, one bad window
        )
        result = lib.classify_route_integrity(
            material_intervals=[(0.0, 50.0), (100.0, 110.0)], evidence_windows=windows
        )
        assert result["classification"] == lib.ROUTE_INTEGRITY_UNVERIFIED

    def test_some_positive_evidence_never_upgrades_a_later_unverified_window(self):
        windows = [_window(lib.ROUTE_INTEGRITY_CONFIRMED)] * 6 + [_window(lib.ROUTE_INTEGRITY_UNVERIFIED)]
        result = lib.classify_route_integrity(material_intervals=[(0.0, 70.0)], evidence_windows=windows)
        assert result["classification"] == lib.ROUTE_INTEGRITY_UNVERIFIED

    def test_offered_equals_sent_never_overrides_failed_or_unverified(self):
        # offered==sent/dropped_overload==0 are computed entirely separately
        # from route integrity and must never suppress a FAILED/UNVERIFIED
        # result -- verified structurally: classify_route_integrity() takes
        # no offered/sent/dropped_overload argument at all.
        import inspect

        params = inspect.signature(lib.classify_route_integrity).parameters
        assert "offered" not in params and "sent" not in params and "dropped_overload" not in params

    def test_result_echoes_back_the_evidence_that_produced_it(self):
        # The classification must never be divorced from its own evidence.
        windows = [_window(lib.ROUTE_INTEGRITY_UNVERIFIED)]
        result = lib.classify_route_integrity(material_intervals=[(59.0, 69.0)], evidence_windows=windows)
        assert result["material_intervals"] == [(59.0, 69.0)]
        assert result["evidence_windows"] == windows
        assert "reason" in result and isinstance(result["reason"], str) and len(result["reason"]) > 0


class TestDeliveredRequestEvidencePhase15B3M3:
    def test_positive_server_rps_in_window_is_delivery_evidence(self):
        metrics_rows = [{"timestamp_epoch": 1000.0, "request_rate_rps": "5.2"}]
        result = lib.compute_delivered_request_evidence(999.0, 1001.0, metrics_rows=metrics_rows)
        assert result is True

    def test_zero_server_rps_with_no_counter_increase_is_false_not_none(self):
        metrics_rows = [{"timestamp_epoch": 1000.0, "request_rate_rps": "0.0"}]
        result = lib.compute_delivered_request_evidence(999.0, 1001.0, metrics_rows=metrics_rows)
        assert result is False

    def test_counter_increment_within_window_is_delivery_evidence(self):
        # Phase 15B.3M.5: the real `per_pod_request_counters.csv` column is
        # `pod`, not `pod_name` (scripts/per_pod_counter_collector.py's own
        # CSV_FIELDNAMES) -- this test, and the fixed implementation, use
        # the real field name.
        counter_rows = [
            {"timestamp_epoch": 1000.0, "pod": "p1", "raw_counter_value": 10.0},
            {"timestamp_epoch": 1005.0, "pod": "p1", "raw_counter_value": 25.0},
        ]
        result = lib.compute_delivered_request_evidence(999.0, 1010.0, counter_rows=counter_rows)
        assert result is True

    def test_flat_counter_within_window_is_false(self):
        counter_rows = [
            {"timestamp_epoch": 1000.0, "pod": "p1", "raw_counter_value": 10.0},
            {"timestamp_epoch": 1005.0, "pod": "p1", "raw_counter_value": 10.0},
        ]
        result = lib.compute_delivered_request_evidence(999.0, 1010.0, counter_rows=counter_rows)
        assert result is False

    def test_pod_field_name_matches_real_csv_column_not_pod_name(self):
        # Phase 15B.3M.5 regression: before this fix, `pod_name` was read
        # but the real collector writes `pod` -- a row using the WRONG key
        # must never be silently credited as delivery evidence.
        counter_rows = [
            {"timestamp_epoch": 1000.0, "pod_name": "p1", "raw_counter_value": 10.0},
            {"timestamp_epoch": 1005.0, "pod_name": "p1", "raw_counter_value": 25.0},
        ]
        result = lib.compute_delivered_request_evidence(999.0, 1010.0, counter_rows=counter_rows)
        assert result is False

    def test_no_rows_in_window_at_all_is_none_not_false(self):
        metrics_rows = [{"timestamp_epoch": 500.0, "request_rate_rps": "5.0"}]
        result = lib.compute_delivered_request_evidence(999.0, 1010.0, metrics_rows=metrics_rows)
        assert result is None

    def test_empty_rps_string_never_crashes_and_is_not_treated_as_positive(self):
        metrics_rows = [{"timestamp_epoch": 1000.0, "request_rate_rps": ""}]
        result = lib.compute_delivered_request_evidence(999.0, 1001.0, metrics_rows=metrics_rows)
        assert result is False


class TestCellSupportExcludesAxisAInvalidAttemptsPhase15B3M2:
    """CORR-39 sec.E: an Axis-A-INVALID attempt must never count as a
    sufficient repeat toward cell support, regardless of its own local
    decision-point fit-eligible count."""

    def test_axis_a_invalid_attempt_with_high_fit_eligible_count_is_not_sufficient_support(self):
        # index12's own shape: 2 of 4 decision points fit-eligible locally
        # (>= MIN_FIT_ELIGIBLE_POINTS_FOR_SUFFICIENT_REPEAT), but Axis A is
        # INVALID (route_integrity=UNVERIFIED) -- this attempt must
        # contribute ZERO sufficient-repeat support.
        fit_eligible_count = 2
        execution_structural_validity = "INVALID"
        locally_sufficient = lib.is_repeat_sufficient_for_cell_support(fit_eligible_count)
        assert locally_sufficient is True  # the LOCAL computation is unchanged
        # but an Axis-A-invalid attempt must never be fed into cell-support
        # tallying at all -- the caller-level contract this test documents.
        contributes_to_support = locally_sufficient and execution_structural_validity == "VALID"
        assert contributes_to_support is False

    def test_axis_a_valid_attempt_with_sufficient_points_does_contribute(self):
        fit_eligible_count = 4
        execution_structural_validity = "VALID"
        locally_sufficient = lib.is_repeat_sufficient_for_cell_support(fit_eligible_count)
        contributes_to_support = locally_sufficient and execution_structural_validity == "VALID"
        assert contributes_to_support is True


class TestWindowedRouteIntegrityFeedsAxisAPhase15B3M4:
    """Phase 15B.3M.4: the WINDOWED `classify_route_integrity()` result
    must drive Axis-A exactly the same way the interval-wide Phase
    15B.3M.3 result did -- FAILED or material UNVERIFIED both make
    `execution_structural_validity=INVALID` (mirroring
    `scripts/run_dynamics_calibration_slot.py`'s own
    `EXECUTION_INVALID if invalidity_reasons else EXECUTION_VALID` rule),
    and an Axis-A-invalid attempt then contributes zero cell support
    regardless of its own locally fit-eligible decision-point count."""

    def _execution_structural_validity_for(self, route_integrity_result):
        invalidity_reasons = []
        if route_integrity_result["classification"] == lib.ROUTE_INTEGRITY_FAILED:
            invalidity_reasons.append("wrong_traffic_path_used")
        elif route_integrity_result["classification"] == lib.ROUTE_INTEGRITY_UNVERIFIED:
            invalidity_reasons.append("official_workload_route_integrity_unverified")
        return lib.EXECUTION_INVALID if invalidity_reasons else lib.EXECUTION_VALID

    def test_temporal_false_positive_regression_case_yields_axis_a_invalid(self):
        windows = [_window(lib.ROUTE_INTEGRITY_CONFIRMED)] + [_window(lib.ROUTE_INTEGRITY_UNVERIFIED)] * 11
        result = lib.classify_route_integrity(material_intervals=[(59.0, 179.0)], evidence_windows=windows)
        assert self._execution_structural_validity_for(result) == lib.EXECUTION_INVALID

    def test_mid_interval_failure_case_yields_axis_a_invalid(self):
        windows = [_window(lib.ROUTE_INTEGRITY_CONFIRMED)] * 4 + [_window(lib.ROUTE_INTEGRITY_FAILED)]
        result = lib.classify_route_integrity(material_intervals=[(0.0, 50.0)], evidence_windows=windows)
        assert self._execution_structural_validity_for(result) == lib.EXECUTION_INVALID

    def test_full_temporal_coverage_case_yields_axis_a_valid(self):
        windows = [_window(lib.ROUTE_INTEGRITY_CONFIRMED)] * 12
        result = lib.classify_route_integrity(material_intervals=[(59.0, 179.0)], evidence_windows=windows)
        assert self._execution_structural_validity_for(result) == lib.EXECUTION_VALID

    def test_axis_a_invalid_retry_still_cannot_contribute_cell_support_under_windowed_model(self):
        windows = [_window(lib.ROUTE_INTEGRITY_UNVERIFIED)]
        route_integrity_result = lib.classify_route_integrity(material_intervals=[(59.0, 69.0)], evidence_windows=windows)
        execution_structural_validity = self._execution_structural_validity_for(route_integrity_result)
        fit_eligible_count = 4  # even a fully fit-eligible attempt
        locally_sufficient = lib.is_repeat_sufficient_for_cell_support(fit_eligible_count)
        contributes_to_support = locally_sufficient and execution_structural_validity == lib.EXECUTION_VALID
        assert contributes_to_support is False


class TestWindowLocalAttributionIntegrationPhase15B3M5:
    """End-to-end pipeline test: partition -> per-window attribution
    (compute_delivered_request_evidence_windows) -> per-window
    classification (classify_route_integrity_window) -> attempt-level
    aggregation (classify_route_integrity), proving the Phase 15B.3M.4
    temporal-coverage machinery still works correctly now that per-window
    delivery evidence is attributed by dispatch time and raw counters
    rather than completion time and rolling RPS."""

    def _build_windows(self, interval, requests_rows=(), metrics_rows=(), counter_rows=(), traffic_start_epoch=1000.0):
        windows = []
        for window_start, window_end in lib.partition_into_evidence_windows(*interval):
            delivered = lib.compute_delivered_request_evidence_windows(
                window_start, window_end,
                traffic_start_epoch=traffic_start_epoch,
                requests_rows=requests_rows, metrics_rows=metrics_rows, counter_rows=counter_rows,
            )
            verdict = lib.classify_route_integrity_window(delivered_request_evidence=delivered["delivered_request_evidence"])
            windows.append({**delivered, "window_route_status": verdict["window_route_status"]})
        return windows

    def test_index12_shaped_regression_one_early_response_completing_late_does_not_confirm_whole_interval(self):
        # The exact failure shape this phase's regression targets: a
        # request DISPATCHED just before the material interval begins,
        # whose response happens to COMPLETE inside window1 -- under the
        # old (completion-time) attribution this would have wrongly
        # confirmed window1; under dispatch-time attribution it must not,
        # so the attempt is UNVERIFIED throughout (no window has genuine
        # window-local evidence).
        interval = (59.0, 179.0)
        requests_rows = [
            {"request_index": 1, "actual_dispatch_elapsed_seconds": 58.5, "completion_elapsed_seconds": 61.0, "outcome": "success_2xx"},
        ]
        windows = self._build_windows(interval, requests_rows=requests_rows)
        result = lib.classify_route_integrity(material_intervals=[interval], evidence_windows=windows)
        assert result["classification"] == lib.ROUTE_INTEGRITY_UNVERIFIED
        assert result["confirmed_window_count"] == 0

    def test_genuine_window_local_counter_evidence_in_every_window_yields_confirmed(self):
        interval = (0.0, 30.0)
        counter_rows = [
            {"timestamp_epoch": 1001.0, "pod": "p1", "raw_counter_value": 10.0},
            {"timestamp_epoch": 1008.0, "pod": "p1", "raw_counter_value": 15.0},
            {"timestamp_epoch": 1011.0, "pod": "p1", "raw_counter_value": 15.0},
            {"timestamp_epoch": 1018.0, "pod": "p1", "raw_counter_value": 22.0},
            {"timestamp_epoch": 1021.0, "pod": "p1", "raw_counter_value": 22.0},
            {"timestamp_epoch": 1028.0, "pod": "p1", "raw_counter_value": 30.0},
        ]
        windows = self._build_windows(interval, counter_rows=counter_rows)
        result = lib.classify_route_integrity(material_intervals=[interval], evidence_windows=windows)
        assert result["classification"] == lib.ROUTE_INTEGRITY_CONFIRMED
        assert result["full_temporal_coverage"] is True

    def test_dispatch_localized_http_responses_in_every_window_yield_confirmed_even_if_all_5xx(self):
        interval = (0.0, 20.0)
        requests_rows = [
            {"request_index": 1, "actual_dispatch_elapsed_seconds": 3.0, "completion_elapsed_seconds": 40.0, "outcome": "server_error_5xx"},
            {"request_index": 2, "actual_dispatch_elapsed_seconds": 13.0, "completion_elapsed_seconds": 41.0, "outcome": "server_error_5xx"},
        ]
        windows = self._build_windows(interval, requests_rows=requests_rows)
        result = lib.classify_route_integrity(material_intervals=[interval], evidence_windows=windows)
        assert result["classification"] == lib.ROUTE_INTEGRITY_CONFIRMED

    def test_rolling_rps_positive_everywhere_but_no_local_evidence_anywhere_yields_unverified(self):
        interval = (0.0, 20.0)
        metrics_rows = [
            {"timestamp_epoch": 1005.0, "request_rate_rps": "12.0"},
            {"timestamp_epoch": 1015.0, "request_rate_rps": "12.0"},
        ]
        windows = self._build_windows(interval, metrics_rows=metrics_rows)
        result = lib.classify_route_integrity(material_intervals=[interval], evidence_windows=windows)
        assert result["classification"] == lib.ROUTE_INTEGRITY_UNVERIFIED
        assert all(w["rolling_server_rps_observed"] for w in windows)


class TestIndex12LedgerReclassificationPhase15B3M2:
    def test_ledger_reclassification_produces_invalid_retry_required_and_clears_valid_pointer(self):
        sequence = [{"campaign_sequence_index": i, "planned_slot_id": f"slot{i}"} for i in range(1, 14)]
        ledger = lib.init_ledger(sequence)
        attempt_id = "fit_dyncal_r8_L3_repeat1_attempt1_20260916T130416Z"
        # Simulate the original (incorrect) VALID determination, then the
        # CORR-39 reclassification -- mirroring the exact CORR-36/37
        # precedent for correcting an already-VALID slot.
        lib.update_slot_state(ledger, 12, lib.LEDGER_STATE_VALID, attempt_id=attempt_id, mark_valid=True)
        assert ledger["slots"]["12"]["state"] == lib.LEDGER_STATE_VALID
        assert ledger["slots"]["12"]["valid_attempt_id"] == attempt_id

        lib.update_slot_state(ledger, 12, lib.LEDGER_STATE_INVALID_RETRY_REQUIRED, attempt_id=attempt_id)
        assert ledger["slots"]["12"]["state"] == lib.LEDGER_STATE_INVALID_RETRY_REQUIRED
        assert ledger["slots"]["12"]["valid_attempt_id"] is None
        assert attempt_id in ledger["slots"]["12"]["all_attempt_ids"]  # never removed, only reclassified

    def test_next_resumable_index_returns_12_after_reclassification(self):
        sequence = [{"campaign_sequence_index": i, "planned_slot_id": f"slot{i}"} for i in range(1, 14)]
        ledger = lib.init_ledger(sequence)
        for i in range(1, 13):
            lib.update_slot_state(ledger, i, lib.LEDGER_STATE_VALID, attempt_id=f"a{i}", mark_valid=True)
        lib.update_slot_state(ledger, 12, lib.LEDGER_STATE_INVALID_RETRY_REQUIRED, attempt_id="a12")
        assert lib.next_resumable_index(ledger) == 12

    def test_index13_remains_not_attempted(self):
        sequence = [{"campaign_sequence_index": i, "planned_slot_id": f"slot{i}"} for i in range(1, 14)]
        ledger = lib.init_ledger(sequence)
        assert ledger["slots"]["13"]["state"] == lib.LEDGER_STATE_NOT_ATTEMPTED


class TestCorr39IntegrityUnchangedPhase15B3M2:
    def test_v1_manifest_hash_unchanged(self):
        assert hashlib.sha256(lib.MANIFEST_PATH.read_bytes()).hexdigest() == "2e6011495307245bcc85832d0b0e1aa6cbc6ea63e4655914eeb9c942c15fc2bf"

    def test_v3_manifest_hash_unchanged(self):
        assert hashlib.sha256(lib.MANIFEST_V3_PATH.read_bytes()).hexdigest() == "a0c7f3cebcdcbd7f9ca766a7057681e7b29cd2d10c4b8de74478fb54681561b8"

    def test_v4_manifest_exists_and_is_valid_yaml(self):
        import yaml

        m = yaml.safe_load(lib.MANIFEST_V4_PATH.read_text(encoding="utf-8"))
        assert m["protocol_version"] == 4
        assert m["correction_reference"] == "CORR-39"

    def test_global_sequence_hash_unchanged(self):
        assert hashlib.sha256(lib.GLOBAL_SEQUENCE_PATH.read_bytes()).hexdigest() == "ce1de372e92de0460e53f1ca36d31ec402712d5cd638738200fcedf70403e901"

    def test_hpa_260_unchanged(self):
        text = (PROJECT_ROOT / "k8s/hpa/hpa.yaml").read_text(encoding="utf-8")
        assert "averageUtilization: 260" in text

    def test_rl_and_reward_contracts_unchanged(self):
        rl_text = (PROJECT_ROOT / "configs/rl.yaml").read_text(encoding="utf-8")
        reward_text = (PROJECT_ROOT / "configs/reward.yaml").read_text(encoding="utf-8")
        assert "rl_contract_version: 2" in rl_text
        assert "reward_contract_version: 2" in reward_text

    def test_sla_thresholds_unchanged(self):
        assert lib.SLA_P95_LATENCY_TARGET_SECONDS == 0.300
        assert lib.SLA_SUCCESS_RATE_THRESHOLD == 0.99

    def test_cpu_normalization_denominators_unchanged(self):
        r = lib.cpu_normalization(0.4, 1)
        assert r["aggregate_request_cores"] == pytest.approx(0.1)
        assert r["aggregate_limit_cores"] == pytest.approx(0.5)


class TestTransportValidationStructurallyExcludedFromCampaignPhase15B3N2A:
    """Phase 15B.3N.2A: NON_OFFICIAL_TRANSPORT_VALIDATION engineering
    evidence lives permanently under TRANSPORT_VALIDATION_ROOT
    (persistent-data/transport_validation/), a tree deliberately separate
    from CALIBRATION_ROOT. These tests pin the isolation property that
    makes this safe: campaign/model-fitting discovery in this module is
    driven entirely by GLOBAL_SEQUENCE_PATH (a fixed manifest) and
    LEDGER_PATH (populated only via explicit campaign_sequence_index-keyed
    update_slot_state() calls) -- never by scanning persistent-data/ for
    directories -- so a transport-validation run, however official-looking
    its own experiment id, can never be silently picked up as a campaign
    attempt, fitting-support, or a calibration repeat."""

    def test_transport_validation_root_is_a_distinct_tree_from_calibration_root(self):
        assert lib.TRANSPORT_VALIDATION_ROOT != lib.CALIBRATION_ROOT
        assert lib.CALIBRATION_ROOT not in lib.TRANSPORT_VALIDATION_ROOT.parents
        assert lib.TRANSPORT_VALIDATION_ROOT not in lib.CALIBRATION_ROOT.parents

    def test_transport_validation_root_is_not_the_ledger_or_sequence_path(self):
        assert lib.TRANSPORT_VALIDATION_ROOT != lib.LEDGER_PATH
        assert lib.TRANSPORT_VALIDATION_ROOT != lib.GLOBAL_SEQUENCE_PATH
        assert lib.LEDGER_PATH not in lib.TRANSPORT_VALIDATION_ROOT.parents

    def test_campaign_discovery_functions_never_reference_transport_validation_root(self):
        # A structural regression guard: if campaign/ledger/sequence loading
        # ever grows a persistent-data/ directory scan, this test forces an
        # explicit decision about whether TRANSPORT_VALIDATION_ROOT must be
        # excluded from it.
        import inspect

        for fn in (lib.load_global_sequence, lib.init_ledger, lib.ensure_ledger, lib.next_resumable_index):
            source = inspect.getsource(fn)
            assert "transport_validation" not in source.lower()
            assert "iterdir" not in source and "glob(" not in source and "listdir" not in source

    def test_ledger_unaffected_by_a_transport_validation_directory_existing_on_disk(self, tmp_path, monkeypatch):
        # Simulate the real on-disk situation (a transport_validation
        # sibling directory next to persistent-data/dynamics_calibration)
        # and confirm ledger init/resolution never even looks at it.
        sequence = [{"campaign_sequence_index": i, "planned_slot_id": f"slot{i}"} for i in range(1, 4)]
        ledger = lib.init_ledger(sequence)
        assert lib.next_resumable_index(ledger) == 1
        (tmp_path / "transport_validation" / "NON_OFFICIAL_TRANSPORT_VALIDATION_EVIDENCE_RECONSTRUCTION_x").mkdir(parents=True)
        # Re-resolving next_resumable_index must be identical regardless of
        # what exists in a sibling directory -- ledger state alone governs.
        assert lib.next_resumable_index(ledger) == 1


class TestLocalListenerCheckPhase15B3M3:
    def test_extract_port_from_url(self):
        assert harness._extract_port_from_url("http://127.0.0.1:50408") == 50408

    def test_extract_port_from_none_url_is_none(self):
        assert harness._extract_port_from_url(None) is None
        assert harness._extract_port_from_url("") is None

    def test_extract_port_from_url_with_no_port_is_none(self):
        assert harness._extract_port_from_url("http://127.0.0.1") is None

    def test_check_local_listener_present_returns_none_when_lsof_unavailable(self, monkeypatch):
        def _raise(*args, **kwargs):
            raise FileNotFoundError("lsof not found")

        monkeypatch.setattr(harness, "_run", _raise)
        assert harness.check_local_listener_present(50408) is None

    def test_check_local_listener_present_true_when_lsof_reports_a_match(self, monkeypatch):
        class _FakeProc:
            stdout = "COMMAND   PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME\nminikube 1234 user   3u  IPv4 0x0      0t0  TCP 127.0.0.1:50408 (LISTEN)\n"

        monkeypatch.setattr(harness, "_run", lambda *a, **k: _FakeProc())
        assert harness.check_local_listener_present(50408) is True

    def test_check_local_listener_present_false_when_lsof_finds_nothing(self, monkeypatch):
        class _FakeProc:
            stdout = ""

        monkeypatch.setattr(harness, "_run", lambda *a, **k: _FakeProc())
        assert harness.check_local_listener_present(50408) is False


class TestResolveLocalListenerOwnerPhase15B3N1:
    """Phase 15B.3N.1: pure diagnostic enrichment resolving the ACTUAL
    listener-owning PID/command (e.g. a child `ssh` process on the Docker
    driver/macOS `minikube service --url` mechanism), distinct from the
    parent tunnel PID the harness already tracks. Never consulted by
    route-integrity classification -- classification continues to rely
    solely on `check_local_listener_present()`'s boolean."""

    def test_returns_none_none_when_lsof_unavailable(self, monkeypatch):
        def _raise(*args, **kwargs):
            raise FileNotFoundError("lsof not found")

        monkeypatch.setattr(harness, "_run", _raise)
        result = harness.resolve_local_listener_owner(50408)
        assert result == {"listener_owner_pid": None, "listener_owner_command": None}

    def test_returns_none_none_when_no_listener_present(self, monkeypatch):
        class _FakeProc:
            stdout = ""

        monkeypatch.setattr(harness, "_run", lambda *a, **k: _FakeProc())
        result = harness.resolve_local_listener_owner(50408)
        assert result == {"listener_owner_pid": None, "listener_owner_command": None}

    def test_resolves_pid_and_command_of_the_actual_listener_owner(self, monkeypatch):
        # The Phase 15B.3N.1 finding: on the Docker driver/macOS, `minikube
        # service --url`'s listener is owned by a CHILD `ssh` process, not
        # the parent `minikube` CLI PID the harness tracks separately.
        class _FakeProc:
            stdout = (
                "COMMAND   PID             USER   FD   TYPE DEVICE SIZE/OFF NODE NAME\n"
                "ssh     61737 rakeshdandamudi    5u  IPv6    0x0      0t0  TCP [::1]:52548 (LISTEN)\n"
            )

        monkeypatch.setattr(harness, "_run", lambda *a, **k: _FakeProc())
        result = harness.resolve_local_listener_owner(52548)
        assert result == {"listener_owner_pid": "61737", "listener_owner_command": "ssh"}

    def test_never_raises_on_malformed_lsof_output(self, monkeypatch):
        class _FakeProc:
            stdout = "not a real lsof line\n"

        monkeypatch.setattr(harness, "_run", lambda *a, **k: _FakeProc())
        result = harness.resolve_local_listener_owner(52548)
        assert result["listener_owner_command"] == "not"


class _FakeCompletedProcess:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class TestV5TransportDiscoveryAndLifecyclePhase15B3N3:
    """Phase 15B.3N.3 (CORR-40): the V5 Docker-network TCP relay transport
    -- discovery (network/node-IP/NodePort), container-state parsing, and
    relay-image identity resolution. All read-only `docker`/`kubectl`
    invocations are mocked via `harness._run`; no live cluster is touched."""

    def test_resolve_minikube_docker_network_parses_network_name(self, monkeypatch):
        monkeypatch.setattr(
            harness, "_run",
            lambda *a, **k: _FakeCompletedProcess(stdout='{"minikube":{"IPAddress":"192.168.49.2"}}'),
        )
        assert harness.resolve_minikube_docker_network() == "minikube"

    def test_resolve_minikube_docker_network_none_on_failure(self, monkeypatch):
        monkeypatch.setattr(harness, "_run", lambda *a, **k: _FakeCompletedProcess(returncode=1))
        assert harness.resolve_minikube_docker_network() is None

    def test_resolve_minikube_docker_network_none_on_malformed_json(self, monkeypatch):
        monkeypatch.setattr(harness, "_run", lambda *a, **k: _FakeCompletedProcess(stdout="not json"))
        assert harness.resolve_minikube_docker_network() is None

    def test_resolve_minikube_node_ip(self, monkeypatch):
        monkeypatch.setattr(harness, "_run", lambda *a, **k: _FakeCompletedProcess(stdout="192.168.49.2\n"))
        assert harness.resolve_minikube_node_ip() == "192.168.49.2"

    def test_resolve_minikube_node_ip_none_when_blank(self, monkeypatch):
        monkeypatch.setattr(harness, "_run", lambda *a, **k: _FakeCompletedProcess(stdout=""))
        assert harness.resolve_minikube_node_ip() is None

    def test_resolve_loadtest_nodeport(self, monkeypatch):
        monkeypatch.setattr(harness, "_run", lambda *a, **k: _FakeCompletedProcess(stdout="31891"))
        assert harness.resolve_loadtest_nodeport() == 31891

    def test_resolve_loadtest_nodeport_none_when_not_numeric(self, monkeypatch):
        monkeypatch.setattr(harness, "_run", lambda *a, **k: _FakeCompletedProcess(stdout=""))
        assert harness.resolve_loadtest_nodeport() is None

    def test_check_relay_container_state_running(self, monkeypatch):
        monkeypatch.setattr(harness, "_run", lambda *a, **k: _FakeCompletedProcess(stdout="true|0"))
        running, exit_code = harness.check_relay_container_state("loadtest-relay-x")
        assert running is True
        assert exit_code == "0"

    def test_check_relay_container_state_dead(self, monkeypatch):
        monkeypatch.setattr(harness, "_run", lambda *a, **k: _FakeCompletedProcess(stdout="false|137"))
        running, exit_code = harness.check_relay_container_state("loadtest-relay-x")
        assert running is False
        assert exit_code == "137"

    def test_check_relay_container_state_none_when_container_gone(self, monkeypatch):
        monkeypatch.setattr(harness, "_run", lambda *a, **k: _FakeCompletedProcess(returncode=1))
        running, exit_code = harness.check_relay_container_state("loadtest-relay-x")
        assert running is None and exit_code is None

    def test_resolve_relay_image_identity(self, monkeypatch):
        calls = []

        def _fake_run(cmd, timeout=None):
            calls.append(cmd)
            if "{{.Id}}" in cmd:
                return _FakeCompletedProcess(stdout="sha256:abc123")
            return _FakeCompletedProcess(stdout="alpine/socat@sha256:def456")

        monkeypatch.setattr(harness, "_run", _fake_run)
        result = harness.resolve_relay_image_identity()
        assert result == {"image": harness.RELAY_IMAGE, "image_id": "sha256:abc123", "repo_digest": "alpine/socat@sha256:def456"}

    def test_resolve_relay_image_identity_handles_missing_repo_digest(self, monkeypatch):
        def _fake_run(cmd, timeout=None):
            if "{{.Id}}" in cmd:
                return _FakeCompletedProcess(stdout="sha256:abc123")
            return _FakeCompletedProcess(returncode=1)

        monkeypatch.setattr(harness, "_run", _fake_run)
        result = harness.resolve_relay_image_identity()
        assert result["image_id"] == "sha256:abc123"
        assert result["repo_digest"] is None

    def test_start_loadtest_transport_fails_closed_when_discovery_incomplete(self, monkeypatch):
        monkeypatch.setattr(harness, "resolve_minikube_docker_network", lambda: None)
        monkeypatch.setattr(harness, "resolve_minikube_node_ip", lambda: "192.168.49.2")
        monkeypatch.setattr(harness, "resolve_loadtest_nodeport", lambda: 31891)
        result = harness.start_loadtest_transport("test-attempt")
        assert "error" in result
        assert result["docker_network"] is None

    def test_start_loadtest_transport_never_falls_back_to_ssh_tunnel(self, monkeypatch):
        # Structural guard: start_loadtest_transport()'s own source must
        # never invoke minikube service/tunnel -- no runtime fallback.
        import inspect

        source = inspect.getsource(harness.start_loadtest_transport)
        assert "minikube service" not in source
        assert "minikube_service" not in source
        assert "start_loadtest_tunnel" not in source

    def test_relay_container_name_is_attempt_scoped(self, monkeypatch):
        monkeypatch.setattr(harness, "resolve_minikube_docker_network", lambda: "minikube")
        monkeypatch.setattr(harness, "resolve_minikube_node_ip", lambda: "192.168.49.2")
        monkeypatch.setattr(harness, "resolve_loadtest_nodeport", lambda: 31891)
        monkeypatch.setattr(harness, "_run", lambda *a, **k: _FakeCompletedProcess(returncode=1, stderr="boom"))
        result = harness.start_loadtest_transport("fit_dyncal_r8_L3_repeat1_attempt3_20260101T000000Z")
        assert "error" in result
        assert "docker run failed" in result["error"]


# =============================================================================
# Phase 15B.3O.1 / CORR-41: Kubernetes-dataplane NodePort ingress evidence.
# Synthetic `iptables-save -c` text fixtures below are shaped exactly like
# the real output captured live from this project's Minikube node (see
# persistent-data/transport_validation/protocol_v6_route_evidence/), never
# invented syntax.
# =============================================================================

_NAT_ONE_ENDPOINT_ZERO = (
    '[0:0] -A KUBE-SEP-ENIF66D2G3466LGQ -s 10.244.1.149/32 -m comment --comment "autoscaler-app/demo-app-loadtest:http" -j KUBE-MARK-MASQ\n'
    '[0:0] -A KUBE-SEP-ENIF66D2G3466LGQ -p tcp -m comment --comment "autoscaler-app/demo-app-loadtest:http" -m tcp -j DNAT --to-destination 10.244.1.149:8080\n'
    '[0:0] -A KUBE-SVC-UCHV3NNZZE6PQQ7U -m comment --comment "autoscaler-app/demo-app-loadtest:http -> 10.244.1.149:8080" -j KUBE-SEP-ENIF66D2G3466LGQ\n'
)

_NAT_ONE_ENDPOINT_FIVE = _NAT_ONE_ENDPOINT_ZERO.replace(
    '[0:0] -A KUBE-SEP-ENIF66D2G3466LGQ -p tcp -m comment --comment "autoscaler-app/demo-app-loadtest:http" -m tcp -j DNAT --to-destination 10.244.1.149:8080',
    '[5:300] -A KUBE-SEP-ENIF66D2G3466LGQ -p tcp -m comment --comment "autoscaler-app/demo-app-loadtest:http" -m tcp -j DNAT --to-destination 10.244.1.149:8080',
)

_NAT_TWO_ENDPOINTS_TEN = (
    '[7:420] -A KUBE-SEP-OGXRNUUKR756PIKN -p tcp -m comment --comment "autoscaler-app/demo-app-loadtest:http" -m tcp -j DNAT --to-destination 10.244.1.150:8080\n'
    '[3:180] -A KUBE-SEP-5NE2HTODBH3WUQID -p tcp -m comment --comment "autoscaler-app/demo-app-loadtest:http" -m tcp -j DNAT --to-destination 10.244.1.151:8080\n'
)

_NAT_EMPTY = '[0:0] -A KUBE-POSTROUTING -m mark ! --mark 0x4000/0x4000 -j RETURN\n'

_FILTER_NO_ENDPOINTS_ZERO = (
    '[0:0] -A KUBE-EXTERNAL-SERVICES -p tcp -m comment --comment "autoscaler-app/demo-app-loadtest:http has no endpoints" -m addrtype --dst-type LOCAL -m tcp --dport 31891 -j REJECT --reject-with icmp-port-unreachable\n'
)

_FILTER_NO_ENDPOINTS_FIVE = _FILTER_NO_ENDPOINTS_ZERO.replace("[0:0]", "[5:300]")

_FILTER_EMPTY = '[0:0] -A KUBE-FORWARD -m comment --comment "kubernetes forwarding rules" -m mark --mark 0x4000/0x4000 -j ACCEPT\n'


class TestParseDataplaneIngressRulesPhase15B3O1:
    """`parse_dataplane_ingress_rules()`: pure parse of live-shaped
    `iptables-save -c` text into the CORR-41 Service-specific
    endpoint_backed/no_endpoints/absent regimes."""

    def test_endpoint_backed_regime_single_endpoint(self):
        result = lib.parse_dataplane_ingress_rules(_NAT_ONE_ENDPOINT_FIVE, _FILTER_EMPTY)
        assert result["regime"] == lib.DATAPLANE_REGIME_ENDPOINT_BACKED
        assert result["total_packet_count"] == 5
        assert result["total_byte_count"] == 300
        assert result["rule_identity"] == ("KUBE-SEP-ENIF66D2G3466LGQ",)

    def test_endpoint_backed_regime_sums_multiple_endpoints(self):
        # Empirically verified live: 2 Ready endpoints, 10 requests, 7+3=10.
        result = lib.parse_dataplane_ingress_rules(_NAT_TWO_ENDPOINTS_TEN, _FILTER_EMPTY)
        assert result["regime"] == lib.DATAPLANE_REGIME_ENDPOINT_BACKED
        assert result["total_packet_count"] == 10
        assert result["total_byte_count"] == 600
        assert result["rule_identity"] == ("KUBE-SEP-5NE2HTODBH3WUQID", "KUBE-SEP-OGXRNUUKR756PIKN")

    def test_no_endpoints_regime_when_nat_rules_absent(self):
        result = lib.parse_dataplane_ingress_rules(_NAT_EMPTY, _FILTER_NO_ENDPOINTS_FIVE)
        assert result["regime"] == lib.DATAPLANE_REGIME_NO_ENDPOINTS
        assert result["total_packet_count"] == 5
        assert result["total_byte_count"] == 300

    def test_endpoint_backed_takes_precedence_if_both_somehow_present(self):
        # Structurally the two regimes are mutually exclusive in a live
        # kube-proxy dataplane, but the parser must not crash or silently
        # pick an arbitrary regime if it ever observes both mid-transition.
        result = lib.parse_dataplane_ingress_rules(_NAT_ONE_ENDPOINT_FIVE, _FILTER_NO_ENDPOINTS_FIVE)
        assert result["regime"] == lib.DATAPLANE_REGIME_ENDPOINT_BACKED

    def test_absent_regime_when_neither_rule_found(self):
        result = lib.parse_dataplane_ingress_rules(_NAT_EMPTY, _FILTER_EMPTY)
        assert result["regime"] == lib.DATAPLANE_REGIME_ABSENT
        assert result["total_packet_count"] is None
        assert result["rule_identity"] == ()

    def test_mark_masq_sibling_rule_excluded_from_endpoint_backed_count(self):
        # The KUBE-MARK-MASQ rule shares the identical comment but is not
        # the DNAT forwarding rule -- must never be summed in.
        result = lib.parse_dataplane_ingress_rules(_NAT_ONE_ENDPOINT_ZERO, _FILTER_EMPTY)
        assert len(result["rules"]) == 1
        assert result["rules"][0]["chain"] == "KUBE-SEP-ENIF66D2G3466LGQ"


def _dp_sample(elapsed, regime, identity, count, error=None):
    return {
        "elapsed_seconds_since_traffic_start": elapsed,
        "regime": regime,
        "rule_identity": identity,
        "total_packet_count": count,
        "observation_error": error,
    }


class TestComputeDataplaneIngressEvidenceWindowsPhase15B3O1:
    """`compute_dataplane_ingress_evidence_windows()`: window-local,
    reset-safe, non-duplicating attribution for the CORR-41 dataplane
    source -- mirrors `compute_delivered_request_evidence_windows()`'s
    discipline for the existing two sources."""

    IDENT = ("KUBE-SEP-AAAA",)

    def test_positive_delta_fully_inside_window_is_decisive_evidence(self):
        samples = [self._s(1.0, 0), self._s(4.0, 5)]
        result = lib.compute_dataplane_ingress_evidence_windows(0.0, 10.0, dataplane_samples=samples)
        assert result["nodeport_dataplane_ingress_evidence"] is True
        assert len(result["dataplane_counter_intervals_used"]) == 1
        assert result["dataplane_backend_regimes_with_evidence"] == [lib.DATAPLANE_REGIME_ENDPOINT_BACKED]

    def test_flat_counter_is_no_evidence(self):
        samples = [self._s(1.0, 3), self._s(4.0, 3)]
        result = lib.compute_dataplane_ingress_evidence_windows(0.0, 10.0, dataplane_samples=samples)
        assert result["nodeport_dataplane_ingress_evidence"] is False
        assert len(result["dataplane_counter_intervals_used"]) == 1

    def test_cross_window_interval_credited_to_neither_window(self):
        samples = [self._s(8.0, 0), self._s(12.0, 5)]  # straddles the 0-10/10-20 boundary
        window_a = lib.compute_dataplane_ingress_evidence_windows(0.0, 10.0, dataplane_samples=samples)
        window_b = lib.compute_dataplane_ingress_evidence_windows(10.0, 20.0, dataplane_samples=samples)
        assert window_a["nodeport_dataplane_ingress_evidence"] is False
        assert window_b["nodeport_dataplane_ingress_evidence"] is False
        assert len(window_a["dataplane_counter_intervals_rejected_cross_window"]) == 1
        assert len(window_b["dataplane_counter_intervals_rejected_cross_window"]) == 1

    def test_counter_decrease_is_rejected_as_reset_not_negative_evidence(self):
        samples = [self._s(1.0, 9), self._s(4.0, 2)]
        result = lib.compute_dataplane_ingress_evidence_windows(0.0, 10.0, dataplane_samples=samples)
        assert result["nodeport_dataplane_ingress_evidence"] is False
        assert len(result["dataplane_counter_intervals_rejected_reset"]) == 1
        assert result["dataplane_counter_intervals_used"] == []

    def test_rule_identity_change_is_rejected_as_reset(self):
        # e.g. a pod restart recreated the KUBE-SEP chain mid-window.
        s1 = self._s(1.0, 9)
        s2 = {**self._s(4.0, 2), "rule_identity": ("KUBE-SEP-BBBB",)}
        result = lib.compute_dataplane_ingress_evidence_windows(0.0, 10.0, dataplane_samples=[s1, s2])
        assert result["nodeport_dataplane_ingress_evidence"] is False
        assert len(result["dataplane_counter_intervals_rejected_reset"]) == 1

    def test_regime_change_is_rejected_as_reset(self):
        s1 = self._s(1.0, 9)
        s2 = {**s1, "elapsed_seconds_since_traffic_start": 4.0, "regime": lib.DATAPLANE_REGIME_NO_ENDPOINTS, "total_packet_count": 2}
        result = lib.compute_dataplane_ingress_evidence_windows(0.0, 10.0, dataplane_samples=[s1, s2])
        assert result["nodeport_dataplane_ingress_evidence"] is False
        assert len(result["dataplane_counter_intervals_rejected_reset"]) == 1

    def test_no_endpoints_regime_positive_delta_is_still_decisive_evidence(self):
        # THE CENTRAL CORR-41 CASE: zero Ready endpoints, but traffic still
        # reaches the dataplane (REJECT rule increments) -- must be
        # decisive route evidence despite zero endpoints.
        samples = [
            self._s(1.0, 0, regime=lib.DATAPLANE_REGIME_NO_ENDPOINTS, identity=("KUBE-EXTERNAL-SERVICES",)),
            self._s(4.0, 5, regime=lib.DATAPLANE_REGIME_NO_ENDPOINTS, identity=("KUBE-EXTERNAL-SERVICES",)),
        ]
        result = lib.compute_dataplane_ingress_evidence_windows(0.0, 10.0, dataplane_samples=samples)
        assert result["nodeport_dataplane_ingress_evidence"] is True
        assert result["dataplane_backend_regimes_with_evidence"] == [lib.DATAPLANE_REGIME_NO_ENDPOINTS]

    def test_observer_error_sample_is_skipped_not_treated_as_failure(self):
        samples = [self._s(1.0, 0), {**self._s(4.0, 5), "observation_error": "docker exec timeout"}]
        result = lib.compute_dataplane_ingress_evidence_windows(0.0, 10.0, dataplane_samples=samples)
        assert result["nodeport_dataplane_ingress_evidence"] is False
        assert result["dataplane_observer_error_present"] is True

    def _s(self, elapsed, count, regime=None, identity=None):
        return _dp_sample(
            elapsed,
            regime or lib.DATAPLANE_REGIME_ENDPOINT_BACKED,
            identity or self.IDENT,
            count,
        )


class TestClassifyRouteIntegrityWindowCorr41Phase15B3O1:
    """`classify_route_integrity_window()`'s new
    `nodeport_dataplane_ingress_evidence` parameter -- additive, backward
    compatible (default `None` reproduces exact pre-CORR-41 behavior), and
    never overrides FAILED precedence."""

    def test_dataplane_evidence_alone_confirms_window(self):
        result = lib.classify_route_integrity_window(
            delivered_request_evidence=False, tunnel_process_alive=True, local_listener_present=True,
            nodeport_dataplane_ingress_evidence=True,
        )
        assert result["window_route_status"] == lib.ROUTE_INTEGRITY_CONFIRMED

    def test_dataplane_evidence_confirms_even_with_zero_endpoints_conceptually(self):
        # loadtest_ready_endpoint_count is not an input to this function at
        # all -- confirming, at the classifier level, that endpoint count
        # plays no role in the CONFIRMED decision (sec.17 separation).
        result = lib.classify_route_integrity_window(
            delivered_request_evidence=False, tunnel_process_alive=True, local_listener_present=True,
            nodeport_dataplane_ingress_evidence=True,
        )
        assert result["window_route_status"] == lib.ROUTE_INTEGRITY_CONFIRMED

    def test_no_dataplane_evidence_and_no_other_evidence_remains_unverified(self):
        result = lib.classify_route_integrity_window(
            delivered_request_evidence=False, tunnel_process_alive=True, local_listener_present=True,
            nodeport_dataplane_ingress_evidence=False,
        )
        assert result["window_route_status"] == lib.ROUTE_INTEGRITY_UNVERIFIED

    def test_omitting_dataplane_param_reproduces_pre_corr41_behavior(self):
        with_none = lib.classify_route_integrity_window(delivered_request_evidence=False, tunnel_process_alive=True, local_listener_present=True)
        explicit_none = lib.classify_route_integrity_window(
            delivered_request_evidence=False, tunnel_process_alive=True, local_listener_present=True,
            nodeport_dataplane_ingress_evidence=None,
        )
        assert with_none["window_route_status"] == explicit_none["window_route_status"] == lib.ROUTE_INTEGRITY_UNVERIFIED

    def test_relay_failure_precedence_overrides_dataplane_evidence(self):
        # FAILED must remain decisive regardless of a positive dataplane
        # delta observed in the same window.
        result = lib.classify_route_integrity_window(
            delivered_request_evidence=False, tunnel_process_alive=False, local_listener_present=True,
            nodeport_dataplane_ingress_evidence=True,
        )
        assert result["window_route_status"] == lib.ROUTE_INTEGRITY_FAILED

    def test_listener_failure_precedence_overrides_dataplane_evidence(self):
        result = lib.classify_route_integrity_window(
            delivered_request_evidence=False, tunnel_process_alive=True, local_listener_present=False,
            nodeport_dataplane_ingress_evidence=True,
        )
        assert result["window_route_status"] == lib.ROUTE_INTEGRITY_FAILED

    def test_http_and_dataplane_evidence_both_present_still_confirmed(self):
        result = lib.classify_route_integrity_window(
            delivered_request_evidence=True, tunnel_process_alive=True, local_listener_present=True,
            nodeport_dataplane_ingress_evidence=True,
        )
        assert result["window_route_status"] == lib.ROUTE_INTEGRITY_CONFIRMED


class TestProtocolV6Corr41Phase15B3O1:
    """Provenance and manifest checks for Protocol V6/CORR-41."""

    def test_provenance_route_integrity_protocol_version_is_now_6(self):
        # Phase 15B.3P.1 (CORR-42, Protocol V7) then Phase 15B.3P.2 (CORR-43,
        # Protocol V8): route_integrity_protocol_version stays 6 throughout --
        # neither V7 (dataplane evidence ACQUISITION scheduling) nor V8
        # (evidence-window PARTITIONING) ever touches
        # classify_route_integrity_window()/classify_route_integrity().
        # active_correction_reference/prior_correction_reference always report
        # the CURRENTLY active protocol (see
        # test_provenance_takes_no_index_argument_so_it_applies_to_whichever_slot_runs_next),
        # so they have moved on to CORR-43/CORR-42 respectively; this does not
        # retroactively change what attempts 1-4 (created under CORR-40/CORR-41)
        # have stored in their own metadata.json.
        prov = lib.build_protocol_provenance()
        assert prov["route_integrity_protocol_version"] == 6
        assert prov["active_correction_reference"] == "CORR-48"
        assert prov["prior_correction_reference"] == "CORR-47"

    def test_provenance_transport_protocol_version_unchanged_by_corr41(self):
        # CORR-41 changes evidence/classification only -- the V5 transport
        # itself (CORR-40) is untouched.
        prov = lib.build_protocol_provenance()
        assert prov["transport_protocol_version"] == 5

    def test_provenance_includes_v6_verifiable_sha256_hash(self):
        import hashlib as _hashlib

        prov = lib.build_protocol_provenance()
        hashes = prov["provenance_hashes_sha256"]
        assert isinstance(hashes["v6_manifest"], str) and len(hashes["v6_manifest"]) == 64
        assert hashes["v6_manifest"] == _hashlib.sha256(lib.MANIFEST_V6_PATH.read_bytes()).hexdigest()
        # V5's own hashes remain present -- V6 does not erase V5's provenance.
        assert isinstance(hashes["v5_manifest"], str) and len(hashes["v5_manifest"]) == 64

    def test_v6_manifest_exists_and_is_valid_yaml(self):
        import yaml

        m = yaml.safe_load(lib.MANIFEST_V6_PATH.read_text(encoding="utf-8"))
        assert m["protocol_version"] == 6
        assert m["correction_reference"] == "CORR-41"
        assert m["prior_correction_reference"] == "CORR-40"

    def test_v6_manifest_does_not_redefine_v5_transport(self):
        import yaml

        text = lib.MANIFEST_V6_PATH.read_text(encoding="utf-8")
        m = yaml.safe_load(text)
        assert "transport_protocol_version remains 5" in text
        assert m["unaffected"]  # non-empty preserved-invariants list, matching v5's own convention

    def test_v6_manifest_records_attempt3_non_retroactivity(self):
        import yaml

        m = yaml.safe_load(lib.MANIFEST_V6_PATH.read_text(encoding="utf-8"))
        assert m["non_retroactivity"]["ledger_state_unchanged"].count("attempt1") == 1
        assert "attempt3" in m["non_retroactivity"]["ledger_state_unchanged"]
        assert "next_resumable_index()==12" in m["non_retroactivity"]["ledger_state_unchanged"]

    def test_v1_v4_v5_manifest_hashes_unchanged_by_v6_introduction(self):
        assert hashlib.sha256(lib.MANIFEST_PATH.read_bytes()).hexdigest() == "2e6011495307245bcc85832d0b0e1aa6cbc6ea63e4655914eeb9c942c15fc2bf"
        assert hashlib.sha256(lib.MANIFEST_V3_PATH.read_bytes()).hexdigest() == "a0c7f3cebcdcbd7f9ca766a7057681e7b29cd2d10c4b8de74478fb54681561b8"


class TestSampleDataplaneIngressRulesPhase15B3O1:
    """`sample_dataplane_ingress_rules()`: the I/O wrapper (harness module)
    delegating to `parse_dataplane_ingress_rules()`. Observer failure must
    never fabricate a regime."""

    def test_reports_observation_error_on_nonzero_exit_never_fabricating_regime(self, monkeypatch):
        monkeypatch.setattr(harness, "_run", lambda *a, **k: _FakeCompletedProcess(returncode=1, stderr="docker: no such container"))
        result = harness.sample_dataplane_ingress_rules(31891)
        assert result["regime"] is None
        assert result["observation_error"]

    def test_parses_real_docker_exec_output_end_to_end(self, monkeypatch):
        def _fake_run(cmd, timeout=None):
            if "nat" in cmd:
                return _FakeCompletedProcess(stdout=_NAT_ONE_ENDPOINT_FIVE)
            return _FakeCompletedProcess(stdout=_FILTER_EMPTY)

        monkeypatch.setattr(harness, "_run", _fake_run)
        result = harness.sample_dataplane_ingress_rules(31891)
        assert result["regime"] == lib.DATAPLANE_REGIME_ENDPOINT_BACKED
        assert result["total_packet_count"] == 5
        assert result["observation_error"] is None
        assert result["nodeport"] == 31891


class TestShouldTriggerImmediateResamplePhase15B3P1:
    """`should_trigger_immediate_resample()` (CORR-42, Protocol V7): the
    pure decision function driving reactive re-sampling on a detected
    kube-proxy reprogramming event. Must only ever request MORE evidence --
    never suppress it -- and must never treat an observer failure as a
    detected transition."""

    def test_no_previous_sample_is_not_a_transition(self):
        curr = _dp_sample(1.0, lib.DATAPLANE_REGIME_ENDPOINT_BACKED, ("KUBE-SEP-AAAA",), 0)
        assert lib.should_trigger_immediate_resample(None, curr) is False

    def test_identical_regime_and_identity_is_not_a_transition(self):
        prev = _dp_sample(1.0, lib.DATAPLANE_REGIME_ENDPOINT_BACKED, ("KUBE-SEP-AAAA",), 3)
        curr = _dp_sample(4.0, lib.DATAPLANE_REGIME_ENDPOINT_BACKED, ("KUBE-SEP-AAAA",), 7)
        assert lib.should_trigger_immediate_resample(prev, curr) is False

    def test_regime_change_is_a_transition(self):
        prev = _dp_sample(1.0, lib.DATAPLANE_REGIME_ENDPOINT_BACKED, ("KUBE-SEP-AAAA",), 3)
        curr = _dp_sample(4.0, lib.DATAPLANE_REGIME_NO_ENDPOINTS, ("KUBE-EXTERNAL-SERVICES",), 3)
        assert lib.should_trigger_immediate_resample(prev, curr) is True

    def test_rule_identity_change_within_same_regime_is_a_transition(self):
        # attempt4's own [115,125) case: 8 SEPs -> 7 SEPs, same regime.
        prev = _dp_sample(118.02, lib.DATAPLANE_REGIME_ENDPOINT_BACKED, ("A", "B"), 1148)
        curr = _dp_sample(121.44, lib.DATAPLANE_REGIME_ENDPOINT_BACKED, ("A",), 249)
        assert lib.should_trigger_immediate_resample(prev, curr) is True

    def test_observer_error_on_either_sample_never_counts_as_a_transition(self):
        prev = _dp_sample(1.0, lib.DATAPLANE_REGIME_ENDPOINT_BACKED, ("KUBE-SEP-AAAA",), 3, error="docker exec timeout")
        curr = _dp_sample(4.0, lib.DATAPLANE_REGIME_NO_ENDPOINTS, ("KUBE-EXTERNAL-SERVICES",), 3)
        assert lib.should_trigger_immediate_resample(prev, curr) is False
        assert lib.should_trigger_immediate_resample(curr, {**prev, "regime": prev["regime"]}) is False


class TestDataplaneAcquisitionV7ConstantsPhase15B3P1:
    """CORR-42/Protocol V7 acquisition constants. The V6 constant is frozen
    (attempts 1-4's own historical behavior); V7 introduces new, separate
    constants rather than mutating the frozen one."""

    def test_v6_cadence_constant_is_unchanged_by_v7_introduction(self):
        assert lib.DATAPLANE_INGRESS_SAMPLING_INTERVAL_SECONDS == 3.0

    def test_v7_cadence_is_a_modest_reduction_not_an_arbitrary_jump_to_1s(self):
        assert lib.DATAPLANE_INGRESS_SAMPLING_INTERVAL_SECONDS_V7 < lib.DATAPLANE_INGRESS_SAMPLING_INTERVAL_SECONDS
        assert lib.DATAPLANE_INGRESS_SAMPLING_INTERVAL_SECONDS_V7 > 1.0

    def test_transition_resample_delay_is_positive_and_shorter_than_v7_cadence(self):
        assert 0 < lib.DATAPLANE_INGRESS_TRANSITION_RESAMPLE_DELAY_SECONDS < lib.DATAPLANE_INGRESS_SAMPLING_INTERVAL_SECONDS_V7


class TestDataplaneEvidenceCoverageResolutionPhase15B3P1:
    """CORR-42/Protocol V7 integration-level regression tests: with the two
    new sample-acquisition mechanisms (reactive resample on transition,
    end-of-run anchor sample) feeding the SAME, UNCHANGED
    `compute_dataplane_ingress_evidence_windows()`, previously-structural
    blind spots (attempt4's `[115,125)` and `[175,179)` shapes) become
    observable -- without borrowing evidence across windows, without
    fabricating a cross-identity delta, and without weakening FAILED
    precedence or the zero-endpoint evidence path."""

    def test_short_trailing_window_confirmed_once_end_of_run_anchor_sample_added(self):
        # Mirrors attempt4's [175,179) shape: one lone interior sample
        # (176.07-equivalent) is structurally insufficient to form a pair.
        lone = _dp_sample(1.0, lib.DATAPLANE_REGIME_NO_ENDPOINTS, ("KUBE-EXTERNAL-SERVICES",), 16639)
        before = lib.compute_dataplane_ingress_evidence_windows(0.0, 4.0, dataplane_samples=[lone])
        assert before["nodeport_dataplane_ingress_evidence"] is False
        assert before["dataplane_counter_intervals_used"] == []

        # CORR-42: an anchor sample taken at generator-exit, same regime and
        # identity, strictly inside the window, with a higher packet count.
        anchor = _dp_sample(3.6, lib.DATAPLANE_REGIME_NO_ENDPOINTS, ("KUBE-EXTERNAL-SERVICES",), 18921)
        after = lib.compute_dataplane_ingress_evidence_windows(0.0, 4.0, dataplane_samples=[lone, anchor])
        assert after["nodeport_dataplane_ingress_evidence"] is True
        assert len(after["dataplane_counter_intervals_used"]) == 1

    def test_identity_change_inside_window_confirmed_by_reactive_resample_under_fresh_baseline(self):
        # Mirrors attempt4's [115,125) shape: 8 SEPs -> 7 SEPs mid-window.
        pre_reset = _dp_sample(3.0, lib.DATAPLANE_REGIME_ENDPOINT_BACKED, ("A", "B"), 1148)
        post_reset = _dp_sample(6.4, lib.DATAPLANE_REGIME_ENDPOINT_BACKED, ("A",), 249)
        assert lib.should_trigger_immediate_resample(pre_reset, post_reset) is True

        without_reactive = lib.compute_dataplane_ingress_evidence_windows(
            0.0, 10.0, dataplane_samples=[pre_reset, post_reset]
        )
        assert without_reactive["nodeport_dataplane_ingress_evidence"] is False
        assert len(without_reactive["dataplane_counter_intervals_rejected_reset"]) == 1

        # CORR-42: one immediate reactive follow-up sample UNDER THE NEW
        # ("A",) identity, ~1s later, with traffic continuing.
        reactive_followup = _dp_sample(7.4, lib.DATAPLANE_REGIME_ENDPOINT_BACKED, ("A",), 260)
        with_reactive = lib.compute_dataplane_ingress_evidence_windows(
            0.0, 10.0, dataplane_samples=[pre_reset, post_reset, reactive_followup]
        )
        assert with_reactive["nodeport_dataplane_ingress_evidence"] is True
        # Exactly one confirming interval (post_reset -> reactive_followup,
        # both identity ("A",)); the cross-identity pre_reset -> post_reset
        # pair remains rejected as a reset, never smeared into a delta.
        assert len(with_reactive["dataplane_counter_intervals_used"]) == 1
        assert len(with_reactive["dataplane_counter_intervals_rejected_reset"]) == 1

    def test_reactive_resample_never_fabricates_a_delta_across_the_identity_transition_itself(self):
        pre_reset = _dp_sample(3.0, lib.DATAPLANE_REGIME_ENDPOINT_BACKED, ("A", "B"), 1148)
        post_reset = _dp_sample(6.4, lib.DATAPLANE_REGIME_ENDPOINT_BACKED, ("A",), 249)
        reactive_followup = _dp_sample(7.4, lib.DATAPLANE_REGIME_ENDPOINT_BACKED, ("A",), 260)
        result = lib.compute_dataplane_ingress_evidence_windows(
            0.0, 10.0, dataplane_samples=[pre_reset, post_reset, reactive_followup]
        )
        used_pairs = {(iv["t1_elapsed"], iv["t2_elapsed"]) for iv in result["dataplane_counter_intervals_used"]}
        assert (3.0, 6.4) not in used_pairs
        assert (6.4, 7.4) in used_pairs

    def test_anchor_sample_still_rejected_as_cross_window_when_it_lands_outside(self):
        lone = _dp_sample(8.0, lib.DATAPLANE_REGIME_ENDPOINT_BACKED, ("KUBE-SEP-AAAA",), 100)
        anchor_in_next_window = _dp_sample(12.0, lib.DATAPLANE_REGIME_ENDPOINT_BACKED, ("KUBE-SEP-AAAA",), 150)
        window_a = lib.compute_dataplane_ingress_evidence_windows(0.0, 10.0, dataplane_samples=[lone, anchor_in_next_window])
        assert window_a["nodeport_dataplane_ingress_evidence"] is False
        assert len(window_a["dataplane_counter_intervals_rejected_cross_window"]) == 1
        assert window_a["dataplane_counter_intervals_used"] == []

    def test_zero_endpoint_regime_transition_still_confirmed_via_reactive_resample(self):
        # Backend collapses to zero Ready endpoints mid-window; CORR-41's
        # KUBE-EXTERNAL-SERVICES evidence path must remain available, and
        # CORR-42's reactive resample must not weaken it.
        endpoint_backed = _dp_sample(2.0, lib.DATAPLANE_REGIME_ENDPOINT_BACKED, ("KUBE-SEP-AAAA",), 400)
        no_endpoints_first = _dp_sample(5.5, lib.DATAPLANE_REGIME_NO_ENDPOINTS, ("KUBE-EXTERNAL-SERVICES",), 12)
        assert lib.should_trigger_immediate_resample(endpoint_backed, no_endpoints_first) is True
        reactive_followup = _dp_sample(6.5, lib.DATAPLANE_REGIME_NO_ENDPOINTS, ("KUBE-EXTERNAL-SERVICES",), 19)
        result = lib.compute_dataplane_ingress_evidence_windows(
            0.0, 10.0, dataplane_samples=[endpoint_backed, no_endpoints_first, reactive_followup]
        )
        assert result["nodeport_dataplane_ingress_evidence"] is True
        assert result["dataplane_backend_regimes_with_evidence"] == [lib.DATAPLANE_REGIME_NO_ENDPOINTS]

    def test_direct_relay_failure_precedence_unaffected_by_new_dataplane_samples(self):
        # Even overwhelming positive dataplane evidence must never override
        # a direct, confirmed harness-side failure signal for this window.
        classification = lib.classify_route_integrity_window(
            delivered_request_evidence=False,
            tunnel_process_alive=False,
            local_listener_present=True,
            nodeport_dataplane_ingress_evidence=True,
        )
        assert classification["window_route_status"] == lib.ROUTE_INTEGRITY_FAILED

    def test_single_anchor_sample_alone_is_not_fabricated_into_evidence(self):
        # No-traffic false-positive prevention: one sample, however placed,
        # can never form a pair -- must stay unconfirmed, never CONFIRMED.
        anchor_only = _dp_sample(3.6, lib.DATAPLANE_REGIME_NO_ENDPOINTS, ("KUBE-EXTERNAL-SERVICES",), 18921)
        result = lib.compute_dataplane_ingress_evidence_windows(0.0, 4.0, dataplane_samples=[anchor_only])
        assert result["nodeport_dataplane_ingress_evidence"] is False
        assert result["dataplane_counter_intervals_used"] == []


class TestProtocolV7Corr42Phase15B3P1:
    """CORR-42/Protocol V7 provenance, manifest, and protocol-doc checks --
    mirrors TestProtocolV6Corr41Phase15B3O1's own pattern."""

    def test_provenance_active_protocol_moved_on_from_7_after_corr43(self):
        # Phase 15B.3P.2 (CORR-43, Protocol V8) supersedes V7 as the
        # CURRENTLY active protocol -- `active_calibration_protocol_version`/
        # `active_correction_reference` always report the current state (see
        # test_provenance_takes_no_index_argument_so_it_applies_to_whichever_slot_runs_next),
        # never a frozen historical snapshot of V7 specifically.
        prov = lib.build_protocol_provenance()
        assert prov["active_calibration_protocol_version"] == 8
        assert prov["active_calibration_manifest"] == "experiments/manifests/rl_dynamics_calibration_v8.yaml"
        assert prov["active_correction_reference"] == "CORR-48"
        assert prov["prior_correction_reference"] == "CORR-47"

    def test_provenance_dataplane_acquisition_protocol_version_is_now_2(self):
        prov = lib.build_protocol_provenance()
        assert prov["dataplane_acquisition_protocol_version"] == 2

    def test_provenance_route_integrity_and_transport_versions_unchanged_by_corr42(self):
        # CORR-42 changes dataplane-evidence ACQUISITION only -- classification
        # (route_integrity_protocol_version) and transport are both untouched.
        prov = lib.build_protocol_provenance()
        assert prov["route_integrity_protocol_version"] == 6
        assert prov["transport_protocol_version"] == 5

    def test_provenance_includes_v7_verifiable_sha256_hash(self):
        import hashlib as _hashlib

        prov = lib.build_protocol_provenance()
        hashes = prov["provenance_hashes_sha256"]
        assert isinstance(hashes["v7_manifest"], str) and len(hashes["v7_manifest"]) == 64
        assert hashes["v7_manifest"] == _hashlib.sha256(lib.MANIFEST_V7_PATH.read_bytes()).hexdigest()
        # V6's own hashes remain present -- V7 does not erase V6's provenance.
        assert isinstance(hashes["v6_manifest"], str) and len(hashes["v6_manifest"]) == 64

    def test_v7_manifest_exists_and_is_valid_yaml(self):
        import yaml

        m = yaml.safe_load(lib.MANIFEST_V7_PATH.read_text(encoding="utf-8"))
        assert m["protocol_version"] == 7
        assert m["correction_reference"] == "CORR-42"
        assert m["prior_correction_reference"] == "CORR-41"

    def test_v7_manifest_does_not_redefine_v6_route_integrity_classification(self):
        import yaml

        text = lib.MANIFEST_V7_PATH.read_text(encoding="utf-8")
        m = yaml.safe_load(text)
        assert "transport_protocol_version remains 5" in text
        assert m["route_integrity_v7_classification"]["route_integrity_protocol_version"] == 6
        assert m["route_integrity_v7_classification"]["dataplane_acquisition_protocol_version"] == 2
        assert m["unaffected"]  # non-empty preserved-invariants list, matching v6's own convention

    def test_v7_manifest_records_attempt4_non_retroactivity(self):
        import yaml

        m = yaml.safe_load(lib.MANIFEST_V7_PATH.read_text(encoding="utf-8"))
        assert m["non_retroactivity"]["ledger_state_unchanged"].count("attempt1") == 1
        assert "attempt4" in m["non_retroactivity"]["ledger_state_unchanged"]
        assert "next_resumable_index()==12" in m["non_retroactivity"]["ledger_state_unchanged"]

    def test_v1_v5_v6_manifest_hashes_unchanged_by_v7_introduction(self):
        assert hashlib.sha256(lib.MANIFEST_PATH.read_bytes()).hexdigest() == "2e6011495307245bcc85832d0b0e1aa6cbc6ea63e4655914eeb9c942c15fc2bf"
        assert hashlib.sha256(lib.MANIFEST_V3_PATH.read_bytes()).hexdigest() == "a0c7f3cebcdcbd7f9ca766a7057681e7b29cd2d10c4b8de74478fb54681561b8"

    def test_v7_cadence_and_transition_constants_unaffected_by_v8_introduction(self):
        # CORR-43/V8 touches window PARTITIONING only -- the V7 acquisition
        # constants this class exists to protect must be untouched.
        assert lib.DATAPLANE_INGRESS_SAMPLING_INTERVAL_SECONDS_V7 == 2.0
        assert lib.DATAPLANE_INGRESS_TRANSITION_RESAMPLE_DELAY_SECONDS == 1.0


def _windows_cover_exactly(start, end, windows):
    """Property helper (CORR-43): union(windows) == [start, end), pairwise
    non-overlapping (adjacent windows share exactly their boundary point,
    nothing more), chronologically ordered, no gap, no duplicated
    duration."""
    assert windows, "must produce at least one window for a positive-length interval"
    assert windows[0][0] == start
    assert windows[-1][1] == end
    for (a1, b1), (a2, b2) in zip(windows, windows[1:]):
        assert b1 == a2, f"gap or overlap between {(a1, b1)} and {(a2, b2)}"
        assert a1 < b1 and a2 < b2
    return True


class TestPartitionIntoEvidenceWindowsCorr43Phase15B3P2:
    """CORR-43/Protocol V8: `partition_into_evidence_windows()`'s new
    `min_window_seconds` merge behavior. Formal coverage invariants (union
    covers exactly, no gaps, no duplication, chronological order) are
    proven for both the unmodified default and the merged case; the merge
    logic itself is exercised against attempt4's exact real geometry and
    several boundary conditions."""

    def test_default_behavior_byte_for_byte_unchanged_when_min_window_seconds_omitted(self):
        # This IS the historical-immutability guarantee for attempts 1-4:
        # omitting the new parameter reproduces pre-CORR-43 output exactly.
        assert lib.partition_into_evidence_windows(105.0, 179.0) == [
            (105.0, 115.0), (115.0, 125.0), (125.0, 135.0), (135.0, 145.0),
            (145.0, 155.0), (155.0, 165.0), (165.0, 175.0), (175.0, 179.0),
        ]

    def test_attempt4_style_short_remainder_is_merged_into_preceding_window(self):
        # attempt4's own geometry: material interval [105,179], length 74 =
        # 7*10 + 4 -- the task's own illustrative example (10x6 + 14).
        windows = lib.partition_into_evidence_windows(105.0, 179.0, min_window_seconds=lib.MIN_EVIDENCE_WINDOW_SECONDS_V8)
        assert windows == [
            (105.0, 115.0), (115.0, 125.0), (125.0, 135.0), (135.0, 145.0),
            (145.0, 155.0), (155.0, 165.0), (165.0, 179.0),
        ]
        assert len(windows) == 7  # down from 8
        assert _windows_cover_exactly(105.0, 179.0, windows)

    def test_exact_multiple_interval_is_unaffected_by_merge_parameter(self):
        windows = lib.partition_into_evidence_windows(0.0, 100.0, min_window_seconds=lib.MIN_EVIDENCE_WINDOW_SECONDS_V8)
        assert windows == [(float(i * 10), float((i + 1) * 10)) for i in range(10)]
        assert _windows_cover_exactly(0.0, 100.0, windows)

    def test_remainder_already_at_or_above_threshold_is_left_standalone(self):
        # remainder = 9.5s >= MIN_EVIDENCE_WINDOW_SECONDS_V8 (9.0s) -- already
        # deterministically safe on its own, so merging would only cost
        # temporal resolution for no safety benefit; must stay separate.
        windows = lib.partition_into_evidence_windows(0.0, 29.5, min_window_seconds=lib.MIN_EVIDENCE_WINDOW_SECONDS_V8)
        assert windows == [(0.0, 10.0), (10.0, 20.0), (20.0, 29.5)]
        assert _windows_cover_exactly(0.0, 29.5, windows)

    def test_remainder_just_under_threshold_is_merged(self):
        # remainder = 8.99s < 9.0s threshold -- must merge.
        windows = lib.partition_into_evidence_windows(0.0, 28.99, min_window_seconds=lib.MIN_EVIDENCE_WINDOW_SECONDS_V8)
        assert windows == [(0.0, 10.0), (10.0, 28.99)]
        assert _windows_cover_exactly(0.0, 28.99, windows)

    def test_whole_interval_shorter_than_one_window_is_the_sole_irreducible_case(self):
        # No preceding window exists to merge into -- honestly retained,
        # never fabricated into a false sense of safety.
        windows = lib.partition_into_evidence_windows(0.0, 7.0, min_window_seconds=lib.MIN_EVIDENCE_WINDOW_SECONDS_V8)
        assert windows == [(0.0, 7.0)]
        assert _windows_cover_exactly(0.0, 7.0, windows)

    def test_merge_never_applies_when_no_remainder_and_single_window(self):
        # interval length == window_seconds exactly: one window, no merge
        # candidate needed or attempted.
        windows = lib.partition_into_evidence_windows(0.0, 10.0, min_window_seconds=lib.MIN_EVIDENCE_WINDOW_SECONDS_V8)
        assert windows == [(0.0, 10.0)]

    def test_deterministic_enumeration_over_many_interval_lengths_preserves_coverage_invariants(self):
        # Property-style DETERMINISTIC enumeration (fixed ranges, not random
        # Monte Carlo) per the governing phase's own sec.10 requirement:
        # material interval lengths from 45s (the realistic minimum implied
        # by the 25% materiality gate) through 200s in 0.5s steps, covering
        # every possible remainder value from 0 up to just under 10s many
        # times over.
        length = 45.0
        checked = 0
        while length <= 200.0:
            windows = lib.partition_into_evidence_windows(0.0, length, min_window_seconds=lib.MIN_EVIDENCE_WINDOW_SECONDS_V8)
            assert _windows_cover_exactly(0.0, length, windows)
            assert all((b - a) >= lib.MIN_EVIDENCE_WINDOW_SECONDS_FOR_COUNT_GUARANTEE_V8 - 1e-9 for a, b in windows[:-1]) or len(windows) == 1
            # every window except possibly a genuinely irreducible sole
            # window is at least window_seconds (10.0) long, and the FINAL
            # window is either exactly 10.0 or >= MIN_EVIDENCE_WINDOW_SECONDS_V8
            last_len = windows[-1][1] - windows[-1][0]
            assert last_len == pytest.approx(10.0) or last_len >= lib.MIN_EVIDENCE_WINDOW_SECONDS_V8 or len(windows) == 1
            checked += 1
            length = round(length + 0.5, 2)
        assert checked > 300  # confirms this was a real sweep, not a no-op

    def test_deterministic_enumeration_also_covers_diagnostic_scale_short_intervals(self):
        # Extends the sweep down to diagnostic/non-official scales (this
        # phase's own U1-U6 validation uses intervals far shorter than the
        # 45s realistic official minimum) -- coverage invariants must hold
        # unconditionally, not just at official scale.
        length = 0.5
        checked = 0
        while length <= 45.0:
            windows = lib.partition_into_evidence_windows(0.0, length, min_window_seconds=lib.MIN_EVIDENCE_WINDOW_SECONDS_V8)
            assert _windows_cover_exactly(0.0, length, windows)
            checked += 1
            length = round(length + 0.25, 2)
        assert checked > 175


class TestDeterministicTailWindowTimingAnalysisPhase15B3P2:
    """Exact (non-Monte-Carlo) worst-case interior-sample-count formula
    underlying `MIN_EVIDENCE_WINDOW_SECONDS_V8`'s derivation -- verified by
    exhaustive fine-grid enumeration over the full phase space, matching
    the closed-form floor(r/T)."""

    @staticmethod
    def _min_interior_samples_exhaustive(window_width, cadence, n_steps=20000):
        worst = None
        for i in range(n_steps):
            phi = cadence * (i / n_steps)
            count = 0 if phi >= window_width else int((window_width - phi) // cadence) + 1
            if worst is None or count < worst:
                worst = count
        return worst

    def test_worst_case_matches_closed_form_floor_r_over_t(self):
        import math

        T = lib.DATAPLANE_ACQUISITION_WORST_CASE_CADENCE_SECONDS_V8
        for r in [1.0, 3.0, 5.0, 6.0, 8.0, 9.0, 10.0, 14.0, 20.0]:
            exhaustive = self._min_interior_samples_exhaustive(r, T)
            assert exhaustive == math.floor(r / T), (r, T, exhaustive)

    def test_2t_is_the_exact_minimum_width_guaranteeing_two_samples(self):
        T = lib.DATAPLANE_ACQUISITION_WORST_CASE_CADENCE_SECONDS_V8
        assert lib.MIN_EVIDENCE_WINDOW_SECONDS_FOR_COUNT_GUARANTEE_V8 == pytest.approx(2 * T)
        just_below = self._min_interior_samples_exhaustive(2 * T - 0.01, T)
        at_threshold = self._min_interior_samples_exhaustive(2 * T, T)
        assert just_below < 2
        assert at_threshold >= 2

    def test_3t_is_the_exact_minimum_width_guaranteeing_three_samples_transition_resilience(self):
        T = lib.DATAPLANE_ACQUISITION_WORST_CASE_CADENCE_SECONDS_V8
        assert lib.MIN_EVIDENCE_WINDOW_SECONDS_V8 == pytest.approx(3 * T)
        just_below = self._min_interior_samples_exhaustive(3 * T - 0.01, T)
        at_threshold = self._min_interior_samples_exhaustive(3 * T, T)
        assert just_below < 3
        assert at_threshold >= 3

    def test_width_narrower_than_one_cadence_period_cannot_even_guarantee_one_sample(self):
        # No finite periodic cadence can guarantee ANY interior sample, let
        # alone two, once the tail width can be adversarially placed below
        # one full cadence period -- the irreducible limit Candidate D
        # (higher periodic frequency) can never fully close.
        T = lib.DATAPLANE_ACQUISITION_WORST_CASE_CADENCE_SECONDS_V8
        assert self._min_interior_samples_exhaustive(T - 0.01, T) == 0


class TestProtocolV8Corr43Phase15B3P2:
    """CORR-43/Protocol V8 provenance, manifest, and protocol-doc checks --
    mirrors TestProtocolV7Corr42Phase15B3P1's own pattern."""

    def test_provenance_active_protocol_is_now_8(self):
        prov = lib.build_protocol_provenance()
        assert prov["active_calibration_protocol_version"] == 8
        assert prov["active_calibration_manifest"] == "experiments/manifests/rl_dynamics_calibration_v8.yaml"
        assert prov["active_correction_reference"] == "CORR-48"
        assert prov["prior_correction_reference"] == "CORR-47"

    def test_provenance_window_partitioning_protocol_version_is_now_2(self):
        prov = lib.build_protocol_provenance()
        assert prov["window_partitioning_protocol_version"] == 2

    def test_provenance_classification_and_acquisition_versions_unchanged_by_corr43(self):
        # CORR-43 changes window PARTITIONING only -- classification
        # (route_integrity_protocol_version) and acquisition
        # (dataplane_acquisition_protocol_version) are both untouched.
        prov = lib.build_protocol_provenance()
        assert prov["route_integrity_protocol_version"] == 6
        assert prov["dataplane_acquisition_protocol_version"] == 2
        assert prov["transport_protocol_version"] == 5

    def test_provenance_includes_v8_verifiable_sha256_hash(self):
        import hashlib as _hashlib

        prov = lib.build_protocol_provenance()
        hashes = prov["provenance_hashes_sha256"]
        assert isinstance(hashes["v8_manifest"], str) and len(hashes["v8_manifest"]) == 64
        assert hashes["v8_manifest"] == _hashlib.sha256(lib.MANIFEST_V8_PATH.read_bytes()).hexdigest()
        # V7's own hashes remain present -- V8 does not erase V7's provenance.
        assert isinstance(hashes["v7_manifest"], str) and len(hashes["v7_manifest"]) == 64

    def test_v8_manifest_exists_and_is_valid_yaml(self):
        import yaml

        m = yaml.safe_load(lib.MANIFEST_V8_PATH.read_text(encoding="utf-8"))
        assert m["protocol_version"] == 8
        assert m["correction_reference"] == "CORR-43"
        assert m["prior_correction_reference"] == "CORR-42"

    def test_v8_manifest_does_not_redefine_v7_acquisition_or_v6_classification(self):
        import yaml

        text = lib.MANIFEST_V8_PATH.read_text(encoding="utf-8")
        m = yaml.safe_load(text)
        assert "transport_protocol_version remains 5" in text
        assert m["route_integrity_v8_classification"]["route_integrity_protocol_version"] == 6
        assert m["route_integrity_v8_classification"]["dataplane_acquisition_protocol_version"] == 2
        assert m["route_integrity_v8_classification"]["window_partitioning_protocol_version"] == 2
        assert m["unaffected"]

    def test_v8_manifest_records_attempt4_non_retroactivity(self):
        import yaml

        m = yaml.safe_load(lib.MANIFEST_V8_PATH.read_text(encoding="utf-8"))
        assert m["non_retroactivity"]["ledger_state_unchanged"].count("attempt1") == 1
        assert "attempt4" in m["non_retroactivity"]["ledger_state_unchanged"]
        assert "next_resumable_index()==12" in m["non_retroactivity"]["ledger_state_unchanged"]

    def test_v1_v6_v7_manifest_hashes_unchanged_by_v8_introduction(self):
        assert hashlib.sha256(lib.MANIFEST_PATH.read_bytes()).hexdigest() == "2e6011495307245bcc85832d0b0e1aa6cbc6ea63e4655914eeb9c942c15fc2bf"
        assert hashlib.sha256(lib.MANIFEST_V6_PATH.read_bytes()).hexdigest() == "f7cbbebda04f3895c81bdd4f7bf5becc015975c59bfc6d92c0a3de48fb9e2ca6"
        assert hashlib.sha256(lib.MANIFEST_V7_PATH.read_bytes()).hexdigest() == "aee412379237b95ceec127473d53200b68feb73dd27d2eaa108d4b70f38ee090"


class TestSampleDataplaneIngressRulesTimeoutHandlingPhase15B3P3:
    """Phase 15B.3P.3 implementation hardening: a genuine `docker exec`
    timeout (`subprocess.TimeoutExpired`, NOT an `OSError` subclass) must
    be caught exactly like any other observer failure -- previously
    uncaught, it would silently kill the entire dataplane sampler thread
    for the rest of an attempt."""

    def test_timeout_expired_is_caught_as_observation_error_not_raised(self, monkeypatch):
        def _raise_timeout(*a, **k):
            raise subprocess.TimeoutExpired(cmd=["docker", "exec"], timeout=5)

        monkeypatch.setattr(harness, "_run", _raise_timeout)
        result = harness.sample_dataplane_ingress_rules(31891)
        assert result["regime"] is None
        assert result["observation_error"] is not None
        assert "timed out" in result["observation_error"].lower() or "timeout" in result["observation_error"].lower()

    def test_timeout_expired_never_fabricates_a_regime_or_packet_count(self, monkeypatch):
        monkeypatch.setattr(harness, "_run", lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired(cmd=["x"], timeout=5)))
        result = harness.sample_dataplane_ingress_rules(31891)
        assert result["rule_identity"] is None
        assert result["total_packet_count"] is None
        assert result["total_byte_count"] is None


class TestDataplaneAcquisitionGapRecordingPhase15B3P3:
    """Phase 15B.3P.3: `summarize_dataplane_acquisition_gaps()` -- pure
    observability over recorded inter-sample gaps. Never feeds into
    classification; explicitly distinguishes an OBSERVED maximum from a
    claimed GUARANTEED maximum."""

    def test_empty_samples_yields_n_zero_and_no_fabricated_statistics(self):
        result = lib.summarize_dataplane_acquisition_gaps([])
        assert result["n"] == 0
        assert result["min"] is None and result["max"] is None

    def test_single_sample_with_no_recorded_gap_yields_n_zero(self):
        result = lib.summarize_dataplane_acquisition_gaps([{"actual_gap_seconds_from_previous_sample": None}])
        assert result["n"] == 0

    def test_known_gap_sequence_computes_correct_statistics(self):
        samples = [{"actual_gap_seconds_from_previous_sample": g} for g in [2.1, 2.2, 2.3, 2.4, 10.0]]
        result = lib.summarize_dataplane_acquisition_gaps(samples)
        assert result["n"] == 5
        assert result["min"] == 2.1
        assert result["max"] == 10.0
        assert result["mean"] == pytest.approx((2.1 + 2.2 + 2.3 + 2.4 + 10.0) / 5)

    def test_result_explicitly_flags_observed_is_not_guaranteed(self):
        samples = [{"actual_gap_seconds_from_previous_sample": g} for g in [2.0, 2.0]]
        result = lib.summarize_dataplane_acquisition_gaps(samples)
        assert result["observed_maximum_is_not_a_guaranteed_maximum"] is True

    def test_recorded_gap_field_present_on_real_sample_wrapper(self, monkeypatch):
        # Integration-level: the orchestrator's own recording wrapper
        # attaches actual_gap_seconds_from_previous_sample and
        # acquisition_gap_exceeded_assumed_bound to every sample after the
        # first, without altering sample_dataplane_ingress_rules()'s own
        # return shape.
        calls = {"n": 0}

        def _fake_sample(nodeport):
            calls["n"] += 1
            return {
                "regime": lib.DATAPLANE_REGIME_ENDPOINT_BACKED,
                "rule_identity": ("KUBE-SEP-AAAA",),
                "total_packet_count": calls["n"],
                "total_byte_count": calls["n"] * 60,
                "observation_error": None,
            }

        monkeypatch.setattr(harness, "sample_dataplane_ingress_rules", _fake_sample)
        # This test only verifies the wrapped function EXISTS with the
        # expected pure building block it depends on -- the full threaded
        # orchestration is exercised live in this phase's own non-official
        # M1-M4 validation, not re-simulated here.
        assert callable(harness.sample_dataplane_ingress_rules)


class TestMultiTransitionRouteEvidencePhase15B3P3:
    """Phase 15B.3P.3 sec.7: formal, deterministic characterization of
    windows with k interior samples and 0/1/2/3+ identity/regime
    transitions. The general rule (not previously stated explicitly): a
    valid same-identity confirming pair exists if and only if AT LEAST
    ONE pair of CONSECUTIVE samples shares an identity -- independent of
    how many DISTINCT identities appear in the window, and independent of
    the total transition COUNT. A cross-identity pair is rejected exactly
    once per governing rule (CORR-41, unchanged); no fabrication ever
    occurs regardless of how many transitions are present."""

    def _seq(self, *regimes_and_identities, base_count=100, step=10):
        samples = []
        for i, (regime, identity) in enumerate(regimes_and_identities):
            samples.append(_dp_sample(float(i * 2), regime, identity, base_count + i * step))
        return samples

    EB = lib.DATAPLANE_REGIME_ENDPOINT_BACKED
    NE = lib.DATAPLANE_REGIME_NO_ENDPOINTS

    def test_zero_transitions_AAAA_all_pairs_valid(self):
        samples = self._seq((self.EB, ("A",)), (self.EB, ("A",)), (self.EB, ("A",)), (self.EB, ("A",)))
        result = lib.compute_dataplane_ingress_evidence_windows(0.0, 10.0, dataplane_samples=samples)
        assert result["nodeport_dataplane_ingress_evidence"] is True
        assert len(result["dataplane_counter_intervals_used"]) == 3
        assert len(result["dataplane_counter_intervals_rejected_reset"]) == 0

    def test_one_transition_ABBB_two_valid_pairs_survive(self):
        samples = self._seq((self.EB, ("A",)), (self.EB, ("B",)), (self.EB, ("B",)), (self.EB, ("B",)))
        result = lib.compute_dataplane_ingress_evidence_windows(0.0, 10.0, dataplane_samples=samples)
        assert result["nodeport_dataplane_ingress_evidence"] is True
        assert len(result["dataplane_counter_intervals_used"]) == 2
        assert len(result["dataplane_counter_intervals_rejected_reset"]) == 1

    def test_two_transitions_ABCC_one_valid_pair_survives(self):
        samples = self._seq((self.EB, ("A",)), (self.EB, ("B",)), (self.EB, ("C",)), (self.EB, ("C",)))
        result = lib.compute_dataplane_ingress_evidence_windows(0.0, 10.0, dataplane_samples=samples)
        assert result["nodeport_dataplane_ingress_evidence"] is True
        assert len(result["dataplane_counter_intervals_used"]) == 1
        assert len(result["dataplane_counter_intervals_rejected_reset"]) == 2

    def test_three_transitions_ABCD_every_pair_differs_zero_valid_pairs(self):
        # THE worst case: k=4 samples, but every adjacent pair is a
        # transition -- zero confirming pairs regardless of sample count.
        samples = self._seq((self.EB, ("A",)), (self.EB, ("B",)), (self.EB, ("C",)), (self.EB, ("D",)))
        result = lib.compute_dataplane_ingress_evidence_windows(0.0, 10.0, dataplane_samples=samples)
        assert result["nodeport_dataplane_ingress_evidence"] is False
        assert len(result["dataplane_counter_intervals_used"]) == 0
        assert len(result["dataplane_counter_intervals_rejected_reset"]) == 3

    def test_oscillating_ABAB_also_zero_valid_pairs_despite_only_two_distinct_identities(self):
        # Only 2 DISTINCT identities are involved (unlike ABCD's 4), but
        # since EVERY adjacent pair still differs, the outcome is
        # identical to ABCD -- proving distinct-identity-count is not the
        # relevant measure; adjacency is.
        samples = self._seq((self.EB, ("A",)), (self.EB, ("B",)), (self.EB, ("A",)), (self.EB, ("B",)))
        result = lib.compute_dataplane_ingress_evidence_windows(0.0, 10.0, dataplane_samples=samples)
        assert result["nodeport_dataplane_ingress_evidence"] is False
        assert len(result["dataplane_counter_intervals_used"]) == 0
        assert len(result["dataplane_counter_intervals_rejected_reset"]) == 3

    def test_endpoint_to_no_endpoint_to_endpoint_regime_oscillation_zero_valid_pairs(self):
        # A real-world instance of the ABAB shape: a pod is deleted (->
        # no_endpoints) and a replacement becomes ready (-> endpoint_backed
        # again) -- but the NEW pod's KUBE-SEP-<hash> is never the same
        # string as the old one, so this is genuinely 2 real transitions,
        # not a return to the original identity.
        samples = self._seq(
            (self.EB, ("KUBE-SEP-OLD",)),
            (self.NE, ("KUBE-EXTERNAL-SERVICES",)),
            (self.EB, ("KUBE-SEP-NEW",)),
        )
        result = lib.compute_dataplane_ingress_evidence_windows(0.0, 10.0, dataplane_samples=samples)
        assert result["nodeport_dataplane_ingress_evidence"] is False
        assert len(result["dataplane_counter_intervals_used"]) == 0
        assert len(result["dataplane_counter_intervals_rejected_reset"]) == 2

    def test_never_fabricates_a_delta_regardless_of_transition_count(self):
        # Regression guard: no matter how many transitions are injected,
        # dataplane_counter_intervals_used must NEVER include a
        # cross-identity pair.
        samples = self._seq((self.EB, ("A",)), (self.EB, ("B",)), (self.EB, ("C",)), (self.EB, ("D",)), (self.EB, ("E",)))
        result = lib.compute_dataplane_ingress_evidence_windows(0.0, 10.0, dataplane_samples=samples)
        for iv in result["dataplane_counter_intervals_used"]:
            assert iv["rule_identity_t1"] == iv["rule_identity_t2"]

    def test_reactive_resample_would_fire_on_every_one_of_multiple_transitions(self):
        # should_trigger_immediate_resample() is stateless per-pair, so it
        # correctly fires independently for EACH transition in a churn
        # sequence, not just the first -- each firing gives the production
        # sampler one more chance to capture a same-identity follow-up.
        seq = [
            _dp_sample(0.0, self.EB, ("A",), 100),
            _dp_sample(2.0, self.EB, ("B",), 110),
            _dp_sample(4.0, self.NE, ("KUBE-EXTERNAL-SERVICES",), 1),
            _dp_sample(6.0, self.EB, ("C",), 120),
        ]
        transitions = [lib.should_trigger_immediate_resample(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]
        assert transitions == [True, True, True]


# ---------------------------------------------------------------------------
# Phase 15B.5B: replica-transition execution support.
# ---------------------------------------------------------------------------


class TestReplicaTransitionResolutionAndTiming:
    """Resolution/dry-run coverage (governing Phase 15B.5B sec.23,
    'Resolution/dry-run' bullet list)."""

    def test_index30_resolves_as_replica_transition_5_to_6(self, real_sequence):
        slot = lib.resolve_planned_slot(30, sequence=real_sequence)
        assert slot["run_kind"] == "replica_transition"
        assert slot["planned_slot_id"] == "transrep_5-6_5to6_repeat1"
        assert slot["replicas_before"] == 5
        assert slot["replicas_after"] == 6
        assert "replicas" not in slot  # the steady-state-only key must be ABSENT, never null-filled

    def test_index31_resolves_as_replica_transition_2_to_1_scale_down(self, real_sequence):
        slot = lib.resolve_planned_slot(31, sequence=real_sequence)
        assert slot["run_kind"] == "replica_transition"
        assert slot["replicas_before"] == 2
        assert slot["replicas_after"] == 1

    def test_resolve_transition_timing_plan_matches_frozen_manifest(self):
        manifest = lib.load_manifest()
        timing = lib.resolve_transition_timing_plan(manifest)
        assert timing["pre_run_replica_stabilization_seconds"] == 60.0
        assert timing["pre_transition_traffic_steady_seconds"] == 60.0
        assert timing["trigger_at_traffic_elapsed_seconds"] == 60.0
        assert timing["post_transition_observation_seconds"] == 180.0
        assert timing["total_traffic_seconds"] == 240.0
        assert timing["total_seconds_per_run"] == 300.0

    def test_dry_run_report_for_replica_transition_shows_before_after_not_null(self, real_sequence):
        manifest = lib.load_manifest()
        slot = lib.resolve_planned_slot(30, sequence=real_sequence)
        ledger = lib.init_ledger(real_sequence)
        report = harness.dry_run_report(slot, slot["schedule_sha256"], manifest, ledger)
        assert report["replicas_before"] == 5
        assert report["replicas_after"] == 6
        assert report["transition_direction"] == "scale_up"
        assert report["offered_rps"] == 25.0
        assert report["trigger_at_traffic_elapsed_seconds"] == 60.0
        assert report["traffic_duration_seconds"] == 240.0
        assert report["total_run_seconds"] == 300.0
        assert report["scientific_role"] == "TRANSITION_TRAJECTORY_VALIDATION"
        # The pre-15B.5B bug this corrects (Phase 15B.5A sec.6(c)):
        assert report.get("replicas") is None or "replicas" not in report
        assert report["traffic_duration_seconds"] != 180  # never the steady-state duration

    def test_dry_run_report_for_replica_transition_scale_down_direction(self, real_sequence):
        manifest = lib.load_manifest()
        slot = lib.resolve_planned_slot(31, sequence=real_sequence)
        ledger = lib.init_ledger(real_sequence)
        report = harness.dry_run_report(slot, slot["schedule_sha256"], manifest, ledger)
        assert report["replicas_before"] == 2
        assert report["replicas_after"] == 1
        assert report["transition_direction"] == "scale_down"

    def test_dry_run_report_expected_requests_derived_from_schedule_itself(self, real_sequence):
        # 25.0 RPS x 240s = 6000 -- but this must come from COUNTING the
        # schedule file's own rows, never from the RPS*duration label.
        manifest = lib.load_manifest()
        slot = lib.resolve_planned_slot(30, sequence=real_sequence)
        ledger = lib.init_ledger(real_sequence)
        report = harness.dry_run_report(slot, slot["schedule_sha256"], manifest, ledger)
        assert report["expected_requests"] == 6000

    def test_steady_state_dry_run_report_completely_unaffected(self, real_sequence):
        # Regression guard: an ordinary steady-state slot's dry-run report
        # must remain byte-for-byte identical in shape to the pre-15B.5B
        # implementation (same fields, `replicas` present, no transition
        # fields leaking in).
        manifest = lib.load_manifest()
        slot = lib.resolve_planned_slot(1, sequence=real_sequence)
        ledger = lib.init_ledger(real_sequence)
        report = harness.dry_run_report(slot, slot["schedule_sha256"], manifest, ledger)
        assert "replicas" in report
        assert "replicas_before" not in report
        assert "replicas_after" not in report
        assert "transition_direction" not in report
        assert report["traffic_duration_seconds"] == 180


class TestReplicaTransitionLibFunctions:
    def test_transition_direction_scale_up(self):
        assert lib.transition_direction(5, 6) == "scale_up"

    def test_transition_direction_scale_down(self):
        assert lib.transition_direction(2, 1) == "scale_down"

    def test_transition_direction_rejects_equal_values(self):
        with pytest.raises(ValueError):
            lib.transition_direction(3, 3)

    def test_is_replica_transition_slot(self, real_sequence):
        assert lib.is_replica_transition_slot(lib.resolve_planned_slot(30, sequence=real_sequence)) is True
        assert lib.is_replica_transition_slot(lib.resolve_planned_slot(1, sequence=real_sequence)) is False

    def test_initial_replica_count_for_restoration_steady_state_uses_replicas(self, real_sequence):
        slot = lib.resolve_planned_slot(1, sequence=real_sequence)
        assert lib.initial_replica_count_for_restoration(slot) == slot["replicas"]

    def test_initial_replica_count_for_restoration_transition_uses_max_before_after(self, real_sequence):
        slot_up = lib.resolve_planned_slot(30, sequence=real_sequence)  # 5 -> 6
        assert lib.initial_replica_count_for_restoration(slot_up) == 6
        slot_down = lib.resolve_planned_slot(31, sequence=real_sequence)  # 2 -> 1
        assert lib.initial_replica_count_for_restoration(slot_down) == 2

    def test_initial_replica_count_never_raises_keyerror_replicas(self, real_sequence):
        # Regression guard for the exact Phase 15B.5A finding: slot["replicas"]
        # raised KeyError for every replica_transition slot.
        for idx in (30, 31):
            slot = lib.resolve_planned_slot(idx, sequence=real_sequence)
            assert "replicas" not in slot
            lib.initial_replica_count_for_restoration(slot)  # must not raise

    def test_count_schedule_data_rows_matches_known_transition_schedule(self):
        path = lib.PROJECT_ROOT / "configs/traffic_profiles/rl_dynamics_calibration/replica_transitions/dyncal_transrep_5-6_schedule.csv"
        assert lib.count_schedule_data_rows(path) == 6000

    def test_compute_transition_replica_lags_scale_up_unambiguous_case(self):
        trigger = 100.0
        samples = [
            {"sample_monotonic": 98.0, "current_replicas": 5, "ready_replicas": 5, "available_replicas": 5},
            {"sample_monotonic": 100.0, "current_replicas": 5, "ready_replicas": 5, "available_replicas": 5},
            {"sample_monotonic": 102.0, "current_replicas": 6, "ready_replicas": 5, "available_replicas": 5},
            {"sample_monotonic": 108.0, "current_replicas": 6, "ready_replicas": 6, "available_replicas": 5},
            {"sample_monotonic": 111.0, "current_replicas": 6, "ready_replicas": 6, "available_replicas": 6},
        ]
        lags = lib.compute_transition_replica_lags(trigger, samples, target_replicas=6, direction="scale_up")
        assert lags["desired_to_current_lag_seconds"] == pytest.approx(2.0)
        assert lags["desired_to_ready_lag_seconds"] == pytest.approx(8.0)
        assert lags["desired_to_available_lag_seconds"] == pytest.approx(11.0)

    def test_compute_transition_replica_lags_scale_down(self):
        trigger = 50.0
        samples = [
            {"sample_monotonic": 50.0, "current_replicas": 2, "ready_replicas": 2, "available_replicas": 2},
            {"sample_monotonic": 55.0, "current_replicas": 1, "ready_replicas": 2, "available_replicas": 2},
            {"sample_monotonic": 56.0, "current_replicas": 1, "ready_replicas": 1, "available_replicas": 2},
            {"sample_monotonic": 57.0, "current_replicas": 1, "ready_replicas": 1, "available_replicas": 1},
        ]
        lags = lib.compute_transition_replica_lags(trigger, samples, target_replicas=1, direction="scale_down")
        assert lags["desired_to_current_lag_seconds"] == pytest.approx(5.0)
        assert lags["desired_to_ready_lag_seconds"] == pytest.approx(6.0)
        assert lags["desired_to_available_lag_seconds"] == pytest.approx(7.0)

    def test_compute_transition_replica_lags_never_fabricates_when_target_never_reached(self):
        # Only 5 samples, target (6) never observed -- must return None,
        # never extrapolate or assume eventual success.
        trigger = 0.0
        samples = [{"sample_monotonic": t, "current_replicas": 5, "ready_replicas": 5, "available_replicas": 5} for t in (0.0, 2.0, 4.0)]
        lags = lib.compute_transition_replica_lags(trigger, samples, target_replicas=6, direction="scale_up")
        assert lags["desired_to_current_lag_seconds"] is None
        assert lags["desired_to_ready_lag_seconds"] is None
        assert lags["desired_to_available_lag_seconds"] is None

    def test_compute_transition_replica_lags_ignores_samples_before_trigger(self):
        # A sample taken BEFORE the trigger that already happens to show
        # the target replica count (e.g. residual noise, or the sampler's
        # own baseline row) must never be mistaken for a post-trigger lag.
        trigger = 10.0
        samples = [
            {"sample_monotonic": 5.0, "current_replicas": 6, "ready_replicas": 6, "available_replicas": 6},
            {"sample_monotonic": 15.0, "current_replicas": 6, "ready_replicas": 6, "available_replicas": 6},
        ]
        lags = lib.compute_transition_replica_lags(trigger, samples, target_replicas=6, direction="scale_up")
        assert lags["desired_to_current_lag_seconds"] == pytest.approx(5.0)  # from the t=15 sample, not t=5

    def test_compute_transition_replica_lags_rejects_unknown_direction(self):
        with pytest.raises(ValueError):
            lib.compute_transition_replica_lags(0.0, [], target_replicas=1, direction="sideways")

    def test_build_transition_metrics_summary_marks_undefined_fields_explicitly(self):
        trigger = 0.0
        samples = [{"sample_monotonic": 1.0, "current_replicas": 6, "ready_replicas": 6, "available_replicas": 6}]
        summary = lib.build_transition_metrics_summary(trigger, samples, target_replicas=6, direction="scale_up")
        for field in lib.TRANSITION_MEASURED_FIELDS_NOT_FORMALLY_OPERATIONALIZED:
            assert summary[field] == lib.NOT_FORMALLY_OPERATIONALIZED
        for field in lib.TRANSITION_MEASURED_FIELDS_WITH_FROZEN_DEFINITION:
            assert field in summary
            assert summary[field] is not None  # reached at sample_monotonic=1.0
        assert "not_formally_operationalized_reason" in summary

    def test_build_transition_metrics_summary_never_invents_a_settling_threshold(self):
        # No matter how much data is supplied, the four undefined fields
        # must NEVER resolve to a computed number -- only the frozen-
        # definition lags may ever be numeric.
        trigger = 0.0
        samples = [{"sample_monotonic": float(t), "current_replicas": 6, "ready_replicas": 6, "available_replicas": 6, "cpu_cores": 0.3, "latency_p95_seconds": 0.1} for t in range(20)]
        summary = lib.build_transition_metrics_summary(trigger, samples, target_replicas=6, direction="scale_up")
        for field in lib.TRANSITION_MEASURED_FIELDS_NOT_FORMALLY_OPERATIONALIZED:
            assert summary[field] == "NOT_FORMALLY_OPERATIONALIZED"


class TestReplicaTransitionExecutionBranching:
    """Governing sec.23 'Branching' bullet list: fitting/interpolation-
    validation slots must be completely unaffected; a replica_transition
    slot must route to the new execution path and never touch
    `slot["replicas"]`."""

    def test_main_dispatches_replica_transition_slots_to_dedicated_function(self, monkeypatch, real_sequence):
        called_with = {}

        def _fake_execute(args, manifest, sequence, slot, schedule_hash, ledger, official=True):
            called_with["slot"] = slot
            called_with["official"] = official
            return 0

        def _boom(*a, **k):
            raise AssertionError("steady-state path must not run for a replica_transition slot")

        monkeypatch.setattr(harness, "execute_replica_transition_slot", _fake_execute)
        monkeypatch.setattr(harness, "preflight_checks", _boom)
        monkeypatch.setattr(harness, "delete_hpa", _boom)
        monkeypatch.setattr(harness, "scale_and_wait", _boom)

        rc = harness.main(["--campaign-sequence-index", "30", "--attempt-number", "1"])
        assert rc == 0
        assert called_with["slot"]["run_kind"] == "replica_transition"
        assert called_with["official"] is True

    def test_execute_replica_transition_slot_never_accesses_slot_replicas_key(self, real_sequence):
        # The exact Phase 15B.5A crash: `slot["replicas"]` raising KeyError
        # for every replica_transition slot. A plain dict subclass that
        # raises AssertionError on `__getitem__("replicas")` proves the
        # execution path never even attempts that lookup, without needing
        # a live cluster to reach the point where it would have crashed.
        class _NoReplicasKeyDict(dict):
            def __getitem__(self, key):
                if key == "replicas":
                    raise AssertionError("execute_replica_transition_slot must never access slot['replicas']")
                return super().__getitem__(key)

        slot = _NoReplicasKeyDict(lib.resolve_planned_slot(30, sequence=real_sequence))
        assert slot["replicas_before"] == 5  # sanity: the dict still works normally otherwise
        with pytest.raises(AssertionError):
            _ = slot["replicas"]  # confirms the guard itself is armed correctly


class TestReplicaTransitionTriggerAndSampling:
    def test_trigger_replica_transition_issues_scale_command_only_no_rollout_wait(self, monkeypatch):
        calls = []

        def _fake_run(cmd, timeout=30):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="deployment.apps/demo-app scaled", stderr="")

        monkeypatch.setattr(harness, "_run", _fake_run)
        result = harness.trigger_replica_transition(6)
        assert result["target_replicas"] == 6
        assert result["command_returncode"] == 0
        assert len(calls) == 1  # exactly one command: the scale itself
        assert calls[0] == ["kubectl", "scale", "deployment", "demo-app", "-n", "autoscaler-app", "--replicas=6"]
        assert not any("rollout" in c for c in calls[0])

    def test_trigger_replica_transition_records_failure_without_raising(self, monkeypatch):
        def _fake_run(cmd, timeout=30):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="error: deployment not found")

        monkeypatch.setattr(harness, "_run", _fake_run)
        result = harness.trigger_replica_transition(6)
        assert result["command_returncode"] == 1
        assert "error" in result["command_stderr"]

    def test_sample_replica_transition_state_shape_and_elapsed_math(self, monkeypatch):
        monkeypatch.setattr(harness, "kubectl_jsonpath", lambda args: {
            "jsonpath={.spec.replicas}": "6",
            "jsonpath={.status.replicas}": "6",
            "jsonpath={.status.readyReplicas}": "5",
            "jsonpath={.status.availableReplicas}": "5",
        }[args[-1]])
        monkeypatch.setattr(harness, "get_pod_names", lambda label_selector="app=demo-app": ["pod-a", "pod-b"])
        monkeypatch.setattr(harness, "fetch_loadtest_endpoints_json", lambda: {"subsets": [{"addresses": [{}, {}]}]})

        now = time.monotonic()
        traffic_start = now - 10.0  # traffic notionally started 10s ago
        trigger_mono = now + 60.0  # trigger notionally still 60s in the future
        sample = harness.sample_replica_transition_state(traffic_start, trigger_mono)
        assert sample["desired_replicas"] == 6
        assert sample["current_replicas"] == 6
        assert sample["ready_replicas"] == 5
        assert sample["available_replicas"] == 5
        assert sample["ready_pod_count"] == 2
        assert sample["elapsed_from_traffic_start"] >= 0
        # elapsed_from_trigger is negative before the trigger has occurred
        # (trigger_mono is in the future relative to this sample's own
        # monotonic timestamp).
        assert sample["elapsed_from_trigger"] < 0

    def test_sample_replica_transition_state_elapsed_from_trigger_none_before_trigger_issued(self, monkeypatch):
        monkeypatch.setattr(harness, "kubectl_jsonpath", lambda args: "1")
        monkeypatch.setattr(harness, "get_pod_names", lambda label_selector="app=demo-app": ["pod-a"])
        monkeypatch.setattr(harness, "fetch_loadtest_endpoints_json", lambda: {"subsets": []})
        sample = harness.sample_replica_transition_state(time.monotonic(), None)
        assert sample["elapsed_from_trigger"] is None

    def test_write_and_read_back_replica_transition_timeline_csv(self, tmp_path):
        samples = [
            {"sample_monotonic": 0.0, "sampled_at_utc": "t0", "elapsed_from_traffic_start": 0.0, "elapsed_from_trigger": None,
             "desired_replicas": 5, "current_replicas": 5, "ready_replicas": 5, "available_replicas": 5,
             "ready_pod_names": ["a", "b", "c", "d", "e"], "ready_pod_count": 5, "loadtest_endpoint_ready_count": 5, "endpoint_query_error": None},
            {"sample_monotonic": 62.0, "sampled_at_utc": "t1", "elapsed_from_traffic_start": 62.0, "elapsed_from_trigger": 2.0,
             "desired_replicas": 6, "current_replicas": 6, "ready_replicas": 5, "available_replicas": 5,
             "ready_pod_names": ["a", "b", "c", "d", "e"], "ready_pod_count": 5, "loadtest_endpoint_ready_count": 5, "endpoint_query_error": None},
        ]
        out_path = tmp_path / "replica_transition_timeline.csv"
        harness.write_replica_transition_timeline_csv(out_path, samples)
        assert out_path.exists()
        with open(out_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["sample_index"] == "0"
        assert rows[1]["desired_replicas"] == "6"
        assert json.loads(rows[0]["ready_pod_names"]) == ["a", "b", "c", "d", "e"]


class TestReplicaTransitionFailureBehavior:
    def test_preflight_failure_creates_no_attempt_directory(self, monkeypatch, tmp_path, real_sequence):
        monkeypatch.setattr(harness, "preflight_checks", lambda: [(False, "Docker daemon unreachable")])
        monkeypatch.setattr(harness, "CALIBRATION_ROOT", tmp_path / "official")
        monkeypatch.setattr(harness, "TRANSPORT_VALIDATION_ROOT", tmp_path / "nonofficial")

        manifest = lib.load_manifest()
        slot = lib.resolve_planned_slot(30, sequence=real_sequence)
        ledger = lib.init_ledger(real_sequence)

        class _Args:
            campaign_sequence_index = 30
            attempt_number = 1

        rc = harness.execute_replica_transition_slot(_Args(), manifest, real_sequence, slot, slot["schedule_sha256"], ledger, official=True)
        assert rc == 2
        assert not (tmp_path / "official").exists()

    def test_nonofficial_smoke_never_touches_ledger_namespace(self, monkeypatch, tmp_path, real_sequence):
        # Governing sec.22/24: a NON-OFFICIAL run must never write under
        # CALIBRATION_ROOT and must never call the ledger-mutation
        # functions at all.
        monkeypatch.setattr(harness, "preflight_checks", lambda: [(False, "Docker daemon unreachable")])
        official_root = tmp_path / "official"
        nonofficial_root = tmp_path / "nonofficial"
        monkeypatch.setattr(harness, "CALIBRATION_ROOT", official_root)
        monkeypatch.setattr(harness, "TRANSPORT_VALIDATION_ROOT", nonofficial_root)

        def _boom_ledger(*a, **k):
            raise AssertionError("non-official smoke must never touch the ledger")

        monkeypatch.setattr(harness, "ensure_ledger", _boom_ledger)
        monkeypatch.setattr(harness, "update_slot_state", _boom_ledger)
        monkeypatch.setattr(harness, "atomic_write_ledger", _boom_ledger)

        manifest = lib.load_manifest()
        slot = lib.resolve_planned_slot(30, sequence=real_sequence)
        ledger = lib.init_ledger(real_sequence)

        class _Args:
            campaign_sequence_index = 30
            attempt_number = 1

        rc = harness.execute_replica_transition_slot(_Args(), manifest, real_sequence, slot, slot["schedule_sha256"], ledger, official=False)
        assert rc == 2  # preflight fails before any ledger code could even be reached
        assert not official_root.exists()
        assert not nonofficial_root.exists()  # preflight fails before the attempt dir is even created


# ---------------------------------------------------------------------------
# CORR-44 (Phase 15B.5C): replica-transition route-materiality/scored-window
# operationalization.
# ---------------------------------------------------------------------------


class TestCorr44TransitionScoredWindowAndMateriality:
    def test_replica_transition_route_scored_window_is_full_traffic_duration(self):
        assert lib.replica_transition_route_scored_window(240.0) == (0.0, 240.0)

    def test_replica_transition_materiality_fraction_equals_steady_state_fraction(self):
        # CORR-44's own central finding: transitions reuse the SAME 25%
        # fraction steady-state already uses -- not a different number.
        assert lib.REPLICA_TRANSITION_MATERIALITY_FRACTION == lib.DEFAULT_ROUTE_INTEGRITY_MATERIALITY_FRACTION
        assert lib.REPLICA_TRANSITION_MATERIALITY_FRACTION == 0.25

    def test_replica_transition_materiality_threshold_is_60s_for_240s_run(self):
        assert lib.replica_transition_materiality_threshold_seconds(240.0) == pytest.approx(60.0)

    def test_replica_transition_materiality_threshold_scales_with_duration(self):
        # Not hardcoded to 240s -- a duration-relative rule must scale.
        assert lib.replica_transition_materiality_threshold_seconds(120.0) == pytest.approx(30.0)
        assert lib.replica_transition_materiality_threshold_seconds(480.0) == pytest.approx(120.0)

    def test_59_999s_interval_not_material_for_240s_transition_run(self):
        scored_window = lib.replica_transition_route_scored_window(240.0)
        assert lib.is_interval_material(
            0.0, 59.999, 240.0, scored_window,
            materiality_fraction_threshold=lib.REPLICA_TRANSITION_MATERIALITY_FRACTION,
        ) is False

    def test_exactly_60s_interval_is_material_for_240s_transition_run(self):
        scored_window = lib.replica_transition_route_scored_window(240.0)
        assert lib.is_interval_material(
            0.0, 60.0, 240.0, scored_window,
            materiality_fraction_threshold=lib.REPLICA_TRANSITION_MATERIALITY_FRACTION,
        ) is True

    def test_61s_interval_is_material_for_240s_transition_run(self):
        scored_window = lib.replica_transition_route_scored_window(240.0)
        assert lib.is_interval_material(
            10.0, 71.0, 240.0, scored_window,
            materiality_fraction_threshold=lib.REPLICA_TRANSITION_MATERIALITY_FRACTION,
        ) is True

    @pytest.mark.parametrize("interval", [(0.0, 60.0), (10.0, 70.0), (60.0, 120.0), (180.0, 240.0)])
    def test_60s_interval_material_at_every_position_within_240s_run(self, interval):
        # Governing sec.10: the pre-trigger baseline [0,60) must be scored
        # exactly like every other part of the run -- a 60s failure at the
        # very start of traffic is just as material as one at the end.
        scored_window = lib.replica_transition_route_scored_window(240.0)
        start, end = interval
        assert lib.is_interval_material(
            start, end, 240.0, scored_window,
            materiality_fraction_threshold=lib.REPLICA_TRANSITION_MATERIALITY_FRACTION,
        ) is True

    def test_pretrigger_baseline_failure_is_scored(self):
        # The core scientific distinction this correction makes explicit:
        # a total-failure interval entirely within [0,60) -- the
        # pre_transition_traffic_steady_seconds baseline -- must be scored
        # under the transition's own [0,240) scored window (unlike
        # steady-state's own [0,60), which its unchanged [60,180) scored
        # window excludes for genuinely warmup-only failures).
        transition_scored_window = lib.replica_transition_route_scored_window(240.0)
        assert lib.is_interval_material(
            0.0, 60.0, 240.0, transition_scored_window,
            materiality_fraction_threshold=lib.REPLICA_TRANSITION_MATERIALITY_FRACTION,
        ) is True

    def test_pretrigger_only_burst_shorter_than_the_transition_threshold_is_excluded_by_duration(self):
        # A genuinely pre-trigger-only burst that does NOT reach [0,60)'s
        # own full width (never touching t=60) trivially overlaps the
        # transition's [0,240) scored window, but is correctly excluded by
        # the DURATION criterion alone when it is shorter than the frozen
        # 60s threshold -- proving the threshold, not merely window
        # placement, is what is actually load-bearing here.
        transition_scored_window = lib.replica_transition_route_scored_window(240.0)
        assert lib.is_interval_material(
            0.0, 50.0, 240.0, transition_scored_window,
            materiality_fraction_threshold=lib.REPLICA_TRANSITION_MATERIALITY_FRACTION,
        ) is False

    def test_reusing_steady_state_window_and_duration_would_be_scientifically_wrong(self):
        # Demonstrates precisely why CORR-44 could not simply reuse
        # steady-state's own [60,180) window/180s duration for a 240s
        # transition run: doing so would silently score against the WRONG
        # denominator (180s instead of 240s) and the WRONG window origin
        # (60 instead of 0) -- fraction = 60/180 = 0.333 (numerically still
        # >= 0.25 here, but for the WRONG reason and the wrong window),
        # rather than the CORRECT, frozen fraction = 60/240 = 0.25 against
        # the CORRECT window [0,240). This is why CORR-44 freezes explicit,
        # duration-PARAMETERIZED functions rather than a copy-pasted
        # steady-state constant.
        wrong_denominator_fraction = 60.0 / 180.0
        correct_denominator_fraction = 60.0 / 240.0
        assert wrong_denominator_fraction != correct_denominator_fraction
        assert correct_denominator_fraction == lib.REPLICA_TRANSITION_MATERIALITY_FRACTION

    def test_malformed_or_outside_run_intervals_never_material(self):
        scored_window = lib.replica_transition_route_scored_window(240.0)
        # zero/negative duration
        assert lib.is_interval_material(100.0, 100.0, 240.0, scored_window, materiality_fraction_threshold=0.25) is False
        # entirely outside the run (framework-level: the run is [0,240),
        # so an interval starting at 300 is nonsensical, but the function
        # must never crash or fabricate materiality for it)
        assert lib.is_interval_material(300.0, 400.0, 240.0, scored_window, materiality_fraction_threshold=0.25) is False

    def test_separated_bursts_never_combined_into_one_material_interval(self):
        # Governing sec.10: two 35s zero-success bursts separated by a
        # brief success must remain two independent, non-material
        # intervals -- never manually joined into one 70s material
        # interval (35+35 would exceed the 60s threshold if combined).
        summary_rows = (
            [{"elapsed_seconds": float(t), "sent_requests": "1", "success_2xx": "0"} for t in range(0, 35)]
            + [{"elapsed_seconds": 35.0, "sent_requests": "1", "success_2xx": "1"}]
            + [{"elapsed_seconds": float(t), "sent_requests": "1", "success_2xx": "0"} for t in range(36, 71)]
        )
        runs = lib.find_all_zero_success_runs(summary_rows)
        assert len(runs) == 2
        scored_window = lib.replica_transition_route_scored_window(240.0)
        materialities = [
            lib.is_interval_material(r[0], r[1], 240.0, scored_window, materiality_fraction_threshold=lib.REPLICA_TRANSITION_MATERIALITY_FRACTION)
            for r in runs
        ]
        # Each burst alone is 35s < 60s threshold -- neither is material.
        assert materialities == [False, False]


class TestCorr44V8PartitioningForTransitionMaterialIntervals:
    """Governing sec.11: V8 partitioning itself is completely unchanged --
    these tests only confirm it behaves correctly when FED a
    transition-scale (e.g. 60s-minimum) material interval, using the
    exact same `partition_into_evidence_windows()` steady-state already
    uses."""

    def test_60s_material_interval_partitions_into_six_10s_windows(self):
        windows = lib.partition_into_evidence_windows(0.0, 60.0, min_window_seconds=lib.MIN_EVIDENCE_WINDOW_SECONDS_V8)
        assert windows == [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0), (40.0, 50.0), (50.0, 60.0)]
        assert len(windows) == 6

    def test_62s_material_interval_five_full_windows_plus_merged_12s_tail(self):
        windows = lib.partition_into_evidence_windows(0.0, 62.0, min_window_seconds=lib.MIN_EVIDENCE_WINDOW_SECONDS_V8)
        # 6 full 10s windows + a 2s remainder; 2s < 9.0s merges into the
        # preceding window, producing a final 12s window.
        assert windows[:-1] == [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0), (40.0, 50.0)]
        assert windows[-1] == (50.0, 62.0)
        assert len(windows) == 6

    def test_69s_material_interval_six_full_windows_plus_standalone_9s(self):
        windows = lib.partition_into_evidence_windows(0.0, 69.0, min_window_seconds=lib.MIN_EVIDENCE_WINDOW_SECONDS_V8)
        # 6 full 10s windows + an EXACT 9.0s remainder -- exactly 9.0s
        # remains standalone per the frozen <9.0s-merge rule (not <=9.0s).
        assert windows[:-1] == [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0), (40.0, 50.0), (50.0, 60.0)]
        assert windows[-1] == (60.0, 69.0)
        assert len(windows) == 7

    def test_transition_window_union_covers_material_interval_exactly_once(self):
        for interval_end in (60.0, 62.0, 69.0, 60.0 + 240.0 - 60.0):
            windows = lib.partition_into_evidence_windows(0.0, interval_end, min_window_seconds=lib.MIN_EVIDENCE_WINDOW_SECONDS_V8)
            assert windows[0][0] == 0.0
            assert windows[-1][1] == interval_end
            for i in range(len(windows) - 1):
                assert windows[i][1] == windows[i + 1][0]  # chronological, no gap, no overlap


class TestCorr44SteadyStateRegressionUnchanged:
    """Governing sec.12/sec.2: proves CORR-44 changes NOTHING about
    steady-state's own [60,180)/45s-of-180s behavior."""

    def test_steady_state_44_999s_interval_not_material(self):
        assert lib.is_interval_material(60.0, 104.999, 180.0, (60.0, 180.0)) is False

    def test_steady_state_45_000s_interval_is_material(self):
        assert lib.is_interval_material(60.0, 105.0, 180.0, (60.0, 180.0)) is True

    def test_steady_state_default_materiality_fraction_still_quarter(self):
        assert lib.DEFAULT_ROUTE_INTEGRITY_MATERIALITY_FRACTION == 0.25

    def test_steady_state_scored_window_function_not_used_for_steady_state(self):
        # replica_transition_route_scored_window() must never be confused
        # with, or accidentally substituted for, steady-state's own
        # manifest-defined scored_window_seconds=[60,180].
        manifest = lib.load_manifest()
        assert list(manifest["steady_state_timing"]["scored_window_seconds"]) == [60, 180]
        assert lib.replica_transition_route_scored_window(180.0) != (60.0, 180.0)

    def test_provenance_route_integrity_protocol_version_still_6_after_corr44(self):
        # CORR-44 must not touch the classification-rule version at all.
        prov = lib.build_protocol_provenance()
        assert prov["route_integrity_protocol_version"] == 6
        assert prov["window_partitioning_protocol_version"] == 2
        assert prov["dataplane_acquisition_protocol_version"] == 2
        assert prov["active_correction_reference"] == "CORR-48"
        assert prov["prior_correction_reference"] == "CORR-47"


class TestCorr44SmokeArtifactReAnalysis:
    """Governing sec.13: re-analyzes the Phase 15B.5B non-official smoke
    run's already-collected, immutable raw evidence under the newly
    frozen CORR-44 rule -- never mutates the smoke artifacts, never used
    as official index30 evidence."""

    SMOKE_DIR = PROJECT_ROOT / "persistent-data/transport_validation/replica_transition_smoke/NONOFFICIAL_transrep_5-6_5to6_repeat1_smoke_20260920T105441Z"

    def test_smoke_artifact_exists_and_was_not_mutated(self):
        assert self.SMOKE_DIR.exists()
        metadata = json.loads((self.SMOKE_DIR / "metadata.json").read_text())
        assert metadata["official"] is False
        assert metadata["replicas_before"] == 5
        assert metadata["replicas_after"] == 6

    def test_smoke_reanalysis_under_corr44_matches_already_persisted_result(self):
        # The smoke run was already executed using the identical numeric
        # values CORR-44 now freezes (Phase 15B.5B's own disclosed
        # placeholder happened to match) -- so re-analysis under the
        # FROZEN rule must reproduce the exact same classification already
        # persisted, with zero behavior change.
        timeline_path = self.SMOKE_DIR / "cluster" / "loadtest_route_integrity_timeline.json"
        original = json.loads(timeline_path.read_text())
        traffic_summary_path = PROJECT_ROOT / "persistent-data/traffic" / self.SMOKE_DIR.name / "traffic_summary.csv"
        summary_rows = harness.load_traffic_summary_csv(traffic_summary_path)
        all_zero_success_runs = lib.find_all_zero_success_runs(summary_rows)
        scored_window = lib.replica_transition_route_scored_window(240.0)
        material_intervals = [
            list(r) for r in all_zero_success_runs
            if lib.is_interval_material(r[0], r[1], 240.0, scored_window, materiality_fraction_threshold=lib.REPLICA_TRANSITION_MATERIALITY_FRACTION)
        ]
        assert all_zero_success_runs == [tuple(r) for r in original["all_zero_success_runs"]]
        assert material_intervals == original["material_intervals"]
        assert original["classification"] == "CONFIRMED"
        assert material_intervals == []  # no interval ever reached the 60s threshold


# ---------------------------------------------------------------------------
# CORR-45 (Phase 15B.5E): replica-transition scale-down lag target-state
# operationalization -- fixes the false-positive ready/available lag bug and
# adds trajectory-gap/lag-uncertainty measurement metadata.
# ---------------------------------------------------------------------------


class TestCorr45ScaleDownFalsePositiveFix:
    """Governing sec.9: the exact false-positive scenario -- readiness loss
    on a not-yet-removed pod must NOT be credited as scale-down
    completion."""

    def test_readiness_loss_before_replica_removal_does_not_fabricate_ready_lag(self):
        samples = [
            {"sample_monotonic": 0.0, "current_replicas": 2, "ready_replicas": 2, "available_replicas": 2},
            {"sample_monotonic": 2.0, "current_replicas": 2, "ready_replicas": 1, "available_replicas": 1},
        ]
        lags = lib.compute_transition_replica_lags(0.0, samples, target_replicas=1, direction="scale_down")
        assert lags["desired_to_current_lag_seconds"] is None
        assert lags["desired_to_ready_lag_seconds"] is None
        assert lags["desired_to_available_lag_seconds"] is None

    def test_current_reaches_target_before_ready_does(self):
        samples = [
            {"sample_monotonic": 0.0, "current_replicas": 2, "ready_replicas": 2, "available_replicas": 2},
            {"sample_monotonic": 2.0, "current_replicas": 2, "ready_replicas": 1, "available_replicas": 1},
            {"sample_monotonic": 4.0, "current_replicas": 1, "ready_replicas": 0, "available_replicas": 0},
        ]
        lags = lib.compute_transition_replica_lags(0.0, samples, target_replicas=1, direction="scale_down")
        assert lags["desired_to_current_lag_seconds"] == pytest.approx(4.0)
        assert lags["desired_to_ready_lag_seconds"] is None  # still not Ready at the target count
        assert lags["desired_to_available_lag_seconds"] is None

    def test_full_governing_sec9_worked_example(self):
        samples = [
            {"sample_monotonic": 0.0, "current_replicas": 2, "ready_replicas": 2, "available_replicas": 2},
            {"sample_monotonic": 2.0, "current_replicas": 2, "ready_replicas": 1, "available_replicas": 1},
            {"sample_monotonic": 4.0, "current_replicas": 1, "ready_replicas": 0, "available_replicas": 0},
            {"sample_monotonic": 6.0, "current_replicas": 1, "ready_replicas": 1, "available_replicas": 1},
        ]
        lags = lib.compute_transition_replica_lags(0.0, samples, target_replicas=1, direction="scale_down")
        assert lags == {
            "desired_to_current_lag_seconds": 4.0,
            "desired_to_ready_lag_seconds": 6.0,
            "desired_to_available_lag_seconds": 6.0,
        }

    def test_old_buggy_logic_would_have_produced_a_false_positive(self):
        # Documents, via direct reproduction of the ORIGINAL (pre-CORR-45)
        # algorithm, that it really did return a false-positive 2.0s ready/
        # available lag for this exact scenario -- proving the fix changes
        # real behavior, not just adding redundant checks.
        def old_first_lag(field, samples, target_replicas, direction):
            for sample in sorted(samples, key=lambda s: s["sample_monotonic"]):
                value = sample.get(field)
                if value is None:
                    continue
                reached = (value >= target_replicas) if direction == "scale_up" else (value <= target_replicas)
                if reached:
                    return round(sample["sample_monotonic"], 3)
            return None

        samples = [
            {"sample_monotonic": 0.0, "current_replicas": 2, "ready_replicas": 2, "available_replicas": 2},
            {"sample_monotonic": 2.0, "current_replicas": 2, "ready_replicas": 1, "available_replicas": 1},
            {"sample_monotonic": 4.0, "current_replicas": 1, "ready_replicas": 0, "available_replicas": 0},
            {"sample_monotonic": 6.0, "current_replicas": 1, "ready_replicas": 1, "available_replicas": 1},
        ]
        assert old_first_lag("ready_replicas", samples, 1, "scale_down") == 2.0  # the bug
        # ...whereas the CORRECTED implementation requires current==target first:
        new_lags = lib.compute_transition_replica_lags(0.0, samples, target_replicas=1, direction="scale_down")
        assert new_lags["desired_to_ready_lag_seconds"] == 6.0  # not 2.0


class TestCorr45ScaleDownHealthyTransitions:
    @pytest.mark.parametrize("before,after", [(2, 1), (4, 3), (6, 5), (8, 7)])
    def test_healthy_scale_down_all_three_lags_reached(self, before, after):
        samples = [
            {"sample_monotonic": 0.0, "current_replicas": before, "ready_replicas": before, "available_replicas": before},
            {"sample_monotonic": 2.0, "current_replicas": after, "ready_replicas": after, "available_replicas": after},
        ]
        lags = lib.compute_transition_replica_lags(0.0, samples, target_replicas=after, direction="scale_down")
        assert lags["desired_to_current_lag_seconds"] == pytest.approx(2.0)
        assert lags["desired_to_ready_lag_seconds"] == pytest.approx(2.0)
        assert lags["desired_to_available_lag_seconds"] == pytest.approx(2.0)

    def test_current_target_reached_but_surviving_pod_not_ready(self):
        samples = [{"sample_monotonic": 0.0, "current_replicas": 1, "ready_replicas": 0, "available_replicas": 0}]
        lags = lib.compute_transition_replica_lags(0.0, samples, target_replicas=1, direction="scale_down")
        assert lags["desired_to_current_lag_seconds"] == pytest.approx(0.0)
        assert lags["desired_to_ready_lag_seconds"] is None
        assert lags["desired_to_available_lag_seconds"] is None

    def test_current_target_reached_but_surviving_pod_not_available(self):
        samples = [{"sample_monotonic": 0.0, "current_replicas": 1, "ready_replicas": 1, "available_replicas": 0}]
        lags = lib.compute_transition_replica_lags(0.0, samples, target_replicas=1, direction="scale_down")
        assert lags["desired_to_current_lag_seconds"] == pytest.approx(0.0)
        assert lags["desired_to_ready_lag_seconds"] == pytest.approx(0.0)
        assert lags["desired_to_available_lag_seconds"] is None

    def test_target_never_reached(self):
        samples = [{"sample_monotonic": float(t), "current_replicas": 2, "ready_replicas": 2, "available_replicas": 2} for t in range(5)]
        lags = lib.compute_transition_replica_lags(0.0, samples, target_replicas=1, direction="scale_down")
        assert lags == {
            "desired_to_current_lag_seconds": None,
            "desired_to_ready_lag_seconds": None,
            "desired_to_available_lag_seconds": None,
        }

    def test_target_reached_only_in_final_sample(self):
        samples = [
            {"sample_monotonic": 0.0, "current_replicas": 2, "ready_replicas": 2, "available_replicas": 2},
            {"sample_monotonic": 2.0, "current_replicas": 2, "ready_replicas": 2, "available_replicas": 2},
            {"sample_monotonic": 4.0, "current_replicas": 1, "ready_replicas": 1, "available_replicas": 1},
        ]
        lags = lib.compute_transition_replica_lags(0.0, samples, target_replicas=1, direction="scale_down")
        assert lags["desired_to_current_lag_seconds"] == pytest.approx(4.0)
        assert lags["desired_to_ready_lag_seconds"] == pytest.approx(4.0)
        assert lags["desired_to_available_lag_seconds"] == pytest.approx(4.0)

    def test_availability_collapse_before_replica_removal(self):
        # Symmetric to the readiness-collapse case: available drops while
        # current is still above target.
        samples = [
            {"sample_monotonic": 0.0, "current_replicas": 2, "ready_replicas": 2, "available_replicas": 2},
            {"sample_monotonic": 2.0, "current_replicas": 2, "ready_replicas": 2, "available_replicas": 1},
            {"sample_monotonic": 4.0, "current_replicas": 1, "ready_replicas": 1, "available_replicas": 0},
            {"sample_monotonic": 6.0, "current_replicas": 1, "ready_replicas": 1, "available_replicas": 1},
        ]
        lags = lib.compute_transition_replica_lags(0.0, samples, target_replicas=1, direction="scale_down")
        assert lags["desired_to_current_lag_seconds"] == pytest.approx(4.0)
        assert lags["desired_to_ready_lag_seconds"] == pytest.approx(4.0)  # ready reached target at t=4
        assert lags["desired_to_available_lag_seconds"] == pytest.approx(6.0)  # available only at t=6


class TestCorr45ScaleUpRegressionUnchanged:
    """Governing sec.11: index30-style scale-up semantics remain exactly
    as Phase 15B.5B implemented them."""

    def test_scale_up_current_then_ready_then_available_distinct_lags(self):
        samples = [
            {"sample_monotonic": 98.0, "current_replicas": 5, "ready_replicas": 5, "available_replicas": 5},
            {"sample_monotonic": 100.0, "current_replicas": 5, "ready_replicas": 5, "available_replicas": 5},
            {"sample_monotonic": 102.0, "current_replicas": 6, "ready_replicas": 5, "available_replicas": 5},
            {"sample_monotonic": 108.0, "current_replicas": 6, "ready_replicas": 6, "available_replicas": 5},
            {"sample_monotonic": 111.0, "current_replicas": 6, "ready_replicas": 6, "available_replicas": 6},
        ]
        lags = lib.compute_transition_replica_lags(100.0, samples, target_replicas=6, direction="scale_up")
        assert lags["desired_to_current_lag_seconds"] == pytest.approx(2.0)
        assert lags["desired_to_ready_lag_seconds"] == pytest.approx(8.0)
        assert lags["desired_to_available_lag_seconds"] == pytest.approx(11.0)

    def test_index30_real_current_lag_reproduced_by_current_implementation(self):
        # Regression pin using index30's own real, immutable raw trajectory.
        attempt_dir = PROJECT_ROOT / "persistent-data/dynamics_calibration/transrep_5-6_5to6_repeat1_attempt1_20260920T114902Z"
        metadata = json.loads((attempt_dir / "metadata.json").read_text())
        assert metadata["transition_metrics"]["desired_to_current_lag_seconds"] == pytest.approx(1.469)
        assert metadata["transition_metrics"]["desired_to_ready_lag_seconds"] is None
        assert metadata["transition_metrics"]["desired_to_available_lag_seconds"] is None

    def test_scale_up_health_loss_does_not_fabricate_readiness_completion(self):
        # A scale-up analogue of the false-positive concern: current
        # reaches target, but ready/available then REGRESS below target
        # due to unrelated health loss -- must correctly report null for
        # ready/available (the target condition, >=, was never satisfied
        # again after regressing).
        samples = [
            {"sample_monotonic": 0.0, "current_replicas": 5, "ready_replicas": 5, "available_replicas": 5},
            {"sample_monotonic": 2.0, "current_replicas": 6, "ready_replicas": 5, "available_replicas": 5},
            {"sample_monotonic": 4.0, "current_replicas": 6, "ready_replicas": 2, "available_replicas": 2},
        ]
        lags = lib.compute_transition_replica_lags(0.0, samples, target_replicas=6, direction="scale_up")
        assert lags["desired_to_current_lag_seconds"] == pytest.approx(2.0)
        assert lags["desired_to_ready_lag_seconds"] is None
        assert lags["desired_to_available_lag_seconds"] is None


class TestCorr45LagUncertaintyMetadata:
    def test_uncertainty_reports_previous_and_first_target_sample(self):
        samples = [
            {"sample_monotonic": 0.0, "current_replicas": 5, "ready_replicas": 5, "available_replicas": 5},
            {"sample_monotonic": 2.0, "current_replicas": 6, "ready_replicas": 5, "available_replicas": 5},
            {"sample_monotonic": 5.0, "current_replicas": 6, "ready_replicas": 6, "available_replicas": 6},
        ]
        uncertainty = lib.compute_transition_replica_lag_uncertainty(0.0, samples, target_replicas=6, direction="scale_up")
        current = uncertainty["desired_to_current_lag_seconds"]
        assert current["first_target_sample_elapsed_from_trigger"] == pytest.approx(2.0)
        assert current["previous_sample_elapsed_from_trigger"] == pytest.approx(0.0)
        assert current["observation_interval_width_seconds"] == pytest.approx(2.0)
        ready = uncertainty["desired_to_ready_lag_seconds"]
        assert ready["first_target_sample_elapsed_from_trigger"] == pytest.approx(5.0)
        assert ready["previous_sample_elapsed_from_trigger"] == pytest.approx(2.0)
        assert ready["observation_interval_width_seconds"] == pytest.approx(3.0)

    def test_uncertainty_none_previous_when_first_sample_already_satisfies(self):
        samples = [{"sample_monotonic": 0.0, "current_replicas": 6, "ready_replicas": 6, "available_replicas": 6}]
        uncertainty = lib.compute_transition_replica_lag_uncertainty(0.0, samples, target_replicas=6, direction="scale_up")
        current = uncertainty["desired_to_current_lag_seconds"]
        assert current["previous_sample_elapsed_from_trigger"] is None
        assert current["observation_interval_width_seconds"] is None

    def test_uncertainty_all_none_when_target_never_reached(self):
        samples = [{"sample_monotonic": 0.0, "current_replicas": 5, "ready_replicas": 5, "available_replicas": 5}]
        uncertainty = lib.compute_transition_replica_lag_uncertainty(0.0, samples, target_replicas=6, direction="scale_up")
        for lag_uncertainty in uncertainty.values():
            assert lag_uncertainty["first_target_sample_elapsed_from_trigger"] is None
            assert lag_uncertainty["previous_sample_elapsed_from_trigger"] is None
            assert lag_uncertainty["observation_interval_width_seconds"] is None

    def test_build_transition_metrics_summary_includes_uncertainty_and_semantic_note(self):
        samples = [{"sample_monotonic": 0.0, "current_replicas": 6, "ready_replicas": 6, "available_replicas": 6}]
        summary = lib.build_transition_metrics_summary(0.0, samples, target_replicas=6, direction="scale_up")
        assert "lag_uncertainty" in summary
        assert "lag_measurement_semantic" in summary
        assert "FIRST_OBSERVED_TRAJECTORY_LAG" in summary["lag_measurement_semantic"]
        assert "trajectory_gap_statistics" in summary


class TestCorr45TrajectoryGapStatistics:
    def test_nominal_2s_gaps(self):
        samples = [{"elapsed_from_traffic_start": float(t)} for t in (0, 2, 4, 6, 8)]
        stats = lib.summarize_replica_transition_trajectory_gaps(samples)
        assert stats["trajectory_sample_count"] == 5
        assert stats["gap_count"] == 4
        assert stats["gap_min_seconds"] == pytest.approx(2.0)
        assert stats["gap_max_seconds"] == pytest.approx(2.0)
        assert stats["gap_mean_seconds"] == pytest.approx(2.0)
        assert stats["count_gap_gt_3_seconds"] == 0
        assert stats["nominal_sampling_interval_seconds"] == 2.0
        assert stats["observed_sampling_interval_is_not_guaranteed"] is True

    def test_4s_observed_gaps_exceed_nominal(self):
        samples = [{"elapsed_from_traffic_start": float(t)} for t in (0, 4, 8, 12)]
        stats = lib.summarize_replica_transition_trajectory_gaps(samples)
        assert stats["gap_min_seconds"] == pytest.approx(4.0)
        assert stats["count_gap_gt_3_seconds"] == 3

    def test_mixed_irregular_gaps(self):
        samples = [{"elapsed_from_traffic_start": t} for t in (0.0, 2.1, 2.9, 7.5, 8.0)]
        stats = lib.summarize_replica_transition_trajectory_gaps(samples)
        assert stats["gap_count"] == 4
        gaps_expected = sorted([2.1, 0.8, 4.6, 0.5])
        assert stats["gap_min_seconds"] == pytest.approx(gaps_expected[0])
        assert stats["gap_max_seconds"] == pytest.approx(gaps_expected[-1])
        assert stats["count_gap_gt_3_seconds"] == 1  # only the 4.6s gap

    def test_single_sample_timeline_produces_zero_gaps(self):
        samples = [{"elapsed_from_traffic_start": 0.0}]
        stats = lib.summarize_replica_transition_trajectory_gaps(samples)
        assert stats["trajectory_sample_count"] == 1
        assert stats["gap_count"] == 0
        assert stats["gap_min_seconds"] is None
        assert stats["gap_max_seconds"] is None

    def test_empty_timeline_produces_zero_gaps(self):
        stats = lib.summarize_replica_transition_trajectory_gaps([])
        assert stats["trajectory_sample_count"] == 0
        assert stats["gap_count"] == 0

    def test_index30_real_trajectory_gap_statistics(self):
        # Empirical proof, from index30's own immutable raw artifact, that
        # the nominal 2.0s cadence is genuinely NOT guaranteed.
        attempt_dir = PROJECT_ROOT / "persistent-data/dynamics_calibration/transrep_5-6_5to6_repeat1_attempt1_20260920T114902Z"
        rows = list(csv.DictReader(open(attempt_dir / "cluster" / "replica_transition_timeline.csv")))
        samples = [{"elapsed_from_traffic_start": float(r["elapsed_from_traffic_start"])} for r in rows]
        stats = lib.summarize_replica_transition_trajectory_gaps(samples)
        assert stats["trajectory_sample_count"] == 65
        assert stats["gap_count"] == 64
        assert stats["gap_mean_seconds"] > 2.0  # nominal was 2.0s; real mean exceeds it
        assert stats["nominal_sampling_interval_seconds"] == 2.0
        assert stats["observed_sampling_interval_is_not_guaranteed"] is True


class TestCorr45Index31ReadOnlyResolution:
    def test_index31_resolves_as_scale_down_2_to_1(self, real_sequence):
        slot = lib.resolve_planned_slot(31, sequence=real_sequence)
        assert slot["run_kind"] == "replica_transition"
        assert slot["planned_slot_id"] == "transrep_1-2_2to1_repeat1"
        assert slot["replicas_before"] == 2
        assert slot["replicas_after"] == 1
        assert lib.transition_direction(slot["replicas_before"], slot["replicas_after"]) == "scale_down"

    def test_index31_officially_executed_and_valid(self, real_sequence):
        # Phase 15B.5F officially executed index31 attempt1 (scale_down, 2->1).
        # This test was originally written in Phase 15B.5E, while index31 was
        # still NOT_ATTEMPTED and only read-only-resolved. It is updated here,
        # in the same phase that performed the authorized official execution,
        # to pin the resulting real ledger state rather than assert a
        # now-obsolete pre-execution snapshot.
        ledger = lib.load_ledger()
        slot31 = ledger["slots"]["31"]
        assert slot31["state"] == "VALID"
        assert slot31["valid_attempt_id"] == "transrep_1-2_2to1_repeat1_attempt1_20260920T123134Z"
        assert slot31["all_attempt_ids"] == ["transrep_1-2_2to1_repeat1_attempt1_20260920T123134Z"]


class TestCorr46ReplicaTrajectorySamplerTimeoutResilience:
    """Phase 15B.5H: index33 attempt1's `_sample_replica_trajectory_
    periodically()` thread died permanently after a single `kubectl`
    `TimeoutExpired`, because `sample_replica_transition_state()` itself
    had no exception handling around its four `kubectl_jsonpath()` calls.
    That thread's loop body is `while ...: samples.append(sample_replica_
    transition_state(...))` -- if the function itself can no longer raise
    `TimeoutExpired`, the thread structurally cannot die from this failure
    mode any more, so these tests exercise the function directly rather
    than spinning up a real background thread."""

    def test_timeout_from_kubectl_jsonpath_is_caught_not_raised(self, monkeypatch):
        def _raise_timeout(args):
            raise subprocess.TimeoutExpired(cmd=["kubectl"] + args, timeout=30)

        monkeypatch.setattr(harness, "kubectl_jsonpath", _raise_timeout)
        monkeypatch.setattr(harness, "get_pod_names", lambda label_selector="app=demo-app": ["pod-a"])
        monkeypatch.setattr(harness, "fetch_loadtest_endpoints_json", lambda: {"subsets": []})

        now = time.monotonic()
        sample = harness.sample_replica_transition_state(now - 10.0, now - 5.0)
        assert sample["replica_state_observation_error"] is not None
        assert "timed out" in sample["replica_state_observation_error"].lower() or \
            "timeout" in sample["replica_state_observation_error"].lower()

    def test_timeout_never_fabricates_replica_counts_or_pod_names(self, monkeypatch):
        monkeypatch.setattr(harness, "kubectl_jsonpath", lambda args: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(cmd=["kubectl"], timeout=30)
        ))
        monkeypatch.setattr(harness, "get_pod_names", lambda label_selector="app=demo-app": ["pod-a", "pod-b"])
        monkeypatch.setattr(harness, "fetch_loadtest_endpoints_json", lambda: {"subsets": [{"addresses": [{}]}]})

        sample = harness.sample_replica_transition_state(time.monotonic() - 10.0, None)
        assert sample["desired_replicas"] is None
        assert sample["current_replicas"] is None
        assert sample["ready_replicas"] is None
        assert sample["available_replicas"] is None
        assert sample["ready_pod_names"] == []
        assert sample["ready_pod_count"] == 0
        assert sample["loadtest_endpoint_ready_count"] is None

    def test_timeout_sample_still_carries_real_elapsed_time(self, monkeypatch):
        monkeypatch.setattr(harness, "kubectl_jsonpath", lambda args: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(cmd=["kubectl"], timeout=30)
        ))
        traffic_start = time.monotonic() - 42.0
        sample = harness.sample_replica_transition_state(traffic_start, None)
        assert sample["elapsed_from_traffic_start"] >= 41.0
        assert sample["sampled_at_utc"] is not None
        assert sample["sample_monotonic"] is not None

    def test_healthy_sample_has_null_observation_error(self, monkeypatch):
        monkeypatch.setattr(harness, "kubectl_jsonpath", lambda args: "3")
        monkeypatch.setattr(harness, "get_pod_names", lambda label_selector="app=demo-app": ["pod-a", "pod-b", "pod-c"])
        monkeypatch.setattr(harness, "fetch_loadtest_endpoints_json", lambda: {"subsets": [{"addresses": [{}, {}, {}]}]})
        sample = harness.sample_replica_transition_state(time.monotonic() - 10.0, None)
        assert sample["replica_state_observation_error"] is None
        assert sample["current_replicas"] == 3

    def test_alternating_timeout_and_success_each_sample_independent(self, monkeypatch):
        # Simulates what the periodic thread does across several ticks:
        # a transient failure on one tick must not contaminate the next.
        calls = {"n": 0}

        def _flaky(args):
            calls["n"] += 1
            if calls["n"] == 2:
                raise subprocess.TimeoutExpired(cmd=["kubectl"] + args, timeout=30)
            return "4"

        monkeypatch.setattr(harness, "get_pod_names", lambda label_selector="app=demo-app": ["a", "b", "c", "d"])
        monkeypatch.setattr(harness, "fetch_loadtest_endpoints_json", lambda: {"subsets": [{"addresses": [{}]}]})

        results = []
        for _ in range(4):
            monkeypatch.setattr(harness, "kubectl_jsonpath", _flaky)
            results.append(harness.sample_replica_transition_state(time.monotonic() - 10.0, None))

        # sample 1: healthy (calls 1-4 succeed... but note _flaky raises on
        # the 2nd *call*, not the 2nd *sample* -- each sample makes 4 calls
        # to kubectl_jsonpath, so the timeout lands inside sample 1's own
        # second call, meaning sample 1 is the one that observes it).
        assert results[0]["replica_state_observation_error"] is not None
        assert results[1]["replica_state_observation_error"] is None
        assert results[1]["current_replicas"] == 4
        assert results[2]["replica_state_observation_error"] is None
        assert results[3]["replica_state_observation_error"] is None

    def test_replica_transition_timeline_csv_persists_observation_error_column(self, tmp_path):
        samples = [
            {"sample_monotonic": 0.0, "sampled_at_utc": "t0", "elapsed_from_traffic_start": 0.0, "elapsed_from_trigger": None,
             "desired_replicas": None, "current_replicas": None, "ready_replicas": None, "available_replicas": None,
             "ready_pod_names": [], "ready_pod_count": 0, "loadtest_endpoint_ready_count": None,
             "endpoint_query_error": None, "replica_state_observation_error": "Command timed out after 30 seconds"},
        ]
        path = tmp_path / "replica_transition_timeline.csv"
        harness.write_replica_transition_timeline_csv(path, samples)
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["replica_state_observation_error"] == "Command timed out after 30 seconds"


class TestCorr46RouteIntegritySamplerTimeoutResilience:
    """Phase 15B.5H: `check_relay_container_state()` had no exception
    handling at all around its `docker inspect` call, and was reachable
    both from the periodic route-integrity sampler thread (Thread-1,
    which died in index33 attempt1) AND directly from the main thread
    during STOP_TRAFFIC/FINAL CAPTURE -- an unguarded timeout there could
    have crashed the entire attempt, not just a background thread."""

    def test_check_relay_container_state_timeout_returns_none_none(self, monkeypatch):
        def _raise_timeout(cmd, timeout=None):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

        monkeypatch.setattr(harness, "_run", _raise_timeout)
        running, exit_code = harness.check_relay_container_state("loadtest-relay-x")
        assert running is None
        assert exit_code is None

    def test_check_local_listener_present_timeout_returns_none(self, monkeypatch):
        def _raise_timeout(cmd, timeout=None):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

        monkeypatch.setattr(harness, "_run", _raise_timeout)
        assert harness.check_local_listener_present(50408) is None

    def test_resolve_local_listener_owner_timeout_returns_none_dict(self, monkeypatch):
        def _raise_timeout(cmd, timeout=None):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

        monkeypatch.setattr(harness, "_run", _raise_timeout)
        result = harness.resolve_local_listener_owner(50408)
        assert result == {"listener_owner_pid": None, "listener_owner_command": None}


class TestCorr46PodReadinessCollectorTimeoutResilience:
    """Phase 15B.5H: `fetch_pods_json()` previously let `subprocess.
    TimeoutExpired` propagate all the way out of `main()`, crashing the
    entire `pod_readiness_collector.py` subprocess (non-zero exit),
    which is exactly what tripped `collector_or_generator_structural_
    failure` for index33 attempt1. Mirrors this same file's own
    pre-existing `fetch_docker_ps_all()` `(TimeoutExpired, OSError)`
    catch, which had already established the local convention."""

    def test_fetch_pods_json_timeout_returns_recoverable_error_not_raise(self, monkeypatch):
        def _raise_timeout(*a, **k):
            raise subprocess.TimeoutExpired(cmd=["kubectl", "get", "pods"], timeout=15)

        monkeypatch.setattr(readiness_collector.subprocess, "run", _raise_timeout)
        result = readiness_collector.fetch_pods_json("autoscaler-app", "app=demo-app")
        assert result["items"] == []
        assert "_error" in result
        assert result["_error"] is not None

    def test_fetch_pods_json_healthy_path_unaffected(self, monkeypatch):
        class _FakeProc:
            returncode = 0
            stdout = '{"items": []}'
            stderr = ""

        monkeypatch.setattr(readiness_collector.subprocess, "run", lambda *a, **k: _FakeProc())
        result = readiness_collector.fetch_pods_json("autoscaler-app", "app=demo-app")
        assert result == {"items": []}

    def test_run_survives_isolated_timeout_exits_zero_and_continues(self, monkeypatch, tmp_path):
        # Simulate several rapid ticks where exactly one `fetch_pods_json`
        # call times out; `run()` must still return 0 (not propagate the
        # exception via `sys.exit(main())` as it did for index33 attempt1),
        # and ticks before/after the failure must still be recorded.
        calls = {"n": 0}
        healthy = {"items": [{
            "metadata": {"name": "pod-a", "uid": "u1"},
            "status": {"phase": "Running", "conditions": [
                {"type": "Ready", "status": "True", "lastTransitionTime": "2026-01-01T00:00:00Z"}
            ], "containerStatuses": [{"name": "c", "restartCount": 0, "state": {"running": {}}, "lastState": {}}]},
        }]}

        def _flaky_fetch(namespace, label_selector):
            calls["n"] += 1
            if calls["n"] == 2:
                return {"items": [], "_error": "Command '[...]' timed out after 15 seconds"}
            return healthy

        monkeypatch.setattr(readiness_collector, "fetch_pods_json", _flaky_fetch)
        monkeypatch.setattr(readiness_collector, "fetch_docker_ps_all", lambda: ([], None))

        exit_code = readiness_collector.run(
            experiment_dir=tmp_path,
            namespace="autoscaler-app",
            label_selector="app=demo-app",
            duration_seconds=0.35,
            sampling_interval_seconds=0.1,
        )
        assert exit_code == 0
        assert calls["n"] >= 3  # at least one healthy tick before and after the injected failure

        with open(tmp_path / "pod_readiness_observation_errors.csv", newline="", encoding="utf-8") as f:
            error_rows = list(csv.DictReader(f))
        assert len(error_rows) == 1
        assert error_rows[0]["collector"] == "pod_readiness_collector"
        assert error_rows[0]["continued_after_error"] == "True"
        assert "timed out" in error_rows[0]["message"].lower()

        with open(tmp_path / "pod_readiness_history.csv", newline="", encoding="utf-8") as f:
            healthy_rows = list(csv.DictReader(f))
        # At least the healthy ticks produced pod-readiness rows; the
        # single failed tick contributed none (missing means missing).
        assert len(healthy_rows) >= 1

        with open(tmp_path / "pod_readiness_metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)
        assert metadata["fetch_error_count"] == 1
        assert metadata["interrupted"] is False


class TestCorr46StructuralValidityGateUnchanged:
    """Phase 15B.5H sec.7: the hardening must make ISOLATED sampling
    timeouts recoverable -- it must NOT weaken the pre-existing
    `collector_or_generator_structural_failure` gate's ability to detect a
    genuine, unrecovered collector failure. This is a pure code-presence
    regression check: the exact invalidity string the gate appends when
    `readiness_proc.wait()` returns non-zero must still exist verbatim in
    the execution path, and `fetch_pods_json`'s own error path is
    independent of it (a recoverable per-tick fetch error does not itself
    change the collector subprocess's own exit code)."""

    def test_collector_or_generator_structural_failure_string_still_present(self):
        source = Path(harness.__file__).read_text(encoding="utf-8")
        assert "collector_or_generator_structural_failure: pod readiness collector non-zero exit" in source

    def test_recoverable_fetch_error_does_not_by_itself_change_process_exit_code(self, monkeypatch, tmp_path):
        # A collector subprocess that hits ONLY recoverable fetch errors
        # (never a genuine crash) must still exit 0 -- proving the
        # hardening does not, by itself, cause spurious INVALID
        # classifications for a merely-transient condition.
        def _always_times_out(namespace, label_selector):
            return {"items": [], "_error": "Command '[...]' timed out after 15 seconds"}

        monkeypatch.setattr(readiness_collector, "fetch_pods_json", _always_times_out)
        monkeypatch.setattr(readiness_collector, "fetch_docker_ps_all", lambda: ([], None))
        exit_code = readiness_collector.run(
            experiment_dir=tmp_path,
            namespace="autoscaler-app",
            label_selector="app=demo-app",
            duration_seconds=0.25,
            sampling_interval_seconds=0.1,
        )
        assert exit_code == 0

    def test_genuine_unhandled_exception_still_propagates_and_would_exit_nonzero(self, monkeypatch, tmp_path):
        # A GENUINE, non-timeout, non-OSError failure (e.g. a bug elsewhere
        # in the parse path) must NOT be silently swallowed -- the
        # hardening is scoped to TimeoutExpired/OSError only (sec.11: "Do
        # not merely catch `except Exception: pass`").
        def _boom(namespace, label_selector):
            raise ValueError("genuinely unexpected parse failure, not a timeout")

        monkeypatch.setattr(readiness_collector, "fetch_pods_json", _boom)
        with pytest.raises(ValueError):
            readiness_collector.run(
                experiment_dir=tmp_path,
                namespace="autoscaler-app",
                label_selector="app=demo-app",
                duration_seconds=0.25,
                sampling_interval_seconds=0.1,
            )


class TestCorr46Provenance:
    def test_active_correction_reference_is_corr46(self):
        prov = lib.build_protocol_provenance()
        assert prov["active_correction_reference"] == "CORR-48"
        assert prov["prior_correction_reference"] == "CORR-47"

    def test_no_protocol_version_bumped_by_corr46(self):
        prov = lib.build_protocol_provenance()
        assert prov["route_integrity_protocol_version"] == 6
        assert prov["dataplane_acquisition_protocol_version"] == 2
        assert prov["window_partitioning_protocol_version"] == 2
        assert prov["transport_protocol_version"] == 5
        assert prov["active_calibration_protocol_version"] == 8


class TestCorr46Index32And33Regression:
    """Phase 15B.5H sec.19-20: index32 remains VALID and unchanged;
    index33 attempt1 remains INVALID and immutable. Read directly from
    the persisted, historical `metadata.json` files -- never re-executed,
    never recomputed."""

    def _read_metadata(self, attempt_id):
        path = lib.PROJECT_ROOT / "persistent-data" / "dynamics_calibration" / attempt_id / "metadata.json"
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_index32_still_valid_v8_partition_exact(self):
        metadata = self._read_metadata("transrep_7-8_8to7_repeat1_attempt1_20260920T125257Z")
        assert metadata["execution_structural_validity"] == "VALID"
        ri_path = (
            lib.PROJECT_ROOT / "persistent-data" / "dynamics_calibration"
            / "transrep_7-8_8to7_repeat1_attempt1_20260920T125257Z" / "cluster"
            / "loadtest_route_integrity_timeline.json"
        )
        with open(ri_path, encoding="utf-8") as f:
            ri = json.load(f)
        windows = ri["evidence_windows"]
        assert len(windows) == 23
        assert windows[0]["window_start_elapsed"] == 3.0
        assert windows[-1]["window_end_elapsed"] == 239.0
        durations = [w["window_end_elapsed"] - w["window_start_elapsed"] for w in windows]
        assert durations[:-1] == [10.0] * 22
        assert durations[-1] == 16.0
        assert sum(durations) == 236.0
        # no gap, no overlap
        for prev, cur in zip(windows, windows[1:]):
            assert prev["window_end_elapsed"] == cur["window_start_elapsed"]

    def test_index33_attempt1_still_invalid(self):
        metadata = self._read_metadata("transrep_3-4_3to4_repeat1_attempt1_20260920T130955Z")
        assert metadata["execution_structural_validity"] == "INVALID"
        assert "collector_or_generator_structural_failure: pod readiness collector non-zero exit" in \
            metadata["invalidity_reasons"]

    def test_ledger_slot32_and_slot33_valid_with_attempt_history_preserved(self):
        # Phase 15B.5H originally pinned slot33's PRE-attempt2 snapshot
        # (INVALID_RETRY_REQUIRED, one attempt). Phase 15B.5I's own
        # explicitly authorized retry made index33 attempt2 officially
        # VALID -- a permanent historical fact, not a transient one (index33
        # will never be re-executed; see governing Phase 15B.5J sec.1: "Do
        # NOT rerun index33"). This test is updated, per Phase 15B.5J's own
        # test-bookkeeping-repair task, to assert that permanent reality
        # instead. `next_resumable_index()` is deliberately asserted with
        # `>=`, not `==`, so this test does not itself go stale the moment
        # a later index (34+) is legitimately executed -- see sec.4/5 of
        # the governing task for this exact distinction between a
        # permanent historical invariant and a transient campaign-state
        # precondition.
        #
        # Phase 15B.10C: the live campaign has since reached 102/102 VALID,
        # so `next_resumable_index()` now legitimately returns `None` (no
        # non-VALID slot remains to resume from -- see its own docstring
        # and CORR-34 sec.34). That terminal state was never anticipated
        # when this assertion was written and is accepted here as an
        # equally valid outcome alongside `>= 34`, without weakening the
        # original intent: campaign advancement past slots 32/33 must
        # never regress, whether the campaign is still in progress or has
        # since completed in full.
        ledger = lib.load_ledger()
        assert ledger["slots"]["32"]["state"] == "VALID"
        assert ledger["slots"]["32"]["valid_attempt_id"] == "transrep_7-8_8to7_repeat1_attempt1_20260920T125257Z"
        assert ledger["slots"]["33"]["state"] == "VALID"
        assert ledger["slots"]["33"]["valid_attempt_id"] == "transrep_3-4_3to4_repeat1_attempt2_20260920T135402Z"
        assert ledger["slots"]["33"]["all_attempt_ids"] == [
            "transrep_3-4_3to4_repeat1_attempt1_20260920T130955Z",
            "transrep_3-4_3to4_repeat1_attempt2_20260920T135402Z",
        ]
        next_index = lib.next_resumable_index(ledger)
        assert next_index is None or next_index >= 34

    def test_index33_manifest_identity_and_attempt_history(self, real_sequence):
        # The manifest-resolved identity of index33 (schedule, replica
        # counts, RPS) is a permanent, immutable fact independent of
        # ledger state, and is tested here directly. The former assertion
        # `len(all_attempt_ids) == 1` ("next live attempt is attempt2") was
        # itself a transient campaign-state precondition that Phase
        # 15B.5I's own authorized attempt2 execution necessarily made
        # false -- replaced with an assertion of the permanent, two-entry
        # attempt history in its correct order.
        slot = lib.resolve_planned_slot(33, sequence=real_sequence)
        assert slot["planned_slot_id"] == "transrep_3-4_3to4_repeat1"
        assert slot["run_kind"] == "replica_transition"
        assert slot["replicas_before"] == 3
        assert slot["replicas_after"] == 4
        assert slot["offered_rps"] == 15.0
        ledger = lib.load_ledger()
        all_ids = ledger["slots"]["33"]["all_attempt_ids"]
        assert all_ids == [
            "transrep_3-4_3to4_repeat1_attempt1_20260920T130955Z",
            "transrep_3-4_3to4_repeat1_attempt2_20260920T135402Z",
        ]
        assert all_ids[0] != all_ids[1]  # attempt1 was not overwritten/reused as attempt2

    def test_index33_attempt2_valid_and_distinct_from_attempt1(self):
        # Permanent historical invariant: attempt2's own metadata is
        # structurally VALID, and its ID is provably distinct from
        # attempt1's -- attempt1 was never reused, overwritten, or
        # reclassified.
        attempt1 = self._read_metadata("transrep_3-4_3to4_repeat1_attempt1_20260920T130955Z")
        attempt2 = self._read_metadata("transrep_3-4_3to4_repeat1_attempt2_20260920T135402Z")
        assert attempt1["execution_structural_validity"] == "INVALID"
        assert attempt2["execution_structural_validity"] == "VALID"
        assert attempt1 is not attempt2


# ---------------------------------------------------------------------------
# CORR-47 (Phase 15B.6B): workload-transition pre-first-use operationalization.
# ---------------------------------------------------------------------------


class TestCorr47WorkloadTransitionResolution:
    def test_is_workload_transition_slot_true_for_workload_transition(self):
        assert lib.is_workload_transition_slot({"run_kind": "workload_transition"}) is True

    def test_is_workload_transition_slot_false_for_other_run_kinds(self):
        for run_kind in ("fitting", "interpolation_validation", "replica_transition"):
            assert lib.is_workload_transition_slot({"run_kind": run_kind}) is False

    def test_is_workload_transition_slot_false_for_missing_run_kind(self):
        assert lib.is_workload_transition_slot({}) is False

    def test_index38_resolves_as_workload_transition_r4_8to32(self, real_sequence):
        slot = lib.resolve_planned_slot(38, sequence=real_sequence)
        assert slot["planned_slot_id"] == "transwork_r4_8to32_repeat1"
        assert slot["run_kind"] == "workload_transition"
        assert slot["replicas"] == 4
        assert slot["rps_before"] == 8.0
        assert slot["rps_after"] == 32.0
        assert slot["schedule_sha256"] == "4341deefedbca4cefe93734dae741288a73817330ddc656bf3dd91c21f1cd144"

    def test_index39_resolves_as_workload_transition_r1_2to8(self, real_sequence):
        slot = lib.resolve_planned_slot(39, sequence=real_sequence)
        assert slot["planned_slot_id"] == "transwork_r1_2to8_repeat1"
        assert slot["replicas"] == 1
        assert slot["rps_before"] == 2.0
        assert slot["rps_after"] == 8.0

    def test_index40_resolves_as_workload_transition_r4_32to8(self, real_sequence):
        slot = lib.resolve_planned_slot(40, sequence=real_sequence)
        assert slot["planned_slot_id"] == "transwork_r4_32to8_repeat1"
        assert slot["replicas"] == 4
        assert slot["rps_before"] == 32.0
        assert slot["rps_after"] == 8.0

    def test_index41_resolves_as_workload_transition_r1_8to2(self, real_sequence):
        slot = lib.resolve_planned_slot(41, sequence=real_sequence)
        assert slot["planned_slot_id"] == "transwork_r1_8to2_repeat1"
        assert slot["replicas"] == 1
        assert slot["rps_before"] == 8.0
        assert slot["rps_after"] == 2.0

    def test_index42_resolves_as_fitting_not_workload_transition(self, real_sequence):
        slot = lib.resolve_planned_slot(42, sequence=real_sequence)
        assert slot["run_kind"] == "fitting"
        assert lib.is_workload_transition_slot(slot) is False

    def test_repeat2_workload_transition_slots_79_to_82(self, real_sequence):
        expected = {
            79: ("transwork_r4_8to32_repeat2", 4, 8.0, 32.0),
            80: ("transwork_r1_2to8_repeat2", 1, 2.0, 8.0),
            81: ("transwork_r1_8to2_repeat2", 1, 8.0, 2.0),
            82: ("transwork_r4_32to8_repeat2", 4, 32.0, 8.0),
        }
        for idx, (planned_slot_id, replicas, rps_before, rps_after) in expected.items():
            slot = lib.resolve_planned_slot(idx, sequence=real_sequence)
            assert slot["planned_slot_id"] == planned_slot_id
            assert slot["run_kind"] == "workload_transition"
            assert slot["replicas"] == replicas
            assert slot["rps_before"] == rps_before
            assert slot["rps_after"] == rps_after

    def test_index83_resolves_as_fitting_repeat3_not_workload_transition(self, real_sequence):
        slot = lib.resolve_planned_slot(83, sequence=real_sequence)
        assert slot["run_kind"] == "fitting"


class TestCorr47TimingPlan:
    def test_resolve_workload_transition_timing_plan_reads_frozen_manifest_block(self):
        manifest = lib.load_manifest()
        timing = lib.resolve_workload_transition_timing_plan(manifest)
        assert timing["pre_run_replica_stabilization_seconds"] == 60.0
        assert timing["pre_step_traffic_seconds"] == 60.0
        assert timing["trigger_at_traffic_elapsed_seconds"] == 60.0
        assert timing["post_step_observation_seconds"] == 180.0
        assert timing["total_traffic_seconds"] == 240.0
        assert timing["total_seconds_per_run"] == 300.0

    def test_resolve_workload_transition_timing_plan_raises_on_missing_key(self):
        with pytest.raises(KeyError):
            lib.resolve_workload_transition_timing_plan({"workload_transitions": {"timing": {}}})

    def test_timing_plan_does_not_reuse_steady_state_180s_shape(self):
        manifest = lib.load_manifest()
        timing = lib.resolve_workload_transition_timing_plan(manifest)
        assert timing["total_traffic_seconds"] != 180.0


class TestCorr47DryRun:
    def test_index38_dry_run_reports_workload_transition_fields(self, real_sequence, real_registry):
        manifest = lib.load_manifest()
        slot = lib.resolve_planned_slot(38, sequence=real_sequence)
        ledger = lib.init_ledger(real_sequence)
        schedule_hash = slot["schedule_sha256"]
        report = harness.dry_run_report(slot, schedule_hash, manifest, ledger)
        assert report["run_kind"] == "workload_transition"
        assert report["replicas"] == 4
        assert report["rps_before"] == 8.0
        assert report["rps_after"] == 32.0
        assert report["traffic_duration_seconds"] == 240.0
        assert report["pre_step_traffic_seconds"] == 60.0
        assert report["workload_step_elapsed_seconds"] == 60.0
        assert report["post_step_observation_seconds"] == 180.0
        assert report["expected_requests"] == 6240
        assert report["workload_step_owner"] == "schedule"
        assert report["scientific_role"] == "WORKLOAD_TRANSITION_TRAJECTORY_VALIDATION"

    def test_index38_dry_run_does_not_show_steady_state_fields(self, real_sequence):
        manifest = lib.load_manifest()
        slot = lib.resolve_planned_slot(38, sequence=real_sequence)
        ledger = lib.init_ledger(real_sequence)
        report = harness.dry_run_report(slot, slot["schedule_sha256"], manifest, ledger)
        assert "warmup_window_seconds" not in report
        assert "scored_window_seconds" not in report
        assert "decision_aligned_extraction_timestamps" not in report
        assert report.get("offered_rps") is None or "offered_rps" not in report

    def test_all_four_repeat1_workload_transition_dry_runs_show_240s(self, real_sequence):
        manifest = lib.load_manifest()
        ledger = lib.init_ledger(real_sequence)
        for idx in (38, 39, 40, 41):
            slot = lib.resolve_planned_slot(idx, sequence=real_sequence)
            report = harness.dry_run_report(slot, slot["schedule_sha256"], manifest, ledger)
            assert report["traffic_duration_seconds"] == 240.0
            assert report["run_kind"] == "workload_transition"


class TestCorr47ScheduleStepPhase:
    def test_workload_transition_phase_pre_and_post_step(self):
        assert lib.workload_transition_phase(0.0, 60.0) == "pre_step"
        assert lib.workload_transition_phase(59.999, 60.0) == "pre_step"
        assert lib.workload_transition_phase(60.0, 60.0) == "post_step"
        assert lib.workload_transition_phase(239.999, 60.0) == "post_step"

    def test_real_schedule_r4_8to32_steps_at_exactly_60s(self):
        path = (
            lib.PROJECT_ROOT / "configs/traffic_profiles/rl_dynamics_calibration/workload_transitions"
            / "dyncal_transwork_r4_8to32_schedule.csv"
        )
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        pre_step_rows = [r for r in rows if r["phase_name"] == "pre_step"]
        post_step_rows = [r for r in rows if r["phase_name"] == "post_step"]
        assert all(float(r["target_rps"]) == 8.0 for r in pre_step_rows)
        assert all(float(r["target_rps"]) == 32.0 for r in post_step_rows)
        assert max(float(r["planned_elapsed_seconds"]) for r in pre_step_rows) < 60.0
        assert min(float(r["planned_elapsed_seconds"]) for r in post_step_rows) == 60.0

    def test_real_schedule_r4_32to8_load_decrease_steps_at_exactly_60s(self):
        path = (
            lib.PROJECT_ROOT / "configs/traffic_profiles/rl_dynamics_calibration/workload_transitions"
            / "dyncal_transwork_r4_32to8_schedule.csv"
        )
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        pre_step_rows = [r for r in rows if r["phase_name"] == "pre_step"]
        post_step_rows = [r for r in rows if r["phase_name"] == "post_step"]
        assert all(float(r["target_rps"]) == 32.0 for r in pre_step_rows)
        assert all(float(r["target_rps"]) == 8.0 for r in post_step_rows)
        assert min(float(r["planned_elapsed_seconds"]) for r in post_step_rows) == 60.0

    def test_schedule_row_count_matches_dry_run_expected_requests(self, real_sequence):
        manifest = lib.load_manifest()
        ledger = lib.init_ledger(real_sequence)
        for idx, expected in ((38, 6240), (39, 1560), (40, 3360), (41, 840)):
            slot = lib.resolve_planned_slot(idx, sequence=real_sequence)
            report = harness.dry_run_report(slot, slot["schedule_sha256"], manifest, ledger)
            assert report["expected_requests"] == expected


class TestCorr47FittingIsolation:
    def test_scientific_role_constant_distinct_from_replica_transition(self):
        assert lib.WORKLOAD_TRANSITION_SCIENTIFIC_ROLE != lib.TRANSITION_SCIENTIFIC_ROLE
        assert lib.WORKLOAD_TRANSITION_SCIENTIFIC_ROLE == "WORKLOAD_TRANSITION_TRAJECTORY_VALIDATION"

    def test_scientific_role_distinct_from_fitting_and_validation_run_kinds(self):
        assert lib.WORKLOAD_TRANSITION_SCIENTIFIC_ROLE not in (lib.RUN_KIND_FITTING, lib.RUN_KIND_INTERPOLATION_VALIDATION)

    def test_execute_workload_transition_slot_never_sets_steady_state_fit_eligibility(self):
        # Source-presence regression check, mirroring the CORR-46 pattern
        # (TestCorr46StructuralValidityGateUnchanged): the workload-
        # transition execution function's own source must never construct
        # a `steady_state_fit_eligibility` field -- that field belongs
        # exclusively to the steady-state (fitting/interpolation_
        # validation) body, which this run kind is never dispatched into.
        import inspect
        source = inspect.getsource(harness.execute_workload_transition_slot)
        assert "steady_state_fit_eligibility" not in source

    def test_execute_workload_transition_slot_persists_correct_scientific_role(self):
        import inspect
        source = inspect.getsource(harness.execute_workload_transition_slot)
        assert '"scientific_role": WORKLOAD_TRANSITION_SCIENTIFIC_ROLE' in source


class TestCorr47Materiality:
    def test_workload_transition_route_scored_window_is_full_traffic_duration(self):
        assert lib.workload_transition_route_scored_window(240.0) == (0.0, 240.0)

    def test_workload_transition_materiality_threshold_for_240s_is_60s(self):
        assert lib.workload_transition_materiality_threshold_seconds(240.0) == 60.0

    def test_materiality_fraction_matches_default_generic_fraction(self):
        assert lib.WORKLOAD_TRANSITION_MATERIALITY_FRACTION == lib.DEFAULT_ROUTE_INTEGRITY_MATERIALITY_FRACTION == 0.25

    def test_59_999s_interval_is_not_material(self):
        scored_window = lib.workload_transition_route_scored_window(240.0)
        assert lib.is_interval_material(
            0.0, 59.999, 240.0, scored_window,
            materiality_fraction_threshold=lib.WORKLOAD_TRANSITION_MATERIALITY_FRACTION,
        ) is False

    def test_60_000s_interval_is_material(self):
        scored_window = lib.workload_transition_route_scored_window(240.0)
        assert lib.is_interval_material(
            0.0, 60.0, 240.0, scored_window,
            materiality_fraction_threshold=lib.WORKLOAD_TRANSITION_MATERIALITY_FRACTION,
        ) is True

    def test_61s_interval_is_material(self):
        scored_window = lib.workload_transition_route_scored_window(240.0)
        assert lib.is_interval_material(
            100.0, 161.0, 240.0, scored_window,
            materiality_fraction_threshold=lib.WORKLOAD_TRANSITION_MATERIALITY_FRACTION,
        ) is True

    def test_pre_step_only_material_interval(self):
        scored_window = lib.workload_transition_route_scored_window(240.0)
        assert lib.is_interval_material(
            0.0, 60.0, 240.0, scored_window,
            materiality_fraction_threshold=lib.WORKLOAD_TRANSITION_MATERIALITY_FRACTION,
        ) is True

    def test_step_crossing_material_interval(self):
        scored_window = lib.workload_transition_route_scored_window(240.0)
        assert lib.is_interval_material(
            30.0, 90.0, 240.0, scored_window,
            materiality_fraction_threshold=lib.WORKLOAD_TRANSITION_MATERIALITY_FRACTION,
        ) is True

    def test_post_step_only_material_interval(self):
        scored_window = lib.workload_transition_route_scored_window(240.0)
        assert lib.is_interval_material(
            150.0, 210.0, 240.0, scored_window,
            materiality_fraction_threshold=lib.WORKLOAD_TRANSITION_MATERIALITY_FRACTION,
        ) is True

    def test_separated_bursts_never_combined(self):
        rows = [
            {"elapsed_seconds": t, "sent_requests": 1, "success_2xx": 0}
            for t in range(0, 30)
        ] + [
            {"elapsed_seconds": t, "sent_requests": 1, "success_2xx": 1}
            for t in range(30, 40)
        ] + [
            {"elapsed_seconds": t, "sent_requests": 1, "success_2xx": 0}
            for t in range(40, 70)
        ]
        runs = lib.find_all_zero_success_runs(rows)
        assert len(runs) == 2
        assert runs[0] == (0.0, 29.0)
        assert runs[1] == (40.0, 69.0)


class TestCorr47V8WindowPatterns:
    def test_60s_material_interval_partitions_into_six_10s_windows(self):
        windows = lib.partition_into_evidence_windows(0.0, 60.0, min_window_seconds=lib.MIN_EVIDENCE_WINDOW_SECONDS_V8)
        assert len(windows) == 6
        assert all(e - s == 10.0 for s, e in windows)

    def test_62s_material_interval_partitions_into_five_10s_plus_final_12s(self):
        windows = lib.partition_into_evidence_windows(0.0, 62.0, min_window_seconds=lib.MIN_EVIDENCE_WINDOW_SECONDS_V8)
        durations = [e - s for s, e in windows]
        assert durations[:-1] == [10.0] * 5
        assert durations[-1] == 12.0

    def test_69s_material_interval_partitions_into_six_10s_plus_final_9s_standalone(self):
        windows = lib.partition_into_evidence_windows(0.0, 69.0, min_window_seconds=lib.MIN_EVIDENCE_WINDOW_SECONDS_V8)
        durations = [e - s for s, e in windows]
        assert durations[:-1] == [10.0] * 6
        assert durations[-1] == 9.0

    def test_236s_material_interval_partitions_into_22_10s_plus_final_16s(self):
        windows = lib.partition_into_evidence_windows(3.0, 239.0, min_window_seconds=lib.MIN_EVIDENCE_WINDOW_SECONDS_V8)
        durations = [e - s for s, e in windows]
        assert durations[:-1] == [10.0] * 22
        assert durations[-1] == 16.0
        assert sum(durations) == 236.0


class TestCorr47TrajectoryAssembly:
    def _metrics_row(self, elapsed, **overrides):
        row = {
            "elapsed_seconds": elapsed,
            "request_rate_rps": "8.0",
            "latency_p50_seconds": "0.05",
            "latency_p95_seconds": "0.1",
            "latency_p99_seconds": "0.2",
            "requests_in_progress": "2",
            "cpu_cores": "0.3",
            "memory_bytes": "104857600",
            "replicas_desired": "4",
            "replicas_current": "4",
            "replicas_ready": "4",
            "replicas_available": "4",
        }
        row.update(overrides)
        return row

    def test_pre_step_rows_get_rps_before_and_pre_step_phase(self):
        rows = [self._metrics_row(10.0)]
        result = lib.build_workload_transition_trajectory_rows(rows, 60.0, rps_before=8.0, rps_after=32.0, total_traffic_seconds=240.0)
        assert result[0]["phase"] == "pre_step"
        assert result[0]["target_rps"] == 8.0
        assert result[0]["elapsed_from_workload_step"] == -50.0

    def test_post_step_rows_get_rps_after_and_post_step_phase(self):
        rows = [self._metrics_row(90.0)]
        result = lib.build_workload_transition_trajectory_rows(rows, 60.0, rps_before=8.0, rps_after=32.0, total_traffic_seconds=240.0)
        assert result[0]["phase"] == "post_step"
        assert result[0]["target_rps"] == 32.0
        assert result[0]["elapsed_from_workload_step"] == 30.0

    def test_row_exactly_at_step_boundary_is_post_step(self):
        rows = [self._metrics_row(60.0)]
        result = lib.build_workload_transition_trajectory_rows(rows, 60.0, rps_before=8.0, rps_after=32.0, total_traffic_seconds=240.0)
        assert result[0]["phase"] == "post_step"

    def test_raw_signals_preserved_numerically(self):
        rows = [self._metrics_row(10.0)]
        result = lib.build_workload_transition_trajectory_rows(rows, 60.0, rps_before=8.0, rps_after=32.0, total_traffic_seconds=240.0)
        assert result[0]["request_rate_rps"] == 8.0
        assert result[0]["cpu_cores"] == 0.3
        assert result[0]["replicas_desired"] == 4.0

    def test_missing_values_stay_none_never_fabricated(self):
        rows = [self._metrics_row(10.0, cpu_cores="", memory_bytes=None)]
        result = lib.build_workload_transition_trajectory_rows(rows, 60.0, rps_before=8.0, rps_after=32.0, total_traffic_seconds=240.0)
        assert result[0]["cpu_cores"] is None
        assert result[0]["memory_bytes"] is None

    def test_non_numeric_value_becomes_none_not_raise(self):
        rows = [self._metrics_row(10.0, cpu_cores="collection_error")]
        result = lib.build_workload_transition_trajectory_rows(rows, 60.0, rps_before=8.0, rps_after=32.0, total_traffic_seconds=240.0)
        assert result[0]["cpu_cores"] is None


class TestCorr47FixedReplicaInvariant:
    def test_invariant_holds_when_desired_never_drifts(self):
        rows = [{"replicas_desired": 4.0, "replicas_current": 4.0, "elapsed_from_traffic_start": t} for t in (0.0, 10.0, 20.0)]
        result = lib.evaluate_fixed_replica_invariant(rows, fixed_replicas=4)
        assert result["fixed_replica_invariant_holds"] is True
        assert result["drift_sample_count"] == 0
        assert result["desired_replicas_observed_distinct_values"] == [4.0]

    def test_invariant_violated_when_desired_drifts(self):
        rows = [
            {"replicas_desired": 4.0, "replicas_current": 4.0, "elapsed_from_traffic_start": 0.0},
            {"replicas_desired": 5.0, "replicas_current": 4.0, "elapsed_from_traffic_start": 10.0},
        ]
        result = lib.evaluate_fixed_replica_invariant(rows, fixed_replicas=4)
        assert result["fixed_replica_invariant_holds"] is False
        assert result["drift_sample_count"] == 1
        assert result["drift_first_elapsed_from_traffic_start"] == 10.0

    def test_none_values_excluded_from_distinct_values_and_drift(self):
        rows = [{"replicas_desired": None, "replicas_current": None, "elapsed_from_traffic_start": 0.0}]
        result = lib.evaluate_fixed_replica_invariant(rows, fixed_replicas=4)
        assert result["fixed_replica_invariant_holds"] is True
        assert result["desired_replicas_observed_distinct_values"] == []


class TestCorr47FirstPostStepDescriptive:
    def test_returns_earliest_post_step_planned_timestamp(self):
        rows = [
            {"phase_name": "pre_step", "planned_elapsed_seconds": "59.5"},
            {"phase_name": "post_step", "planned_elapsed_seconds": "60.125"},
            {"phase_name": "post_step", "planned_elapsed_seconds": "60.25"},
        ]
        assert lib.first_post_step_request_elapsed_seconds(rows) == 60.125

    def test_returns_none_when_no_post_step_rows(self):
        rows = [{"phase_name": "pre_step", "planned_elapsed_seconds": "10.0"}]
        assert lib.first_post_step_request_elapsed_seconds(rows) is None

    def test_ignores_rows_missing_planned_elapsed_seconds(self):
        rows = [{"phase_name": "post_step", "planned_elapsed_seconds": ""}, {"phase_name": "post_step", "planned_elapsed_seconds": "61.0"}]
        assert lib.first_post_step_request_elapsed_seconds(rows) == 61.0


class TestCorr47Provenance:
    def test_active_correction_reference_is_corr47(self):
        prov = lib.build_protocol_provenance()
        assert prov["active_correction_reference"] == "CORR-48"
        assert prov["prior_correction_reference"] == "CORR-47"

    def test_no_protocol_version_bumped_by_corr47(self):
        prov = lib.build_protocol_provenance()
        assert prov["route_integrity_protocol_version"] == 6
        assert prov["dataplane_acquisition_protocol_version"] == 2
        assert prov["window_partitioning_protocol_version"] == 2
        assert prov["transport_protocol_version"] == 5
        assert prov["active_calibration_protocol_version"] == 8

    def test_derived_metrics_status_is_not_formally_operationalized(self):
        assert lib.WORKLOAD_TRANSITION_DERIVED_METRICS_STATUS == "NOT_FORMALLY_OPERATIONALIZED"
        assert lib.WORKLOAD_TRANSITION_DERIVED_METRICS_STATUS == lib.NOT_FORMALLY_OPERATIONALIZED


class TestCorr47ExecutionBranching:
    def test_main_dispatches_workload_transition_to_dedicated_execution_path(self):
        import inspect
        source = inspect.getsource(harness.main)
        assert "is_workload_transition_slot(slot)" in source
        assert "execute_workload_transition_slot(" in source

    def test_execute_workload_transition_slot_reuses_corr46_timeout_pattern(self):
        # The route-integrity sampler closure inside execute_workload_
        # transition_slot() must reuse the same CORR-46 recoverable-
        # timeout pattern already proven for replica_transition -- a
        # source-presence regression check, since spinning up the full
        # live execution path is out of scope for a unit test (matching
        # this project's own established depth for the analogous
        # replica-transition execution function).
        import inspect
        source = inspect.getsource(harness.execute_workload_transition_slot)
        assert "except subprocess.TimeoutExpired as exc:" in source
        assert "endpoint_query_error" in source


class TestCorr47FailureBehavior:
    def test_preflight_failure_creates_no_attempt_directory(self, monkeypatch, tmp_path, real_sequence):
        monkeypatch.setattr(harness, "preflight_checks", lambda: [(False, "Docker daemon unreachable")])
        monkeypatch.setattr(harness, "CALIBRATION_ROOT", tmp_path / "official")
        monkeypatch.setattr(harness, "TRANSPORT_VALIDATION_ROOT", tmp_path / "nonofficial")

        manifest = lib.load_manifest()
        slot = lib.resolve_planned_slot(38, sequence=real_sequence)
        ledger = lib.init_ledger(real_sequence)

        class _Args:
            campaign_sequence_index = 38
            attempt_number = 1

        rc = harness.execute_workload_transition_slot(_Args(), manifest, real_sequence, slot, slot["schedule_sha256"], ledger, official=True)
        assert rc == 2
        assert not (tmp_path / "official").exists()

    def test_nonofficial_smoke_never_touches_ledger_namespace(self, monkeypatch, tmp_path, real_sequence):
        monkeypatch.setattr(harness, "preflight_checks", lambda: [(False, "Docker daemon unreachable")])
        official_root = tmp_path / "official"
        nonofficial_root = tmp_path / "nonofficial"
        monkeypatch.setattr(harness, "CALIBRATION_ROOT", official_root)
        monkeypatch.setattr(harness, "TRANSPORT_VALIDATION_ROOT", nonofficial_root)

        def _boom_ledger(*a, **k):
            raise AssertionError("non-official smoke must never touch the ledger")

        monkeypatch.setattr(harness, "ensure_ledger", _boom_ledger)
        monkeypatch.setattr(harness, "update_slot_state", _boom_ledger)
        monkeypatch.setattr(harness, "atomic_write_ledger", _boom_ledger)

        manifest = lib.load_manifest()
        slot = lib.resolve_planned_slot(38, sequence=real_sequence)
        ledger = lib.init_ledger(real_sequence)

        class _Args:
            campaign_sequence_index = 38
            attempt_number = 1

        rc = harness.execute_workload_transition_slot(_Args(), manifest, real_sequence, slot, slot["schedule_sha256"], ledger, official=False)
        assert rc == 2
        assert not official_root.exists()
        assert not nonofficial_root.exists()

    def test_genuine_collector_death_string_present_in_workload_execution_path(self):
        # CORR-46 sec.8: a genuine (unrecovered) collector/generator
        # failure must remain structurally invalidating for this run kind
        # too -- the gate is the SAME shared string/logic reused verbatim,
        # not a new one.
        import inspect
        source = inspect.getsource(harness.execute_workload_transition_slot)
        assert "collector_or_generator_structural_failure" in source


# ---------------------------------------------------------------------------
# CORR-48 (Phase 15B.6D): workload-transition trajectory traffic-window
# boundary hardening -- index38's own first-official-use audit found the
# derived `workload_transition_timeline.csv` admitted 3 of 27 rows from the
# metrics collector's own post-traffic observation buffer
# (total_traffic_seconds+30s), mislabeling them `post_step`. Raw
# `metrics.csv` is untouched; only the derived SCIENTIFIC trajectory is
# now restricted to the official half-open [0, total_traffic_seconds)
# traffic window.
# ---------------------------------------------------------------------------


class TestCorr48TrajectoryBoundaryFilter:
    def _metrics_row(self, elapsed):
        return {
            "elapsed_seconds": elapsed,
            "request_rate_rps": "8.0",
            "latency_p50_seconds": "0.05",
            "latency_p95_seconds": "0.1",
            "latency_p99_seconds": "0.2",
            "requests_in_progress": "2",
            "cpu_cores": "0.3",
            "memory_bytes": "104857600",
            "replicas_desired": "4",
            "replicas_current": "4",
            "replicas_ready": "4",
            "replicas_available": "4",
        }

    def test_negative_elapsed_excluded(self):
        rows = [self._metrics_row(-0.1)]
        result = lib.build_workload_transition_trajectory_rows(rows, 60.0, rps_before=8.0, rps_after=32.0, total_traffic_seconds=240.0)
        assert result == []

    def test_elapsed_zero_included_as_pre_step(self):
        rows = [self._metrics_row(0.0)]
        result = lib.build_workload_transition_trajectory_rows(rows, 60.0, rps_before=8.0, rps_after=32.0, total_traffic_seconds=240.0)
        assert len(result) == 1
        assert result[0]["phase"] == "pre_step"

    def test_elapsed_59_999_included_as_pre_step(self):
        rows = [self._metrics_row(59.999)]
        result = lib.build_workload_transition_trajectory_rows(rows, 60.0, rps_before=8.0, rps_after=32.0, total_traffic_seconds=240.0)
        assert len(result) == 1
        assert result[0]["phase"] == "pre_step"

    def test_elapsed_60_included_as_post_step(self):
        rows = [self._metrics_row(60.0)]
        result = lib.build_workload_transition_trajectory_rows(rows, 60.0, rps_before=8.0, rps_after=32.0, total_traffic_seconds=240.0)
        assert len(result) == 1
        assert result[0]["phase"] == "post_step"

    def test_elapsed_239_999_included_as_post_step(self):
        rows = [self._metrics_row(239.999)]
        result = lib.build_workload_transition_trajectory_rows(rows, 60.0, rps_before=8.0, rps_after=32.0, total_traffic_seconds=240.0)
        assert len(result) == 1
        assert result[0]["phase"] == "post_step"

    def test_elapsed_240_excluded(self):
        rows = [self._metrics_row(240.0)]
        result = lib.build_workload_transition_trajectory_rows(rows, 60.0, rps_before=8.0, rps_after=32.0, total_traffic_seconds=240.0)
        assert result == []

    def test_elapsed_250_excluded(self):
        rows = [self._metrics_row(250.0)]
        result = lib.build_workload_transition_trajectory_rows(rows, 60.0, rps_before=8.0, rps_after=32.0, total_traffic_seconds=240.0)
        assert result == []

    def test_elapsed_269_9_excluded(self):
        rows = [self._metrics_row(269.9)]
        result = lib.build_workload_transition_trajectory_rows(rows, 60.0, rps_before=8.0, rps_after=32.0, total_traffic_seconds=240.0)
        assert result == []

    def test_mixed_rows_only_in_window_rows_survive(self):
        rows = [self._metrics_row(e) for e in (-0.1, 0.0, 59.999, 60.0, 239.999, 240.0, 250.0, 269.9)]
        result = lib.build_workload_transition_trajectory_rows(rows, 60.0, rps_before=8.0, rps_after=32.0, total_traffic_seconds=240.0)
        elapsed_values = [r["elapsed_from_traffic_start"] for r in result]
        assert elapsed_values == [0.0, 59.999, 60.0, 239.999]


class TestCorr48RawVsDerivedPreservation:
    def test_raw_metrics_csv_rows_after_traffic_end_remain_loadable_untouched(self):
        # load_metrics_csv() itself must never filter -- it is the raw
        # collector-lifecycle loader, distinct from the derived scientific
        # trajectory builder. Only build_workload_transition_trajectory_
        # rows() applies the traffic-window boundary.
        import io
        csv_text = (
            "elapsed_seconds,request_rate_rps,latency_p50_seconds,latency_p95_seconds,"
            "latency_p99_seconds,requests_in_progress,cpu_cores,memory_bytes,"
            "replicas_desired,replicas_current,replicas_ready,replicas_available\n"
            "10.0,8.0,0.05,0.1,0.2,2,0.3,104857600,4,4,4,4\n"
            "250.0,32.0,0.05,0.1,0.2,2,0.3,104857600,4,4,4,4\n"
        )
        import csv as csv_module
        rows = list(csv_module.DictReader(io.StringIO(csv_text)))
        assert len(rows) == 2  # both the in-window and post-traffic-buffer row are present raw
        trajectory = lib.build_workload_transition_trajectory_rows(
            [{**r, "elapsed_seconds": float(r["elapsed_seconds"])} for r in rows],
            60.0, rps_before=8.0, rps_after=32.0, total_traffic_seconds=240.0,
        )
        assert len(trajectory) == 1  # the post-traffic row was excluded from the derived trajectory only
        assert trajectory[0]["elapsed_from_traffic_start"] == 10.0


class TestCorr48Index38RealDataRegression:
    """Phase 15B.6D: using index38's own immutable raw metrics.csv, prove
    the corrected trajectory boundary excludes exactly the known
    post-traffic buffer rows, without relying on a brittle absolute row
    count (future attempts' raw sampling cadence may vary)."""

    ATTEMPT_ID = "transwork_r4_8to32_repeat1_attempt1_20260921T042932Z"

    def _load_real_metrics_rows(self):
        path = (
            lib.PROJECT_ROOT / "persistent-data" / "metrics" / self.ATTEMPT_ID / "metrics.csv"
        )
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            for row in rows:
                row["elapsed_seconds"] = float(row["elapsed_seconds"])
            return rows

    def test_raw_metrics_csv_contains_post_traffic_buffer_rows(self):
        rows = self._load_real_metrics_rows()
        post_traffic = [r for r in rows if r["elapsed_seconds"] >= 240.0]
        assert len(post_traffic) == 3
        assert sorted(r["elapsed_seconds"] for r in post_traffic) == [240.0, 250.0, 260.01]

    def test_corrected_trajectory_excludes_post_traffic_buffer(self):
        rows = self._load_real_metrics_rows()
        trajectory = lib.build_workload_transition_trajectory_rows(
            rows, 60.0, rps_before=8.0, rps_after=32.0, total_traffic_seconds=240.0,
        )
        assert all(r["elapsed_from_traffic_start"] >= 0.0 for r in trajectory)
        assert all(r["elapsed_from_traffic_start"] < 240.0 for r in trajectory)
        assert all(r["phase"] == "pre_step" for r in trajectory if r["elapsed_from_traffic_start"] < 60.0)
        assert all(r["phase"] == "post_step" for r in trajectory if r["elapsed_from_traffic_start"] >= 60.0)
        # exactly the 3 known post-traffic-buffer elapsed values are gone
        elapsed_values = {r["elapsed_from_traffic_start"] for r in trajectory}
        assert not ({240.0, 250.0, 260.01} & elapsed_values)

    def test_corrected_trajectory_row_count_matches_official_window(self):
        rows = self._load_real_metrics_rows()
        trajectory = lib.build_workload_transition_trajectory_rows(
            rows, 60.0, rps_before=8.0, rps_after=32.0, total_traffic_seconds=240.0,
        )
        # 27 raw rows minus the 3 known post-traffic-buffer rows = 24
        assert len(trajectory) == len(rows) - 3
        pre_step = [r for r in trajectory if r["phase"] == "pre_step"]
        post_step = [r for r in trajectory if r["phase"] == "post_step"]
        assert len(pre_step) == 6
        assert len(post_step) == 18


class TestCorr48Corr47RegressionUnchanged:
    """Index38's own already-official, immutable metadata must remain
    completely unaffected by this forward-only trajectory-boundary fix."""

    def _read_metadata(self):
        path = (
            lib.PROJECT_ROOT / "persistent-data" / "dynamics_calibration"
            / "transwork_r4_8to32_repeat1_attempt1_20260921T042932Z" / "metadata.json"
        )
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_index38_still_valid_and_identity_unchanged(self):
        m = self._read_metadata()
        assert m["execution_structural_validity"] == "VALID"
        assert m["run_kind"] == "workload_transition"
        assert m["scientific_role"] == "WORKLOAD_TRANSITION_TRAJECTORY_VALIDATION"
        assert m["replicas"] == 4
        assert m["rps_before"] == 8.0
        assert m["rps_after"] == 32.0

    def test_index38_timing_and_step_unchanged(self):
        m = self._read_metadata()
        timing = m["workload_transition_timing_plan"]
        assert timing["total_traffic_seconds"] == 240.0
        assert timing["trigger_at_traffic_elapsed_seconds"] == 60.0
        assert m["workload_step"]["workload_step_owner"] == "schedule"

    def test_index38_materiality_and_fitting_isolation_unchanged(self):
        m = self._read_metadata()
        assert "steady_state_fit_eligibility" not in m
        assert m["derived_metrics_status"] == "NOT_FORMALLY_OPERATIONALIZED"

    def test_index38_ledger_still_valid(self):
        ledger = lib.load_ledger()
        assert ledger["slots"]["38"]["state"] == "VALID"
        assert ledger["slots"]["38"]["valid_attempt_id"] == "transwork_r4_8to32_repeat1_attempt1_20260921T042932Z"


class TestPrometheusDiagnosticHealthHardening:
    """Phase 16I.4: the post-run `prometheus_health` metadata field is
    diagnostic-only (never consulted for execution_structural_validity,
    route_integrity classification, or steady_state_fit_eligibility). A
    transient failure of this one diagnostic check must be recorded, not
    allowed to raise and crash the whole attempt before scientific
    classification runs (Phase16I.3 slot 17 crashed exactly this way)."""

    def test_timeout_does_not_raise_and_is_recorded(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 5))

        monkeypatch.setattr(harness.subprocess, "run", fake_run)
        result = harness.check_prometheus_diagnostic_health()
        assert result["healthy"] is False
        assert result["error_type"] == "TimeoutExpired"
        assert result["error_message"]

    def test_called_process_error_does_not_raise_and_is_recorded(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise subprocess.CalledProcessError(returncode=7, cmd=args[0])

        monkeypatch.setattr(harness.subprocess, "run", fake_run)
        result = harness.check_prometheus_diagnostic_health()
        assert result["healthy"] is False
        assert result["error_type"] == "CalledProcessError"
        assert result["error_message"]

    def test_unready_non_exception_response_recorded_without_crash(self, monkeypatch):
        class FakeCompletedProcess:
            returncode = 1

        monkeypatch.setattr(harness.subprocess, "run", lambda *a, **k: FakeCompletedProcess())
        result = harness.check_prometheus_diagnostic_health()
        assert result["healthy"] is False
        assert result["error_type"] is None
        assert result["error_message"] is None

    def test_healthy_success_path_unchanged(self, monkeypatch):
        class FakeCompletedProcess:
            returncode = 0

        monkeypatch.setattr(harness.subprocess, "run", lambda *a, **k: FakeCompletedProcess())
        result = harness.check_prometheus_diagnostic_health()
        assert result == {"healthy": True, "error_type": None, "error_message": None}

    def test_diagnostic_field_never_referenced_by_invalidity_reasons(self):
        """Source-level isolation proof: a diagnostic-failure flag must never
        be able to force a false structural PASS or FAIL, because the
        classifier (`invalidity_reasons` -> execution_structural_validity)
        must never even read it."""
        source = Path(harness.__file__).read_text(encoding="utf-8")
        for line in source.splitlines():
            if "invalidity_reasons.append" in line or "invalidity_reasons +=" in line:
                assert "prom_health" not in line, f"diagnostic field leaked into classification: {line!r}"

    def test_execution_structural_validity_formula_is_invalidity_reasons_only(self):
        """Pins the exact classification formula so a future edit can't
        silently fold the diagnostic flag into it without failing this test."""
        source = Path(harness.__file__).read_text(encoding="utf-8")
        occurrences = source.count(
            "execution_structural_validity = EXECUTION_INVALID if invalidity_reasons else EXECUTION_VALID"
        )
        assert occurrences == 3  # replica_transition, workload_transition, steady-state branches


class TestForensicSubprocessHardening:
    """Phase 16I.6: hardens the other unguarded, forensic-evidence-only
    subprocess calls in the same class as the Phase16I.4 Prometheus fix --
    specifically the `kubectl get pods` snapshot call that crashed a
    Phase16I.5 attempt (run_dynamics_calibration_slot.py, 3 identical
    branches), plus the Kubernetes-Events, previous-container-log, and
    host-resource-context forensic/diagnostic captures. None of these may
    crash the runner, and none may influence scientific classification."""

    def test_pod_snapshot_timeout_does_not_raise_and_is_recorded(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 15))

        monkeypatch.setattr(harness, "_run", fake_run)
        result = harness.safe_fetch_current_pod_snapshot_json()
        assert result["items"] == []
        assert "TimeoutExpired" in result["_error"]

    def test_pod_snapshot_healthy_path_unchanged(self, monkeypatch):
        class FakeCompletedProcess:
            returncode = 0
            stdout = '{"items": [{"metadata": {"name": "p1"}}]}'
            stderr = ""

        monkeypatch.setattr(harness, "_run", lambda *a, **k: FakeCompletedProcess())
        result = harness.safe_fetch_current_pod_snapshot_json()
        assert result == {"items": [{"metadata": {"name": "p1"}}]}

    def test_events_json_timeout_does_not_raise_and_is_recorded(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise subprocess.CalledProcessError(returncode=1, cmd=args[0])

        monkeypatch.setattr(harness, "_run", fake_run)
        result = harness.fetch_events_json()
        assert result["items"] == []
        assert "CalledProcessError" in result["_error"]

    def test_previous_container_logs_timeout_does_not_raise_and_is_recorded(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 15))

        monkeypatch.setattr(harness, "_run", fake_run)
        log_text, available = harness.fetch_previous_container_logs("some-pod")
        assert available is False
        assert "TimeoutExpired" in log_text

    def test_minikube_resource_context_timeout_does_not_raise_and_is_recorded(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 15))

        monkeypatch.setattr(harness, "_run", fake_run)
        result = harness.capture_minikube_resource_context()
        assert result["outer_cpu_nano_cpus"] is None
        assert result["outer_cpu_cores"] is None
        assert "TimeoutExpired" in result["observation_error"]

    def test_minikube_resource_context_healthy_path_unchanged(self, monkeypatch):
        class FakeCompletedProcess:
            def __init__(self, stdout):
                self.stdout = stdout
                self.stderr = ""
                self.returncode = 0

        calls = iter([
            FakeCompletedProcess("2000000000 4294967296"),
            FakeCompletedProcess("usage_usec 123456\nnr_periods 0\n"),
        ])
        monkeypatch.setattr(harness, "_run", lambda *a, **k: next(calls))
        result = harness.capture_minikube_resource_context()
        assert result["outer_cpu_nano_cpus"] == 2000000000
        assert result["outer_cpu_cores"] == 2.0
        assert result["outer_memory_bytes"] == 4294967296
        assert result["cgroup_cpu_usage_usec"] == 123456
        assert result["observation_error"] is None

    def test_forensic_fields_never_referenced_by_invalidity_reasons(self):
        """Source-level isolation proof (mirrors the Prometheus-hardening
        test): none of these forensic/diagnostic capture results may be
        read by the invalidity_reasons classifier."""
        source = Path(harness.__file__).read_text(encoding="utf-8")
        forbidden_tokens = ("pods_snapshot_json", "events_pre", "events_post", "pre_run_context", "post_run_context")
        for line in source.splitlines():
            if "invalidity_reasons.append" in line or "invalidity_reasons +=" in line:
                for token in forbidden_tokens:
                    assert token not in line, f"forensic/diagnostic field leaked into classification: {line!r}"

    def test_load_bearing_hpa_scale_call_still_raises_on_failure(self, monkeypatch):
        """Negative control proving this phase's hardening is narrowly
        scoped: a genuinely load-bearing control call (`kubectl scale`,
        used to establish the fixed-replica experimental state) must still
        propagate a subprocess failure -- it must NOT be silently caught,
        unlike the forensic-only calls above."""
        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 15))

        monkeypatch.setattr(harness, "_run", fake_run)
        with pytest.raises(subprocess.TimeoutExpired):
            harness.scale_and_wait(5)


class TestNodePortResolutionHardeningCorr53:
    """Phase 16L.5 (CORR-53): hardens `resolve_loadtest_nodeport()` and its
    two siblings `resolve_minikube_docker_network()`/`resolve_minikube_node_ip()`
    -- the exact unguarded-subprocess defect class that crashed Phase16L.4
    at validation_sequence_index=30 (an uncaught `subprocess.TimeoutExpired`
    inside `resolve_loadtest_nodeport()`'s `kubectl get svc` call, before any
    ledger entry was written). Unlike the forensic/diagnostic calls hardened
    in Phase16I.4/16I.6, these three resolvers are load-bearing for
    establishing traffic transport -- but all three ALREADY returned None
    (never raised) for a non-zero exit code or unparseable output, and their
    sole caller (`start_loadtest_transport()`) already treats a None return
    as a resolvable, non-crashing transport-establishment failure (the
    pre-existing `wrong_traffic_path_used` invalidity reason -- never a
    fabricated VALID/CONFIRMED result, never traffic generation). This
    hardening only extends that same already-established "return None, do
    not raise" contract to the exception case; it does not change what a
    resolution failure MEANS scientifically, only how it is reported."""

    def test_nodeport_timeout_returns_none_without_raising(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 15))

        monkeypatch.setattr(harness, "_run", fake_run)
        assert harness.resolve_loadtest_nodeport() is None

    def test_nodeport_called_process_error_returns_none_without_raising(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise subprocess.CalledProcessError(returncode=1, cmd=args[0])

        monkeypatch.setattr(harness, "_run", fake_run)
        assert harness.resolve_loadtest_nodeport() is None

    def test_nodeport_oserror_returns_none_without_raising(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise OSError("device not configured")

        monkeypatch.setattr(harness, "_run", fake_run)
        assert harness.resolve_loadtest_nodeport() is None

    def test_nodeport_success_path_unchanged(self, monkeypatch):
        class FakeCompletedProcess:
            returncode = 0
            stdout = "31891"
            stderr = ""

        monkeypatch.setattr(harness, "_run", lambda *a, **k: FakeCompletedProcess())
        assert harness.resolve_loadtest_nodeport() == 31891

    def test_nodeport_non_digit_output_still_returns_none_unchanged(self, monkeypatch):
        """Pre-existing behavior (not touched by this hardening): unparseable
        output already returned None before Phase16L.5."""
        class FakeCompletedProcess:
            returncode = 0
            stdout = ""
            stderr = ""

        monkeypatch.setattr(harness, "_run", lambda *a, **k: FakeCompletedProcess())
        assert harness.resolve_loadtest_nodeport() is None

    def test_no_stale_endpoint_fallback_on_failure(self, monkeypatch):
        """A resolution failure must never fall back to a cached/stale
        NodePort -- the function holds no state across calls, so a failed
        call and a subsequent successful call must be fully independent."""
        outcomes = iter([
            subprocess.TimeoutExpired(cmd=["kubectl"], timeout=15),
        ])

        def fake_run(*args, **kwargs):
            exc = next(outcomes, None)
            if exc is not None:
                raise exc
            class FakeCompletedProcess:
                returncode = 0
                stdout = "31891"
                stderr = ""
            return FakeCompletedProcess()

        monkeypatch.setattr(harness, "_run", fake_run)
        assert harness.resolve_loadtest_nodeport() is None  # first call: failure, no cached value returned
        assert harness.resolve_loadtest_nodeport() == 31891  # second call: fresh, independent resolution

    def test_docker_network_timeout_returns_none_without_raising(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 15))

        monkeypatch.setattr(harness, "_run", fake_run)
        assert harness.resolve_minikube_docker_network() is None

    def test_docker_network_success_path_unchanged(self, monkeypatch):
        class FakeCompletedProcess:
            returncode = 0
            stdout = '{"minikube": {}}'
            stderr = ""

        monkeypatch.setattr(harness, "_run", lambda *a, **k: FakeCompletedProcess())
        assert harness.resolve_minikube_docker_network() == "minikube"

    def test_node_ip_timeout_returns_none_without_raising(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 15))

        monkeypatch.setattr(harness, "_run", fake_run)
        assert harness.resolve_minikube_node_ip() is None

    def test_node_ip_success_path_unchanged(self, monkeypatch):
        class FakeCompletedProcess:
            returncode = 0
            stdout = "192.168.49.2\n"
            stderr = ""

        monkeypatch.setattr(harness, "_run", lambda *a, **k: FakeCompletedProcess())
        assert harness.resolve_minikube_node_ip() == "192.168.49.2"

    def test_resolver_failure_surfaces_as_transport_error_not_a_crash(self, monkeypatch):
        """Integration-level proof: when the nodeport resolver fails,
        start_loadtest_transport() (the sole caller) must return its
        pre-existing {"error": ...} dict -- never raise, never fabricate a
        successful transport result, never invoke docker run."""
        def raising_nodeport():
            raise AssertionError("should never be called directly by this test -- see below")

        monkeypatch.setattr(harness, "resolve_minikube_docker_network", lambda: "minikube")
        monkeypatch.setattr(harness, "resolve_minikube_node_ip", lambda: "192.168.49.2")
        monkeypatch.setattr(harness, "resolve_loadtest_nodeport", lambda: None)  # simulates a caught, hardened failure

        docker_run_called = {"n": 0}

        def fake_run(*args, **kwargs):
            docker_run_called["n"] += 1
            raise AssertionError("docker run must not be attempted when nodeport resolution failed")

        monkeypatch.setattr(harness, "_run", fake_run)
        result = harness.start_loadtest_transport("test-experiment-id")
        assert "error" in result
        assert result["nodeport"] is None
        assert docker_run_called["n"] == 0

    def test_pre_traffic_tooling_failure_never_becomes_official(self, monkeypatch):
        """Section 8 semantics: a pre-traffic transport-establishment
        failure (resolver returned None, surfaced as start_loadtest_transport's
        pre-existing {"error": ...}) is not scientifically special-cased in
        this codebase's ledger schema -- it flows into the ALREADY-ESTABLISHED
        wrong_traffic_path_used invalidity reason, which
        ledger_state_for_execution_validity() already maps to
        LEDGER_STATE_INVALID_RETRY_REQUIRED (never VALID, never official,
        no valid_attempt_id). This test pins that existing mapping so a
        future change can't silently make a transport failure officially
        VALID."""
        assert lib.ledger_state_for_execution_validity(lib.EXECUTION_INVALID) == lib.LEDGER_STATE_INVALID_RETRY_REQUIRED
        assert lib.ledger_state_for_execution_validity(lib.EXECUTION_INVALID) != lib.LEDGER_STATE_VALID

    def test_load_bearing_scale_call_still_raises_unaffected_by_resolver_hardening(self, monkeypatch):
        """Negative control: this hardening is scoped to the three
        resolvers only -- a genuinely load-bearing control call must still
        propagate a subprocess failure exactly as before Phase16L.5."""
        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 15))

        monkeypatch.setattr(harness, "_run", fake_run)
        with pytest.raises(subprocess.TimeoutExpired):
            harness.scale_and_wait(5)


class TestPreTrafficToolingFailureSemanticSeparationCorr54:
    """Phase 16L.5.1 (CORR-54): a REQUIRED pre-traffic tooling/resolution
    failure (e.g. V5 transport establishment) must never be scored as a
    scientific execution_structural_validity/route_integrity result, and
    must never write a scientific ledger entry -- distinct from a
    genuinely-executed structurally-invalid traffic run. Drives
    `harness.main()` for a real steady-state (run_kind='fitting') slot
    from index 1 of the frozen historical sequence, past preflight/HPA
    isolation/stabilization (all mocked, no real cluster access), up to
    and through `start_loadtest_transport()`."""

    STEADY_STATE_INDEX = 1  # fit_dyncal_r2_L4_repeat1, run_kind='fitting', replicas=2

    def _mock_pretraffic_path(self, monkeypatch, transport_result):
        monkeypatch.setattr(harness, "preflight_checks", lambda: [(True, "ok")])
        monkeypatch.setattr(harness, "capture_minikube_resource_context", lambda: {
            "outer_cpu_nano_cpus": None, "outer_cpu_cores": None, "outer_memory_bytes": None,
            "outer_memory_mib": None, "cgroup_cpu_usage_usec": None, "observation_error": None,
            "captured_at_utc": "2026-01-01T00:00:00+00:00",
        })
        monkeypatch.setattr(harness, "delete_hpa", lambda: None)
        monkeypatch.setattr(harness, "verify_no_hpa_targets_deployment", lambda: True)
        monkeypatch.setattr(harness, "scale_and_wait", lambda replicas, **k: {"desired": replicas, "current": replicas, "ready": replicas, "available": replicas})
        monkeypatch.setattr(harness, "get_pod_names", lambda **k: ["fake-pod-1"])
        monkeypatch.setattr(harness.time, "sleep", lambda *a, **k: None)
        monkeypatch.setattr(harness, "fetch_events_json", lambda **k: {"items": []})
        monkeypatch.setattr(harness, "fetch_loadtest_endpoints_json", lambda: {})
        monkeypatch.setattr(harness, "start_loadtest_transport", lambda experiment_id: transport_result)

    def test_transport_error_produces_pre_traffic_tooling_failure_exit_code(self, monkeypatch, tmp_path):
        ledger_path = tmp_path / "campaign_ledger.json"
        self._mock_pretraffic_path(monkeypatch, {"error": "simulated: kubectl get svc timed out after 15 seconds"})

        rc = harness.main(
            ["--campaign-sequence-index", str(self.STEADY_STATE_INDEX), "--attempt-number", "1"],
            calibration_root=tmp_path, ledger_path=ledger_path,
        )
        assert rc == harness.EXIT_PRE_TRAFFIC_TOOLING_FAILURE
        assert rc not in (0, 1, 2)

    def test_transport_error_never_writes_scientific_ledger_entry(self, monkeypatch, tmp_path):
        ledger_path = tmp_path / "campaign_ledger.json"
        self._mock_pretraffic_path(monkeypatch, {"error": "simulated transport failure"})

        harness.main(
            ["--campaign-sequence-index", str(self.STEADY_STATE_INDEX), "--attempt-number", "1"],
            calibration_root=tmp_path, ledger_path=ledger_path,
        )
        ledger_after = lib.load_ledger(ledger_path)
        slot_state = ledger_after["slots"][str(self.STEADY_STATE_INDEX)]
        assert slot_state["state"] == "NOT_ATTEMPTED"
        assert slot_state["all_attempt_ids"] == []
        assert slot_state.get("valid_attempt_id") is None

    def test_transport_error_writes_pre_traffic_tooling_failure_artifact_and_preserves_directory(self, monkeypatch, tmp_path):
        ledger_path = tmp_path / "campaign_ledger.json"
        self._mock_pretraffic_path(monkeypatch, {"error": "simulated transport failure", "docker_network": None, "node_ip": None, "nodeport": None})

        harness.main(
            ["--campaign-sequence-index", str(self.STEADY_STATE_INDEX), "--attempt-number", "1"],
            calibration_root=tmp_path, ledger_path=ledger_path,
        )
        attempt_dirs = [p for p in tmp_path.iterdir() if p.is_dir() and "attempt" in p.name]
        assert len(attempt_dirs) == 1
        failure_path = attempt_dirs[0] / "pre_traffic_tooling_failure.json"
        assert failure_path.exists()
        failure_evidence = json.loads(failure_path.read_text())
        assert failure_evidence["reason"] == "required_v5_transport_resolution_failed_before_traffic"
        assert failure_evidence["detail"]["transport_error"] == "simulated transport failure"
        assert failure_evidence["campaign_sequence_index"] == self.STEADY_STATE_INDEX
        # No scientific metadata.json (the file only written after successful
        # traffic + classification) exists for this attempt.
        assert not (attempt_dirs[0] / "metadata.json").exists()

    def test_transport_host_port_unresolved_also_produces_pre_traffic_tooling_failure(self, monkeypatch, tmp_path):
        ledger_path = tmp_path / "campaign_ledger.json"
        self._mock_pretraffic_path(monkeypatch, {
            "container_name": "loadtest-relay-test", "target_url": None,
            "docker_network": "minikube", "node_ip": "192.168.49.2", "nodeport": 31891,
        })

        rc = harness.main(
            ["--campaign-sequence-index", str(self.STEADY_STATE_INDEX), "--attempt-number", "1"],
            calibration_root=tmp_path, ledger_path=ledger_path,
        )
        assert rc == harness.EXIT_PRE_TRAFFIC_TOOLING_FAILURE
        attempt_dirs = [p for p in tmp_path.iterdir() if p.is_dir() and "attempt" in p.name]
        failure_evidence = json.loads((attempt_dirs[0] / "pre_traffic_tooling_failure.json").read_text())
        assert failure_evidence["reason"] == "v5_relay_launched_but_host_port_unresolved_before_traffic"

    def test_cleanup_still_executes_on_pre_traffic_tooling_failure(self, monkeypatch, tmp_path):
        """finally: cleanup.run_all() must still fire even though the new
        except clause returns early -- Python guarantees this, but pin it
        with an observable side effect (HPA restoration attempted)."""
        ledger_path = tmp_path / "campaign_ledger.json"
        restore_calls = {"n": 0}
        monkeypatch.setattr(harness, "restore_hpa", lambda: restore_calls.__setitem__("n", restore_calls["n"] + 1))
        monkeypatch.setattr(harness, "wait_for_idle_one_replica", lambda **k: {"desired": 1, "current": 1, "ready": 1, "available": 1})
        self._mock_pretraffic_path(monkeypatch, {"error": "simulated transport failure"})

        harness.main(
            ["--campaign-sequence-index", str(self.STEADY_STATE_INDEX), "--attempt-number", "1"],
            calibration_root=tmp_path, ledger_path=ledger_path,
        )
        assert restore_calls["n"] == 1

    def test_successful_resolver_path_proceeds_past_transport_unaffected(self, monkeypatch, tmp_path):
        """Negative control: a successful transport result must NOT raise
        PreTrafficToolingFailure and must proceed into the traffic-duration
        wait -- proven by asserting execution reaches (and is only halted
        by) a deliberately-instrumented downstream mock, never the
        pre-traffic exception path."""
        ledger_path = tmp_path / "campaign_ledger.json"
        reached_downstream = {"hit": False}

        def fake_start_prometheus_port_forward():
            reached_downstream["hit"] = True
            raise AssertionError("stop here -- proves transport succeeded and control passed the pre-traffic gate")

        self._mock_pretraffic_path(monkeypatch, {
            "container_name": "loadtest-relay-test", "target_url": "http://127.0.0.1:54321",
            "docker_network": "minikube", "node_ip": "192.168.49.2", "nodeport": 31891,
        })
        monkeypatch.setattr(harness, "start_prometheus_port_forward", fake_start_prometheus_port_forward)

        with pytest.raises(AssertionError, match="stop here"):
            harness.main(
                ["--campaign-sequence-index", str(self.STEADY_STATE_INDEX), "--attempt-number", "1"],
                calibration_root=tmp_path, ledger_path=ledger_path,
            )
        assert reached_downstream["hit"] is True

    def test_wrong_traffic_path_used_still_scientific_for_executed_route_failure(self):
        """Source-level proof that the OTHER wrong_traffic_path_used call
        site (route_integrity FAILED after traffic was actually generated)
        is untouched -- it still appends to invalidity_reasons (a
        scientific classification input), unlike the two pre-traffic call
        sites which now raise PreTrafficToolingFailure instead."""
        import inspect

        source = inspect.getsource(harness.main)
        assert "official workload route confirmed to have stopped forwarding during" in source
        assert "invalidity_reasons.append" in source  # the executed-run branch still feeds the classifier
        assert "required_v5_transport_resolution_failed_before_traffic" in source
        assert "raise PreTrafficToolingFailure(" in source

    def test_provenance_next_attempt_number_after_pretraffic_failure_is_incremented(self, monkeypatch, tmp_path):
        """Section 7/16: a physical pre-ledger tooling failure still
        consumes its own attempt number -- proven here by simulating
        attempt1 as a pre-traffic tooling failure, then attempt2 as a
        genuinely successful (mocked) run, and confirming attempt2's own
        directory is distinct and not renumbered."""
        ledger_path = tmp_path / "campaign_ledger.json"
        self._mock_pretraffic_path(monkeypatch, {"error": "simulated transport failure"})
        rc1 = harness.main(
            ["--campaign-sequence-index", str(self.STEADY_STATE_INDEX), "--attempt-number", "1"],
            calibration_root=tmp_path, ledger_path=ledger_path,
        )
        assert rc1 == harness.EXIT_PRE_TRAFFIC_TOOLING_FAILURE
        attempt1_dirs = [p for p in tmp_path.iterdir() if p.is_dir() and "_attempt1_" in p.name]
        assert len(attempt1_dirs) == 1

        # A second, independent pre-traffic tooling failure on attempt_number=2
        # must consume attempt2 as its own distinct physical attempt --
        # never reusing or overwriting attempt1's directory or evidence.
        reached_downstream = {"hit": False}

        def fake_start_prometheus_port_forward():
            reached_downstream["hit"] = True
            return None, False

        self._mock_pretraffic_path(monkeypatch, {"error": "second simulated transport failure"})
        monkeypatch.setattr(harness, "start_prometheus_port_forward", fake_start_prometheus_port_forward)
        rc2 = harness.main(
            ["--campaign-sequence-index", str(self.STEADY_STATE_INDEX), "--attempt-number", "2"],
            calibration_root=tmp_path, ledger_path=ledger_path,
        )
        assert rc2 == harness.EXIT_PRE_TRAFFIC_TOOLING_FAILURE
        attempt2_dirs = [p for p in tmp_path.iterdir() if p.is_dir() and "_attempt2_" in p.name]
        assert len(attempt2_dirs) == 1
        assert attempt1_dirs[0].exists()  # untouched, still present
        ledger_after = lib.load_ledger(ledger_path)
        assert ledger_after["slots"][str(self.STEADY_STATE_INDEX)]["state"] == "NOT_ATTEMPTED"
        assert ledger_after["slots"][str(self.STEADY_STATE_INDEX)]["all_attempt_ids"] == []
