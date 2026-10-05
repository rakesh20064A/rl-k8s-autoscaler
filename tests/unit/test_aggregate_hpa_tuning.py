"""Unit tests for scripts/aggregate_hpa_tuning.py (Phase 11C.4P).

Fully offline -- builds small synthetic fixture directories mirroring the
real persistent-data/ layout (traffic/<id>/{metadata.json, requests.csv},
hpa/<id>/hpa_status.csv, logs/<id>_trial_metadata.json). No Kubernetes,
Prometheus, or network access, and no dependency on any real trial data.
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.aggregate_hpa_tuning import (
    ManifestError,
    StructuralValidationError,
    aggregate_candidate,
    aggregate_trial,
    compute_latency,
    compute_reliability,
    compute_replica_efficiency,
    load_manifest,
    verify_trial_structural,
)

REAL_MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "experiments/manifests/hpa_tuning_official_9_trials.yaml"
)
REAL_SCHEDULE_SHA = "4f648411e039f9229cd35a06374e6870dc75eca67e385e67f4f52bb0b5ec55ad"

HPA_STATUS_FIELDNAMES = [
    "timestamp_utc", "timestamp_epoch", "elapsed_seconds",
    "hpa_status_current_replicas", "hpa_status_desired_replicas",
    "hpa_current_cpu_utilization_percent", "hpa_current_cpu_average_value_millicores",
    "hpa_target_cpu_utilization_percent", "hpa_min_replicas", "hpa_max_replicas",
    "able_to_scale", "scaling_active", "scaling_limited", "last_scale_time",
    "deployment_spec_desired_replicas", "deployment_status_replicas",
    "deployment_status_ready_replicas", "deployment_status_available_replicas",
    "prom_current_cpu_utilization_percent", "prom_current_cpu_average_value_millicores",
    "prom_desired_replicas", "kubectl_error_count",
]

REQUESTS_FIELDNAMES = [
    "request_index", "planned_elapsed_seconds", "actual_dispatch_elapsed_seconds",
    "dispatch_lag_seconds", "completion_elapsed_seconds", "configured_intensity",
    "phase_name", "target_rps", "outcome", "http_status", "client_latency_seconds",
]

TINY_PROFILE_YAML = """
profile: tiny_test
seed: 42
endpoint: /work
intensity: 3
phases:
  - type: constant
    name: phase_a
    duration_seconds: 10
    target_rps: 2
  - type: constant
    name: recovery_2rps
    duration_seconds: 20
    target_rps: 6
"""


def _write_hpa_status_csv(path: Path, replica_rows):
    """replica_rows: list of (elapsed_seconds, hpa_desired, dep_desired, current, ready, available)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HPA_STATUS_FIELDNAMES)
        writer.writeheader()
        for elapsed, hpa_d, dep_d, cur, ready, avail in replica_rows:
            writer.writerow({
                "timestamp_utc": "", "timestamp_epoch": "", "elapsed_seconds": elapsed,
                "hpa_status_current_replicas": cur, "hpa_status_desired_replicas": hpa_d,
                "hpa_current_cpu_utilization_percent": 100.0,
                "hpa_current_cpu_average_value_millicores": 100.0,
                "hpa_target_cpu_utilization_percent": 200.0,
                "hpa_min_replicas": 1, "hpa_max_replicas": 8,
                "able_to_scale": 1, "scaling_active": 1, "scaling_limited": 0,
                "last_scale_time": "",
                "deployment_spec_desired_replicas": dep_d,
                "deployment_status_replicas": cur,
                "deployment_status_ready_replicas": ready,
                "deployment_status_available_replicas": avail,
                "prom_current_cpu_utilization_percent": 100.0,
                "prom_current_cpu_average_value_millicores": 100.0,
                "prom_desired_replicas": dep_d,
                "kubectl_error_count": 0,
            })


def _write_requests_csv(path: Path, rows):
    """rows: list of dicts with phase_name, outcome, client_latency_seconds (or None)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUESTS_FIELDNAMES)
        writer.writeheader()
        for i, r in enumerate(rows):
            writer.writerow({
                "request_index": i,
                "planned_elapsed_seconds": 0.0,
                "actual_dispatch_elapsed_seconds": 0.0,
                "dispatch_lag_seconds": 0.001,
                "completion_elapsed_seconds": 0.1,
                "configured_intensity": 3,
                "phase_name": r["phase_name"],
                "target_rps": 3.0,
                "outcome": r["outcome"],
                "http_status": r.get("http_status", 200 if r["outcome"] == "success_2xx" else ""),
                "client_latency_seconds": "" if r["client_latency_seconds"] is None else r["client_latency_seconds"],
            })


def _build_trial_dir(root: Path, exp_id: str, target: int, schedule_sha=REAL_SCHEDULE_SHA,
                      status="valid", connection_mode="fresh",
                      target_url="http://127.0.0.1:60000",
                      offered=10, sent=10,
                      replica_rows=None, request_rows=None,
                      kubectl_errors_present=False):
    traffic_dir = root / "traffic" / exp_id
    hpa_dir = root / "hpa" / exp_id
    logs_dir = root / "logs"
    traffic_dir.mkdir(parents=True, exist_ok=True)
    hpa_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    with open(traffic_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_offered": offered,
            "total_sent": sent,
            "connection_mode": connection_mode,
            "target_url": target_url,
            "schedule_generation_info": {"schedule_sha256": schedule_sha},
        }, f)

    with open(logs_dir / f"{exp_id}_trial_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"experiment_id": exp_id, "target": str(target), "status": status}, f)

    if request_rows is None:
        request_rows = [
            {"phase_name": "phase_a", "outcome": "success_2xx", "client_latency_seconds": 0.05}
            for _ in range(offered)
        ]
    _write_requests_csv(traffic_dir / "requests.csv", request_rows)

    if replica_rows is None:
        replica_rows = [(0.0, 1, 1, 1, 1, 1)]
    if kubectl_errors_present:
        replica_rows = [(t, a, b, c, d, e) for (t, a, b, c, d, e) in replica_rows]
        _write_hpa_status_csv(hpa_dir / "hpa_status.csv", replica_rows)
        # inject a nonzero kubectl_error_count row
        with open(hpa_dir / "hpa_status.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HPA_STATUS_FIELDNAMES)
            row = {k: "" for k in HPA_STATUS_FIELDNAMES}
            row["elapsed_seconds"] = replica_rows[-1][0] + 1
            row["kubectl_error_count"] = 1
            writer.writerow(row)
    else:
        _write_hpa_status_csv(hpa_dir / "hpa_status.csv", replica_rows)


class TestRealManifest(unittest.TestCase):
    """Exercises the real frozen manifest checked into the repo."""

    def test_real_manifest_has_nine_trials_three_per_candidate_one_per_block(self):
        manifest = load_manifest(REAL_MANIFEST_PATH)
        self.assertEqual(len(manifest.trials), 9)
        per_candidate = {200: [], 230: [], 260: []}
        for t in manifest.trials:
            per_candidate[t.candidate].append(t.block)
        for candidate, blocks in per_candidate.items():
            self.assertEqual(len(blocks), 3, f"candidate {candidate} must have 3 trials")
            self.assertEqual(sorted(blocks), [1, 2, 3], f"candidate {candidate} must cover blocks 1-3 exactly once each")

    def test_real_manifest_schedule_hash_matches_corr24(self):
        manifest = load_manifest(REAL_MANIFEST_PATH)
        self.assertEqual(manifest.schedule_sha256, REAL_SCHEDULE_SHA)


class TestManifestValidation(unittest.TestCase):
    def _minimal_valid_trials(self):
        trials = []
        for block, order in [(1, [200, 230, 260]), (2, [260, 200, 230]), (3, [230, 260, 200])]:
            for candidate in order:
                trials.append({
                    "experiment_id": f"hpa_tune_t{candidate}_b{block}r{block}_synthetic",
                    "candidate": candidate,
                    "block": block,
                    "repeat": block,
                })
        return trials

    def _write_manifest(self, tmpdir: Path, trials, schedule_sha=REAL_SCHEDULE_SHA):
        path = Path(tmpdir) / "manifest.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump({
                "schedule_sha256": schedule_sha,
                "expected_planned_requests": 4950,
                "profile": "configs/traffic_profiles/hpa_tuning/hpa_tuning_v2.yaml",
                "scored_end_seconds": 1230.0,
                "trials": trials,
            }, f)
        return path

    def test_valid_synthetic_manifest_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_manifest(tmp, self._minimal_valid_trials())
            manifest = load_manifest(path)
            self.assertEqual(len(manifest.trials), 9)

    def test_wrong_trial_count_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            trials = self._minimal_valid_trials()[:8]
            path = self._write_manifest(tmp, trials)
            with self.assertRaises(ManifestError):
                load_manifest(path)

    def test_duplicate_experiment_id_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            trials = self._minimal_valid_trials()
            trials[1]["experiment_id"] = trials[0]["experiment_id"]
            path = self._write_manifest(tmp, trials)
            with self.assertRaises(ManifestError):
                load_manifest(path)

    def test_unexpected_candidate_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            trials = self._minimal_valid_trials()
            trials[0]["candidate"] = 220  # superseded family, CORR-24
            path = self._write_manifest(tmp, trials)
            with self.assertRaises(ManifestError):
                load_manifest(path)

    def test_missing_block_for_candidate_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            trials = self._minimal_valid_trials()
            # Duplicate block 1's 200% entry in place of block 2's, so 200%
            # never appears in block 2.
            trials[4]["block"] = 1
            trials[4]["repeat"] = 1
            trials[4]["experiment_id"] = "hpa_tune_t200_dup"
            path = self._write_manifest(tmp, trials)
            with self.assertRaises(ManifestError):
                load_manifest(path)


class TestStructuralValidation(unittest.TestCase):
    def test_valid_trial_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_trial_dir(root, "exp1", target=200, offered=4950, sent=4950)
            manifest = load_manifest(REAL_MANIFEST_PATH)
            trial = next(t for t in manifest.trials if True)
            trial.experiment_id = "exp1"
            trial.candidate = 200
            result = verify_trial_structural(trial, root, manifest)
            self.assertTrue(result["structurally_valid"])

    def test_missing_artifact_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = load_manifest(REAL_MANIFEST_PATH)
            trial = manifest.trials[0]
            trial.experiment_id = "does_not_exist"
            with self.assertRaises(StructuralValidationError):
                verify_trial_structural(trial, root, manifest)

    def test_schedule_hash_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_trial_dir(root, "exp_bad_hash", target=200, schedule_sha="deadbeef")
            manifest = load_manifest(REAL_MANIFEST_PATH)
            trial = manifest.trials[0]
            trial.experiment_id = "exp_bad_hash"
            trial.candidate = 200
            with self.assertRaises(StructuralValidationError):
                verify_trial_structural(trial, root, manifest)

    def test_offered_not_equal_sent_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_trial_dir(root, "exp_mismatch", target=200, offered=4950, sent=4949)
            manifest = load_manifest(REAL_MANIFEST_PATH)
            trial = manifest.trials[0]
            trial.experiment_id = "exp_mismatch"
            trial.candidate = 200
            with self.assertRaises(StructuralValidationError):
                verify_trial_structural(trial, root, manifest)

    def test_wrong_connection_mode_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_trial_dir(root, "exp_keepalive", target=200, connection_mode="keepalive", offered=4950, sent=4950)
            manifest = load_manifest(REAL_MANIFEST_PATH)
            trial = manifest.trials[0]
            trial.experiment_id = "exp_keepalive"
            trial.candidate = 200
            with self.assertRaises(StructuralValidationError):
                verify_trial_structural(trial, root, manifest)

    def test_port_forward_target_url_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_trial_dir(root, "exp_portforward", target=200, target_url="http://localhost:8080", offered=4950, sent=4950)
            manifest = load_manifest(REAL_MANIFEST_PATH)
            trial = manifest.trials[0]
            trial.experiment_id = "exp_portforward"
            trial.candidate = 200
            with self.assertRaises(StructuralValidationError):
                verify_trial_structural(trial, root, manifest)

    def test_kubectl_error_present_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_trial_dir(root, "exp_kubectl_err", target=200, kubectl_errors_present=True, offered=4950, sent=4950)
            manifest = load_manifest(REAL_MANIFEST_PATH)
            trial = manifest.trials[0]
            trial.experiment_id = "exp_kubectl_err"
            trial.candidate = 200
            with self.assertRaises(StructuralValidationError):
                verify_trial_structural(trial, root, manifest)

    def test_target_candidate_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_trial_dir(root, "exp_mismatch_target", target=230)
            manifest = load_manifest(REAL_MANIFEST_PATH)
            trial = manifest.trials[0]
            trial.experiment_id = "exp_mismatch_target"
            trial.candidate = 200  # manifest says 200, harness recorded 230
            with self.assertRaises(StructuralValidationError):
                verify_trial_structural(trial, root, manifest)

    def test_invalid_harness_status_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_trial_dir(root, "exp_invalid_status", target=200, status="invalid")
            manifest = load_manifest(REAL_MANIFEST_PATH)
            trial = manifest.trials[0]
            trial.experiment_id = "exp_invalid_status"
            trial.candidate = 200
            with self.assertRaises(StructuralValidationError):
                verify_trial_structural(trial, root, manifest)


class TestLatencyAndReliability(unittest.TestCase):
    def test_success_and_error_counting(self):
        rows = [
            {"phase_name": "p", "outcome": "success_2xx", "client_latency_seconds": 0.05},
            {"phase_name": "p", "outcome": "success_2xx", "client_latency_seconds": 0.06},
            {"phase_name": "p", "outcome": "timeout", "client_latency_seconds": 5.0},
            {"phase_name": "p", "outcome": "connection_error", "client_latency_seconds": None},
        ]
        reliability = compute_reliability(rows)
        self.assertEqual(reliability["n"], 4)
        self.assertEqual(reliability["success_2xx"], 2)
        self.assertEqual(reliability["timeout"], 1)
        self.assertEqual(reliability["connection_error"], 1)
        self.assertAlmostEqual(reliability["success_rate"], 2 / 4)

    def test_latency_percentiles_and_tail_exclude_connection_errors(self):
        rows = [
            {"client_latency_seconds": 0.05, "outcome": "success_2xx"},
            {"client_latency_seconds": 0.35, "outcome": "success_2xx"},
            {"client_latency_seconds": None, "outcome": "connection_error"},
        ]
        lat = compute_latency(rows)
        # Only the two rows with a parseable latency are scored.
        self.assertEqual(lat["n_latency_scored"], 2)
        self.assertEqual(lat["gt300_count"], 1)
        self.assertAlmostEqual(lat["gt300_fraction"], 0.5)
        self.assertAlmostEqual(lat["max_ms"], 350.0)

    def test_timeout_included_in_latency_scoring(self):
        rows = [
            {"client_latency_seconds": 0.05, "outcome": "success_2xx"},
            {"client_latency_seconds": 5.0, "outcome": "timeout"},
        ]
        lat = compute_latency(rows)
        self.assertEqual(lat["n_latency_scored"], 2)
        self.assertEqual(lat["gt300_count"], 1)

    def test_empty_latency_set_returns_none_not_crash(self):
        rows = [{"client_latency_seconds": None, "outcome": "connection_error"}]
        lat = compute_latency(rows)
        self.assertEqual(lat["n_latency_scored"], 0)
        self.assertIsNone(lat["p95_ms"])
        self.assertIsNone(lat["gt300_fraction"])


class TestReplicaIntegrationDelegation(unittest.TestCase):
    def test_delegates_to_shared_integrator_and_matches_manual_computation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hpa_status.csv"
            # 1 replica [0,100), 2 replicas [100,1230)
            _write_hpa_status_csv(path, [(0.0, 1, 1, 1, 1, 1), (100.0, 2, 2, 2, 2, 2)])
            result = compute_replica_efficiency(path, scored_end=1230.0)
            expected_seconds = 1 * 100 + 2 * 1130
            self.assertAlmostEqual(result["current"]["replica_seconds"], expected_seconds)
            self.assertAlmostEqual(result["ready"]["replica_seconds"], expected_seconds)
            self.assertEqual(result["current"]["peak"], 2)
            # All five CORR-25 fields must be present -- delegation to
            # integrate_all_fields(), not a partial reimplementation.
            self.assertEqual(
                set(result.keys()),
                {"hpa_desired", "deployment_desired", "current", "ready", "available"},
            )


class TestMedianMinMaxAndCandidateAggregation(unittest.TestCase):
    def test_aggregate_candidate_median_min_max(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_path = Path(tmp) / "profile.yaml"
            profile_path.write_text(TINY_PROFILE_YAML, encoding="utf-8")

            # Three synthetic 200% trials with distinct replica-seconds and p95s.
            trial_results = []
            for i, (rs_current, p95) in enumerate([(1000.0, 100.0), (1100.0, 120.0), (900.0, 80.0)]):
                exp_id = f"synthetic_200_{i}"
                replica_rows = [(0.0, 1, 1, 1, 1, 1), (5.0, 2, 2, 2, 2, 2)]
                request_rows = [
                    {"phase_name": "phase_a", "outcome": "success_2xx", "client_latency_seconds": p95 / 1000.0}
                    for _ in range(5)
                ]
                _build_trial_dir(
                    root, exp_id, target=200, offered=5, sent=5,
                    replica_rows=replica_rows, request_rows=request_rows,
                )
                from scripts.aggregate_hpa_tuning import Manifest, TrialSpec
                fake_manifest = Manifest(
                    schedule_sha256=REAL_SCHEDULE_SHA, expected_planned_requests=5,
                    profile=str(profile_path.relative_to(Path(tmp)) if False else profile_path),
                    scored_end_seconds=1230.0, trials=[],
                )
                trial_spec = TrialSpec(experiment_id=exp_id, candidate=200, block=i + 1, repeat=i + 1)
                trial_results.append(aggregate_trial(trial_spec, root, fake_manifest))

            summary = aggregate_candidate(200, trial_results)
            self.assertEqual(summary["valid_runs"], 3)
            self.assertAlmostEqual(summary["p95_ms_stats"]["median"], 100.0)
            self.assertAlmostEqual(summary["p95_ms_stats"]["min"], 80.0)
            self.assertAlmostEqual(summary["p95_ms_stats"]["max"], 120.0)


if __name__ == "__main__":
    unittest.main()
