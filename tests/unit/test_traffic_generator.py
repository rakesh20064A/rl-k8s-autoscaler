"""Unit tests for host-side open-loop traffic generator.

All tests run completely offline and mock network interactions.
"""

import asyncio
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import yaml

from loadgen.traffic_generator import (
    BurstPhase,
    CompletedRequest,
    ConstantPhase,
    PeriodicPhase,
    PlannedRequest,
    RampPhase,
    ScheduleValidationError,
    StepPhase,
    TrafficGenerator,
    TrafficProfile,
    generate_arrival_schedule,
    load_planned_schedule,
    parse_args,
    parse_traffic_profile,
    SCHEDULE_CSV_FIELDNAMES,
)

_real_sleep = asyncio.sleep


class _VirtualClock:
    """Virtual clock for deterministic, millisecond-fast asyncio scheduling tests."""

    def __init__(self, start_time: float = 1000.0) -> None:
        self.current_time = start_time

    def monotonic(self) -> float:
        return self.current_time

    async def sleep(self, delay: float) -> None:
        self.current_time += delay
        await _real_sleep(0)


class TestProfileCalculations(unittest.TestCase):
    """Test mathematical target rate calculations for all phase types."""

    def test_constant_phase(self):
        phase = ConstantPhase("c1", duration_seconds=10.0, target_rps=25.0)
        self.assertEqual(phase.get_target_rps(0.0), 25.0)
        self.assertEqual(phase.get_target_rps(5.0), 25.0)
        self.assertEqual(phase.get_target_rps(10.0), 25.0)

    def test_ramp_phase(self):
        phase = RampPhase("r1", duration_seconds=100.0, start_rps=10.0, end_rps=50.0)
        self.assertEqual(phase.get_target_rps(0.0), 10.0)
        self.assertEqual(phase.get_target_rps(50.0), 30.0)
        self.assertEqual(phase.get_target_rps(100.0), 50.0)

    def test_step_phase(self):
        steps = [
            {"at_second": 0, "rps": 5.0},
            {"at_second": 20, "rps": 40.0},
            {"at_second": 50, "rps": 15.0},
        ]
        phase = StepPhase("s1", duration_seconds=80.0, steps=steps)
        self.assertEqual(phase.get_target_rps(0.0), 5.0)
        self.assertEqual(phase.get_target_rps(19.9), 5.0)
        self.assertEqual(phase.get_target_rps(20.0), 40.0)
        self.assertEqual(phase.get_target_rps(49.9), 40.0)
        self.assertEqual(phase.get_target_rps(50.0), 15.0)
        self.assertEqual(phase.get_target_rps(75.0), 15.0)

    def test_burst_phase(self):
        phase = BurstPhase(
            name="b1",
            duration_seconds=100.0,
            baseline_rps=5.0,
            burst_rps=50.0,
            burst_duration_seconds=10.0,
            inter_burst_interval_seconds=20.0,
        )
        # Cycle length = 30s (0-10s burst, 10-30s baseline)
        self.assertEqual(phase.get_target_rps(0.0), 50.0)
        self.assertEqual(phase.get_target_rps(9.9), 50.0)
        self.assertEqual(phase.get_target_rps(10.0), 5.0)
        self.assertEqual(phase.get_target_rps(29.9), 5.0)
        self.assertEqual(phase.get_target_rps(30.0), 50.0)  # Next cycle burst

    def test_periodic_phase(self):
        phase = PeriodicPhase(
            name="p1",
            duration_seconds=60.0,
            mean_rps=20.0,
            amplitude_rps=10.0,
            period_seconds=20.0,
        )
        # Period = 20s. sin(0)=0 -> 20. sin(pi/2)=1 at t=5s -> 30. sin(3pi/2)=-1 at t=15s -> 10.
        self.assertAlmostEqual(phase.get_target_rps(0.0), 20.0, places=4)
        self.assertAlmostEqual(phase.get_target_rps(5.0), 30.0, places=4)
        self.assertAlmostEqual(phase.get_target_rps(10.0), 20.0, places=4)
        self.assertAlmostEqual(phase.get_target_rps(15.0), 10.0, places=4)


class TestProfileParsing(unittest.TestCase):
    """Test parsing of unified YAML configuration."""

    def test_parse_unified_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_content = {
                "profile": "test_unified",
                "description": "Multi-phase test profile",
                "seed": 123,
                "endpoint": "/work",
                "intensity": 3,
                "phases": [
                    {"name": "p1", "type": "constant", "duration_seconds": 10.0, "target_rps": 5.0},
                    {"name": "p2", "type": "ramp", "duration_seconds": 20.0, "start_rps": 5.0, "end_rps": 25.0},
                ],
            }
            yaml.safe_dump(yaml_content, f)
            temp_path = Path(f.name)

        try:
            profile = parse_traffic_profile(temp_path)
            self.assertEqual(profile.profile_name, "test_unified")
            self.assertEqual(profile.seed, 123)
            self.assertEqual(profile.intensity, 3)
            self.assertEqual(profile.total_duration, 30.0)
            self.assertEqual(len(profile.phases), 2)

            rate, p_name, p_type = profile.get_phase_and_target_rps(5.0)
            self.assertEqual(rate, 5.0)
            self.assertEqual(p_name, "p1")

            rate2, p_name2, p_type2 = profile.get_phase_and_target_rps(20.0)
            # In p2 (at 10s offset into 20s ramp from 5 to 25): rate = 15.0
            self.assertAlmostEqual(rate2, 15.0, places=4)
            self.assertEqual(p_name2, "p2")
        finally:
            temp_path.unlink()


class TestArrivalScheduleGeneration(unittest.TestCase):
    """Test deterministic arrival schedule generation properties."""

    def setUp(self):
        self.phases = [
            ConstantPhase("p1", duration_seconds=10.0, target_rps=10.0),
            ConstantPhase("p2", duration_seconds=10.0, target_rps=20.0),
        ]
        self.profile = TrafficProfile(
            profile_name="test_sched",
            description="test",
            seed=42,
            endpoint="/work",
            intensity=3,
            phases=self.phases,
        )

    def test_deterministic_schedule_reproducibility(self):
        s1 = generate_arrival_schedule(self.profile, seed=42, arrival_process="deterministic")
        s2 = generate_arrival_schedule(self.profile, seed=42, arrival_process="deterministic")

        self.assertEqual(len(s1), len(s2))
        for r1, r2 in zip(s1, s2):
            self.assertEqual(r1.planned_elapsed_seconds, r2.planned_elapsed_seconds)
            self.assertEqual(r1.target_rps, r2.target_rps)

    def test_poisson_schedule_same_seed_identical(self):
        s1 = generate_arrival_schedule(self.profile, seed=101, arrival_process="poisson")
        s2 = generate_arrival_schedule(self.profile, seed=101, arrival_process="poisson")

        self.assertEqual(len(s1), len(s2))
        for r1, r2 in zip(s1, s2):
            self.assertEqual(r1.planned_elapsed_seconds, r2.planned_elapsed_seconds)
            self.assertEqual(r1.target_rps, r2.target_rps)

    def test_poisson_schedule_different_seed_differs(self):
        s1 = generate_arrival_schedule(self.profile, seed=101, arrival_process="poisson")
        s2 = generate_arrival_schedule(self.profile, seed=202, arrival_process="poisson")

        # Timestamps must differ
        timestamps1 = [r.planned_elapsed_seconds for r in s1]
        timestamps2 = [r.planned_elapsed_seconds for r in s2]
        self.assertNotEqual(timestamps1, timestamps2)

    def test_schedule_bounded_by_duration(self):
        schedule = generate_arrival_schedule(self.profile, seed=42, arrival_process="poisson")
        for req in schedule:
            self.assertLess(req.planned_elapsed_seconds, self.profile.total_duration)

    def test_schedule_monotonic(self):
        schedule = generate_arrival_schedule(self.profile, seed=42, arrival_process="poisson")
        for i in range(len(schedule) - 1):
            self.assertLess(schedule[i].planned_elapsed_seconds, schedule[i + 1].planned_elapsed_seconds)


class TestScheduleReplay(unittest.TestCase):
    """Test exact-replay of a persisted traffic_schedule.csv (Phase 11C.1).

    All tests here are fully offline -- no HTTP client is ever instantiated,
    matching the requirement that schedule generation/replay validation must
    never send requests to the live demo-app.
    """

    def setUp(self):
        self.phases = [
            ConstantPhase("p1", duration_seconds=10.0, target_rps=10.0),
            ConstantPhase("p2", duration_seconds=10.0, target_rps=20.0),
        ]
        self.profile = TrafficProfile(
            profile_name="test_replay",
            description="test",
            seed=42,
            endpoint="/work",
            intensity=3,
            phases=self.phases,
        )

    def _write_schedule_csv(self, path: Path, schedule):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SCHEDULE_CSV_FIELDNAMES)
            writer.writeheader()
            for req in schedule:
                writer.writerow(req.to_dict())

    def test_round_trip_generate_save_load_matches_exactly(self):
        original = generate_arrival_schedule(self.profile, seed=42, arrival_process="poisson")
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "traffic_schedule.csv"
            self._write_schedule_csv(csv_path, original)
            loaded = load_planned_schedule(csv_path)

        self.assertEqual(len(original), len(loaded))
        for orig_req, loaded_req in zip(original, loaded):
            self.assertEqual(orig_req.request_index, loaded_req.request_index)
            # planned_elapsed_seconds is persisted with 6 decimal places (see to_dict())
            self.assertAlmostEqual(
                orig_req.planned_elapsed_seconds, loaded_req.planned_elapsed_seconds, places=6
            )
            self.assertEqual(orig_req.phase_name, loaded_req.phase_name)
            self.assertEqual(orig_req.phase_type, loaded_req.phase_type)
            self.assertEqual(orig_req.configured_intensity, loaded_req.configured_intensity)
            self.assertEqual(orig_req.endpoint, loaded_req.endpoint)

    def test_replay_never_rerolls_randomness(self):
        """Loading a persisted Poisson schedule must reproduce it verbatim,
        regardless of what seed/arrival_process the caller happens to pass
        elsewhere -- load_planned_schedule() takes no seed at all."""
        original = generate_arrival_schedule(self.profile, seed=7, arrival_process="poisson")
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "traffic_schedule.csv"
            self._write_schedule_csv(csv_path, original)
            loaded_once = load_planned_schedule(csv_path)
            loaded_twice = load_planned_schedule(csv_path)

        self.assertEqual(
            [r.planned_elapsed_seconds for r in loaded_once],
            [r.planned_elapsed_seconds for r in loaded_twice],
        )

    def test_missing_file_raises_validation_error(self):
        with self.assertRaises(ScheduleValidationError):
            load_planned_schedule(Path("/nonexistent/traffic_schedule.csv"))

    def test_missing_column_raises_validation_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "traffic_schedule.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["request_index", "planned_elapsed_seconds"])
                writer.writeheader()
                writer.writerow({"request_index": 0, "planned_elapsed_seconds": 0.0})
            with self.assertRaises(ScheduleValidationError):
                load_planned_schedule(csv_path)

    def test_empty_file_raises_validation_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "traffic_schedule.csv"
            csv_path.write_text("")
            with self.assertRaises(ScheduleValidationError):
                load_planned_schedule(csv_path)

    def test_header_only_raises_validation_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "traffic_schedule.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=SCHEDULE_CSV_FIELDNAMES)
                writer.writeheader()
            with self.assertRaises(ScheduleValidationError):
                load_planned_schedule(csv_path)

    def test_duplicate_request_index_raises_validation_error(self):
        original = generate_arrival_schedule(self.profile, seed=42, arrival_process="deterministic")
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "traffic_schedule.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=SCHEDULE_CSV_FIELDNAMES)
                writer.writeheader()
                rows = [r.to_dict() for r in original[:3]]
                rows[1]["request_index"] = rows[0]["request_index"]  # force a duplicate
                for row in rows:
                    writer.writerow(row)
            with self.assertRaises(ScheduleValidationError):
                load_planned_schedule(csv_path)

    def test_non_monotonic_timestamps_raise_validation_error(self):
        original = generate_arrival_schedule(self.profile, seed=42, arrival_process="deterministic")
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "traffic_schedule.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=SCHEDULE_CSV_FIELDNAMES)
                writer.writeheader()
                rows = [r.to_dict() for r in original[:3]]
                # Swap two rows so elapsed time goes backwards
                rows[0], rows[1] = rows[1], rows[0]
                for row in rows:
                    writer.writerow(row)
            with self.assertRaises(ScheduleValidationError):
                load_planned_schedule(csv_path)

    def test_setup_directory_and_schedule_uses_replay_when_provided(self):
        """TrafficGenerator.setup_directory_and_schedule() must consume an existing
        schedule verbatim (never re-generate) when replay_schedule_path is set."""
        original = generate_arrival_schedule(self.profile, seed=99, arrival_process="poisson")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            source_csv = tmp_root / "source_schedule.csv"
            self._write_schedule_csv(source_csv, original)

            profile_path = tmp_root / "profile.yaml"
            profile_path.write_text(
                "seed: 42\nendpoint: /work\nintensity: 3\n"
                "phases:\n- type: constant\n  duration_seconds: 20\n  target_rps: 10\n"
            )

            generator = TrafficGenerator(
                experiment_id="replay_test",
                profile_path="profile.yaml",
                arrival_process="deterministic",  # deliberately different from source
                seed=1234,  # deliberately different seed
                project_root=tmp_root,
                replay_schedule_path=str(source_csv),
            )
            generator.setup_directory_and_schedule()

            self.assertEqual(len(generator.planned_schedule), len(original))
            for orig_req, replayed_req in zip(original, generator.planned_schedule):
                self.assertAlmostEqual(
                    orig_req.planned_elapsed_seconds, replayed_req.planned_elapsed_seconds, places=6
                )


class TestDispatcherAndOutcomes(unittest.IsolatedAsyncioTestCase):
    """Test asynchronous open-loop dispatcher and outcome classification."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)

        # Create mock smoke profile
        self.profile_dir = self.root_path / "configs" / "traffic_profiles"
        self.profile_dir.mkdir(parents=True)
        self.profile_file = self.profile_dir / "smoke.yaml"

        profile_content = {
            "profile": "smoke",
            "seed": 42,
            "endpoint": "/work",
            "intensity": 3,
            "phases": [
                {"name": "p1", "type": "constant", "duration_seconds": 2.0, "target_rps": 5.0}
            ],
        }
        with open(self.profile_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(profile_content, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_refuse_overwrite_existing_directory(self):
        exp_id = "exp_collision"
        target_dir = self.root_path / "persistent-data" / "traffic" / exp_id
        target_dir.mkdir(parents=True)

        generator = TrafficGenerator(
            experiment_id=exp_id,
            profile_path=str(self.profile_file.relative_to(self.root_path)),
            project_root=self.root_path,
        )

        with self.assertRaises(FileExistsError):
            generator.setup_directory_and_schedule()

    async def test_outcome_classification(self):
        generator = TrafficGenerator(
            experiment_id="exp_outcomes",
            profile_path=str(self.profile_file.relative_to(self.root_path)),
            project_root=self.root_path,
        )
        generator.setup_directory_and_schedule()

        mock_file = MagicMock()
        start_mono = 1000.0

        # 1. Test 200 OK -> success_2xx
        req_200 = PlannedRequest(0, 0.1, "p1", "constant", 5.0, 3)
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.get = MagicMock(return_value=mock_resp)

        # Patch client.get to be an async coroutine
        async def mock_get_200(*args, **kwargs):
            return mock_resp

        mock_client.get = mock_get_200
        await generator._dispatch_single_request(mock_client, req_200, start_mono, mock_file)
        self.assertEqual(generator.completed_requests[-1].outcome, "success_2xx")
        self.assertEqual(generator.completed_requests[-1].http_status, 200)

        # 2. Test 404 -> client_error_4xx
        req_404 = PlannedRequest(1, 0.2, "p1", "constant", 5.0, 3)
        mock_resp_404 = MagicMock(status_code=404)

        async def mock_get_404(*args, **kwargs):
            return mock_resp_404

        mock_client.get = mock_get_404
        await generator._dispatch_single_request(mock_client, req_404, start_mono, mock_file)
        self.assertEqual(generator.completed_requests[-1].outcome, "client_error_4xx")
        self.assertEqual(generator.completed_requests[-1].http_status, 404)

        # 3. Test 503 -> server_error_5xx
        req_503 = PlannedRequest(2, 0.3, "p1", "constant", 5.0, 3)
        mock_resp_503 = MagicMock(status_code=503)

        async def mock_get_503(*args, **kwargs):
            return mock_resp_503

        mock_client.get = mock_get_503
        await generator._dispatch_single_request(mock_client, req_503, start_mono, mock_file)
        self.assertEqual(generator.completed_requests[-1].outcome, "server_error_5xx")

        # 4. Test timeout -> timeout
        req_to = PlannedRequest(3, 0.4, "p1", "constant", 5.0, 3)

        async def mock_get_timeout(*args, **kwargs):
            raise httpx.TimeoutException("Read timeout")

        mock_client.get = mock_get_timeout
        await generator._dispatch_single_request(mock_client, req_to, start_mono, mock_file)
        self.assertEqual(generator.completed_requests[-1].outcome, "timeout")

        # 5. Test connection error -> connection_error
        req_conn = PlannedRequest(4, 0.5, "p1", "constant", 5.0, 3)

        async def mock_get_conn_err(*args, **kwargs):
            raise httpx.ConnectError("Connection refused")

        mock_client.get = mock_get_conn_err
        await generator._dispatch_single_request(mock_client, req_conn, start_mono, mock_file)
        self.assertEqual(generator.completed_requests[-1].outcome, "connection_error")

        # Phase 15B.3M.2 (CORR-39 resolution): diagnostic error_type/error_message
        # must now be captured for both timeout and connection_error outcomes,
        # rather than discarding the exception entirely.
        self.assertEqual(generator.completed_requests[-2].outcome, "timeout")
        self.assertEqual(generator.completed_requests[-2].error_type, "TimeoutException")
        self.assertIn("Read timeout", generator.completed_requests[-2].error_message)
        self.assertEqual(generator.completed_requests[-1].error_type, "ConnectError")
        self.assertIn("Connection refused", generator.completed_requests[-1].error_message)

        # A successful request must never carry error diagnostics.
        self.assertIsNone(generator.completed_requests[0].error_type)
        self.assertIsNone(generator.completed_requests[0].error_message)

        # to_dict() must always emit both columns (empty string, never absent).
        row = generator.completed_requests[-1].to_dict()
        self.assertIn("error_type", row)
        self.assertIn("error_message", row)
        success_row = generator.completed_requests[0].to_dict()
        self.assertEqual(success_row["error_type"], "")
        self.assertEqual(success_row["error_message"], "")

    def test_concurrency_saturation_dropped_overload(self):
        """When in_flight_count >= max_in_flight, request is recorded as dropped_overload without blocking."""
        generator = TrafficGenerator(
            experiment_id="exp_overload",
            profile_path=str(self.profile_file.relative_to(self.root_path)),
            max_in_flight=2,
            project_root=self.root_path,
        )
        generator.setup_directory_and_schedule()
        mock_file = MagicMock()

        generator.in_flight_count = 2  # Concurrency bound reached
        req = PlannedRequest(10, 0.5, "p1", "constant", 10.0, 3)

        generator._record_dropped_overload(req, start_mono=1000.0, requests_file=mock_file)

        self.assertEqual(len(generator.completed_requests), 1)
        record = generator.completed_requests[0]
        self.assertEqual(record.outcome, "dropped_overload")
        self.assertIsNone(record.http_status)
        self.assertIsNone(record.client_latency_seconds)


class TestMetadataAndSummaries(unittest.TestCase):
    """Test output metadata and summary aggregation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)

        self.profile_dir = self.root_path / "configs" / "traffic_profiles"
        self.profile_dir.mkdir(parents=True)
        self.profile_file = self.profile_dir / "smoke.yaml"

        profile_content = {
            "profile": "smoke_test",
            "seed": 42,
            "endpoint": "/work",
            "intensity": 3,
            "phases": [
                {"name": "p1", "type": "constant", "duration_seconds": 3.0, "target_rps": 2.0}
            ],
        }
        with open(self.profile_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(profile_content, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_metadata_finalization(self):
        generator = TrafficGenerator(
            experiment_id="exp_meta",
            profile_path=str(self.profile_file.relative_to(self.root_path)),
            project_root=self.root_path,
        )
        generator.setup_directory_and_schedule()

        generator.start_mono = 1000.0
        generator.end_mono = 1003.02
        generator.start_utc = "2026-09-06T00:00:00Z"
        generator.end_utc = "2026-09-06T00:00:03Z"

        # Simulate 2 completed requests
        generator.completed_requests = [
            CompletedRequest(0, 0.0, 0.001, 0.001, 0.05, 3, "p1", 2.0, "success_2xx", 200, 0.049),
            CompletedRequest(1, 0.5, 0.502, 0.002, 0.56, 3, "p1", 2.0, "success_2xx", 200, 0.058),
        ]

        generator._write_metadata()

        with open(generator.metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["experiment_id"], "exp_meta")
        self.assertEqual(data["profile"], "smoke_test")
        self.assertEqual(data["seed"], 42)
        self.assertEqual(data["requested_duration_seconds"], 3.0)
        self.assertAlmostEqual(data["actual_duration_seconds"], 3.02, places=2)
        self.assertEqual(data["outcome_totals"]["success_2xx"], 2)
        self.assertEqual(data["test_non_benchmark"], True)
        self.assertIn("dispatch_lag_stats", data)
        self.assertIn("client_latency_stats", data)

    def test_traffic_summary_aggregation(self):
        generator = TrafficGenerator(
            experiment_id="exp_summary",
            profile_path=str(self.profile_file.relative_to(self.root_path)),
            project_root=self.root_path,
        )
        generator.setup_directory_and_schedule()

        # Add mock completed requests across 2 seconds
        generator.completed_requests = [
            CompletedRequest(0, 0.1, 0.101, 0.001, 0.15, 3, "p1", 2.0, "success_2xx", 200, 0.049),
            CompletedRequest(1, 0.6, 0.602, 0.002, 0.65, 3, "p1", 2.0, "server_error_5xx", 500, 0.048),
            CompletedRequest(2, 1.2, 1.202, 0.002, 1.25, 3, "p1", 2.0, "success_2xx", 200, 0.048),
        ]

        generator._write_traffic_summary()

        self.assertTrue(generator.summary_path.exists())
        with open(generator.summary_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        self.assertGreaterEqual(len(lines), 4)  # Header + 3 seconds


class TestCLIArgs(unittest.TestCase):
    """Test CLI argument parser."""

    def test_cli_parsing(self):
        args = parse_args([
            "--profile", "configs/traffic_profiles/smoke_test.yaml",
            "--experiment-id", "exp_cli_test",
            "--target-url", "http://localhost:8080",
            "--arrival-process", "deterministic",
            "--seed", "99",
            "--max-in-flight", "25",
            "--timeout", "3.0",
        ])
        self.assertEqual(args.profile, "configs/traffic_profiles/smoke_test.yaml")
        self.assertEqual(args.experiment_id, "exp_cli_test")
        self.assertEqual(args.target_url, "http://localhost:8080")
        self.assertEqual(args.arrival_process, "deterministic")
        self.assertEqual(args.seed, 99)
        self.assertEqual(args.max_in_flight, 25)
        self.assertEqual(args.timeout, 3.0)

    def test_connection_mode_defaults_to_keepalive(self):
        args = parse_args([
            "--profile", "configs/traffic_profiles/smoke_test.yaml",
            "--experiment-id", "exp_cli_test",
        ])
        self.assertEqual(args.connection_mode, "keepalive")

    def test_connection_mode_fresh_accepted(self):
        args = parse_args([
            "--profile", "configs/traffic_profiles/smoke_test.yaml",
            "--experiment-id", "exp_cli_test",
            "--connection-mode", "fresh",
        ])
        self.assertEqual(args.connection_mode, "fresh")

    def test_connection_mode_rejects_invalid_value(self):
        with self.assertRaises(SystemExit):
            parse_args([
                "--profile", "configs/traffic_profiles/smoke_test.yaml",
                "--experiment-id", "exp_cli_test",
                "--connection-mode", "bogus",
            ])


class TestConnectionMode(unittest.TestCase):
    """Test TrafficGenerator's connection_mode validation and metadata persistence."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_root = Path(self.tmpdir.name)
        self.profile_path = self.tmp_root / "profile.yaml"
        self.profile_path.write_text(
            "seed: 42\nendpoint: /work\nintensity: 3\n"
            "phases:\n- type: constant\n  duration_seconds: 5\n  target_rps: 2\n"
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_invalid_connection_mode_rejected(self):
        with self.assertRaises(ValueError):
            TrafficGenerator(
                experiment_id="bad_mode_test",
                profile_path="profile.yaml",
                project_root=self.tmp_root,
                connection_mode="not-a-real-mode",
            )

    def test_default_connection_mode_is_keepalive(self):
        gen = TrafficGenerator(
            experiment_id="default_mode_test",
            profile_path="profile.yaml",
            project_root=self.tmp_root,
        )
        self.assertEqual(gen.connection_mode, "keepalive")

    def test_fresh_connection_mode_accepted(self):
        gen = TrafficGenerator(
            experiment_id="fresh_mode_test",
            profile_path="profile.yaml",
            project_root=self.tmp_root,
            connection_mode="fresh",
        )
        self.assertEqual(gen.connection_mode, "fresh")


class TestProfileDurationAndLifecycleSemantics(unittest.IsolatedAsyncioTestCase):
    """Prove lifecycle boundary, wait-to-deadline, bounded grace period, and summary semantics."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)
        self.profile_file = self.root_path / "profile_25s.yaml"
        profile_content = {
            "profile": "test_boundary_25s",
            "seed": 42,
            "endpoint": "/work",
            "intensity": 3,
            "phases": [
                {"name": "p1", "type": "constant", "duration_seconds": 25.0, "target_rps": 1.0}
            ],
        }
        with open(self.profile_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(profile_content, f)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    async def test_25s_profile_final_arrival_at_24_4s_waits_until_deadline_no_invented_requests(self) -> None:
        """Requirement E1, E2, E3:

        1. 25s profile with final arrival at 24.4s does not finalize at 24.4s.
        2. It reaches approximately the 25s profile deadline.
        3. No request is invented after the final planned arrival.
        """
        clock = _VirtualClock(1000.0)
        generator = TrafficGenerator(
            experiment_id="exp_boundary_24_4s",
            profile_path=str(self.profile_file.relative_to(self.root_path)),
            project_root=self.root_path,
            timeout_seconds=5.0,
        )

        mock_schedule = [
            PlannedRequest(0, 0.0, "p1", "constant", 1.0, 3),
            PlannedRequest(1, 10.0, "p1", "constant", 1.0, 3),
            PlannedRequest(2, 24.4, "p1", "constant", 1.0, 3),
        ]

        with patch("loadgen.traffic_generator.time.monotonic", side_effect=clock.monotonic), \
             patch("loadgen.traffic_generator.asyncio.sleep", side_effect=clock.sleep), \
             patch("loadgen.traffic_generator.generate_arrival_schedule", return_value=mock_schedule), \
             patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:

            resp = MagicMock()
            resp.status_code = 200
            mock_get.return_value = resp

            await generator.run_async()

        # 1. Does NOT finalize at 24.4s
        self.assertNotEqual(generator.end_mono - generator.start_mono, 24.4)
        # 2. Reaches approximately 25.0s profile deadline
        self.assertAlmostEqual(generator.end_mono - generator.start_mono, 25.0, delta=0.01)
        # 3. No request is invented after the final planned arrival at 24.4s
        self.assertEqual(len(generator.completed_requests), 3)
        self.assertEqual(mock_get.call_count, 3)

        # Metadata checks
        with open(generator.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["requested_duration_seconds"], 25.0)
        self.assertEqual(meta["schedule_end_elapsed_seconds"], 24.4)
        self.assertAlmostEqual(meta["actual_runtime_seconds"], 25.0, places=2)
        self.assertAlmostEqual(meta["duration_seconds"], 25.0, places=2)
        self.assertAlmostEqual(meta["actual_duration_seconds"], 25.0, places=2)

    async def test_profile_with_no_request_close_to_end_lasts_full_profile_duration(self) -> None:
        """Requirement E4:

        A profile with no request close to the end still lasts for the full profile duration.
        """
        clock = _VirtualClock(1000.0)
        generator = TrafficGenerator(
            experiment_id="exp_idle_end",
            profile_path=str(self.profile_file.relative_to(self.root_path)),
            project_root=self.root_path,
            timeout_seconds=5.0,
        )

        # Only 2 requests at start; last 20 seconds are completely idle
        mock_schedule = [
            PlannedRequest(0, 0.0, "p1", "constant", 1.0, 3),
            PlannedRequest(1, 5.0, "p1", "constant", 1.0, 3),
        ]

        with patch("loadgen.traffic_generator.time.monotonic", side_effect=clock.monotonic), \
             patch("loadgen.traffic_generator.asyncio.sleep", side_effect=clock.sleep), \
             patch("loadgen.traffic_generator.generate_arrival_schedule", return_value=mock_schedule), \
             patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:

            resp = MagicMock()
            resp.status_code = 200
            mock_get.return_value = resp

            await generator.run_async()

        self.assertAlmostEqual(generator.end_mono - generator.start_mono, 25.0, delta=0.01)
        self.assertEqual(len(generator.completed_requests), 2)

        # Traffic summary covers 25 rows (0..24), with 0 offered and 0 sent for seconds 6..24
        with open(generator.summary_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        self.assertEqual(len(reader), 25)
        for sec in range(6, 25):
            self.assertEqual(int(reader[sec]["offered_requests"]), 0)
            self.assertEqual(int(reader[sec]["sent_requests"]), 0)

    async def test_inflight_request_at_deadline_bounded_grace_period(self) -> None:
        """Requirement E5:

        In-flight requests active at deadline are allowed a bounded grace period to complete.
        """
        clock = _VirtualClock(1000.0)
        generator = TrafficGenerator(
            experiment_id="exp_inflight_grace",
            profile_path=str(self.profile_file.relative_to(self.root_path)),
            project_root=self.root_path,
            timeout_seconds=5.0,
        )

        # Request dispatched at 24.5s
        mock_schedule = [
            PlannedRequest(0, 24.5, "p1", "constant", 1.0, 3),
        ]

        async def simulated_delayed_get(url: str, timeout: float = 5.0) -> MagicMock:
            # Simulate 1.0s server processing latency (completes at t = 25.5s)
            clock.current_time += 1.0
            await _real_sleep(0)
            resp = MagicMock()
            resp.status_code = 200
            return resp

        with patch("loadgen.traffic_generator.time.monotonic", side_effect=clock.monotonic), \
             patch("loadgen.traffic_generator.asyncio.sleep", side_effect=clock.sleep), \
             patch("loadgen.traffic_generator.generate_arrival_schedule", return_value=mock_schedule), \
             patch("httpx.AsyncClient.get", side_effect=simulated_delayed_get):

            await generator.run_async()

        # Generator stayed open until 25.0s, then allowed grace period for in-flight request
        actual_runtime = generator.end_mono - generator.start_mono
        self.assertGreaterEqual(actual_runtime, 25.5)
        self.assertEqual(len(generator.completed_requests), 1)
        self.assertEqual(generator.completed_requests[0].outcome, "success_2xx")

        with open(generator.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["requested_duration_seconds"], 25.0)
        self.assertEqual(meta["schedule_end_elapsed_seconds"], 24.5)
        self.assertGreaterEqual(meta["actual_runtime_seconds"], 25.5)

    async def test_inflight_request_hanging_bounded_by_timeout(self) -> None:
        """Requirement E5:

        Hanging in-flight request at deadline is bounded by request timeout and does not hang indefinitely.
        """
        short_profile_file = self.root_path / "profile_short.yaml"
        with open(short_profile_file, "w", encoding="utf-8") as f:
            yaml.safe_dump({
                "profile": "test_short",
                "phases": [{"name": "p1", "type": "constant", "duration_seconds": 0.1, "target_rps": 1.0}],
            }, f)

        generator = TrafficGenerator(
            experiment_id="exp_inflight_timeout",
            profile_path=str(short_profile_file.relative_to(self.root_path)),
            project_root=self.root_path,
            timeout_seconds=0.05,
        )

        mock_schedule = [
            PlannedRequest(0, 0.05, "p1", "constant", 1.0, 3),
        ]

        async def hung_get(url: str, timeout: float = 0.05) -> MagicMock:
            # Simulates target server stalling indefinitely
            await _real_sleep(5.0)
            resp = MagicMock()
            resp.status_code = 200
            return resp

        with patch("loadgen.traffic_generator.generate_arrival_schedule", return_value=mock_schedule), \
             patch("httpx.AsyncClient.get", side_effect=hung_get):

            # Runs in ~0.15s real time and terminates cleanly
            await generator.run_async()

        self.assertEqual(len(generator.completed_requests), 1)
        self.assertEqual(generator.completed_requests[0].outcome, "timeout")

        with open(generator.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["outcome_totals"]["timeout"], 1)

    async def test_near_boundary_in_bounds_request_dispatched_despite_extra_clock_read(self) -> None:
        """CORR-51 (Phase16L.1): reproduces the exact production halt mechanism.

        Schedule i2bval_r2_V1's final row (request_index 1920) is planned at
        179.999994s -- only 6 microseconds before the 180s duration boundary.
        The wait loop correctly decides (via `rem <= 0.0005`) to proceed with
        dispatch while wall-clock time is still measurably before the profile
        deadline. The pre-fix code then performed a SECOND, redundant
        `time.monotonic()` read immediately afterward purely to re-check the
        profile deadline; because that second read can itself advance past
        the deadline (ordinary event-loop/scheduling overhead is far larger
        than 6 microseconds), the already-approved request was silently
        dropped without being recorded anywhere. The fix (a single cached
        `profile_deadline_reached` flag set only inside the wait loop itself)
        removes that redundant read entirely, so this scenario is now
        dispatched regardless of how close the two clock reads would have
        been. This test drives the exact numeric shape of the real defect
        via a controlled monotonic() sequence rather than approximating it
        through many compounding sleep() iterations."""
        # Controlled, explicit sequence of time.monotonic() return values:
        #   1. start_mono capture
        #   2. first (and only, under the fix) wait-loop read: still clearly
        #      before the profile deadline, decides rem <= 0.0005 -> proceed
        #   3. onward: dispatch bookkeeping / lifecycle-tail reads. Under the
        #      PRE-FIX code, a second boundary re-check would have landed
        #      here and crossed the deadline; under the fix, no such
        #      dispatch-gating read occurs, so these values are unconstrained
        #      apart from letting the run finish.
        clock_values = [
            1000.0,        # 1: start_mono
            1024.9997,     # 2: wait-loop now (rem = 1024.999994 - 1024.9997 = 0.000294 <= 0.0005)
            1025.0002,     # 3: keep-lifecycle-open loop's rem check (already past 1025.0 -- harmless)
            1025.0003,     # 4: dispatch actual_dispatch_mono
            1025.0005,     # 5: dispatch completion_mono
        ]
        value_iter = iter(clock_values)

        def controlled_monotonic() -> float:
            try:
                return next(value_iter)
            except StopIteration:
                return clock_values[-1]

        generator = TrafficGenerator(
            experiment_id="exp_near_boundary_corr51",
            profile_path=str(self.profile_file.relative_to(self.root_path)),
            project_root=self.root_path,
            timeout_seconds=5.0,
        )

        # Mirrors the shape of the real defect: single request 6 microseconds
        # before the (here, 25s) profile duration boundary.
        near_boundary_offset = 25.0 - 0.000006
        mock_schedule = [
            PlannedRequest(0, near_boundary_offset, "p1", "constant", 1.0, 3),
        ]

        with patch("loadgen.traffic_generator.time.monotonic", side_effect=controlled_monotonic), \
             patch("loadgen.traffic_generator.asyncio.sleep", side_effect=lambda d: _real_sleep(0)), \
             patch("loadgen.traffic_generator.generate_arrival_schedule", return_value=mock_schedule), \
             patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:

            resp = MagicMock()
            resp.status_code = 200
            mock_get.return_value = resp

            await generator.run_async()

        self.assertEqual(len(generator.completed_requests), 1)
        self.assertEqual(generator.completed_requests[0].outcome, "success_2xx")
        self.assertEqual(mock_get.call_count, 1)

        with open(generator.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["total_offered"], 1)
        self.assertEqual(meta["total_sent"], 1)
        self.assertEqual(meta["outcome_totals"]["dropped_overload"], 0)

    async def test_request_genuinely_past_profile_deadline_while_waiting_is_still_not_dispatched(self) -> None:
        """CORR-51/CORR-52: if the profile deadline genuinely elapses (a
        real stall in the SCHEDULING clock itself, not merely a slow
        concurrent HTTP response -- see CORR-52/Phase16L.3, which
        established that an open-loop generator must not let one slow
        response block unrelated scheduled dispatches) while still waiting
        for a future in-bounds-by-timestamp request, that request must
        still be correctly abandoned. Uses a call-count-triggered stall
        injected via the wait loop's own sleep() (the mechanism that
        actually gates dispatch decisions), landing partway through
        request 0's own convergence (its 1.0s-away deadline needs ~20
        calls at the 0.05s cap; call #10 lands well within that). Request
        0 -- already admitted and mid-wait when the stall hits -- still
        dispatches (the same bounded latitude CORR-11 already grants
        in-flight completions past the nominal deadline); request 1's
        later, fresh upfront gate then correctly sees the exhausted
        window and aborts."""
        call_count = {"n": 0}
        clock = _VirtualClock(1000.0)
        generator = TrafficGenerator(
            experiment_id="exp_genuinely_late_corr51",
            profile_path=str(self.profile_file.relative_to(self.root_path)),
            project_root=self.root_path,
            timeout_seconds=5.0,
        )

        mock_schedule = [
            PlannedRequest(0, 1.0, "p1", "constant", 1.0, 3),
            PlannedRequest(1, 2.0, "p1", "constant", 1.0, 3),
        ]

        async def stalling_sleep(delay: float) -> None:
            call_count["n"] += 1
            clock.current_time += delay
            if call_count["n"] == 10:
                clock.current_time += 500.0  # genuine multi-minute-scale stall
            await _real_sleep(0)

        with patch("loadgen.traffic_generator.time.monotonic", side_effect=clock.monotonic), \
             patch("loadgen.traffic_generator.asyncio.sleep", side_effect=stalling_sleep), \
             patch("loadgen.traffic_generator.generate_arrival_schedule", return_value=mock_schedule), \
             patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            resp = MagicMock()
            resp.status_code = 200
            mock_get.return_value = resp

            await generator.run_async()

        # Only the first request was dispatched; the second was correctly
        # abandoned because the profile deadline was genuinely reached first.
        self.assertEqual(len(generator.completed_requests), 1)
        self.assertEqual(generator.completed_requests[0].request_index, 0)

        with open(generator.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["total_offered"], 2)
        self.assertEqual(meta["total_sent"], 1)

    def test_traffic_summary_covers_full_intended_profile_window(self) -> None:
        """Requirement E6:

        Traffic summary covers the complete 0-25s workload window rather than stopping at final arrival.
        """
        generator = TrafficGenerator(
            experiment_id="exp_summary_coverage",
            profile_path=str(self.profile_file.relative_to(self.root_path)),
            project_root=self.root_path,
        )
        generator.setup_directory_and_schedule()

        # Completed request only in second 2
        generator.completed_requests = [
            CompletedRequest(0, 2.1, 2.102, 0.002, 2.15, 3, "p1", 1.0, "success_2xx", 200, 0.048),
        ]

        generator._write_traffic_summary()

        self.assertTrue(generator.summary_path.exists())
        with open(generator.summary_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        # Must have exactly 25 1-second interval rows for 0..24
        self.assertEqual(len(reader), 25)
        self.assertEqual(int(reader[0]["elapsed_seconds"]), 0)
        self.assertEqual(int(reader[24]["elapsed_seconds"]), 24)
        self.assertEqual(int(reader[2]["sent_requests"]), 1)
        self.assertEqual(int(reader[24]["sent_requests"]), 0)

    def test_planned_arrival_schedule_unimpaired_and_reproducible(self) -> None:
        """Requirement E7:

        Planned arrival schedule itself remains unchanged, bounded, and reproducible.
        """
        profile = parse_traffic_profile(self.profile_file)

        # Deterministic reproducibility
        s_det1 = generate_arrival_schedule(profile, seed=42, arrival_process="deterministic")
        s_det2 = generate_arrival_schedule(profile, seed=42, arrival_process="deterministic")
        self.assertEqual(len(s_det1), len(s_det2))
        for r1, r2 in zip(s_det1, s_det2):
            self.assertEqual(r1.planned_elapsed_seconds, r2.planned_elapsed_seconds)
            self.assertLess(r1.planned_elapsed_seconds, profile.total_duration)

        # Poisson reproducibility with same seed
        s_pois1 = generate_arrival_schedule(profile, seed=42, arrival_process="poisson")
        s_pois2 = generate_arrival_schedule(profile, seed=42, arrival_process="poisson")
        self.assertEqual(len(s_pois1), len(s_pois2))
        for r1, r2 in zip(s_pois1, s_pois2):
            self.assertEqual(r1.planned_elapsed_seconds, r2.planned_elapsed_seconds)
            self.assertLess(r1.planned_elapsed_seconds, profile.total_duration)

        # Strictly strictly bounded by profile duration
        self.assertLess(s_det1[-1].planned_elapsed_seconds, 25.0)
        self.assertLess(s_pois1[-1].planned_elapsed_seconds, 25.0)


class _OvershootClock(_VirtualClock):
    """Virtual clock that adds a fixed overshoot to every sleep(), modeling
    ordinary real-world asyncio.sleep()/OS scheduling imprecision (CORR-52,
    Phase16L.3) -- empirically observed in live production runs to range
    from ~99 microseconds to ~2.3 milliseconds. `monotonic()` itself adds no
    extra overhead, isolating sleep-overshoot as the sole injected variable."""

    def __init__(self, start_time: float = 1000.0, overshoot_seconds: float = 0.0001) -> None:
        super().__init__(start_time)
        self.overshoot_seconds = overshoot_seconds

    async def sleep(self, delay: float) -> None:
        self.current_time += delay + self.overshoot_seconds
        await _real_sleep(0)


class TestCorr52FinalArrivalDeadlineSemantics(unittest.IsolatedAsyncioTestCase):
    """CORR-52 (Phase16L.3): a schedule row that passes both fixed-value
    membership checks (planned_elapsed_seconds < total_duration and
    req_deadline_mono < profile_deadline_mono) must be dispatched regardless
    of ordinary asyncio.sleep()/OS scheduling overshoot encountered while
    converging on its own deadline -- see docs/experimental_design_corrections.md
    CORR-11/CORR-12 for the pre-existing schedule-domain-membership contract
    this restores. CORR-51 (Phase16L.1) fixed a narrower, related race (a
    redundant second clock read after loop exit); CORR-52 fixes the dominant
    mechanism for razor-thin (sub-millisecond) schedule margins: a recurring
    internal profile-deadline recheck inside the same convergence loop."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)
        self.profile_file = self.root_path / "profile_180s.yaml"
        with open(self.profile_file, "w", encoding="utf-8") as f:
            yaml.safe_dump({
                "profile": "test_corr52_180s", "seed": 42, "endpoint": "/work", "intensity": 3,
                "phases": [{"name": "p1", "type": "constant", "duration_seconds": 180.0, "target_rps": 10.666667}],
            }, f)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    async def _run_two_request_schedule(self, overshoot_seconds: float):
        clock = _OvershootClock(1000.0, overshoot_seconds=overshoot_seconds)
        generator = TrafficGenerator(
            experiment_id=f"exp_corr52_{int(overshoot_seconds * 1_000_000)}us",
            profile_path=str(self.profile_file.relative_to(self.root_path)),
            project_root=self.root_path, timeout_seconds=5.0,
        )
        mock_schedule = [
            PlannedRequest(1919, 179.906244, "p1", "constant", 10.6667, 3),
            PlannedRequest(1920, 179.999994, "p1", "constant", 10.6667, 3),
        ]
        with patch("loadgen.traffic_generator.time.monotonic", side_effect=clock.monotonic), \
             patch("loadgen.traffic_generator.asyncio.sleep", side_effect=clock.sleep), \
             patch("loadgen.traffic_generator.generate_arrival_schedule", return_value=mock_schedule), \
             patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            resp = MagicMock()
            resp.status_code = 200
            mock_get.return_value = resp
            await generator.run_async()
        return generator, mock_get

    async def test_1_final_row_dispatched_under_realistic_and_larger_overshoot(self) -> None:
        """Test 1 (Section 14): the exact real i2bval_r2_V1 boundary shape
        (6-microsecond margin) dispatches both requests -- offered==sent --
        across a range of injected overshoots from 0 up through 10ms, far
        exceeding the empirically observed real-world range."""
        for overshoot_us in (0, 25, 100, 250, 1000, 2000, 5000, 10000):
            with self.subTest(overshoot_us=overshoot_us):
                generator, mock_get = await self._run_two_request_schedule(overshoot_us / 1_000_000.0)
                self.assertEqual(len(generator.completed_requests), 2)
                with open(generator.metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.assertEqual(meta["total_offered"], 2)
                self.assertEqual(meta["total_sent"], 2)

    async def test_2_final_dispatched_request_matches_frozen_schedule_row_exactly(self) -> None:
        """Test 2: the dispatched final request is exactly request_index=1920,
        planned_elapsed_seconds=179.999994 -- not an invented substitute."""
        generator, _ = await self._run_two_request_schedule(0.001)
        final = [r for r in generator.completed_requests if r.request_index == 1920]
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0].planned_elapsed_seconds, 179.999994)

    async def test_3_out_of_window_entry_still_excluded(self) -> None:
        """Test 3: a synthetic entry with planned_elapsed_seconds >= total_duration
        remains excluded, regardless of the CORR-52 change."""
        clock = _OvershootClock(1000.0, overshoot_seconds=0.001)
        generator = TrafficGenerator(
            experiment_id="exp_corr52_out_of_window",
            profile_path=str(self.profile_file.relative_to(self.root_path)),
            project_root=self.root_path, timeout_seconds=5.0,
        )
        mock_schedule = [
            PlannedRequest(0, 179.0, "p1", "constant", 10.6667, 3),
            PlannedRequest(1, 180.0, "p1", "constant", 10.6667, 3),  # exactly at duration -- excluded
            PlannedRequest(2, 180.5, "p1", "constant", 10.6667, 3),  # past duration -- excluded
        ]
        with patch("loadgen.traffic_generator.time.monotonic", side_effect=clock.monotonic), \
             patch("loadgen.traffic_generator.asyncio.sleep", side_effect=clock.sleep), \
             patch("loadgen.traffic_generator.generate_arrival_schedule", return_value=mock_schedule), \
             patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            resp = MagicMock()
            resp.status_code = 200
            mock_get.return_value = resp
            await generator.run_async()
        self.assertEqual(len(generator.completed_requests), 1)
        self.assertEqual(generator.completed_requests[0].request_index, 0)

    async def test_4_explicit_stop_still_authoritative(self) -> None:
        """Test 4: stop_requested still prevents further dispatch even
        though the recurring profile-deadline recheck was removed."""
        clock = _OvershootClock(1000.0, overshoot_seconds=0.0001)
        generator = TrafficGenerator(
            experiment_id="exp_corr52_stop_requested",
            profile_path=str(self.profile_file.relative_to(self.root_path)),
            project_root=self.root_path, timeout_seconds=5.0,
        )
        mock_schedule = [
            PlannedRequest(0, 1.0, "p1", "constant", 10.6667, 3),
            PlannedRequest(1, 2.0, "p1", "constant", 10.6667, 3),
        ]

        async def stop_after_first_sleep(delay: float) -> None:
            generator.stop_requested = True
            clock.current_time += delay + clock.overshoot_seconds
            await _real_sleep(0)

        with patch("loadgen.traffic_generator.time.monotonic", side_effect=clock.monotonic), \
             patch("loadgen.traffic_generator.asyncio.sleep", side_effect=stop_after_first_sleep), \
             patch("loadgen.traffic_generator.generate_arrival_schedule", return_value=mock_schedule), \
             patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            resp = MagicMock()
            resp.status_code = 200
            mock_get.return_value = resp
            await generator.run_async()
        self.assertEqual(len(generator.completed_requests), 0)

    async def test_5_request_timeout_unchanged(self) -> None:
        """Test 5: the 5-second request timeout behavior is unaffected."""
        clock = _OvershootClock(1000.0, overshoot_seconds=0.0001)
        generator = TrafficGenerator(
            experiment_id="exp_corr52_timeout_unchanged",
            profile_path=str(self.profile_file.relative_to(self.root_path)),
            project_root=self.root_path, timeout_seconds=5.0,
        )
        mock_schedule = [PlannedRequest(0, 179.999994, "p1", "constant", 10.6667, 3)]

        async def hanging_get(url: str, timeout: float = 5.0):
            raise __import__("httpx").TimeoutException("simulated timeout")

        with patch("loadgen.traffic_generator.time.monotonic", side_effect=clock.monotonic), \
             patch("loadgen.traffic_generator.asyncio.sleep", side_effect=clock.sleep), \
             patch("loadgen.traffic_generator.generate_arrival_schedule", return_value=mock_schedule), \
             patch("httpx.AsyncClient.get", side_effect=hanging_get):
            await generator.run_async()
        self.assertEqual(len(generator.completed_requests), 1)
        self.assertEqual(generator.completed_requests[0].outcome, "timeout")

    async def test_6_max_in_flight_unchanged(self) -> None:
        """Test 6: max_in_flight remains a configurable, enforced ceiling;
        CORR-52 does not touch the concurrency-check branch at all. Uses a
        short profile (mirroring this file's existing
        test_inflight_request_hanging_bounded_by_timeout pattern) so the
        first request's slow-but-bounded response keeps it in-flight long
        enough for the second, near-simultaneous request to be dropped for
        overload rather than needing an unresolvable mock."""
        short_profile_file = self.root_path / "profile_short_corr52.yaml"
        with open(short_profile_file, "w", encoding="utf-8") as f:
            yaml.safe_dump({
                "profile": "test_corr52_short",
                "phases": [{"name": "p1", "type": "constant", "duration_seconds": 0.2, "target_rps": 10.0}],
            }, f)

        generator = TrafficGenerator(
            experiment_id="exp_corr52_max_in_flight",
            profile_path=str(short_profile_file.relative_to(self.root_path)),
            project_root=self.root_path, timeout_seconds=0.15, max_in_flight=1,
        )
        mock_schedule = [
            PlannedRequest(0, 0.05, "p1", "constant", 10.0, 3),
            PlannedRequest(1, 0.0501, "p1", "constant", 10.0, 3),
        ]

        async def slow_get(url: str, timeout: float = 0.15):
            await _real_sleep(0.1)  # still in-flight when request 1 arrives, resolves before profile end
            resp = MagicMock()
            resp.status_code = 200
            return resp

        with patch("loadgen.traffic_generator.generate_arrival_schedule", return_value=mock_schedule), \
             patch("httpx.AsyncClient.get", side_effect=slow_get):
            await generator.run_async()

        dropped = [r for r in generator.completed_requests if r.outcome == "dropped_overload"]
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0].request_index, 1)
        self.assertEqual(generator.max_in_flight, 1)

    async def test_7_unaffected_schedules_i2bval_r1_v1_and_r6_v1_behaviorally_equivalent(self) -> None:
        """Test 7: representative unaffected frozen schedules (comfortable
        56-187ms boundary margins) dispatch identically regardless of
        CORR-52, proving no behavioral change for the three already-official
        Phase16L slots."""
        project_root = Path("/Users/rakeshdandamudi/Capstone/rl-k8s-autoscaler")
        for cell in ("i2bval_r1_V1", "i2bval_r6_V1"):
            with self.subTest(cell=cell):
                schedule_path = (
                    project_root
                    / "configs/traffic_profiles/rl_dynamics_calibration_iteration2b/final_unseen"
                    / f"{cell}_schedule.csv"
                )
                real_schedule = load_planned_schedule(schedule_path)
                clock = _OvershootClock(1000.0, overshoot_seconds=0.0023)  # max observed live lag
                tmp = tempfile.TemporaryDirectory()
                root = Path(tmp.name)
                profile_file = root / "profile.yaml"
                with open(profile_file, "w", encoding="utf-8") as f:
                    yaml.safe_dump({
                        "profile": cell, "seed": 42, "endpoint": "/work", "intensity": 3,
                        "phases": [{"name": "steady_state", "type": "constant", "duration_seconds": 180.0,
                                    "target_rps": float(real_schedule[0].target_rps)}],
                    }, f)
                generator = TrafficGenerator(
                    experiment_id=f"exp_corr52_unaffected_{cell}",
                    profile_path=str(profile_file.relative_to(root)),
                    project_root=root, timeout_seconds=5.0,
                )
                with patch("loadgen.traffic_generator.time.monotonic", side_effect=clock.monotonic), \
                     patch("loadgen.traffic_generator.asyncio.sleep", side_effect=clock.sleep), \
                     patch("loadgen.traffic_generator.generate_arrival_schedule", return_value=real_schedule), \
                     patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
                    resp = MagicMock()
                    resp.status_code = 200
                    mock_get.return_value = resp
                    await generator.run_async()
                self.assertEqual(len(generator.completed_requests), len(real_schedule))
                tmp.cleanup()

    async def test_8_three_near_boundary_schedule_families_all_fully_dispatch(self) -> None:
        """Test 8: all three near-boundary schedule families
        (i2bval_r1_V2, i2bval_r2_V1, i2bval_r4_V1) fully dispatch offline
        under realistic injected overshoot, with zero schedule-file changes."""
        project_root = Path("/Users/rakeshdandamudi/Capstone/rl-k8s-autoscaler")
        for cell in ("i2bval_r1_V2", "i2bval_r2_V1", "i2bval_r4_V1"):
            with self.subTest(cell=cell):
                schedule_path = (
                    project_root
                    / "configs/traffic_profiles/rl_dynamics_calibration_iteration2b/final_unseen"
                    / f"{cell}_schedule.csv"
                )
                real_schedule = load_planned_schedule(schedule_path)
                clock = _OvershootClock(1000.0, overshoot_seconds=0.0023)
                tmp = tempfile.TemporaryDirectory()
                root = Path(tmp.name)
                profile_file = root / "profile.yaml"
                with open(profile_file, "w", encoding="utf-8") as f:
                    yaml.safe_dump({
                        "profile": cell, "seed": 42, "endpoint": "/work", "intensity": 3,
                        "phases": [{"name": "steady_state", "type": "constant", "duration_seconds": 180.0,
                                    "target_rps": float(real_schedule[0].target_rps)}],
                    }, f)
                generator = TrafficGenerator(
                    experiment_id=f"exp_corr52_nearboundary_{cell}",
                    profile_path=str(profile_file.relative_to(root)),
                    project_root=root, timeout_seconds=5.0,
                )
                with patch("loadgen.traffic_generator.time.monotonic", side_effect=clock.monotonic), \
                     patch("loadgen.traffic_generator.asyncio.sleep", side_effect=clock.sleep), \
                     patch("loadgen.traffic_generator.generate_arrival_schedule", return_value=real_schedule), \
                     patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
                    resp = MagicMock()
                    resp.status_code = 200
                    mock_get.return_value = resp
                    await generator.run_async()
                self.assertEqual(len(generator.completed_requests), len(real_schedule))
                tmp.cleanup()

    async def test_9_incomplete_replay_still_fails_structural_classifier(self) -> None:
        """Test 9: a genuinely incomplete replay -- caused by a real stall in
        the SCHEDULING clock itself (the mechanism the wait loop relies on),
        not merely a slow concurrent HTTP response, which an open-loop
        generator must not let block unrelated scheduled dispatches -- still
        produces offered!=sent, so check_offered_equals_sent() in
        dynamics_calibration_lib.py (never modified) continues to fail
        closed. No weakening of V8 validity. Shares its stall model with
        test_10 (see that test for the full no-burst rationale): the stall
        is injected via one large sleep() call while converging toward
        request 1's own deadline, so request 1 (already admitted, mid-wait
        when the stall hit) still dispatches, but request 2's later,
        fresh upfront gate correctly sees the exhausted window and aborts
        -- yielding offered=3, sent=2."""
        # Each request's deadline is exactly 20 x 0.05s from the previous one
        # (1.0s apart, 0.05s sleep cap), so request 0 converges in exactly 20
        # sleep calls; call #25 deterministically lands partway through
        # request 1's own convergence (calls 21-40), regardless of Python/
        # asyncio version -- not tied to any observed production timing.
        call_count = {"n": 0}
        clock = _VirtualClock(1000.0)

        async def stalling_sleep(delay: float) -> None:
            call_count["n"] += 1
            clock.current_time += delay
            if call_count["n"] == 25:
                clock.current_time += 500.0  # genuine multi-minute-scale stall
            await _real_sleep(0)

        generator = TrafficGenerator(
            experiment_id="exp_corr52_incomplete_replay_still_fails",
            profile_path=str(self.profile_file.relative_to(self.root_path)),
            project_root=self.root_path, timeout_seconds=5.0,
        )
        mock_schedule = [
            PlannedRequest(0, 1.0, "p1", "constant", 10.6667, 3),
            PlannedRequest(1, 2.0, "p1", "constant", 10.6667, 3),
            PlannedRequest(2, 3.0, "p1", "constant", 10.6667, 3),
        ]

        with patch("loadgen.traffic_generator.time.monotonic", side_effect=clock.monotonic), \
             patch("loadgen.traffic_generator.asyncio.sleep", side_effect=stalling_sleep), \
             patch("loadgen.traffic_generator.generate_arrival_schedule", return_value=mock_schedule), \
             patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            resp = MagicMock()
            resp.status_code = 200
            mock_get.return_value = resp
            await generator.run_async()

        with open(generator.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertNotEqual(meta["total_offered"], meta["total_sent"])
        self.assertEqual(meta["total_offered"], 3)
        self.assertEqual(meta["total_sent"], 2)

        import sys as _sys
        _sys.path.insert(0, str(Path("/Users/rakeshdandamudi/Capstone/rl-k8s-autoscaler")))
        from scripts.dynamics_calibration_lib import check_offered_equals_sent
        ok, _ = check_offered_equals_sent(meta)
        self.assertFalse(ok)

    async def test_10_genuine_stall_produces_at_most_one_straggler_never_a_burst(self) -> None:
        """Test 10 (Section 13/14 mandatory stall check): a genuine stall in
        the scheduling clock itself, occurring while request 1 is already
        admitted and mid-convergence, may let that ONE already-admitted
        request dispatch somewhat late (the same bounded latitude CORR-11
        already grants in-flight HTTP completions past the nominal
        deadline) -- but every LATER request must still see a fresh upfront
        gate reflecting the now-exhausted window and be abandoned. This is
        not a burst: exactly one request is affected, never more, however
        many additional requests remain in the frozen schedule after the
        stall."""
        # Same deterministic call-count trigger as test_9: call #25 lands
        # partway through request 1's own convergence (request 0 takes
        # exactly 20 calls to converge from a 1.0s-away deadline).
        call_count = {"n": 0}
        clock = _VirtualClock(1000.0)

        async def stalling_sleep(delay: float) -> None:
            call_count["n"] += 1
            clock.current_time += delay
            if call_count["n"] == 25:
                clock.current_time += 500.0  # genuine multi-minute-scale stall
            await _real_sleep(0)

        generator = TrafficGenerator(
            experiment_id="exp_corr52_no_catch_up_burst",
            profile_path=str(self.profile_file.relative_to(self.root_path)),
            project_root=self.root_path, timeout_seconds=5.0,
        )
        # Five requests well within the schedule -- proving the abandonment
        # after the stall applies to ALL remaining requests, not just the
        # next one, i.e. genuinely no burst of any size.
        mock_schedule = [
            PlannedRequest(0, 1.0, "p1", "constant", 10.6667, 3),
            PlannedRequest(1, 2.0, "p1", "constant", 10.6667, 3),
            PlannedRequest(2, 3.0, "p1", "constant", 10.6667, 3),
            PlannedRequest(3, 4.0, "p1", "constant", 10.6667, 3),
            PlannedRequest(4, 5.0, "p1", "constant", 10.6667, 3),
        ]

        with patch("loadgen.traffic_generator.time.monotonic", side_effect=clock.monotonic), \
             patch("loadgen.traffic_generator.asyncio.sleep", side_effect=stalling_sleep), \
             patch("loadgen.traffic_generator.generate_arrival_schedule", return_value=mock_schedule), \
             patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            resp = MagicMock()
            resp.status_code = 200
            mock_get.return_value = resp
            await generator.run_async()

        # Requests 0 and 1 dispatch (1 was already admitted when the stall
        # hit); requests 2, 3, and 4 are all correctly abandoned -- not a
        # burst of any size, regardless of how many remained in the schedule.
        dispatched_indices = sorted(r.request_index for r in generator.completed_requests)
        self.assertEqual(dispatched_indices, [0, 1])


if __name__ == "__main__":
    unittest.main()
