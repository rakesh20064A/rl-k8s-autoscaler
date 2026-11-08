"""tests/unit/test_calibrate_epsilon_v2.py

V2 Phase 5B PART B: offline tests for scripts/calibrate_epsilon_v2.py.
Uses fake models/probability extractors for the pure-computation tests
(no SB3/torch dependency), plus one proof run against the REAL completed
sessions and the latest real checkpoint (session_001, the latest verified
as of this phase) writing to a TEMP path -- explicitly NOT the official
calibration (session_047 does not exist yet; training is live).
"""

from __future__ import annotations

import json

import pytest

import scripts.calibrate_epsilon_v2 as calib


class _FakeModel:
    pass


def _fake_extractor_factory(margins):
    """Returns a probability extractor that, called once per observation
    in order, yields a 3-vector whose top1-top2 margin is exactly the
    corresponding value in `margins` (deterministic, order-preserving)."""
    calls = {"i": 0}

    def extractor(model, obs):
        m = margins[calls["i"]]
        calls["i"] += 1
        # top1 = (1+m)/2, top2 = (1-m)/2, third = 0 -- margin = top1-top2 = m.
        return [0.0, (1.0 - m) / 2.0, (1.0 + m) / 2.0]

    return extractor


class TestPercentileHelper:
    def test_median_of_odd_list(self):
        assert calib._percentile([1.0, 2.0, 3.0], 50.0) == 2.0

    def test_min_at_0th_percentile(self):
        assert calib._percentile([5.0, 1.0, 3.0], 0.0) == 1.0

    def test_max_at_100th_percentile(self):
        assert calib._percentile([5.0, 1.0, 3.0], 100.0) == 5.0

    def test_single_value(self):
        assert calib._percentile([7.0], 10.0) == 7.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            calib._percentile([], 10.0)


class TestClampHelper:
    def test_within_bounds_unchanged(self):
        assert calib._clamp(0.02, floor=0.0, cap=0.05) == 0.02

    def test_above_cap_clamped_to_cap(self):
        assert calib._clamp(0.5, floor=0.0, cap=0.05) == 0.05

    def test_below_floor_clamped_to_floor(self):
        assert calib._clamp(-0.1, floor=0.0, cap=0.05) == 0.0


class TestComputeEpsilon:
    def test_epsilon_is_10th_percentile_when_within_cap_and_floor(self):
        margins = [0.01 * i for i in range(1, 101)]  # 0.01 .. 1.00
        extractor = _fake_extractor_factory(margins)
        observations = [[0.0] * 11 for _ in margins]
        result = calib.compute_epsilon(
            _FakeModel(), observations, percentile=10.0, cap=0.05, floor=0.0, probability_extractor=extractor,
        )
        # linear-interpolation percentile of [0.01..1.00] at p=10: rank=0.1*99=9.9 -> between sorted[9]=0.10 and sorted[10]=0.11.
        assert result["raw_percentile_value"] == pytest.approx(0.109, abs=1e-6)
        # capped: raw 0.109 > cap 0.05 -> epsilon clamped to the cap.
        assert result["epsilon"] == 0.05

    def test_epsilon_capped_at_005(self):
        margins = [0.5] * 20  # 10th percentile would be 0.5, way above the cap
        extractor = _fake_extractor_factory(margins)
        observations = [[0.0] * 11 for _ in margins]
        result = calib.compute_epsilon(_FakeModel(), observations, cap=0.05, floor=0.0, probability_extractor=extractor)
        assert result["epsilon"] == 0.05

    def test_epsilon_never_below_floor(self):
        """`compute_margin_stats().margin` is always >= 0 by construction
        (top1 >= top2 after sorting), so the floor clamp is a defensive
        bound rather than one real margins ever trigger -- the clamp
        arithmetic itself is covered directly by TestClampHelper."""
        margins = [0.0] * 20
        extractor = _fake_extractor_factory(margins)
        observations = [[0.0] * 11 for _ in margins]
        result = calib.compute_epsilon(_FakeModel(), observations, cap=0.05, floor=0.0, probability_extractor=extractor)
        assert result["epsilon"] >= 0.0

    def test_epsilon_within_bounds_uses_raw_value(self):
        margins = sorted([0.02] * 9 + [0.1])  # 10th percentile of 10 values lands at 0.02
        extractor = _fake_extractor_factory(margins)
        observations = [[0.0] * 11 for _ in margins]
        result = calib.compute_epsilon(_FakeModel(), observations, cap=0.05, floor=0.0, probability_extractor=extractor)
        assert result["epsilon"] == pytest.approx(result["raw_percentile_value"])
        assert 0.0 < result["epsilon"] < 0.05

    def test_tie_fraction_counts_margins_at_or_below_epsilon(self):
        margins = [0.0, 0.0, 0.0, 0.0, 1.0]  # 4/5 at or below any epsilon >= 0
        extractor = _fake_extractor_factory(margins)
        observations = [[0.0] * 11 for _ in margins]
        result = calib.compute_epsilon(_FakeModel(), observations, percentile=10.0, cap=0.05, floor=0.0, probability_extractor=extractor)
        assert result["tie_fraction_at_epsilon"] == pytest.approx(0.8)

    def test_empty_observations_raises(self):
        with pytest.raises(ValueError):
            calib.compute_epsilon(_FakeModel(), [], probability_extractor=_fake_extractor_factory([]))


class TestFindCompletedSessions:
    def test_only_sessions_with_both_checkpoint_and_metadata_count(self, tmp_path):
        complete = tmp_path / "session_000"
        complete.mkdir()
        (complete / "ppo_model.zip").write_bytes(b"x")
        (complete / "metadata.json").write_text("{}")

        incomplete = tmp_path / "session_001"
        incomplete.mkdir()
        (incomplete / "ppo_model.zip").write_bytes(b"x")  # metadata.json missing -> in progress

        result = calib.find_completed_sessions(tmp_path)
        assert [p.name for p in result] == ["session_000"]

    def test_missing_directory_returns_empty(self, tmp_path):
        assert calib.find_completed_sessions(tmp_path / "does_not_exist") == []


class TestLoadLastNSessionsObservations:
    def _make_session(self, root, name, rows):
        d = root / name
        d.mkdir()
        (d / "ppo_model.zip").write_bytes(b"x")
        (d / "metadata.json").write_text("{}")
        with open(d / "step_telemetry.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return d

    def test_uses_only_the_last_n_completed_sessions(self, tmp_path):
        for i in range(7):
            self._make_session(tmp_path, f"session_{i:03d}", [{"normalized_observation": [float(i)] * 11}])
        observations, used = calib.load_last_n_sessions_observations(tmp_path, n=5)
        assert used == [f"session_{i:03d}" for i in range(2, 7)]
        assert len(observations) == 5

    def test_fewer_than_n_available_uses_all_of_them(self, tmp_path):
        for i in range(2):
            self._make_session(tmp_path, f"session_{i:03d}", [{"normalized_observation": [0.0] * 11}])
        observations, used = calib.load_last_n_sessions_observations(tmp_path, n=5)
        assert used == ["session_000", "session_001"]
        assert len(observations) == 2


class TestRunCalibrationEndToEnd:
    def test_record_fields_and_official_flag_default_false(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        for i in range(3):
            d = sessions_dir / f"session_{i:03d}"
            d.mkdir(parents=True)
            (d / "ppo_model.zip").write_bytes(b"x")
            (d / "metadata.json").write_text("{}")
            with open(d / "step_telemetry.jsonl", "w", encoding="utf-8") as f:
                for m in [0.01, 0.02, 0.03]:
                    f.write(json.dumps({"normalized_observation": [m] * 11}) + "\n")

        model_path = tmp_path / "ppo_model.zip"
        model_path.write_bytes(b"fake model bytes")

        margins = [0.01] * 9
        extractor = _fake_extractor_factory(margins)

        record = calib.run_calibration(
            model_path, sessions_dir, n_sessions=5, official=False,
            model_loader=lambda p: _FakeModel(), probability_extractor=extractor,
        )
        assert record["official"] is False
        assert record["model_sha256"] == __import__("hashlib").sha256(model_path.read_bytes()).hexdigest()
        assert record["sessions_used_count"] == 3
        assert record["fewer_than_requested_sessions_disclosed"] is True
        assert 0.0 <= record["epsilon"] <= 0.05

    def test_model_load_failure_raises(self, tmp_path):
        model_path = tmp_path / "missing.zip"
        with pytest.raises(RuntimeError):
            calib.run_calibration(model_path, tmp_path / "sessions", model_loader=lambda p: None)


class TestProofRunAgainstRealCompletedSessions:
    """V2 Phase 5B: run the real script once against the actual completed
    V2 final_min2 sessions and the latest real checkpoint, writing to a
    TEMP path -- proves the end-to-end wiring works. Explicitly NOT the
    official calibration (training is live; session_047 does not exist
    yet). Read-only against persistent-data/rl_training_v2/."""

    def test_proof_run_against_real_sessions_and_latest_checkpoint(self, tmp_path):
        real_sessions_dir = calib.DEFAULT_SESSIONS_DIR
        completed = calib.find_completed_sessions(real_sessions_dir)
        if not completed:
            pytest.skip("no completed V2 final_min2 sessions exist yet")

        latest_checkpoint = completed[-1] / "ppo_model.zip"
        output_path = tmp_path / "EPSILON_CALIBRATION_PROOF.json"

        try:
            from stable_baselines3 import PPO  # noqa: F401
        except ImportError:
            pytest.skip("stable_baselines3 not importable in this environment")

        record = calib.run_calibration(
            latest_checkpoint, real_sessions_dir, n_sessions=5, official=False,
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        assert output_path.exists()
        assert record["official"] is False
        assert record["sessions_used_count"] >= 1
        assert record["sessions_used_count"] <= 5
        assert 0.0 <= record["epsilon"] <= 0.05
