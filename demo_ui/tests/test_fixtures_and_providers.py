from __future__ import annotations

from demo_ui.fixtures import build_experiment_history, build_fixture_pair, schedule_hash
from demo_ui.models import Controller, TrafficProfile
from demo_ui.providers import FixtureExperimentController


class TestFixtureDeterminism:
    def test_schedule_hash_is_deterministic(self):
        a = schedule_hash(TrafficProfile.RAMP, 42, 300.0)
        b = schedule_hash(TrafficProfile.RAMP, 42, 300.0)
        assert a == b

    def test_schedule_hash_differs_by_profile(self):
        a = schedule_hash(TrafficProfile.RAMP, 42, 300.0)
        b = schedule_hash(TrafficProfile.BURST, 42, 300.0)
        assert a != b

    def test_fixture_pair_shares_schedule_hash(self):
        hpa, ppo = build_fixture_pair(TrafficProfile.CONSTANT)
        assert hpa.schedule_hash == ppo.schedule_hash
        assert hpa.is_demo_data is True
        assert ppo.is_demo_data is True

    def test_fixture_pair_has_parallel_time_series(self):
        hpa, ppo = build_fixture_pair(TrafficProfile.STEP)
        assert len(hpa.timestamps) == len(hpa.replicas) == len(hpa.cpu_pct)
        assert len(ppo.timestamps) == len(ppo.replicas) == len(ppo.cpu_pct)

    def test_experiment_history_covers_all_profiles(self):
        records = build_experiment_history()
        assert len(records) == len(list(TrafficProfile))
        assert all(r.valid for r in records)


class TestFixtureExperimentController:
    def test_start_stop_reset_are_recorded_not_executed(self):
        controller = FixtureExperimentController()
        controller.start(Controller.PPO, TrafficProfile.BURST)
        controller.stop()
        controller.reset()
        assert controller.calls == [
            ("start", Controller.PPO, TrafficProfile.BURST),
            ("stop",),
            ("reset",),
        ]
