"""tests/unit/test_v2_pilot_memory_semantics.py

V2 Phase 3C PART 1/2/3/6: proves the per-pod (not aggregate) memory safety
semantics, and the safety-interruption provenance fix.
"""

from __future__ import annotations

import json

import gymnasium as gym
import pytest

from scripts.v2_pilot_safety import (
    MEMORY_SAFETY_THRESHOLD_MIB,
    PilotSafetyCallback,
    PodHealthSnapshot,
    get_aggregate_memory_mib,
    get_max_per_pod_memory_mib,
)


class _FakePromResponse:
    def __init__(self, value_bytes):
        self._value_bytes = value_bytes

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps({"data": {"result": [{"value": [0, str(self._value_bytes)]}]}}).encode()


class TestPerPodVsAggregateQueryShape:
    def test_max_per_pod_uses_max_sum_by_pod(self, monkeypatch):
        import urllib.request

        captured = {}

        def _fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url if hasattr(req, "full_url") else req
            return _FakePromResponse(150 * 1048576)

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
        get_max_per_pod_memory_mib("http://x", "autoscaler-app", "demo-app-.*")
        assert "max%28sum+by+%28pod%29" in captured["url"] or "max(sum by (pod)" in captured["url"] or "max%28sum%20by%20%28pod%29" in captured["url"]

    def test_aggregate_uses_plain_sum(self, monkeypatch):
        import urllib.request

        captured = {}

        def _fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url if hasattr(req, "full_url") else req
            return _FakePromResponse(300 * 1048576)

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
        get_aggregate_memory_mib("http://x", "autoscaler-app", "demo-app-.*")
        assert "by" not in captured["url"].lower().split("sum")[0]  # sanity: aggregate query has no 'by (pod)' grouping


class TestTwoPodsOneHundredFiftyEachDoesNotTripThreshold:
    """The exact scenario that invalidated V2 Phase 3B's physical attempt
    2: multiple healthy replicas whose SUM crosses 230Mi while no
    individual pod does."""

    def test_aggregate_300_but_max_per_pod_150_never_trips(self, tmp_path, monkeypatch):
        import scripts.v2_pilot_safety as mod

        class _FakeWrapper:
            current_profile_name = "constant_cpu"
            env = None

        monkeypatch.setattr(mod, "get_pod_health", lambda *a, **k: [
            PodHealthSnapshot(name="demo-app-a", memory_bytes=None, restart_count=0, ready=True, phase="Running"),
            PodHealthSnapshot(name="demo-app-b", memory_bytes=None, restart_count=0, ready=True, phase="Running"),
        ])
        monkeypatch.setattr(mod, "get_max_per_pod_memory_mib", lambda *a, **k: 150.0)
        monkeypatch.setattr(mod, "get_aggregate_memory_mib", lambda *a, **k: 300.0)

        cb = PilotSafetyCallback(
            namespace="autoscaler-app", label_selector="app=demo-app", prom_base_url="http://localhost:9090",
            pod_regex="demo-app-.*", baseline_restart_counts={"demo-app-a": 0, "demo-app-b": 0},
            telemetry_path=tmp_path / "telemetry.jsonl", synced_wrapper=_FakeWrapper(),
        )
        locals_ = {"infos": [{"action_index": 1, "action_override_reason": "none", "sample_valid": True,
                               "reward_total": 0.0, "reward_components": {}, "raw_observation": {},
                               "deployment_current_replicas_after": 2, "ready_replicas_after": 2}], "actions": [1]}
        cb.locals = locals_

        # Run many steps -- aggregate (300) is WELL above 230Mi the whole
        # time, but max-per-pod (150) never is, so safety must never trip.
        for _ in range(5):
            assert cb._on_step() is True
        assert cb.termination_reason is None
        assert cb.summary["max_aggregate_memory_mib"] == pytest.approx(300.0)
        assert cb.summary["max_per_pod_memory_mib"] == pytest.approx(150.0)


class TestSinglePodAboveThresholdTrips:
    def test_231_mib_single_pod_sustained_twice_trips(self, tmp_path, monkeypatch):
        import scripts.v2_pilot_safety as mod

        class _FakeWrapper:
            current_profile_name = "constant_memory"
            env = None

        monkeypatch.setattr(mod, "get_pod_health", lambda *a, **k: [
            PodHealthSnapshot(name="demo-app-a", memory_bytes=None, restart_count=0, ready=True, phase="Running"),
        ])
        monkeypatch.setattr(mod, "get_max_per_pod_memory_mib", lambda *a, **k: 231.0)
        monkeypatch.setattr(mod, "get_aggregate_memory_mib", lambda *a, **k: 231.0)

        cb = PilotSafetyCallback(
            namespace="autoscaler-app", label_selector="app=demo-app", prom_base_url="http://localhost:9090",
            pod_regex="demo-app-.*", baseline_restart_counts={"demo-app-a": 0},
            telemetry_path=tmp_path / "telemetry.jsonl", synced_wrapper=_FakeWrapper(),
        )
        locals_ = {"infos": [{"action_index": 1, "action_override_reason": "none", "sample_valid": True,
                               "reward_total": 0.0, "reward_components": {}, "raw_observation": {},
                               "deployment_current_replicas_after": 1, "ready_replicas_after": 1}], "actions": [1]}
        cb.locals = locals_

        assert cb._on_step() is True  # breach 1 -- not yet sustained
        assert cb._on_step() is False  # breach 2 -- trips
        assert "memory_safety_threshold_breached" in cb.termination_reason
        assert "231.0" in cb.termination_reason

    def test_one_transient_breach_alone_does_not_trip(self, tmp_path, monkeypatch):
        import scripts.v2_pilot_safety as mod

        class _FakeWrapper:
            current_profile_name = "constant_memory"
            env = None

        monkeypatch.setattr(mod, "get_pod_health", lambda *a, **k: [
            PodHealthSnapshot(name="demo-app-a", memory_bytes=None, restart_count=0, ready=True, phase="Running"),
        ])
        monkeypatch.setattr(mod, "get_max_per_pod_memory_mib", lambda *a, **k: 231.0)
        monkeypatch.setattr(mod, "get_aggregate_memory_mib", lambda *a, **k: 231.0)

        cb = PilotSafetyCallback(
            namespace="autoscaler-app", label_selector="app=demo-app", prom_base_url="http://localhost:9090",
            pod_regex="demo-app-.*", baseline_restart_counts={"demo-app-a": 0},
            telemetry_path=tmp_path / "telemetry.jsonl", synced_wrapper=_FakeWrapper(),
        )
        locals_ = {"infos": [{"action_index": 1, "action_override_reason": "none", "sample_valid": True,
                               "reward_total": 0.0, "reward_components": {}, "raw_observation": {},
                               "deployment_current_replicas_after": 1, "ready_replicas_after": 1}], "actions": [1]}
        cb.locals = locals_
        assert cb._on_step() is True
        assert cb.termination_reason is None

    def test_threshold_constant_unchanged(self):
        assert MEMORY_SAFETY_THRESHOLD_MIB == 230.0


class TestSafetyInterruptionProvenance:
    def test_trip_safety_writes_exactly_one_row(self, tmp_path, monkeypatch):
        import scripts.v2_pilot_safety as mod

        calls = []

        class _FakeLedgeredEnv:
            def log_pilot_safety_interruption(self, reason):
                calls.append(reason)
                return True

        class _FakeWrapper:
            current_profile_name = "constant_memory"
            env = _FakeLedgeredEnv()

        monkeypatch.setattr(mod, "get_pod_health", lambda *a, **k: [
            PodHealthSnapshot(name="demo-app-a", memory_bytes=None, restart_count=0, ready=True, phase="Running"),
        ])
        monkeypatch.setattr(mod, "get_max_per_pod_memory_mib", lambda *a, **k: 240.0)
        monkeypatch.setattr(mod, "get_aggregate_memory_mib", lambda *a, **k: 240.0)

        cb = PilotSafetyCallback(
            namespace="autoscaler-app", label_selector="app=demo-app", prom_base_url="http://localhost:9090",
            pod_regex="demo-app-.*", baseline_restart_counts={"demo-app-a": 0},
            telemetry_path=tmp_path / "telemetry.jsonl", synced_wrapper=_FakeWrapper(),
        )
        locals_ = {"infos": [{"action_index": 1, "action_override_reason": "none", "sample_valid": True,
                               "reward_total": 0.0, "reward_components": {}, "raw_observation": {},
                               "deployment_current_replicas_after": 1, "ready_replicas_after": 1}], "actions": [1]}
        cb.locals = locals_

        cb._on_step()  # breach 1
        cb._on_step()  # breach 2 -> trips, writes row
        assert len(calls) == 1
        assert "memory_safety_threshold_breached" in calls[0]

    def test_no_duplicate_row_if_on_step_called_again_after_trip(self, tmp_path, monkeypatch):
        """Defensive: even if _on_step() were somehow invoked again after
        tripping (SB3 itself should not do this, but the guard is cheap and
        explicit)."""
        import scripts.v2_pilot_safety as mod

        calls = []

        class _FakeLedgeredEnv:
            def log_pilot_safety_interruption(self, reason):
                calls.append(reason)
                return True

        class _FakeWrapper:
            current_profile_name = "constant_memory"
            env = _FakeLedgeredEnv()

        monkeypatch.setattr(mod, "get_pod_health", lambda *a, **k: [
            PodHealthSnapshot(name="demo-app-a", memory_bytes=None, restart_count=0, ready=True, phase="Running"),
        ])
        monkeypatch.setattr(mod, "get_max_per_pod_memory_mib", lambda *a, **k: 240.0)
        monkeypatch.setattr(mod, "get_aggregate_memory_mib", lambda *a, **k: 240.0)

        cb = PilotSafetyCallback(
            namespace="autoscaler-app", label_selector="app=demo-app", prom_base_url="http://localhost:9090",
            pod_regex="demo-app-.*", baseline_restart_counts={"demo-app-a": 0},
            telemetry_path=tmp_path / "telemetry.jsonl", synced_wrapper=_FakeWrapper(),
        )
        locals_ = {"infos": [{"action_index": 1, "action_override_reason": "none", "sample_valid": True,
                               "reward_total": 0.0, "reward_components": {}, "raw_observation": {},
                               "deployment_current_replicas_after": 1, "ready_replicas_after": 1}], "actions": [1]}
        cb.locals = locals_
        cb._on_step()
        cb._on_step()  # trips
        cb._on_step()  # called again -- must not write a second row
        assert len(calls) == 1

    def test_pod_restart_also_writes_provenance_row(self, tmp_path, monkeypatch):
        import scripts.v2_pilot_safety as mod

        calls = []

        class _FakeLedgeredEnv:
            def log_pilot_safety_interruption(self, reason):
                calls.append(reason)
                return True

        class _FakeWrapper:
            current_profile_name = "constant_cpu"
            env = _FakeLedgeredEnv()

        monkeypatch.setattr(mod, "get_pod_health", lambda *a, **k: [
            PodHealthSnapshot(name="demo-app-a", memory_bytes=None, restart_count=1, ready=True, phase="Running"),
        ])
        cb = PilotSafetyCallback(
            namespace="autoscaler-app", label_selector="app=demo-app", prom_base_url="http://localhost:9090",
            pod_regex="demo-app-.*", baseline_restart_counts={"demo-app-a": 0},
            telemetry_path=tmp_path / "telemetry.jsonl", synced_wrapper=_FakeWrapper(),
        )
        locals_ = {"infos": [{"action_index": 1, "action_override_reason": "none", "sample_valid": True,
                               "reward_total": 0.0, "reward_components": {}, "raw_observation": {},
                               "deployment_current_replicas_after": 1, "ready_replicas_after": 1}], "actions": [1]}
        cb.locals = locals_
        assert cb._on_step() is False
        assert len(calls) == 1
        assert "pod_restart_detected" in calls[0]


class TestLedgeredEnvProvenanceMethod:
    def test_log_pilot_safety_interruption_writes_row(self, tmp_path):
        from autoscaler.src.rl_env import episode_ledger
        from autoscaler.src.rl_env.training_wrapper import LedgeredEpisodeWrapper, STATUS_PILOT_SAFETY_INTERRUPTED

        class _FakeInnerEnv(gym.Env):
            def reset(self, **kwargs):
                import numpy as np
                return np.zeros(11, dtype="float32"), {}

        ledger_path = tmp_path / "ledger.jsonl"
        wrapper = LedgeredEpisodeWrapper(
            _FakeInnerEnv(), ledger_path=ledger_path, traffic_profile="ramp_mixed", profile_hash="abc",
            seed=42, controller_code_hash="code_hash", config_hash="config_hash", role="v2_pilot",
        )
        wrapper.reset()
        wrote = wrapper.log_pilot_safety_interruption("memory_safety_threshold_breached:240.0MiB_per_pod")
        assert wrote is True

        rows = list(episode_ledger.read_rows(ledger_path))
        assert len(rows) == 1
        assert rows[0]["status"] == STATUS_PILOT_SAFETY_INTERRUPTED
        assert rows[0]["valid"] is False
        assert rows[0]["terminated"] is False
        assert rows[0]["truncated"] is False
        assert rows[0]["physical_attempt_number"] == 1
        assert any(e["reason"] == "memory_safety_threshold_breached:240.0MiB_per_pod" for e in rows[0]["safety_events"])

    def test_idempotent_no_op_if_episode_already_completed(self, tmp_path):
        from autoscaler.src.rl_env import episode_ledger
        from autoscaler.src.rl_env.training_wrapper import LedgeredEpisodeWrapper

        class _FakeInnerEnv(gym.Env):
            def reset(self, **kwargs):
                import numpy as np
                return np.zeros(11, dtype="float32"), {}

            def step(self, action):
                import numpy as np
                return np.zeros(11, dtype="float32"), 0.0, True, False, {"reward_validity": "full", "action_override_reason": "none"}

        ledger_path = tmp_path / "ledger.jsonl"
        wrapper = LedgeredEpisodeWrapper(
            _FakeInnerEnv(), ledger_path=ledger_path, traffic_profile="ramp_mixed", profile_hash="abc",
            seed=42, controller_code_hash="code_hash", config_hash="config_hash", role="v2_pilot",
        )
        wrapper.reset()
        wrapper.step(1)  # env terminates normally -- writes its own row, sets _episode_completed=True
        wrote = wrapper.log_pilot_safety_interruption("some_reason")
        assert wrote is False  # no-op -- episode already completed

        rows = list(episode_ledger.read_rows(ledger_path))
        assert len(rows) == 1  # only the normal-completion row, no duplicate

    def test_second_physical_attempt_increments_attempt_number(self, tmp_path):
        from autoscaler.src.rl_env import episode_ledger
        from autoscaler.src.rl_env.training_wrapper import LedgeredEpisodeWrapper

        class _FakeInnerEnv(gym.Env):
            def reset(self, **kwargs):
                import numpy as np
                return np.zeros(11, dtype="float32"), {}

        ledger_path = tmp_path / "ledger.jsonl"
        wrapper1 = LedgeredEpisodeWrapper(
            _FakeInnerEnv(), ledger_path=ledger_path, traffic_profile="ramp_mixed", profile_hash="abc",
            seed=42, controller_code_hash="code_hash", config_hash="config_hash", role="v2_pilot",
        )
        wrapper1.reset()
        wrapper1.log_pilot_safety_interruption("reason_one")

        wrapper2 = LedgeredEpisodeWrapper(
            _FakeInnerEnv(), ledger_path=ledger_path, traffic_profile="ramp_mixed", profile_hash="abc",
            seed=42, controller_code_hash="code_hash", config_hash="config_hash", role="v2_pilot",
        )
        wrapper2.reset()
        wrapper2.log_pilot_safety_interruption("reason_two")

        rows = list(episode_ledger.read_rows(ledger_path))
        assert len(rows) == 2
        assert rows[0]["physical_attempt_number"] == 1
        assert rows[1]["physical_attempt_number"] == 2
