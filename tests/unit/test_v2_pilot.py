"""tests/unit/test_v2_pilot.py

V2 Phase 3A PART 13: targeted, fully-faked-cluster tests for the new pilot
trainer code (scripts.v2_traffic_orchestrator, scripts.v2_pilot_safety),
required BEFORE live execution per PART 13's own instruction.
"""

from __future__ import annotations

import pytest

from scripts.v2_pilot_safety import (
    MEMORY_SAFETY_THRESHOLD_MIB,
    SUSTAINED_BREACH_SAMPLES,
    PilotSafetyCallback,
    get_aggregate_memory_mib,
    get_pod_health,
)
from scripts.v2_traffic_orchestrator import (
    V2ProfileFirewallViolation,
    resolve_v2_profile_for_episode,
    resolve_v2_profile_metadata,
)
from autoscaler.src.rl_env.training_contract_v2 import V2_TRAINING_PROFILE_ALLOWLIST
from autoscaler.src.rl_env.training_schedule_v2 import build_training_profile_rotation


class TestResolveV2ProfileMetadata:
    def test_valid_profile_resolves(self):
        meta = resolve_v2_profile_metadata("constant_cpu")
        assert meta.name == "constant_cpu"
        assert meta.path.exists()
        assert meta.seed == 42

    def test_all_allowlisted_profiles_resolve(self):
        for name in V2_TRAINING_PROFILE_ALLOWLIST:
            meta = resolve_v2_profile_metadata(name)
            assert meta.duration_seconds > 0

    def test_non_allowlisted_profile_rejected(self):
        with pytest.raises(V2ProfileFirewallViolation):
            resolve_v2_profile_metadata("some_v1_only_profile")

    def test_v1_profile_name_rejected_even_if_coincidentally_shaped(self):
        with pytest.raises(V2ProfileFirewallViolation):
            resolve_v2_profile_metadata("constant")  # a real V1 profile name, not in the V2 allowlist


class TestResolveV2ProfileForEpisode:
    def test_matches_training_schedule_rotation(self):
        schedule = build_training_profile_rotation(42, 10)
        for i in range(10):
            assert resolve_v2_profile_for_episode(i, 42) == schedule[i]

    def test_negative_episode_index_rejected(self):
        with pytest.raises(ValueError):
            resolve_v2_profile_for_episode(-1, 42)


class _FakeCompletedProcess:
    def __init__(self, returncode=0, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


class TestGetPodHealth:
    def test_parses_healthy_pod(self):
        fake_json = """{
            "items": [{
                "metadata": {"name": "demo-app-abc"},
                "status": {"phase": "Running", "containerStatuses": [{"ready": true, "restartCount": 0}]}
            }]
        }"""

        def fake_run(cmd, **kwargs):
            return _FakeCompletedProcess(0, fake_json)

        pods = get_pod_health("autoscaler-app", "app=demo-app", run=fake_run)
        assert len(pods) == 1
        assert pods[0].ready is True
        assert pods[0].restart_count == 0
        assert pods[0].phase == "Running"

    def test_parses_restarted_pod(self):
        fake_json = """{
            "items": [{
                "metadata": {"name": "demo-app-abc"},
                "status": {"phase": "Running", "containerStatuses": [{"ready": true, "restartCount": 3}]}
            }]
        }"""

        def fake_run(cmd, **kwargs):
            return _FakeCompletedProcess(0, fake_json)

        pods = get_pod_health("autoscaler-app", "app=demo-app", run=fake_run)
        assert pods[0].restart_count == 3

    def test_kubectl_failure_returns_empty_list_not_raise(self):
        def fake_run(cmd, **kwargs):
            return _FakeCompletedProcess(1, "")

        pods = get_pod_health("autoscaler-app", "app=demo-app", run=fake_run)
        assert pods == []

    def test_not_ready_pod_detected(self):
        fake_json = """{
            "items": [{
                "metadata": {"name": "demo-app-abc"},
                "status": {"phase": "Running", "containerStatuses": [{"ready": false, "restartCount": 0}]}
            }]
        }"""

        def fake_run(cmd, **kwargs):
            return _FakeCompletedProcess(0, fake_json)

        pods = get_pod_health("autoscaler-app", "app=demo-app", run=fake_run)
        assert pods[0].ready is False


class TestGetAggregateMemoryMib:
    def test_parses_value(self, monkeypatch):
        import json
        import urllib.request

        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps({"data": {"result": [{"value": [0, str(100 * 1048576)]}]}}).encode()

        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _FakeResponse())
        val = get_aggregate_memory_mib("http://x", "ns", "demo-app-.*")
        assert val == pytest.approx(100.0)

    def test_no_data_returns_none(self, monkeypatch):
        import json
        import urllib.request

        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps({"data": {"result": []}}).encode()

        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _FakeResponse())
        assert get_aggregate_memory_mib("http://x", "ns", "demo-app-.*") is None

    def test_query_exception_returns_none_not_raise(self, monkeypatch):
        import urllib.request

        def _raise(*a, **k):
            raise OSError("connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", _raise)
        assert get_aggregate_memory_mib("http://x", "ns", "demo-app-.*") is None


class _FakeWrapper:
    current_profile_name = "constant_cpu"


def _make_callback(
    tmp_path, monkeypatch, pod_health_fn=None, memory_fn=None, aggregate_memory_fn=None, baseline_restarts=None,
):
    """`memory_fn` patches `get_max_per_pod_memory_mib` -- the metric that
    actually drives the safety trip (V2 Phase 3C). `aggregate_memory_fn`
    (defaults to the same value as `memory_fn` unless given separately)
    patches `get_aggregate_memory_mib`, which is reporting-only and must
    never itself trip safety -- see test_v2_pilot_retry.py for tests that
    deliberately diverge the two."""
    import scripts.v2_pilot_safety as mod

    cb = PilotSafetyCallback(
        namespace="autoscaler-app", label_selector="app=demo-app",
        prom_base_url="http://localhost:19090", pod_regex="demo-app-.*",
        baseline_restart_counts=baseline_restarts or {"demo-app-abc": 0},
        telemetry_path=tmp_path / "telemetry.jsonl", synced_wrapper=_FakeWrapper(),
    )
    if pod_health_fn is not None:
        monkeypatch.setattr(mod, "get_pod_health", pod_health_fn)
    if memory_fn is not None:
        monkeypatch.setattr(mod, "get_max_per_pod_memory_mib", memory_fn)
        monkeypatch.setattr(mod, "get_aggregate_memory_mib", aggregate_memory_fn or memory_fn)
    return cb


def _healthy_pods():
    from scripts.v2_pilot_safety import PodHealthSnapshot

    return [PodHealthSnapshot(name="demo-app-abc", memory_bytes=None, restart_count=0, ready=True, phase="Running")]


def _step_locals(action_index=1, override_reason="none", sample_valid=True):
    return {
        "infos": [{
            "action_index": action_index, "action_override_reason": override_reason, "sample_valid": sample_valid,
            "reward_total": -0.1, "reward_components": {"sla": 0.1}, "raw_observation": {"cpu_per_replica": 1.0},
            "deployment_current_replicas_after": 1, "ready_replicas_after": 1,
        }],
        "actions": [1],
    }


class TestPilotSafetyCallbackNormalOperation:
    def test_continues_when_healthy(self, tmp_path, monkeypatch):
        cb = _make_callback(tmp_path, monkeypatch, pod_health_fn=lambda *a, **k: _healthy_pods(), memory_fn=lambda *a, **k: 60.0)
        cb.locals = _step_locals()
        assert cb._on_step() is True
        assert cb.termination_reason is None

    def test_telemetry_row_written(self, tmp_path, monkeypatch):
        cb = _make_callback(tmp_path, monkeypatch, pod_health_fn=lambda *a, **k: _healthy_pods(), memory_fn=lambda *a, **k: 60.0)
        cb.locals = _step_locals()
        cb._on_step()
        content = cb.telemetry_path.read_text()
        assert "constant_cpu" in content
        assert "reward_total" in content

    def test_counts_tracked(self, tmp_path, monkeypatch):
        cb = _make_callback(tmp_path, monkeypatch, pod_health_fn=lambda *a, **k: _healthy_pods(), memory_fn=lambda *a, **k: 60.0)
        cb.locals = _step_locals(action_index=2, override_reason="replica_bound_clamp")
        cb._on_step()
        assert cb.summary["profile_counts"]["constant_cpu"] == 1
        assert cb.summary["action_counts"][1] == 1  # sampled action from _step_locals' "actions"
        assert cb.summary["effective_action_counts"][2] == 1
        assert cb.summary["override_counts"]["replica_bound_clamp"] == 1

    def test_invalid_step_counted(self, tmp_path, monkeypatch):
        cb = _make_callback(tmp_path, monkeypatch, pod_health_fn=lambda *a, **k: _healthy_pods(), memory_fn=lambda *a, **k: 60.0)
        cb.locals = _step_locals(sample_valid=False)
        cb._on_step()
        assert cb.summary["invalid_step_count"] == 1
        assert cb.summary["invalid_step_rate"] == pytest.approx(1.0)


class TestPilotSafetyCallbackHardSafety:
    def test_memory_breach_requires_sustained_samples(self, tmp_path, monkeypatch):
        cb = _make_callback(tmp_path, monkeypatch, pod_health_fn=lambda *a, **k: _healthy_pods(), memory_fn=lambda *a, **k: 240.0)
        cb.locals = _step_locals()
        assert SUSTAINED_BREACH_SAMPLES == 2
        assert cb._on_step() is True  # first breach alone is not enough
        assert cb.termination_reason is None
        assert cb._on_step() is False  # second consecutive breach trips it
        assert cb.termination_reason is not None
        assert "memory_safety_threshold_breached" in cb.termination_reason

    def test_memory_breach_resets_on_a_healthy_sample(self, tmp_path, monkeypatch):
        # One value per _on_step() call -- both the per-pod AND aggregate
        # queries are invoked within a single _on_step(), so a plain
        # iterator would be consumed twice per step; this returns the same
        # value to both calls within a step, advancing only between steps.
        values = [240.0, 60.0, 240.0, 240.0]
        state = {"calls": 0}

        def _next_value(*a, **k):
            idx = state["calls"] // 2
            state["calls"] += 1
            return values[idx]

        cb = _make_callback(tmp_path, monkeypatch, pod_health_fn=lambda *a, **k: _healthy_pods(), memory_fn=_next_value)
        cb.locals = _step_locals()
        assert cb._on_step() is True  # breach 1
        assert cb._on_step() is True  # healthy resets the counter
        assert cb._on_step() is True  # breach 1 again
        assert cb._on_step() is False  # breach 2 -> trips

    def test_pod_restart_detected_trips_immediately(self, tmp_path, monkeypatch):
        from scripts.v2_pilot_safety import PodHealthSnapshot

        restarted = [PodHealthSnapshot(name="demo-app-abc", memory_bytes=None, restart_count=1, ready=True, phase="Running")]
        cb = _make_callback(
            tmp_path, monkeypatch, pod_health_fn=lambda *a, **k: restarted, memory_fn=lambda *a, **k: 60.0,
            baseline_restarts={"demo-app-abc": 0},
        )
        cb.locals = _step_locals()
        assert cb._on_step() is False
        assert "pod_restart_detected" in cb.termination_reason

    def test_zero_ready_replicas_trips_immediately(self, tmp_path, monkeypatch):
        from scripts.v2_pilot_safety import PodHealthSnapshot

        not_ready = [PodHealthSnapshot(name="demo-app-abc", memory_bytes=None, restart_count=0, ready=False, phase="Running")]
        cb = _make_callback(tmp_path, monkeypatch, pod_health_fn=lambda *a, **k: not_ready, memory_fn=lambda *a, **k: 60.0)
        cb.locals = _step_locals()
        assert cb._on_step() is False
        assert cb.termination_reason == "zero_ready_replicas"

    def test_safety_threshold_is_the_fixed_230_mib(self):
        assert MEMORY_SAFETY_THRESHOLD_MIB == 230.0

    def test_monitoring_query_failure_does_not_trip_safety(self, tmp_path, monkeypatch):
        """A transient Prometheus query failure (None returned) must never
        itself be treated as a safety breach -- only a genuine high reading
        does."""
        cb = _make_callback(tmp_path, monkeypatch, pod_health_fn=lambda *a, **k: _healthy_pods(), memory_fn=lambda *a, **k: None)
        cb.locals = _step_locals()
        assert cb._on_step() is True
        assert cb.termination_reason is None
