"""tests/unit/test_v2_normalized_observation_telemetry.py

V2 Phase 4A PART 3/11: proves `PilotSafetyCallback` now persists the EXACT
11-element normalized vector SB3 received from `env.step()` (via SB3's own
`self.locals["new_obs"]`), never a separately-recomputed approximation --
this is additive instrumentation only; it must not change any existing
counted/derived telemetry field.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.v2_pilot_safety import PilotSafetyCallback


class _FakeWrapper:
    current_profile_name = "constant_cpu"


def _make_callback(tmp_path, monkeypatch, pod_health_fn, memory_fn):
    import scripts.v2_pilot_safety as mod

    cb = PilotSafetyCallback(
        namespace="autoscaler-app", label_selector="app=demo-app",
        prom_base_url="http://localhost:19090", pod_regex="demo-app-.*",
        baseline_restart_counts={"demo-app-abc": 0}, telemetry_path=tmp_path / "telemetry.jsonl",
        synced_wrapper=_FakeWrapper(),
    )
    monkeypatch.setattr(mod, "get_pod_health", pod_health_fn)
    monkeypatch.setattr(mod, "get_max_per_pod_memory_mib", memory_fn)
    monkeypatch.setattr(mod, "get_aggregate_memory_mib", memory_fn)
    return cb


def _healthy_pods():
    from scripts.v2_pilot_safety import PodHealthSnapshot

    return [PodHealthSnapshot(name="demo-app-abc", memory_bytes=None, restart_count=0, ready=True, phase="Running")]


def _step_locals_with_obs(new_obs_row):
    return {
        "infos": [{
            "action_index": 1, "action_override_reason": "none", "sample_valid": True,
            "reward_total": -0.1, "reward_components": {"sla": 0.1}, "raw_observation": {"cpu_per_replica": 1.0},
            "deployment_current_replicas_after": 1, "ready_replicas_after": 1,
        }],
        "actions": [1],
        "new_obs": np.array([new_obs_row], dtype=np.float32),
    }


class TestNormalizedObservationTelemetry:
    def test_length_is_11(self, tmp_path, monkeypatch):
        cb = _make_callback(tmp_path, monkeypatch, pod_health_fn=lambda *a, **k: _healthy_pods(), memory_fn=lambda *a, **k: 60.0)
        vector = [0.1 * i for i in range(11)]
        cb.locals = _step_locals_with_obs(vector)
        cb._on_step()
        row = json.loads(cb.telemetry_path.read_text().strip().splitlines()[-1])
        assert len(row["normalized_observation"]) == 11

    def test_exact_match_to_sb3s_own_new_obs(self, tmp_path, monkeypatch):
        """The persisted vector must be numerically identical (within float32
        round-trip tolerance) to the exact array SB3 handed the callback --
        never recomputed via a separate normalization path."""
        cb = _make_callback(tmp_path, monkeypatch, pod_health_fn=lambda *a, **k: _healthy_pods(), memory_fn=lambda *a, **k: 60.0)
        vector = [0.5, -0.5, 0.25, 1.0, 0.0, -1.0, 0.75, -0.25, 0.1, 0.9, 0.3]
        cb.locals = _step_locals_with_obs(vector)
        cb._on_step()
        row = json.loads(cb.telemetry_path.read_text().strip().splitlines()[-1])
        assert row["normalized_observation"] == pytest.approx(vector, abs=1e-6)

    def test_none_when_new_obs_unavailable(self, tmp_path, monkeypatch):
        """Backward-compatible: existing tests that manually construct
        `.locals` without a `new_obs` key (e.g. test_v2_pilot.py's
        `_step_locals()`) must keep working -- the new field degrades to
        `None`, never raises."""
        cb = _make_callback(tmp_path, monkeypatch, pod_health_fn=lambda *a, **k: _healthy_pods(), memory_fn=lambda *a, **k: 60.0)
        cb.locals = {
            "infos": [{"action_index": 1, "action_override_reason": "none", "sample_valid": True}],
            "actions": [1],
        }
        cb._on_step()
        row = json.loads(cb.telemetry_path.read_text().strip().splitlines()[-1])
        assert row["normalized_observation"] is None

    def test_existing_telemetry_fields_unchanged(self, tmp_path, monkeypatch):
        """Adding normalized_observation must not alter any pre-existing
        counted/derived field (profile/action/override counts, invalid-step
        detection, safety-trip logic)."""
        cb = _make_callback(tmp_path, monkeypatch, pod_health_fn=lambda *a, **k: _healthy_pods(), memory_fn=lambda *a, **k: 60.0)
        cb.locals = _step_locals_with_obs([0.0] * 11)
        result = cb._on_step()
        assert result is True
        assert cb.summary["profile_counts"]["constant_cpu"] == 1
        assert cb.summary["action_counts"][1] == 1
        assert cb.summary["effective_action_counts"][1] == 1
        assert cb.summary["invalid_step_count"] == 0
        assert cb.termination_reason is None
