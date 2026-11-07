"""tests/unit/test_v2_pilot_retry.py

V2 Phase 3B PART 4: proves the Prometheus tooling correction and retry
provenance machinery, all via fakes -- no real cluster/Prometheus needed.
"""

from __future__ import annotations

import json

import pytest

from scripts.v2_pilot_safety import PROMETHEUS_URL, verify_prometheus_internal_paths
from scripts.train_ppo_live_v2 import resolve_next_pilot_session_dir


class TestSingleAuthoritativeUrl:
    def test_prometheus_url_is_the_standard_port(self):
        assert PROMETHEUS_URL == "http://localhost:9090"

    def test_backend_receives_the_shared_url(self):
        """V2 Phase 3A's bug: LiveKubernetesBackend was constructed with no
        explicit prometheus_url, silently using its own unrelated default.
        Confirms train_ppo_live_v2.py's source now passes PROMETHEUS_URL
        explicitly."""
        source = open("scripts/train_ppo_live_v2.py", encoding="utf-8").read()
        assert "LiveKubernetesBackend(prometheus_url=PROMETHEUS_URL)" in source

    def test_orchestrator_receives_the_shared_url(self):
        source = open("scripts/train_ppo_live_v2.py", encoding="utf-8").read()
        assert "PrometheusClient(base_url=PROMETHEUS_URL)" in source

    def test_safety_callback_receives_the_shared_url(self):
        source = open("scripts/train_ppo_live_v2.py", encoding="utf-8").read()
        assert "prom_base_url=PROMETHEUS_URL" in source

    def test_no_hardcoded_alternate_port_remains_in_trainer(self):
        """The exact bug from physical attempt 1: a hardcoded literal
        'http://localhost:19090' URL. A code comment DOCUMENTING that past
        bug (for posterity) is fine -- an actual literal URL assignment is
        not."""
        source = open("scripts/train_ppo_live_v2.py", encoding="utf-8").read()
        assert '"http://localhost:19090"' not in source
        assert "'http://localhost:19090'" not in source


class _FakePromClient:
    def __init__(self, ready=True, ready_message="Ready", backend_result=(1.0, None), drain_result=(None, None)):
        self._ready = ready
        self._ready_message = ready_message
        self._backend_result = backend_result
        self._drain_result = drain_result
        self.queries_made = []

    def check_readiness(self):
        return self._ready, self._ready_message

    def query(self, promql):
        self.queries_made.append(promql)
        if "demo_app_requests_in_progress" in promql:
            return self._drain_result
        return self._backend_result


class TestVerifyPrometheusInternalPaths:
    def test_all_ok_when_healthy(self, monkeypatch):
        import collectors.metrics_collector as mc

        fake = _FakePromClient()
        monkeypatch.setattr(mc, "PrometheusClient", lambda base_url: fake)
        result = verify_prometheus_internal_paths("http://localhost:9090")
        assert result["all_ok"] is True
        assert result["backend_query_ok"] is True
        assert result["drain_query_ok"] is True

    def test_absent_drain_metric_is_not_a_failure(self, monkeypatch):
        """(None, None) -- an empty/absent series -- must NOT be treated as
        a Prometheus connectivity failure (PrometheusClient.query()'s own
        documented contract)."""
        import collectors.metrics_collector as mc

        fake = _FakePromClient(drain_result=(None, None))
        monkeypatch.setattr(mc, "PrometheusClient", lambda base_url: fake)
        result = verify_prometheus_internal_paths("http://localhost:9090")
        assert result["drain_query_ok"] is True
        assert result["all_ok"] is True

    def test_connection_refused_fails_closed(self, monkeypatch):
        import collectors.metrics_collector as mc

        fake = _FakePromClient(ready=False, ready_message="connection refused", backend_result=(None, "connection refused"))
        monkeypatch.setattr(mc, "PrometheusClient", lambda base_url: fake)
        result = verify_prometheus_internal_paths("http://localhost:9090")
        assert result["all_ok"] is False
        assert result["ready"] is False
        assert result["backend_query_ok"] is False

    def test_drain_query_error_fails_closed(self, monkeypatch):
        import collectors.metrics_collector as mc

        fake = _FakePromClient(drain_result=(None, "query timeout"))
        monkeypatch.setattr(mc, "PrometheusClient", lambda base_url: fake)
        result = verify_prometheus_internal_paths("http://localhost:9090")
        assert result["all_ok"] is False
        assert result["drain_query_ok"] is False

    def test_uses_the_exact_drain_promql_shape(self, monkeypatch):
        import collectors.metrics_collector as mc

        fake = _FakePromClient()
        monkeypatch.setattr(mc, "PrometheusClient", lambda base_url: fake)
        verify_prometheus_internal_paths("http://localhost:9090", namespace="autoscaler-app", route="/work")
        assert any("demo_app_requests_in_progress" in q and "autoscaler-app" in q and "/work" in q for q in fake.queries_made)


class TestPreflightFailsBeforeClusterMutation:
    def test_run_pilot_training_raises_before_acquire_controller_on_preflight_failure(self, monkeypatch, tmp_path):
        import scripts.train_ppo_live_v2 as mod
        import scripts.v2_pilot_safety as safety_mod

        # run_pilot_training() does `from scripts.v2_pilot_safety import
        # verify_prometheus_internal_paths` as a LOCAL import, re-resolved
        # from safety_mod's own namespace on every call -- patch it there.
        monkeypatch.setattr(
            safety_mod, "verify_prometheus_internal_paths",
            lambda *a, **k: {"all_ok": False, "ready": False, "backend_query_ok": False, "drain_query_ok": False},
        )

        def _explode(*a, **k):
            raise AssertionError("acquire_controller must never be called when Prometheus preflight fails")

        monkeypatch.setattr(mod, "acquire_controller", _explode)
        monkeypatch.setattr(mod, "resolve_next_pilot_session_dir", lambda: tmp_path / "session_999")

        with pytest.raises(RuntimeError, match="Prometheus internal-path preflight failed"):
            mod.run_pilot_training(seed=42)

        assert not (tmp_path / "session_999").exists()  # no session directory created either


class TestRetrySessionProvenance:
    def test_first_call_returns_session_000(self, tmp_path):
        assert resolve_next_pilot_session_dir(tmp_path) == tmp_path / "session_000"

    def test_skips_existing_sessions(self, tmp_path):
        (tmp_path / "session_000").mkdir(parents=True)
        assert resolve_next_pilot_session_dir(tmp_path) == tmp_path / "session_001"

    def test_skips_multiple_existing_sessions(self, tmp_path):
        (tmp_path / "session_000").mkdir(parents=True)
        (tmp_path / "session_001").mkdir(parents=True)
        assert resolve_next_pilot_session_dir(tmp_path) == tmp_path / "session_002"

    def test_never_reuses_a_failed_attempts_directory(self, tmp_path):
        """Direct proof against the real on-disk evidence from V2 Phase 3A
        physical attempt 1 (session_000, preserved, must never be touched)."""
        real_pilot_root = __import__("pathlib").Path("persistent-data/rl_training_v2/pilot")
        if (real_pilot_root / "session_000").exists():
            next_dir = resolve_next_pilot_session_dir(real_pilot_root)
            assert next_dir != real_pilot_root / "session_000"

    def test_real_pilot_root_next_session_follows_existing_count(self):
        """V2 Phase 3C/3D PART 5/6: direct proof against the REAL on-disk
        evidence -- whatever contiguous session_NNN directories already
        exist (session_000, session_001, session_002, ... from each prior
        physical attempt), the next resolved session must be the very next
        integer, and must not collide with any existing one. Written to
        stay correct as more physical attempts accumulate across phases,
        rather than hardcoding a specific number that goes stale."""
        real_pilot_root = __import__("pathlib").Path("persistent-data/rl_training_v2/pilot")
        if not (real_pilot_root / "session_000").exists():
            pytest.skip("no physical attempt evidence present in this environment")
        n = 0
        while (real_pilot_root / f"session_{n:03d}").exists():
            n += 1
        expected_next = real_pilot_root / f"session_{n:03d}"
        actual_next = resolve_next_pilot_session_dir(real_pilot_root)
        assert actual_next == expected_next
        assert not actual_next.exists()


class TestPreviousAttemptEvidencePreserved:
    def test_attempt1_ledger_row_still_present_and_unmodified(self):
        ledger_path = __import__("pathlib").Path(
            "persistent-data/rl_training_v2/pilot/ppo_pilot_episode_ledger_v2.jsonl"
        )
        if not ledger_path.exists():
            pytest.skip("physical attempt 1 evidence not present in this environment")
        rows = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
        attempt1_rows = [r for r in rows if r.get("physical_attempt_id", "").endswith("__attempt1")]
        assert len(attempt1_rows) >= 1
        row = attempt1_rows[0]
        assert row["status"] == "safety_aborted"
        assert row["valid"] is False
        assert row["n_decision_steps"] == 3

    def test_attempt1_telemetry_file_still_present(self):
        telemetry_path = __import__("pathlib").Path(
            "persistent-data/rl_training_v2/pilot/session_000/step_telemetry.jsonl"
        )
        if not telemetry_path.exists():
            pytest.skip("physical attempt 1 evidence not present in this environment")
        lines = [line for line in telemetry_path.read_text().splitlines() if line.strip()]
        assert len(lines) == 2  # exact count preserved from physical attempt 1 -- never appended to or truncated
