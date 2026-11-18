"""tests/unit/test_v2_live_cluster_state.py

V2 Phase 9F: offline tests for scripts/v2_live_cluster_state.py's shared
RECOVER logic (`no_runner_process_alive`, `cluster_is_canonical_idle`,
`ensure_canonical_idle_state`) -- never touches a real cluster/subprocess.
Runs under the scientific .venv.
"""

from __future__ import annotations

import scripts.v2_live_cluster_state as cluster_state


class _FakeCompletedProcess:
    def __init__(self, stdout: str):
        self.stdout = stdout


class TestNoRunnerProcessAlive:
    def test_true_when_pgrep_finds_nothing(self):
        assert cluster_state.no_runner_process_alive(run_fn=lambda *a, **k: _FakeCompletedProcess("")) is True

    def test_false_when_pgrep_finds_a_pid(self):
        assert cluster_state.no_runner_process_alive(run_fn=lambda *a, **k: _FakeCompletedProcess("12345\n")) is False

    def test_fails_closed_when_pgrep_itself_cannot_run(self):
        def raising_run(*args, **kwargs):
            raise FileNotFoundError("pgrep not found")

        assert cluster_state.no_runner_process_alive(run_fn=raising_run) is False


class TestEnsureCanonicalIdleState:
    def test_never_mutates_cluster_while_a_runner_is_alive(self, monkeypatch):
        monkeypatch.setattr(cluster_state, "no_runner_process_alive", lambda: False)
        calls = []
        monkeypatch.setattr(cluster_state, "cluster_is_canonical_idle", lambda: calls.append("checked") or True)

        result = cluster_state.ensure_canonical_idle_state()

        assert calls == []  # never even checked the cluster
        assert result["ran"] is False
        assert "alive" in result["reason"]

    def test_no_op_when_cluster_already_canonical_idle(self, monkeypatch):
        monkeypatch.setattr(cluster_state, "no_runner_process_alive", lambda: True)
        monkeypatch.setattr(cluster_state, "cluster_is_canonical_idle", lambda: True)

        result = cluster_state.ensure_canonical_idle_state()

        assert result == {"ran": False, "reason": "cluster already in the canonical idle state"}

    def test_restores_idle_state_when_not_canonical_and_no_runner_alive(self, monkeypatch):
        monkeypatch.setattr(cluster_state, "no_runner_process_alive", lambda: True)
        monkeypatch.setattr(cluster_state, "cluster_is_canonical_idle", lambda: False)

        import scripts.evaluate_v2_campaign as campaign

        monkeypatch.setattr(
            campaign, "restore_canonical_idle_state",
            lambda: {"restored": True, "hpa_names": ["demo-app-hpa-v2"], "deleted_hpa_names": ["stray-hpa"], "replicas": 2},
        )

        result = cluster_state.ensure_canonical_idle_state()

        assert result["ran"] is True
        assert result["restored"] is True
        assert result["deleted_hpa_names"] == ["stray-hpa"]

    def test_reports_error_instead_of_raising_on_cluster_failure(self, monkeypatch):
        monkeypatch.setattr(cluster_state, "no_runner_process_alive", lambda: True)

        def raising_check():
            raise RuntimeError("kubectl unreachable")

        monkeypatch.setattr(cluster_state, "cluster_is_canonical_idle", raising_check)

        result = cluster_state.ensure_canonical_idle_state()

        assert result["ran"] is False
        assert "kubectl unreachable" in result["error"]


class TestRemoveHpaBeforePpoStartUnchanged:
    """Not new -- just a smoke check that the pre-existing function was
    left untouched by this phase's additions to the same module."""

    def test_is_idempotent_no_op_when_no_hpa_present(self, monkeypatch):
        import types

        fake_kubectl = types.SimpleNamespace(list_hpa_names=lambda: [], delete_hpa=lambda: None)
        fake_backend = types.SimpleNamespace(_kubectl=fake_kubectl)
        monkeypatch.setattr(
            "autoscaler.src.rl_env.live_backend.LiveKubernetesBackend",
            lambda **kwargs: fake_backend,
        )

        assert cluster_state.remove_hpa_before_ppo_start() == []
