"""tests/unit/test_evaluate_ppo_vs_hpa.py

Phase 16V/Index38A: tests for scripts/evaluate_ppo_vs_hpa.py. The --dry-run
path is exercised directly. run_live_evaluation() and its per-episode
helpers require a real live Kubernetes cluster (LiveKubernetesBackend,
real kubectl/Prometheus) and are NOT invoked here -- only their presence,
signatures, and the CLI's gating (--execute requires
--i-confirm-live-cluster-interaction) are checked offline.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


import scripts.evaluate_ppo_vs_hpa as eval_mod

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestEvaluationPlan:
    def test_plan_has_six_episodes(self):
        plan = eval_mod.build_evaluation_plan()
        assert len(plan) == 6

    def test_plan_alternates_ppo_and_hpa_never_simultaneous(self):
        plan = eval_mod.build_evaluation_plan()
        controllers = [p["controller"] for p in plan]
        assert controllers == ["PPO", "HPA", "PPO", "HPA", "PPO", "HPA"]

    def test_plan_uses_identical_workload_across_all_episodes(self):
        plan = eval_mod.build_evaluation_plan()
        profiles = {p["workload_profile"] for p in plan}
        hashes = {p["workload_hash"] for p in plan}
        assert len(profiles) == 1
        assert len(hashes) == 1

    def test_plan_has_distinct_episode_ids(self):
        plan = eval_mod.build_evaluation_plan()
        ids = [p["episode_id"] for p in plan]
        assert len(ids) == len(set(ids))

    def test_no_planned_episode_is_marked_executed(self):
        plan = eval_mod.build_evaluation_plan()
        assert all(p["executed"] is False for p in plan)

    def test_ppo_and_hpa_episode_ids_are_distinguishable(self):
        plan = eval_mod.build_evaluation_plan()
        ppo_ids = {p["episode_id"] for p in plan if p["controller"] == "PPO"}
        hpa_ids = {p["episode_id"] for p in plan if p["controller"] == "HPA"}
        assert ppo_ids.isdisjoint(hpa_ids)


class TestFinalHoldoutUntouched:
    """Index38B: attempt1 (PPO repeat1) genuinely executed against the real
    ledger on disk -- `untouched` MUST correctly report False now and
    forever after (STEP5: never pretend it remains untouched). This
    reflects real, permanent, on-disk scientific state, not a test bug."""

    def test_holdout_status_reports_touched_by_real_attempt1_evidence(self):
        """Exact row counts are NOT asserted here -- the real final-eval
        ledgers keep growing as the live evaluation progresses through its
        remaining episodes; only the boolean 'has this ever been executed'
        signal is stable and meaningful to pin down."""
        status = eval_mod.verify_final_holdout_untouched()
        assert status["untouched"] is False
        assert status["appeared_in_training_ledger"] is False
        assert status["appeared_in_ppo_final_eval_ledger"] is True


class TestDryRunValidation:
    def test_dry_run_report_reflects_real_attempt1_evidence_and_is_not_blindly_ok(self):
        """Index38B: run_dry_run_validation()'s `ok` gate requires a
        PRISTINE (zero prior episodes) start -- correctly False now that
        attempt1 is real, on-disk evidence. A resume must go through
        verify_resumable_from_attempt1() instead of this blind
        fresh-start gate (STEP5)."""
        report = eval_mod.run_dry_run_validation()
        assert report["ok"] is False
        assert report["n_ppo_final_eval_episodes_executed_so_far"] >= 1
        assert report["traffic_generated"] is False
        assert report["kubernetes_mutation"] is False

    def test_dry_run_ledgers_are_distinct_from_training_ledger(self):
        assert str(eval_mod.PPO_FINAL_EVAL_LEDGER) != str(eval_mod.TRAINING_LEDGER)
        assert str(eval_mod.HPA_FINAL_EVAL_LEDGER) != str(eval_mod.TRAINING_LEDGER)
        assert str(eval_mod.PPO_FINAL_EVAL_LEDGER) != str(eval_mod.HPA_FINAL_EVAL_LEDGER)


class TestNoTuningFromResults:
    def test_module_never_writes_to_any_config_or_hyperparameter_file(self):
        source = Path(eval_mod.__file__).read_text()
        for forbidden in ("configs/rl.yaml", "configs/reward.yaml", "phase_16u_ppo_hyperparameters"):
            # Allowed to READ these paths (via rl_live_common), never open(...) them for writing.
            assert f'open("{forbidden}"' not in source
            assert f"open('{forbidden}'" not in source


class TestCliGating:
    def test_default_invocation_runs_dry_run_without_mutation(self):
        """Index38B: exit code now correctly reflects report["ok"]=False
        (real attempt1 evidence exists, see TestDryRunValidation) -- the
        important invariant checked here is that --dry-run NEVER touches
        the cluster or generates traffic, regardless of exit code."""
        proc = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate_ppo_vs_hpa.py")],
            capture_output=True, text=True, timeout=30, cwd=PROJECT_ROOT,
        )
        assert '"traffic_generated": false' in proc.stdout
        assert '"kubernetes_mutation": false' in proc.stdout

    def test_execute_flag_refuses(self):
        proc = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate_ppo_vs_hpa.py"), "--execute"],
            capture_output=True, text=True, timeout=30, cwd=PROJECT_ROOT,
        )
        assert proc.returncode == 2
        assert "Refusing" in proc.stderr


class TestLiveEvaluationImplementedButGated:
    """Index38A: run_live_evaluation() is now implemented (no longer a
    frozen NotImplementedError skeleton), but still requires a real
    cluster to execute -- verified here only at the CLI-gating level,
    never by actually invoking it."""

    def test_run_live_evaluation_is_callable_not_a_stub(self):
        import inspect
        assert callable(eval_mod.run_live_evaluation)
        src = inspect.getsource(eval_mod.run_live_evaluation)
        assert "NotImplementedError" not in src

    def test_execute_without_confirmation_flag_is_refused(self):
        proc = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate_ppo_vs_hpa.py"), "--execute"],
            capture_output=True, text=True, timeout=30, cwd=PROJECT_ROOT,
        )
        assert proc.returncode == 2
        assert "i-confirm-live-cluster-interaction" in proc.stderr

    def test_resolve_holdout_profile_metadata_matches_frozen_hash(self):
        meta = eval_mod.resolve_holdout_profile_metadata()
        assert meta.sha256 == "308f4f90ea3cea97f1927cec97efeaa652c4424e9fe5bcea421358f3425e3409"
        assert meta.seed == 777
        assert meta.duration_seconds == 1500.0

    def test_run_ppo_episode_and_run_hpa_episode_exist_with_expected_signature(self):
        import inspect
        ppo_sig = inspect.signature(eval_mod.run_ppo_episode)
        hpa_sig = inspect.signature(eval_mod.run_hpa_episode)
        assert list(ppo_sig.parameters) == ["repeat_index", "sequence_index", "model", "env_config", "profile_meta"]
        assert list(hpa_sig.parameters) == ["repeat_index", "sequence_index", "env_config", "profile_meta"]

    def test_no_learn_call_anywhere_in_module(self):
        """AST-based, not substring-based: the module's own docstrings
        legitimately mention 'model.learn()' in prose (documenting that it
        is NEVER called) -- a naive substring check would false-positive
        on that prose. This walks actual Call nodes only."""
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(eval_mod))
        learn_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "learn"
        ]
        assert learn_calls == []

    def test_no_checkpoint_save_or_session_020_path(self):
        import inspect
        src = inspect.getsource(eval_mod)
        assert "session_020" not in src
        assert "model.save(" not in src


def _sample(t, target_up=True, desired=1, ready=1, cpu=None, p95=None):
    return {
        "time_s": t, "request_rate_rps": None, "error_rate_rps": None,
        "latency_p95_ms": p95, "requests_in_progress": None, "cpu_cores": cpu,
        "memory_bytes": None, "target_up": target_up,
        "replicas_desired": desired, "replicas_current": desired,
        "replicas_ready": ready, "replicas_available": ready,
        "latency_p50_ms": None, "latency_p99_ms": None, "success_rate_rps": None,
    }


class TestClassifyTermination:
    """Index38B STEP3B / STEP6: a safety-terminated episode (e.g. the real
    PPO1: 24/50 steps, 4 consecutive target_up=False samples) must never be
    silently reported as a normal full-duration completion."""

    def test_24_of_50_with_trailing_invalid_run_is_classified_safety_terminated(self):
        samples = [_sample(i * 30, target_up=True) for i in range(21)]
        samples += [_sample((21 + i) * 30, target_up=False) for i in range(4)]  # 24 total steps + t=0
        result = eval_mod._classify_termination(samples, nominal_steps=50, env_terminated=True)
        assert result["episode_status"] == "terminated_safety"
        assert result["termination_reason"] == "consecutive_invalid_steps_limit_reached"
        assert result["full_holdout_completed"] is False
        assert result["valid_for_primary_comparison"] is False
        assert result["observed_decision_steps"] == 24
        assert result["nominal_decision_steps"] == 50

    def test_full_duration_completion_is_classified_as_such(self):
        samples = [_sample(i * 30, target_up=True) for i in range(51)]  # t=0 + 50 steps
        result = eval_mod._classify_termination(samples, nominal_steps=50, env_terminated=False)
        assert result["episode_status"] == "completed_full_duration"
        assert result["termination_reason"] == "none"
        assert result["full_holdout_completed"] is True
        assert result["valid_for_primary_comparison"] is True
        assert result["observed_decision_steps"] == 50


class TestScalingEventsUseEffectiveReplicas:
    """Index38B STEP3B: scaling_events must count DESIRED-replica changes
    (actual/effective scaling decisions), never READY-replica flicker
    caused by unrelated pod restarts."""

    def test_flickering_ready_with_constant_desired_counts_zero_scaling_events(self):
        samples = [
            _sample(0, desired=1, ready=1), _sample(30, desired=1, ready=0),
            _sample(60, desired=1, ready=1), _sample(90, desired=1, ready=0),
        ]
        result = eval_mod._summarize_arm(
            "PPO", 1, "final_eval_ppo_repeat1", samples, [],
            eval_mod.resolve_holdout_profile_metadata(), nominal_steps=50, env_terminated=False,
        )
        assert result["scaling_events"] == 0

    def test_genuine_desired_replica_change_is_counted(self):
        samples = [
            _sample(0, desired=1, ready=1), _sample(30, desired=1, ready=1),
            _sample(60, desired=2, ready=1), _sample(90, desired=2, ready=2),
        ]
        result = eval_mod._summarize_arm(
            "PPO", 1, "final_eval_ppo_repeat1", samples, [],
            eval_mod.resolve_holdout_profile_metadata(), nominal_steps=50, env_terminated=False,
        )
        assert result["scaling_events"] == 1


class TestResumeGate:
    def test_resume_refused_when_no_prior_evidence_exists(self, tmp_path, monkeypatch):
        monkeypatch.setattr(eval_mod, "PPO_FINAL_EVAL_LEDGER", tmp_path / "ppo.jsonl")
        monkeypatch.setattr(eval_mod, "HPA_FINAL_EVAL_LEDGER", tmp_path / "hpa.jsonl")
        result = eval_mod.verify_resumable_from_attempt1({"ppo_repeats": [1], "hpa_repeats": []})
        assert result["resumable"] is False
        assert result["ppo_rows_found"] == 0

    def test_resume_permitted_when_evidence_matches_exact_expected_identity(self, tmp_path, monkeypatch):
        ppo_ledger = tmp_path / "ppo.jsonl"
        monkeypatch.setattr(eval_mod, "PPO_FINAL_EVAL_LEDGER", ppo_ledger)
        monkeypatch.setattr(eval_mod, "HPA_FINAL_EVAL_LEDGER", tmp_path / "hpa.jsonl")
        eval_mod.append_row(ppo_ledger, {
            "episode_id": "final_eval_ppo_repeat1", "repeat_index": 1,
            "workload_hash": "308f4f90ea3cea97f1927cec97efeaa652c4424e9fe5bcea421358f3425e3409",
        })
        result = eval_mod.verify_resumable_from_attempt1({"ppo_repeats": [1], "hpa_repeats": []})
        assert result["resumable"] is True
        assert result["ppo_rows_found"] == 1

    def test_resume_refused_when_prior_evidence_has_wrong_workload_hash(self, tmp_path, monkeypatch):
        ppo_ledger = tmp_path / "ppo.jsonl"
        monkeypatch.setattr(eval_mod, "PPO_FINAL_EVAL_LEDGER", ppo_ledger)
        monkeypatch.setattr(eval_mod, "HPA_FINAL_EVAL_LEDGER", tmp_path / "hpa.jsonl")
        eval_mod.append_row(ppo_ledger, {
            "episode_id": "final_eval_ppo_repeat1", "repeat_index": 1, "workload_hash": "WRONG_HASH",
        })
        result = eval_mod.verify_resumable_from_attempt1({"ppo_repeats": [1], "hpa_repeats": []})
        assert result["resumable"] is False
        assert result["prior_ppo_evidence_matches_expected_identity"] is False

    def test_resume_does_not_mutate_existing_ledger_rows(self, tmp_path, monkeypatch):
        ppo_ledger = tmp_path / "ppo.jsonl"
        monkeypatch.setattr(eval_mod, "PPO_FINAL_EVAL_LEDGER", ppo_ledger)
        monkeypatch.setattr(eval_mod, "HPA_FINAL_EVAL_LEDGER", tmp_path / "hpa.jsonl")
        eval_mod.append_row(ppo_ledger, {
            "episode_id": "final_eval_ppo_repeat1", "repeat_index": 1,
            "workload_hash": "308f4f90ea3cea97f1927cec97efeaa652c4424e9fe5bcea421358f3425e3409",
        })
        before = ppo_ledger.read_text()
        eval_mod.verify_resumable_from_attempt1({"ppo_repeats": [1], "hpa_repeats": []})
        assert ppo_ledger.read_text() == before

    def test_verify_final_holdout_untouched_correctly_reports_false_after_real_rows_exist(self, tmp_path, monkeypatch):
        ppo_ledger = tmp_path / "ppo.jsonl"
        monkeypatch.setattr(eval_mod, "PPO_FINAL_EVAL_LEDGER", ppo_ledger)
        monkeypatch.setattr(eval_mod, "HPA_FINAL_EVAL_LEDGER", tmp_path / "hpa.jsonl")
        monkeypatch.setattr(eval_mod, "TRAINING_LEDGER", tmp_path / "training.jsonl")
        eval_mod.append_row(ppo_ledger, {"episode_id": "final_eval_ppo_repeat1"})
        status = eval_mod.verify_final_holdout_untouched()
        assert status["untouched"] is False
        assert status["appeared_in_ppo_final_eval_ledger"] is True


class TestIndex38CHpa2RetryResumeGate:
    """Index38C: HPA repeat2 physical attempt1 failed (pure tooling/infra
    failure -- DrainQueryError during post-workload drain, no ledger row
    was ever written for it). Attempt2 resumes at the SAME logical
    sequence_index=4, expecting PPO1, PPO2, HPA1 already committed and nothing
    for HPA repeat2 yet."""

    def _seed_ledgers(self, tmp_path, monkeypatch, ppo_repeats, hpa_repeats):
        ppo_ledger, hpa_ledger = tmp_path / "ppo.jsonl", tmp_path / "hpa.jsonl"
        monkeypatch.setattr(eval_mod, "PPO_FINAL_EVAL_LEDGER", ppo_ledger)
        monkeypatch.setattr(eval_mod, "HPA_FINAL_EVAL_LEDGER", hpa_ledger)
        expected_hash = "308f4f90ea3cea97f1927cec97efeaa652c4424e9fe5bcea421358f3425e3409"
        for r in ppo_repeats:
            eval_mod.append_row(ppo_ledger, {"episode_id": f"final_eval_ppo_repeat{r}", "repeat_index": r, "workload_hash": expected_hash})
        for r in hpa_repeats:
            eval_mod.append_row(hpa_ledger, {"episode_id": f"final_eval_hpa_repeat{r}", "repeat_index": r, "workload_hash": expected_hash})
        return ppo_ledger, hpa_ledger

    def test_hpa2_attempt2_resume_permitted_with_ppo1_ppo2_hpa1_committed(self, tmp_path, monkeypatch):
        self._seed_ledgers(tmp_path, monkeypatch, ppo_repeats=[1, 2], hpa_repeats=[1])
        result = eval_mod.verify_resumable_from_attempt1({"ppo_repeats": [1, 2], "hpa_repeats": [1]})
        assert result["resumable"] is True

    def test_hpa2_attempt2_resume_refused_if_hpa2_row_already_exists(self, tmp_path, monkeypatch):
        """A real HPA2 row existing already would mean attempt1 actually
        succeeded (contradicting the failure evidence) -- the gate must
        refuse rather than silently proceeding to a THIRD attempt."""
        self._seed_ledgers(tmp_path, monkeypatch, ppo_repeats=[1, 2], hpa_repeats=[1, 2])
        result = eval_mod.verify_resumable_from_attempt1({"ppo_repeats": [1, 2], "hpa_repeats": [1]})
        assert result["resumable"] is False

    def test_hpa2_attempt2_resume_refused_if_ppo2_missing(self, tmp_path, monkeypatch):
        """Guards against accidentally resuming from the wrong sequence
        position, which could otherwise lead to PPO2 (or PPO1) being
        silently re-run -- explicitly prohibited (outcome-aware retry)."""
        self._seed_ledgers(tmp_path, monkeypatch, ppo_repeats=[1], hpa_repeats=[1])
        result = eval_mod.verify_resumable_from_attempt1({"ppo_repeats": [1, 2], "hpa_repeats": [1]})
        assert result["resumable"] is False

    def test_drain_query_error_is_a_distinguishable_exception_type(self):
        """STEP4: the supervisor's telemetry-failure classification relies
        on DrainQueryError being a distinct, catchable type (not a bare
        RuntimeError/string) -- verifies the type this phase's fix depends
        on actually exists with this exact name/module."""
        from scripts.traffic_orchestrator import DrainQueryError
        assert issubclass(DrainQueryError, Exception)
