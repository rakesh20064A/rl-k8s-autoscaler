"""tests/unit/test_evaluate_v2_campaign.py

V2 Phase 5B PART A: offline tests for scripts/evaluate_v2_campaign.py.
Exercises the full --dry-run control flow (plan construction, Latin-square
order, ledger/resume, tooling-failure retry-once, the fail-closed
hpa_v2_cpu_only arm, and campaign-level cleanup/settle calls) entirely
against injected fakes -- no subprocess, no cluster, no sleep.
"""

from __future__ import annotations

import json

import pytest

import scripts.evaluate_v2_campaign as campaign_mod


@pytest.fixture()
def preregistration():
    return campaign_mod.load_preregistration()


class TestRunPlanConstruction:
    def test_27_runs_total(self, preregistration):
        runs = campaign_mod.build_run_plan(preregistration)
        assert len(runs) == 27

    def test_9_blocks_of_3_arms(self, preregistration):
        runs = campaign_mod.build_run_plan(preregistration)
        blocks = {r.block_index for r in runs}
        assert len(blocks) == 9
        for b in blocks:
            arms_in_block = {r.arm_id for r in runs if r.block_index == b}
            assert arms_in_block == set(campaign_mod.ARM_IDS)

    def test_3_repeats_per_profile_per_arm(self, preregistration):
        runs = campaign_mod.build_run_plan(preregistration)
        from collections import Counter

        counts = Counter((r.profile, r.arm_id) for r in runs)
        assert all(c == 3 for c in counts.values())

    def test_cyclic_latin_square_rotation(self, preregistration):
        runs = campaign_mod.build_run_plan(preregistration)
        for block_index in sorted({r.block_index for r in runs}):
            order = [r.arm_id for r in runs if r.block_index == block_index]
            shift = block_index % len(campaign_mod.ARM_IDS)
            expected = list(campaign_mod.ARM_IDS[shift:]) + list(campaign_mod.ARM_IDS[:shift])
            assert order == expected

    def test_every_arm_appears_once_per_position_across_any_3_consecutive_blocks(self, preregistration):
        runs = campaign_mod.build_run_plan(preregistration)
        by_block = {}
        for r in runs:
            by_block.setdefault(r.block_index, {})[r.position_in_block] = r.arm_id
        for start in range(0, 9, 3):
            for position in range(3):
                arms_at_position = {by_block[start + i][position] for i in range(3)}
                assert arms_at_position == set(campaign_mod.ARM_IDS)

    def test_run_ids_are_unique(self, preregistration):
        runs = campaign_mod.build_run_plan(preregistration)
        assert len({r.run_id for r in runs}) == 27


class TestWriteRunPlan:
    def test_plan_written_before_first_run(self, tmp_path, preregistration):
        runs = campaign_mod.build_run_plan(preregistration)
        plan_path = tmp_path / "RUN_PLAN.json"
        plan = campaign_mod.write_run_plan(runs, "test_campaign", preregistration, plan_path)
        assert plan_path.exists()
        assert plan["n_runs"] == 27
        with open(plan_path, encoding="utf-8") as f:
            reloaded = json.load(f)
        assert reloaded["campaign_id"] == "test_campaign"
        assert len(reloaded["runs"]) == 27


class TestEvalProfileFirewall:
    def test_resolves_a_real_eval_profile(self):
        meta = campaign_mod.resolve_eval_profile_metadata("eval_cpu_step")
        assert meta.name == "eval_cpu_step"
        assert meta.duration_seconds == 1200.0

    def test_refuses_a_training_profile_name(self):
        with pytest.raises(campaign_mod.EvalProfileFirewallViolation):
            campaign_mod.resolve_eval_profile_metadata("constant_cpu")

    def test_refuses_an_unknown_name(self):
        with pytest.raises(campaign_mod.EvalProfileFirewallViolation):
            campaign_mod.resolve_eval_profile_metadata("not_a_real_profile")


class TestDryRunFullCampaign:
    def test_full_campaign_completes(self, tmp_path):
        ctx = campaign_mod.build_dry_run_context("dr1")
        result = campaign_mod.run_campaign("dr1", ctx, output_root=tmp_path)
        assert result["ok"] is True
        assert result["status"] == "COMPLETE"

    def test_ledger_has_27_valid_entries(self, tmp_path):
        """V2 Phase 5C / CORR-62: CONTROLLER_CPU_ONLY_HPA_V2 now exists in
        controller_exclusivity.py (post-training amendment) -- all 3 arms,
        including hpa_v2_cpu_only, succeed by default."""
        ctx = campaign_mod.build_dry_run_context("dr2")
        campaign_mod.run_campaign("dr2", ctx, output_root=tmp_path)
        ledger = campaign_mod.load_ledger(tmp_path / "dr2" / "ledger.jsonl")
        assert len(ledger) == 27
        statuses = {row["status"] for row in ledger}
        assert statuses == {"VALID"}

    def test_cpu_only_arm_succeeds_now_that_the_phase5c_constant_exists(self, tmp_path):
        ctx = campaign_mod.build_dry_run_context("dr3")
        campaign_mod.run_campaign("dr3", ctx, output_root=tmp_path)
        ledger = campaign_mod.load_ledger(tmp_path / "dr3" / "ledger.jsonl")
        cpu_only_rows = [r for r in ledger if r["arm_id"] == "hpa_v2_cpu_only"]
        assert len(cpu_only_rows) == 9
        assert all(r["status"] == "VALID" for r in cpu_only_rows)
        assert all(r["attempt_number"] == 1 for r in cpu_only_rows)

    def test_cpu_only_arm_still_fails_closed_if_the_constant_is_ever_absent(self, tmp_path, monkeypatch):
        """Regression coverage for the pre-Phase-5C state: if
        CONTROLLER_CPU_ONLY_HPA_V2 were ever removed/renamed, the campaign
        must still fail closed for that arm, never silently skip it or
        crash the whole campaign."""
        import autoscaler.src.rl_env.controller_exclusivity as ce

        monkeypatch.delattr(ce, "CONTROLLER_CPU_ONLY_HPA_V2", raising=False)
        ctx = campaign_mod.build_dry_run_context("dr3b")
        campaign_mod.run_campaign("dr3b", ctx, output_root=tmp_path)
        ledger = campaign_mod.load_ledger(tmp_path / "dr3b" / "ledger.jsonl")
        cpu_only_rows = [r for r in ledger if r["arm_id"] == "hpa_v2_cpu_only"]
        assert len(cpu_only_rows) == 9
        assert all(r["status"] == "FAILED_CLOSED" for r in cpu_only_rows)
        assert all(r["attempt_number"] == 1 for r in cpu_only_rows)

    def test_run_status_ends_complete_with_no_current_run(self, tmp_path):
        ctx = campaign_mod.build_dry_run_context("dr4")
        campaign_mod.run_campaign("dr4", ctx, output_root=tmp_path)
        with open(tmp_path / "dr4" / "run_status.json", encoding="utf-8") as f:
            status = json.load(f)
        assert status["status"] == "COMPLETE"
        assert status["current_run"] is None

    def test_resume_skips_already_terminal_runs(self, tmp_path):
        ctx1 = campaign_mod.build_dry_run_context("dr5")
        campaign_mod.run_campaign("dr5", ctx1, output_root=tmp_path)
        ledger_path = tmp_path / "dr5" / "ledger.jsonl"
        n_rows_first = len(campaign_mod.load_ledger(ledger_path))

        ctx2 = campaign_mod.build_dry_run_context("dr5")
        result2 = campaign_mod.run_campaign("dr5", ctx2, output_root=tmp_path)
        assert result2["ok"] is True
        n_rows_second = len(campaign_mod.load_ledger(ledger_path))
        assert n_rows_second == n_rows_first  # nothing re-run
        assert len(ctx2.orchestrator.started) == 0  # no traffic started on the resumed (no-op) pass

    def test_cleanup_and_settle_called_once_per_successful_run(self, tmp_path):
        ctx = campaign_mod.build_dry_run_context("dr6")
        campaign_mod.run_campaign("dr6", ctx, output_root=tmp_path)
        # All 27 runs are now VALID (CONTROLLER_CPU_ONLY_HPA_V2 exists).
        assert len(ctx.settle_fn.calls) == 27
        assert all(c == 120.0 for c in ctx.settle_fn.calls)

    def test_traffic_started_and_stopped_for_every_run(self, tmp_path):
        ctx = campaign_mod.build_dry_run_context("dr7")
        campaign_mod.run_campaign("dr7", ctx, output_root=tmp_path)
        assert len(ctx.orchestrator.started) == 27
        assert len(ctx.orchestrator.stopped) == 27
        assert ctx.orchestrator.drained == 27


class TestToolingFailureRetryPolicy:
    def test_tooling_failure_retried_exactly_once_then_succeeds(self, tmp_path):
        ctx = campaign_mod.build_dry_run_context("dr8")
        calls = {"n": 0}
        real_ppo_fn = ctx.ppo_episode_fn

        def flaky_ppo(profile_meta, episode_id):
            calls["n"] += 1
            if calls["n"] == 1:
                raise campaign_mod.ToolingFailureError("simulated transient failure")
            return real_ppo_fn(profile_meta, episode_id)

        ctx.ppo_episode_fn = flaky_ppo
        result = campaign_mod.run_campaign("dr8", ctx, output_root=tmp_path)
        assert result["ok"] is True
        ledger = campaign_mod.load_ledger(tmp_path / "dr8" / "ledger.jsonl")
        first_ppo_run_rows = [r for r in ledger if r["run_id"] == "eval_cpu_step__repeat0__ppo_v2"]
        assert len(first_ppo_run_rows) == 2
        assert first_ppo_run_rows[0]["status"] == "TOOLING_FAILURE"
        assert first_ppo_run_rows[0]["attempt_number"] == 1
        assert first_ppo_run_rows[1]["status"] == "VALID"
        assert first_ppo_run_rows[1]["attempt_number"] == 2

    def test_tooling_failure_on_both_attempts_stops_the_campaign(self, tmp_path):
        ctx = campaign_mod.build_dry_run_context("dr9")

        def always_fails(profile_meta, episode_id):
            raise campaign_mod.ToolingFailureError("simulated persistent failure")

        ctx.ppo_episode_fn = always_fails
        result = campaign_mod.run_campaign("dr9", ctx, output_root=tmp_path)
        assert result["ok"] is False
        assert result["status"] == "FAILED"
        ledger = campaign_mod.load_ledger(tmp_path / "dr9" / "ledger.jsonl")
        # Only the first run's 2 attempts were made before the campaign stopped.
        assert len(ledger) == 2
        assert ledger[0]["status"] == "TOOLING_FAILURE"
        assert ledger[1]["status"] == "FAILED_AFTER_RETRY"
        with open(tmp_path / "dr9" / "run_status.json", encoding="utf-8") as f:
            status = json.load(f)
        assert status["status"] == "FAILED"

    def test_safety_termination_is_never_retried(self, tmp_path):
        ctx = campaign_mod.build_dry_run_context("dr10")
        calls = {"n": 0}

        def terminating_ppo(profile_meta, episode_id):
            calls["n"] += 1
            raise campaign_mod.SafetyTerminationError("simulated memory safety breach")

        ctx.ppo_episode_fn = terminating_ppo
        result = campaign_mod.run_campaign("dr10", ctx, output_root=tmp_path)
        assert result["ok"] is True  # safety termination is a counted outcome, not a campaign-stopping failure
        ledger = campaign_mod.load_ledger(tmp_path / "dr10" / "ledger.jsonl")
        ppo_rows = [r for r in ledger if r["arm_id"] == "ppo_v2"]
        # All 9 ppo_v2 runs across the campaign terminate on safety grounds,
        # each exactly once -- never retried (attempt_number always 1).
        assert len(ppo_rows) == 9
        assert all(r["status"] == "SAFETY_TERMINATION" for r in ppo_rows)
        assert all(r["attempt_number"] == 1 for r in ppo_rows)
        assert calls["n"] == 9  # one call per ppo_v2 run, never a retry of any single one


class TestPreflightFailureIsToolingFailure:
    def test_failed_preflight_is_treated_as_tooling_failure(self, tmp_path):
        ctx = campaign_mod.build_dry_run_context("dr11")
        ctx.preflight_fn = lambda: {"ok": False, "reason": "simulated"}
        result = campaign_mod.run_campaign("dr11", ctx, output_root=tmp_path)
        assert result["ok"] is False
        ledger = campaign_mod.load_ledger(tmp_path / "dr11" / "ledger.jsonl")
        assert ledger[0]["status"] == "TOOLING_FAILURE"


class TestEndToEndDryCampaignToAnalysis:
    """V2 Phase 5C-A: the full runner -> analysis pipeline, entirely
    offline. A 27-run dry campaign (with the fake episodes now emitting the
    real EpisodeSample/RunSeries shape) feeds its `normalized_series.json`
    files directly into analyze_v2_evaluation.py with no manual conversion
    step, producing tables, figures, and pre-registered comparison
    verdicts."""

    def test_full_pipeline_produces_tables_figures_and_verdicts(self, tmp_path):
        import scripts.analyze_v2_evaluation as av

        ctx = campaign_mod.build_dry_run_context("e2e")
        result = campaign_mod.run_campaign("e2e", ctx, output_root=tmp_path)
        assert result["ok"] is True

        campaign_dir = tmp_path / "e2e"
        run_metrics = av.analyze_evaluation_campaign(campaign_dir)
        # All 27 runs are VALID now (CONTROLLER_CPU_ONLY_HPA_V2 exists,
        # V2 Phase 5C / CORR-62) and all carry normalized_series.json.
        assert len(run_metrics) == 27
        assert {m["arm_id"] for m in run_metrics} == {"ppo_v2", "hpa_v2_cpu_memory", "hpa_v2_cpu_only"}

        aggregated = av.aggregate_per_profile_arm(run_metrics)
        assert len(aggregated) == 9  # 3 profiles x 3 arms with data

        output_dir = tmp_path / "analysis"
        tables = av.write_evaluation_tables(aggregated, output_dir)
        assert tables["csv"].exists()
        assert tables["md"].exists()
        figures = av.plot_evaluation_figures(aggregated, output_dir)
        assert len(figures) == len(av.PRIMARY_METRICS)
        assert all(p.exists() for p in figures)

        comparison = av.compare_arms_per_profile(run_metrics, "ppo_v2", "hpa_v2_cpu_memory")
        for profile in {m["profile"] for m in run_metrics}:
            # The fake ppo_v2 episode is deterministically lower-latency
            # than the fake hpa_v2_cpu_memory episode on EVERY repeat --
            # the pre-registered "better in all 3 repeats" rule must find
            # ppo_v2 the decidable winner on mean_p95_latency_ms.
            assert comparison[profile]["mean_p95_latency_ms"] == "ppo_v2_better"


class _FakeKubectlForLiveContext:
    def __init__(self, existing_hpa_names):
        self._existing = list(existing_hpa_names)
        self.delete_hpa_calls = 0

    def list_hpa_names(self):
        return list(self._existing)

    def delete_hpa(self):
        self.delete_hpa_calls += 1
        self._existing = []


class _FakeBackendForLiveContext:
    def __init__(self, hpa_name=None, hpa_manifest_path=None, min_replicas=None, initial_replicas=None, existing_hpa_names=()):
        self._kubectl = _FakeKubectlForLiveContext(existing_hpa_names)


class TestBuildLiveContextRemovesConflictingHpaBeforeStart:
    """V2 Phase 5F / CORR-63: `build_live_context()`'s ppo_v2 and
    hpa_v2_cpu_only wrappers must each remove a conflicting HPA left
    behind by a prior arm's own canonical-idle-state cleanup, BEFORE
    calling into their own `acquire_controller(...)` (which fails closed
    if any such HPA is present) -- reproduces the exact live failures
    found in V2 Phase 5D/5F (ppo_v2 blocked by demo-app-hpa-v2;
    hpa_v2_cpu_only blocked by demo-app-hpa-v2 left over from the
    hpa_v2_cpu_memory arm's own cleanup)."""

    def _build_ctx_with_fake_backend(self, monkeypatch, existing_hpa_names, ppo_result=None, cpu_only_result=None):
        import scripts.evaluate_v2_campaign as campaign_mod

        captured = {}

        def fake_backend_factory(**kwargs):
            backend = _FakeBackendForLiveContext(existing_hpa_names=existing_hpa_names, **{k: v for k, v in kwargs.items() if k in ("hpa_name", "hpa_manifest_path", "min_replicas", "initial_replicas")})
            captured["backend"] = backend
            return backend

        monkeypatch.setattr("autoscaler.src.rl_env.live_backend.LiveKubernetesBackend", fake_backend_factory)
        monkeypatch.setattr(campaign_mod, "run_ppo_v2_eval_episode", lambda *a, **k: ppo_result or {"ok": True})
        monkeypatch.setattr(campaign_mod, "run_hpa_v2_cpu_only_eval_episode", lambda *a, **k: cpu_only_result or {"ok": True})

        ctx = campaign_mod.build_live_context(__import__("pathlib").Path("model.zip"), 0.05)
        return ctx, captured

    def test_ppo_episode_deletes_hpa_when_present(self, monkeypatch):
        import scripts.evaluate_v2_campaign as campaign_mod

        ctx, captured = self._build_ctx_with_fake_backend(monkeypatch, existing_hpa_names=["demo-app-hpa-v2"])
        profile_meta = campaign_mod.resolve_eval_profile_metadata("eval_cpu_step")
        ctx.ppo_episode_fn(profile_meta, "e1")
        assert captured["backend"]._kubectl.delete_hpa_calls == 1

    def test_ppo_episode_is_a_noop_when_no_hpa_present(self, monkeypatch):
        import scripts.evaluate_v2_campaign as campaign_mod

        ctx, captured = self._build_ctx_with_fake_backend(monkeypatch, existing_hpa_names=[])
        profile_meta = campaign_mod.resolve_eval_profile_metadata("eval_cpu_step")
        ctx.ppo_episode_fn(profile_meta, "e1")
        assert captured["backend"]._kubectl.delete_hpa_calls == 0

    def test_cpu_only_episode_deletes_conflicting_hpa_when_present(self, monkeypatch):
        """Reproduces the exact V2 Phase 5F failure: demo-app-hpa-v2 (the
        hpa_v2_cpu_memory arm's own HPA) left present blocks the
        hpa_v2_cpu_only arm's own acquire_controller call."""
        import scripts.evaluate_v2_campaign as campaign_mod

        ctx, captured = self._build_ctx_with_fake_backend(monkeypatch, existing_hpa_names=["demo-app-hpa-v2"])
        profile_meta = campaign_mod.resolve_eval_profile_metadata("eval_cpu_step")
        ctx.cpu_only_episode_fn(profile_meta, "e1")
        assert captured["backend"]._kubectl.delete_hpa_calls == 1

    def test_cpu_only_episode_is_a_noop_when_only_its_own_hpa_present(self, monkeypatch):
        import scripts.evaluate_v2_campaign as campaign_mod

        ctx, captured = self._build_ctx_with_fake_backend(monkeypatch, existing_hpa_names=["demo-app-hpa-v2-cpu"])
        profile_meta = campaign_mod.resolve_eval_profile_metadata("eval_cpu_step")
        ctx.cpu_only_episode_fn(profile_meta, "e1")
        assert captured["backend"]._kubectl.delete_hpa_calls == 0

    def test_cpu_only_episode_is_a_noop_when_no_hpa_present(self, monkeypatch):
        import scripts.evaluate_v2_campaign as campaign_mod

        ctx, captured = self._build_ctx_with_fake_backend(monkeypatch, existing_hpa_names=[])
        profile_meta = campaign_mod.resolve_eval_profile_metadata("eval_cpu_step")
        ctx.cpu_only_episode_fn(profile_meta, "e1")
        assert captured["backend"]._kubectl.delete_hpa_calls == 0


class TestRerunAfterFix:
    """V2 Phase 5F / CORR-63: the explicit --rerun-after-fix escape hatch
    for a run stuck at FAILED_AFTER_RETRY after a confirmed tooling-side
    fix -- appends a new attempt, never overwrites the prior failed rows,
    and tags every appended row with the CORR id."""

    def _seed_failed_after_retry(self, tmp_path, campaign_id="rerun1"):
        ctx = campaign_mod.build_dry_run_context(campaign_id)
        ctx.ppo_episode_fn = lambda *a, **k: (_ for _ in ()).throw(campaign_mod.ToolingFailureError("simulated bug"))
        campaign_mod.run_campaign(campaign_id, ctx, output_root=tmp_path)
        ledger_before = campaign_mod.load_ledger(tmp_path / campaign_id / "ledger.jsonl")
        failed_rows = [r for r in ledger_before if r["run_id"] == "eval_cpu_step__repeat0__ppo_v2"]
        assert len(failed_rows) == 2
        assert failed_rows[-1]["status"] == "FAILED_AFTER_RETRY"
        return ledger_before

    def test_rerun_appends_a_new_attempt_without_touching_prior_rows(self, tmp_path):
        campaign_id = "rerun1"
        ledger_before = self._seed_failed_after_retry(tmp_path, campaign_id)

        ctx2 = campaign_mod.build_dry_run_context(campaign_id)  # the "fix": fresh fake context, ppo_episode_fn now succeeds
        result = campaign_mod.rerun_run_after_fix(
            campaign_id, "eval_cpu_step__repeat0__ppo_v2", ctx2, "CORR-63", output_root=tmp_path,
        )
        assert result["ok"] is True
        assert result["final_status"] == "VALID"
        assert result["corr_id"] == "CORR-63"

        ledger_after = campaign_mod.load_ledger(tmp_path / campaign_id / "ledger.jsonl")
        assert ledger_after[:2] == ledger_before  # prior 2 rows byte-identical, never overwritten
        assert len(ledger_after) == 3
        new_row = ledger_after[2]
        assert new_row["attempt_number"] == 3
        assert new_row["status"] == "VALID"
        assert new_row["corr_id"] == "CORR-63"
        assert new_row["rerun_after_fix"] is True

    def test_rerun_on_a_run_with_no_prior_attempts_raises(self, tmp_path):
        campaign_id = "rerun2"
        campaign_dir = tmp_path / campaign_id
        runs = campaign_mod.build_run_plan(campaign_mod.load_preregistration())
        campaign_mod.write_run_plan(runs, campaign_id, campaign_mod.load_preregistration(), campaign_dir / "RUN_PLAN.json")
        # RUN_PLAN.json exists (the run_id IS a real planned run) but no
        # ledger row has ever been written for it -- nothing to rerun yet.
        ctx2 = campaign_mod.build_dry_run_context(campaign_id)
        with pytest.raises(ValueError, match="no prior ledger rows"):
            campaign_mod.rerun_run_after_fix(campaign_id, "eval_cpu_step__repeat0__ppo_v2", ctx2, "CORR-99", output_root=tmp_path)

    def test_rerun_unknown_run_id_raises(self, tmp_path):
        campaign_id = "rerun3"
        ledger_before = self._seed_failed_after_retry(tmp_path, campaign_id)
        assert ledger_before
        ctx2 = campaign_mod.build_dry_run_context(campaign_id)
        with pytest.raises(ValueError, match="not found"):
            campaign_mod.rerun_run_after_fix(campaign_id, "not_a_real_run_id", ctx2, "CORR-63", output_root=tmp_path)

    def test_rerun_that_fails_again_is_recorded_as_failed_after_retry_again(self, tmp_path):
        campaign_id = "rerun4"
        self._seed_failed_after_retry(tmp_path, campaign_id)

        ctx2 = campaign_mod.build_dry_run_context(campaign_id)
        ctx2.ppo_episode_fn = lambda *a, **k: (_ for _ in ()).throw(campaign_mod.ToolingFailureError("still broken"))
        result = campaign_mod.rerun_run_after_fix(
            campaign_id, "eval_cpu_step__repeat0__ppo_v2", ctx2, "CORR-63", output_root=tmp_path,
        )
        assert result["ok"] is False
        assert result["final_status"] == "FAILED_AFTER_RETRY"

        ledger_after = campaign_mod.load_ledger(tmp_path / campaign_id / "ledger.jsonl")
        run_rows = [r for r in ledger_after if r["run_id"] == "eval_cpu_step__repeat0__ppo_v2"]
        assert len(run_rows) == 4  # original 2 + this rerun's own 2 attempts
        assert [r["attempt_number"] for r in run_rows] == [1, 2, 3, 4]
        assert all(r.get("corr_id") == "CORR-63" for r in run_rows[2:])

    def test_main_refuses_rerun_after_fix_without_execute(self):
        rc = campaign_mod.main(["--rerun-after-fix", "some_run_id", "--corr-id", "CORR-63"])
        assert rc == 2

    def test_main_refuses_rerun_after_fix_without_corr_id(self):
        rc = campaign_mod.main([
            "--execute", "--i-confirm-live-cluster-interaction", "--rerun-after-fix", "some_run_id",
            "--epsilon-calibration-path", "/dev/null",
        ])
        assert rc == 2
