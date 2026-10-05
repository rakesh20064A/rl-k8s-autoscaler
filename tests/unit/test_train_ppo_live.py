"""tests/unit/test_train_ppo_live.py

Phase 16V: tests for scripts/train_ppo_live.py. Only the --dry-run path is
exercised (as the task requires) -- run_live_training() is never invoked
here; it is covered only by static/structural assertions (it exists, it is
gated behind two explicit flags, it is not reachable from --dry-run).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import scripts.train_ppo_live as train_mod

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestDryRunValidation:
    def test_dry_run_report_is_ok(self):
        report = train_mod.run_dry_run_validation(seed=42)
        assert report["ok"] is True
        assert report["traffic_generated"] is False
        assert report["kubernetes_mutation"] is False

    def test_dry_run_rejects_non_frozen_seed(self):
        report = train_mod.run_dry_run_validation(seed=999)
        assert report["seed_is_frozen_live_training_seed"] is False
        assert report["ok"] is False

    def test_dry_run_uses_scripted_backend_never_live_backend(self):
        source = Path(train_mod.__file__).read_text()
        # Isolate just the run_dry_run_validation() function body.
        dry_run_src = source.split("def run_dry_run_validation")[1].split("\ndef ")[0]
        assert "LiveKubernetesBackend" not in dry_run_src
        assert "ScriptedBackend" in dry_run_src

    def test_dry_run_reports_all_required_dependency_versions(self):
        report = train_mod.run_dry_run_validation(seed=42)
        deps = report["dependency_versions"]
        for name in ("python", "stable_baselines3", "gymnasium", "torch", "numpy", "pandas", "kubernetes"):
            assert name in deps
            assert deps[name] != "MISSING", f"{name} unexpectedly missing"

    def test_dry_run_workload_firewall_rejects_final_holdout_and_unseen_eval(self):
        report = train_mod.run_dry_run_validation(seed=42)
        assert report["workload_firewall"]["iteration_rl_final_holdout_v1"] == "CORRECTLY_REJECTED"
        assert report["workload_firewall"]["unseen_eval"] == "CORRECTLY_REJECTED"


class TestCliGating:
    def test_default_invocation_is_dry_run_and_exits_zero(self):
        proc = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "train_ppo_live.py")],
            capture_output=True, text=True, timeout=30, cwd=PROJECT_ROOT,
        )
        assert proc.returncode == 0
        assert '"traffic_generated": false' in proc.stdout
        assert '"kubernetes_mutation": false' in proc.stdout

    def test_execute_live_training_without_confirmation_flag_refuses(self):
        proc = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "train_ppo_live.py"), "--execute-live-training"],
            capture_output=True, text=True, timeout=30, cwd=PROJECT_ROOT,
        )
        assert proc.returncode == 2
        assert "Refusing" in proc.stderr

    def test_no_cli_flag_can_override_frozen_hyperparameters(self):
        """Static check: the argument parser exposes only --dry-run,
        --execute-live-training, --i-confirm-live-cluster-interaction, and
        --seed -- never a reward/action/PPO-hyperparameter override flag."""
        # Reconstruct main()'s parser definition by inspection instead of
        # calling main() (which would require argv plumbing) -- assert on
        # the actual source text for the flags it registers.
        source = Path(train_mod.__file__).read_text()
        add_argument_lines = [line for line in source.splitlines() if "add_argument(" in line]
        flags = []
        for line in add_argument_lines:
            if '"--' in line:
                flags.append(line.split('"--')[1].split('"')[0])
        allowed = {"dry-run", "execute-live-training", "i-confirm-live-cluster-interaction", "seed"}
        assert set(flags) == allowed, f"unexpected CLI flags: {set(flags) - allowed}"


class TestLiveTrainingSkeletonNeverAutoInvoked:
    def test_run_live_training_function_exists_but_main_does_not_call_it_by_default(self):
        assert hasattr(train_mod, "run_live_training")
        # main() with no args must resolve to the dry-run path, proven above.

    def test_run_live_training_checks_hpa_before_any_other_live_action(self):
        source = Path(train_mod.__file__).read_text()
        live_fn_src = source.split("def run_live_training")[1].split("def main")[0]
        hpa_check_idx = live_fn_src.find("hpa_exists")
        learn_idx = live_fn_src.find("model.learn")
        assert hpa_check_idx != -1 and learn_idx != -1
        assert hpa_check_idx < learn_idx
