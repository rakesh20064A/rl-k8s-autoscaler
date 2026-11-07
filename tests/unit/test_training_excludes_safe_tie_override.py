"""tests/unit/test_training_excludes_safe_tie_override.py

V2 Phase 2.1 PART 8: explicitly verifies that the safe near-tie-HOLD
override (autoscaler/src/rl_env/action_selection.py,
autoscaler/src/rl_env/safe_inference.py,
autoscaler/src/rl_env/safe_ppo_controller.py) is NEVER imported by any PPO
TRAINING script. PPO's own on-policy training requires genuinely stochastic
action sampling (SB3's `model.learn()` samples from the policy's own
categorical distribution internally) -- inserting the safe selector's
deterministic argmax-with-tie-break logic into that path would corrupt the
training signal (the advantage estimator assumes actions were drawn from
the policy's actual sampling distribution, not silently overridden).

This is a structural/import-graph guard, not a runtime behavioral test --
`model.learn()` itself is Stable-Baselines3 library code (not reimplemented
by this project), so there is no project-level sampling function to unit-
test directly. What IS directly verifiable and enforced here: no training
entry point imports or references the safe-selector modules at all.
"""

from __future__ import annotations

import ast
from pathlib import Path

TRAINING_SCRIPTS = [
    Path("scripts/train_ppo_live.py"),
    Path("scripts/train_ppo_live_session.py"),
]

SAFE_TIE_MODULES = {
    "autoscaler.src.rl_env.action_selection",
    "autoscaler.src.rl_env.safe_inference",
    "autoscaler.src.rl_env.safe_ppo_controller",
}


def _imported_modules(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


class TestTrainingScriptsNeverImportSafeSelector:
    def test_train_ppo_live_does_not_import_safe_tie_modules(self):
        imports = _imported_modules(TRAINING_SCRIPTS[0])
        assert imports.isdisjoint(SAFE_TIE_MODULES)

    def test_train_ppo_live_session_does_not_import_safe_tie_modules(self):
        imports = _imported_modules(TRAINING_SCRIPTS[1])
        assert imports.isdisjoint(SAFE_TIE_MODULES)

    def test_training_wrapper_does_not_import_safe_tie_modules(self):
        imports = _imported_modules(Path("autoscaler/src/rl_env/training_wrapper.py"))
        assert imports.isdisjoint(SAFE_TIE_MODULES)

    def test_training_scripts_call_learn_not_deterministic_predict(self):
        """Positive confirmation that training genuinely goes through
        SB3's stochastic `model.learn()` (library-internal sampling),
        rather than looping `model.predict(..., deterministic=True)` the
        way the frozen EVALUATION script does."""
        for path in TRAINING_SCRIPTS:
            source = path.read_text(encoding="utf-8")
            assert ".learn(" in source
            assert "deterministic=True" not in source


class TestSafeTieModulesOnlyUsedByInferencePaths:
    """Positive confirmation of WHERE the safe selector IS used -- the V2
    inference path only, never training."""

    def test_safe_ppo_controller_used_only_by_v2_eval_script(self):
        import subprocess

        result = subprocess.run(
            ["grep", "-rl", "--include=*.py", "safe_ppo_controller", "scripts/", "autoscaler/"],
            capture_output=True, text=True, check=False,
        )
        users = {line for line in result.stdout.splitlines() if line}
        assert users == {"scripts/evaluate_ppo_vs_hpa_v2.py", "autoscaler/src/rl_env/safe_ppo_controller.py"}
