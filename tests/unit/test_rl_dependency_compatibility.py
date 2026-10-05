"""tests/unit/test_rl_dependency_compatibility.py

Phase 15A.2 (CORR-28): validates the corrected RL runtime dependency stack
(stable-baselines3==2.6.0, gymnasium==1.1.1, torch==2.5.1, tensorboard==2.18.0)
and re-confirms the frozen RL MDP contract (rl_contract_version==2,
reward_contract_version==2, HPA baseline==260) was not disturbed by this
dependency-only phase.

These tests DO import gymnasium/stable_baselines3/torch (now installed in
.venv) to verify runtime version/compatibility -- they do not train any
model, run `.learn()`, or touch Kubernetes/Prometheus/network.
"""

import importlib.metadata
import re
from pathlib import Path

import yaml
from packaging.requirements import Requirement
from packaging.version import Version

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RL_REQUIREMENTS_PATH = PROJECT_ROOT / "requirements" / "rl.txt"
RL_CONFIG_PATH = PROJECT_ROOT / "configs" / "rl.yaml"
REWARD_CONFIG_PATH = PROJECT_ROOT / "configs" / "reward.yaml"
HPA_CONFIG_PATH = PROJECT_ROOT / "configs" / "hpa.yaml"
HPA_MANIFEST_PATH = PROJECT_ROOT / "k8s" / "hpa" / "hpa.yaml"


def _load(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_pin(requirements_text: str, package: str) -> str:
    """Extract an exact `==X.Y.Z` pin for `package` from a requirements.txt-style file."""
    match = re.search(rf"^{re.escape(package)}==([\w.]+)\s*$", requirements_text, re.MULTILINE)
    assert match, f"expected an exact '{package}==...' pin in {RL_REQUIREMENTS_PATH}"
    return match.group(1)


class TestRequirementsFilePins:
    def test_stable_baselines3_pinned_at_2_6_0(self):
        text = RL_REQUIREMENTS_PATH.read_text(encoding="utf-8")
        assert _parse_pin(text, "stable-baselines3") == "2.6.0"

    def test_gymnasium_pinned_at_1_1_1(self):
        text = RL_REQUIREMENTS_PATH.read_text(encoding="utf-8")
        assert _parse_pin(text, "gymnasium") == "1.1.1"

    def test_torch_pinned_at_2_5_1(self):
        text = RL_REQUIREMENTS_PATH.read_text(encoding="utf-8")
        assert _parse_pin(text, "torch") == "2.5.1"

    def test_tensorboard_pinned_at_2_18_0(self):
        text = RL_REQUIREMENTS_PATH.read_text(encoding="utf-8")
        assert _parse_pin(text, "tensorboard") == "2.18.0"

    def test_no_disallowed_rl_frameworks_introduced(self):
        text = RL_REQUIREMENTS_PATH.read_text(encoding="utf-8").lower()
        for disallowed in ("ray[", "rllib", "tensorflow", "cleanrl", "jax", "sb3-contrib"):
            assert disallowed not in text, f"unexpected/disallowed RL framework reference: {disallowed!r}"


class TestInstalledVersionsMatchPins:
    """Requires the corrected stack to actually be installed in .venv (Phase 15A.2)."""

    def test_installed_stable_baselines3_matches_pin(self):
        assert importlib.metadata.version("stable-baselines3") == "2.6.0"

    def test_installed_gymnasium_matches_pin(self):
        assert importlib.metadata.version("gymnasium") == "1.1.1"

    def test_installed_torch_matches_pin(self):
        assert importlib.metadata.version("torch") == "2.5.1"

    def test_installed_tensorboard_matches_pin(self):
        assert importlib.metadata.version("tensorboard") == "2.18.0"

    def test_base_numpy_pin_unaffected_by_rl_stack_install(self):
        """The RL install must not have silently bumped the shared base.txt
        numpy pin used by the rest of the project (collectors, traffic
        generator, aggregation scripts)."""
        assert importlib.metadata.version("numpy") == "1.26.4"


class TestSb3GymnasiumCompatibility:
    """Verifies compatibility by reading stable-baselines3's OWN declared
    dependency metadata, rather than hardcoding a belief about it -- this
    is the same check that revealed the original CORR-28 incompatibility."""

    def test_installed_sb3_declares_a_gymnasium_constraint_satisfied_by_1_1_1(self):
        requires = importlib.metadata.requires("stable-baselines3") or []
        gymnasium_reqs = [r for r in requires if Requirement(r).name == "gymnasium"]
        assert gymnasium_reqs, "stable-baselines3 metadata must declare a gymnasium requirement"
        req = Requirement(gymnasium_reqs[0])
        installed_gymnasium = Version(importlib.metadata.version("gymnasium"))
        assert installed_gymnasium in req.specifier, (
            f"installed gymnasium {installed_gymnasium} does not satisfy "
            f"stable-baselines3's declared constraint {req.specifier}"
        )

    def test_installed_sb3_declares_a_torch_constraint_satisfied_by_2_5_1(self):
        requires = importlib.metadata.requires("stable-baselines3") or []
        torch_reqs = [r for r in requires if Requirement(r).name == "torch"]
        assert torch_reqs, "stable-baselines3 metadata must declare a torch requirement"
        req = Requirement(torch_reqs[0])
        installed_torch = Version(importlib.metadata.version("torch"))
        assert installed_torch in req.specifier, (
            f"installed torch {installed_torch} does not satisfy "
            f"stable-baselines3's declared constraint {req.specifier}"
        )

    def test_previous_sb3_2_4_1_pin_would_have_been_incompatible(self):
        """Documents the ORIGINAL defect (CORR-28): SB3 2.4.1's own declared
        gymnasium constraint excludes 1.1.1. Uses PyPI-independent, already
        -verified constraint strings recorded in CORR-28 rather than a live
        network call, so this test remains offline-safe."""
        sb3_2_4_1_gymnasium_constraint = Requirement("gymnasium>=0.29.1,<1.1.0").specifier
        assert Version("1.1.1") not in sb3_2_4_1_gymnasium_constraint


class TestPpoConfigFieldsAcceptedBySb3:
    def test_all_configured_ppo_fields_are_known_sb3_ppo_constructor_kwargs(self):
        import inspect

        from stable_baselines3 import PPO

        rl_config = _load(RL_CONFIG_PATH)
        ppo_cfg = rl_config["ppo"]
        accepted = set(inspect.signature(PPO.__init__).parameters.keys())

        direct_fields = [
            "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma",
            "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm", "policy",
        ]
        for field in direct_fields:
            assert field in ppo_cfg, f"configs/rl.yaml ppo block missing expected field {field!r}"
            assert field in accepted, f"SB3 2.6.0 PPO.__init__ no longer accepts {field!r} (renamed/removed)"

        # net_arch is applied via policy_kwargs, not a top-level PPO kwarg.
        assert "net_arch" in ppo_cfg
        assert "policy_kwargs" in accepted


class TestRlContractUnchangedByDependencyPhase:
    """Phase 15A.2 is dependency-only -- the MDP contract frozen in
    Phase 15A.1 (v2) must be completely unaffected."""

    def test_rl_contract_version_still_2(self):
        assert _load(RL_CONFIG_PATH)["rl_contract_version"] == 2

    def test_reward_contract_version_still_2(self):
        assert _load(REWARD_CONFIG_PATH)["reward_contract_version"] == 2

    def test_observation_vector_still_11_elements_in_order(self):
        entries = _load(RL_CONFIG_PATH)["observation_v2"]
        assert len(entries) == 11
        assert [e["index"] for e in entries] == list(range(11))

    def test_action_map_still_unchanged(self):
        action_map = _load(RL_CONFIG_PATH)["action_space"]["action_map"]
        assert {int(k): int(v) for k, v in action_map.items()} == {0: -1, 1: 0, 2: 1}

    def test_hpa_baseline_still_260(self):
        assert _load(HPA_CONFIG_PATH)["target_cpu_utilization_percent"] == 260
        manifest = _load(HPA_MANIFEST_PATH)
        assert manifest["spec"]["metrics"][0]["resource"]["target"]["averageUtilization"] == 260
