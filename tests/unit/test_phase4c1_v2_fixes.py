"""tests/unit/test_phase4c1_v2_fixes.py

V2 Phase 4C-1 / 4C-1b: regression coverage for the two confirmed R4F
defects fixed this phase (P1-1: the V2 evaluation path silently ran the V1
environment/config; P1-2: V2 scripts had no explicit, canonical HPA
identity and never touched demo-app-hpa-v2) plus the additive decision-
record fields that close R4F sec.4.5's gap. Entirely offline: no
Kubernetes/Prometheus access, no real `kubectl` subprocess is ever invoked
(every `KubectlRunner` here is constructed with an injected fake `run`
callable or exercised only through its pure name/path attributes).
"""

from __future__ import annotations

import numpy as np
import pytest

from autoscaler.src.rl_env.action_selection import compute_margin_stats, select_safe_deterministic_action
from autoscaler.src.rl_env.config_v2 import (
    DEFAULT_RL_V2_YAML,
    EXPECTED_CONTROLLER_GENERATION,
    EXPECTED_WORKLOAD_CONTRACT,
    V2ConfigContractError,
    load_env_config_v2_from_yaml,
    load_safe_action_epsilon_from_v2_config,
)
from autoscaler.src.rl_env.config import DEFAULT_RL_YAML
from autoscaler.src.rl_env.controller_exclusivity import (
    CONTROLLER_CPU_MEMORY_HPA,
    CONTROLLER_PPO,
    V1_HPA_MANIFEST_PATH,
    V1_HPA_NAME,
    V2_HPA_MANIFEST_PATH,
    V2_HPA_NAME,
    ControllerConflictError,
    acquire_controller,
    detect_conflicting_hpas,
    release_controller,
)
from autoscaler.src.rl_env.decision_record_v2 import build_decision_record_v2
from autoscaler.src.rl_env.live_backend import DEFAULT_HPA_NAME, KubectlRunner, LiveKubernetesBackend
from autoscaler.src.rl_env.reward_v2 import RewardConfigV2, compute_reward_v2
from autoscaler.src.rl_env.safe_inference import infer_safe_action

# -- P1-1: V2 env/config identity -------------------------------------------


class TestEvaluationPathUsesV2EnvClass:
    def test_evaluate_script_never_constructs_v1_env_or_loader(self):
        """scripts/evaluate_ppo_vs_hpa_v2.py must import/construct
        KubernetesAutoscalerEnvV2 and load_env_config_v2_from_yaml -- never
        the V1 KubernetesAutoscalerEnv / load_env_config_from_yaml. A bare
        'KubernetesAutoscalerEnv(' construction call (not the V2 subclass
        name) would be the exact R4F P1-1 regression."""
        source = open("scripts/evaluate_ppo_vs_hpa_v2.py", encoding="utf-8").read()
        assert "KubernetesAutoscalerEnvV2" in source
        assert "from autoscaler.src.rl_env.environment import KubernetesAutoscalerEnv" not in source
        assert "load_env_config_v2_from_yaml" in source
        assert "from autoscaler.src.rl_env.config import load_env_config_from_yaml" not in source
        assert "KubernetesAutoscalerEnv(" not in source  # only the V2 subclass name may appear

    def test_safe_ppo_controller_type_hint_is_v2(self):
        source = open("autoscaler/src/rl_env/safe_ppo_controller.py", encoding="utf-8").read()
        assert "from .environment_v2 import KubernetesAutoscalerEnvV2" in source
        assert "from .environment import KubernetesAutoscalerEnv" not in source


class TestV2ConfigLoaderIdentity:
    def test_loads_real_rl_v2_yaml_with_v2_identity(self):
        cfg = load_env_config_v2_from_yaml()
        assert cfg.controller_generation == EXPECTED_CONTROLLER_GENERATION == "v2"
        assert cfg.workload_contract == EXPECTED_WORKLOAD_CONTRACT == "multi_resource_v2"

    def test_rejects_v1_rl_yaml(self):
        """Supplying configs/rl.yaml (the frozen V1 file) must be rejected,
        never silently accepted as if it were a V2 config."""
        with pytest.raises(V2ConfigContractError):
            load_env_config_v2_from_yaml(rl_yaml_path=DEFAULT_RL_YAML)

    def test_default_rl_v2_yaml_path_is_distinct_from_v1(self):
        assert DEFAULT_RL_V2_YAML != DEFAULT_RL_YAML
        assert DEFAULT_RL_V2_YAML.name == "rl_v2.yaml"


class TestRewardComponentNames:
    def test_component_names_are_exactly_the_v2_seven(self):
        result = compute_reward_v2(
            decision_state_valid=True,
            outcome_state_valid=True,
            p95_latency_ms=250.0,
            error_rate=0.0,
            deployment_current_replicas=2,
            cpu_per_replica=1.0,
            memory_per_replica=0.5,
            requested_delta=0,
            effective_delta=0,
            override_reason="none",
            is_reversal=False,
            config=RewardConfigV2(),
        )
        assert set(result.components.keys()) == {
            "reliability", "sla", "pressure", "waste", "action", "reversal", "boundary",
        }


class TestEpsilonSourcedFromYaml:
    def test_epsilon_reader_matches_live_rl_v2_yaml(self):
        epsilon, status = load_safe_action_epsilon_from_v2_config()
        assert epsilon == 0.05
        assert status == "provisional_unvalidated"

    def test_dry_run_report_epsilon_matches_yaml_when_not_overridden(self):
        from scripts.evaluate_ppo_vs_hpa_v2 import run_dry_run_validation

        report = run_dry_run_validation(epsilon=None)
        yaml_epsilon, yaml_status = load_safe_action_epsilon_from_v2_config()
        assert report["near_tie_epsilon"] == yaml_epsilon
        assert report["epsilon_status"] == yaml_status

    def test_dry_run_report_epsilon_override_still_works(self):
        from scripts.evaluate_ppo_vs_hpa_v2 import run_dry_run_validation

        report = run_dry_run_validation(epsilon=0.25)
        assert report["near_tie_epsilon"] == 0.25


class TestDecisionRecordRawArgmaxAndSafeSelected:
    def test_success_case_both_fields_populated_and_naming_consistent(self):
        """Both raw_argmax_* and safe_selected_* use the SAME ACTION_NAMES
        convention end-to-end through the real selector/margin-stats
        functions (not a hand-built fixture) -- confirms there is no
        naming-convention mismatch between the two in the real pipeline."""
        probabilities = (0.1, 0.2, 0.7)
        inference_result = infer_safe_action(model=_ArgmaxStubModel(probabilities), observation=np.zeros(11), epsilon=0.05)
        record = build_decision_record_v2(
            timestamp_utc="t", observation_vector=[0.0] * 11,
            raw_cpu_cores=None, raw_memory_bytes=None, raw_request_rate_rps=None, raw_p95_latency_ms=None,
            current_replicas=1, ready_replicas=1, desired_replicas=2,
            inference_result=inference_result, effective_action_index=inference_result.effective_action_index,
            effective_action_name=inference_result.effective_action_name,
            effective_override_reason=inference_result.override_reason, model_identifier="stub",
        )
        expected = compute_margin_stats(probabilities)
        assert record.raw_argmax_action_index == expected.argmax_action_index == 2
        assert record.raw_argmax_action_name == expected.argmax_action_name
        assert record.safe_selected_action_index == inference_result.effective_action_index
        assert record.safe_selected_action_name == inference_result.effective_action_name
        # Same ACTION_NAMES convention on both sides (e.g. "SCALE_UP_ONE"),
        # confirmed via the real selector rather than asserted by fiat.
        selector_result = select_safe_deterministic_action(probabilities, epsilon=0.05)
        assert record.raw_argmax_action_name == selector_result.action_name

    def test_failure_case_raw_argmax_none_selected_is_hold(self):
        inference_result = infer_safe_action(model=None, observation=np.zeros(11), epsilon=0.05)
        record = build_decision_record_v2(
            timestamp_utc="t", observation_vector=[0.0] * 11,
            raw_cpu_cores=None, raw_memory_bytes=None, raw_request_rate_rps=None, raw_p95_latency_ms=None,
            current_replicas=1, ready_replicas=1, desired_replicas=1,
            inference_result=inference_result, effective_action_index=inference_result.effective_action_index,
            effective_action_name=inference_result.effective_action_name,
            effective_override_reason=inference_result.override_reason, model_identifier="stub",
        )
        assert record.raw_argmax_action_index is None
        assert record.raw_argmax_action_name is None
        assert record.safe_selected_action_name == "HOLD"


class _ArgmaxStubModel:
    """Minimal stand-in exposing exactly what `extract_action_probabilities`
    needs -- a `.policy` producing a fixed categorical distribution -- so
    `infer_safe_action()` can be exercised end-to-end without SB3/torch."""

    def __init__(self, probabilities):
        self._probabilities = probabilities

    def predict(self, *args, **kwargs):  # pragma: no cover - not used by this path
        raise NotImplementedError


@pytest.fixture(autouse=True)
def _patch_probability_extraction(monkeypatch):
    import autoscaler.src.rl_env.safe_inference as safe_inference_mod

    def _fake_extract(model, observation):
        if isinstance(model, _ArgmaxStubModel):
            return np.asarray(model._probabilities, dtype=float)
        raise RuntimeError("unexpected model in fake extractor")

    monkeypatch.setattr(safe_inference_mod, "extract_action_probabilities", _fake_extract)


# -- P1-2: explicit V2 HPA identity ------------------------------------------


class TestLiveKubernetesBackendV1DefaultsUnchanged:
    def test_kubectlrunner_v1_defaults(self):
        runner = KubectlRunner()
        assert runner.hpa_name == DEFAULT_HPA_NAME == "demo-app-hpa"
        assert runner.hpa_manifest_path == "k8s/hpa/hpa.yaml"

    def test_backend_omitting_hpa_args_keeps_v1_identity(self):
        backend = LiveKubernetesBackend()
        assert backend._kubectl.hpa_name == "demo-app-hpa"
        assert backend._kubectl.hpa_manifest_path == "k8s/hpa/hpa.yaml"

    def test_backend_with_explicit_v2_identity(self):
        backend = LiveKubernetesBackend(hpa_name=V2_HPA_NAME, hpa_manifest_path=V2_HPA_MANIFEST_PATH)
        assert backend._kubectl.hpa_name == "demo-app-hpa-v2"
        assert backend._kubectl.hpa_manifest_path == "k8s/hpa/hpa_v2_cpu_memory.yaml"

    def test_injected_kubectl_instance_identity_wins_regardless_of_hpa_kwargs(self):
        custom = KubectlRunner(hpa_name="custom-hpa", hpa_manifest_path="custom.yaml")
        backend = LiveKubernetesBackend(kubectl=custom, hpa_name=V2_HPA_NAME, hpa_manifest_path=V2_HPA_MANIFEST_PATH)
        assert backend._kubectl is custom
        assert backend._kubectl.hpa_name == "custom-hpa"


class TestEachV2ScriptConstructsExplicitV2Identity:
    @pytest.mark.parametrize("path", [
        "scripts/train_ppo_final_v2.py",
        "scripts/train_ppo_live_v2.py",
        "scripts/evaluate_ppo_vs_hpa_v2.py",
    ])
    def test_script_passes_v2_hpa_identity_to_backend(self, path):
        source = open(path, encoding="utf-8").read()
        assert "hpa_name=V2_HPA_NAME" in source
        assert "hpa_manifest_path=V2_HPA_MANIFEST_PATH" in source


class _FakeKubectl:
    def __init__(self, existing_hpas):
        self._existing_hpas = list(existing_hpas)
        self.delete_hpa_called = False
        self.restore_hpa_called = False
        self.applied_manifest_paths = []

    def list_hpa_names(self):
        return list(self._existing_hpas)

    def delete_hpa(self):
        self.delete_hpa_called = True
        self._existing_hpas = []

    def restore_hpa(self):
        self.restore_hpa_called = True
        self._existing_hpas = [V2_HPA_NAME]


class TestV2PreflightFailsClosedOnV1Hpa:
    def test_detect_conflicting_hpas_flags_v1_hpa_for_v2_controller(self):
        conflicting = detect_conflicting_hpas(CONTROLLER_CPU_MEMORY_HPA, _FakeKubectl([V1_HPA_NAME]))
        assert conflicting == [V1_HPA_NAME]

    def test_acquire_controller_refuses_and_never_deletes_v1_hpa(self, tmp_path):
        fake = _FakeKubectl([V1_HPA_NAME])
        lock_path = tmp_path / "lock.json"
        with pytest.raises(ControllerConflictError):
            acquire_controller(CONTROLLER_CPU_MEMORY_HPA, fake, lock_path=lock_path)
        assert fake.delete_hpa_called is False
        assert fake.list_hpa_names() == [V1_HPA_NAME]  # untouched

    def test_hpa_arm_acquisition_succeeds_when_only_v2_hpa_or_none_present(self, tmp_path):
        lock_path = tmp_path / "lock.json"
        acquire_controller(CONTROLLER_CPU_MEMORY_HPA, _FakeKubectl([]), lock_path=lock_path)
        release_controller(CONTROLLER_CPU_MEMORY_HPA, lock_path=lock_path)
        acquire_controller(CONTROLLER_CPU_MEMORY_HPA, _FakeKubectl([V2_HPA_NAME]), lock_path=lock_path)
        release_controller(CONTROLLER_CPU_MEMORY_HPA, lock_path=lock_path)

    def test_ppo_arm_refuses_when_any_hpa_present(self, tmp_path):
        lock_path = tmp_path / "lock.json"
        for existing in ([V1_HPA_NAME], [V2_HPA_NAME], [V1_HPA_NAME, V2_HPA_NAME]):
            with pytest.raises(ControllerConflictError):
                acquire_controller(CONTROLLER_PPO, _FakeKubectl(existing), lock_path=lock_path)

    def test_ppo_arm_succeeds_when_no_hpa_present(self, tmp_path):
        lock_path = tmp_path / "lock.json"
        acquire_controller(CONTROLLER_PPO, _FakeKubectl([]), lock_path=lock_path)
        release_controller(CONTROLLER_PPO, lock_path=lock_path)


class TestV2RestoreAppliesV2Manifest:
    def test_kubectlrunner_restore_hpa_applies_the_v2_manifest_path(self):
        applied_cmds = []

        def fake_run(cmd, timeout):
            applied_cmds.append(cmd)
            import subprocess

            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        runner = KubectlRunner(hpa_name=V2_HPA_NAME, hpa_manifest_path=V2_HPA_MANIFEST_PATH, run=fake_run)
        runner.restore_hpa()
        assert applied_cmds == [["kubectl", "apply", "-f", V2_HPA_MANIFEST_PATH]]

    def test_kubectlrunner_delete_hpa_targets_the_v2_name_not_v1(self):
        applied_cmds = []

        def fake_run(cmd, timeout):
            applied_cmds.append(cmd)
            import subprocess

            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        runner = KubectlRunner(hpa_name=V2_HPA_NAME, hpa_manifest_path=V2_HPA_MANIFEST_PATH, run=fake_run)
        runner.delete_hpa()
        assert applied_cmds[0][:4] == ["kubectl", "delete", "hpa", V2_HPA_NAME]
        assert V1_HPA_NAME not in applied_cmds[0]

    def test_v1_defaults_still_target_v1_identity(self):
        applied_cmds = []

        def fake_run(cmd, timeout):
            applied_cmds.append(cmd)
            import subprocess

            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        runner = KubectlRunner(run=fake_run)  # no explicit hpa_name/hpa_manifest_path -- V1 defaults
        runner.restore_hpa()
        assert applied_cmds == [["kubectl", "apply", "-f", V1_HPA_MANIFEST_PATH]]


class TestHpaEvalArmExists:
    def test_run_hpa_v2_episode_is_defined_and_uses_cpu_memory_controller(self):
        import scripts.evaluate_ppo_vs_hpa_v2 as eval_mod

        assert hasattr(eval_mod, "run_hpa_v2_episode")
        source = open("scripts/evaluate_ppo_vs_hpa_v2.py", encoding="utf-8").read()
        assert "acquire_controller(CONTROLLER_CPU_MEMORY_HPA" in source
        assert "--execute" in source  # still gated behind the same confirmation flag as the PPO arm
