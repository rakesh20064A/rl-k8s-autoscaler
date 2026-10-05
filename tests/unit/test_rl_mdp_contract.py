"""tests/unit/test_rl_mdp_contract.py

Phase 15A / 15A.1 (CORR-27): offline, read-only validation of the frozen RL
MDP contract (docs/rl_mdp_spec.md, v2) as reflected in configs/rl.yaml,
configs/reward.yaml, configs/experiment.yaml, configs/hpa.yaml, and
k8s/hpa/hpa.yaml. No Kubernetes/Prometheus/network access, no RL code
exercised (none exists yet). A handful of tests assert on the frozen prose in
docs/rl_mdp_spec.md itself -- acceptable at this pure-specification stage
(no environment/controller code exists yet to test the *behavior* directly)
and intended as a lightweight regression guard against silently reverting a
frozen v2 rule while later phases edit the document.
"""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RL_CONFIG_PATH = PROJECT_ROOT / "configs" / "rl.yaml"
REWARD_CONFIG_PATH = PROJECT_ROOT / "configs" / "reward.yaml"
EXPERIMENT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"
HPA_CONFIG_PATH = PROJECT_ROOT / "configs" / "hpa.yaml"
HPA_MANIFEST_PATH = PROJECT_ROOT / "k8s" / "hpa" / "hpa.yaml"
RL_MDP_SPEC_PATH = PROJECT_ROOT / "docs" / "rl_mdp_spec.md"
UNSEEN_EVAL_PATH = PROJECT_ROOT / "configs/traffic_profiles/unseen_eval.yaml"

# Frozen expected observation vector, docs/rl_mdp_spec.md sec.7 (v2).
# Only indices 0/1's raw range changed from v1 (4.0 -> 5.0); normalized
# min/max and names/order are unchanged.
EXPECTED_OBSERVATION_V2 = [
    ("cpu_per_replica", 0.0, 5.0, 0.0, 1.0),
    ("cpu_per_replica_delta", -5.0, 5.0, -1.0, 1.0),
    ("memory_per_replica", 0.0, 4.0, 0.0, 1.0),
    ("in_progress_per_replica", 0.0, 20.0, 0.0, 1.0),
    ("request_rate_norm", 0.0, 100.0, 0.0, 1.0),
    ("request_rate_delta", -100.0, 100.0, -1.0, 1.0),
    ("p95_latency_norm", 0.0, 2000.0, 0.0, 1.0),
    ("p95_latency_delta", -2000.0, 2000.0, -1.0, 1.0),
    ("error_rate", 0.0, 1.0, 0.0, 1.0),
    ("ready_replica_norm", 1.0, 8.0, 0.0, 1.0),
    ("seconds_since_last_scale_norm", 0.0, 300.0, 0.0, 1.0),
]

EXPECTED_ACTION_MAP = {0: -1, 1: 0, 2: 1}


def _load(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _rl_config():
    return _load(RL_CONFIG_PATH)


def _reward_config():
    return _load(REWARD_CONFIG_PATH)


def _experiment_config():
    return _load(EXPERIMENT_CONFIG_PATH)


def _spec_text() -> str:
    return RL_MDP_SPEC_PATH.read_text(encoding="utf-8")


class TestObservationVectorContract:
    def test_observation_v2_has_fixed_unique_order(self):
        config = _rl_config()
        entries = config["observation_v2"]
        indices = [e["index"] for e in entries]
        assert indices == list(range(len(entries))), "observation_v2 indices must be sequential starting at 0"
        names = [e["name"] for e in entries]
        assert len(names) == len(set(names)), "observation_v2 names must be unique"

    def test_observation_v2_matches_frozen_spec(self):
        config = _rl_config()
        entries = config["observation_v2"]
        actual = [
            (e["name"], float(e["raw_min"]), float(e["raw_max"]), float(e["min"]), float(e["max"]))
            for e in entries
        ]
        assert actual == EXPECTED_OBSERVATION_V2, (
            "observation_v2 diverged from the frozen docs/rl_mdp_spec.md contract -- "
            "bump rl_contract_version and update docs/rl_mdp_spec.md before changing this."
        )

    def test_cpu_raw_range_is_exactly_5x_request(self):
        """CORR-27: v1's 4.0x clip was too low (below the 5.0x physical
        ceiling and below Phase-10's observed 4.63x saturation mean)."""
        config = _rl_config()
        cpu = config["observation_v2"][0]
        cpu_delta = config["observation_v2"][1]
        assert cpu["name"] == "cpu_per_replica"
        assert float(cpu["raw_min"]) == 0.0
        assert float(cpu["raw_max"]) == 5.0
        assert cpu_delta["name"] == "cpu_per_replica_delta"
        assert float(cpu_delta["raw_min"]) == -5.0
        assert float(cpu_delta["raw_max"]) == 5.0

    def test_no_legacy_observation_v1_block_remains(self):
        config = _rl_config()
        assert "observation_v1" not in config, "the old observation_v1 block must be renamed to observation_v2, not left alongside it"

    def test_rl_contract_version_frozen_at_2(self):
        config = _rl_config()
        assert config["rl_contract_version"] == 2

    def test_observation_version_tag_bumped_to_v2(self):
        config = _rl_config()
        assert config["agent"]["observation_version"] == "v2"

    def test_history_horizons_align_with_decision_interval(self):
        rl_config = _rl_config()
        experiment_config = _experiment_config()
        decision_interval = experiment_config["rl"]["control_interval_seconds"]
        ewma_span_steps = rl_config["history"]["ewma_span_steps"]
        assert isinstance(ewma_span_steps, int) and ewma_span_steps > 0
        ewma_span_seconds = ewma_span_steps * decision_interval
        assert ewma_span_seconds % decision_interval == 0

        reward_config = _reward_config()
        reversal_window = reward_config["reversal"]["reversal_window_seconds"]
        assert reversal_window % decision_interval == 0, (
            "reversal_window_seconds must be an exact multiple of the RL decision interval."
        )


class TestActionEncodingContract:
    def test_action_map_is_exact_frozen_mapping_unchanged_by_v2(self):
        config = _rl_config()
        action_map = config["action_space"]["action_map"]
        assert {int(k): int(v) for k, v in action_map.items()} == EXPECTED_ACTION_MAP

    def test_n_actions_matches_map_length(self):
        config = _rl_config()
        assert config["action_space"]["n_actions"] == len(config["action_space"]["action_map"])


class TestAsymmetricGatingRule:
    """v2 (CORR-27): scale-up has no extra cooldown beyond pending-scale
    gating; scale-down retains the 60s cooldown; emergency scale-up must not
    be blocked solely by a recent scale-down. No environment code exists yet
    to exercise this behaviorally, so these tests pin the frozen prose and
    the absence of a symmetric-cooldown config key."""

    def test_scale_down_cooldown_is_exactly_60s_from_experiment_config(self):
        experiment_config = _experiment_config()
        assert experiment_config["scaling"]["cooldown_seconds"] == 60

    def test_no_symmetric_or_scale_up_specific_cooldown_key_exists(self):
        rl_config = _rl_config()
        reward_config = _reward_config()
        for config in (rl_config, reward_config):
            flat_text = str(config)
            assert "scale_up_cooldown" not in flat_text

    def test_spec_documents_no_extra_scale_up_cooldown(self):
        text = _spec_text()
        assert "no extra 60s scale-up-specific cooldown" in text.lower() or "no additional 60s" in text.lower() \
            or "no additional cooldown" in text.lower()

    def test_spec_documents_pending_scale_gating(self):
        text = _spec_text()
        assert "pending_scale_gate" in text
        assert "pending-scale gating" in text.lower()

    def test_spec_documents_emergency_reversal_not_blocked(self):
        text = _spec_text()
        assert "emergency reversal" in text.lower()
        assert "must not" in text.lower() and "blocked" in text.lower()


class TestReplicaBoundsContract:
    def test_experiment_replica_bounds_are_1_and_8(self):
        experiment_config = _experiment_config()
        assert experiment_config["scaling"]["min_replicas"] == 1
        assert experiment_config["scaling"]["max_replicas"] == 8

    def test_hpa_configs_agree_with_experiment_bounds(self):
        experiment_config = _experiment_config()
        hpa_config = _load(HPA_CONFIG_PATH)
        hpa_manifest = _load(HPA_MANIFEST_PATH)
        assert hpa_config["min_replicas"] == experiment_config["scaling"]["min_replicas"]
        assert hpa_config["max_replicas"] == experiment_config["scaling"]["max_replicas"]
        assert hpa_manifest["spec"]["minReplicas"] == experiment_config["scaling"]["min_replicas"]
        assert hpa_manifest["spec"]["maxReplicas"] == experiment_config["scaling"]["max_replicas"]


class TestSelectedHpaBaselineUnchanged:
    """Phase 15A.1 must not reopen or alter the CORR-26 HPA selection."""

    def test_hpa_baseline_still_260(self):
        hpa_config = _load(HPA_CONFIG_PATH)
        assert hpa_config["target_cpu_utilization_percent"] == 260

    def test_hpa_manifest_still_260(self):
        hpa_manifest = _load(HPA_MANIFEST_PATH)
        assert hpa_manifest["spec"]["metrics"][0]["resource"]["target"]["averageUtilization"] == 260

    def test_hpa_behavior_unchanged(self):
        hpa_manifest = _load(HPA_MANIFEST_PATH)
        scale_up = hpa_manifest["spec"]["behavior"]["scaleUp"]
        scale_down = hpa_manifest["spec"]["behavior"]["scaleDown"]
        assert scale_up["stabilizationWindowSeconds"] == 0
        assert scale_down["stabilizationWindowSeconds"] == 120


class TestRewardContractV2:
    def test_reward_contract_version_frozen_at_2(self):
        config = _reward_config()
        assert config["reward_contract_version"] == 2

    def test_all_weights_are_numeric(self):
        config = _reward_config()
        for name, value in config["weights"].items():
            assert isinstance(value, (int, float)), f"weight {name} is not numeric: {value!r}"

    def test_sla_p95_target_matches_experiment_sla(self):
        reward_config = _reward_config()
        experiment_config = _experiment_config()
        assert reward_config["sla"]["p95_target_ms"] == experiment_config["sla"]["p95_latency_ms"]

    def test_resource_penalty_uses_current_replicas_not_ready(self):
        """CORR-27: aligns with the CORR-26 HPA-selection methodology's own
        primary efficiency metric (current/desired replica-seconds)."""
        config = _reward_config()
        assert config["resource"]["metric"] == "deployment_current_replicas"

    def test_boundary_penalty_fires_only_on_replica_bound_clamp(self):
        config = _reward_config()
        assert config["boundary"]["triggers_on"] == "replica_bound_clamp"

    def test_efficiency_band_not_derived_from_hpa_260_target(self):
        """CORR-27: the v1 band's upper bound (2.6) was the HPA-selected
        target itself -- an unwanted coupling. v2 must use a different,
        independently-derived value."""
        config = _reward_config()
        high = config["efficiency"]["target_cpu_high"]
        assert high != 2.6, "efficiency band must not merely retain the HPA-selected 260% value"
        assert high == 3.0

    def test_efficiency_band_within_unit_fraction_of_request_and_below_physical_ceiling(self):
        config = _reward_config()
        low = config["efficiency"]["target_cpu_low"]
        high = config["efficiency"]["target_cpu_high"]
        assert 0.0 < low < high
        # CPU limit/request ratio is 500m/100m = 5.0 -- the physical ceiling per pod.
        assert high < 5.0

    def test_no_undefined_legacy_keys_reintroduced(self):
        """Phase 15A consolidated the old, ambiguous separate 'latency' block into
        the single SLA term -- guard against silent reintroduction."""
        config = _reward_config()
        assert "latency" not in config, "the old ambiguous separate latency-penalty block must not be reintroduced"


class TestOverrideAndInvalidStepSemantics:
    """No environment code exists yet to exercise these rules behaviorally;
    these tests pin the frozen v2 prose in docs/rl_mdp_spec.md as a
    regression guard (CORR-27)."""

    def test_no_traffic_latency_is_semantic_zero_not_invalid(self):
        text = _spec_text()
        assert "no_traffic" in text
        assert "not a telemetry failure" in text.lower() or "not a measurement" in text.lower()

    def test_positive_traffic_missing_latency_is_structurally_invalid(self):
        text = _spec_text()
        assert "structurally invalid" in text.lower()

    def test_fourth_consecutive_invalid_step_terminates_episode(self):
        text = _spec_text()
        assert "4th consecutive invalid" in text or "consecutive_invalid_steps" in text
        assert "120 seconds" in text

    def test_reward_validity_partial_on_invalid_step(self):
        text = _spec_text()
        assert 'reward_validity="partial"' in text or "reward_validity" in text

    def test_reward_not_a_live_inference_input(self):
        text = _spec_text()
        assert "not a policy input" in text.lower() or "does not receive or need the reward" in text.lower()


class TestTrainValidationHeldOutSeparation:
    def test_unseen_eval_not_referenced_by_rl_or_reward_config(self):
        for path in (RL_CONFIG_PATH, REWARD_CONFIG_PATH):
            text = path.read_text(encoding="utf-8")
            assert "unseen_eval" not in text, (
                f"{path} must not reference the held-out unseen_eval.yaml workload."
            )

    def test_unseen_eval_profile_file_untouched_by_this_phase(self):
        assert UNSEEN_EVAL_PATH.exists()
        content = UNSEEN_EVAL_PATH.read_text(encoding="utf-8")
        assert "seed: 99" in content, "unseen_eval.yaml's distinguishing seed must remain unchanged"

    def test_unseen_eval_marked_not_eligible_as_sole_final_held_out(self):
        """CORR-27: Phase 15A inspected unseen_eval.yaml's magnitude while
        justifying request_rate_norm's bound -- this leakage must be
        documented, and the file's eligibility as a clean sole final
        held-out workload formally revoked (without deleting or altering
        the file itself)."""
        text = _spec_text()
        assert "NOT_ELIGIBLE_AS_THE_SOLE_FORMAL_FINAL_HELD_OUT_EVALUATION_PROFILE" in text

    def test_request_rate_bound_justification_does_not_cite_unseen_eval(self):
        rl_config_text = RL_CONFIG_PATH.read_text(encoding="utf-8")
        assert "unseen_eval" not in rl_config_text
