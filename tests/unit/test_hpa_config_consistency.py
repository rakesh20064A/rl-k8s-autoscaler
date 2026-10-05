"""tests/unit/test_hpa_config_consistency.py

Minimal static consistency check between configs/hpa.yaml (the project's
human-edited source of truth for HPA parameters) and k8s/hpa/hpa.yaml (the
actual autoscaling/v2 manifest applied to the cluster). These two files are
currently kept in sync manually (Phase 11B) rather than by a generator, so
this test exists to catch silent divergence early.
"""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HPA_CONFIG_PATH = PROJECT_ROOT / "configs" / "hpa.yaml"
HPA_MANIFEST_PATH = PROJECT_ROOT / "k8s" / "hpa" / "hpa.yaml"


def _load_hpa_config():
    with open(HPA_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_hpa_manifest():
    with open(HPA_MANIFEST_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_hpa_manifest_exists():
    assert HPA_MANIFEST_PATH.exists(), (
        "k8s/hpa/hpa.yaml is missing — configs/hpa.yaml alone does not "
        "configure a live HPA."
    )


def test_target_utilization_matches():
    config = _load_hpa_config()
    manifest = _load_hpa_manifest()
    manifest_target = manifest["spec"]["metrics"][0]["resource"]["target"]["averageUtilization"]
    assert manifest_target == config["target_cpu_utilization_percent"]


def test_replica_bounds_match():
    config = _load_hpa_config()
    manifest = _load_hpa_manifest()
    assert manifest["spec"]["minReplicas"] == config["min_replicas"]
    assert manifest["spec"]["maxReplicas"] == config["max_replicas"]


def test_scale_up_behavior_matches():
    config = _load_hpa_config()
    manifest = _load_hpa_manifest()
    scale_up = manifest["spec"]["behavior"]["scaleUp"]
    assert scale_up["stabilizationWindowSeconds"] == config["scale_up_stabilization_seconds"]
    policy = scale_up["policies"][0]
    assert policy["value"] == config["scale_up_max_pods"]
    assert policy["periodSeconds"] == config["scale_up_window_seconds"]


def test_scale_down_behavior_matches():
    config = _load_hpa_config()
    manifest = _load_hpa_manifest()
    scale_down = manifest["spec"]["behavior"]["scaleDown"]
    assert scale_down["stabilizationWindowSeconds"] == config["scale_down_stabilization_seconds"]
    policy = scale_down["policies"][0]
    assert policy["value"] == config["scale_down_max_pods"]
    assert policy["periodSeconds"] == config["scale_down_window_seconds"]


def test_manifest_targets_demo_app_in_correct_namespace():
    manifest = _load_hpa_manifest()
    assert manifest["metadata"]["namespace"] == "autoscaler-app"
    assert manifest["spec"]["scaleTargetRef"]["kind"] == "Deployment"
    assert manifest["spec"]["scaleTargetRef"]["name"] == "demo-app"
