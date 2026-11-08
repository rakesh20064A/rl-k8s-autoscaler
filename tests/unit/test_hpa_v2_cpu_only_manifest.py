"""tests/unit/test_hpa_v2_cpu_only_manifest.py

V2 Phase 5A / CORR-60: structural checks on the new
k8s/hpa/hpa_v2_cpu_only_min2.yaml manifest -- the third evaluation arm
(CPU-only HPA, V2 replica floor). NOT applied by this phase; manifest
preparation only.
"""

from __future__ import annotations

from pathlib import Path

import yaml

HPA_V2_CPU_MEMORY_PATH = Path("k8s/hpa/hpa_v2_cpu_memory.yaml")
HPA_V2_CPU_ONLY_PATH = Path("k8s/hpa/hpa_v2_cpu_only_min2.yaml")


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestManifestExists:
    def test_manifest_exists(self):
        assert HPA_V2_CPU_ONLY_PATH.is_file()


class TestManifestStructure:
    def test_uses_autoscaling_v2_api(self):
        manifest = _load(HPA_V2_CPU_ONLY_PATH)
        assert manifest["apiVersion"] == "autoscaling/v2"
        assert manifest["kind"] == "HorizontalPodAutoscaler"

    def test_distinct_object_name_from_other_two_hpas(self):
        cpu_only = _load(HPA_V2_CPU_ONLY_PATH)
        cpu_memory = _load(HPA_V2_CPU_MEMORY_PATH)
        assert cpu_only["metadata"]["name"] == "demo-app-hpa-v2-cpu"
        assert cpu_only["metadata"]["name"] != cpu_memory["metadata"]["name"]

    def test_same_scale_target_ref_as_cpu_memory_variant(self):
        cpu_only = _load(HPA_V2_CPU_ONLY_PATH)
        cpu_memory = _load(HPA_V2_CPU_MEMORY_PATH)
        assert cpu_only["spec"]["scaleTargetRef"] == cpu_memory["spec"]["scaleTargetRef"]

    def test_replica_bounds_min_2_max_8(self):
        manifest = _load(HPA_V2_CPU_ONLY_PATH)
        assert manifest["spec"]["minReplicas"] == 2
        assert manifest["spec"]["maxReplicas"] == 8

    def test_single_cpu_metric_only(self):
        manifest = _load(HPA_V2_CPU_ONLY_PATH)
        metrics = manifest["spec"]["metrics"]
        assert len(metrics) == 1
        assert metrics[0]["type"] == "Resource"
        assert metrics[0]["resource"]["name"] == "cpu"
        assert metrics[0]["resource"]["target"]["averageUtilization"] == 260

    def test_behavior_identical_to_cpu_memory_variant(self):
        cpu_only = _load(HPA_V2_CPU_ONLY_PATH)
        cpu_memory = _load(HPA_V2_CPU_MEMORY_PATH)
        assert cpu_only["spec"]["behavior"] == cpu_memory["spec"]["behavior"]
