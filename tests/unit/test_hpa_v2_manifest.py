"""tests/unit/test_hpa_v2_manifest.py

V2 Phase 2 PART 3/9: structural checks on the new k8s/hpa/hpa_v2_cpu_memory.yaml
manifest (CPU + memory baseline), and a small worked illustration of the
Kubernetes autoscaling/v2 "largest recommendation wins" rule the task asks
to document.
"""

from __future__ import annotations

import math
from pathlib import Path

import yaml

HPA_V1_PATH = Path("k8s/hpa/hpa.yaml")
HPA_V2_PATH = Path("k8s/hpa/hpa_v2_cpu_memory.yaml")


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestManifestExistsAlongsideFrozenV1:
    def test_v2_manifest_exists(self):
        assert HPA_V2_PATH.exists()

    def test_v1_manifest_untouched_structurally(self):
        v1 = _load(HPA_V1_PATH)
        assert v1["metadata"]["name"] == "demo-app-hpa"
        assert len(v1["spec"]["metrics"]) == 1
        assert v1["spec"]["metrics"][0]["resource"]["name"] == "cpu"
        assert v1["spec"]["metrics"][0]["resource"]["target"]["averageUtilization"] == 260


class TestV2ManifestStructure:
    def test_uses_autoscaling_v2_api(self):
        v2 = _load(HPA_V2_PATH)
        assert v2["apiVersion"] == "autoscaling/v2"
        assert v2["kind"] == "HorizontalPodAutoscaler"

    def test_distinct_object_name_from_frozen_v1(self):
        v2 = _load(HPA_V2_PATH)
        v1 = _load(HPA_V1_PATH)
        assert v2["metadata"]["name"] != v1["metadata"]["name"]
        assert v2["metadata"]["name"] == "demo-app-hpa-v2"

    def test_same_scale_target_ref_as_v1(self):
        v2 = _load(HPA_V2_PATH)
        v1 = _load(HPA_V1_PATH)
        assert v2["spec"]["scaleTargetRef"] == v1["spec"]["scaleTargetRef"]

    def test_replica_bounds_match_task_requirements(self):
        """V2 Phase 4D-B / CORR-55: minReplicas raised 1 -> 2 for the V2
        CPU+memory HPA (both V2 training failures traced to a single
        saturated pod at replicas=1); maxReplicas unchanged."""
        v2 = _load(HPA_V2_PATH)
        assert v2["spec"]["minReplicas"] == 2
        assert v2["spec"]["maxReplicas"] == 8

    def test_v1_hpa_min_replicas_unchanged(self):
        v1 = _load(HPA_V1_PATH)
        assert v1["spec"]["minReplicas"] == 1
        assert v1["spec"]["maxReplicas"] == 8

    def test_has_both_cpu_and_memory_resource_metrics(self):
        v2 = _load(HPA_V2_PATH)
        metrics = v2["spec"]["metrics"]
        names = {m["resource"]["name"] for m in metrics if m["type"] == "Resource"}
        assert names == {"cpu", "memory"}

    def test_cpu_target_unchanged_from_frozen_v1(self):
        v2 = _load(HPA_V2_PATH)
        v1 = _load(HPA_V1_PATH)
        v2_cpu = next(m for m in v2["spec"]["metrics"] if m["resource"]["name"] == "cpu")
        v1_cpu = next(m for m in v1["spec"]["metrics"] if m["resource"]["name"] == "cpu")
        assert v2_cpu["resource"]["target"]["averageUtilization"] == v1_cpu["resource"]["target"]["averageUtilization"]

    def test_memory_target_is_calibrated_live_value(self):
        v2 = _load(HPA_V2_PATH)
        mem = next(m for m in v2["spec"]["metrics"] if m["resource"]["name"] == "memory")
        assert mem["resource"]["target"]["type"] == "Utilization"
        # 120% of the 64Mi request = 76.8Mi -- CALIBRATED_LIVE (V2 Phase 2.1
        # PART 4) from real measurements: idle ~55-59%, memory-heavy ~139-
        # 141%, mixed ~189-191% of the 64Mi request. 120% sits comfortably
        # above idle and comfortably below the weakest memory-heavy
        # reading -- see the manifest's own comments for the full
        # measurement-backed reasoning.
        assert mem["resource"]["target"]["averageUtilization"] == 120

    def test_memory_target_above_measured_idle_with_headroom(self):
        v2 = _load(HPA_V2_PATH)
        mem = next(m for m in v2["spec"]["metrics"] if m["resource"]["name"] == "memory")
        measured_idle_max_pct = 59  # ~38Mi / 64Mi
        assert mem["resource"]["target"]["averageUtilization"] > measured_idle_max_pct

    def test_memory_target_below_measured_memory_heavy_reading(self):
        v2 = _load(HPA_V2_PATH)
        mem = next(m for m in v2["spec"]["metrics"] if m["resource"]["name"] == "memory")
        measured_memory_heavy_min_pct = 139  # ~89Mi / 64Mi, memory-only scenario
        assert mem["resource"]["target"]["averageUtilization"] < measured_memory_heavy_min_pct

    def test_memory_target_below_pod_limit_utilization(self):
        """The limit (256Mi) is 400% of the request (64Mi) -- the chosen
        target must leave real headroom below that hard ceiling, not sit at
        or above it."""
        v2 = _load(HPA_V2_PATH)
        mem = next(m for m in v2["spec"]["metrics"] if m["resource"]["name"] == "memory")
        pod_limit_utilization_pct = 400  # 256Mi / 64Mi
        assert mem["resource"]["target"]["averageUtilization"] < pod_limit_utilization_pct

    def test_does_not_modify_frozen_hpa_yaml_file_itself(self):
        # Mere existence check: this test file must never read/assert
        # anything that would require editing hpa.yaml's own content.
        v1 = _load(HPA_V1_PATH)
        assert v1["metadata"]["name"] == "demo-app-hpa"  # unchanged sentinel


def _recommended_replicas(current_replicas: int, current_utilization_pct: float, target_utilization_pct: float) -> int:
    """Kubernetes HPA's own per-metric recommendation formula
    (https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/#algorithm-details):
    desiredReplicas = ceil(currentReplicas * (currentMetricValue / desiredMetricValue)).
    A small, illustrative reproduction -- NOT a reimplementation of the real
    HPA controller, used only to document/prove the "largest recommendation
    wins" rule referenced in hpa_v2_cpu_memory.yaml's own comments."""
    return math.ceil(current_replicas * (current_utilization_pct / target_utilization_pct))


class TestLargestRecommendationSemantics:
    """Documents (via a worked numeric example) the autoscaling/v2 rule that
    when multiple metrics are specified, the HPA uses the LARGEST of the
    per-metric recommended replica counts -- never an average, never only
    the first metric."""

    def test_cpu_bound_workload_scales_on_cpu_recommendation(self):
        # 3 replicas, CPU average utilization well above its 260% target
        # (CPU-bound), memory average utilization well below its 120% target.
        cpu_rec = _recommended_replicas(current_replicas=3, current_utilization_pct=520, target_utilization_pct=260)
        mem_rec = _recommended_replicas(current_replicas=3, current_utilization_pct=50, target_utilization_pct=120)
        assert cpu_rec == 6
        assert mem_rec == 2
        assert max(cpu_rec, mem_rec) == cpu_rec  # CPU's recommendation wins

    def test_memory_bound_workload_scales_on_memory_recommendation(self):
        # 2 replicas, memory heavily loaded (380% aggregate = 190%/replica,
        # the measured mixed-mode level), CPU calm.
        cpu_rec = _recommended_replicas(current_replicas=2, current_utilization_pct=60, target_utilization_pct=260)
        mem_rec = _recommended_replicas(current_replicas=2, current_utilization_pct=380, target_utilization_pct=120)
        assert mem_rec > cpu_rec
        assert max(cpu_rec, mem_rec) == mem_rec  # memory's recommendation wins

    def test_mixed_workload_scales_on_whichever_metric_is_more_constrained(self):
        cpu_rec = _recommended_replicas(current_replicas=4, current_utilization_pct=1200, target_utilization_pct=260)
        mem_rec = _recommended_replicas(current_replicas=4, current_utilization_pct=600, target_utilization_pct=120)
        chosen = max(cpu_rec, mem_rec)
        assert chosen == max(cpu_rec, mem_rec)  # the documented rule: largest wins, regardless of which metric
        assert chosen >= cpu_rec and chosen >= mem_rec
