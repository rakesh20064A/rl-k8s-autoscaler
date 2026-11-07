"""tests/unit/test_demo_app_deployment_probes.py

V2 Phase 4B-R3: locks in the demo-app Deployment's probe contract after a
TCP-socket liveness correction was proposed, LIVE-VALIDATED, and REJECTED
(it prevented the liveness-restart but caused a worse failure: with only 1
replica, a sustained-NotReady pod left HPA with zero Ready pods to sample
CPU from, producing 4+ minutes of genuine total unavailability with no
recovery path -- see k8s/app/deployment.yaml's own inline provenance
comment). This test proves the final, reverted state: readiness and
liveness are BOTH HTTP GET /health, matching the pre-Phase-4B-R3 contract,
and the resource requests/limits this phase was explicitly forbidden from
touching remain unchanged.
"""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEPLOYMENT_PATH = PROJECT_ROOT / "k8s" / "app" / "deployment.yaml"


def _container_spec() -> dict:
    with open(DEPLOYMENT_PATH, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    return manifest["spec"]["template"]["spec"]["containers"][0]


class TestReadinessProbe:
    def test_readiness_is_http_health(self):
        probe = _container_spec()["readinessProbe"]
        assert "httpGet" in probe
        assert probe["httpGet"]["path"] == "/health"
        assert probe["httpGet"]["port"] == 8080


class TestLivenessProbe:
    def test_liveness_is_http_health_not_tcp(self):
        """The TCP-socket correction was explicitly rejected after live
        validation -- liveness must remain HTTP GET /health, matching
        readiness, not tcpSocket."""
        probe = _container_spec()["livenessProbe"]
        assert "httpGet" in probe, "liveness must be HTTP, not tcpSocket -- the TCP correction was rejected"
        assert "tcpSocket" not in probe
        assert probe["httpGet"]["path"] == "/health"
        assert probe["httpGet"]["port"] == 8080

    def test_liveness_thresholds_unchanged(self):
        probe = _container_spec()["livenessProbe"]
        assert probe["initialDelaySeconds"] == 10
        assert probe["periodSeconds"] == 10
        assert probe["failureThreshold"] == 5
        assert probe["timeoutSeconds"] == 3


class TestResourceLimitsUnchanged:
    """V2 Phase 4B-R3 explicitly forbade touching CPU/memory requests/limits
    regardless of the probe-correction outcome."""

    def test_cpu_request_and_limit(self):
        resources = _container_spec()["resources"]
        assert resources["requests"]["cpu"] == "100m"
        assert resources["limits"]["cpu"] == "500m"

    def test_memory_request_and_limit(self):
        resources = _container_spec()["resources"]
        assert resources["requests"]["memory"] == "64Mi"
        assert resources["limits"]["memory"] == "256Mi"
