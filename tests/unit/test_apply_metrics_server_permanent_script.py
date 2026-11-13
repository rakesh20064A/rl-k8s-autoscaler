"""tests/unit/test_apply_metrics_server_permanent_script.py

V2 Phase 5A / CORR-60: offline-only checks for
k8s/metrics-server/metrics-server.yaml and
scripts/apply_metrics_server_permanent.sh -- neither is applied/run by
this phase. Mirrors the grep-based static-guard pattern already used by
tests/unit/test_v2_cluster_prepare_script.py.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "k8s" / "metrics-server" / "metrics-server.yaml"
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "apply_metrics_server_permanent.sh"


def _load_all(path: Path):
    with open(path, encoding="utf-8") as f:
        return list(yaml.safe_load_all(f))


class TestManifestParsesAndHasExpectedArgs:
    def test_manifest_exists_and_parses(self):
        docs = _load_all(MANIFEST_PATH)
        assert len(docs) >= 1
        assert all(d is not None for d in docs)

    def test_auth_reader_rolebinding_present(self):
        """V2 Phase 5D (live-setup): metrics-server panics on startup
        without read access to the extension-apiserver-authentication
        ConfigMap -- found live when this manifest was first actually
        deployed. Binds to the cluster's own built-in
        extension-apiserver-authentication-reader Role."""
        docs = _load_all(MANIFEST_PATH)
        bindings = [d for d in docs if d.get("kind") == "RoleBinding" and d.get("metadata", {}).get("namespace") == "kube-system"]
        assert any(b["roleRef"]["name"] == "extension-apiserver-authentication-reader" for b in bindings)

    def test_auth_delegator_clusterrolebinding_present(self):
        """V2 Phase 5D (live-setup): the HPA controller's own calls to the
        metrics.k8s.io API fail with a SubjectAccessReview-forbidden error
        without this binding -- found live during the same deployment."""
        docs = _load_all(MANIFEST_PATH)
        bindings = [d for d in docs if d.get("kind") == "ClusterRoleBinding"]
        assert any(b["roleRef"]["name"] == "system:auth-delegator" for b in bindings)

    def test_deployment_present(self):
        docs = _load_all(MANIFEST_PATH)
        deployments = [d for d in docs if d.get("kind") == "Deployment"]
        assert len(deployments) == 1

    def test_metric_resolution_is_15s(self):
        docs = _load_all(MANIFEST_PATH)
        deployment = next(d for d in docs if d.get("kind") == "Deployment")
        args = deployment["spec"]["template"]["spec"]["containers"][0]["args"]
        assert "--metric-resolution=15s" in args

    def test_kubelet_insecure_tls_present(self):
        docs = _load_all(MANIFEST_PATH)
        deployment = next(d for d in docs if d.get("kind") == "Deployment")
        args = deployment["spec"]["template"]["spec"]["containers"][0]["args"]
        assert "--kubelet-insecure-tls" in args

    def test_image_is_a_real_pinned_image_not_a_placeholder(self):
        """V2 Phase 5B PART D: the image tag was resolved via the one
        permitted read-only cluster query
        (`kubectl get deployment metrics-server -n kube-system -o
        jsonpath='{.spec.template.spec.containers[0].image}'`) and must now
        be a real, digest-pinned image -- never the Phase 5A placeholder."""
        docs = _load_all(MANIFEST_PATH)
        deployment = next(d for d in docs if d.get("kind") == "Deployment")
        image = deployment["spec"]["template"]["spec"]["containers"][0]["image"]
        assert "PLACEHOLDER" not in image
        assert image.startswith("registry.k8s.io/metrics-server/metrics-server:")
        assert "@sha256:" in image  # pinned by digest, not just a mutable tag


class TestScriptStaticGuards:
    def test_script_exists_and_is_executable(self):
        assert SCRIPT_PATH.is_file()
        assert SCRIPT_PATH.stat().st_mode & 0o111

    def test_script_has_bash_strict_mode(self):
        assert "set -euo pipefail" in SCRIPT_PATH.read_text(encoding="utf-8")

    def test_script_syntax_is_valid_bash(self):
        result = subprocess.run(["bash", "-n", str(SCRIPT_PATH)], capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr

    def test_script_has_a_documented_rollback(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        assert "ROLLBACK" in source
        assert "minikube addons enable metrics-server" in source

    def test_script_never_touches_demo_app_hpa_or_prometheus(self):
        code_lines = "\n".join(
            line for line in SCRIPT_PATH.read_text(encoding="utf-8").splitlines() if not line.strip().startswith("#")
        )
        assert not re.search(r"\bhpa\b", code_lines, re.IGNORECASE)
        assert "demo-app" not in code_lines
        assert "prometheus" not in code_lines.lower()

    def test_script_refuses_to_run_while_image_placeholder_is_unresolved(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        assert "PLACEHOLDER_IMAGE_TAG_UNVERIFIED_OFFLINE" in source
        assert re.search(r"grep\s+-q\s+.*PLACEHOLDER", source)
