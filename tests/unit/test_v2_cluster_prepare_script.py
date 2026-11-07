"""tests/unit/test_v2_cluster_prepare_script.py

V2 Phase 4D-C: offline-only checks for scripts/v2_cluster_prepare.sh --
syntax validity and a grep-based guard against the forbidden actions this
script must never perform (HPA apply/delete, scaling demo-app itself,
building images, starting traffic/training, or touching Minikube/Docker
Desktop settings directly). Does not mock kubectl or execute the script
against a real cluster -- that is exercised manually per the phase's own
execution rules (no background-process monitoring, bounded waits only).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "v2_cluster_prepare.sh"


def test_script_exists_and_is_executable():
    assert SCRIPT_PATH.is_file()
    assert SCRIPT_PATH.stat().st_mode & 0o111  # at least one executable bit set


def test_script_has_bash_strict_mode():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "set -euo pipefail" in source


def test_script_syntax_is_valid_bash():
    result = subprocess.run(["bash", "-n", str(SCRIPT_PATH)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_script_never_applies_or_deletes_an_hpa():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert not re.search(r"kubectl\s+apply\s+.*hpa", source, re.IGNORECASE)
    assert not re.search(r"kubectl\s+delete\s+hpa", source, re.IGNORECASE)


def test_script_never_scales_demo_app_itself():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "scale deployment demo-app" not in source
    assert "scale deployment/demo-app" not in source


def test_script_never_builds_images_or_starts_traffic_or_training():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    for forbidden in ("docker build", "docker-build", "traffic_orchestrator", "train_ppo", "evaluate_ppo"):
        assert forbidden not in source, f"found forbidden reference: {forbidden!r}"


def test_script_never_touches_minikube_or_helm_lifecycle():
    code_lines = "\n".join(
        line for line in SCRIPT_PATH.read_text(encoding="utf-8").splitlines() if not line.strip().startswith("#")
    )
    assert not re.search(r"minikube\s+(stop|delete)\b", code_lines)
    assert not re.search(r"helm\s+(install|upgrade|uninstall)\b", code_lines)


def test_script_uses_jsonpath_not_jq():
    """Project standard: kubectl JSONPath, never `jq -r` (per this phase's
    own explicit instruction)."""
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "jq -r" not in source
    assert "jsonpath=" in source


def test_script_supports_restore_grafana_flag():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "--restore-grafana" in source


def test_script_reuses_existing_configure_metrics_server():
    """Must delegate to configure_metrics_server.sh's own idempotent
    --metric-resolution patch logic, never reimplement a raw `kubectl
    patch deployment metrics-server` call itself."""
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "configure_metrics_server.sh" in source
    assert "kubectl patch" not in source
