#!/usr/bin/env bash
# scripts/apply_metrics_server_permanent.sh
# V2 Phase 5A / CORR-60: NOT RUN by this phase -- prepared for a future,
# explicitly-decided switch from Minikube's `metrics-server` ADDON (which
# resets to its own packaged --metric-resolution=60s default on every
# `minikube addons enable metrics-server` / cluster recreation) to a
# self-managed Deployment (k8s/metrics-server/metrics-server.yaml) that
# persists this project's selected 15s resolution permanently.
#
# BEFORE running this script: read the addon's LIVE image reference
# (`kubectl get deployment metrics-server -n kube-system -o
# jsonpath='{.spec.template.spec.containers[0].image}'`) and replace the
# PLACEHOLDER_IMAGE_TAG_UNVERIFIED_OFFLINE placeholder in
# k8s/metrics-server/metrics-server.yaml with it -- this script does NOT
# do that substitution automatically (never silently assumes an image).
#
# Sequence: disable the addon -> apply the self-managed manifest -> bounded
# rollout wait -> verify the live args match this project's selected
# resolution -> verify `kubectl top nodes` actually returns data (proves
# the self-managed instance is genuinely serving the metrics API, not just
# "Running").
#
# This script NEVER touches demo-app, any HPA object, or Prometheus --
# verified offline by tests/unit/test_apply_metrics_server_permanent_script.py.
#
# ROLLBACK (if the self-managed Deployment fails to come up healthy):
#   kubectl delete -f k8s/metrics-server/metrics-server.yaml --ignore-not-found
#   minikube addons enable metrics-server
#   ./scripts/configure_metrics_server.sh 15s
# This restores the original addon-managed metrics-server at this
# project's selected 15s resolution, exactly as it was before this script
# ran.
#
# Usage:
#   ./scripts/apply_metrics_server_permanent.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${SCRIPT_DIR}/../k8s/metrics-server/metrics-server.yaml"
NAMESPACE=kube-system
DEPLOYMENT=metrics-server
EXPECTED_RESOLUTION="--metric-resolution=15s"

echo "=== apply_metrics_server_permanent.sh ==="

if grep -q "PLACEHOLDER_IMAGE_TAG_UNVERIFIED_OFFLINE" "${MANIFEST}"; then
    echo "ERROR: ${MANIFEST} still contains the unresolved image placeholder." >&2
    echo "  Read the live addon image first:" >&2
    echo "    kubectl get deployment metrics-server -n kube-system -o jsonpath='{.spec.template.spec.containers[0].image}'" >&2
    echo "  and substitute it into the manifest before running this script." >&2
    exit 1
fi

echo "--- Step 1: disable the Minikube metrics-server addon ---"
minikube addons disable metrics-server

echo ""
echo "--- Step 2: apply the self-managed manifest ---"
kubectl apply -f "${MANIFEST}"

echo ""
echo "--- Step 3: bounded rollout wait ---"
kubectl rollout status deployment/"${DEPLOYMENT}" -n "${NAMESPACE}" --timeout=90s

echo ""
echo "--- Step 4: verify live args match the selected resolution ---"
LIVE_ARGS="$(kubectl get deployment "${DEPLOYMENT}" -n "${NAMESPACE}" \
    -o jsonpath='{.spec.template.spec.containers[0].args}')"
echo "  Live args: ${LIVE_ARGS}"
if [[ "${LIVE_ARGS}" != *"${EXPECTED_RESOLUTION}"* ]]; then
    echo "ERROR: live args do not contain ${EXPECTED_RESOLUTION}." >&2
    echo "  Rollback: see this script's header comment." >&2
    exit 1
fi

echo ""
echo "--- Step 5: verify kubectl top actually works ---"
sleep 5
if ! kubectl top nodes &>/dev/null; then
    echo "ERROR: 'kubectl top nodes' failed -- the self-managed instance is not yet serving the metrics API." >&2
    echo "  Rollback: see this script's header comment." >&2
    exit 1
fi
kubectl top nodes

echo ""
echo "=== metrics-server is now self-managed, permanently at ${EXPECTED_RESOLUTION} ==="
