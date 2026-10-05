#!/usr/bin/env bash
# scripts/install_monitoring.sh
# Installs Prometheus via kube-prometheus-stack Helm chart
# using declarative values from k8s/monitoring/values.yaml.
# Resource-conscious configuration appropriate for 2 CPU / 3072 MB Minikube.
# Idempotent: safe to re-run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MONITORING_NS=monitoring
HELM_RELEASE=kube-prom-stack
CHART=prometheus-community/kube-prometheus-stack
CHART_VERSION="82.9.0"
VALUES_FILE="${PROJECT_ROOT}/k8s/monitoring/values.yaml"

echo "=== Installing Prometheus Monitoring Stack ==="
echo "  Namespace:     ${MONITORING_NS}"
echo "  Helm release:  ${HELM_RELEASE}"
echo "  Chart:         ${CHART}:${CHART_VERSION}"
echo "  Values file:   ${VALUES_FILE}"
echo ""

# Verify cluster is reachable
if ! kubectl cluster-info &>/dev/null; then
    echo "ERROR: Cannot reach Kubernetes cluster."
    echo "  Run ./scripts/start_minikube.sh first."
    exit 1
fi

# Verify values file exists
if [[ ! -f "${VALUES_FILE}" ]]; then
    echo "ERROR: Values file not found at ${VALUES_FILE}"
    exit 1
fi

# Add Helm repo if missing
if ! helm repo list 2>/dev/null | grep -q prometheus-community; then
    echo "Adding prometheus-community Helm repo..."
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
fi
helm repo update prometheus-community 2>&1 | tail -3

# Create namespace
kubectl create namespace "${MONITORING_NS}" --dry-run=client -o yaml | kubectl apply -f -

if helm status "${HELM_RELEASE}" -n "${MONITORING_NS}" &>/dev/null; then
    echo "Helm release ${HELM_RELEASE} already exists. Upgrading..."
    helm upgrade "${HELM_RELEASE}" "${CHART}" \
        --version "${CHART_VERSION}" \
        -f "${VALUES_FILE}" \
        -n "${MONITORING_NS}" \
        --wait --timeout=8m
else
    echo "Installing ${HELM_RELEASE}..."
    helm install "${HELM_RELEASE}" "${CHART}" \
        --version "${CHART_VERSION}" \
        -f "${VALUES_FILE}" \
        -n "${MONITORING_NS}" \
        --wait --timeout=8m
fi

echo ""
echo "=== Verifying monitoring pods ==="
kubectl get pods -n "${MONITORING_NS}"

echo ""
echo "=== Access instructions ==="
echo "  Prometheus: kubectl port-forward -n ${MONITORING_NS} svc/${HELM_RELEASE}-prometheus 9090:9090"
echo ""
echo "=== Monitoring installation complete ==="
