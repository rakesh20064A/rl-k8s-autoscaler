#!/usr/bin/env bash
# scripts/start_minikube.sh
# Starts Minikube with conservative resource allocation suitable for an 8 GB Mac.
# If a Minikube cluster already exists, inspects it before deciding to start or recreate.
# Never runs minikube delete automatically.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Conservative resource allocation: 2 CPUs, 3072 MB RAM
# 3072 MB matches the existing saved Minikube profile and leaves ~840 MB headroom
# inside Docker Desktop's ~3916 MB VM for Docker overhead, Prometheus, and Grafana.
# Increase only if post-monitoring measurements show memory pressure.
MINIKUBE_CPUS=2
MINIKUBE_MEMORY=3072
MINIKUBE_DRIVER=docker
KUBERNETES_VERSION=v1.35.1

echo "=== Minikube startup ==="
echo "  Driver:  ${MINIKUBE_DRIVER}"
echo "  CPUs:    ${MINIKUBE_CPUS}"
echo "  Memory:  ${MINIKUBE_MEMORY} MB"
echo "  k8s:     ${KUBERNETES_VERSION}"
echo ""

# Verify Docker daemon is reachable before attempting to start Minikube
if ! docker info &>/dev/null; then
    echo "ERROR: Docker daemon is not running."
    echo "  Start Docker Desktop and try again."
    exit 1
fi

# Check if a Minikube profile already exists
MINIKUBE_STATUS=$(minikube status --format='{{.Host}}' 2>/dev/null || echo "None")

if [[ "${MINIKUBE_STATUS}" == "Running" ]]; then
    echo "Minikube is already running."
    minikube status
    echo ""
    echo "If you want different resource limits, delete the cluster manually:"
    echo "  minikube delete"
    echo "Then re-run this script."
elif [[ "${MINIKUBE_STATUS}" == "Stopped" ]]; then
    echo "Minikube exists but is stopped. Starting..."
    minikube start
else
    echo "No running Minikube cluster found. Creating one..."
    minikube start \
        --driver="${MINIKUBE_DRIVER}" \
        --cpus="${MINIKUBE_CPUS}" \
        --memory="${MINIKUBE_MEMORY}" \
        --kubernetes-version="${KUBERNETES_VERSION}" \
        --addons=metrics-server
fi

echo ""
echo "=== Validating cluster ==="
kubectl get nodes -o wide
echo ""
kubectl get pods -A | head -20
echo ""

echo "=== Minikube startup complete ==="
echo ""
echo "Next steps:"
echo "  ./scripts/install_monitoring.sh"
echo "  ./scripts/deploy_app.sh"
