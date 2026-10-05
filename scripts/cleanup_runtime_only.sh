#!/usr/bin/env bash
# scripts/cleanup_runtime_only.sh
# Removes ONLY replaceable runtime Kubernetes resources.
# NEVER removes persistent-data/, datasets/, trained models, or experiment results.
# Prints exactly what will be removed before doing it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

APP_NS=autoscaler-app

echo "=== Runtime cleanup (persistent data is NOT touched) ==="
echo ""
echo "The following resources WILL be removed:"
echo "  - Deployment: demo-app in ${APP_NS}"
echo "  - HPA: demo-app-hpa in ${APP_NS} (if present)"
echo "  - ConfigMap: demo-app-config in ${APP_NS}"
echo "  - Service: demo-app in ${APP_NS}"
echo "  - ServiceMonitor: demo-app in ${APP_NS} (if present)"
echo ""
echo "The following will NOT be touched:"
echo "  - persistent-data/"
echo "  - experiments/results/"
echo "  - experiments/reports/"
echo "  - datasets/"
echo "  - Prometheus/Grafana (use a separate command to uninstall monitoring)"
echo "  - Minikube cluster"
echo ""

read -rp "Proceed with cleanup? [y/N] " confirm
if [[ "${confirm}" != "y" && "${confirm}" != "Y" ]]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo ""
echo "Removing HPA if present..."
kubectl delete hpa demo-app-hpa -n "${APP_NS}" --ignore-not-found

echo "Removing Deployment..."
kubectl delete deployment demo-app -n "${APP_NS}" --ignore-not-found

echo "Removing Service..."
kubectl delete service demo-app -n "${APP_NS}" --ignore-not-found

echo "Removing ConfigMap..."
kubectl delete configmap demo-app-config -n "${APP_NS}" --ignore-not-found

echo "Removing ServiceMonitor if present..."
kubectl delete servicemonitor demo-app -n "${APP_NS}" --ignore-not-found 2>/dev/null || true

echo ""
echo "=== Cleanup complete ==="
echo ""
echo "persistent-data/ and experiment results are intact."
echo "To redeploy the application: ./scripts/deploy_app.sh"
