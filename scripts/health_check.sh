#!/usr/bin/env bash
# scripts/health_check.sh
# Smoke-tests the full stack.
# Returns exit code 0 only when all required components are healthy.
# Called by CI, the experiment runner, and the demo workflow.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

APP_NS=autoscaler-app
MONITORING_NS=monitoring

PASS=0
FAIL=0

check() {
    local label="$1"
    local result="$2"   # "ok" or anything else
    if [[ "${result}" == "ok" ]]; then
        echo "  [PASS] ${label}"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] ${label}: ${result}"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== Stack health check ==="
echo ""

# Minikube running
MINIKUBE_HOST=$(minikube status --format='{{.Host}}' 2>/dev/null || echo "Unknown")
check "Minikube host" "$([ "${MINIKUBE_HOST}" = "Running" ] && echo ok || echo "${MINIKUBE_HOST}")"

# Node ready
NODE_READY=$(kubectl get nodes --no-headers 2>/dev/null | awk '{print $2}' | head -1)
check "Node Ready" "$([ "${NODE_READY}" = "Ready" ] && echo ok || echo "${NODE_READY}")"

# Metrics Server
METRICS_POD=$(kubectl get pods -n kube-system -l k8s-app=metrics-server --no-headers 2>/dev/null | awk '{print $3}' | head -1)
check "Metrics Server" "$([ "${METRICS_POD}" = "Running" ] && echo ok || echo "${METRICS_POD:-not found}")"

# kubectl top nodes works
if kubectl top nodes &>/dev/null; then
    check "kubectl top nodes" ok
else
    check "kubectl top nodes" "metrics API unavailable"
fi

# Application deployment
APP_READY=$(kubectl get deploy demo-app -n "${APP_NS}" --no-headers 2>/dev/null | awk '{print $4}' | head -1)
APP_DESIRED=$(kubectl get deploy demo-app -n "${APP_NS}" --no-headers 2>/dev/null | awk '{print $2}' | head -1)
check "App deployment ready" "$([ "${APP_READY}" = "${APP_DESIRED}" ] && [ -n "${APP_READY}" ] && echo ok || echo "${APP_READY}/${APP_DESIRED}")"

# Application /health endpoint
APP_URL=$(minikube service demo-app -n "${APP_NS}" --url 2>/dev/null | head -1 || echo "")
if [[ -n "${APP_URL}" ]]; then
    HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "${APP_URL}/health" 2>/dev/null || echo "000")
    check "App /health endpoint" "$([ "${HTTP_CODE}" = "200" ] && echo ok || echo "HTTP ${HTTP_CODE}")"
else
    check "App /health endpoint" "service URL unavailable"
fi

# Prometheus pod
PROM_POD=$(kubectl get pods -n "${MONITORING_NS}" -l "app.kubernetes.io/name=prometheus" --no-headers 2>/dev/null | awk '{print $3}' | head -1)
check "Prometheus pod" "$([ "${PROM_POD}" = "Running" ] && echo ok || echo "${PROM_POD:-not found}")"

# Grafana pod
GRAFANA_POD=$(kubectl get pods -n "${MONITORING_NS}" -l "app.kubernetes.io/name=grafana" --no-headers 2>/dev/null | awk '{print $3}' | head -1)
check "Grafana pod" "$([ "${GRAFANA_POD}" = "Running" ] && echo ok || echo "${GRAFANA_POD:-not found}")"

# persistent-data directories writable
for subdir in metrics traffic rl hpa logs models metadata backups; do
    DIR="${PROJECT_ROOT}/persistent-data/${subdir}"
    mkdir -p "${DIR}"
    if touch "${DIR}/.write_test" 2>/dev/null; then
        rm -f "${DIR}/.write_test"
    else
        check "persistent-data/${subdir} writable" "permission denied"
    fi
done
check "persistent-data/ writable" ok

echo ""
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="

if [[ "${FAIL}" -gt 0 ]]; then
    echo ""
    echo "WARN: Some checks failed. Review the output above."
    exit 1
fi

echo ""
echo "Stack is healthy. Proceed with experiments."
