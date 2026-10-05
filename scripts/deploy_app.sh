#!/usr/bin/env bash
# scripts/deploy_app.sh
# Builds the FastAPI demo application Docker image inside Minikube's Docker daemon
# and deploys it to Kubernetes using the manifests in k8s/app/.
# Idempotent: re-running applies updated manifests and rebuilds the image.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

APP_NS=autoscaler-app
IMAGE_NAME=rl-autoscaler-demo-app
IMAGE_TAG=local

echo "=== Deploying demo application ==="
echo "  Namespace: ${APP_NS}"
echo "  Image:     ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""

# Verify cluster is reachable
if ! kubectl cluster-info &>/dev/null; then
    echo "ERROR: Cannot reach Kubernetes cluster."
    exit 1
fi

# Apply namespace
kubectl apply -f "${PROJECT_ROOT}/k8s/namespaces/app-namespace.yaml"

# Build inside Minikube's Docker daemon so the image is available to pods
# without requiring a registry.
echo "Building Docker image inside Minikube's Docker daemon..."
eval "$(minikube docker-env)"
docker build \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    "${PROJECT_ROOT}/app"

echo "Image built: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""

# Apply Kubernetes manifests
echo "Applying Kubernetes manifests..."
kubectl apply -f "${PROJECT_ROOT}/k8s/app/configmap.yaml"
kubectl apply -f "${PROJECT_ROOT}/k8s/app/deployment.yaml"
kubectl apply -f "${PROJECT_ROOT}/k8s/app/service.yaml"

# Apply service-monitor only if Prometheus CRDs exist
if kubectl get crd servicemonitors.monitoring.coreos.com &>/dev/null; then
    kubectl apply -f "${PROJECT_ROOT}/k8s/app/service-monitor.yaml"
    echo "ServiceMonitor applied."
else
    echo "WARNING: ServiceMonitor CRD not found. Run install_monitoring.sh first."
fi

echo ""
echo "=== Waiting for deployment rollout ==="
kubectl rollout status deployment/demo-app -n "${APP_NS}" --timeout=120s

echo ""
echo "=== Verifying application ==="
kubectl get deploy,pods,svc -n "${APP_NS}"

echo ""
APP_URL=$(minikube service demo-app -n "${APP_NS}" --url 2>/dev/null | head -1)
if [[ -n "${APP_URL}" ]]; then
    echo "Application URL: ${APP_URL}"
    echo "Testing /health endpoint..."
    if curl -sf "${APP_URL}/health" | grep -q "ok"; then
        echo "  /health: OK"
    else
        echo "  WARNING: /health did not return expected response."
        echo "  Check: kubectl logs -n ${APP_NS} -l app=demo-app"
    fi
else
    echo "  Could not determine service URL automatically."
    echo "  Use: minikube service demo-app -n ${APP_NS} --url"
fi

echo ""
echo "=== Application deployment complete ==="
