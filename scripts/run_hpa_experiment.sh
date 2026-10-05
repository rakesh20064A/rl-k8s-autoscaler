#!/usr/bin/env bash
# scripts/run_hpa_experiment.sh
# Runs a complete HPA experiment: resets replicas, enables HPA, starts collector,
# starts traffic generator, waits for completion, saves results, disables HPA.
# Usage: ./scripts/run_hpa_experiment.sh <traffic_profile.yaml> [--seed SEED] [--run-id RUN_ID]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
APP_NS=autoscaler-app
TRAFFIC_PROFILE="${1:-${PROJECT_ROOT}/configs/traffic_profiles/mixed.yaml}"

if [[ ! -f "${TRAFFIC_PROFILE}" ]]; then
    echo "ERROR: Traffic profile not found: ${TRAFFIC_PROFILE}"
    exit 1
fi

if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "ERROR: .venv not found. Run ./scripts/bootstrap.sh first."
    exit 1
fi

echo "=== HPA Experiment ==="
echo "  Profile: ${TRAFFIC_PROFILE}"
echo "  Namespace: ${APP_NS}"
echo ""

# Step 1: Health check
echo "Running health check..."
"${SCRIPT_DIR}/health_check.sh"

# Step 2: Ensure RL agent is NOT running (controller exclusivity)
echo "Verifying RL agent is not active..."
if kubectl get deployment rl-agent -n "${APP_NS}" &>/dev/null; then
    echo "WARNING: RL agent deployment found. Scale it to 0 for a fair HPA experiment."
    kubectl scale deployment rl-agent --replicas=0 -n "${APP_NS}" 2>/dev/null || true
fi

# Step 3: Reset to initial replicas
INITIAL_REPLICAS=1
echo "Resetting demo-app to ${INITIAL_REPLICAS} replica(s)..."
kubectl scale deployment demo-app --replicas="${INITIAL_REPLICAS}" -n "${APP_NS}"
kubectl rollout status deployment/demo-app -n "${APP_NS}" --timeout=60s

# Step 4: Enable HPA
echo "Applying HPA..."
kubectl apply -f "${PROJECT_ROOT}/k8s/hpa/hpa.yaml"
sleep 5

# Step 5: Run the experiment via the Python orchestrator
echo "Launching experiment orchestrator..."
"${VENV_PYTHON}" -m experiments.run \
    --controller hpa \
    --traffic "${TRAFFIC_PROFILE}" \
    --config "${PROJECT_ROOT}/configs/experiment.yaml" \
    "${@:2}"

# Step 6: Summarize
echo ""
echo "=== HPA experiment complete. Results in persistent-data/hpa/ ==="
