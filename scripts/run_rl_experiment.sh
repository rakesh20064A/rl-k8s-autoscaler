#!/usr/bin/env bash
# scripts/run_rl_experiment.sh
# Runs a complete RL experiment: deletes HPA, resets replicas, starts RL agent,
# starts traffic, waits for completion, saves results.
# Usage: ./scripts/run_rl_experiment.sh <traffic_profile.yaml> [--seed SEED] [--run-id RUN_ID]

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

echo "=== RL Experiment ==="
echo "  Profile:   ${TRAFFIC_PROFILE}"
echo "  Namespace: ${APP_NS}"
echo ""

# Step 1: Health check
"${SCRIPT_DIR}/health_check.sh"

# Step 2: Controller exclusivity — disable HPA before running RL
echo "Removing HPA to prevent conflict with RL agent..."
kubectl delete hpa demo-app-hpa -n "${APP_NS}" --ignore-not-found

# Step 3: Reset to initial replicas
INITIAL_REPLICAS=1
echo "Resetting demo-app to ${INITIAL_REPLICAS} replica(s)..."
kubectl scale deployment demo-app --replicas="${INITIAL_REPLICAS}" -n "${APP_NS}"
kubectl rollout status deployment/demo-app -n "${APP_NS}" --timeout=60s

# Step 4: Run the experiment via the Python orchestrator
echo "Launching RL experiment orchestrator..."
"${VENV_PYTHON}" -m experiments.run \
    --controller rl \
    --traffic "${TRAFFIC_PROFILE}" \
    --config "${PROJECT_ROOT}/configs/experiment.yaml" \
    --rl-config "${PROJECT_ROOT}/configs/rl.yaml" \
    "${@:2}"

echo ""
echo "=== RL experiment complete. Results in persistent-data/rl/ ==="
