#!/usr/bin/env bash
# scripts/run_calibration_level.sh
# Execute a single-level capacity calibration run with synchronized traffic generation and telemetry collection.

set -euo pipefail

RPS="${1:-1}"
DURATION="${2:-90}"
INTENSITY="3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PROFILE_PATH="configs/traffic_profiles/calibration/cal_${RPS}rps_${DURATION}s.yaml"

if [[ ! -f "${PROFILE_PATH}" ]]; then
    echo "ERROR: Calibration profile not found at ${PROFILE_PATH}" >&2
    exit 1
fi

# Verify port-forwards
echo "=== Pre-run Port-Forward Health Checks ==="
if ! curl -s -f http://localhost:8080/health > /dev/null; then
    echo "ERROR: demo-app port-forward (http://localhost:8080) is unreachable." >&2
    exit 1
fi
if ! curl -s -f http://localhost:9090/-/ready > /dev/null; then
    echo "ERROR: Prometheus port-forward (http://localhost:9090) is unreachable." >&2
    exit 1
fi
echo "Port-forwards (8080 and 9090) are responsive."

# Verify demo-app replica count and status
echo "=== Pre-run demo-app Status ==="
DESIRED_REPLICAS="$(kubectl get deployment -n autoscaler-app demo-app -o jsonpath='{.spec.replicas}')"
READY_REPLICAS="$(kubectl get deployment -n autoscaler-app demo-app -o jsonpath='{.status.readyReplicas}')"
POD_NAME="$(kubectl get pods -n autoscaler-app -l app=demo-app -o jsonpath='{.items[0].metadata.name}')"
PRE_RESTARTS="$(kubectl get pod -n autoscaler-app "${POD_NAME}" -o jsonpath='{.status.containerStatuses[0].restartCount}')"

echo "Deployment: desired=${DESIRED_REPLICAS}, ready=${READY_REPLICAS}"
echo "Pod: ${POD_NAME} (restarts=${PRE_RESTARTS})"

if [[ "${DESIRED_REPLICAS}" != "1" || "${READY_REPLICAS}" != "1" ]]; then
    echo "ERROR: Calibration requires exactly 1 desired and 1 ready replica." >&2
    exit 1
fi

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EXP_ID="cal_single_r${RPS}_i3_${TIMESTAMP}"

echo "=========================================================="
echo "Starting Calibration Run: ${EXP_ID}"
echo "Target RPS: ${RPS} | Duration: ${DURATION}s | Intensity: ${INTENSITY}"
echo "=========================================================="

# 1. Start metrics collector in background
echo "Starting metrics collector (mode=test, duration=${DURATION}s)..."
.venv/bin/python -m collectors.metrics_collector \
    --experiment-id "${EXP_ID}" \
    --mode test \
    --traffic-profile "cal_${RPS}rps_${DURATION}s" \
    --duration-seconds "${DURATION}" \
    --seed 42 &
COLLECTOR_PID=$!

# Small pause to allow collector initialization
sleep 0.5

# 2. Start traffic generator in foreground
echo "Starting traffic generator (deterministic, rate=${RPS} RPS, duration=${DURATION}s)..."
.venv/bin/python -m loadgen.traffic_generator \
    --profile "${PROFILE_PATH}" \
    --experiment-id "${EXP_ID}" \
    --target-url http://localhost:8080 \
    --arrival-process deterministic \
    --seed 42

# 3. Wait for metrics collector
echo "Waiting for metrics collector to finalize..."
wait "${COLLECTOR_PID}"

echo "Run ${EXP_ID} completed."

# 4. Verify artifact persistence
echo "=== Verifying Artifacts ==="
TRAFFIC_DIR="persistent-data/traffic/${EXP_ID}"
METRICS_DIR="persistent-data/metrics/${EXP_ID}"

for f in "${TRAFFIC_DIR}/requests.csv" "${TRAFFIC_DIR}/traffic_summary.csv" "${TRAFFIC_DIR}/metadata.json" \
         "${METRICS_DIR}/metrics.csv" "${METRICS_DIR}/metadata.json" "${METRICS_DIR}/events.jsonl"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: Missing expected artifact: ${f}" >&2
        exit 1
    fi
    echo "Verified: ${f} ($(wc -l < "${f}") lines)"
done

# 5. Check post-run health
POST_RESTARTS="$(kubectl get pod -n autoscaler-app "${POD_NAME}" -o jsonpath='{.status.containerStatuses[0].restartCount}')"
echo "Post-run pod restarts: ${POST_RESTARTS} (pre-run: ${PRE_RESTARTS})"
if [[ "${POST_RESTARTS}" != "${PRE_RESTARTS}" ]]; then
    echo "WARNING: Pod restarted during calibration run!" >&2
fi

echo "Calibration level ${RPS} RPS finished successfully. Experiment ID: ${EXP_ID}"
