#!/usr/bin/env bash
# scripts/run_hpa_tuning_trial.sh
#
# Runs exactly ONE HPA tuning/smoke trial: preflight checks, load-generator
# tunnel + Prometheus port-forward lifecycle, the persistent metrics
# collector, the HPA status collector, and the traffic generator (schedule
# REPLAY only -- never regenerates a schedule), followed by graceful cleanup
# and a run-level metadata record.
#
# Phase 11C.4B: the load generator reaches the app through the dedicated
# demo-app-loadtest NodePort Service via `minikube service --url`, not
# `kubectl port-forward svc/demo-app` (which pins traffic to one pod and
# never exercises Service load-balancing -- see CORR-21).
#
# This is deliberately narrow. It does NOT:
#   - know anything about RL
#   - run all 9 official tuning trials automatically (call it once per trial)
#   - choose a winner or compare candidates
#   - alter configs/hpa.yaml, k8s/hpa/hpa.yaml, or any workload profile
#   - change the live HPA target (use scripts/set_hpa_target.sh separately,
#     BEFORE invoking this script, then pass the resulting value via --target
#     so this script can verify it matches what's actually live)
#
# Usage:
#   ./scripts/run_hpa_tuning_trial.sh \
#       --experiment-id hpa_tune_t230_20260908T120000Z \
#       --target 230 \
#       --profile configs/traffic_profiles/hpa_tuning/hpa_tuning_v1.yaml \
#       --schedule configs/traffic_profiles/hpa_tuning/canonical_schedule.csv \
#       [--expected-hash <sha256>]   # defaults to reading <schedule>.sha256 if present
#       [--skip-target-check]        # only for smoke runs against whatever target is live

set -uo pipefail  # deliberately NOT -e: we want to reach cleanup/invalid-marking on failure

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

APP_NS="autoscaler-app"
MONITORING_NS="monitoring"
HPA_NAME="demo-app-hpa"
LOADTEST_SVC="demo-app-loadtest"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"

EXPERIMENT_ID=""
TARGET=""
PROFILE_PATH=""
SCHEDULE_PATH=""
EXPECTED_HASH=""
SKIP_TARGET_CHECK=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment-id) EXPERIMENT_ID="$2"; shift 2 ;;
        --target) TARGET="$2"; shift 2 ;;
        --profile) PROFILE_PATH="$2"; shift 2 ;;
        --schedule) SCHEDULE_PATH="$2"; shift 2 ;;
        --expected-hash) EXPECTED_HASH="$2"; shift 2 ;;
        --skip-target-check) SKIP_TARGET_CHECK=1; shift ;;
        *) echo "ERROR: unrecognized argument '$1'" >&2; exit 1 ;;
    esac
done

for required in EXPERIMENT_ID TARGET PROFILE_PATH SCHEDULE_PATH; do
    if [[ -z "${!required}" ]]; then
        echo "ERROR: --${required,,} is required." >&2
        exit 1
    fi
done

INVALID_LOG_DIR="persistent-data/logs"
mkdir -p "${INVALID_LOG_DIR}"

STARTED_APP_TUNNEL=0
STARTED_PROM_PORT_FORWARD=0
APP_TUNNEL_PID=""
APP_TUNNEL_URL=""
PROM_PF_PID=""
METRICS_COLLECTOR_PID=""
HPA_COLLECTOR_PID=""
RUN_STATUS="unknown"
INVALID_REASON=""

mark_invalid() {
    local reason="$1"
    RUN_STATUS="invalid"
    INVALID_REASON="${reason}"
    echo "ERROR: trial marked INVALID: ${reason}" >&2
}

cleanup() {
    echo "=== cleanup ==="
    if [[ -n "${METRICS_COLLECTOR_PID}" ]] && kill -0 "${METRICS_COLLECTOR_PID}" 2>/dev/null; then
        kill "${METRICS_COLLECTOR_PID}" 2>/dev/null || true
        wait "${METRICS_COLLECTOR_PID}" 2>/dev/null || true
    fi
    if [[ -n "${HPA_COLLECTOR_PID}" ]] && kill -0 "${HPA_COLLECTOR_PID}" 2>/dev/null; then
        kill "${HPA_COLLECTOR_PID}" 2>/dev/null || true
        wait "${HPA_COLLECTOR_PID}" 2>/dev/null || true
    fi
    if [[ "${STARTED_APP_TUNNEL}" -eq 1 && -n "${APP_TUNNEL_PID}" ]]; then
        kill "${APP_TUNNEL_PID}" 2>/dev/null || true
        wait "${APP_TUNNEL_PID}" 2>/dev/null || true
    fi
    if [[ "${STARTED_PROM_PORT_FORWARD}" -eq 1 && -n "${PROM_PF_PID}" ]]; then
        kill "${PROM_PF_PID}" 2>/dev/null || true
    fi

    local meta_path="${INVALID_LOG_DIR}/${EXPERIMENT_ID}_trial_metadata.json"
    cat > "${meta_path}" <<EOF
{
  "experiment_id": "${EXPERIMENT_ID}",
  "target": "${TARGET}",
  "profile": "${PROFILE_PATH}",
  "schedule": "${SCHEDULE_PATH}",
  "status": "${RUN_STATUS}",
  "invalid_reason": "${INVALID_REASON}",
  "recorded_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    echo "Trial metadata written: ${meta_path} (status=${RUN_STATUS})"
}
trap cleanup EXIT

echo "=== run_hpa_tuning_trial.sh ==="
echo "  Experiment ID: ${EXPERIMENT_ID}"
echo "  Target:        ${TARGET}%"
echo "  Profile:       ${PROFILE_PATH}"
echo "  Schedule:      ${SCHEDULE_PATH}"
echo ""

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
echo "=== Preflight ==="

if ! docker info &>/dev/null; then
    mark_invalid "Docker daemon unreachable"; exit 1
fi

if ! minikube status -p minikube 2>&1 | grep -q "host: Running"; then
    mark_invalid "Minikube not running"; exit 1
fi

RUNNING_KIND="$(docker ps --format '{{.Names}}' | grep -E '^(desktop-|kind-)' || true)"
if [[ -n "${RUNNING_KIND}" ]]; then
    mark_invalid "Unrelated Docker Desktop Kind cluster is running: ${RUNNING_KIND}"; exit 1
fi

if ! kubectl get nodes minikube -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' | grep -q "True"; then
    mark_invalid "Node minikube is not Ready"; exit 1
fi

MS_ARGS="$(kubectl get deployment -n kube-system metrics-server -o jsonpath='{.spec.template.spec.containers[0].args}' 2>&1)"
if ! echo "${MS_ARGS}" | grep -q "metric-resolution=15s"; then
    mark_invalid "Metrics Server is not at --metric-resolution=15s (got: ${MS_ARGS})"; exit 1
fi
MS_READY="$(kubectl get deployment -n kube-system metrics-server -o jsonpath='{.status.readyReplicas}' 2>&1)"
if [[ "${MS_READY}" != "1" ]]; then
    mark_invalid "Metrics Server not Ready (readyReplicas=${MS_READY})"; exit 1
fi

if ! kubectl get pods -n "${MONITORING_NS}" -l app.kubernetes.io/name=prometheus -o jsonpath='{.items[0].status.phase}' 2>&1 | grep -q "Running"; then
    mark_invalid "Prometheus pod not Running"; exit 1
fi

RL_PROC="$(ps aux | grep -iE "python.*(rl_agent|autoscal|ppo)" | grep -v grep || true)"
if [[ -n "${RL_PROC}" ]]; then
    mark_invalid "An RL/custom-controller-like process appears to be running: ${RL_PROC}"; exit 1
fi

HPA_COUNT="$(kubectl get hpa -n "${APP_NS}" --no-headers 2>/dev/null | wc -l | tr -d ' ')"
if [[ "${HPA_COUNT}" != "1" ]]; then
    mark_invalid "Expected exactly 1 HPA in ${APP_NS}, found ${HPA_COUNT}"; exit 1
fi

LIVE_TARGET="$(kubectl get hpa "${HPA_NAME}" -n "${APP_NS}" -o jsonpath='{.spec.metrics[0].resource.target.averageUtilization}' 2>&1)"
if [[ "${SKIP_TARGET_CHECK}" -eq 0 ]] && [[ "${LIVE_TARGET}" != "${TARGET}" ]]; then
    mark_invalid "Live HPA target is ${LIVE_TARGET}%, expected ${TARGET}%"; exit 1
fi
if [[ "${SKIP_TARGET_CHECK}" -eq 1 ]]; then
    echo "  (--skip-target-check set; live HPA target is ${LIVE_TARGET}%, recorded as-is)"
    TARGET="${LIVE_TARGET}"
fi

SCALE_UP_STAB="$(kubectl get hpa "${HPA_NAME}" -n "${APP_NS}" -o jsonpath='{.spec.behavior.scaleUp.stabilizationWindowSeconds}')"
SCALE_DOWN_STAB="$(kubectl get hpa "${HPA_NAME}" -n "${APP_NS}" -o jsonpath='{.spec.behavior.scaleDown.stabilizationWindowSeconds}')"
if [[ "${SCALE_UP_STAB}" != "0" || "${SCALE_DOWN_STAB}" != "120" ]]; then
    mark_invalid "HPA behavior does not match Phase 11B (scaleUp stab=${SCALE_UP_STAB}, scaleDown stab=${SCALE_DOWN_STAB})"; exit 1
fi

CPU_REQ="$(kubectl get deployment -n "${APP_NS}" demo-app -o jsonpath='{.spec.template.spec.containers[0].resources.requests.cpu}')"
CPU_LIM="$(kubectl get deployment -n "${APP_NS}" demo-app -o jsonpath='{.spec.template.spec.containers[0].resources.limits.cpu}')"
MEM_REQ="$(kubectl get deployment -n "${APP_NS}" demo-app -o jsonpath='{.spec.template.spec.containers[0].resources.requests.memory}')"
MEM_LIM="$(kubectl get deployment -n "${APP_NS}" demo-app -o jsonpath='{.spec.template.spec.containers[0].resources.limits.memory}')"
if [[ "${CPU_REQ}" != "100m" || "${CPU_LIM}" != "500m" || "${MEM_REQ}" != "64Mi" || "${MEM_LIM}" != "256Mi" ]]; then
    mark_invalid "demo-app resources changed: cpu=${CPU_REQ}/${CPU_LIM} mem=${MEM_REQ}/${MEM_LIM}"; exit 1
fi

MIN_R="$(kubectl get hpa "${HPA_NAME}" -n "${APP_NS}" -o jsonpath='{.spec.minReplicas}')"
MAX_R="$(kubectl get hpa "${HPA_NAME}" -n "${APP_NS}" -o jsonpath='{.spec.maxReplicas}')"
if [[ "${MIN_R}" != "1" || "${MAX_R}" != "8" ]]; then
    mark_invalid "Replica bounds changed: min=${MIN_R} max=${MAX_R}"; exit 1
fi

DESIRED="$(kubectl get deployment -n "${APP_NS}" demo-app -o jsonpath='{.spec.replicas}')"
CURRENT="$(kubectl get deployment -n "${APP_NS}" demo-app -o jsonpath='{.status.replicas}')"
READY="$(kubectl get deployment -n "${APP_NS}" demo-app -o jsonpath='{.status.readyReplicas}')"
AVAILABLE="$(kubectl get deployment -n "${APP_NS}" demo-app -o jsonpath='{.status.availableReplicas}')"
if [[ "${DESIRED}" != "1" || "${CURRENT}" != "1" || "${READY}" != "1" || "${AVAILABLE}" != "1" ]]; then
    mark_invalid "demo-app did not start at 1/1/1/1 (desired=${DESIRED} current=${CURRENT} ready=${READY} available=${AVAILABLE})"; exit 1
fi

if [[ ! -f "${SCHEDULE_PATH}" ]]; then
    mark_invalid "Schedule file not found: ${SCHEDULE_PATH}"; exit 1
fi
ACTUAL_HASH="$(shasum -a 256 "${SCHEDULE_PATH}" | awk '{print $1}')"
if [[ -z "${EXPECTED_HASH}" && -f "${SCHEDULE_PATH}.sha256" ]]; then
    EXPECTED_HASH="$(awk '{print $1}' "${SCHEDULE_PATH}.sha256")"
fi
if [[ -n "${EXPECTED_HASH}" && "${ACTUAL_HASH}" != "${EXPECTED_HASH}" ]]; then
    mark_invalid "Schedule hash mismatch: expected ${EXPECTED_HASH}, got ${ACTUAL_HASH}"; exit 1
fi
echo "  Schedule SHA-256 verified: ${ACTUAL_HASH}"

for dir in "persistent-data/traffic/${EXPERIMENT_ID}" "persistent-data/metrics/${EXPERIMENT_ID}" "persistent-data/hpa/${EXPERIMENT_ID}"; do
    if [[ -d "${dir}" ]]; then
        mark_invalid "Output directory already exists: ${dir}"; exit 1
    fi
done

# Lightweight distribution preflight (Phase 11C.4B): confirm the dedicated
# load-generator Service actually fronts the same pods as demo-app, has no
# accidental session affinity, and its endpoint count matches Ready replicas
# -- NOT a full distribution test, just a cheap sanity check before traffic.
LOADTEST_SELECTOR="$(kubectl get svc "${LOADTEST_SVC}" -n "${APP_NS}" -o jsonpath='{.spec.selector}' 2>&1)"
DEMO_SELECTOR="$(kubectl get svc demo-app -n "${APP_NS}" -o jsonpath='{.spec.selector}' 2>&1)"
if [[ "${LOADTEST_SELECTOR}" != "${DEMO_SELECTOR}" ]]; then
    mark_invalid "${LOADTEST_SVC} selector (${LOADTEST_SELECTOR}) does not match demo-app selector (${DEMO_SELECTOR})"; exit 1
fi

LOADTEST_AFFINITY="$(kubectl get svc "${LOADTEST_SVC}" -n "${APP_NS}" -o jsonpath='{.spec.sessionAffinity}' 2>&1)"
if [[ -n "${LOADTEST_AFFINITY}" && "${LOADTEST_AFFINITY}" != "None" ]]; then
    mark_invalid "${LOADTEST_SVC} sessionAffinity is '${LOADTEST_AFFINITY}', expected None"; exit 1
fi

ENDPOINT_COUNT="$(kubectl get endpoints "${LOADTEST_SVC}" -n "${APP_NS}" -o jsonpath='{.subsets[0].addresses[*].ip}' 2>/dev/null | wc -w | tr -d ' ')"
if [[ "${ENDPOINT_COUNT}" != "${READY}" ]]; then
    mark_invalid "${LOADTEST_SVC} endpoint count (${ENDPOINT_COUNT}) does not match Ready replica count (${READY})"; exit 1
fi
echo "  Distribution preflight OK: ${LOADTEST_SVC} selector matches demo-app, sessionAffinity=None, endpoints=${ENDPOINT_COUNT} (Ready=${READY})"

echo "Preflight passed."
echo ""

# ---------------------------------------------------------------------------
# Load-generator tunnel + Prometheus port-forward lifecycle
# ---------------------------------------------------------------------------
# Phase 11C.4B correction: the load generator no longer reaches the app via
# `kubectl port-forward svc/demo-app`. Per Kubernetes documentation,
# `kubectl port-forward service/<name>` selects ONE backing pod and tunnels
# to it for the life of the port-forward process -- it does NOT traverse the
# normal Service data plane (kube-proxy) that load-balances across all Ready
# replicas. That pinned all HPA-tuning traffic to a single pod, starving any
# newly-scaled-up replica (see task_outputs/phase_11c4b_multi_replica_traffic_delivery_correction.md
# and CORR-21). The load generator now reaches the app through the dedicated
# `demo-app-loadtest` NodePort Service via `minikube service --url`, a
# genuine Service-load-balanced path. The Prometheus port-forward is
# unaffected (observer path only, never a load-balancing concern) and keeps
# its original reuse-if-already-up behavior.
echo "=== Load-generator tunnel (${LOADTEST_SVC}) ==="
APP_TUNNEL_LOG="/tmp/mk_svc_${EXPERIMENT_ID}.log"
minikube service "${LOADTEST_SVC}" -n "${APP_NS}" --url >"${APP_TUNNEL_LOG}" 2>&1 &
APP_TUNNEL_PID=$!
STARTED_APP_TUNNEL=1

for _ in $(seq 1 30); do
    APP_TUNNEL_URL="$(grep -Eo 'http://[0-9.]+:[0-9]+' "${APP_TUNNEL_LOG}" 2>/dev/null | head -n1)"
    if [[ -n "${APP_TUNNEL_URL}" ]]; then
        break
    fi
    sleep 1
done

if [[ -z "${APP_TUNNEL_URL}" ]]; then
    mark_invalid "minikube service tunnel for ${LOADTEST_SVC} did not produce a URL within 30s (see ${APP_TUNNEL_LOG})"; exit 1
fi
if ! curl -s -f "${APP_TUNNEL_URL}/health" > /dev/null 2>&1; then
    mark_invalid "Could not reach ${LOADTEST_SVC} through tunnel at ${APP_TUNNEL_URL}"; exit 1
fi
echo "  Tunnel established: ${APP_TUNNEL_URL} -> svc/${LOADTEST_SVC} (pid ${APP_TUNNEL_PID})"

if curl -s -f http://localhost:9090/-/ready > /dev/null 2>&1; then
    echo "  Prometheus port-forward already active on :9090"
else
    kubectl port-forward -n "${MONITORING_NS}" svc/kube-prom-stack-kube-prome-prometheus 9090:9090 >/tmp/pf_prom_${EXPERIMENT_ID}.log 2>&1 &
    PROM_PF_PID=$!
    STARTED_PROM_PORT_FORWARD=1
    sleep 3
    if ! curl -s -f http://localhost:9090/-/ready > /dev/null 2>&1; then
        mark_invalid "Could not establish Prometheus port-forward"; exit 1
    fi
    echo "  Started Prometheus port-forward (pid ${PROM_PF_PID})"
fi
echo ""

# ---------------------------------------------------------------------------
# Compute profile total duration (offline, no dispatch) for collector durations
# ---------------------------------------------------------------------------
TOTAL_DURATION="$("${VENV_PYTHON}" -c "
from loadgen.traffic_generator import parse_traffic_profile
from pathlib import Path
print(int(parse_traffic_profile(Path('${PROFILE_PATH}')).total_duration))
")"
COLLECTOR_DURATION=$((TOTAL_DURATION + 30))
echo "Profile total duration: ${TOTAL_DURATION}s -> collectors run for ${COLLECTOR_DURATION}s"
echo ""

# ---------------------------------------------------------------------------
# Launch collectors (background) then traffic generator (foreground, replay)
# ---------------------------------------------------------------------------
echo "=== Starting collectors ==="
"${VENV_PYTHON}" -m collectors.metrics_collector \
    --experiment-id "${EXPERIMENT_ID}" \
    --mode test \
    --traffic-profile "$(basename "${PROFILE_PATH}" .yaml)" \
    --duration-seconds "${COLLECTOR_DURATION}" \
    --seed 42 &
METRICS_COLLECTOR_PID=$!
echo "  metrics_collector started (pid ${METRICS_COLLECTOR_PID})"

"${VENV_PYTHON}" -m collectors.hpa_status_collector \
    --experiment-id "${EXPERIMENT_ID}" \
    --hpa-name "${HPA_NAME}" \
    --namespace "${APP_NS}" \
    --sampling-interval-seconds 5 \
    --duration-seconds "${COLLECTOR_DURATION}" &
HPA_COLLECTOR_PID=$!
echo "  hpa_status_collector started (pid ${HPA_COLLECTOR_PID})"

sleep 1

echo ""
echo "=== Starting traffic generator (schedule REPLAY, no regeneration) ==="
# --connection-mode fresh (Phase 11C.4B): required so kube-proxy actually
# load-balances each request across all Ready backend pods -- httpx's
# default keep-alive connection reuse was empirically shown to pin ~99% of
# traffic to a single pod even through a genuinely load-balanced Service
# (see the phase_11c4b report/CORR-21 distribution-smoke results).
"${VENV_PYTHON}" -m loadgen.traffic_generator \
    --profile "${PROFILE_PATH}" \
    --experiment-id "${EXPERIMENT_ID}" \
    --target-url "${APP_TUNNEL_URL}" \
    --replay-schedule "${SCHEDULE_PATH}" \
    --arrival-process deterministic \
    --connection-mode fresh
GEN_EXIT=$?

if [[ "${GEN_EXIT}" -ne 0 ]]; then
    mark_invalid "Traffic generator exited with code ${GEN_EXIT}"; exit 1
fi

echo ""
echo "=== Waiting for collectors to finalize ==="
wait "${METRICS_COLLECTOR_PID}"
METRICS_EXIT=$?
wait "${HPA_COLLECTOR_PID}"
HPA_EXIT=$?
METRICS_COLLECTOR_PID=""  # already reaped, prevent cleanup() from re-killing
HPA_COLLECTOR_PID=""

if [[ "${METRICS_EXIT}" -ne 0 ]]; then
    mark_invalid "metrics_collector exited with code ${METRICS_EXIT}"; exit 1
fi
if [[ "${HPA_EXIT}" -ne 0 ]]; then
    mark_invalid "hpa_status_collector exited with code ${HPA_EXIT}"; exit 1
fi

echo ""
echo "=== Verifying artifacts ==="
for f in \
    "persistent-data/traffic/${EXPERIMENT_ID}/requests.csv" \
    "persistent-data/traffic/${EXPERIMENT_ID}/traffic_schedule.csv" \
    "persistent-data/traffic/${EXPERIMENT_ID}/metadata.json" \
    "persistent-data/metrics/${EXPERIMENT_ID}/metrics.csv" \
    "persistent-data/metrics/${EXPERIMENT_ID}/metadata.json" \
    "persistent-data/hpa/${EXPERIMENT_ID}/hpa_status.csv" \
    "persistent-data/hpa/${EXPERIMENT_ID}/hpa_events.jsonl" ; do
    if [[ ! -f "${f}" ]]; then
        mark_invalid "Missing expected artifact: ${f}"; exit 1
    fi
    echo "  verified: ${f}"
done

RUN_STATUS="valid"
echo ""
echo "=== Trial ${EXPERIMENT_ID} completed successfully (target=${TARGET}%) ==="
