#!/usr/bin/env bash
# scripts/v2_cluster_prepare.sh
# V2 Phase 4D-C: laptop-safe, idempotent cluster preparation for the V2
# multi-metric validation phases. Makes the ALREADY-RUNNING Minikube
# cluster lighter (Grafana scaled to 0 -- RL never reads it) and verifies
# Metrics Server's --metric-resolution, then REPORTS (never changes) the
# V2 canonical idle state (HPAs present, demo-app replica/ready count, the
# project's own controller lock file).
#
# This script NEVER:
#   - applies or deletes any HPA object
#   - scales demo-app (the RL/HPA-under-test deployment) itself
#   - builds or rebuilds any container image
#   - starts traffic, training, or evaluation
#   - touches Docker Desktop / Minikube's own CPU/memory/disk settings
#   - runs `minikube stop/delete` or any `helm install/upgrade/uninstall`
#
# Idempotent: re-running it with the cluster already in the target state
# is a no-op (Grafana already 0, Metrics Server already at the target
# resolution). Safe to run after any Minikube/Docker Desktop restart.
#
# Usage:
#   ./scripts/v2_cluster_prepare.sh                  # prepare (scale Grafana to 0, set 15s resolution, report state)
#   ./scripts/v2_cluster_prepare.sh --restore-grafana # scale Grafana back to 1 replica
#   ./scripts/v2_cluster_prepare.sh --metric-resolution 30s   # override the default 15s target

set -euo pipefail

MONITORING_NAMESPACE="monitoring"
APP_NAMESPACE="autoscaler-app"
METRIC_RESOLUTION="15s"
RESTORE_GRAFANA=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --restore-grafana)
            RESTORE_GRAFANA=1
            shift
            ;;
        --metric-resolution)
            METRIC_RESOLUTION="${2:?--metric-resolution requires a value, e.g. 15s}"
            shift 2
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== v2_cluster_prepare.sh ==="
echo ""

# -- (a) verify API reachable + node Ready, fail fast --------------------
echo "--- Step 1: cluster reachability ---"
if ! kubectl get nodes >/dev/null 2>&1; then
    echo "ERROR: kubectl cannot reach the API server. Is Minikube running?" >&2
    exit 1
fi
NODE_STATUS="$(kubectl get nodes -o jsonpath='{.items[0].status.conditions[?(@.type=="Ready")].status}')"
if [[ "${NODE_STATUS}" != "True" ]]; then
    echo "ERROR: node is not Ready (status=${NODE_STATUS})." >&2
    exit 1
fi
echo "  OK: API reachable, node Ready."
echo ""

# -- (b) scale Grafana to 0 (or restore to 1) -----------------------------
echo "--- Step 2: Grafana replica count ---"
GRAFANA_DEPLOYMENT="$(kubectl get deployment -n "${MONITORING_NAMESPACE}" \
    -l app.kubernetes.io/name=grafana -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
if [[ -z "${GRAFANA_DEPLOYMENT}" ]]; then
    echo "  WARNING: no deployment with label app.kubernetes.io/name=grafana found in ${MONITORING_NAMESPACE} -- skipping Grafana step."
else
    CURRENT_GRAFANA_REPLICAS="$(kubectl get deployment "${GRAFANA_DEPLOYMENT}" -n "${MONITORING_NAMESPACE}" \
        -o jsonpath='{.spec.replicas}')"
    echo "  Grafana deployment: ${GRAFANA_DEPLOYMENT} (current replicas: ${CURRENT_GRAFANA_REPLICAS})"
    if [[ "${RESTORE_GRAFANA}" -eq 1 ]]; then
        if [[ "${CURRENT_GRAFANA_REPLICAS}" == "1" ]]; then
            echo "  Already at 1 replica -- no change needed."
        else
            kubectl scale deployment "${GRAFANA_DEPLOYMENT}" -n "${MONITORING_NAMESPACE}" --replicas=1
            echo "  Restored ${GRAFANA_DEPLOYMENT} to 1 replica (was ${CURRENT_GRAFANA_REPLICAS})."
        fi
    else
        if [[ "${CURRENT_GRAFANA_REPLICAS}" == "0" ]]; then
            echo "  Already at 0 replicas -- no change needed."
        else
            kubectl scale deployment "${GRAFANA_DEPLOYMENT}" -n "${MONITORING_NAMESPACE}" --replicas=0
            echo "  Scaled ${GRAFANA_DEPLOYMENT} to 0 (was ${CURRENT_GRAFANA_REPLICAS})."
        fi
    fi
fi
echo ""

if [[ "${RESTORE_GRAFANA}" -eq 1 ]]; then
    echo "=== --restore-grafana requested: nothing else to do ==="
    exit 0
fi

# -- (c) ensure metrics-server --metric-resolution=${METRIC_RESOLUTION} --
echo "--- Step 3: Metrics Server resolution ---"
if [[ -x "${SCRIPT_DIR}/configure_metrics_server.sh" ]]; then
    echo "  Reusing existing scripts/configure_metrics_server.sh (idempotent, bounded 90s rollout wait)."
    "${SCRIPT_DIR}/configure_metrics_server.sh" "${METRIC_RESOLUTION}"
else
    echo "ERROR: scripts/configure_metrics_server.sh not found or not executable." >&2
    exit 1
fi
echo ""

# -- (d) report (never change) V2 canonical state -------------------------
echo "--- Step 4: V2 canonical state (report only, no changes made here) ---"
echo "  HPAs in ${APP_NAMESPACE}:"
kubectl get hpa -n "${APP_NAMESPACE}" -o wide 2>&1 || true
echo ""
echo "  Deployment demo-app replicas:"
kubectl get deployment demo-app -n "${APP_NAMESPACE}" \
    -o jsonpath='    desired={.spec.replicas} ready={.status.readyReplicas}{"\n"}' 2>&1 || true
echo ""
echo "  Controller lock file:"
LOCK_PATH="${SCRIPT_DIR}/../persistent-data/controller/active_controller.json"
if [[ -f "${LOCK_PATH}" ]]; then
    cat "${LOCK_PATH}"
else
    echo "  (lock file does not exist -- treated as 'none' by controller_exclusivity.py)"
fi
echo ""

echo "=== v2_cluster_prepare.sh complete ==="
