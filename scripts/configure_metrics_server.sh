#!/usr/bin/env bash
# scripts/configure_metrics_server.sh
# Sets the Minikube metrics-server addon's --metric-resolution to the project's
# selected HPA-vs-RL research value (default 15s), without touching any other
# argument, the image, resources, ports, probes, or TLS settings.
#
# Background (Phase 11A / 11A.1, see task_outputs/phase_11a_*.md):
# Minikube's built-in metrics-server addon ships with --metric-resolution=60s as
# its own packaging default (not an upstream metrics-server default, and not a
# choice made by this project). The upstream kubernetes-sigs/metrics-server Helm
# chart itself defaults to 15s. 15s was selected as the initial Phase-11B research
# configuration to reduce the information-freshness asymmetry between HPA (fed by
# Metrics Server) and RL (fed by Prometheus, 10s scrape interval).
#
# This script assumes the metrics-server addon Deployment already exists
# (kube-system/metrics-server) — it does not install or reinstall metrics-server.
# It is idempotent: re-running it with the same resolution is a no-op rollout.
#
# IMPORTANT: this is a live-cluster patch, not a persisted manifest. It does NOT
# survive `minikube delete` / cluster recreation — after recreating the cluster,
# re-run this script (or re-enable the addon and re-run it) to restore 15s.
#
# Usage:
#   ./scripts/configure_metrics_server.sh [resolution]
#   resolution defaults to 15s if omitted.

set -euo pipefail

NAMESPACE=kube-system
DEPLOYMENT=metrics-server
RESOLUTION="${1:-15s}"

if ! [[ "${RESOLUTION}" =~ ^[0-9]+s$ ]]; then
    echo "ERROR: resolution must look like '15s', '30s', '60s', etc. Got: ${RESOLUTION}"
    exit 1
fi

echo "=== configure_metrics_server.sh ==="
echo "  Target resolution: --metric-resolution=${RESOLUTION}"
echo ""

if ! kubectl get deployment "${DEPLOYMENT}" -n "${NAMESPACE}" &>/dev/null; then
    echo "ERROR: deployment/${DEPLOYMENT} not found in namespace ${NAMESPACE}."
    echo "  metrics-server addon must already be enabled (minikube addons enable metrics-server)."
    echo "  This script only edits an existing installation; it does not install one."
    exit 1
fi

CURRENT_ARGS_JSON="$(kubectl get deployment "${DEPLOYMENT}" -n "${NAMESPACE}" \
    -o jsonpath='{.spec.template.spec.containers[0].args}' | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)))')"

echo "Current args: ${CURRENT_ARGS_JSON}"

# Rebuild the args array, replacing only the --metric-resolution entry
# (or appending it if somehow absent), preserving every other argument verbatim.
NEW_ARGS_JSON="$(python3 - "${CURRENT_ARGS_JSON}" "${RESOLUTION}" <<'PYEOF'
import json
import sys

current = json.loads(sys.argv[1])
resolution = sys.argv[2]
target_flag = f"--metric-resolution={resolution}"

new_args = []
replaced = False
for arg in current:
    if arg.startswith("--metric-resolution="):
        new_args.append(target_flag)
        replaced = True
    else:
        new_args.append(arg)
if not replaced:
    new_args.append(target_flag)

print(json.dumps(new_args))
PYEOF
)"

echo "New args:     ${NEW_ARGS_JSON}"
echo ""

if [[ "${CURRENT_ARGS_JSON}" == "${NEW_ARGS_JSON}" ]]; then
    echo "No change needed — already at ${RESOLUTION}."
    exit 0
fi

kubectl patch deployment "${DEPLOYMENT}" -n "${NAMESPACE}" \
    --type=json \
    -p="[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/args\",\"value\":${NEW_ARGS_JSON}}]"

echo ""
echo "Waiting for rollout..."
kubectl rollout status deployment/"${DEPLOYMENT}" -n "${NAMESPACE}" --timeout=90s

echo ""
echo "Verifying live args..."
kubectl get deployment "${DEPLOYMENT}" -n "${NAMESPACE}" \
    -o jsonpath='{.spec.template.spec.containers[0].args}{"\n"}'

echo ""
echo "=== metrics-server now running at --metric-resolution=${RESOLUTION} ==="
