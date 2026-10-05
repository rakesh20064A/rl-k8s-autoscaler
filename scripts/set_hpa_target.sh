#!/usr/bin/env bash
# scripts/set_hpa_target.sh
#
# Phase 11C candidate-target patch mechanism. Changes ONLY
# spec.metrics[0].resource.target.averageUtilization on the live demo-app-hpa
# object -- never behavior, never minReplicas/maxReplicas, never any resource
# configuration. Verifies the live value after patching.
#
# By design this script REJECTS any value outside the three explicitly
# pre-committed tuning candidates unless the caller passes --allow-arbitrary,
# which exists only for exceptional, deliberate use and must never be used to
# silently substitute an unintended value during a tuning run.
#
# Phase 11C.4E: the candidate family was changed from 220/230/250 to
# 200/230/260 (see docs/experimental_design_corrections.md CORR-24 and
# task_outputs/phase_11c4e_hpa_candidate_and_schedule_redesign.md). The old
# family is SUPERSEDED_BEFORE_ANY_VALID_OFFICIAL_TUNING_RESULT -- Phase
# 11C.4D's controlled, repeated, final-path CPU map showed the 220-vs-230
# tolerance-adjusted boundary gap (11m) was smaller than observed CPU
# variability at every tested workload level. No valid official result is
# discarded by this change (the sole 220% attempt was already invalid for an
# unrelated traffic-delivery defect, CORR-21/22).
#
# This script does NOT rewrite configs/hpa.yaml or k8s/hpa/hpa.yaml -- those
# remain the canonical Phase-11B initial baseline (230%) until Phase 11D
# freezes the final winner. This is a live, temporary tuning patch only; the
# selected candidate value for a given run must be recorded in that run's own
# metadata by the run script that calls this, not inferred after the fact.
#
# Usage:
#   ./scripts/set_hpa_target.sh <200|230|260> [--namespace NS] [--hpa-name NAME] [--allow-arbitrary]
#
# NOTE: per Phase 11C.1 instructions, this script is prepared but has not been
# executed against the live cluster as part of Phase 11C.1 -- the canonical
# repo HPA remains at 230% (Phase 11B) until a future phase actually runs it.

set -euo pipefail

NAMESPACE="autoscaler-app"
HPA_NAME="demo-app-hpa"
ALLOW_ARBITRARY=0
TARGET=""

ALLOWED_VALUES=(200 230 260)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --namespace)
            NAMESPACE="$2"; shift 2 ;;
        --hpa-name)
            HPA_NAME="$2"; shift 2 ;;
        --allow-arbitrary)
            ALLOW_ARBITRARY=1; shift ;;
        [0-9]*)
            TARGET="$1"; shift ;;
        *)
            echo "ERROR: unrecognized argument '$1'" >&2
            exit 1 ;;
    esac
done

if [[ -z "${TARGET}" ]]; then
    echo "Usage: $0 <200|230|260> [--namespace NS] [--hpa-name NAME] [--allow-arbitrary]" >&2
    exit 1
fi

if ! [[ "${TARGET}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: target must be a positive integer percentage. Got: ${TARGET}" >&2
    exit 1
fi

is_allowed=0
for v in "${ALLOWED_VALUES[@]}"; do
    if [[ "${TARGET}" == "${v}" ]]; then
        is_allowed=1
        break
    fi
done

if [[ "${is_allowed}" -ne 1 && "${ALLOW_ARBITRARY}" -ne 1 ]]; then
    echo "ERROR: ${TARGET}% is not one of the pre-committed Phase-11C tuning" >&2
    echo "  candidates (${ALLOWED_VALUES[*]}). Refusing to apply an unplanned" >&2
    echo "  target -- this guards against accidentally tuning on an untested" >&2
    echo "  value. Pass --allow-arbitrary to override deliberately." >&2
    exit 1
fi

echo "=== set_hpa_target.sh ==="
echo "  HPA:       ${NAMESPACE}/${HPA_NAME}"
echo "  Target:    ${TARGET}%"
echo ""

if ! kubectl get hpa "${HPA_NAME}" -n "${NAMESPACE}" &>/dev/null; then
    echo "ERROR: hpa/${HPA_NAME} not found in namespace ${NAMESPACE}." >&2
    echo "  This script only patches an existing HPA; it does not create one." >&2
    exit 1
fi

CURRENT_TARGET="$(kubectl get hpa "${HPA_NAME}" -n "${NAMESPACE}" \
    -o jsonpath='{.spec.metrics[0].resource.target.averageUtilization}')"
echo "Current live target: ${CURRENT_TARGET}%"

if [[ "${CURRENT_TARGET}" == "${TARGET}" ]]; then
    echo "No change needed -- already at ${TARGET}%."
    exit 0
fi

kubectl patch hpa "${HPA_NAME}" -n "${NAMESPACE}" \
    --type=json \
    -p="[{\"op\":\"replace\",\"path\":\"/spec/metrics/0/resource/target/averageUtilization\",\"value\":${TARGET}}]"

NEW_TARGET="$(kubectl get hpa "${HPA_NAME}" -n "${NAMESPACE}" \
    -o jsonpath='{.spec.metrics[0].resource.target.averageUtilization}')"

if [[ "${NEW_TARGET}" != "${TARGET}" ]]; then
    echo "ERROR: patch applied but live target reads '${NEW_TARGET}%', expected '${TARGET}%'." >&2
    exit 1
fi

echo "Verified live target is now ${NEW_TARGET}%."

# Sanity-check that behavior/min/max were not touched by this patch.
kubectl get hpa "${HPA_NAME}" -n "${NAMESPACE}" -o jsonpath='  minReplicas={.spec.minReplicas} maxReplicas={.spec.maxReplicas} scaleUpStabilization={.spec.behavior.scaleUp.stabilizationWindowSeconds} scaleDownStabilization={.spec.behavior.scaleDown.stabilizationWindowSeconds}{"\n"}'

echo ""
echo "=== demo-app-hpa target set to ${TARGET}% (record this value in the run's own metadata) ==="
