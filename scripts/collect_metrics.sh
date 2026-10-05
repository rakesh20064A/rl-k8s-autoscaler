#!/usr/bin/env bash
# scripts/collect_metrics.sh
# Starts the persistent metrics collector in the foreground.
# Accepts optional arguments to override experiment metadata.
# Usage: ./scripts/collect_metrics.sh [--run-id RUN_ID] [--controller hpa|rl] [--profile PROFILE]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"

if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "ERROR: .venv not found. Run ./scripts/bootstrap.sh first."
    exit 1
fi

echo "Starting metrics collector..."
exec "${VENV_PYTHON}" -m collectors.metrics_collector \
    --config "${PROJECT_ROOT}/configs/experiment.yaml" \
    "$@"
