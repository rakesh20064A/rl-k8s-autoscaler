#!/usr/bin/env bash
# scripts/bootstrap.sh
# Checks all required local prerequisites, creates the Python 3.10 virtual environment,
# and installs dependencies.
# Run once before any other project work.
# Safe to re-run — idempotent.

set -euo pipefail

# Resolve project root relative to this script regardless of working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

echo "Project root: ${PROJECT_ROOT}"
echo ""
echo "=== Phase 0: Checking prerequisites ==="

check_tool() {
    local tool="$1"
    local install_hint="$2"
    if ! command -v "${tool}" &>/dev/null; then
        echo "MISSING: ${tool}"
        echo "  Install hint: ${install_hint}"
        return 1
    fi
    echo "  OK: ${tool} ($(${tool} --version 2>&1 | head -1))"
    return 0
}

MISSING=0

check_tool minikube  "brew install minikube"  || MISSING=1
check_tool kubectl   "brew install kubectl"   || MISSING=1
check_tool docker    "Install Docker Desktop from https://docs.docker.com/desktop/install/mac-install/" || MISSING=1
check_tool helm      "brew install helm"      || MISSING=1
check_tool python3.10 "pyenv install 3.10.14 && pyenv global 3.10.14" || MISSING=1
check_tool git       "brew install git"       || MISSING=1
check_tool curl      "brew install curl"      || MISSING=1

if [[ "${MISSING}" -eq 1 ]]; then
    echo ""
    echo "ERROR: One or more required tools are missing. Install them and re-run bootstrap.sh."
    exit 1
fi

echo ""
echo "=== Creating Python 3.10 virtual environment ==="

VENV_DIR="${PROJECT_ROOT}/.venv"

# Prefer the exact pyenv 3.10.14 interpreter when present (reproducible
# across machines); fall back to whatever `python3.10` resolves to on PATH
# otherwise. Either way, verify the resulting venv is actually a 3.10.x
# interpreter before proceeding -- never silently continue on a mismatched
# Python version.
PYENV_PY310="${HOME}/.pyenv/versions/3.10.14/bin/python3.10"
if [[ -x "${PYENV_PY310}" ]]; then
    PYTHON_FOR_VENV="${PYENV_PY310}"
else
    PYTHON_FOR_VENV="python3.10"
fi

if [[ -d "${VENV_DIR}" ]]; then
    echo "  .venv already exists, skipping creation."
else
    "${PYTHON_FOR_VENV}" -m venv "${VENV_DIR}"
    echo "  Created ${VENV_DIR} (using ${PYTHON_FOR_VENV})"
fi

VENV_PY_VERSION="$("${VENV_DIR}/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [[ "${VENV_PY_VERSION}" != "3.10" ]]; then
    echo "ERROR: ${VENV_DIR} is Python ${VENV_PY_VERSION}, expected 3.10.x." >&2
    exit 1
fi
echo "  Verified .venv is Python ${VENV_PY_VERSION}."

# Upgrade pip silently to avoid noise
"${VENV_DIR}/bin/pip" install --upgrade pip setuptools wheel --quiet

echo ""
echo "=== Installing base dependencies ==="
"${VENV_DIR}/bin/pip" install -r requirements/base.txt --quiet
echo "  Base dependencies installed."

echo ""
echo "=== Installing RL/training dependencies ==="
"${VENV_DIR}/bin/pip" install -r requirements/rl.txt --quiet
echo "  RL dependencies installed."

echo ""
echo "=== Installing development dependencies ==="
"${VENV_DIR}/bin/pip" install -r requirements/dev.txt --quiet
echo "  Dev dependencies installed."

echo ""
echo "=== Validating key imports ==="
"${VENV_DIR}/bin/python" -c "import yaml, kubernetes, requests, fastapi, prometheus_client, pandas, numpy" \
    && echo "  Core imports OK."

echo ""
echo "=== Ensuring persistent-data directories exist ==="
for subdir in metrics traffic rl hpa logs models metadata backups; do
    mkdir -p "${PROJECT_ROOT}/persistent-data/${subdir}"
done
echo "  persistent-data/ subdirectories ready."

echo ""
echo "=== Bootstrap complete ==="
echo ""
echo "To activate the virtual environment:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Next step: start Docker Desktop, then run:"
echo "  ./scripts/start_minikube.sh"
