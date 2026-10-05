#!/usr/bin/env bash
# scripts/backup_results.sh
# Copies all experiment results and persistent data to a timestamped backup archive.
# Non-destructive: never deletes source files.
# Run after any major experiment batch to protect against accidental loss.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TIMESTAMP=$(date -u +"%Y-%m-%dT%H%M%SZ")
BACKUP_DIR="${PROJECT_ROOT}/persistent-data/backups/${TIMESTAMP}"

echo "=== Backup: persistent-data -> ${BACKUP_DIR} ==="
echo ""

mkdir -p "${BACKUP_DIR}"

# Directories to back up (never including the backups directory itself)
BACKUP_SOURCES=(
    "${PROJECT_ROOT}/persistent-data/metrics"
    "${PROJECT_ROOT}/persistent-data/traffic"
    "${PROJECT_ROOT}/persistent-data/rl"
    "${PROJECT_ROOT}/persistent-data/hpa"
    "${PROJECT_ROOT}/persistent-data/logs"
    "${PROJECT_ROOT}/persistent-data/models"
    "${PROJECT_ROOT}/persistent-data/metadata"
    "${PROJECT_ROOT}/experiments/results"
    "${PROJECT_ROOT}/experiments/reports"
    "${PROJECT_ROOT}/datasets/processed"
    "${PROJECT_ROOT}/datasets/generated"
)

for src in "${BACKUP_SOURCES[@]}"; do
    if [[ -d "${src}" ]]; then
        rel="${src#${PROJECT_ROOT}/}"
        dest="${BACKUP_DIR}/${rel}"
        mkdir -p "$(dirname "${dest}")"
        cp -r "${src}" "${dest}"
        # Count files copied
        count=$(find "${dest}" -type f | wc -l | tr -d ' ')
        echo "  Backed up ${rel} (${count} files)"
    fi
done

# Write backup manifest
MANIFEST="${BACKUP_DIR}/backup_manifest.yaml"
cat > "${MANIFEST}" <<EOF
timestamp: "${TIMESTAMP}"
backup_dir: "${BACKUP_DIR}"
host: "$(hostname)"
git_commit: "$(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
EOF

echo ""
echo "Backup manifest: ${MANIFEST}"
echo ""

# Verify: check that at least some files were copied
TOTAL=$(find "${BACKUP_DIR}" -type f | wc -l | tr -d ' ')
if [[ "${TOTAL}" -eq 0 ]]; then
    echo "WARNING: No files were backed up. The persistent-data directories may be empty."
else
    echo "Backup complete: ${TOTAL} files in ${BACKUP_DIR}"
fi

echo ""
echo "IMPORTANT: This backup is NOT deleted by cleanup_runtime_only.sh."
