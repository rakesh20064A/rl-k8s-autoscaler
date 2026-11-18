#!/usr/bin/env python3
"""scripts/demo_recover.py -- terminal equivalent of the demo UI's RECOVER
button (task sec.6).

Reconciles `demo_ui.experiment_store`'s run history against reality (a
run's own status file is trusted as ground truth over a dead/stale pid --
see `experiment_store.reconcile()`), then, ONLY if no
`scripts/run_demo_episode_v2.py` process is alive anywhere on this host,
checks whether the live cluster is already in its canonical idle state
(controller lock `none`, HPA list `[]` or `[demo-app-hpa-v2]`) and restores
it if not (`scripts/v2_live_cluster_state.ensure_canonical_idle_state()`).

Never mutates the cluster while a runner might still be mid-cleanup
itself. Safe to run at any time, including when nothing is wrong (it
simply reports there was nothing to do).

Usage:
    .venv/bin/python scripts/demo_recover.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from demo_ui import experiment_store  # noqa: E402
from scripts.v2_live_cluster_state import ensure_canonical_idle_state, no_runner_process_alive  # noqa: E402


def run_recover() -> Dict[str, Any]:
    """Pure (no printing): reconciles the run store, then runs the
    cluster-cleanup check -- the exact same two calls the demo UI's
    RECOVER button makes. Returns a plain dict report."""
    reconciled = experiment_store.reconcile()
    runner_alive = not no_runner_process_alive()
    cleanup = None if runner_alive else ensure_canonical_idle_state()
    return {
        "active_run": (reconciled["active"] or {}).get("run_id"),
        "newly_terminal": [r["run_id"] for r in reconciled["newly_terminal"]],
        "newly_stale": [r["run_id"] for r in reconciled["newly_stale"]],
        "runner_process_alive": runner_alive,
        "cluster_cleanup": cleanup if cleanup is not None else {"ran": False, "reason": "a run_demo_episode_v2.py process is still alive"},
    }


def main(argv=None) -> int:  # noqa: ARG001 -- no args today; kept for a consistent CLI shape with the other scripts
    report = run_recover()
    print(json.dumps(report, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
