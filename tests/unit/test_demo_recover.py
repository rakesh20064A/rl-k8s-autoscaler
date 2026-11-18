"""tests/unit/test_demo_recover.py

V2 Phase 9F: offline test for scripts/demo_recover.py -- the terminal
equivalent of the demo UI's RECOVER button (task sec.6). Never touches a
real cluster/subprocess: `experiment_store.reconcile()` and
`scripts.v2_live_cluster_state`'s helpers are exercised against a
monkeypatched store path and faked cluster functions.
"""

from __future__ import annotations

import json
import os
import time

from demo_ui import experiment_store
from scripts import demo_recover


def test_run_recover_reports_reconciliation_and_skips_cleanup_when_runner_alive(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_store, "STORE_PATH", tmp_path / "runs.json")
    monkeypatch.setattr(demo_recover, "no_runner_process_alive", lambda: False)

    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps({"status": "completed", "stage": "COMPLETED", "summary": {"n_samples": 1}}))
    old = time.time() - (experiment_store.STALE_STATUS_FILE_AGE_S + 600)
    os.utime(status_path, (old, old))
    run_id = experiment_store.start_run("ppo_v2", "demo_cpu_short")
    experiment_store.update_run(run_id, pid=2**31 - 1, status_path=str(status_path))

    report = demo_recover.run_recover()

    assert report["newly_terminal"] == [run_id]
    assert report["newly_stale"] == []
    assert report["runner_process_alive"] is True
    assert report["cluster_cleanup"]["ran"] is False


def test_run_recover_runs_cleanup_when_no_runner_alive(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_store, "STORE_PATH", tmp_path / "runs.json")
    monkeypatch.setattr(demo_recover, "no_runner_process_alive", lambda: True)
    monkeypatch.setattr(demo_recover, "ensure_canonical_idle_state", lambda: {"ran": True, "restored": True})

    report = demo_recover.run_recover()

    assert report["runner_process_alive"] is False
    assert report["cluster_cleanup"] == {"ran": True, "restored": True}


def test_main_prints_json_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(experiment_store, "STORE_PATH", tmp_path / "runs.json")
    monkeypatch.setattr(demo_recover, "no_runner_process_alive", lambda: True)
    monkeypatch.setattr(demo_recover, "ensure_canonical_idle_state", lambda: {"ran": False, "reason": "cluster already in the canonical idle state"})

    rc = demo_recover.main([])

    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["runner_process_alive"] is False
