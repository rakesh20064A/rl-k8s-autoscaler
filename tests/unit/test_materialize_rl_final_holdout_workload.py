"""tests/unit/test_materialize_rl_final_holdout_workload.py

Phase 16V: tests for scripts/materialize_rl_final_holdout_workload.py.
Pure file I/O -- no Kubernetes/Prometheus/traffic involvement whatsoever.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import yaml

import scripts.materialize_rl_final_holdout_workload as materialize_mod

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestMaterializedFile:
    def test_target_file_exists_and_matches_frozen_spec(self):
        with open(PROJECT_ROOT / "task_outputs" / "phase_16u_training_and_evaluation_workload_split.json") as f:
            spec = json.load(f)["final_evaluation_workload"]
        target = PROJECT_ROOT / spec["new_file_name_reserved"]
        assert target.exists(), "run scripts/materialize_rl_final_holdout_workload.py first"

        doc = yaml.safe_load(target.read_text())
        assert doc["seed"] == spec["seed"]
        assert doc["duration_seconds"] == spec["duration_seconds"]
        assert doc["intensity"] == spec["intensity"]
        assert doc["endpoint"] == spec["endpoint"]
        assert sum(p["duration_seconds"] for p in doc["phases"]) == spec["duration_seconds"]

    def test_seed_and_peak_are_novel(self):
        with open(PROJECT_ROOT / "task_outputs" / "phase_16u_training_and_evaluation_workload_split.json") as f:
            spec = json.load(f)["final_evaluation_workload"]
        assert spec["seed"] not in (42, 99)
        assert spec["shape_specification"]["peak_rps"] not in {70, 60, 50, 45, 20, 5, 80}

    def test_rendering_is_deterministic(self):
        spec = materialize_mod._load_spec()
        rendered1 = materialize_mod._render_yaml(spec)
        rendered2 = materialize_mod._render_yaml(spec)
        assert rendered1 == rendered2  # same spec -> byte-identical output every time

    def test_main_exits_zero_against_the_real_committed_spec(self):
        # Exercises the full main() path (including the relative_to() print
        # logic) against the real, already-materialized project file --
        # re-materializing it is a no-op (same frozen spec -> same bytes).
        assert materialize_mod.main() == 0

    def test_never_imports_or_calls_traffic_generator(self):
        source = Path(importlib.util.find_spec("scripts.materialize_rl_final_holdout_workload").origin).read_text()
        assert "traffic_generator" not in source
        assert "subprocess" not in source
        assert "kubectl" not in source
