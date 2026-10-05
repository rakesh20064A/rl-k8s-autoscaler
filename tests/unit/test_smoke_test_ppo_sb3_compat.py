"""tests/unit/test_smoke_test_ppo_sb3_compat.py

Phase 16V: wraps scripts/smoke_test_ppo_sb3_compat.py in the regular test
suite. Slower than the rest of the unit tests (actually constructs and
briefly "trains" a real SB3 PPO instance against ScriptedBackend), but
still fully offline -- no live Kubernetes/Prometheus/network access.
"""

from __future__ import annotations

import scripts.smoke_test_ppo_sb3_compat as smoke_mod


class TestSmokeTest:
    def test_smoke_test_passes_and_reports_zero_scientific_weight(self):
        report = smoke_mod.run_smoke_test()
        assert report["ok"] is True
        assert report["scientific_weight"] == "ZERO"
        assert report["live_kubernetes_touched"] is False
        assert report["traffic_generated"] is False
        assert report["counts_toward_frozen_training_budget"] is False

    def test_smoke_test_uses_scripted_backend_only(self):
        report = smoke_mod.run_smoke_test()
        assert "ScriptedBackend" in report["backend_used"]

    def test_banner_present_and_unambiguous(self):
        assert "ZERO SCIENTIFIC WEIGHT" in smoke_mod.ZERO_SCIENTIFIC_WEIGHT_BANNER
        assert "NOT PPO TRAINING EVIDENCE" in smoke_mod.ZERO_SCIENTIFIC_WEIGHT_BANNER
