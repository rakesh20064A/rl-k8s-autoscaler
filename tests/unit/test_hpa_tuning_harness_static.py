"""Static consistency checks for the HPA tuning shell harness (Phase 11C.4E).

These are plain text/content assertions against the checked-in shell scripts
-- no subprocess execution, no live cluster access. Their purpose is to catch
a regression back to defects already fixed in prior phases:

- Phase 11C.4B/CORR-21: the load generator must never fall back to
  `kubectl port-forward svc/demo-app` for application traffic (pins traffic
  to a single pod, defeating multi-replica load balancing). It must use the
  corrected `demo-app-loadtest` NodePort tunnel and `--connection-mode fresh`.
- Phase 11C.4E/CORR-24: `scripts/set_hpa_target.sh` must enforce the current
  candidate family (200/230/260), not the superseded 220/230/250 family.
"""

from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HARNESS_SCRIPT = PROJECT_ROOT / "scripts" / "run_hpa_tuning_trial.sh"
SET_TARGET_SCRIPT = PROJECT_ROOT / "scripts" / "set_hpa_target.sh"


class TestHpaTuningHarnessStatic(unittest.TestCase):
    def setUp(self):
        self.harness_text = HARNESS_SCRIPT.read_text(encoding="utf-8")
        self.set_target_text = SET_TARGET_SCRIPT.read_text(encoding="utf-8")

    def test_harness_never_port_forwards_to_demo_app_service_for_traffic(self):
        self.assertNotIn(
            'port-forward -n "${APP_NS}" svc/demo-app ',
            self.harness_text,
            "Harness must not fall back to kubectl port-forward svc/demo-app "
            "for application traffic -- it pins traffic to one pod (CORR-21).",
        )

    def test_harness_uses_loadtest_nodeport_tunnel(self):
        self.assertIn("minikube service", self.harness_text)
        self.assertIn("demo-app-loadtest", self.harness_text)

    def test_harness_uses_fresh_connection_mode_for_traffic(self):
        self.assertIn("--connection-mode fresh", self.harness_text)

    def test_harness_still_allows_prometheus_port_forward(self):
        # The Prometheus observer path is explicitly allowed to remain a
        # port-forward (CORR-21 §9) -- this is not the defect being guarded.
        self.assertIn("kube-prom-stack-kube-prome-prometheus", self.harness_text)

    def test_harness_has_distribution_preflight(self):
        self.assertIn("LOADTEST_SELECTOR", self.harness_text)
        self.assertIn("sessionAffinity", self.harness_text)
        self.assertIn("ENDPOINT_COUNT", self.harness_text)

    def test_set_hpa_target_enforces_current_candidate_family(self):
        self.assertIn("ALLOWED_VALUES=(200 230 260)", self.set_target_text)
        self.assertNotIn("ALLOWED_VALUES=(220 230 250)", self.set_target_text)


if __name__ == "__main__":
    unittest.main()
