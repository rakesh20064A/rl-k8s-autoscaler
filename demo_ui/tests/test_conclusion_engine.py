from __future__ import annotations

from demo_ui.comparison_engine import compare
from demo_ui.conclusion_engine import ConclusionThresholds, generate_conclusion
from demo_ui.models import ConclusionCategory, Controller, RunSummary, TrafficProfile


def _run(controller: Controller, replicas, timestamps, **overrides) -> RunSummary:
    base = dict(
        controller=controller, traffic_profile=TrafficProfile.RAMP, seed=42, duration_s=timestamps[-1],
        schedule_hash="same", timestamps=timestamps, replicas=replicas,
        cpu_pct=[50.0] * len(timestamps), memory_pct=[20.0] * len(timestamps), rps=[10.0] * len(timestamps),
        latency_avg_ms=[40.0] * len(timestamps), latency_p95_ms=[80.0] * len(timestamps), latency_p99_ms=[100.0] * len(timestamps),
        successful_requests=1000, errors=5, timeouts=1, sla_violations=2,
    )
    base.update(overrides)
    return RunSummary(**base)


class TestGenerateConclusion:
    def test_similar_when_no_dimension_is_material(self):
        hpa = _run(Controller.HPA, [2, 2, 2], [0, 10, 20], latency_p95_ms=[100.0] * 3, sla_violations=1)
        ppo = _run(Controller.PPO, [2, 2, 2], [0, 10, 20], latency_p95_ms=[101.0] * 3, sla_violations=1)
        result = compare(hpa, ppo)
        conclusion = generate_conclusion(result, TrafficProfile.RAMP)
        assert conclusion.category == ConclusionCategory.SIMILAR
        assert all(d.ppo_favored is None for d in conclusion.dimensions if d.name in ("Latency", "SLA"))

    def test_ppo_better_overall_when_all_material_dimensions_favor_ppo(self):
        hpa = _run(Controller.HPA, [3, 3, 3], [0, 10, 20], latency_p95_ms=[200.0] * 3, sla_violations=10)
        ppo = _run(Controller.PPO, [1, 1, 1], [0, 10, 20], latency_p95_ms=[100.0] * 3, sla_violations=1)
        result = compare(hpa, ppo)
        conclusion = generate_conclusion(result, TrafficProfile.RAMP)
        assert conclusion.category == ConclusionCategory.PPO_BETTER_OVERALL
        assert "PPO achieved" in conclusion.headline

    def test_hpa_better_overall_when_all_material_dimensions_favor_hpa(self):
        hpa = _run(Controller.HPA, [1, 1, 1], [0, 10, 20], latency_p95_ms=[100.0] * 3, sla_violations=1)
        ppo = _run(Controller.PPO, [3, 3, 3], [0, 10, 20], latency_p95_ms=[200.0] * 3, sla_violations=10)
        result = compare(hpa, ppo)
        conclusion = generate_conclusion(result, TrafficProfile.RAMP)
        assert conclusion.category == ConclusionCategory.HPA_BETTER_OVERALL

    def test_trade_off_when_dimensions_split(self):
        # PPO: much better latency, but much worse replica-seconds (more replicas held longer)
        hpa = _run(Controller.HPA, [1, 1, 1], [0, 10, 20], latency_p95_ms=[300.0] * 3, sla_violations=1)
        ppo = _run(Controller.PPO, [5, 5, 5], [0, 10, 20], latency_p95_ms=[100.0] * 3, sla_violations=1)
        result = compare(hpa, ppo)
        conclusion = generate_conclusion(result, TrafficProfile.RAMP)
        assert conclusion.category == ConclusionCategory.TRADE_OFF

    def test_conclusion_is_demo_data_flagged(self):
        hpa = _run(Controller.HPA, [1, 1], [0, 10])
        ppo = _run(Controller.PPO, [1, 1], [0, 10])
        result = compare(hpa, ppo)
        conclusion = generate_conclusion(result, TrafficProfile.RAMP, is_demo_data=True)
        assert conclusion.is_demo_data is True

    def test_dimensions_are_traceable_to_actual_values(self):
        hpa = _run(Controller.HPA, [1, 1, 1], [0, 10, 20], latency_p95_ms=[200.0] * 3)
        ppo = _run(Controller.PPO, [1, 1, 1], [0, 10, 20], latency_p95_ms=[100.0] * 3)
        result = compare(hpa, ppo)
        conclusion = generate_conclusion(result, TrafficProfile.RAMP)
        latency_dim = next(d for d in conclusion.dimensions if d.name == "Latency")
        assert "50.0%" in latency_dim.verdict
        assert latency_dim.ppo_favored is True

    def test_thresholds_are_configurable(self):
        hpa = _run(Controller.HPA, [1, 1, 1], [0, 10, 20], latency_p95_ms=[100.0] * 3)
        ppo = _run(Controller.PPO, [1, 1, 1], [0, 10, 20], latency_p95_ms=[94.0] * 3)  # 6% lower
        result = compare(hpa, ppo)
        loose = generate_conclusion(result, TrafficProfile.RAMP, thresholds=ConclusionThresholds(material_pct=10.0))
        tight = generate_conclusion(result, TrafficProfile.RAMP, thresholds=ConclusionThresholds(material_pct=2.0))
        loose_latency = next(d for d in loose.dimensions if d.name == "Latency")
        tight_latency = next(d for d in tight.dimensions if d.name == "Latency")
        assert loose_latency.ppo_favored is None  # below 10% threshold
        assert tight_latency.ppo_favored is True  # above 2% threshold
