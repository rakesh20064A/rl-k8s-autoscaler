"""tests/unit/test_observation_v2_aggregation.py

V2 Phase 2 PART 2/9: focused coverage for CPU/memory per-replica
aggregation and normalization that tests/unit/test_rl_env_observation.py
does not already cover directly (that file proves CPU/request-rate/latency/
replica normalization and missing-data holding; this file adds
`per_replica_ratios()`'s explicit sum-then-divide aggregation behavior
across MULTIPLE ready replicas, and memory_per_replica normalization/
missing-metric behavior specifically -- memory_bytes is the field the V2
Phase 2 audit most needed verified: it comes from
`container_memory_working_set_bytes` (container-level, not node-level --
see live_backend.py's `_query_dict()`), summed across all pods matching the
Deployment's pod-name regex, then divided by ready_replicas here.
"""

from __future__ import annotations

import pytest

from autoscaler.src.rl_env.observation import (
    CPU_REQUEST_CORES,
    MEMORY_REQUEST_BYTES,
    HistoryState,
    ObservationConfig,
    build_observation,
    per_replica_ratios,
)
from autoscaler.src.rl_env.types import RawMetricSnapshot, ReplicaState


def _metrics(cpu_cores=None, memory_bytes=None, in_progress=0.0, request_rate=10.0, latency_ms=50.0, error_rate=0.0, target_up=True):
    return RawMetricSnapshot(
        request_rate_rps=request_rate, error_rate_rps=error_rate, latency_p95_ms=latency_ms,
        requests_in_progress=in_progress, cpu_cores=cpu_cores, memory_bytes=memory_bytes,
        target_up=target_up, query_error=False,
    )


class TestCpuAggregationAcrossReplicas:
    """cpu_cores from Prometheus is already a CLUSTER-WIDE sum across every
    matching pod (live_backend.py's `_query_dict()`: `sum(rate(...))`);
    per_replica_ratios() must divide that sum by ready_replicas to get a
    genuine per-replica average, not merely use the raw sum."""

    def test_single_replica_full_value_attributed(self):
        cpu, _, _ = per_replica_ratios(_metrics(cpu_cores=0.3), ready_replicas=1)
        assert cpu == pytest.approx(0.3 / CPU_REQUEST_CORES)

    def test_four_replicas_averages_the_summed_value(self):
        # 1.2 cores summed across 4 ready replicas -> 0.3 cores/replica average.
        cpu, _, _ = per_replica_ratios(_metrics(cpu_cores=1.2), ready_replicas=4)
        assert cpu == pytest.approx(0.3 / CPU_REQUEST_CORES)

    def test_doubling_replicas_at_same_total_halves_per_replica_value(self):
        cpu_2, _, _ = per_replica_ratios(_metrics(cpu_cores=1.0), ready_replicas=2)
        cpu_4, _, _ = per_replica_ratios(_metrics(cpu_cores=1.0), ready_replicas=4)
        assert cpu_4 == pytest.approx(cpu_2 / 2)

    def test_zero_ready_replicas_never_divides_by_zero(self):
        cpu, mem, inprog = per_replica_ratios(_metrics(cpu_cores=1.0, memory_bytes=1000), ready_replicas=0)
        assert cpu is None
        assert mem is None
        assert inprog is None


class TestMemoryAggregationAcrossReplicas:
    """Mirrors the CPU aggregation tests exactly -- memory_bytes is also a
    cluster-wide sum (container_memory_working_set_bytes, summed across all
    matching pods, see live_backend.py), so the same sum-then-divide
    semantics must apply."""

    def test_single_replica_full_value_attributed(self):
        _, mem, _ = per_replica_ratios(_metrics(memory_bytes=33_554_432), ready_replicas=1)  # 32MiB
        assert mem == pytest.approx(33_554_432 / MEMORY_REQUEST_BYTES)

    def test_four_replicas_averages_the_summed_value(self):
        # 128MiB summed across 4 replicas -> 32MiB/replica average.
        _, mem, _ = per_replica_ratios(_metrics(memory_bytes=134_217_728), ready_replicas=4)
        assert mem == pytest.approx(33_554_432 / MEMORY_REQUEST_BYTES)

    def test_memory_request_constant_is_64_mebibytes(self):
        assert MEMORY_REQUEST_BYTES == 64 * 1024 * 1024

    def test_memory_bytes_none_yields_none_ratio(self):
        _, mem, _ = per_replica_ratios(_metrics(memory_bytes=None, cpu_cores=0.1), ready_replicas=2)
        assert mem is None


class TestMemoryNormalizationInBuildObservation:
    def test_memory_at_request_normalizes_to_point_two_five_of_raw_max(self):
        # raw_max=4.0 (configs/rl_v2.yaml index 2); memory == exactly the
        # 64MiB request per replica -> raw ratio 1.0 -> normalized 1.0/4.0.
        config = ObservationConfig()
        result = build_observation(
            metrics=_metrics(cpu_cores=0.1, memory_bytes=MEMORY_REQUEST_BYTES),
            replicas=ReplicaState(desired=1, current=1, ready=1, available=1),
            seconds_since_last_effective_action=10.0,
            history=HistoryState.new(),
            previous_normalized=None,
            config=config,
        )
        assert result.normalized[2] == pytest.approx(0.25)

    def test_memory_above_raw_max_clips_to_one(self):
        config = ObservationConfig()
        result = build_observation(
            metrics=_metrics(cpu_cores=0.1, memory_bytes=MEMORY_REQUEST_BYTES * 100),
            replicas=ReplicaState(desired=1, current=1, ready=1, available=1),
            seconds_since_last_effective_action=10.0,
            history=HistoryState.new(),
            previous_normalized=None,
            config=config,
        )
        assert result.normalized[2] == pytest.approx(1.0)


class TestMissingMemoryMetricHoldsPreviousValue:
    def test_missing_memory_bytes_holds_previous_normalized_index_2(self):
        history = HistoryState.new()
        config = ObservationConfig()
        replicas = ReplicaState(desired=1, current=1, ready=1, available=1)

        first = build_observation(
            metrics=_metrics(cpu_cores=0.1, memory_bytes=MEMORY_REQUEST_BYTES),
            replicas=replicas, seconds_since_last_effective_action=10.0,
            history=history, previous_normalized=None, config=config,
        )
        assert first.normalized[2] == pytest.approx(0.25)

        # Second step: memory_bytes missing entirely -> step is invalid
        # (determine_validity treats ANY of cpu/memory/in_progress being
        # None as OVERRIDE_STALE_OR_MISSING_METRIC) -- every held level
        # feature, including memory_per_replica, must retain its prior value.
        second = build_observation(
            metrics=_metrics(cpu_cores=0.1, memory_bytes=None),
            replicas=replicas, seconds_since_last_effective_action=20.0,
            history=history, previous_normalized=first.normalized, config=config,
        )
        assert second.validity.sample_valid is False
        assert second.normalized[2] == pytest.approx(0.25)  # held, not zeroed or fabricated
