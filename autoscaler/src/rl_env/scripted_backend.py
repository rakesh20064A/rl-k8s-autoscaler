"""autoscaler/src/rl_env/scripted_backend.py

TEST_ONLY_SYNTHETIC_METRICS -- a deterministic DynamicsBackend implementation
for validating environment MECHANICS (Phase 15B.1). It does NOT model how
workload/replicas actually produce CPU, latency, memory, or error-rate
readings -- it simply plays back an explicitly supplied timeline of raw
metric snapshots keyed by exact decision-time (seconds). This is a
deliberate design choice: there is not yet sufficient empirical evidence to
freeze a general CPU/latency dynamics function suitable for PPO training
(see docs/experimental_design_corrections.md CORR-29). This backend must
NEVER be used to train a PPO policy -- only to exercise/validate the
Gymnasium environment's mechanics (observation construction, EWMA, action
gating, reward decomposition, invalid-telemetry handling, episode
termination).

Replica-transition mechanics ARE implemented here (not fabricated) because
they are already empirically grounded (CORR-25): on scale-up, `desired`/
`current` update immediately while `ready`/`available` catch up only after
a configurable readiness delay (default 10s, matching the ~5-15s lag
observed across all 9 official HPA tuning trials -- NOT claimed as a final
calibrated distribution, only a deterministic mechanics-test default); on
scale-down, all four fields update immediately (CORR-25 sec.4).
"""

from __future__ import annotations

from typing import Dict, Optional

from .types import BackendSnapshot, RawMetricSnapshot, ReplicaState

TEST_ONLY_SYNTHETIC_METRICS = True


class ScriptedBackend:
    """Implements backend.DynamicsBackend. See module docstring."""

    def __init__(
        self,
        timeline: Dict[float, RawMetricSnapshot],
        initial_replicas: int = 1,
        min_replicas: int = 1,
        max_replicas: int = 8,
        scale_up_ready_delay_seconds: float = 10.0,
    ) -> None:
        if not timeline:
            raise ValueError("ScriptedBackend requires a non-empty explicit metric timeline")
        self._timeline = dict(timeline)
        self.initial_replicas = initial_replicas
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.scale_up_ready_delay_seconds = scale_up_ready_delay_seconds

        self._time: float = 0.0
        self._desired = initial_replicas
        self._current = initial_replicas
        self._ready = initial_replicas
        self._available = initial_replicas
        self._pending_ready_transition: Optional[tuple] = None  # (scheduled_time, target_ready)

    def _missing_snapshot_metrics(self) -> RawMetricSnapshot:
        """No scripted entry exists for this exact time -- treated as a
        target-down/query-error tick, never as a fabricated reading."""
        return RawMetricSnapshot(
            request_rate_rps=None,
            error_rate_rps=None,
            latency_p95_ms=None,
            requests_in_progress=None,
            cpu_cores=None,
            memory_bytes=None,
            target_up=False,
            query_error=True,
        )

    def _snapshot_at(self, t: float) -> BackendSnapshot:
        metrics = self._timeline.get(t, self._missing_snapshot_metrics())
        return BackendSnapshot(
            time_s=t,
            metrics=metrics,
            replicas=ReplicaState(
                desired=self._desired, current=self._current,
                ready=self._ready, available=self._available,
            ),
        )

    def reset(self) -> BackendSnapshot:
        self._time = 0.0
        self._desired = self._current = self._ready = self._available = self.initial_replicas
        self._pending_ready_transition = None
        return self._snapshot_at(0.0)

    def current_snapshot(self) -> BackendSnapshot:
        return self._snapshot_at(self._time)

    def set_desired_replicas(self, desired: int) -> None:
        desired = max(self.min_replicas, min(self.max_replicas, desired))
        if desired == self._desired:
            return
        if desired > self._desired:
            # Scale-up: desired & current move immediately; ready/available
            # catch up only after the configured readiness delay.
            self._current = desired
            self._pending_ready_transition = (self._time + self.scale_up_ready_delay_seconds, desired)
            self._desired = desired
        else:
            # Scale-down: all four fields move immediately (CORR-25 sec.4).
            self._desired = desired
            self._current = desired
            self._ready = desired
            self._available = desired
            self._pending_ready_transition = None

    def advance(self, dt_seconds: float) -> BackendSnapshot:
        if dt_seconds < 0:
            raise ValueError("dt_seconds must be non-negative")
        target_time = self._time + dt_seconds
        if self._pending_ready_transition is not None:
            scheduled_time, target_ready = self._pending_ready_transition
            if scheduled_time <= target_time:
                self._ready = target_ready
                self._available = target_ready
                self._pending_ready_transition = None
        self._time = target_time
        return self._snapshot_at(self._time)
