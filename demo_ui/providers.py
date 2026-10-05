"""providers.py -- Data provider abstraction (sec.18).

Defines Protocol interfaces so UI components never couple directly to
Kubernetes/Prometheus/PPO runtime. This phase ships ONLY offline fixture
implementations. A future `PrometheusMetricsProvider` /
`KubernetesClusterProvider` can implement the same Protocols without any UI
component changing.

HARD BOUNDARY: no implementation in this file may perform real subprocess,
network, or Kubernetes API calls in this phase. Live variants are documented
as TODOs only.
"""

from __future__ import annotations

from typing import List, Optional, Protocol

from demo_ui.models import (
    Controller,
    ExperimentRecord,
    HpaSnapshot,
    LogEvent,
    PodStatus,
    PpoSnapshot,
    RunSummary,
    TrafficProfile,
)


class MetricsProvider(Protocol):
    """Time-series + log data for a single in-progress or completed run.

    Future live implementation (PrometheusMetricsProvider): queries the
    in-cluster Prometheus via the project's existing `PrometheusClient`
    (collectors/metrics_collector.py) through an already-established,
    externally-managed port-forward -- this class must NEVER start/stop
    that port-forward itself (mirrors LiveKubernetesBackend's own
    documented boundary)."""

    def get_run_summary(self, run_id: str) -> RunSummary: ...

    def get_logs(self, run_id: str) -> List[LogEvent]: ...


class ClusterProvider(Protocol):
    """Live cluster/workload state for display only.

    Future live implementation (KubernetesClusterProvider): READ-ONLY calls
    only (e.g. the project's `LiveKubernetesBackend.current_snapshot()`
    pattern) -- never issues `kubectl scale`, never deletes/creates the
    HPA. Mutating the cluster is explicitly out of scope for this UI
    layer; experiment orchestration (if ever wired up) belongs in
    `ExperimentController`, not here."""

    def get_pods(self) -> List[PodStatus]: ...

    def get_hpa_snapshot(self) -> Optional[HpaSnapshot]: ...

    def get_ppo_snapshot(self) -> Optional[PpoSnapshot]: ...


class ExperimentController(Protocol):
    """Starts/stops/resets an experiment run.

    Future live implementation: TODO -- must reuse the project's existing
    detached-supervisor pattern (tracked Prometheus port-forward, tracked
    caffeinate, HPA removed only after dependencies verified healthy, PPO
    launched via `start_new_session=True`, signal-safe cleanup). This phase
    implements ONLY an offline fixture simulator; no subprocess, no
    kubectl, no Docker call exists anywhere in this class in this phase."""

    def start(self, controller: Controller, traffic_profile: TrafficProfile) -> None: ...

    def stop(self) -> None: ...

    def reset(self) -> None: ...


class ResultStore(Protocol):
    """Experiment history persistence (sec.17).

    Future live implementation: TODO -- read/write the
    `experiments/<id>/{hpa,ppo}/...` layout. This phase lists only fixture
    records; it never creates or modifies real scientific experiment
    records."""

    def list_experiments(self) -> List[ExperimentRecord]: ...

    def get_run_summary(self, experiment_id: str, controller: Controller) -> Optional[RunSummary]: ...


class FixtureMetricsProvider:
    """Offline MetricsProvider backed entirely by demo_ui.fixtures. Safe to
    use at any time: performs no I/O beyond reading the in-memory fixture
    module."""

    def __init__(self, summaries: dict, logs: dict) -> None:
        self._summaries = summaries
        self._logs = logs

    def get_run_summary(self, run_id: str) -> RunSummary:
        return self._summaries[run_id]

    def get_logs(self, run_id: str) -> List[LogEvent]:
        return self._logs.get(run_id, [])


class FixtureClusterProvider:
    """Offline ClusterProvider backed entirely by demo_ui.fixtures."""

    def __init__(self, pods: List[PodStatus], hpa: Optional[HpaSnapshot], ppo: Optional[PpoSnapshot]) -> None:
        self._pods = pods
        self._hpa = hpa
        self._ppo = ppo

    def get_pods(self) -> List[PodStatus]:
        return list(self._pods)

    def get_hpa_snapshot(self) -> Optional[HpaSnapshot]:
        return self._hpa

    def get_ppo_snapshot(self) -> Optional[PpoSnapshot]:
        return self._ppo


class FixtureExperimentController:
    """Offline ExperimentController: mutates only in-memory state, never
    touches live infrastructure. `start`/`stop`/`reset` are no-ops beyond
    recording the call for the state machine driving the UI."""

    def __init__(self) -> None:
        self.calls: List[tuple] = []

    def start(self, controller: Controller, traffic_profile: TrafficProfile) -> None:
        self.calls.append(("start", controller, traffic_profile))

    def stop(self) -> None:
        self.calls.append(("stop",))

    def reset(self) -> None:
        self.calls.append(("reset",))


class FixtureResultStore:
    """Offline ResultStore backed entirely by demo_ui.fixtures."""

    def __init__(self, records: List[ExperimentRecord], summaries: dict) -> None:
        self._records = records
        self._summaries = summaries

    def list_experiments(self) -> List[ExperimentRecord]:
        return list(self._records)

    def get_run_summary(self, experiment_id: str, controller: Controller) -> Optional[RunSummary]:
        return self._summaries.get((experiment_id, controller))
