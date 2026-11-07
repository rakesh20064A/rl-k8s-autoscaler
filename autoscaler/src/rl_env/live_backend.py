"""autoscaler/src/rl_env/live_backend.py

Phase 16V: the live Kubernetes/Prometheus `DynamicsBackend` implementation
(see `backend.py`'s Protocol docstring, option 3 -- "a future live
Kubernetes/Prometheus backend"). Obtains every observation from the
project's ALREADY-EXISTING, already-tested telemetry/mutation mechanisms:

  * `collectors.metrics_collector.PrometheusClient` -- the same instant-query
    HTTP client every offline collection script already uses. This module
    does not open its own port-forward; it assumes one is already running,
    exactly like every other Prometheus-consuming script in this project
    (docs/experimental_design_corrections.md: "must NEVER automatically
    manipulate Kubernetes resources or start/stop port-forwards internally").
  * The same `kubectl` subprocess command SHAPES already used by
    `scripts/run_dynamics_calibration_slot.py`'s `scale_and_wait()`,
    `wait_for_idle_one_replica()`, `verify_no_hpa_targets_deployment()`,
    `delete_hpa()`, `restore_hpa()` -- reimplemented here as small,
    dependency-injectable functions (via `KubectlRunner`) rather than
    imported directly, because that script is a __main__ entry point, not a
    library module, and importing internal functions from it would couple
    this environment code to an unrelated script's argument-parsing/global
    state. The exact command shapes, namespaces, and constants are
    unchanged from that script -- see `phase_16v_implementation_audit.md`
    for the line-by-line correspondence.
  * The SAME None-vs-0.0 zero/null policy for request/error rate already
    used by `collectors.metrics_collector.MetricsCollector.collect_sample()`
    (lines ~512-534 there) -- ported here as a small, explicitly-cited
    function rather than re-derived, since `RawMetricSnapshot`'s contract
    requires exactly this distinction.

Unlike `ScriptedBackend` (idealized, logical time), this backend's
`advance()` performs a REAL `time.sleep()` for the decision interval before
reading fresh state -- there is no "scheduled transition" concept here: the
real Kubernetes cluster's own convergence speed is whatever it is, observed
honestly.

Failure handling (Phase16V task sec.5, "post-action emergency handling"):
Any INTERNAL exception while querying Prometheus or reading back Deployment
status (`advance()`, `current_snapshot()`, `reset()`) is caught and
converted into a `target_up=False` `RawMetricSnapshot` -- this is not a
fabrication, it is exactly the frozen contract's own designated pathway for
"we could not obtain a trustworthy reading" (docs/rl_mdp_spec.md sec.3,
`OVERRIDE_PROMETHEUS_UNAVAILABLE`). It never raises out of `advance()`.
`set_desired_replicas()`, by contrast, DOES raise `KubectlApiError` on
failure -- an actuation failure is a materially different event
(`k8s_api_failure`, sec.13) that the calling environment must be able to
distinguish and record, not silently swallow (see the corresponding fix in
`environment.py`, Phase16V).
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from collectors.metrics_collector import PrometheusClient

from .types import BackendSnapshot, RawMetricSnapshot, ReplicaState

DEFAULT_APP_NAMESPACE = "autoscaler-app"
DEFAULT_DEPLOYMENT = "demo-app"
DEFAULT_HPA_NAME = "demo-app-hpa"


class KubectlApiError(RuntimeError):
    """Raised when a `kubectl` mutation (scale) fails or times out. Mirrors
    docs/rl_mdp_spec.md sec.13's `k8s_api_failure` override reason -- the
    caller (environment.py) must catch this and record it, never let it
    propagate uncaught, and never blindly retry within the same step."""


class RestorationTimeoutError(RuntimeError):
    """Raised when a bounded wait for the Deployment to reach a target
    replica state does not converge in time (reset, or post-episode
    restoration per phase_16u_direct_ppo_protocol.yaml Section 4/
    phase_16u_safety_and_fallback_protocol.yaml). Callers must stop the
    training/evaluation loop for manual inspection on this error -- never
    proceed with an unconfirmed state, never silently retry indefinitely."""


CommandRunner = Callable[[List[str], float], subprocess.CompletedProcess]


def _default_command_runner(cmd: List[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)  # noqa: S603,S607


@dataclass
class KubectlRunner:
    """Thin, dependency-injectable wrapper around the exact `kubectl`
    command shapes already used by `scripts/run_dynamics_calibration_slot.py`
    (`scale_and_wait`, `wait_for_idle_one_replica`,
    `verify_no_hpa_targets_deployment`, `delete_hpa`, `restore_hpa`). Kept
    separate from that script (a __main__ entry point) so it can be
    constructed with a fake `run` callable in tests without any subprocess
    or Kubernetes dependency."""

    app_namespace: str = DEFAULT_APP_NAMESPACE
    deployment: str = DEFAULT_DEPLOYMENT
    hpa_name: str = DEFAULT_HPA_NAME
    hpa_manifest_path: str = "k8s/hpa/hpa.yaml"
    run: CommandRunner = field(default=_default_command_runner)
    timeout_s: float = 30.0

    def _jsonpath(self, args: List[str]) -> str:
        proc = self.run(["kubectl"] + args, self.timeout_s)
        return (proc.stdout or "").strip()

    def get_replica_state(self) -> ReplicaState:
        """Mirrors scale_and_wait()/wait_for_idle_one_replica()'s four
        separate jsonpath reads of the Deployment's own .spec/.status --
        NOT kube-state-metrics (CORR-25's evidence hierarchy, types.py)."""
        base = ["get", "deployment", self.deployment, "-n", self.app_namespace, "-o"]
        desired = self._jsonpath(base + ["jsonpath={.spec.replicas}"])
        current = self._jsonpath(base + ["jsonpath={.status.replicas}"])
        ready = self._jsonpath(base + ["jsonpath={.status.readyReplicas}"])
        available = self._jsonpath(base + ["jsonpath={.status.availableReplicas}"])

        def _as_int(s: str) -> int:
            try:
                return int(s)
            except (TypeError, ValueError):
                return 0

        return ReplicaState(
            desired=_as_int(desired), current=_as_int(current),
            ready=_as_int(ready), available=_as_int(available),
        )

    def scale(self, replicas: int) -> None:
        """Fire-and-forget `kubectl scale` -- NOT `scale_and_wait()`'s
        blocking `rollout status` wait. A live backend must observe
        whatever real convergence state exists at the next decision point,
        not force synchronous convergence mid-step (that would fabricate a
        faster system than actually exists)."""
        proc = self.run(
            ["kubectl", "scale", "deployment", self.deployment, "-n", self.app_namespace, f"--replicas={replicas}"],
            self.timeout_s,
        )
        if proc.returncode != 0:
            raise KubectlApiError(f"kubectl scale failed (rc={proc.returncode}): {proc.stderr!r}")

    def wait_for_replica_state(
        self, target_replicas: int, timeout_seconds: float, poll_interval_seconds: float = 2.0,
        sleep_fn: Callable[[float], None] = time.sleep, clock_fn: Callable[[], float] = time.monotonic,
    ) -> ReplicaState:
        """Bounded poll until desired==current==ready==available==target,
        mirroring wait_for_idle_one_replica()/scale_and_wait()'s polling
        loop shape. Raises RestorationTimeoutError on timeout -- never
        returns a state silently claiming success it did not observe."""
        deadline = clock_fn() + timeout_seconds
        state = self.get_replica_state()
        while clock_fn() < deadline:
            state = self.get_replica_state()
            if (state.desired, state.current, state.ready, state.available) == (target_replicas,) * 4:
                return state
            sleep_fn(poll_interval_seconds)
        raise RestorationTimeoutError(
            f"Deployment {self.deployment!r} did not reach replicas={target_replicas} "
            f"within {timeout_seconds}s (last observed: {state})"
        )

    def hpa_exists(self) -> bool:
        """Mirrors verify_no_hpa_targets_deployment() -- True if any HPA
        object exists in the namespace (this project runs exactly one
        Deployment/HPA pair per namespace, so namespace-scoped absence is
        equivalent to 'no HPA targets this Deployment')."""
        out = self._jsonpath(["get", "hpa", "-n", self.app_namespace, "--no-headers"])
        return out != ""

    def list_hpa_names(self) -> List[str]:
        """V2 Phase 2: unlike `hpa_exists()` (a single present/absent bit),
        this names WHICH HPA object(s) exist -- needed because V2 introduces
        a second, independently-nameable HPA variant
        (k8s/hpa/hpa_v2_cpu_memory.yaml's `demo-app-hpa-v2`, alongside the
        frozen `demo-app-hpa`) that Kubernetes would otherwise allow to
        coexist with the first, independently issuing scale decisions
        against the same Deployment -- a genuine double-controller hazard
        (see controller_exclusivity.py). Returns an empty list if no HPA
        object exists in the namespace."""
        out = self._jsonpath(["get", "hpa", "-n", self.app_namespace, "-o", "jsonpath={.items[*].metadata.name}"])
        return out.split() if out else []

    def delete_hpa(self) -> None:
        self.run(["kubectl", "delete", "hpa", self.hpa_name, "-n", self.app_namespace, "--ignore-not-found"], self.timeout_s)

    def restore_hpa(self) -> None:
        self.run(["kubectl", "apply", "-f", self.hpa_manifest_path], self.timeout_s)


def _query_dict(app_namespace: str, deployment: str, route: str) -> Dict[str, str]:
    """The exact six PromQL strings this backend needs, taken verbatim from
    collectors/metrics_collector.py's `self.queries` (lines ~368-380) --
    only the replica-related queries are omitted here since ReplicaState is
    sourced from the Deployment object directly (CORR-25), not
    kube-state-metrics."""
    return {
        "request_rate_rps": f'sum(rate(demo_app_requests_total{{namespace="{app_namespace}",route="{route}"}}[1m]))',
        "error_rate_rps": f'sum(rate(demo_app_requests_total{{namespace="{app_namespace}",route="{route}",status_code=~"5.."}}[1m]))',
        "latency_p95_seconds": f'histogram_quantile(0.95, sum(rate(demo_app_request_duration_seconds_bucket{{namespace="{app_namespace}",route="{route}"}}[1m])) by (le))',
        "requests_in_progress": f'sum(demo_app_requests_in_progress{{namespace="{app_namespace}",route="{route}"}})',
        "cpu_cores": f'sum(rate(container_cpu_usage_seconds_total{{namespace="{app_namespace}",pod=~"{deployment}-.*"}}[1m]))',
        "memory_bytes": f'sum(container_memory_working_set_bytes{{namespace="{app_namespace}",pod=~"{deployment}-.*"}})',
    }


def _snapshot_metrics(prom: PrometheusClient, queries: Dict[str, str], app_namespace: str, deployment: str) -> RawMetricSnapshot:
    """Query Prometheus for one decision step and apply the SAME null/zero
    policy as collect_sample() (collectors/metrics_collector.py lines
    ~512-534), ported verbatim rather than re-derived:
      - target down -> every rate field is None (never inferred as 0).
      - request_rate_rps itself missing/errored -> error_rate_rps also None
        (an error rate is meaningless without a request-rate denominator).
      - request_rate_rps present -> error_rate_rps absent-without-error
        becomes a genuine 0.0 (real "no 5xx traffic", not a fabrication).
      - ANY query error (including on cpu/memory/in-progress/latency)
        forces `query_error=True`, matching determine_validity()'s
        unconditional invalidity on query_error (observation.py).
    Latency is converted seconds -> milliseconds here (RawMetricSnapshot's
    contract; the PromQL/metrics_collector.py convention keeps latency in
    seconds).
    """
    target_up, _ = prom.check_target_health(namespace=app_namespace, app_name=deployment)
    if not target_up:
        return RawMetricSnapshot(
            request_rate_rps=None, error_rate_rps=None, latency_p95_ms=None,
            requests_in_progress=None, cpu_cores=None, memory_bytes=None,
            target_up=False, query_error=False,
        )

    values: Dict[str, Optional[float]] = {}
    any_query_error = False
    for name, promql in queries.items():
        val, err = prom.query(promql)
        if err:
            any_query_error = True
        values[name] = val

    total_rate = values.get("request_rate_rps")
    if total_rate is None:
        request_rate = None
        error_rate = None
    else:
        request_rate = total_rate
        error_rate = values.get("error_rate_rps")
        if error_rate is None:
            error_rate = 0.0  # absent 5xx series with a real total -> genuine zero

    latency_s = values.get("latency_p95_seconds")
    latency_ms = None if latency_s is None else latency_s * 1000.0

    return RawMetricSnapshot(
        request_rate_rps=request_rate,
        error_rate_rps=error_rate,
        latency_p95_ms=latency_ms,
        requests_in_progress=values.get("requests_in_progress"),
        cpu_cores=values.get("cpu_cores"),
        memory_bytes=values.get("memory_bytes"),
        target_up=True,
        query_error=any_query_error,
    )


class LiveKubernetesBackend:
    """Implements `backend.DynamicsBackend` against the real Minikube
    cluster. See module docstring for the exact reused mechanisms and the
    exception-handling contract."""

    def __init__(
        self,
        prometheus_url: str = "http://localhost:9090",
        app_namespace: str = DEFAULT_APP_NAMESPACE,
        deployment: str = DEFAULT_DEPLOYMENT,
        route: str = "/work",
        min_replicas: int = 1,
        max_replicas: int = 8,
        initial_replicas: int = 1,
        pre_episode_stabilization_s: float = 60.0,
        restoration_max_wait_s: float = 120.0,
        kubectl: Optional[KubectlRunner] = None,
        prom_client: Optional[PrometheusClient] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        hpa_name: Optional[str] = None,
        hpa_manifest_path: Optional[str] = None,
    ) -> None:
        """V2 Phase 4C-1 (P1-2): `hpa_name`/`hpa_manifest_path` are additive,
        optional overrides of `KubectlRunner`'s own V1 defaults
        (`DEFAULT_HPA_NAME` / `"k8s/hpa/hpa.yaml"`) -- ignored (and
        irrelevant) whenever `kubectl` is explicitly injected, since the
        caller's own `KubectlRunner` already decides its HPA identity. Every
        existing caller that omits both keeps BYTE-IDENTICAL V1 behaviour:
        no `KubectlRunner` keyword changes unless a value was actually
        given."""
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.initial_replicas = initial_replicas
        self.pre_episode_stabilization_s = pre_episode_stabilization_s
        self.restoration_max_wait_s = restoration_max_wait_s
        if kubectl is not None:
            self._kubectl = kubectl
        else:
            kubectl_kwargs: Dict[str, str] = {"app_namespace": app_namespace, "deployment": deployment}
            if hpa_name is not None:
                kubectl_kwargs["hpa_name"] = hpa_name
            if hpa_manifest_path is not None:
                kubectl_kwargs["hpa_manifest_path"] = hpa_manifest_path
            self._kubectl = KubectlRunner(**kubectl_kwargs)
        self._prom = prom_client or PrometheusClient(base_url=prometheus_url)
        self._queries = _query_dict(app_namespace, deployment, route)
        self._app_namespace = app_namespace
        self._deployment = deployment
        self._sleep = sleep_fn

        self._time_s = 0.0
        self._last_desired_issued: Optional[int] = None

    # -- internal helpers ---------------------------------------------------

    def _safe_metrics(self) -> RawMetricSnapshot:
        """Never raises: any internal exception (HTTP/query failure) is
        converted to a target_up=False snapshot -- the frozen contract's own
        'we could not get a trustworthy reading' pathway (sec.3)."""
        try:
            return _snapshot_metrics(self._prom, self._queries, self._app_namespace, self._deployment)
        except Exception:  # noqa: BLE001 -- must never crash the live episode loop
            return RawMetricSnapshot(
                request_rate_rps=None, error_rate_rps=None, latency_p95_ms=None,
                requests_in_progress=None, cpu_cores=None, memory_bytes=None,
                target_up=False, query_error=False,
            )

    def _safe_replica_state(self) -> ReplicaState:
        """Never raises: a kubectl readback failure is reported as an
        all-zero replica state (equivalent to 'we could not observe
        capacity'), which the environment will see alongside a
        target_up=False metrics snapshot on the same tick in the common
        failure case (both derive from the same real infrastructure)."""
        try:
            return self._kubectl.get_replica_state()
        except Exception:  # noqa: BLE001
            return ReplicaState(desired=0, current=0, ready=0, available=0)

    # -- DynamicsBackend Protocol --------------------------------------------

    def reset(self) -> BackendSnapshot:
        """Scale to `initial_replicas`, bounded-wait for convergence
        (raises RestorationTimeoutError if it does not converge -- never
        proceeds on an unconfirmed state), then perform the REAL 60s
        pre-episode stabilization sleep (docs/rl_mdp_spec.md sec.20) before
        taking the very first metrics reading. Idempotent-safe: calling
        this again mid-training (next episode) repeats exactly the same
        procedure."""
        current_state = self._safe_replica_state()
        if (current_state.desired, current_state.current, current_state.ready, current_state.available) != (
            self.initial_replicas,
        ) * 4:
            self._kubectl.scale(self.initial_replicas)
            self._kubectl.wait_for_replica_state(self.initial_replicas, self.restoration_max_wait_s)
        self._last_desired_issued = self.initial_replicas

        self._sleep(self.pre_episode_stabilization_s)

        self._time_s = 0.0
        metrics = self._safe_metrics()
        replicas = self._safe_replica_state()
        return BackendSnapshot(time_s=self._time_s, metrics=metrics, replicas=replicas)

    def current_snapshot(self) -> BackendSnapshot:
        return BackendSnapshot(time_s=self._time_s, metrics=self._safe_metrics(), replicas=self._safe_replica_state())

    def set_desired_replicas(self, desired: int) -> None:
        """Clamps defensively (backend.py's documented requirement), then
        issues a fire-and-forget `kubectl scale`. RAISES `KubectlApiError`
        on failure -- unlike metrics/replica reads, an actuation failure
        must be visible to the caller (environment.py) as a distinct
        `k8s_api_failure` event, never silently swallowed."""
        desired = max(self.min_replicas, min(self.max_replicas, desired))
        if desired == self._last_desired_issued:
            return
        self._kubectl.scale(desired)  # raises KubectlApiError on failure
        self._last_desired_issued = desired

    def advance(self, dt_seconds: float) -> BackendSnapshot:
        """REAL `time.sleep(dt_seconds)` -- there is no simulated/scheduled
        transition here, only the genuine passage of wall-clock time on the
        live cluster. Never raises (see `_safe_metrics`/`_safe_replica_state`)."""
        if dt_seconds < 0:
            raise ValueError("dt_seconds must be non-negative")
        self._sleep(dt_seconds)
        self._time_s += dt_seconds
        metrics = self._safe_metrics()
        replicas = self._safe_replica_state()
        return BackendSnapshot(time_s=self._time_s, metrics=metrics, replicas=replicas)
