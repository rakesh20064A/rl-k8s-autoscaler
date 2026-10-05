"""scripts/traffic_orchestrator.py

Phase 16Y: the live-PPO-training traffic-orchestration abstraction Phase16X
found missing. Reuses, rather than reinvents, three already-proven
mechanisms:

  * `loadgen.traffic_generator` (invoked as `python -m loadgen.traffic_generator`)
    -- the exact CLI already used correctly by
    `scripts/run_dynamics_calibration_slot.py` (three call sites).
  * The V5 Docker-relay workload transport
    (`resolve_minikube_docker_network` / `resolve_minikube_node_ip` /
    `resolve_loadtest_nodeport` / relay `docker run` / `docker port` /
    `check_relay_container_state`), mirrored here (not imported -- that
    script is a __main__ entry point, not a library, exactly the same
    reasoning as `autoscaler/src/rl_env/live_backend.py`'s `KubectlRunner`)
    via a dependency-injectable `V5TransportManager`.
  * `scripts.rl_live_common`'s existing workload firewall
    (`assert_training_eligible`, `TRAINING_PROFILES`, `FINAL_HOLDOUT_PROFILE`,
    `EXCLUDED_FROM_FORMAL_COMPARISON`) -- reused directly, not duplicated,
    then hardened further (Section "Firewall hardening" below).

Two protocol choices this module resolves, NEITHER already frozen anywhere
in Phase16U/16W/docs/rl_mdp_spec.md, cited rather than invented (see
`task_outputs/phase_16y_implementation_and_root_cause_report.md` for full
justification and the honest disclosure that no prior artifact addressed
these):

  * `--connection-mode fresh` -- NOT the generator's own default
    (`keepalive`). Cited from `docs/experimental_design_corrections.md`
    CORR-21 sec.7's explicit, already-frozen general principle: "keepalive
    must never be used for multi-replica HPA/RL benchmark traffic" --
    genuinely applicable here (PPO training targets a real, potentially
    multi-replica Service via the same NodePort/relay route), not a new
    invention.
  * `--arrival-process poisson`, `--seed <profile's own frozen seed>`,
    `--replay-schedule` omitted -- ALL are the traffic generator's own
    documented default behavior when unspecified, verified deterministic
    (same profile + same seed -> byte-identical generated schedule every
    replay, confirmed by direct source inspection of
    `generate_arrival_schedule()`, no wall-clock/OS-entropy seeding
    anywhere in the module). Using a tool's own designed default is not
    the same kind of scientific choice as overriding it, and is treated
    here as not requiring separate authorization.

CRITICAL SEPARATE FINDING (documented in the Phase16Y report, NOT fixed by
this module, out of this phase's authorized scope): `loadgen.traffic_
generator.parse_traffic_profile()` does not recognize the flat top-level
schema used by `step.yaml`/`burst.yaml`/`ramp.yaml`/`periodic.yaml`, and
silently degrades all four to a flat 2.0 RPS constant load for their full
duration -- verified directly (`parse_traffic_profile()` on all six
training profiles). `verify_profile_shape_not_degenerate()` below is a
defense this module adds specifically because of that finding: it
refuses to authorize live use of any non-`constant` profile whose parsed
shape collapses to a single `ConstantPhase`, catching exactly this defect
at runtime rather than silently running degenerate traffic.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import gymnasium as gym

from autoscaler.src.rl_env import episode_ledger
from autoscaler.src.rl_env.checkpoint_state import resolve_profile_for_episode
from collectors.metrics_collector import PrometheusClient
from loadgen.traffic_generator import ConstantPhase, parse_traffic_profile
from scripts import rl_live_common as common

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")
APP_NS = "autoscaler-app"
DEPLOYMENT = "demo-app"
LOADTEST_SVC = "demo-app-loadtest"

TRAFFIC_PROVENANCE_LEDGER_PATH = PROJECT_ROOT / "persistent-data" / "rl_training" / "traffic_provenance_ledger.jsonl"
TRAFFIC_OUTPUT_BASE_DIR = PROJECT_ROOT / "persistent-data" / "traffic"

# Frozen registry: training-profile name -> path, mirroring
# scripts.rl_live_common.TRAINING_PROFILES exactly (single source of truth
# for MEMBERSHIP; this dict adds the file path each name resolves to).
AUTHORIZED_TRAINING_PROFILE_PATHS: Dict[str, str] = {
    name: f"configs/traffic_profiles/{name}.yaml" for name in common.TRAINING_PROFILES
}

DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_MAX_IN_FLIGHT = 50
ROUTE_HEALTH_CHECK_PATH = "/health"
WORKLOAD_ROUTE = "/work"

# Index33 correction: bounded server-side drain after the client generator
# is stopped (Index32 diagnosis -- stop_generator() only terminates the
# CLIENT process; server-side in-flight /work requests were never awaited,
# letting their CPU-starvation effects bleed into the next episode's
# canonical-reset window). Deliberately much smaller than
# LiveKubernetesBackend.restoration_max_wait_s (120s, unchanged) so a drain
# failure surfaces fast and distinctly rather than being absorbed into that
# existing budget.
DEFAULT_DRAIN_TIMEOUT_SECONDS = 30.0
DEFAULT_DRAIN_POLL_INTERVAL_SECONDS = 2.0
# demo_app_requests_in_progress is a Prometheus Gauge of a non-negative
# integer count; an instant query always returns it as a float, so this is
# a defensive tolerance for float representation only, never a policy of
# "close enough to zero."
DRAIN_ZERO_EPSILON = 0.5
GENERATOR_START_GRACE_SECONDS = 5.0

# Phase16Z, task sec.7: a NARROW, seconds-denominated early-exit tolerance,
# NOT a percentage of profile duration (the prior 90%-of-duration rule
# could silently mask up to 240s of idle-but-uncaught time for the
# 2400s "mixed" profile -- multiple full PPO decision intervals). Grounded
# in measured overhead, not invented: `python3 -c "import loadgen.traffic_
# generator"` measured ~0.1-0.2s wall-clock on this machine (Phase16Z);
# the generator's own pre-dispatch setup (argument parsing, TrafficProfile
# construction, on-the-fly schedule generation via
# generate_arrival_schedule(), directory/file setup) adds modest further
# overhead, conservatively bounded well under 2s for these profile sizes.
# 2.0s is a ~10x margin over the measured baseline, and is small enough
# that it can NEVER mask even one full 30-second PPO decision interval --
# any exit earlier than (profile_duration - 2.0s) is now unconditionally
# treated as unexpected, regardless of the profile's own total duration.
GENERATOR_EARLY_EXIT_TOLERANCE_SECONDS = 2.0


class TrafficOrchestratorError(RuntimeError):
    """Base class for all fail-closed errors this module raises."""


class ProfileFirewallViolation(TrafficOrchestratorError):
    """Raised for any profile that is not one of the six authorized
    training profiles, or whose content hash does not match what was
    frozen at Phase16Y's own verification time."""


class ProfileShapeIntegrityError(TrafficOrchestratorError):
    """Raised when an authorized profile's file parses into a degenerate
    shape (see module docstring's CRITICAL SEPARATE FINDING)."""


class RouteEstablishmentFailure(TrafficOrchestratorError):
    """Raised when the V5 transport route cannot be established or
    verified alive before an episode's traffic would need to start."""


class TrafficGeneratorFailure(TrafficOrchestratorError):
    """Raised when the traffic generator fails to start, or exits
    unexpectedly early during an active episode. Callers (the session
    runner / environment wrapper) must treat this as a tooling failure,
    never silently continue as if the workload were still active, and
    never reclassify it as a safety abort."""


class DrainTimeoutError(TrafficOrchestratorError):
    """Index33: raised when server-side in-flight /work requests did not
    reach quiescence within the bounded drain window after the client
    generator was stopped. A distinct, surfaced orchestration/recovery
    failure -- callers must never silently proceed into canonical
    deployment reset on this condition."""


class DrainQueryError(TrafficOrchestratorError):
    """Index33: raised when the in-progress-requests Prometheus query
    itself fails (unreachable, malformed response, etc.). Fails closed --
    never silently assumes zero in-flight requests on a query error."""


CommandRunner = Callable[[List[str], float], subprocess.CompletedProcess]


def _default_command_runner(cmd: List[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)  # noqa: S603,S607


PopenFactory = Callable[..., "subprocess.Popen"]


@dataclass(frozen=True)
class ProfileMetadata:
    name: str
    path: Path
    sha256: str
    seed: int
    duration_seconds: float
    n_phases: int


def resolve_profile_metadata(profile_name: str, expected_sha256: Optional[str] = None) -> ProfileMetadata:
    """Fail-closed profile resolution (task sec.7): rejects anything not
    in the frozen six-name registry, and (if `expected_sha256` is given)
    rejects a profile whose current file content no longer matches it.
    ALSO fails closed via `verify_profile_shape_not_degenerate()` -- see
    module docstring."""
    common.assert_training_eligible(profile_name)  # raises WorkloadFirewallViolation for anything not in the 6
    if profile_name not in AUTHORIZED_TRAINING_PROFILE_PATHS:
        raise ProfileFirewallViolation(f"{profile_name!r} passed the firewall but has no registered path")

    path = PROJECT_ROOT / AUTHORIZED_TRAINING_PROFILE_PATHS[profile_name]
    if not path.exists():
        raise ProfileFirewallViolation(f"authorized profile file missing on disk: {path}")

    actual_sha256 = common.sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ProfileFirewallViolation(
            f"{profile_name!r} content hash changed: expected {expected_sha256}, got {actual_sha256} "
            "-- a frozen training profile must never be silently edited."
        )

    profile = parse_traffic_profile(path)
    verify_profile_shape_not_degenerate(profile_name, profile)

    return ProfileMetadata(
        name=profile_name, path=path, sha256=actual_sha256, seed=profile.seed,
        duration_seconds=profile.total_duration, n_phases=len(profile.phases),
    )


def verify_profile_shape_not_degenerate(profile_name: str, profile) -> None:
    """Raises ProfileShapeIntegrityError if a non-`constant` profile parses
    into exactly one ConstantPhase -- the exact signature of the discovered
    flat-schema parser gap (module docstring). `constant.yaml` is
    correctly, intentionally a single ConstantPhase and is exempted."""
    if profile_name == "constant":
        return
    if len(profile.phases) == 1 and isinstance(profile.phases[0], ConstantPhase):
        raise ProfileShapeIntegrityError(
            f"profile {profile_name!r} parsed into a single flat ConstantPhase "
            f"({profile.phases[0].target_rps if hasattr(profile.phases[0], 'target_rps') else '?'} RPS for "
            f"{profile.phases[0].duration_seconds}s) instead of its intended shape -- this is the known "
            "loadgen.traffic_generator.parse_traffic_profile() flat-schema gap documented in "
            "task_outputs/phase_16y_implementation_and_root_cause_report.md. Refusing to authorize live use "
            "of this profile until the parser (or the profile file's schema) is fixed in a separate phase."
        )


# ---------------------------------------------------------------------------
# V5 transport (mirrors run_dynamics_calibration_slot.py's relay path)
# ---------------------------------------------------------------------------


@dataclass
class RouteHandle:
    target_url: str
    container_name: str
    node_ip: str
    nodeport: str
    established_utc: str


@dataclass
class V5TransportManager:
    app_namespace: str = APP_NS
    loadtest_service: str = LOADTEST_SVC
    run: CommandRunner = field(default=_default_command_runner)
    timeout_s: float = 30.0

    def _run(self, cmd: List[str]) -> subprocess.CompletedProcess:
        return self.run(cmd, self.timeout_s)

    def resolve_minikube_docker_network(self) -> Optional[str]:
        proc = self._run(["docker", "inspect", "minikube", "--format", "{{json .NetworkSettings.Networks}}"])
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        import json as _json
        try:
            networks = _json.loads(proc.stdout)
        except ValueError:
            return None
        return next(iter(networks), None)

    def resolve_minikube_node_ip(self) -> Optional[str]:
        proc = self._run(["minikube", "-p", "minikube", "ip"])
        ip = proc.stdout.strip()
        return ip if proc.returncode == 0 and ip else None

    def resolve_loadtest_nodeport(self) -> Optional[str]:
        proc = self._run([
            "kubectl", "get", "svc", self.loadtest_service, "-n", self.app_namespace,
            "-o", "jsonpath={.spec.ports[0].nodePort}",
        ])
        port = proc.stdout.strip()
        return port if proc.returncode == 0 and port else None

    def establish_route(self, run_id: str) -> RouteHandle:
        """Mirrors start_loadtest_transport()'s structural (no-HTTP)
        resolution + relay container launch. Raises RouteEstablishmentFailure
        on any missing prerequisite -- never guesses or proceeds partially.

        Phase16AB fix: the relay's container port 8080 is now explicitly
        PUBLISHED to a Docker-assigned ephemeral host port via `-p 8080`
        (never a fixed/hard-coded host port). Verified directly against the
        real local Docker daemon (task_outputs/phase_16ab_real_docker_route_
        smoke_report.md): `docker run ... -p 8080 ...` publishes to
        `0.0.0.0:<ephemeral>` AND `[::]:<ephemeral>`; `docker port <c> 8080`
        reports both lines; `http://localhost:<ephemeral>` is genuinely
        reachable from the host and round-trips to the application's
        `/health` endpoint through the full relay -> NodePort -> Service ->
        pod path. The PRIOR command (Phase16Y/16Z/16AA) never published the
        port at all, so `docker port` always returned empty -- see
        task_outputs/phase_16x_retry_safety_and_tooling_event_audit.md for
        the original failure this corrects."""
        network = self.resolve_minikube_docker_network()
        if not network:
            raise RouteEstablishmentFailure("could not resolve the Minikube Docker network")
        node_ip = self.resolve_minikube_node_ip()
        if not node_ip:
            raise RouteEstablishmentFailure("could not resolve the Minikube node IP")
        nodeport = self.resolve_loadtest_nodeport()
        if not nodeport:
            raise RouteEstablishmentFailure(f"could not resolve the {self.loadtest_service!r} NodePort")

        container_name = f"rl-traffic-relay-{run_id}"
        run_cmd = [
            "docker", "run", "-d", "--rm", "--name", container_name, "--network", network,
            "-p", "8080",  # Phase16AB: publish container port 8080 to an ephemeral host port -- never hard-coded
            "alpine/socat", "TCP-LISTEN:8080,fork,reuseaddr", f"TCP:{node_ip}:{nodeport}",
        ]
        proc = self._run(run_cmd)
        if proc.returncode != 0:
            raise RouteEstablishmentFailure(f"relay container launch failed: {proc.stderr}")

        port_proc = self._run(["docker", "port", container_name, "8080"])
        host_port = self._parse_published_host_port(port_proc.stdout)
        if not host_port:
            self.teardown_route(RouteHandle(target_url="", container_name=container_name, node_ip=node_ip, nodeport=nodeport, established_utc=""))
            raise RouteEstablishmentFailure("relay container started but its published host port could not be resolved")

        return RouteHandle(
            target_url=f"http://localhost:{host_port}",
            container_name=container_name, node_ip=node_ip, nodeport=nodeport,
            established_utc=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _parse_published_host_port(docker_port_stdout: str) -> str:
        """`docker port <container> 8080` prints one line per bound address,
        e.g. `0.0.0.0:50713` and `[::]:50713` (verified empirically, real
        Docker daemon, Phase16AB). Prefers the IPv4 (`0.0.0.0:`) line --
        reachable from the host as `localhost:<port>` -- and falls back to
        any other non-empty line (e.g. a bracketed IPv6 line) only if no
        IPv4 line is present. Returns "" (never guesses/fabricates a port)
        if the output is empty -- i.e. the port was never published."""
        lines = [ln for ln in docker_port_stdout.strip().splitlines() if ln.strip()]
        if not lines:
            return ""
        ipv4_lines = [ln for ln in lines if ln.startswith("0.0.0.0:") or ln.startswith("127.0.0.1:")]
        chosen = ipv4_lines[0] if ipv4_lines else lines[0]
        return chosen.rsplit(":", 1)[-1].strip()

    def check_relay_container_state(self, container_name: str) -> "tuple[Optional[bool], Optional[str]]":
        proc = self._run(["docker", "inspect", "--format", "{{.State.Running}}|{{.State.ExitCode}}", container_name])
        if proc.returncode != 0 or "|" not in proc.stdout:
            return None, None
        running_str, exit_code = proc.stdout.strip().split("|", 1)
        return running_str == "true", exit_code

    def teardown_route(self, route: RouteHandle) -> None:
        self._run(["docker", "rm", "-f", route.container_name])


def verify_route_alive(route: RouteHandle, http_get: Callable[[str, float], "tuple[bool, str]"]) -> bool:
    """`http_get(url, timeout) -> (ok, message)` is injected for
    testability. Hits the app's own lightweight, zero-workload
    `GET /health` endpoint -- never the `/work` endpoint, never real
    training traffic."""
    ok, _msg = http_get(f"{route.target_url}{ROUTE_HEALTH_CHECK_PATH}", DEFAULT_TIMEOUT_SECONDS)
    return ok


# ---------------------------------------------------------------------------
# Traffic-generator subprocess lifecycle
# ---------------------------------------------------------------------------


@dataclass
class GeneratorHandle:
    process: "subprocess.Popen"
    profile: ProfileMetadata
    run_id: str
    route: RouteHandle
    start_monotonic: float
    start_utc: str
    command: List[str]


def resolve_next_available_run_id(episode_id: str, traffic_base_dir: Path = TRAFFIC_OUTPUT_BASE_DIR) -> str:
    """Retry-safe physical run_id allocation: a scientific retry (same
    `episode_id`, new physical attempt) must not collide with a prior
    attempt's traffic output directory. An in-memory counter alone is not
    safe here because a retry is typically a brand-new process, which
    resets any in-memory counter to 0 -- so this scans PERSISTED on-disk
    evidence under `traffic_base_dir` instead, choosing the first
    `<episode_id>_run{N}` (N starting at 1) whose directory does not
    already exist. Never deletes, renames, or overwrites an existing
    directory; different episode_ids are independent by construction
    (distinct prefixes)."""
    n = 1
    while (traffic_base_dir / f"{episode_id}_run{n}").exists():
        n += 1
    return f"{episode_id}_run{n}"


def build_generator_command(profile: ProfileMetadata, route: RouteHandle, run_id: str) -> List[str]:
    """The exact `python -m loadgen.traffic_generator` invocation for live
    PPO training -- `--seed` from the profile's own frozen field,
    `--replay-schedule` OMITTED (generator synthesizes deterministically,
    verified reproducible), `--connection-mode fresh` (CORR-21, cited not
    invented), `--arrival-process poisson` (the generator's own default,
    stated explicitly rather than silently relying on its absence)."""
    return [
        VENV_PYTHON, "-m", "loadgen.traffic_generator",
        "--profile", str(profile.path),
        "--experiment-id", run_id,
        "--target-url", route.target_url,
        "--arrival-process", "poisson",
        "--connection-mode", "fresh",
        "--seed", str(profile.seed),
        "--timeout", str(DEFAULT_TIMEOUT_SECONDS),
        "--max-in-flight", str(DEFAULT_MAX_IN_FLIGHT),
    ]


def start_generator(
    profile: ProfileMetadata, route: RouteHandle, run_id: str,
    popen_factory: PopenFactory = subprocess.Popen,
    clock: Callable[[], float] = time.monotonic,
) -> GeneratorHandle:
    command = build_generator_command(profile, route, run_id)
    process = popen_factory(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return GeneratorHandle(
        process=process, profile=profile, run_id=run_id, route=route,
        start_monotonic=clock(), start_utc=datetime.now(timezone.utc).isoformat(), command=command,
    )


def verify_generator_started(
    handle: GeneratorHandle, grace_seconds: float = GENERATOR_START_GRACE_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> bool:
    """A bounded grace period, then a liveness poll -- catches the
    'generator fails before episode traffic begins' case (task sec.8) so
    the caller never proceeds to a scientific PPO step believing traffic
    is live when it never started."""
    sleep_fn(grace_seconds)
    return handle.process.poll() is None


def poll_generator(handle: GeneratorHandle) -> Optional[int]:
    return handle.process.poll()


def generator_exit_was_expected(
    handle: GeneratorHandle, exit_code: int, clock: Callable[[], float] = time.monotonic,
) -> bool:
    """Phase16Z (task sec.7): True only if the generator exited with code 0
    AND ran for at least `profile.duration_seconds -
    GENERATOR_EARLY_EXIT_TOLERANCE_SECONDS` -- a narrow, fixed,
    seconds-denominated tolerance for process-launch/clock-measurement
    jitter only, NOT a percentage of the profile's duration (see the
    constant's own definition for the measured justification). This
    directly enforces the frozen "one episode = one full traffic-profile
    execution" rule: PPO must never be allowed to continue learning from a
    materially idle tail after the generator has genuinely stopped early.
    Anything else (a nonzero exit at any time, or a zero exit before this
    tolerance window) is 'unexpected' and must be treated as a tooling
    failure (task sec.8), never silently absorbed."""
    if exit_code != 0:
        return False
    elapsed = clock() - handle.start_monotonic
    return elapsed >= handle.profile.duration_seconds - GENERATOR_EARLY_EXIT_TOLERANCE_SECONDS


def stop_generator(
    handle: GeneratorHandle, terminate_wait_seconds: float = 10.0,
) -> Optional[int]:
    """Terminate -> wait -> escalate to kill only if necessary (task
    sec.4's 'escalate kill only when necessary'). Returns the final exit
    code. Safe to call on an already-exited process."""
    if handle.process.poll() is None:
        handle.process.terminate()
        try:
            handle.process.wait(timeout=terminate_wait_seconds)
        except subprocess.TimeoutExpired:
            handle.process.kill()
            handle.process.wait(timeout=terminate_wait_seconds)
    return handle.process.returncode


def drain_in_flight_requests(
    query_fn: Callable[[], "tuple[Optional[float], Optional[str]]"],
    timeout_seconds: float = DEFAULT_DRAIN_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_DRAIN_POLL_INTERVAL_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> float:
    """Index33 correction sec.2: bounded, deterministic wait for server-side
    `/work` in-flight requests to reach quiescence. Called AFTER
    `stop_generator()` has already stopped the CLIENT process and BEFORE the
    next episode's canonical deployment reset begins consuming its own
    separate `restoration_max_wait_s` budget (Index32 diagnosis: that gap
    let stale server-side CPU-starvation effects bleed into the next
    episode's reset window). Uses the project's EXISTING
    `demo_app_requests_in_progress` Prometheus gauge via the injected
    `query_fn` -- no new metric invented.

    `query_fn` must return `(value, error)` matching
    `PrometheusClient.query()`'s own contract: `(float, None)` on success,
    `(None, None)` if the series is genuinely absent (no error) -- treated
    as quiescent, since this gauge is only ever registered once `/work` has
    been called at least once and an absent-without-error read here most
    plausibly reflects a pod that has already restarted and lost all
    in-process state, which IS zero in-flight requests, not an unknown --
    and `(None, error_str)` on a real query failure, which is RAISED, never
    silently treated as zero.

    Returns the last observed value on success (>= 0, < DRAIN_ZERO_EPSILON).
    Raises `DrainQueryError` on a query failure (fail-closed). Raises
    `DrainTimeoutError` if the bounded window elapses without reaching
    quiescence -- callers must never catch this and silently proceed."""
    deadline = clock() + timeout_seconds
    while True:
        value, err = query_fn()
        if err is not None:
            raise DrainQueryError(f"in-flight-request query failed: {err}")
        observed = value if value is not None else 0.0
        if observed < DRAIN_ZERO_EPSILON:
            return observed
        if clock() >= deadline:
            raise DrainTimeoutError(
                f"server-side in-flight requests did not reach quiescence within "
                f"{timeout_seconds}s (last observed={observed})"
            )
        sleep_fn(poll_interval_seconds)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def record_traffic_provenance(
    handle: GeneratorHandle, *, episode_id: str, session_index: Optional[int],
    exit_code: Optional[int], reason: str, ledger_path: Path = TRAFFIC_PROVENANCE_LEDGER_PATH,
) -> None:
    """Appends one immutable provenance row per traffic-generator
    lifecycle event (task sec.6). Never overwritten, never rewound."""
    row = {
        "profile_name": handle.profile.name,
        "profile_path": str(handle.profile.path.relative_to(PROJECT_ROOT)),
        "profile_sha256": handle.profile.sha256,
        "profile_seed": handle.profile.seed,
        "traffic_run_id": handle.run_id,
        "start_utc": handle.start_utc,
        "end_utc": datetime.now(timezone.utc).isoformat(),
        "exit_code": exit_code,
        "reason": reason,
        "episode_id": episode_id,
        "session_index": session_index,
        "target_url": handle.route.target_url,
        "route_container": handle.route.container_name,
        "command": handle.command,
    }
    episode_ledger.append_row(ledger_path, row)


# ---------------------------------------------------------------------------
# Facade combining the above, plus the environment-lifecycle wrapper
# ---------------------------------------------------------------------------


@dataclass
class TrafficOrchestrator:
    """Ties transport + generator lifecycle + provenance together behind a
    small, dependency-injectable interface. Every dependency (command
    runner, Popen factory, HTTP client, clocks/sleeps) is injectable so the
    whole class is exercisable offline with fakes -- no real subprocess,
    no real network, in tests."""

    transport: V5TransportManager = field(default_factory=V5TransportManager)
    popen_factory: PopenFactory = subprocess.Popen
    http_get: Callable[[str, float], "tuple[bool, str]"] = None  # type: ignore[assignment]
    sleep_fn: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    ledger_path: Path = TRAFFIC_PROVENANCE_LEDGER_PATH
    session_index: Optional[int] = None

    traffic_base_dir: Path = TRAFFIC_OUTPUT_BASE_DIR

    # Index33 correction: server-side drain dependencies, injectable like
    # every other dependency on this class (no real Prometheus/network call
    # in tests -- `drain_query_fn` overrides `prom_client` entirely when set).
    prom_client: PrometheusClient = field(default_factory=PrometheusClient)
    drain_namespace: str = APP_NS
    drain_route: str = WORKLOAD_ROUTE
    drain_timeout_seconds: float = DEFAULT_DRAIN_TIMEOUT_SECONDS
    drain_poll_interval_seconds: float = DEFAULT_DRAIN_POLL_INTERVAL_SECONDS
    drain_query_fn: Optional[Callable[[], "tuple[Optional[float], Optional[str]]"]] = None

    _route: Optional[RouteHandle] = field(default=None, init=False, repr=False)

    def ensure_route_established(self, run_id_seed: str) -> RouteHandle:
        if self._route is not None:
            running, _ = self.transport.check_relay_container_state(self._route.container_name)
            if running:
                return self._route
        self._route = self.transport.establish_route(run_id_seed)
        return self._route

    def verify_route(self) -> bool:
        if self._route is None:
            return False
        return verify_route_alive(self._route, self.http_get)

    def start_episode_traffic(self, profile_name: str, episode_id: str) -> GeneratorHandle:
        profile = resolve_profile_metadata(profile_name)
        run_id = resolve_next_available_run_id(episode_id, self.traffic_base_dir)
        route = self.ensure_route_established(run_id)
        if not verify_route_alive(route, self.http_get):
            raise RouteEstablishmentFailure(f"route {route.target_url} not alive before episode {episode_id}")
        handle = start_generator(profile, route, run_id, popen_factory=self.popen_factory, clock=self.clock)
        if not verify_generator_started(handle, sleep_fn=self.sleep_fn):
            stop_generator(handle)
            record_traffic_provenance(
                handle, episode_id=episode_id, session_index=self.session_index,
                exit_code=handle.process.poll(), reason="failed_to_start", ledger_path=self.ledger_path,
            )
            raise TrafficGeneratorFailure(f"generator failed to start for profile {profile_name!r}, episode {episode_id}")
        return handle

    def check_liveness_or_raise(self, handle: GeneratorHandle, episode_id: str) -> None:
        status = poll_generator(handle)
        if status is not None and not generator_exit_was_expected(handle, status, clock=self.clock):
            record_traffic_provenance(
                handle, episode_id=episode_id, session_index=self.session_index,
                exit_code=status, reason="unexpected_early_exit", ledger_path=self.ledger_path,
            )
            raise TrafficGeneratorFailure(
                f"traffic generator for profile {handle.profile.name!r} exited unexpectedly "
                f"(code={status}) during episode {episode_id!r}"
            )

    def stop_episode_traffic(self, handle: GeneratorHandle, episode_id: str) -> None:
        exit_code = stop_generator(handle)
        reason = "episode_end_cleanup" if exit_code in (0, None, -15) else "episode_end_nonzero_exit"
        record_traffic_provenance(
            handle, episode_id=episode_id, session_index=self.session_index,
            exit_code=exit_code, reason=reason, ledger_path=self.ledger_path,
        )

    def drain_server_side_requests(self) -> float:
        """Index33 correction sec.2: call AFTER `stop_episode_traffic()` and
        BEFORE relying on the next episode's deployment-readiness wait.
        Raises `DrainQueryError`/`DrainTimeoutError` (never silently
        proceeds) -- see `drain_in_flight_requests()` for the full
        contract."""
        query_fn = self.drain_query_fn
        if query_fn is None:
            promql = (
                f'sum(demo_app_requests_in_progress{{namespace="{self.drain_namespace}",'
                f'route="{self.drain_route}"}})'
            )
            query_fn = lambda: self.prom_client.query(promql)  # noqa: E731
        return drain_in_flight_requests(
            query_fn=query_fn,
            timeout_seconds=self.drain_timeout_seconds,
            poll_interval_seconds=self.drain_poll_interval_seconds,
            sleep_fn=self.sleep_fn,
            clock=self.clock,
        )

    def force_cleanup(self) -> None:
        """Safety-net teardown for exception paths -- always safe to call,
        never raises."""
        if self._route is not None:
            try:
                self.transport.teardown_route(self._route)
            except Exception:  # noqa: BLE001
                pass
            self._route = None


class TrafficSyncedWrapper(gym.Wrapper):
    """Phase16Y, task sec.5: synchronizes traffic-generator lifecycle with
    the RL episode lifecycle at the environment layer, per the frozen
    ordering (docs/rl_mdp_spec.md sec.20): canonical reset -> readiness ->
    stabilization -> traffic starts at episode t=0 -> PPO steps while
    traffic is active -> traffic stops at episode end -> next episode's
    workload follows the frozen scheduling rule
    (resolve_profile_for_episode()). Must be the OUTERMOST wrapper
    (wrapping `LedgeredEpisodeWrapper`, which itself wraps the raw
    `KubernetesAutoscalerEnv`), for two reasons: (1) it mutates
    `self.unwrapped.episode_duration_s` to match each episode's assigned
    profile's own duration -- the frozen contract's own "episode = one
    full traffic-profile execution" rule (sec.20), which the environment
    class itself does not enforce -- and `.unwrapped` walks through any
    number of nested wrappers to reach the raw env regardless of nesting
    depth; (2) it updates `self.env.traffic_profile`/`self.env.profile_hash`
    (the LedgeredEpisodeWrapper instance's own mutable attributes) with the
    freshly-resolved per-episode profile BEFORE calling `self.env.reset()`,
    so that wrapper's own ledger-row construction (which reads
    `self.traffic_profile` while building `self._episode_id`) reflects the
    CORRECT profile for this specific episode rather than a stale
    constructor-time placeholder -- profiles vary per episode within a
    single session/rollout, unlike the single-profile-per-wrapper-lifetime
    assumption a naive integration would otherwise make."""

    def __init__(self, env: gym.Env, orchestrator: TrafficOrchestrator, episode_index_provider: Callable[[], int], training_seed: int):
        super().__init__(env)
        self.orchestrator = orchestrator
        self._episode_index_provider = episode_index_provider
        self.training_seed = training_seed
        self._current_handle: Optional[GeneratorHandle] = None
        self._current_episode_id: Optional[str] = None

    def reset(self, **kwargs):
        episode_index = self._episode_index_provider()
        profile_name = resolve_profile_for_episode(episode_index, self.training_seed)
        profile = resolve_profile_metadata(profile_name)  # fail-closed: firewall + hash + shape-integrity

        # Frozen contract (docs/rl_mdp_spec.md sec.20): episode duration ==
        # this episode's assigned profile's own total duration.
        self.unwrapped.episode_duration_s = profile.duration_seconds

        # Propagate the freshly-resolved profile to the inner
        # LedgeredEpisodeWrapper's OWN mutable attributes BEFORE calling its
        # reset() -- that wrapper builds its episode_id from
        # self.traffic_profile at reset() time, and profiles vary per
        # episode here (see class docstring, reason 2).
        if hasattr(self.env, "traffic_profile"):
            self.env.traffic_profile = profile_name
        if hasattr(self.env, "profile_hash"):
            self.env.profile_hash = profile.sha256
        # Phase17: `episode_index` here IS the one authoritative global
        # schedule index (it is what resolve_profile_for_episode() itself
        # is keyed on) -- propagate it down so the inner
        # LedgeredEpisodeWrapper derives a globally-unique
        # scientific_episode_id from it, rather than maintaining its own
        # independent, per-process-restarting counter for this purpose.
        if hasattr(self.env, "global_schedule_episode_index"):
            self.env.global_schedule_episode_index = episode_index

        obs, info = self.env.reset(**kwargs)  # performs canonical reset + readiness + stabilization internally

        episode_id = f"episode_{episode_index}_{profile_name}"
        self._current_episode_id = episode_id
        # Traffic starts exactly here -- immediately after stabilization,
        # i.e. episode t=0 -- never before.
        self._current_handle = self.orchestrator.start_episode_traffic(profile_name, episode_id)
        return obs, info

    def step(self, action: int):
        if self._current_handle is not None:
            self.orchestrator.check_liveness_or_raise(self._current_handle, self._current_episode_id)

        obs, reward, terminated, truncated, info = self.env.step(action)

        if (terminated or truncated) and self._current_handle is not None:
            self.orchestrator.stop_episode_traffic(self._current_handle, self._current_episode_id)
            self._current_handle = None
            # Index33 correction: stop_episode_traffic() only terminated the
            # CLIENT generator process -- block here (bounded) until
            # server-side in-flight /work requests actually reach
            # quiescence, before the NEXT reset() call relies on
            # deployment-readiness convergence. Raises (never silently
            # continues) if drain fails or times out.
            self.orchestrator.drain_server_side_requests()

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self._current_handle is not None:
            self.orchestrator.stop_episode_traffic(self._current_handle, self._current_episode_id)
            self._current_handle = None
        self.orchestrator.force_cleanup()
        super().close()
