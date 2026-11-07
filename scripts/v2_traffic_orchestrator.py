"""scripts/v2_traffic_orchestrator.py

V2 Phase 3A PART 2/4: the V2-specific analogs of `scripts.traffic_orchestrator`'s
profile-resolution functions. Reuses the GENERIC parts of that module
unmodified (V5TransportManager, start_generator/stop_generator/
verify_generator_started/drain_in_flight_requests/resolve_next_available_run_id
-- none of these are coupled to V1's profile registry). Only
`resolve_profile_metadata()`/`resolve_profile_for_episode()` and the two
classes that call them directly (`TrafficOrchestrator.start_episode_traffic`,
`TrafficSyncedWrapper.reset()`) are V1-firewall-coupled -- this module
provides V2-firewalled equivalents rather than modifying the frozen-adjacent
V1 versions.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from autoscaler.src.rl_env.training_contract_v2 import V2_TRAINING_PROFILE_ALLOWLIST
from autoscaler.src.rl_env.training_schedule_v2 import build_training_profile_rotation
from loadgen.traffic_generator import parse_traffic_profile
from scripts import rl_live_common as common
from scripts import traffic_orchestrator as to

V2_PROFILE_DIR = to.PROJECT_ROOT / "configs" / "traffic_profiles" / "v2"

# V2 Phase 3D PART 3: retry budget for a TRANSIENT post-episode drain query
# failure (e.g. a brief monitoring-pod liveness-probe restart, confirmed
# live during V2 Phase 3C physical attempt 3's own investigation -- the
# underlying Prometheus container exited cleanly (reason=Completed,
# exitCode=0) after a liveness-probe timeout, not OOMKilled, and the whole
# cluster's monitoring stack recovered within well under a minute). ~75s
# comfortably covers that observed recovery time without masking a
# genuinely dead Prometheus for an unbounded period.
DEFAULT_DRAIN_QUERY_RETRY_BUDGET_SECONDS = 75.0
DEFAULT_DRAIN_QUERY_RETRY_BACKOFF_SECONDS = 5.0


def drain_in_flight_requests_with_retry(
    query_fn: Callable[[], "tuple[Optional[float], Optional[str]]"],
    timeout_seconds: float = to.DEFAULT_DRAIN_TIMEOUT_SECONDS,
    poll_interval_seconds: float = to.DEFAULT_DRAIN_POLL_INTERVAL_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    retry_budget_seconds: float = DEFAULT_DRAIN_QUERY_RETRY_BUDGET_SECONDS,
    retry_backoff_seconds: float = DEFAULT_DRAIN_QUERY_RETRY_BACKOFF_SECONDS,
) -> float:
    """V2 Phase 3D PART 3: a retry WRAPPER around the frozen-adjacent,
    COMPLETELY UNMODIFIED `to.drain_in_flight_requests()` -- that function
    itself is never touched, so V1's frozen scripts/evaluate_ppo_vs_hpa.py
    (which calls it, via `TrafficOrchestrator.drain_server_side_requests`,
    directly and unconditionally) is byte-for-byte unaffected by this
    addition; only V2's OWN orchestrator (see
    `V2TrafficOrchestrator.drain_server_side_requests` below) calls this
    wrapper at all.

    Retries the ENTIRE drain operation on `DrainQueryError` (a genuine
    Prometheus query failure -- connection refused, HTTP 5xx, network
    error) with a fixed backoff, for a bounded total retry budget. Does
    NOT retry on `DrainTimeoutError` -- that signal means Prometheus
    answered successfully but server-side requests never reached
    quiescence, which is a different, non-transient condition that must
    propagate immediately, unchanged. An unavailable Prometheus is NEVER
    interpreted as "zero in-flight requests" -- every retry still requires
    a genuine successful query result before concluding anything about
    drain status; if the budget expires, this raises `DrainQueryError`
    exactly as the wrapped function would have on its very first attempt,
    just later and with the accumulated context."""
    deadline = clock() + retry_budget_seconds
    while True:
        try:
            return to.drain_in_flight_requests(
                query_fn, timeout_seconds=timeout_seconds, poll_interval_seconds=poll_interval_seconds,
                sleep_fn=sleep_fn, clock=clock,
            )
        except to.DrainQueryError as exc:
            if clock() >= deadline:
                raise to.DrainQueryError(
                    f"drain query remained unavailable past the {retry_budget_seconds}s retry budget "
                    f"(transient-failure retry exhausted); last error: {exc}"
                ) from exc
            sleep_fn(retry_backoff_seconds)


class V2ProfileFirewallViolation(to.TrafficOrchestratorError):
    """Raised when a profile name is not in V2_TRAINING_PROFILE_ALLOWLIST,
    or its file is missing/unreadable -- fail-closed, mirrors V1's
    ProfileFirewallViolation in spirit without reusing V1's firewall
    function (which knows nothing about V2 profile names/paths)."""


def resolve_v2_profile_metadata(profile_name: str) -> to.ProfileMetadata:
    """V2 equivalent of resolve_profile_metadata() -- fail-closed against
    V2_TRAINING_PROFILE_ALLOWLIST and configs/traffic_profiles/v2/, never
    the V1 six-name registry."""
    if profile_name not in V2_TRAINING_PROFILE_ALLOWLIST:
        raise V2ProfileFirewallViolation(
            f"{profile_name!r} is not in V2_TRAINING_PROFILE_ALLOWLIST {V2_TRAINING_PROFILE_ALLOWLIST}"
        )
    path = V2_PROFILE_DIR / f"{profile_name}.yaml"
    if not path.exists():
        raise V2ProfileFirewallViolation(f"authorized V2 profile file missing on disk: {path}")

    actual_sha256 = common.sha256_file(path)
    profile = parse_traffic_profile(path)
    return to.ProfileMetadata(
        name=profile_name, path=path, sha256=actual_sha256, seed=profile.seed,
        duration_seconds=profile.total_duration, n_phases=len(profile.phases),
    )


def resolve_v2_profile_for_episode(episode_index: int, training_seed: int) -> str:
    """V2 equivalent of checkpoint_state.resolve_profile_for_episode() --
    delegates to the already-tested, deterministic, perfectly-balanced
    training_schedule_v2.build_training_profile_rotation() rather than a
    separately re-derived cycling algorithm. Cheap for the small episode
    counts a single pilot/session needs (rebuilds the schedule up to
    `episode_index` each call); not intended for very large episode
    indices without memoizing at the call site."""
    if episode_index < 0:
        raise ValueError(f"episode_index must be >= 0, got {episode_index}")
    schedule = build_training_profile_rotation(training_seed, episode_index + 1)
    return schedule[episode_index]


class V2TrafficOrchestrator(to.TrafficOrchestrator):
    """Overrides `start_episode_traffic` to resolve against the V2 profile
    registry, and `drain_server_side_requests` to use the bounded-retry
    wrapper (V2 Phase 3D PART 3) -- every other method (route management,
    generator process lifecycle, provenance) is inherited unchanged from
    `TrafficOrchestrator`."""

    def drain_server_side_requests(self) -> float:
        """V2 Phase 3D PART 3: identical to `TrafficOrchestrator.
        drain_server_side_requests()` except it calls the bounded-retry
        wrapper instead of `to.drain_in_flight_requests()` directly --
        same authoritative PromQL (via `self.drain_query_fn`/
        `self.prom_client`), same definition of "drained" (DRAIN_ZERO_EPSILON,
        unchanged), same timeout/poll-interval semantics. The ONLY
        difference is tolerating a bounded number of TRANSIENT query
        failures before failing closed."""
        query_fn = self.drain_query_fn
        if query_fn is None:
            promql = (
                f'sum(demo_app_requests_in_progress{{namespace="{self.drain_namespace}",'
                f'route="{self.drain_route}"}})'
            )
            query_fn = lambda: self.prom_client.query(promql)  # noqa: E731
        return drain_in_flight_requests_with_retry(
            query_fn=query_fn,
            timeout_seconds=self.drain_timeout_seconds,
            poll_interval_seconds=self.drain_poll_interval_seconds,
            sleep_fn=self.sleep_fn,
            clock=self.clock,
        )

    def start_episode_traffic(self, profile_name: str, episode_id: str) -> to.GeneratorHandle:
        profile = resolve_v2_profile_metadata(profile_name)
        run_id = to.resolve_next_available_run_id(episode_id, self.traffic_base_dir)
        route = self.ensure_route_established(run_id)
        if not to.verify_route_alive(route, self.http_get):
            raise to.RouteEstablishmentFailure(f"route {route.target_url} not alive before episode {episode_id}")
        handle = to.start_generator(profile, route, run_id, popen_factory=self.popen_factory, clock=self.clock)
        if not to.verify_generator_started(handle, sleep_fn=self.sleep_fn):
            to.stop_generator(handle)
            to.record_traffic_provenance(
                handle, episode_id=episode_id, session_index=self.session_index,
                exit_code=handle.process.poll(), reason="failed_to_start", ledger_path=self.ledger_path,
            )
            raise to.TrafficGeneratorFailure(f"generator failed to start for V2 profile {profile_name!r}, episode {episode_id}")
        return handle


class V2TrafficSyncedWrapper(to.TrafficSyncedWrapper):
    """Overrides ONLY `reset()` to resolve against the V2 profile registry
    and the V2 deterministic rotation -- `step()`/`close()` are inherited
    unchanged from `TrafficSyncedWrapper` (neither calls a profile resolver
    directly)."""

    current_profile_name: Optional[str] = None  # set on first reset(); readable by an external safety/telemetry callback

    def reset(self, **kwargs):
        episode_index = self._episode_index_provider()
        profile_name = resolve_v2_profile_for_episode(episode_index, self.training_seed)
        profile = resolve_v2_profile_metadata(profile_name)
        self.current_profile_name = profile_name  # V2 Phase 3A: readable by an external safety/telemetry callback

        self.unwrapped.episode_duration_s = profile.duration_seconds

        if hasattr(self.env, "traffic_profile"):
            self.env.traffic_profile = profile_name
        if hasattr(self.env, "profile_hash"):
            self.env.profile_hash = profile.sha256
        if hasattr(self.env, "global_schedule_episode_index"):
            self.env.global_schedule_episode_index = episode_index

        obs, info = self.env.reset(**kwargs)

        episode_id = f"v2_episode_{episode_index}_{profile_name}"
        self._current_episode_id = episode_id
        self._current_handle = self.orchestrator.start_episode_traffic(profile_name, episode_id)
        return obs, info
