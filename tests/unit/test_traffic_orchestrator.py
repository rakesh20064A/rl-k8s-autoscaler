"""tests/unit/test_traffic_orchestrator.py

Phase 16Y: offline tests for scripts/traffic_orchestrator.py. NO real
subprocess, NO real HTTP, NO real docker/kubectl/minikube access -- every
dependency is a fake/mock via constructor injection.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

import gymnasium as gym
import pytest

from scripts import traffic_orchestrator as to


def _proc(rc: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr=stderr)


@dataclass
class FakeDockerKubectl:
    """Fake command runner standing in for docker/kubectl/minikube CLI calls.

    Phase16AB fix: `docker port` now models REAL Docker behavior --
    it only returns a host-port mapping for a container whose `docker run`
    command actually contained a publish argument (`-p`/`-P`). A `docker
    run` WITHOUT a publish flag correctly results in empty `docker port`
    output, exactly reproducing the real-Docker failure this phase fixes
    (previously, this fake unconditionally returned a fake host port
    regardless of whether the simulated `docker run` published anything --
    the exact blind spot that let the Phase16X-Retry defect through)."""

    network: str = "minikube"
    node_ip: str = "192.168.49.2"
    nodeport: str = "31234"
    host_port: str = "54321"
    relay_run_should_fail: bool = False
    relay_running: bool = True
    calls: List[List[str]] = field(default_factory=list)
    _published_containers: dict = field(default_factory=dict)

    def run(self, cmd: List[str], timeout: float) -> subprocess.CompletedProcess:
        self.calls.append(cmd)
        if cmd[:2] == ["docker", "inspect"] and "minikube" in cmd:
            return _proc(stdout=f'{{"{self.network}": {{}}}}')
        if cmd[:2] == ["minikube", "-p"] or (cmd[:1] == ["minikube"] and "ip" in cmd):
            return _proc(stdout=self.node_ip)
        if cmd[:2] == ["kubectl", "get"]:
            return _proc(stdout=self.nodeport)
        if cmd[:2] == ["docker", "run"]:
            if self.relay_run_should_fail:
                return _proc(rc=1, stderr="docker run failed")
            container_name = cmd[cmd.index("--name") + 1]
            published = "-p" in cmd or "-P" in cmd
            self._published_containers[container_name] = published
            return _proc(stdout="containerid123")
        if cmd[:2] == ["docker", "port"]:
            container_name = cmd[2]
            if self._published_containers.get(container_name, False):
                return _proc(stdout=f"0.0.0.0:{self.host_port}\n[::]:{self.host_port}")
            return _proc(stdout="")  # real Docker: no publish flag -> empty output
        if cmd[:2] == ["docker", "inspect"]:  # relay container state
            state = "true" if self.relay_running else "false"
            return _proc(stdout=f"{state}|0")
        if cmd[:2] == ["docker", "rm"]:
            return _proc(rc=0)
        raise AssertionError(f"unexpected command: {cmd}")


@dataclass
class FakePopen:
    """Stands in for subprocess.Popen -- controllable poll()/wait()/terminate()/kill()."""

    args: List[str] = field(default_factory=list)
    _returncode: Optional[int] = None
    terminate_called: bool = False
    kill_called: bool = False
    stdout: Optional[object] = None
    stderr: Optional[object] = None

    def poll(self) -> Optional[int]:
        return self._returncode

    def wait(self, timeout: Optional[float] = None) -> int:
        if self._returncode is None:
            self._returncode = 0
        return self._returncode

    def terminate(self) -> None:
        self.terminate_called = True
        if self._returncode is None:
            self._returncode = -15

    def kill(self) -> None:
        self.kill_called = True
        self._returncode = -9

    @property
    def returncode(self):
        return self._returncode

    def set_exit(self, code: int) -> None:
        self._returncode = code


def _fake_popen_factory(handles: List[FakePopen], exit_code: Optional[int] = None):
    def factory(cmd, **kwargs):
        p = FakePopen(args=cmd, _returncode=exit_code)
        handles.append(p)
        return p
    return factory


def _fake_http_get_always_ok(url: str, timeout: float):
    return True, "ok"


def _fake_http_get_always_fail(url: str, timeout: float):
    return False, "connection refused"


def _no_sleep(_seconds: float) -> None:
    return None


class _FakeClock:
    def __init__(self, start=0.0):
        self.t = start

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


# --- Profile resolution / firewall ------------------------------------------------


class TestProfileResolutionFirewall:
    def test_authorized_working_profiles_accepted(self):
        """Phase16Z: after the parse_traffic_profile() fix, all six frozen
        training profiles parse correctly and pass the shape-integrity
        check -- not just constant/mixed (the Phase16Y-era state)."""
        for name in ("constant", "step", "burst", "ramp", "periodic", "mixed"):
            meta = to.resolve_profile_metadata(name)
            assert meta.name == name

    def test_final_holdout_rejected(self):
        with pytest.raises(Exception):  # noqa: B017 -- WorkloadFirewallViolation from rl_live_common
            to.resolve_profile_metadata("iteration_rl_final_holdout_v1")

    def test_unseen_eval_rejected(self):
        with pytest.raises(Exception):  # noqa: B017 -- WorkloadFirewallViolation from rl_live_common
            to.resolve_profile_metadata("unseen_eval")

    def test_unregistered_profile_rejected(self):
        with pytest.raises(Exception):  # noqa: B017
            to.resolve_profile_metadata("totally_made_up_profile")

    def test_tampered_profile_hash_rejected(self):
        with pytest.raises(to.ProfileFirewallViolation, match="content hash changed"):
            to.resolve_profile_metadata("constant", expected_sha256="0" * 64)

    def test_matching_hash_accepted(self):
        real_meta = to.resolve_profile_metadata("constant")
        # must not raise
        to.resolve_profile_metadata("constant", expected_sha256=real_meta.sha256)

    def test_shape_integrity_check_still_rejects_a_genuinely_degenerate_profile(self):
        """Phase16Z: the parser fix means all six real training profiles
        now pass (see test_authorized_working_profiles_accepted). This test
        proves verify_profile_shape_not_degenerate() itself still correctly
        rejects a genuinely malformed non-constant profile -- the safety
        check is still doing real work, not merely disabled."""
        from loadgen.traffic_generator import ConstantPhase, TrafficProfile

        malformed = TrafficProfile(
            profile_name="step", description="synthetic malformed", seed=42,
            endpoint="/work", intensity=3,
            phases=[ConstantPhase("default", 900.0, 2.0)],  # exactly the old bug's signature
        )
        with pytest.raises(to.ProfileShapeIntegrityError):
            to.verify_profile_shape_not_degenerate("step", malformed)

    def test_shape_integrity_check_accepts_constant_as_a_single_constant_phase(self):
        """constant.yaml is INTENTIONALLY a single ConstantPhase -- must
        never be flagged as degenerate."""
        from loadgen.traffic_generator import ConstantPhase, TrafficProfile

        genuine = TrafficProfile(
            profile_name="constant", description="", seed=42, endpoint="/work",
            intensity=3, phases=[ConstantPhase("constant", 600.0, 20.0)],
        )
        to.verify_profile_shape_not_degenerate("constant", genuine)  # must not raise


# --- Generator command construction -----------------------------------------------


class TestGeneratorCommand:
    def test_command_uses_frozen_profile_seed_and_omits_replay_schedule(self):
        profile = to.resolve_profile_metadata("constant")
        route = to.RouteHandle(target_url="http://localhost:9999", container_name="c", node_ip="1.2.3.4", nodeport="30000", established_utc="t")
        cmd = to.build_generator_command(profile, route, "run1")
        assert cmd[0] == to.VENV_PYTHON
        assert cmd[1:3] == ["-m", "loadgen.traffic_generator"]
        assert "--profile" in cmd and str(profile.path) in cmd
        assert "--experiment-id" in cmd and "run1" in cmd
        assert "--target-url" in cmd and route.target_url in cmd
        assert "--seed" in cmd and str(profile.seed) in cmd
        assert "--connection-mode" in cmd and "fresh" in cmd
        assert "--arrival-process" in cmd and "poisson" in cmd
        assert "--replay-schedule" not in cmd

    def test_correct_profile_hash_and_seed_propagation(self):
        profile = to.resolve_profile_metadata("mixed")
        assert profile.seed == 42
        assert len(profile.sha256) == 64


# --- V5 transport ------------------------------------------------------------------


class TestV5Transport:
    def test_establish_route_success(self):
        fk = FakeDockerKubectl()
        manager = to.V5TransportManager(run=fk.run)
        route = manager.establish_route("run1")
        assert route.target_url == f"http://localhost:{fk.host_port}"
        assert route.node_ip == fk.node_ip

    def test_establish_route_fails_when_network_unresolvable(self):
        fk = FakeDockerKubectl(network="")
        manager = to.V5TransportManager(run=fk.run)
        with pytest.raises(to.RouteEstablishmentFailure):
            manager.establish_route("run1")

    def test_establish_route_fails_when_nodeport_unresolvable(self):
        fk = FakeDockerKubectl(nodeport="")
        manager = to.V5TransportManager(run=fk.run)
        with pytest.raises(to.RouteEstablishmentFailure):
            manager.establish_route("run1")

    def test_establish_route_fails_when_relay_launch_fails(self):
        fk = FakeDockerKubectl(relay_run_should_fail=True)
        manager = to.V5TransportManager(run=fk.run)
        with pytest.raises(to.RouteEstablishmentFailure):
            manager.establish_route("run1")

    def test_route_health_check_ok(self):
        fk = FakeDockerKubectl()
        manager = to.V5TransportManager(run=fk.run)
        route = manager.establish_route("run1")
        assert to.verify_route_alive(route, _fake_http_get_always_ok) is True

    def test_route_health_check_failure(self):
        fk = FakeDockerKubectl()
        manager = to.V5TransportManager(run=fk.run)
        route = manager.establish_route("run1")
        assert to.verify_route_alive(route, _fake_http_get_always_fail) is False

    def test_relay_run_command_contains_the_publish_argument(self):
        """Phase16AB: proves the production `docker run` command actually
        includes a port-publish flag -- the exact thing that was missing
        and caused the Phase16X-Retry failure."""
        fk = FakeDockerKubectl()
        manager = to.V5TransportManager(run=fk.run)
        manager.establish_route("run1")
        run_calls = [c for c in fk.calls if c[:2] == ["docker", "run"]]
        assert len(run_calls) == 1
        assert "-p" in run_calls[0] or "-P" in run_calls[0]

    def test_no_hardcoded_host_port_in_the_publish_argument(self):
        """Docker must assign the ephemeral host port -- the command must
        never request a SPECIFIC fixed host port (e.g. '-p 1234:8080')."""
        fk = FakeDockerKubectl()
        manager = to.V5TransportManager(run=fk.run)
        manager.establish_route("run1")
        run_cmd = next(c for c in fk.calls if c[:2] == ["docker", "run"])
        publish_value = run_cmd[run_cmd.index("-p") + 1]
        assert ":" not in publish_value  # a fixed host:container mapping would contain ':'

    def test_unpublished_relay_reproduces_the_original_phase16x_retry_failure(self):
        """Regression test (task sec.4): simulates the EXACT pre-fix
        command shape (docker run with no publish flag) and proves it
        reproduces the identical RouteEstablishmentFailure that halted
        Phase16X-Retry at 63 seconds with zero timesteps committed."""
        class UnpublishedFakeDockerKubectl(FakeDockerKubectl):
            def run(self, cmd, timeout):
                if cmd[:2] == ["docker", "run"]:
                    # Reproduce the ORIGINAL defect: strip any -p/-P the
                    # caller passed, so docker port always reports nothing --
                    # exactly the pre-Phase16AB production behavior.
                    stripped = [tok for tok in cmd if tok not in ("-p", "-P")]
                    return super().run(stripped, timeout)
                return super().run(cmd, timeout)

        fk = UnpublishedFakeDockerKubectl()
        manager = to.V5TransportManager(run=fk.run)
        with pytest.raises(to.RouteEstablishmentFailure, match="published host port could not be resolved"):
            manager.establish_route("run1")

    def test_published_relay_parses_ipv4_line_preferentially(self):
        """docker port real output is two lines (IPv4 + IPv6, verified
        empirically against a real Docker daemon, Phase16AB). The IPv4
        0.0.0.0 line must be preferred -- it is what 'localhost:<port>' on
        the Mac host actually resolves through."""
        host_port = to.V5TransportManager._parse_published_host_port("0.0.0.0:50713\n[::]:50713")
        assert host_port == "50713"

    def test_target_url_is_host_reachable_localhost_form(self):
        fk = FakeDockerKubectl(host_port="50713")
        manager = to.V5TransportManager(run=fk.run)
        route = manager.establish_route("run1")
        assert route.target_url == "http://localhost:50713"
        assert "localhost" in route.target_url  # never a container-only/internal address


# --- Generator lifecycle -------------------------------------------------------------


def _route():
    return to.RouteHandle(target_url="http://localhost:1234", container_name="c", node_ip="1.2.3.4", nodeport="30000", established_utc="t")


class TestGeneratorLifecycle:
    def test_start_and_verify_started(self):
        handles = []
        profile = to.resolve_profile_metadata("constant")
        handle = to.start_generator(profile, _route(), "run1", popen_factory=_fake_popen_factory(handles))
        assert to.verify_generator_started(handle, sleep_fn=_no_sleep) is True

    def test_verify_started_false_if_died_immediately(self):
        handles = []
        profile = to.resolve_profile_metadata("constant")
        handle = to.start_generator(profile, _route(), "run1", popen_factory=_fake_popen_factory(handles, exit_code=1))
        assert to.verify_generator_started(handle, sleep_fn=_no_sleep) is False

    def test_poll_generator_reflects_process_state(self):
        handles = []
        profile = to.resolve_profile_metadata("constant")
        handle = to.start_generator(profile, _route(), "run1", popen_factory=_fake_popen_factory(handles))
        assert to.poll_generator(handle) is None
        handle.process.set_exit(0)
        assert to.poll_generator(handle) == 0

    def test_generator_exit_expected_when_ran_full_duration_and_exit_zero(self):
        clock = _FakeClock()
        handles = []
        profile = to.resolve_profile_metadata("constant")  # 600s
        handle = to.start_generator(profile, _route(), "run1", popen_factory=_fake_popen_factory(handles), clock=clock)
        clock.advance(600.0)
        assert to.generator_exit_was_expected(handle, 0, clock=clock) is True

    def test_generator_exit_unexpected_when_too_early(self):
        clock = _FakeClock()
        handles = []
        profile = to.resolve_profile_metadata("constant")  # 600s
        handle = to.start_generator(profile, _route(), "run1", popen_factory=_fake_popen_factory(handles), clock=clock)
        clock.advance(10.0)  # far short of duration - tolerance
        assert to.generator_exit_was_expected(handle, 0, clock=clock) is False

    def test_generator_exit_expected_just_inside_the_narrow_tolerance(self):
        """Phase16Z: within GENERATOR_EARLY_EXIT_TOLERANCE_SECONDS (2.0s)
        of the profile's own declared duration must be accepted -- process-
        launch/clock jitter, not a real early stop."""
        clock = _FakeClock()
        handles = []
        profile = to.resolve_profile_metadata("constant")  # 600s
        handle = to.start_generator(profile, _route(), "run1", popen_factory=_fake_popen_factory(handles), clock=clock)
        clock.advance(600.0 - 1.5)  # 1.5s short -- inside the 2.0s tolerance
        assert to.generator_exit_was_expected(handle, 0, clock=clock) is True

    def test_generator_exit_unexpected_just_outside_the_narrow_tolerance(self):
        """Phase16Z: the OLD 90%-of-duration rule would have accepted this
        (540s >= 90% of 600s); the NEW narrow seconds-based tolerance must
        NOT -- exiting 5s early on a 600s profile is a real early stop that
        must be caught, not silently tolerated as 'close enough'."""
        clock = _FakeClock()
        handles = []
        profile = to.resolve_profile_metadata("constant")  # 600s
        handle = to.start_generator(profile, _route(), "run1", popen_factory=_fake_popen_factory(handles), clock=clock)
        clock.advance(600.0 - 5.0)  # 5s short -- outside the 2.0s tolerance
        assert to.generator_exit_was_expected(handle, 0, clock=clock) is False

    def test_generator_exit_unexpected_when_nonzero_even_if_full_duration(self):
        clock = _FakeClock()
        handles = []
        profile = to.resolve_profile_metadata("constant")
        handle = to.start_generator(profile, _route(), "run1", popen_factory=_fake_popen_factory(handles), clock=clock)
        clock.advance(600.0)
        assert to.generator_exit_was_expected(handle, 1, clock=clock) is False

    def test_stop_generator_terminates_running_process(self):
        handles = []
        profile = to.resolve_profile_metadata("constant")
        handle = to.start_generator(profile, _route(), "run1", popen_factory=_fake_popen_factory(handles))
        to.stop_generator(handle)
        assert handle.process.terminate_called is True

    def test_stop_generator_escalates_to_kill_only_on_timeout(self):
        class HangingThenKillablePopen(FakePopen):
            def wait(self, timeout=None):
                if not self.kill_called:
                    raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)
                return self._returncode

        handles = []

        def factory(cmd, **kwargs):
            p = HangingThenKillablePopen(args=cmd)
            handles.append(p)
            return p

        profile = to.resolve_profile_metadata("constant")
        handle = to.start_generator(profile, _route(), "run1", popen_factory=factory)
        to.stop_generator(handle, terminate_wait_seconds=0.01)
        assert handle.process.terminate_called is True
        assert handle.process.kill_called is True

    def test_stop_generator_safe_on_already_exited_process(self):
        handles = []
        profile = to.resolve_profile_metadata("constant")
        handle = to.start_generator(profile, _route(), "run1", popen_factory=_fake_popen_factory(handles, exit_code=0))
        exit_code = to.stop_generator(handle)
        assert exit_code == 0
        assert handle.process.terminate_called is False  # already exited, no need to terminate

    def test_no_orphan_subprocess_after_stop(self):
        handles = []
        profile = to.resolve_profile_metadata("constant")
        handle = to.start_generator(profile, _route(), "run1", popen_factory=_fake_popen_factory(handles))
        to.stop_generator(handle)
        assert handle.process.poll() is not None  # process is confirmed dead, not left running


# --- TrafficOrchestrator facade -----------------------------------------------------


class TestTrafficOrchestratorFacade:
    def _orchestrator(self, fk_docker=None, popen_handles=None, http_ok=True, exit_code=None):
        fk_docker = fk_docker or FakeDockerKubectl()
        popen_handles = popen_handles if popen_handles is not None else []
        return to.TrafficOrchestrator(
            transport=to.V5TransportManager(run=fk_docker.run),
            popen_factory=_fake_popen_factory(popen_handles, exit_code=exit_code),
            http_get=_fake_http_get_always_ok if http_ok else _fake_http_get_always_fail,
            sleep_fn=_no_sleep,
        )

    def test_start_episode_traffic_success(self, tmp_path):
        orch = self._orchestrator()
        orch.ledger_path = tmp_path / "ledger.jsonl"
        handle = orch.start_episode_traffic("constant", "episode_0_constant")
        assert handle.profile.name == "constant"

    def test_start_episode_traffic_route_failure_raises(self, tmp_path):
        orch = self._orchestrator(fk_docker=FakeDockerKubectl(network=""))
        orch.ledger_path = tmp_path / "ledger.jsonl"
        with pytest.raises(to.RouteEstablishmentFailure):
            orch.start_episode_traffic("constant", "episode_0_constant")

    def test_start_episode_traffic_route_unhealthy_raises(self, tmp_path):
        orch = self._orchestrator(http_ok=False)
        orch.ledger_path = tmp_path / "ledger.jsonl"
        with pytest.raises(to.RouteEstablishmentFailure):
            orch.start_episode_traffic("constant", "episode_0_constant")

    def test_start_episode_traffic_generator_prestart_failure_raises(self, tmp_path):
        orch = self._orchestrator(exit_code=1)  # dies immediately
        orch.ledger_path = tmp_path / "ledger.jsonl"
        with pytest.raises(to.TrafficGeneratorFailure):
            orch.start_episode_traffic("constant", "episode_0_constant")
        from autoscaler.src.rl_env.episode_ledger import read_rows
        rows = list(read_rows(orch.ledger_path))
        assert rows[0]["reason"] == "failed_to_start"

    def test_check_liveness_ok_while_generator_running(self, tmp_path):
        orch = self._orchestrator()
        orch.ledger_path = tmp_path / "ledger.jsonl"
        handle = orch.start_episode_traffic("constant", "ep0")
        orch.check_liveness_or_raise(handle, "ep0")  # must not raise

    def test_check_liveness_raises_on_unexpected_early_exit(self, tmp_path):
        orch = self._orchestrator()
        orch.ledger_path = tmp_path / "ledger.jsonl"
        handle = orch.start_episode_traffic("constant", "ep0")
        handle.process.set_exit(1)  # simulate a mid-episode crash
        with pytest.raises(to.TrafficGeneratorFailure):
            orch.check_liveness_or_raise(handle, "ep0")

    def test_stop_episode_traffic_records_provenance(self, tmp_path):
        orch = self._orchestrator()
        orch.ledger_path = tmp_path / "ledger.jsonl"
        handle = orch.start_episode_traffic("constant", "ep0")
        orch.stop_episode_traffic(handle, "ep0")
        from autoscaler.src.rl_env.episode_ledger import read_rows
        rows = list(read_rows(orch.ledger_path))
        assert any(r["reason"] == "episode_end_cleanup" for r in rows)

    def test_force_cleanup_tears_down_route_and_never_raises(self, tmp_path):
        orch = self._orchestrator()
        orch.ledger_path = tmp_path / "ledger.jsonl"
        orch.start_episode_traffic("constant", "ep0")
        orch.force_cleanup()  # must not raise even though a generator handle is still "active"

    def test_route_reused_across_episodes_when_still_alive(self, tmp_path):
        fk = FakeDockerKubectl()
        orch = self._orchestrator(fk_docker=fk)
        orch.ledger_path = tmp_path / "ledger.jsonl"
        h1 = orch.start_episode_traffic("constant", "ep0")
        orch.stop_episode_traffic(h1, "ep0")
        n_docker_run_calls_before = sum(1 for c in fk.calls if c[:2] == ["docker", "run"])
        orch.start_episode_traffic("constant", "ep1")
        n_docker_run_calls_after = sum(1 for c in fk.calls if c[:2] == ["docker", "run"])
        assert n_docker_run_calls_after == n_docker_run_calls_before  # route reused, not re-established


class TestResolveNextAvailableRunId:
    """Phase (Index29): retry-safe physical traffic run_id allocation.

    A scientific retry reuses the same logical `episode_id` but must land
    on a NEW, unused physical traffic output directory -- an in-memory
    counter is not safe here because a retry is normally a brand-new
    process (counter resets to 0). These tests use only `tmp_path`
    fixtures; no real `persistent-data/traffic/` directories are ever
    touched."""

    def test_no_existing_directory_returns_run1(self, tmp_path):
        run_id = to.resolve_next_available_run_id("episode_58_periodic", tmp_path)
        assert run_id == "episode_58_periodic_run1"

    def test_run1_exists_returns_run2(self, tmp_path):
        (tmp_path / "episode_58_periodic_run1").mkdir()
        run_id = to.resolve_next_available_run_id("episode_58_periodic", tmp_path)
        assert run_id == "episode_58_periodic_run2"

    def test_run1_and_run2_exist_returns_run3(self, tmp_path):
        (tmp_path / "episode_58_periodic_run1").mkdir()
        (tmp_path / "episode_58_periodic_run2").mkdir()
        run_id = to.resolve_next_available_run_id("episode_58_periodic", tmp_path)
        assert run_id == "episode_58_periodic_run3"

    def test_different_episode_ids_do_not_interfere(self, tmp_path):
        (tmp_path / "episode_58_periodic_run1").mkdir()
        run_id = to.resolve_next_available_run_id("episode_59_mixed", tmp_path)
        assert run_id == "episode_59_mixed_run1"

    def test_no_existing_artifacts_are_modified_or_deleted(self, tmp_path):
        existing = tmp_path / "episode_58_periodic_run1"
        existing.mkdir()
        marker = existing / "requests.csv"
        marker.write_text("original-evidence")
        to.resolve_next_available_run_id("episode_58_periodic", tmp_path)
        assert existing.exists()
        assert marker.read_text() == "original-evidence"

    def test_start_episode_traffic_uses_resolved_run_id_as_experiment_id(self, tmp_path):
        (tmp_path / "episode_58_periodic_run1").mkdir()
        orch = TestTrafficOrchestratorFacade()._orchestrator()
        orch.ledger_path = tmp_path / "ledger.jsonl"
        orch.traffic_base_dir = tmp_path
        handle = orch.start_episode_traffic("constant", "episode_58_periodic")
        assert handle.run_id == "episode_58_periodic_run2"
        assert "--experiment-id" in handle.command
        assert handle.command[handle.command.index("--experiment-id") + 1] == "episode_58_periodic_run2"

    def test_logical_episode_id_and_schedule_identity_unchanged(self, tmp_path):
        # The physical run_id changes across retries, but the logical
        # episode_id/profile passed in is untouched -- scientific identity
        # (global_schedule_episode_index, scientific_episode_id) is derived
        # upstream from episode_id, not from the physical run_id.
        (tmp_path / "episode_58_periodic_run1").mkdir()
        orch = TestTrafficOrchestratorFacade()._orchestrator()
        orch.ledger_path = tmp_path / "ledger.jsonl"
        orch.traffic_base_dir = tmp_path
        handle = orch.start_episode_traffic("periodic", "episode_58_periodic")
        assert handle.run_id.startswith("episode_58_periodic_run")
        assert handle.profile.name == "periodic"


class _FakeDrainClock:
    """Deterministic, instant fake clock/sleep pair: `sleep()` advances the
    clock by exactly the requested amount with no real wall-clock delay, so
    these tests run instantly and never flake under machine load."""

    def __init__(self):
        self.now = 0.0

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class TestDrainInFlightRequests:
    def test_immediate_drain_returns_without_sleeping(self):
        fake = _FakeDrainClock()
        calls = {"n": 0}

        def query_fn():
            calls["n"] += 1
            return 0.0, None

        result = to.drain_in_flight_requests(query_fn, sleep_fn=fake.sleep, clock=fake.clock)
        assert result == 0.0
        assert calls["n"] == 1
        assert fake.now == 0.0  # never slept

    def test_eventual_drain_succeeds(self):
        fake = _FakeDrainClock()
        values = iter([5.0, 2.0, 0.0])

        def query_fn():
            return next(values), None

        result = to.drain_in_flight_requests(
            query_fn, timeout_seconds=30.0, poll_interval_seconds=2.0, sleep_fn=fake.sleep, clock=fake.clock,
        )
        assert result == 0.0
        assert fake.now == 4.0  # two polls before quiescence, 2s apart

    def test_timeout_raises_drain_timeout_error(self):
        fake = _FakeDrainClock()

        def query_fn():
            return 5.0, None  # never drains

        with pytest.raises(to.DrainTimeoutError):
            to.drain_in_flight_requests(
                query_fn, timeout_seconds=10.0, poll_interval_seconds=2.0, sleep_fn=fake.sleep, clock=fake.clock,
            )

    def test_timeout_never_exceeds_the_configured_bound(self):
        fake = _FakeDrainClock()

        def query_fn():
            return 5.0, None

        with pytest.raises(to.DrainTimeoutError):
            to.drain_in_flight_requests(
                query_fn, timeout_seconds=10.0, poll_interval_seconds=2.0, sleep_fn=fake.sleep, clock=fake.clock,
            )
        assert fake.now <= 12.0  # bounded: at most one extra poll interval past the deadline

    def test_query_failure_raises_drain_query_error_not_silent_zero(self):
        fake = _FakeDrainClock()

        def query_fn():
            return None, "connection refused"

        with pytest.raises(to.DrainQueryError):
            to.drain_in_flight_requests(query_fn, sleep_fn=fake.sleep, clock=fake.clock)

    def test_absent_metric_without_error_is_treated_as_quiescent(self):
        fake = _FakeDrainClock()

        def query_fn():
            return None, None  # genuinely absent series, no error

        result = to.drain_in_flight_requests(query_fn, sleep_fn=fake.sleep, clock=fake.clock)
        assert result == 0.0

    def test_small_float_below_epsilon_counts_as_drained(self):
        fake = _FakeDrainClock()

        def query_fn():
            return 0.0001, None

        result = to.drain_in_flight_requests(query_fn, sleep_fn=fake.sleep, clock=fake.clock)
        assert result < to.DRAIN_ZERO_EPSILON


class TestTrafficOrchestratorDrainServerSideRequests:
    def test_uses_injected_drain_query_fn_without_real_prometheus_call(self):
        orch = to.TrafficOrchestrator(drain_query_fn=lambda: (0.0, None))
        result = orch.drain_server_side_requests()
        assert result == 0.0

    def test_default_query_builds_promql_with_existing_metric_name_only(self, monkeypatch):
        captured = {}

        class _FakeProm:
            def query(self, promql):
                captured["promql"] = promql
                return 0.0, None

        orch = to.TrafficOrchestrator(prom_client=_FakeProm())
        orch.drain_server_side_requests()
        assert "demo_app_requests_in_progress" in captured["promql"]
        assert captured["promql"].count("demo_app_requests_in_progress") == 1  # no invented metric

    def test_timeout_propagates_from_orchestrator_method(self):
        orch = to.TrafficOrchestrator(
            drain_query_fn=lambda: (5.0, None), drain_timeout_seconds=1.0, drain_poll_interval_seconds=1.0,
            sleep_fn=lambda s: None,
        )
        with pytest.raises(to.DrainTimeoutError):
            orch.drain_server_side_requests()


class TestTrafficSyncedWrapperDrainsAfterEpisodeEnd:
    def test_step_calls_drain_after_stop_episode_traffic_on_episode_end(self, tmp_path):
        fk = FakeDockerKubectl()
        orch = to.TrafficOrchestrator(
            transport=to.V5TransportManager(run=fk.run),
            popen_factory=_fake_popen_factory([], exit_code=None),
            http_get=_fake_http_get_always_ok,
            sleep_fn=_no_sleep,
            drain_query_fn=lambda: (0.0, None),
        )
        orch.ledger_path = tmp_path / "ledger.jsonl"

        drain_calls = {"n": 0}
        real_drain = orch.drain_server_side_requests

        def counting_drain():
            drain_calls["n"] += 1
            return real_drain()

        orch.drain_server_side_requests = counting_drain

        class _FakeInnerEnv(gym.Env):
            traffic_profile = None
            profile_hash = None
            global_schedule_episode_index = None

            def reset(self, **kwargs):
                return {}, {}

            def step(self, action):
                return {}, 0.0, True, False, {}  # always terminates

        inner = _FakeInnerEnv()
        wrapper = to.TrafficSyncedWrapper(inner, orch, lambda: 0, training_seed=42)
        wrapper.reset()
        wrapper.step(0)

        assert drain_calls["n"] == 1

    def test_step_does_not_call_drain_when_episode_not_ending(self, tmp_path):
        fk = FakeDockerKubectl()
        orch = to.TrafficOrchestrator(
            transport=to.V5TransportManager(run=fk.run),
            popen_factory=_fake_popen_factory([], exit_code=None),
            http_get=_fake_http_get_always_ok,
            sleep_fn=_no_sleep,
            drain_query_fn=lambda: (0.0, None),
        )
        orch.ledger_path = tmp_path / "ledger.jsonl"

        drain_calls = {"n": 0}
        orch.drain_server_side_requests = lambda: drain_calls.__setitem__("n", drain_calls["n"] + 1)

        class _FakeInnerEnv(gym.Env):
            traffic_profile = None
            profile_hash = None
            global_schedule_episode_index = None

            def reset(self, **kwargs):
                return {}, {}

            def step(self, action):
                return {}, 0.0, False, False, {}  # episode continues

        inner = _FakeInnerEnv()
        wrapper = to.TrafficSyncedWrapper(inner, orch, lambda: 0, training_seed=42)
        wrapper.reset()
        wrapper.step(0)

        assert drain_calls["n"] == 0
