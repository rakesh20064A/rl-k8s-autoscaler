# app/tests/test_workload_v2.py
# V2 multi-resource /work workload tests. Uses only small memory_mb values
# (<=4MB) to keep the suite fast -- these tests prove the touch/release/
# bound mechanics, not realistic production sizing.
#
# Design constraints (same as test_app.py):
#   - no live Kubernetes/Prometheus/Docker required
#   - FastAPI TestClient only, in-process

import gc

import pytest
from fastapi.testclient import TestClient

try:
    from app.src.main import app
    from app.src.workload import (
        MAX_MEMORY_HOLD_MS,
        MAX_WORK_MEMORY_MB,
        VALID_RESOURCE_MODES,
        perform_cpu_work,
        perform_memory_work,
        perform_work,
    )
except ImportError:
    import sys
    sys.path.insert(0, "app/src")
    from main import app  # type: ignore[import]
    from workload import (  # type: ignore[import]
        MAX_MEMORY_HOLD_MS,
        MAX_WORK_MEMORY_MB,
        VALID_RESOURCE_MODES,
        perform_cpu_work,
        perform_memory_work,
        perform_work,
    )

# V2 Phase 1.1: DEFAULT_MAX_IN_FLIGHT is the load generator's own declared
# worst-case concurrency ceiling (scripts/traffic_orchestrator.py); the pod's
# declared memory limit is k8s/app/deployment.yaml's memory.limits (256Mi).
# These two constants are duplicated here (not imported) deliberately --
# this is a cross-cutting safety-math assertion, not logic shared at runtime
# between the app and the load generator.
_POD_MEMORY_LIMIT_MIB = 256
_LOADGEN_MAX_IN_FLIGHT = 50


@pytest.fixture(scope="module")
def client():
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Backward compatibility: resource_mode="cpu" must match V1 exactly
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_old_intensity_only_request_still_200(self, client):
        assert client.get("/work?intensity=2").status_code == 200

    def test_default_resource_mode_is_cpu(self, client):
        body = client.get("/work?intensity=2").json()
        assert body["resource_mode"] == "cpu"

    def test_cpu_mode_response_still_has_all_v1_fields(self, client):
        body = client.get("/work?intensity=2").json()
        for field in ("intensity", "elapsed_ms", "iterations_completed", "result"):
            assert field in body

    def test_cpu_mode_performs_no_memory_work(self, client):
        body = client.get("/work?intensity=2").json()
        assert body["memory_mb_requested"] == 0
        assert body["memory_mb_touched"] == 0

    def test_perform_work_cpu_mode_matches_perform_cpu_work(self):
        """resource_mode="cpu" must produce the identical CPU-side fields
        `perform_cpu_work` alone would -- V2 only ever adds fields.
        `elapsed_ms` is excluded: two separate real CPU loop executions
        naturally differ by microseconds (same non-timing-assertion
        constraint test_app.py's own module docstring documents)."""
        direct = perform_cpu_work(2)
        via_dispatch = perform_work(2, resource_mode="cpu", memory_mb=0)
        for key in direct:
            if key == "elapsed_ms":
                continue
            assert via_dispatch[key] == direct[key]


# ---------------------------------------------------------------------------
# Memory / mixed modes
# ---------------------------------------------------------------------------

class TestMemoryWork:
    def test_memory_mode_performs_bounded_memory_work(self):
        result = perform_memory_work(2)
        assert result["memory_mb_requested"] == 2
        assert result["memory_mb_touched"] == 2
        assert result["memory_elapsed_ms"] >= 0

    def test_memory_mode_zero_mb_is_a_no_op(self):
        result = perform_memory_work(0)
        assert result["memory_mb_touched"] == 0

    def test_maximum_bound_enforced_by_clamping(self):
        """Direct calls to perform_memory_work (bypassing the route's own
        `le=` validation) must still never exceed MAX_WORK_MEMORY_MB."""
        result = perform_memory_work(MAX_WORK_MEMORY_MB + 1000)
        assert result["memory_mb_requested"] == MAX_WORK_MEMORY_MB

    def test_route_rejects_memory_above_max(self, client):
        r = client.get(f"/work?intensity=1&resource_mode=memory&memory_mb={MAX_WORK_MEMORY_MB + 1}")
        assert r.status_code == 422

    def test_memory_mode_skips_cpu_work(self, client):
        body = client.get("/work?intensity=9&resource_mode=memory&memory_mb=1").json()
        assert body["iterations_completed"] == 0

    def test_mixed_mode_performs_both(self, client):
        body = client.get("/work?intensity=2&resource_mode=mixed&memory_mb=1").json()
        assert body["iterations_completed"] > 0
        assert body["memory_mb_touched"] == 1

    def test_invalid_resource_mode_rejected(self, client):
        assert client.get("/work?resource_mode=disk&memory_mb=1").status_code == 422

    def test_valid_resource_modes_are_exactly_cpu_memory_mixed(self):
        assert VALID_RESOURCE_MODES == ("cpu", "memory", "mixed")

    def test_perform_work_rejects_invalid_mode_directly(self):
        with pytest.raises(ValueError):
            perform_work(3, resource_mode="disk")


class TestAggregateMemorySafety:
    """V2 Phase 1.1: proves the DISCLOSED worst-case trade-off documented in
    workload.py's own comments, so a future change to either constant
    cannot silently re-introduce the Phase-1 OOM risk without failing a
    test. This does NOT assert full concurrency-level safety (the
    documented trade-off explicitly accepts that a true 50-way concurrent
    burst can still exceed the pod limit) -- it asserts the numbers match
    what was disclosed and that the realistic low-concurrency case is safe."""

    def test_max_memory_bound_is_the_recalculated_safe_value(self):
        assert MAX_WORK_MEMORY_MB == 16

    def test_worst_case_full_concurrency_is_disclosed_not_silently_safe(self):
        worst_case_mib = MAX_WORK_MEMORY_MB * _LOADGEN_MAX_IN_FLIGHT
        assert worst_case_mib == 800
        assert worst_case_mib > _POD_MEMORY_LIMIT_MIB  # disclosed trade-off, not a false safety guarantee

    def test_strict_worst_case_safe_bound_would_be_under_pod_limit(self):
        baseline_reserved_mib = 60
        strict_safe_per_request_mib = (_POD_MEMORY_LIMIT_MIB - baseline_reserved_mib) / _LOADGEN_MAX_IN_FLIGHT
        assert strict_safe_per_request_mib < MAX_WORK_MEMORY_MB  # confirms 16MB is ABOVE the strict bound (the trade-off)
        assert round(strict_safe_per_request_mib, 1) == 3.9

    def test_low_concurrency_realistic_case_stays_within_pod_limit(self):
        """A handful of genuinely concurrent memory-mode requests (far below
        the load generator's full 50-way ceiling) must stay safely within
        the pod's memory limit at the new 16MB bound."""
        realistic_concurrency = 8
        assert MAX_WORK_MEMORY_MB * realistic_concurrency < _POD_MEMORY_LIMIT_MIB


class TestMemoryHoldRelease:
    def test_default_hold_respected_in_elapsed_time(self):
        result = perform_memory_work(1, hold_ms=50)
        assert result["memory_hold_ms"] == 50
        assert result["memory_elapsed_ms"] >= 50

    def test_zero_hold_does_not_sleep(self):
        result = perform_memory_work(1, hold_ms=0)
        assert result["memory_hold_ms"] == 0
        assert result["memory_elapsed_ms"] < 50

    def test_hold_ms_clamped_to_max(self):
        result = perform_memory_work(1, hold_ms=MAX_MEMORY_HOLD_MS + 10_000)
        assert result["memory_hold_ms"] == MAX_MEMORY_HOLD_MS

    def test_negative_hold_clamped_to_zero(self):
        result = perform_memory_work(1, hold_ms=-5)
        assert result["memory_hold_ms"] == 0

    def test_zero_memory_mb_ignores_hold(self):
        result = perform_memory_work(0, hold_ms=5000)
        assert result["memory_hold_ms"] == 0
        assert result["memory_elapsed_ms"] < 50

    def test_route_accepts_memory_hold_ms(self, client):
        body = client.get(
            "/work?intensity=1&resource_mode=memory&memory_mb=1&memory_hold_ms=20"
        ).json()
        assert body["memory_hold_ms"] == 20

    def test_route_rejects_memory_hold_ms_above_max(self, client):
        r = client.get(
            f"/work?intensity=1&resource_mode=memory&memory_mb=1&memory_hold_ms={MAX_MEMORY_HOLD_MS + 1}"
        )
        assert r.status_code == 422

    def test_route_rejects_negative_memory_hold_ms(self, client):
        r = client.get("/work?intensity=1&resource_mode=memory&memory_mb=1&memory_hold_ms=-1")
        assert r.status_code == 422

    def test_cpu_mode_response_still_reports_zero_hold(self, client):
        body = client.get("/work?intensity=2").json()
        assert body["memory_hold_ms"] == 0


class TestMemoryNotRetained:
    def test_no_module_level_buffer_created(self):
        """The allocated mapping must never become reachable from module
        globals -- calling perform_memory_work repeatedly must not grow
        any persistent reference gc can find rooted at the module."""
        import app.src.workload as workload_mod
        before = {k: v for k, v in vars(workload_mod).items() if isinstance(v, (bytearray, bytes))}
        for _ in range(3):
            perform_memory_work(1)
        after = {k: v for k, v in vars(workload_mod).items() if isinstance(v, (bytearray, bytes))}
        assert before == after == {}

    def test_memory_released_after_call_no_uncollectable_growth(self):
        gc.collect()
        baseline = len(gc.get_objects())
        for _ in range(5):
            perform_memory_work(2)
        gc.collect()
        after = len(gc.get_objects())
        # A handful of transient objects is normal; a persistent leak would
        # show up as growth proportional to the number of calls (hundreds+).
        assert after - baseline < 200


class TestMmapExplicitReleaseSemantics:
    """V2 Phase 2.2 PART 1: proves the OS-level release mechanism itself
    (mmap.close() / munmap), not merely that the Python object is
    collectible -- the distinction that mattered here, since a plain
    bytearray's Python object WAS already being correctly collected
    (TestMemoryNotRetained above) even while real pod RSS stayed elevated
    (glibc arena retention, confirmed live in V2 Phase 2.1 PART 4)."""

    def test_mapping_is_closed_after_a_normal_call(self, monkeypatch):
        import mmap as mmap_module

        import app.src.workload as workload_mod

        created = []
        real_mmap = mmap_module.mmap

        def _spy_mmap(*args, **kwargs):
            m = real_mmap(*args, **kwargs)
            created.append(m)
            return m

        monkeypatch.setattr(workload_mod.mmap, "mmap", _spy_mmap)
        perform_memory_work(1, hold_ms=0)

        assert len(created) == 1
        assert created[0].closed is True

    def test_mapping_is_closed_even_when_touching_pages_raises(self, monkeypatch):
        """Release-on-exception: if writing to the mapping raises partway
        through, the mapping must STILL be closed (the `finally` block),
        and the original exception must still propagate -- never silently
        swallowed."""
        import app.src.workload as workload_mod

        class _ExplodingMapping:
            def __init__(self, real_mapping):
                self._real = real_mapping
                self.closed = False
                self._writes = 0

            def __setitem__(self, key, value):
                self._writes += 1
                if self._writes == 3:
                    raise RuntimeError("simulated page-touch failure")
                self._real[key] = value

            def close(self):
                self.closed = True
                self._real.close()

        import mmap as mmap_module
        real_mmap = mmap_module.mmap
        spies = []

        def _fake_mmap(*args, **kwargs):
            wrapper = _ExplodingMapping(real_mmap(*args, **kwargs))
            spies.append(wrapper)
            return wrapper

        monkeypatch.setattr(workload_mod.mmap, "mmap", _fake_mmap)

        with pytest.raises(RuntimeError, match="simulated page-touch failure"):
            perform_memory_work(1, hold_ms=0)

        assert len(spies) == 1
        assert spies[0].closed is True  # released despite the exception

    def test_mapping_is_closed_even_when_sleep_raises(self, monkeypatch):
        """Same guarantee, but the failure happens during the hold (sleep)
        phase rather than the touch phase."""
        import app.src.workload as workload_mod

        def _raising_sleep(_seconds):
            raise KeyboardInterrupt("simulated interrupt during hold")

        import mmap as mmap_module
        real_mmap = mmap_module.mmap
        created = []

        def _spy_mmap(*args, **kwargs):
            m = real_mmap(*args, **kwargs)
            created.append(m)
            return m

        monkeypatch.setattr(workload_mod.mmap, "mmap", _spy_mmap)
        monkeypatch.setattr(workload_mod.time, "sleep", _raising_sleep)

        with pytest.raises(KeyboardInterrupt):
            perform_memory_work(1, hold_ms=50)

        assert len(created) == 1
        assert created[0].closed is True

    def test_repeated_calls_never_accumulate_open_mappings(self, monkeypatch):
        """Direct proof that repeated/concurrent-style sequential memory
        workload calls never retain an application-owned open mapping --
        every single one is closed before its call returns."""
        import mmap as mmap_module
        import app.src.workload as workload_mod

        created = []
        real_mmap = mmap_module.mmap

        def _spy_mmap(*args, **kwargs):
            m = real_mmap(*args, **kwargs)
            created.append(m)
            return m

        monkeypatch.setattr(workload_mod.mmap, "mmap", _spy_mmap)
        for _ in range(10):
            perform_memory_work(2, hold_ms=0)

        assert len(created) == 10
        assert all(m.closed for m in created)
