from __future__ import annotations

from demo_ui.live_adapter import LiveExperimentController
from demo_ui.models import Controller, TrafficProfile


class TestLiveExperimentController:
    def test_defaults_to_dry_run(self):
        ctrl = LiveExperimentController()
        assert ctrl.dry_run is True
        assert ctrl.i_understand_this_will_mutate_the_live_cluster is False

    def test_start_never_raises_and_only_plans(self):
        ctrl = LiveExperimentController()
        ctrl.start(Controller.PPO, TrafficProfile.RAMP)
        assert ctrl.is_running is True
        assert len(ctrl.planned_operations) > 0
        assert all("would" in op.detail for op in ctrl.planned_operations)

    def test_start_is_idempotent_against_streamlit_reruns(self):
        ctrl = LiveExperimentController()
        ctrl.start(Controller.HPA, TrafficProfile.CONSTANT)
        first_plan = list(ctrl.planned_operations)
        ctrl.start(Controller.PPO, TrafficProfile.BURST)  # simulates a rerun re-invoking start()
        assert ctrl.planned_operations == first_plan  # second call was a no-op

    def test_execution_gate_is_hardcoded_closed_even_with_explicit_opt_in(self):
        ctrl = LiveExperimentController(dry_run=False, i_understand_this_will_mutate_the_live_cluster=True)
        # _EXECUTION_ENABLED is hard-coded False in this phase, so even an
        # explicit opt-in must not raise LiveMutationNotImplementedError
        # (that branch is unreachable) and must not mutate anything.
        ctrl.start(Controller.PPO, TrafficProfile.STEP)
        assert ctrl.is_running is True

    def test_stop_clears_running_flag(self):
        ctrl = LiveExperimentController()
        ctrl.start(Controller.HPA, TrafficProfile.MIXED)
        ctrl.stop()
        assert ctrl.is_running is False

    def test_reset_clears_plan(self):
        ctrl = LiveExperimentController()
        ctrl.start(Controller.HPA, TrafficProfile.MIXED)
        ctrl.reset()
        assert ctrl.planned_operations == []
        assert ctrl.is_running is False

    def test_describe_plan_is_human_readable(self):
        ctrl = LiveExperimentController()
        ctrl.start(Controller.PPO, TrafficProfile.PERIODIC)
        lines = ctrl.describe_plan()
        assert all(isinstance(line, str) for line in lines)
        assert len(lines) == len(ctrl.planned_operations)

    def test_no_real_mutation_path_exists_in_source(self):
        import inspect

        import demo_ui.live_adapter as mod
        source = inspect.getsource(mod)
        for forbidden in ["subprocess.run(", "subprocess.Popen(", "import subprocess", "import docker"]:
            assert forbidden not in source
