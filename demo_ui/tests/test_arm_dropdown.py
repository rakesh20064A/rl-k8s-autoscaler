"""demo_ui/tests/test_arm_dropdown.py

Coverage for replacing the Arm radio buttons with an `st.selectbox` on
the Experiment page's control row (same row/styling as the traffic
pattern / resource type / load level selects):

  - options are exactly "PPO" / "HPA", default "PPO";
  - "PPO" maps to the internal arm id "ppo_v2", "HPA" maps to
    "hpa_v2_cpu_memory" (unchanged internal ids -- UI/label-only change);
  - the short labels "PPO"/"HPA" (never a longer display name, never the
    old "(CPU+memory)" suffix) are what Live Details / History /
    Comparison show for an arm;
  - the dropdown is disabled exactly when the other controls are (an
    active run locks it), in both Live cluster and Sample data (offline)
    modes, since both share `render_controls_v2()`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

from demo_ui import experiment_store  # noqa: E402
from demo_ui.components.controls_v2 import ARM_HELP, ARM_LABELS  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def _fresh_store(tmp_path, monkeypatch):
    store_path = tmp_path / "runs.json"
    monkeypatch.setattr(experiment_store, "STORE_PATH", store_path)
    return store_path


class TestArmLabelsAndMapping:
    def test_labels_are_exactly_ppo_and_hpa(self):
        assert ARM_LABELS == {"ppo_v2": "PPO", "hpa_v2_cpu_memory": "HPA"}

    def test_no_cpu_memory_suffix_anywhere_in_labels(self):
        for label in ARM_LABELS.values():
            assert "CPU" not in label and "memory" not in label.lower()

    def test_help_tooltip_text(self):
        assert ARM_HELP == "HPA uses CPU (260%) and memory (120%) targets, 2-8 replicas"


class TestExperimentPageArmDropdown:
    def test_arm_widget_is_a_selectbox_not_a_radio(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Experiment").run()
        assert not at.exception
        arm_selectbox_keys = {sb.key for sb in at.selectbox}
        assert "du_v2_arm_widget" in arm_selectbox_keys
        radio_keys = {r.key for r in at.radio}
        assert "du_v2_arm_widget" not in radio_keys

    def test_default_is_ppo(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Experiment").run()
        arm_select = at.selectbox(key="du_v2_arm_widget")
        assert arm_select.value == "PPO"

    def test_options_are_exactly_ppo_and_hpa(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Experiment").run()
        arm_select = at.selectbox(key="du_v2_arm_widget")
        assert list(arm_select.options) == ["PPO", "HPA"]

    def test_help_tooltip_present_on_widget(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Experiment").run()
        arm_select = at.selectbox(key="du_v2_arm_widget")
        assert arm_select.help == ARM_HELP

    def test_selecting_ppo_maps_to_ppo_v2_internal_id_sample_data_mode(self):
        """Drives the dropdown and START in Sample data (offline) mode --
        the same `render_controls_v2()` control path Live cluster mode
        uses -- and checks the produced fixture record carries the exact
        internal arm id "ppo_v2", never a display label."""
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Experiment").run()
        at.segmented_control(key="du_data_source_widget").set_value("Sample data (offline)").run()
        at.selectbox(key="du_v2_arm_widget").set_value("PPO").run()
        at.button(key="du_v2_start_button").click().run()
        assert not at.exception
        rec = at.session_state.get("du_v2_demo_record")
        assert rec is not None
        assert rec["controller"] == "ppo_v2"

    def test_selecting_hpa_maps_to_hpa_v2_cpu_memory_internal_id_sample_data_mode(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Experiment").run()
        at.segmented_control(key="du_data_source_widget").set_value("Sample data (offline)").run()
        at.selectbox(key="du_v2_arm_widget").set_value("HPA").run()
        at.button(key="du_v2_start_button").click().run()
        assert not at.exception
        rec = at.session_state.get("du_v2_demo_record")
        assert rec is not None
        assert rec["controller"] == "hpa_v2_cpu_memory"

    def test_arm_dropdown_disabled_while_run_active_live_cluster_mode(self):
        """Live cluster mode (the default data source): an active v2 run
        must disable the arm dropdown via the same `controls_disabled`
        lock already used for the other 3 selects."""
        run_id = experiment_store.start_run("hpa_v2_cpu_memory", "demo_cpu_short")
        experiment_store.update_run(
            run_id, generation="v2", stage="EPISODE_RUNNING", workload_hash="h1",
            workload_seed=42, workload_duration_s=40.0, pid=os.getpid(),
        )
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Experiment").run()
        assert not at.exception
        arm_select = at.selectbox(key="du_v2_arm_widget")
        pattern_select = at.selectbox(key="du_v2_pattern_widget")
        assert arm_select.disabled is True
        assert arm_select.disabled == pattern_select.disabled

    def test_arm_dropdown_enabled_when_no_active_run(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Experiment").run()
        arm_select = at.selectbox(key="du_v2_arm_widget")
        assert arm_select.disabled is False

    def test_arm_dropdown_in_same_row_as_traffic_selects_sample_data_mode(self):
        """Sample data mode uses the exact same `render_controls_v2()`
        control strip -- there is no separate Sample-mode control path to
        drift out of sync."""
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Experiment").run()
        at.segmented_control(key="du_data_source_widget").set_value("Sample data (offline)").run()
        assert not at.exception
        selectbox_keys = {sb.key for sb in at.selectbox}
        assert {
            "du_v2_arm_widget", "du_v2_pattern_widget", "du_v2_resource_widget", "du_v2_load_widget",
        }.issubset(selectbox_keys)


class TestShortArmLabelsOnOtherPages:
    def test_history_page_shows_short_labels_end_to_end(self):
        run_id_hpa = experiment_store.start_run("hpa_v2_cpu_memory", "demo_cpu_short")
        experiment_store.update_run(
            run_id_hpa, generation="v2", status="completed", profile_name="demo_cpu_short",
        )
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("History").run()
        assert not at.exception
        html = "".join(m.value for m in at.markdown)
        assert "<td>HPA</td>" in html
        assert "CPU+memory" not in html
        assert "HPA (CPU+memory)" not in html

    def test_live_details_header_shows_short_label(self):
        run_id = experiment_store.start_run("ppo_v2", "demo_cpu_short")
        experiment_store.update_run(
            run_id, generation="v2", status="running", stage="EPISODE_RUNNING",
            workload_hash="h1", workload_seed=42, workload_duration_s=40.0, pid=os.getpid(),
        )
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Live Details").run()
        assert not at.exception
        html = "".join(m.value for m in at.markdown)
        assert "PPO" in html
        assert "CPU+memory" not in html

    def test_comparison_page_never_shows_cpu_memory_suffix(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Comparison").run()
        assert not at.exception
        html = "".join(m.value for m in at.markdown) + "".join(c.value for c in at.caption)
        assert "CPU+memory" not in html
        assert "HPA (CPU+memory)" not in html
