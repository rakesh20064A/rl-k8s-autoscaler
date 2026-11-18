"""demo_ui/tests/test_demo_traffic_catalog_ui.py

Coverage for the Experiment page's 3-select demo traffic picker (traffic
pattern / resource type / load level), whose combination picks one of the
36 generated `configs/traffic_profiles/v2_demo/demo_<pattern>_<resource>_
<load>.yaml` profiles (scripts/generate_demo_profiles.py):

  - the 3 selects map to the correct profile filename for every one of
    the 36 combinations;
  - the Experiment page (Sample data / offline mode) renders without
    exception for every one of the 36 combinations;
  - no "V1"/"V2"/"Generation" version wording appears in any new
    user-visible Streamlit call this task added (a static grep-style
    check, separate from test_single_system_ui.py's rendered-text check).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

from demo_ui import experiment_store  # noqa: E402
from scripts.generate_demo_profiles import (  # noqa: E402
    ALL_COMBOS,
    LOAD_LABELS,
    PATTERN_LABELS,
    RESOURCE_LABELS,
    profile_slug,
)

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(autouse=True)
def _fresh_store(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_store, "STORE_PATH", tmp_path / "runs.json")


class TestThreeSelectsMapToCorrectProfileFilename:
    @pytest.mark.parametrize("combo", ALL_COMBOS, ids=lambda c: "_".join(c))
    def test_catalog_mapping_function_matches_expected_filename(self, combo):
        pattern, resource, load = combo
        assert profile_slug(pattern, resource, load) == f"demo_{pattern}_{resource}_{load}"

    @pytest.mark.parametrize("combo", ALL_COMBOS, ids=lambda c: "_".join(c))
    def test_ui_selects_resolve_to_expected_profile_name(self, combo):
        """Drives the actual 3 Experiment-page selects to this combination
        and checks the record produced by START carries the exact expected
        profile_name -- not just that the pure mapping function agrees
        with itself."""
        pattern, resource, load = combo
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Experiment").run()
        at.segmented_control(key="du_data_source_widget").set_value("Sample data (offline)").run()
        at.selectbox(key="du_v2_pattern_widget").set_value(PATTERN_LABELS[pattern]).run()
        at.selectbox(key="du_v2_resource_widget").set_value(RESOURCE_LABELS[resource]).run()
        at.selectbox(key="du_v2_load_widget").set_value(LOAD_LABELS[load]).run()
        assert not at.exception
        at.button(key="du_v2_start_button").click().run()
        assert not at.exception

        rec = at.session_state.get("du_v2_demo_record")
        assert rec is not None
        assert rec["profile_name"] == profile_slug(pattern, resource, load)


class TestSampleDataRendersForEveryCombination:
    @pytest.mark.parametrize("combo", ALL_COMBOS, ids=lambda c: "_".join(c))
    def test_renders_without_exception(self, combo):
        pattern, resource, load = combo
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Experiment").run()
        at.segmented_control(key="du_data_source_widget").set_value("Sample data (offline)").run()
        at.selectbox(key="du_v2_pattern_widget").set_value(PATTERN_LABELS[pattern]).run()
        at.selectbox(key="du_v2_resource_widget").set_value(RESOURCE_LABELS[resource]).run()
        at.selectbox(key="du_v2_load_widget").set_value(LOAD_LABELS[load]).run()
        assert not at.exception, f"{combo}: selecting raised {at.exception}"

        at.button(key="du_v2_start_button").click().run()
        assert not at.exception, f"{combo}: START raised {at.exception}"

        # The preview chart + success banner must both be present -- not
        # just "didn't crash".
        assert any("Traffic Preview" in m.value for m in at.markdown)
        assert any(s.value.startswith("SIMULATED DATA loaded") for s in at.success)


# Files this task added or materially changed to build the 3-select demo
# traffic picker -- the ones a new "V1"/"V2"/"Generation" user-visible
# string would most plausibly have been introduced into.
_NEW_OR_CHANGED_FILES = (
    "demo_ui/components/controls_v2.py",
    "demo_ui/views/experiment.py",
    "demo_ui/views/live_details_v2.py",
    "demo_ui/views/live_details.py",
    "demo_ui/profile_preview.py",
    "demo_ui/fixtures_v2.py",
    "scripts/generate_demo_profiles.py",
)

# Streamlit calls that put a string on screen (label=, title text, body
# text, button/selectbox labels, markdown/caption/success/error text).
_UI_CALL_PATTERN = re.compile(
    r"\bst\.(markdown|caption|success|error|warning|info|title|header|subheader|"
    r"selectbox|radio|button|text|write)\("
)
_FORBIDDEN_TOKEN_PATTERN = re.compile(r"\bV1\b|\bV2\b|\bGeneration\b")


class TestNoVersionWordingInNewUiStrings:
    def test_grep_new_ui_files_for_forbidden_version_tokens(self):
        violations = []
        for rel_path in _NEW_OR_CHANGED_FILES:
            path = REPO_ROOT / rel_path
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if _UI_CALL_PATTERN.search(line) and _FORBIDDEN_TOKEN_PATTERN.search(line):
                    violations.append(f"{rel_path}:{lineno}: {line.strip()}")
        assert not violations, "forbidden version wording found in UI calls:\n" + "\n".join(violations)
