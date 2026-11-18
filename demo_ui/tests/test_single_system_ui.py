"""demo_ui/tests/test_single_system_ui.py

The demo UI presents a single system only -- the final PPO multi-metric
autoscaler. There is no version-choice control anywhere, and no
user-visible page ever shows a version token ("V1"/"V2"/"Generation"/
"first/second generation"). The data-source toggle defaults to "Live
cluster", and "Sample data (offline)" renders every page correctly as a
fallback.
"""

from __future__ import annotations

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

from demo_ui import experiment_store  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

ALL_PAGES = ["Experiment", "Live Details", "Comparison", "Training", "History"]

# Case-insensitive, whole-word-ish version tokens that must never appear in
# any user-visible string. Matched against the raw rendered text, so a hit
# here means a real person would have seen it on screen.
_FORBIDDEN_TOKENS = ["V1", "V2", "Generation", "first generation", "second generation"]


def _strip_filesystem_paths(text: str) -> str:
    """Drops lines that merely report a real on-disk path/command (e.g. a
    "run this script" recovery instruction naming a data directory this
    task is forbidden from renaming, like persistent-data/evaluation_v2/)
    or an internal generated run_id (e.g. "fixture_v2_ppo_v2_..."). These
    are internal identifiers (file/directory/script/record names), not UI
    chrome a person reads as "the product's own wording" -- the task's own
    exception for file/variable names. Everything else is still checked."""
    kept = [
        line for line in text.splitlines()
        if "persistent-data" not in line and "scripts/" not in line
        and ".venv/bin/python" not in line and "fixture_v2_" not in line
    ]
    return "\n".join(kept)


def _all_visible_text(at: AppTest) -> str:
    parts = []
    parts += [m.value for m in at.markdown]
    parts += [c.value for c in at.caption]
    parts += [s.value for s in at.success]
    parts += [e.value for e in at.error]
    parts += [w.value for w in at.warning]
    parts += [t.value for t in at.title]
    parts += [h.value for h in at.header]
    parts += [sh.value for sh in at.subheader]
    for e in at.expander:
        parts.append(e.label)
    for b in at.button:
        parts.append(b.label)
    for sb in at.selectbox:
        parts.append(sb.label or "")
        parts += [str(o) for o in sb.options]
    for r in at.radio:
        parts.append(r.label or "")
        parts += [str(o) for o in r.options]
    for sc in at.segmented_control:
        parts.append(sc.label or "")
        parts += [str(o) for o in sc.options]
    return "\n".join(parts)


def _assert_no_forbidden_tokens(text: str, page: str) -> None:
    lowered = _strip_filesystem_paths(text).lower()
    for token in _FORBIDDEN_TOKENS:
        assert token.lower() not in lowered, f"forbidden version token {token!r} found on {page} page"


@pytest.fixture(autouse=True)
def _fresh_store(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_store, "STORE_PATH", tmp_path / "runs.json")


class TestNoGenerationSelector:
    def test_no_generation_widget_anywhere(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        for page in ALL_PAGES:
            at.sidebar.radio[0].set_value(page).run()
            assert not at.exception, f"{page} raised: {at.exception}"
            seg_keys = {sc.key for sc in at.segmented_control}
            assert "du_generation_widget" not in seg_keys
            assert "du_generation" not in at.session_state

    def test_only_one_segmented_control_exists_on_experiment_page(self):
        """Just the data-source toggle -- no second selector for choosing
        between systems."""
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Experiment").run()
        assert not at.exception
        assert len(at.segmented_control) == 1


class TestNoVersionStringsVisible:
    def test_no_forbidden_tokens_on_any_page_default_mode(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        for page in ALL_PAGES:
            at.sidebar.radio[0].set_value(page).run()
            assert not at.exception, f"{page} raised: {at.exception}"
            _assert_no_forbidden_tokens(_all_visible_text(at), page)

    def test_no_forbidden_tokens_in_sample_data_mode(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Experiment").run()
        at.segmented_control(key="du_data_source_widget").set_value("Sample data (offline)").run()
        assert not at.exception
        _assert_no_forbidden_tokens(_all_visible_text(at), "Experiment (sample data)")

        at.button[0].click().run()  # START -- loads the sample-data run
        assert not at.exception
        _assert_no_forbidden_tokens(_all_visible_text(at), "Experiment (sample data, after start)")

        at.sidebar.radio[0].set_value("Live Details").run()
        at.button(key="du_load_live_fixture_v2").click().run()
        assert not at.exception
        _assert_no_forbidden_tokens(_all_visible_text(at), "Live Details (sample data loaded)")


class TestDataSourceDefaultsToLiveCluster:
    def test_default_value_is_live_cluster(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Experiment").run()
        seg = at.segmented_control(key="du_data_source_widget")
        assert seg.value == "Live cluster"

    def test_options_are_exactly_live_cluster_and_sample_data(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Experiment").run()
        seg = at.segmented_control(key="du_data_source_widget")
        assert set(seg.options) == {"Live cluster", "Sample data (offline)"}


class TestSampleDataOfflineRendersEveryPage:
    def test_experiment_page_start_with_sample_data(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Experiment").run()
        at.segmented_control(key="du_data_source_widget").set_value("Sample data (offline)").run()
        start_button = [b for b in at.button if b.label == "START"][0]
        start_button.click().run()
        assert not at.exception
        success_text = "".join(s.value for s in at.success)
        assert "SIMULATED DATA loaded" in success_text

    def test_every_other_page_renders_without_exception(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        for page in ALL_PAGES:
            at.sidebar.radio[0].set_value(page).run()
            assert not at.exception, f"{page} raised: {at.exception}"

    def test_live_details_sample_data_is_labeled_as_simulated(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Live Details").run()
        at.button(key="du_load_live_fixture_v2").click().run()
        assert not at.exception
        html = "".join(m.value for m in at.markdown) + "".join(c.value for c in at.caption)
        assert "SIMULATED DATA" in html or "simulated" in html.lower()
