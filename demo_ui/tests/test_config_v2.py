"""demo_ui/tests/test_config_v2.py

V2 Phase 5E: demo_ui/config_v2.py has no Streamlit dependency, so these
tests run under the scientific .venv directly, not just the UI venv.
Covers both the raising `load_config_v2()` and the crash-proof
`load_config_v2_safe()` every UI view must use.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from demo_ui.config_v2 import ConfigV2, load_config_v2, load_config_v2_safe


def _write_config(tmp_path: Path, model_path: str, epsilon_path: str, campaign_dir: str) -> Path:
    config_path = tmp_path / "config_v2.yaml"
    config_path.write_text(
        f'model_path: "{model_path}"\n'
        f'epsilon_calibration_path: "{epsilon_path}"\n'
        f'evaluation_campaign_dir: "{campaign_dir}"\n',
        encoding="utf-8",
    )
    return config_path


class TestLoadConfigV2(object):
    def test_loads_relative_paths_against_project_root(self, tmp_path):
        config_path = _write_config(tmp_path, "models/x.zip", "eps.json", "campaign")
        cfg = load_config_v2(config_path)
        assert cfg.model_path.is_absolute()
        assert str(cfg.model_path).endswith("models/x.zip")

    def test_absolute_paths_pass_through_unchanged(self, tmp_path):
        model_abs = tmp_path / "model.zip"
        config_path = _write_config(tmp_path, str(model_abs), "eps.json", "campaign")
        cfg = load_config_v2(config_path)
        assert cfg.model_path == model_abs

    def test_missing_required_field_raises(self, tmp_path):
        config_path = tmp_path / "config_v2.yaml"
        config_path.write_text('model_path: "x.zip"\n', encoding="utf-8")
        with pytest.raises(ValueError, match="epsilon_calibration_path"):
            load_config_v2(config_path)

    def test_missing_file_raises_oserror(self, tmp_path):
        with pytest.raises(OSError):
            load_config_v2(tmp_path / "does_not_exist.yaml")

    def test_live_enabled_requires_both_files(self, tmp_path):
        model_path = tmp_path / "model.zip"
        model_path.write_bytes(b"x")
        config_path = _write_config(tmp_path, str(model_path), str(tmp_path / "missing_eps.json"), "campaign")
        cfg = load_config_v2(config_path)
        assert cfg.model_exists is True
        assert cfg.epsilon_calibration_exists is False
        assert cfg.live_enabled is False
        assert "epsilon calibration" in cfg.live_disabled_reason

    def test_live_enabled_true_when_both_files_present(self, tmp_path):
        model_path = tmp_path / "model.zip"
        model_path.write_bytes(b"x")
        eps_path = tmp_path / "eps.json"
        eps_path.write_text("{}")
        config_path = _write_config(tmp_path, str(model_path), str(eps_path), "campaign")
        cfg = load_config_v2(config_path)
        assert cfg.live_enabled is True
        assert cfg.live_disabled_reason is None


class TestLoadConfigV2Safe:
    def test_returns_config_and_no_error_on_success(self, tmp_path):
        model_path = tmp_path / "model.zip"
        model_path.write_bytes(b"x")
        eps_path = tmp_path / "eps.json"
        eps_path.write_text("{}")
        config_path = _write_config(tmp_path, str(model_path), str(eps_path), "campaign")
        cfg, error = load_config_v2_safe(config_path)
        assert isinstance(cfg, ConfigV2)
        assert error is None

    def test_returns_none_and_error_message_when_file_missing(self, tmp_path):
        cfg, error = load_config_v2_safe(tmp_path / "does_not_exist.yaml")
        assert cfg is None
        assert error is not None
        assert "does_not_exist.yaml" in error

    def test_returns_none_and_error_message_when_field_missing(self, tmp_path):
        config_path = tmp_path / "config_v2.yaml"
        config_path.write_text('model_path: "x.zip"\n', encoding="utf-8")
        cfg, error = load_config_v2_safe(config_path)
        assert cfg is None
        assert "epsilon_calibration_path" in error

    def test_returns_none_and_error_message_on_malformed_yaml(self, tmp_path):
        config_path = tmp_path / "config_v2.yaml"
        config_path.write_text("model_path: [unterminated\n", encoding="utf-8")
        cfg, error = load_config_v2_safe(config_path)
        assert cfg is None
        assert error is not None

    def test_never_raises_regardless_of_failure_mode(self, tmp_path):
        # The whole point: a page calling this must never crash.
        for bad_path in (tmp_path / "missing.yaml", tmp_path):
            cfg, error = load_config_v2_safe(bad_path)
            assert cfg is None
            assert isinstance(error, str)
