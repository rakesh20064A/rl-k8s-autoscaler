"""demo_ui/config_v2.py -- loads demo_ui/config_v2.yaml and determines
whether V2 LIVE mode may be offered.

Pure Python + stdlib/PyYAML only (no Streamlit import), same discipline as
experiment_store.py/models.py -- importable and unit-testable in isolation.

V2 LIVE requires BOTH the final model checkpoint AND the epsilon
calibration file to exist on disk; if either is missing, V2 LIVE is
disabled with a clear, specific message naming which file is missing --
DEMO (fixture) mode is unaffected either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config_v2.yaml"


@dataclass(frozen=True)
class ConfigV2:
    model_path: Path
    epsilon_calibration_path: Path
    evaluation_campaign_dir: Path

    @property
    def model_exists(self) -> bool:
        return self.model_path.is_file()

    @property
    def epsilon_calibration_exists(self) -> bool:
        return self.epsilon_calibration_path.is_file()

    @property
    def evaluation_campaign_exists(self) -> bool:
        return self.evaluation_campaign_dir.is_dir()

    @property
    def live_enabled(self) -> bool:
        """V2 LIVE requires BOTH files -- missing either one disables it
        entirely (DEMO mode is unaffected)."""
        return self.model_exists and self.epsilon_calibration_exists

    @property
    def live_disabled_reason(self) -> Optional[str]:
        if self.live_enabled:
            return None
        missing = []
        if not self.model_exists:
            missing.append(f"model checkpoint not found at {self.model_path}")
        if not self.epsilon_calibration_exists:
            missing.append(f"epsilon calibration file not found at {self.epsilon_calibration_path}")
        return "V2 LIVE is disabled: " + "; ".join(missing) + ". DEMO mode is still available."


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def load_config_v2_safe(config_path: Path = DEFAULT_CONFIG_PATH) -> "tuple[Optional[ConfigV2], Optional[str]]":
    """The ONE call site every UI view should use -- `load_config_v2()`
    itself still raises on genuine misconfiguration (missing/malformed
    YAML), but no page render should ever crash over that. Returns
    `(config, None)` on success or `(None, error_message)` on any
    OSError/ValueError/YAML error, so a view can show a clear message and
    keep DEMO/V1 usable either way."""
    try:
        return load_config_v2(config_path), None
    except (OSError, ValueError) as exc:
        return None, f"V2 config unavailable ({config_path}): {exc}"
    except yaml.YAMLError as exc:
        return None, f"V2 config is not valid YAML ({config_path}): {exc}"


def load_config_v2(config_path: Path = DEFAULT_CONFIG_PATH) -> ConfigV2:
    """Never raises on a missing/malformed MODEL or epsilon file -- those
    are reported via `live_enabled`/`live_disabled_reason`, not an
    exception, so the UI can still render DEMO mode. Raises only if the
    config YAML itself is missing or has no usable path fields -- those are
    genuine misconfiguration, not an expected "training not done yet"
    state."""
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    for required in ("model_path", "epsilon_calibration_path", "evaluation_campaign_dir"):
        if required not in data:
            raise ValueError(f"{config_path} is missing required field {required!r}")

    return ConfigV2(
        model_path=_resolve(data["model_path"]),
        epsilon_calibration_path=_resolve(data["epsilon_calibration_path"]),
        evaluation_campaign_dir=_resolve(data["evaluation_campaign_dir"]),
    )
