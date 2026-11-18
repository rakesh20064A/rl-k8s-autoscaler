"""models.py -- Pure data schemas shared by the demo UI's state machine,
providers, comparison engine and components. No Streamlit/Plotly import
here: this module must be importable and unit-testable in complete
isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class PodState(str, Enum):
    READY = "READY"
    STARTING = "STARTING"
    TERMINATING = "TERMINATING"
    FAILED = "FAILED"


class PpoAction(str, Enum):
    SCALE_DOWN = "SCALE_DOWN"
    KEEP = "KEEP"
    SCALE_UP = "SCALE_UP"


@dataclass
class EventMarker:
    """A vertical marker overlaid on time-series charts, correlated with a
    LogEvent by timestamp so graphs and the evidence log share one basis."""
    timestamp: float
    label: str
    category: str  # e.g. "PPO_SCALE_UP", "HPA_SCALE", "DEPLOYMENT_REPLICA_CHANGE", "POD_READY"


@dataclass
class PodStatus:
    name: str
    state: PodState


@dataclass
class HpaSnapshot:
    cpu_target_pct: float
    current_cpu_pct: float
    current_replicas: int
    desired_replicas: int
    min_replicas: int
    max_replicas: int
    latest_scaling_event: Optional[str] = None


@dataclass
class PpoDecision:
    timestamp: float
    observation_summary: str
    replicas_before: int
    action: PpoAction
    replicas_after: int
    reward: float


@dataclass
class PpoSnapshot:
    latest_action: Optional[PpoAction]
    latest_reward: Optional[float]
    observation_timestamp: Optional[float]
    decisions: List[PpoDecision] = field(default_factory=list)


