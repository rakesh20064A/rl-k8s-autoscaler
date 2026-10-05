"""models.py -- Pure data schemas shared by the demo UI's state machine,
providers, comparison engine and components. No Streamlit/Plotly import
here: this module must be importable and unit-testable in complete
isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Controller(str, Enum):
    HPA = "HPA"
    PPO = "PPO"


class TrafficProfile(str, Enum):
    CONSTANT = "constant"
    RAMP = "ramp"
    BURST = "burst"
    STEP = "step"
    PERIODIC = "periodic"
    MIXED = "mixed"


class LogSource(str, Enum):
    TRAFFIC = "TRAFFIC"
    PROMETHEUS = "PROMETHEUS"
    HPA = "HPA"
    PPO = "PPO"
    KUBERNETES = "KUBERNETES"
    SYSTEM = "SYSTEM"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


class PodState(str, Enum):
    READY = "READY"
    STARTING = "STARTING"
    TERMINATING = "TERMINATING"
    FAILED = "FAILED"


class PpoAction(str, Enum):
    SCALE_DOWN = "SCALE_DOWN"
    KEEP = "KEEP"
    SCALE_UP = "SCALE_UP"


# ---------------------------------------------------------------------------
# Single-controller (Auto OFF) state machine states
# ---------------------------------------------------------------------------

class SingleRunState(str, Enum):
    IDLE = "IDLE"
    PREPARING = "PREPARING"
    RUNNING_SELECTED_CONTROLLER = "RUNNING_SELECTED_CONTROLLER"
    RESETTING = "RESETTING"
    COMPLETE = "COMPLETE"


# ---------------------------------------------------------------------------
# Auto (HPA-then-PPO fair-replay) state machine states
# ---------------------------------------------------------------------------

class AutoRunState(str, Enum):
    IDLE = "IDLE"
    PREPARING_HPA = "PREPARING_HPA"
    HPA_RUNNING = "HPA_RUNNING"
    HPA_COMPLETE = "HPA_COMPLETE"
    RESETTING = "RESETTING"
    BASELINE_STABLE = "BASELINE_STABLE"
    PREPARING_PPO = "PREPARING_PPO"
    PPO_RUNNING = "PPO_RUNNING"
    PPO_COMPLETE = "PPO_COMPLETE"
    VALIDATING = "VALIDATING"
    COMPARISON_READY = "COMPARISON_READY"


@dataclass(frozen=True)
class ResetSnapshot:
    """The canonical post-reset environment state. Reset must converge here,
    never to replicas=0 -- see project discipline on reset semantics."""
    traffic_rps: float = 0.0
    replicas: int = 1
    application_ready: bool = False
    metrics_stabilizing: bool = True


@dataclass
class LogEvent:
    timestamp: float  # seconds since experiment epoch (shared basis with charts)
    source: LogSource
    level: LogLevel
    event_type: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimeSeriesPoint:
    timestamp: float
    value: float


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


@dataclass
class RunSummary:
    """One controller's full result for one experiment run -- the unit the
    comparison engine consumes. Time series are parallel-indexed lists
    sharing one timestamp basis with `events`/`logs`."""
    controller: Controller
    traffic_profile: TrafficProfile
    seed: int
    duration_s: float
    schedule_hash: str

    timestamps: List[float]
    replicas: List[int]
    cpu_pct: List[float]
    memory_pct: List[float]
    rps: List[float]
    latency_avg_ms: List[float]
    latency_p95_ms: List[float]
    latency_p99_ms: List[float]
    successful_requests: int
    errors: int
    timeouts: int
    sla_violations: int

    events: List[EventMarker] = field(default_factory=list)
    logs: List[LogEvent] = field(default_factory=list)
    pods: List[PodStatus] = field(default_factory=list)
    hpa: Optional[HpaSnapshot] = None
    ppo: Optional[PpoSnapshot] = None

    is_demo_data: bool = True


@dataclass
class ComparisonMetric:
    name: str
    hpa_value: float
    ppo_value: float
    unit: str = ""
    lower_is_better: bool = True

    @property
    def pct_difference(self) -> float:
        """PPO vs HPA, signed: negative means PPO is lower. Returns 0.0 when
        hpa_value is 0 to avoid a division error (documented, not hidden)."""
        if self.hpa_value == 0:
            return 0.0
        return ((self.ppo_value - self.hpa_value) / abs(self.hpa_value)) * 100.0

    @property
    def ppo_is_better(self) -> bool:
        if self.ppo_value == self.hpa_value:
            return False
        if self.lower_is_better:
            return self.ppo_value < self.hpa_value
        return self.ppo_value > self.hpa_value


@dataclass
class ComparisonResult:
    metrics: List[ComparisonMetric]
    hpa_replica_seconds: float
    ppo_replica_seconds: float
    hpa_avg_replicas: float
    ppo_avg_replicas: float
    hpa_peak_replicas: int
    ppo_peak_replicas: int
    hpa_scaling_events: int
    ppo_scaling_events: int
    same_schedule_verified: bool


class ConclusionCategory(str, Enum):
    PPO_BETTER_OVERALL = "PPO_BETTER_OVERALL"
    HPA_BETTER_OVERALL = "HPA_BETTER_OVERALL"
    TRADE_OFF = "TRADE_OFF"
    SIMILAR = "SIMILAR"


@dataclass
class ConclusionDimension:
    name: str  # e.g. "Latency", "SLA", "Resource efficiency", "Scaling behaviour"
    verdict: str  # short human-readable finding for this dimension
    ppo_favored: Optional[bool]  # None = no material difference


@dataclass
class Conclusion:
    category: ConclusionCategory
    headline: str
    dimensions: List[ConclusionDimension]
    is_demo_data: bool = True


@dataclass
class ExperimentRecord:
    """Fixture-backed shape matching the future on-disk
    experiments/<id>/{hpa,ppo}/... layout (sec.17)."""
    experiment_id: str
    traffic_profile: TrafficProfile
    controllers: List[Controller]
    timestamp_label: str
    status: str  # e.g. "COMPLETE", "FAILED", "PARTIAL"
    valid: bool
