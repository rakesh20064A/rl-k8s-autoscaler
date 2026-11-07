"""autoscaler/src/rl_env/training_wrapper.py

Phase 16V, task sec.6: a thin `gymnasium.Wrapper` around
`KubernetesAutoscalerEnv` that appends one immutable ledger row per
completed episode (episode_ledger.append_row), distinguishing exactly the
outcomes the task requires:

  * `"truncated"`   -- the episode ran to its natural time-limit end
                       (`truncated=True`, `terminated=False` throughout) --
                       the normal, expected outcome (docs/rl_mdp_spec.md
                       sec.20: "not a failure").
  * `"safety_aborted"` -- the frozen contract's OWN deterministic safety
                       termination fired (`terminated=True`: 4 consecutive
                       invalid steps, or zero Ready replicas for >120s,
                       docs/rl_mdp_spec.md sec.13).
  * `"tooling_failed"` -- an exception escaped the normal step()/reset()
                       Gymnasium contract entirely (e.g.
                       live_backend.RestorationTimeoutError at reset, or any
                       other unhandled exception) -- logged by the CALLER
                       (the training runner script's own try/except around
                       reset()/the whole training loop, since a Wrapper
                       cannot itself return a valid Gymnasium 5-tuple/obs
                       for a call that never completed).
  * `"session_boundary_forced_reset"` (Phase16W) -- a resumable live-training
                       session ended (its target rollout count completed
                       normally) while an episode was still in progress
                       (neither `terminated` nor `truncated` had yet fired).
                       SB3's own `PPO.load()` forces a fresh `env.reset()`
                       at the start of the next session regardless (its
                       rollout buffer and environment object are never
                       pickled) -- this status honestly records that the
                       in-progress episode's trajectory was cut short by the
                       session boundary itself, not by any frozen-contract
                       termination condition. Logged by the CALLER (the
                       session-based training runner) via
                       `log_session_boundary_if_episode_in_progress()`
                       immediately before exiting a session, never silently
                       folded into `"truncated"`.

`valid` is a derived boolean, not a fourth independent status: `True` only
for `"truncated"` (a clean, full-length rollout). A `"safety_aborted"`
episode's recorded transitions are still genuine, honest data (nothing is
fabricated) but must not be treated as if the episode reached its intended
length. This mapping is deliberate -- see the module docstring's cross-
reference in phase_16v_implementation_audit.md.

This wrapper does not change any observation/action/reward semantics --
`step()`/`reset()` delegate to the wrapped environment unchanged; it only
observes the returned tuples to append ledger rows.

Phase16AB addition: every row now also carries `physical_attempt_id` /
`physical_attempt_number` (via `episode_ledger.next_physical_attempt_
number()`), distinguishing the stable SCIENTIFIC identity from the
physical EXECUTION-ATTEMPT identity (unique per attempt, computed by
scanning the ledger file itself, never in-memory state that resets every
process restart). This ensures a retried attempt at the same scientific
slot (e.g. after an earlier tooling failure) never overwrites, reuses, or
is indistinguishable from the prior attempt's row -- both coexist
permanently in the ledger.

Phase17 correction: the legacy `episode_id` (e.g. `"training_mixed_42_1"`)
is built from this wrapper's own PER-PROCESS `_episode_index` counter,
which restarts at 0 every time a new `LedgeredEpisodeWrapper` is
constructed (i.e. every rollout process). It is therefore NOT a globally
unique scientific-slot identity: two DIFFERENT global schedule positions
across different rollouts can produce an IDENTICAL legacy `episode_id` if
they happen to land on the same per-process index and the same assigned
profile -- which Phase16AE demonstrated concretely (see
`task_outputs/phase_17_provenance_correction_report.md`). Every row now
additionally carries:

  * `global_schedule_episode_index` -- the AUTHORITATIVE, globally
    monotonic index used by `resolve_profile_for_episode()` to assign this
    episode's traffic profile. Supplied externally via the new
    `global_schedule_episode_index` attribute (set by
    `scripts.traffic_orchestrator.TrafficSyncedWrapper.reset()` before
    calling this wrapper's own `reset()`, mirroring how `traffic_profile`
    is already propagated) -- this wrapper never maintains a second,
    independent schedule counter of its own. When no external value is
    supplied (e.g. plain `ScriptedBackend` tests with no
    `TrafficSyncedWrapper`), falls back to the legacy per-process index for
    backward compatibility.
  * `scientific_episode_id` -- a deterministic function of role,
    `global_schedule_episode_index`, assigned profile, and seed
    (`f"{role}_schedule_{global_index:06d}_{profile}_{seed}"`). THIS is the
    identity `physical_attempt_number`/`physical_attempt_id` are now scoped
    to (via `episode_ledger.next_physical_attempt_number(...,
    field="scientific_episode_id")`), not the legacy `episode_id` -- so a
    genuine retry of the SAME global schedule slot is correctly counted as
    attempt 2/3/..., while two DIFFERENT global slots that happen to share
    a legacy `episode_id` each independently start at attempt 1.

The legacy `episode_id` field is preserved, unchanged, on every row for
backward compatibility; it is no longer used for attempt-number scoping.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import gymnasium as gym

from . import episode_ledger

STATUS_TRUNCATED = "truncated"
STATUS_SAFETY_ABORTED = "safety_aborted"
STATUS_TOOLING_FAILED = "tooling_failed"
STATUS_SESSION_BOUNDARY_FORCED_RESET = "session_boundary_forced_reset"
# V2 Phase 3C PART 3: an EXTERNAL caller-side safety mechanism (e.g. a V2
# pilot's SB3 callback) stopped model.learn() mid-episode -- a different
# event from STATUS_SESSION_BOUNDARY_FORCED_RESET (planned session-length
# boundary) or STATUS_SAFETY_ABORTED (the environment's OWN consecutive-
# invalid-telemetry logic). Added because V2 Phase 3A/3B's physical attempt
# 2 (PPO pilot) demonstrated this exact gap: a callback-level stop never
# reaches this wrapper's own step()'s terminated/truncated branch, so no
# ledger row was written for that attempt at all.
STATUS_PILOT_SAFETY_INTERRUPTED = "pilot_safety_interrupted"
ALL_STATUSES = (
    STATUS_TRUNCATED, STATUS_SAFETY_ABORTED, STATUS_TOOLING_FAILED,
    STATUS_SESSION_BOUNDARY_FORCED_RESET, STATUS_PILOT_SAFETY_INTERRUPTED,
)


class LedgeredEpisodeWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        ledger_path: Path,
        traffic_profile: str,
        profile_hash: str,
        seed: int,
        controller_code_hash: str,
        config_hash: str,
        role: str,  # "training" | "final_evaluation_ppo" | "final_evaluation_hpa"
    ) -> None:
        super().__init__(env)
        self.ledger_path = ledger_path
        self.traffic_profile = traffic_profile
        self.profile_hash = profile_hash
        self.seed = seed
        self.controller_code_hash = controller_code_hash
        self.config_hash = config_hash
        self.role = role

        self._episode_id: Optional[str] = None
        self._episode_index = 0
        # Phase17: settable externally (mirrors `traffic_profile`) by a
        # caller that knows the authoritative global schedule index (e.g.
        # TrafficSyncedWrapper, BEFORE calling this wrapper's reset()).
        # None (the default) means "no external authority supplied" --
        # reset() then falls back to the legacy per-process index.
        self.global_schedule_episode_index: Optional[int] = None
        self._resolved_global_schedule_episode_index: Optional[int] = None
        self._scientific_episode_id: Optional[str] = None
        self._n_steps = 0
        self._total_reward = 0.0
        self._reward_validity_counts: Dict[str, int] = {"full": 0, "partial": 0}
        self._safety_events: list = []
        self._start_utc: Optional[str] = None
        self._ppo_update_index: Optional[int] = None
        self._episode_completed = True  # no episode in progress until the first reset()

    def set_ppo_update_index(self, index: int) -> None:
        """Set by the training loop so the ledger row records which PPO
        rollout/update this episode's steps contributed to."""
        self._ppo_update_index = index

    def reset(self, **kwargs):
        self._episode_index += 1
        self._episode_id = f"{self.role}_{self.traffic_profile}_{self.seed}_{self._episode_index}"
        # Phase17: resolve the GLOBAL schedule index (the one authoritative
        # source, supplied externally when available) and derive the
        # globally-unique scientific_episode_id from it -- computed here,
        # at the same point as the legacy episode_id, so both are already
        # set before self.env.reset() is called (and therefore available
        # even if that call raises -- see log_tooling_failure()).
        self._resolved_global_schedule_episode_index = (
            self.global_schedule_episode_index
            if self.global_schedule_episode_index is not None
            else self._episode_index - 1
        )
        self._scientific_episode_id = (
            f"{self.role}_schedule_{self._resolved_global_schedule_episode_index:06d}_"
            f"{self.traffic_profile}_{self.seed}"
        )
        self._n_steps = 0
        self._total_reward = 0.0
        self._reward_validity_counts = {"full": 0, "partial": 0}
        self._safety_events = []
        self._start_utc = datetime.now(timezone.utc).isoformat()
        self._episode_completed = False
        return self.env.reset(**kwargs)

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._n_steps += 1
        self._total_reward += float(reward)
        validity = info.get("reward_validity", "full")
        self._reward_validity_counts[validity] = self._reward_validity_counts.get(validity, 0) + 1

        override_reason = info.get("action_override_reason")
        if override_reason in (
            "stale_or_missing_metric", "prometheus_unavailable",
            "policy_inference_failure", "k8s_api_failure",
        ):
            self._safety_events.append({"step_index": info.get("step_index"), "reason": override_reason})

        if terminated or truncated:
            status = STATUS_SAFETY_ABORTED if terminated else STATUS_TRUNCATED
            attempt_n = episode_ledger.next_physical_attempt_number(
                self.ledger_path, self._scientific_episode_id, field="scientific_episode_id",
            )
            row = {
                "episode_id": self._episode_id,
                "scientific_episode_id": self._scientific_episode_id,
                "global_schedule_episode_index": self._resolved_global_schedule_episode_index,
                "physical_attempt_id": f"{self._scientific_episode_id}__attempt{attempt_n}",
                "physical_attempt_number": attempt_n,
                "role": self.role,
                "seed": self.seed,
                "traffic_profile": self.traffic_profile,
                "profile_hash": self.profile_hash,
                "n_decision_steps": self._n_steps,
                "ppo_update_index": self._ppo_update_index,
                "start_utc": self._start_utc,
                "end_utc": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "valid": status == STATUS_TRUNCATED,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "total_reward": self._total_reward,
                "reward_validity_counts": dict(self._reward_validity_counts),
                "safety_events": self._safety_events,
                "controller_code_hash": self.controller_code_hash,
                "config_hash": self.config_hash,
            }
            episode_ledger.append_row(self.ledger_path, row)
            self._episode_completed = True

        return obs, reward, terminated, truncated, info

    def log_session_boundary_if_episode_in_progress(self) -> bool:
        """Phase16W: called by the session-based training runner
        immediately before a session exits (after its target rollout count
        has completed). If the current episode never reached
        `terminated`/`truncated` (i.e. the session boundary itself cut it
        short), appends a `session_boundary_forced_reset` row with whatever
        partial stats were accumulated. Returns True if a row was appended,
        False if the prior episode had already completed cleanly (nothing
        to log). Idempotent-safe to call multiple times -- a second call
        with no new episode in progress is a no-op."""
        if self._episode_completed:
            return False
        attempt_n = episode_ledger.next_physical_attempt_number(
            self.ledger_path, self._scientific_episode_id, field="scientific_episode_id",
        )
        row = {
            "episode_id": self._episode_id,
            "scientific_episode_id": self._scientific_episode_id,
            "global_schedule_episode_index": self._resolved_global_schedule_episode_index,
            "physical_attempt_id": f"{self._scientific_episode_id}__attempt{attempt_n}",
            "physical_attempt_number": attempt_n,
            "role": self.role,
            "seed": self.seed,
            "traffic_profile": self.traffic_profile,
            "profile_hash": self.profile_hash,
            "n_decision_steps": self._n_steps,
            "ppo_update_index": self._ppo_update_index,
            "start_utc": self._start_utc,
            "end_utc": datetime.now(timezone.utc).isoformat(),
            "status": STATUS_SESSION_BOUNDARY_FORCED_RESET,
            "valid": False,
            "terminated": False,
            "truncated": False,
            "total_reward": self._total_reward,
            "reward_validity_counts": dict(self._reward_validity_counts),
            "safety_events": self._safety_events,
            "controller_code_hash": self.controller_code_hash,
            "config_hash": self.config_hash,
        }
        episode_ledger.append_row(self.ledger_path, row)
        self._episode_completed = True
        return True

    def log_pilot_safety_interruption(self, reason: str) -> bool:
        """V2 Phase 3C PART 3: called by an EXTERNAL safety mechanism (e.g.
        scripts.v2_pilot_safety.PilotSafetyCallback) immediately after it
        decides to stop model.learn() early, BEFORE this wrapper's own
        step() has any chance to observe a terminated/truncated signal for
        the in-progress episode (a callback-level SB3 stop never reaches
        this wrapper's step() at all for that final step). Idempotent-safe
        exactly like log_session_boundary_if_episode_in_progress(): returns
        False (no row written) if the episode already completed normally
        through this wrapper's own step() on this same final call -- this
        is what prevents a duplicate row when both the environment's own
        termination AND the external safety stop could plausibly fire for
        the same step. Never fabricates terminated=True/truncated=True --
        both are recorded as False, since the environment itself never
        signalled either."""
        if self._episode_completed:
            return False
        attempt_n = episode_ledger.next_physical_attempt_number(
            self.ledger_path, self._scientific_episode_id, field="scientific_episode_id",
        )
        row = {
            "episode_id": self._episode_id,
            "scientific_episode_id": self._scientific_episode_id,
            "global_schedule_episode_index": self._resolved_global_schedule_episode_index,
            "physical_attempt_id": f"{self._scientific_episode_id}__attempt{attempt_n}",
            "physical_attempt_number": attempt_n,
            "role": self.role,
            "seed": self.seed,
            "traffic_profile": self.traffic_profile,
            "profile_hash": self.profile_hash,
            "n_decision_steps": self._n_steps,
            "ppo_update_index": self._ppo_update_index,
            "start_utc": self._start_utc,
            "end_utc": datetime.now(timezone.utc).isoformat(),
            "status": STATUS_PILOT_SAFETY_INTERRUPTED,
            "valid": False,
            "terminated": False,
            "truncated": False,
            "total_reward": self._total_reward,
            "reward_validity_counts": dict(self._reward_validity_counts),
            "safety_events": self._safety_events + [{"step_index": self._n_steps, "reason": reason}],
            "controller_code_hash": self.controller_code_hash,
            "config_hash": self.config_hash,
        }
        episode_ledger.append_row(self.ledger_path, row)
        self._episode_completed = True
        return True

    def log_tooling_failure(self, error: Exception, phase: str) -> None:
        """Called by the OUTER training/eval loop (never by this wrapper's
        own step()/reset(), which cannot itself observe an exception that
        prevents a valid Gymnasium return) when reset() or another
        infrastructure operation raises. Appends a row with
        status='tooling_failed', valid=False -- never silently dropped."""
        episode_id = self._episode_id or f"{self.role}_{self.traffic_profile}_{self.seed}_{self._episode_index}"
        if self._scientific_episode_id is not None:
            scientific_episode_id = self._scientific_episode_id
            global_schedule_episode_index = self._resolved_global_schedule_episode_index
        else:
            # reset() never ran even once -- fall back exactly as the
            # legacy episode_id fallback above does.
            global_schedule_episode_index = (
                self.global_schedule_episode_index
                if self.global_schedule_episode_index is not None
                else self._episode_index
            )
            scientific_episode_id = (
                f"{self.role}_schedule_{global_schedule_episode_index:06d}_"
                f"{self.traffic_profile}_{self.seed}"
            )
        attempt_n = episode_ledger.next_physical_attempt_number(
            self.ledger_path, scientific_episode_id, field="scientific_episode_id",
        )
        row = {
            "episode_id": episode_id,
            "scientific_episode_id": scientific_episode_id,
            "global_schedule_episode_index": global_schedule_episode_index,
            "physical_attempt_id": f"{scientific_episode_id}__attempt{attempt_n}",
            "physical_attempt_number": attempt_n,
            "role": self.role,
            "seed": self.seed,
            "traffic_profile": self.traffic_profile,
            "profile_hash": self.profile_hash,
            "n_decision_steps": self._n_steps,
            "ppo_update_index": self._ppo_update_index,
            "start_utc": self._start_utc or datetime.now(timezone.utc).isoformat(),
            "end_utc": datetime.now(timezone.utc).isoformat(),
            "status": STATUS_TOOLING_FAILED,
            "valid": False,
            "terminated": False,
            "truncated": False,
            "total_reward": self._total_reward,
            "reward_validity_counts": dict(self._reward_validity_counts),
            "safety_events": self._safety_events,
            "controller_code_hash": self.controller_code_hash,
            "config_hash": self.config_hash,
            "tooling_failure_phase": phase,
            "tooling_failure_error": f"{type(error).__name__}: {error}",
        }
        episode_ledger.append_row(self.ledger_path, row)
        self._episode_completed = True
