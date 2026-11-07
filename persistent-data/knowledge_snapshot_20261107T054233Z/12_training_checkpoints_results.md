# Part 12 — Training, Checkpoints, and Results

VERIFIED this session (via research agent, re-stated here) unless marked
FROM_CODE.

## V1 (frozen)

`models/ppo/`: `metadata.json` (1784B), `ppo_model.zip` (484,495B),
`rng_state.pkl` (12,008B) — all Oct 5, 16:58 (committed with the repo's
only 2 commits). `models/ppo_v2/`: **empty**. `results/`: `evaluation/`,
`figures/`, `provenance/` subdirectories present (contents not
individually enumerated this pass).

## V2 pilot (`persistent-data/rl_training_v2/pilot/`)

Top level also has `ppo_pilot_episode_ledger_v2.jsonl` (21,402B) and
`traffic_provenance_ledger_v2.jsonl` (24,169B).

| Session | Files present | telemetry rows | SESSION_FAILED.json |
|---|---|---|---|
| session_000 | step_telemetry.jsonl only | 2 | No |
| session_001 | +metadata/model/metrics | 7 | No |
| session_002 | step_telemetry.jsonl only | 139 | No |
| session_003 | +metadata/model/metrics | 256 | No |

Note: session_000/002 lack metadata.json/ppo_model.zip/training_metrics.json
that session_001/003 have — consistent with this conversation's own
history (session_000 = attempt1, Prometheus-port-mismatch failure before
any checkpoint could be written; session_002 = attempt3, stopped by a
genuine Prometheus pod restart during drain, also before its own
checkpoint write completed). **Pilot telemetry schema does NOT include
`normalized_observation`** (confirmed by direct read of
`pilot/session_002/step_telemetry.jsonl`'s first row keys) — this field was
only added in the Phase 4A instrumentation fix, which postdates all pilot
sessions.

## V2 final campaign (`persistent-data/rl_training_v2/final/`)

Full `campaign_metadata.json` (VERIFIED, re-read this session — unchanged
from every prior recorded value):

```json
{
  "campaign_classification": "V2_FINAL_TRAINING_CAMPAIGN",
  "campaign_last_updated_utc": "2026-10-12T07:53:38.624683+00:00",
  "campaign_started_utc": "2026-10-11T13:14:59.722807+00:00",
  "cumulative_timesteps": 768,
  "cumulative_updates": 3,
  "failed_physical_attempt": 2,
  "failed_scientific_session_index": 3,
  "failure_reason": "pod_restart_detected:demo-app-6579f6d9b-fdflt",
  "latest_verified_model_sha256": "c721144f006d6ff5281dac09eba7050d87c7668aaf23c6acb2ad63e7ba76d3f2",
  "latest_verified_session_dir": ".../sessions/session_002",
  "n_steps_per_session": 256,
  "next_global_schedule_episode_index": 42,
  "planned_sessions": 20,
  "remaining_timesteps": 4352,
  "resume_eligible": true,
  "sessions_completed": ["session_000", "session_001", "session_002"],
  "status": "stopped_on_failure",
  "total_timesteps_budget": 5120,
  "training_seed": 42
}
```

(Paths inside the file are the correct current repo path, not the deleted
old repo — this file is unaffected by the `.venv` symlink issue.)

Sessions:

| Session | Files | telemetry rows | SESSION_FAILED.json |
|---|---|---|---|
| session_000/001/002 | candidate_metadata, metadata, ppo_model.zip, rng_state.pkl, step_telemetry, training_metrics | 256 each | No |
| session_003 | same set | 121 | **Yes** |
| session_003__attempt2 | same set | 17 | **Yes** |

`ppo_update_metrics.jsonl`: exists, 4 rows (one per completed PPO update
across sessions 000-002, plus formatting consistent with
`cumulative_updates: 3`). First row:
```json
{"approx_kl": 0.00217..., "clip_fraction": 0.0, "cumulative_timesteps": 256,
 "entropy_loss": -1.0978..., "explained_variance": 0.00415...,
 "learning_rate": 0.0003, "loss": 436.811..., "policy_gradient_loss": -0.00472...,
 "session_index": 0, "update_index": 0, "value_loss": 934.944...}
```

`normalized_observation` check: **final-campaign `session_002`'s telemetry
DOES have it** (`['timestamp_utc','profile','sampled_action',
'effective_action','override_reason','reward_total','reward_components',
'raw_observation','normalized_observation','deployment_current_replicas_after',
'ready_replicas_after']`) — confirms the Phase 4A instrumentation fix was
genuinely implemented and real data exists (this directly answers Part 17
Q4).

## Is resume possible, from which checkpoint, what would break it

Resume IS possible: from `session_002`
(`sha256 c721144f006d6ff5281dac09eba7050d87c7668aaf23c6acb2ad63e7ba76d3f2`,
`rng_state.pkl` alongside it), at `next_global_schedule_episode_index=42`,
scientific session index 3 (next physical attempt would be
`session_003__attempt3` per the project's own retry-safe naming
convention, confirmed working twice already). `verify_resume_provenance()`
would need to re-pass its `governing_hashes_snapshot` check against the
CURRENT state of `environment_v2.py`/`reward_v2.py`/`observation.py`/
`rl_v2.yaml`/`reward_v2.yaml` — none of which have changed since
`session_002` was created (confirmed via Part 05's git-diff inventory,
none of these 5 files appear as modified). **What would break it**: (1)
the still-unresolved live-cluster non-canonical state (Part 04) — a
restoration is required before ANY live resume attempt; (2) the two
confirmed P1 defects from Phase 4B-R4F (V1-environment coupling in
`evaluate_ppo_vs_hpa_v2.py`; the missing V2-HPA restoration path) do NOT
themselves block a resume (training never calls those scripts), but the
task's own standing instruction requires the full V2 setup "implemented,
tested, validated and frozen" before final training resumes; (3) the
genuinely unresolved root-cause question from Phase 4B-R2/R3/R4
(constant_cpu-workload-induced pod restarts under the current probe/
resource contract) — resuming without addressing this risks a THIRD
failure at the same scientific session index for the same underlying
reason, as already happened twice.
