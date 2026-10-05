#!/usr/bin/env python3
"""scripts/analyze_final_evaluation.py

Index39: OFFLINE-ONLY final-evaluation analysis and result synthesis.

Reads only existing, already-committed JSON/JSONL evidence under
persistent-data/evaluation/final_session019/ and
persistent-data/rl_training/{ppo,hpa}_final_evaluation_ledger.jsonl.

Performs NO Kubernetes call, NO Prometheus live query, NO model
load/predict/learn, NO traffic generation. Deterministic given the files on
disk. Never overwrites original episode artifacts or ledgers -- writes only
new, additively-named files under task_outputs/.

Section-10 metric definitions are copied VERBATIM from
task_outputs/phase_16u_direct_ppo_protocol.yaml (frozen BEFORE any
evaluation episode ran) -- never reinterpreted here.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "persistent-data" / "evaluation" / "final_session019"
PPO_LEDGER = PROJECT_ROOT / "persistent-data" / "rl_training" / "ppo_final_evaluation_ledger.jsonl"
HPA_LEDGER = PROJECT_ROOT / "persistent-data" / "rl_training" / "hpa_final_evaluation_ledger.jsonl"
TASK_OUTPUTS = PROJECT_ROOT / "task_outputs"

EXPECTED_HOLDOUT_HASH = "308f4f90ea3cea97f1927cec97efeaa652c4424e9fe5bcea421358f3425e3409"
EXPECTED_SEED = 777
SLA_P95_MS = 300.0  # configs/experiment.yaml sla.p95_latency_ms, frozen

EPISODE_FILES = {
    "PPO1": "episode_01_ppo_repeat1.json",
    "HPA1": "episode_02_hpa_repeat1.json",
    "PPO2": "episode_03_ppo_repeat2.json",
    "HPA2": "episode_04_hpa_repeat2.json",
    "PPO3": "episode_05_ppo_repeat3.json",
    "HPA3": "episode_06_hpa_repeat3.json",
}
HPA2_ATTEMPT1_FAILURE_FILE = "episode_04_hpa_repeat2_physical_attempt1_FAILED.json"


class ProtocolMismatchError(RuntimeError):
    """Raised if the on-disk evidence or protocol text does not match the
    frozen Section-10 definitions this script implements -- fail closed,
    never silently reinterpret."""


def _load(name: str) -> Dict[str, Any]:
    path = EVAL_DIR / EPISODE_FILES[name]
    with open(path) as f:
        return json.load(f)


def _nominal_steps(episode: Dict[str, Any]) -> int:
    return episode.get("nominal_decision_steps") or int(round(episode["workload_duration_seconds"] / 30.0))


def _classify_legacy(episode: Dict[str, Any]) -> Dict[str, Any]:
    """PPO1 predates the Index38B classification fields (episode_status,
    full_holdout_completed, valid_for_primary_comparison, etc.) -- this
    derives them identically to _classify_termination() in
    scripts/evaluate_ppo_vs_hpa.py, from the same raw fields, without
    editing the historical row/file itself."""
    if episode.get("episode_status") is not None:
        return episode
    nominal = _nominal_steps(episode)
    observed = max(0, episode["n_decision_steps"] - 1)
    full = observed >= nominal
    episode = dict(episode)
    episode["nominal_decision_steps"] = nominal
    episode["observed_decision_steps"] = observed
    episode["full_holdout_completed"] = full
    episode["valid_for_primary_comparison"] = full
    episode["episode_status"] = "completed_full_duration" if full else "terminated_safety"
    return episode


def compute_section10_metrics(episode: Dict[str, Any]) -> Dict[str, Any]:
    """Frozen Section-10 per-episode metrics (phase_16u_direct_ppo_protocol.yaml
    Section 10), computed from the episode's own recorded `samples`
    (and `decisions` for PPO's reaction-time/churn)."""
    samples = episode["samples"]
    decisions = episode.get("decisions") or []

    # 1. sla_violation_rate: fraction of decision steps with p95 > 300ms.
    # Decision STEPS exclude the initial t=0 snapshot (sample[0] is the
    # pre-episode snapshot, not a decision outcome).
    decision_samples = samples[1:] if len(samples) > 1 else []
    sla_breaches = [s for s in decision_samples if (s.get("latency_p95_ms") or 0.0) > SLA_P95_MS]
    sla_violation_rate = (len(sla_breaches) / len(decision_samples)) if decision_samples else None

    # 2. error_rate_mean: mean per-step (error_rate_rps / request_rate_rps),
    # 0 if request_rate_rps == 0.
    per_step_error_rates = []
    for s in decision_samples:
        req = s.get("request_rate_rps")
        err = s.get("error_rate_rps")
        if req is None or err is None:
            continue
        per_step_error_rates.append(0.0 if req == 0 else err / req)
    error_rate_mean = (sum(per_step_error_rates) / len(per_step_error_rates)) if per_step_error_rates else None

    # 3. resource_proxy_mean: mean per-step (deployment_current_replicas-1)/7.
    resource_proxies = [(s["replicas_current"] - 1) / 7.0 for s in decision_samples]
    resource_proxy_mean = (sum(resource_proxies) / len(resource_proxies)) if resource_proxies else None

    # 4. reaction_time_seconds: for EACH SLA-breach onset (p95 crosses above
    # 300ms from below), seconds until the first EFFECTIVE scale-up action
    # at or after that onset -- independent of whether the breach itself
    # later auto-resolves before the scale-up is observed. Averaged across
    # onsets; excluded (None) if no breach occurs.
    onset_times: List[float] = []
    was_breaching = False
    for s in decision_samples:
        is_breaching = (s.get("latency_p95_ms") or 0.0) > SLA_P95_MS
        if is_breaching and not was_breaching:
            onset_times.append(s["time_s"])
        was_breaching = is_breaching

    reaction_times: List[float] = []
    for onset_time in onset_times:
        if decisions:
            hit = next(
                (d for d in decisions if d["decision_time_s"] >= onset_time and d["replicas_after"] > d["replicas_before"]),
                None,
            )
            hit_time = hit["decision_time_s"] if hit else None
        else:
            # HPA: no `decisions` log exists; detect an effective scale-up
            # directly from the replicas_current time series.
            hit_time = None
            prev = None
            for s2 in decision_samples:
                if s2["time_s"] < onset_time:
                    prev = s2
                    continue
                if prev is not None and s2["replicas_current"] > prev["replicas_current"]:
                    hit_time = s2["time_s"]
                    break
                prev = s2
        if hit_time is not None:
            reaction_times.append(hit_time - onset_time)
    reaction_time_seconds = (sum(reaction_times) / len(reaction_times)) if reaction_times else None

    # 5. action_churn_count: count of EFFECTIVE non-HOLD actions (diagnostic only).
    if decisions:
        action_churn_count = sum(1 for d in decisions if d["replicas_after"] != d["replicas_before"])
    else:
        action_churn_count = sum(
            1 for a, b in zip(decision_samples, decision_samples[1:]) if a["replicas_current"] != b["replicas_current"]
        )

    return {
        "sla_violation_rate": sla_violation_rate,
        "error_rate_mean": error_rate_mean,
        "resource_proxy_mean": resource_proxy_mean,
        "reaction_time_seconds": reaction_time_seconds,
        "n_sla_breach_onsets_with_reaction_captured": len(reaction_times),
        "action_churn_count": action_churn_count,
    }


def run_analysis() -> Dict[str, Any]:
    episodes = {name: _classify_legacy(_load(name)) for name in EPISODE_FILES}
    hpa2_attempt1 = json.load(open(EVAL_DIR / HPA2_ATTEMPT1_FAILURE_FILE))

    # --- Validation (fail closed on any mismatch) ---------------------
    hashes = {name: ep["workload_hash"] for name, ep in episodes.items()}
    if len(set(hashes.values())) != 1 or next(iter(hashes.values())) != EXPECTED_HOLDOUT_HASH:
        raise ProtocolMismatchError(f"holdout hash mismatch across episodes: {hashes}")
    seeds = {name: ep["workload_seed"] for name, ep in episodes.items()}
    if any(s != EXPECTED_SEED for s in seeds.values()):
        raise ProtocolMismatchError(f"workload seed mismatch: {seeds}")

    full_duration_valid = {name: ep for name, ep in episodes.items() if ep["valid_for_primary_comparison"]}
    hpa_valid = sorted(int(n[3]) for n in full_duration_valid if n.startswith("HPA"))
    ppo_valid = sorted(int(n[3]) for n in full_duration_valid if n.startswith("PPO"))
    if hpa_valid != [1, 2, 3]:
        raise ProtocolMismatchError(f"expected HPA valid repeats [1,2,3], got {hpa_valid}")
    if ppo_valid != [3]:
        raise ProtocolMismatchError(f"expected PPO valid full repeat [3] only, got {ppo_valid}")

    metrics = {name: compute_section10_metrics(ep) for name, ep in episodes.items()}

    safety_terminated = [name for name, ep in episodes.items() if ep["episode_status"] == "terminated_safety"]
    if sorted(safety_terminated) != ["PPO1", "PPO2"]:
        raise ProtocolMismatchError(f"expected safety-terminated = [PPO1, PPO2], got {sorted(safety_terminated)}")

    formal_n3_applicable = len(ppo_valid) == 3 and len(hpa_valid) == 3

    return {
        "episodes": episodes,
        "metrics": metrics,
        "hpa2_attempt1_failure": hpa2_attempt1,
        "hpa_valid_repeats": hpa_valid,
        "ppo_valid_repeats": ppo_valid,
        "formal_n3_applicable": formal_n3_applicable,
        "holdout_hash": EXPECTED_HOLDOUT_HASH,
        "workload_seed": EXPECTED_SEED,
    }


def write_metrics_json(result: Dict[str, Any], path: Path) -> None:
    out = {
        "classification": "OFFLINE_FINAL_EVALUATION_ANALYSIS",
        "holdout_hash": result["holdout_hash"],
        "workload_seed": result["workload_seed"],
        "formal_n3_rule_applicable": result["formal_n3_applicable"],
        "hpa_valid_full_repeats": result["hpa_valid_repeats"],
        "ppo_valid_full_repeats": result["ppo_valid_repeats"],
        "per_episode": {
            name: {
                "episode_status": ep["episode_status"],
                "termination_reason": ep.get("termination_reason"),
                "full_holdout_completed": ep["full_holdout_completed"],
                "valid_for_primary_comparison": ep["valid_for_primary_comparison"],
                "observed_decision_steps": ep["observed_decision_steps"],
                "nominal_decision_steps": ep["nominal_decision_steps"],
                "mean_replicas": ep["mean_replicas"],
                "max_replicas": ep["max_replicas"],
                "mean_latency_p95_ms": ep["mean_latency_p95_ms"],
                "section10_metrics": result["metrics"][name],
            }
            for name, ep in result["episodes"].items()
        },
        "hpa2_physical_attempt1_tooling_failure": {
            "status": result["hpa2_attempt1_failure"]["status"],
            "traffic_completed": result["hpa2_attempt1_failure"]["traffic_completed"],
            "metrics_complete": result["hpa2_attempt1_failure"]["metrics_complete"],
            "valid_for_primary_comparison": result["hpa2_attempt1_failure"]["valid_for_primary_comparison"],
            "excluded_from_scientific_aggregation": True,
        },
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)


def write_table_csv(result: Dict[str, Any], path: Path) -> None:
    fieldnames = [
        "episode", "controller", "repeat", "episode_status", "valid_for_primary_comparison",
        "observed_decision_steps", "nominal_decision_steps", "mean_replicas", "max_replicas",
        "mean_latency_p95_ms", "sla_violation_rate", "error_rate_mean", "resource_proxy_mean",
        "reaction_time_seconds", "action_churn_count",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, ep in result["episodes"].items():
            m = result["metrics"][name]
            writer.writerow({
                "episode": name, "controller": ep["controller"], "repeat": ep["repeat_index"],
                "episode_status": ep["episode_status"], "valid_for_primary_comparison": ep["valid_for_primary_comparison"],
                "observed_decision_steps": ep["observed_decision_steps"], "nominal_decision_steps": ep["nominal_decision_steps"],
                "mean_replicas": ep["mean_replicas"], "max_replicas": ep["max_replicas"],
                "mean_latency_p95_ms": ep["mean_latency_p95_ms"],
                "sla_violation_rate": m["sla_violation_rate"], "error_rate_mean": m["error_rate_mean"],
                "resource_proxy_mean": m["resource_proxy_mean"], "reaction_time_seconds": m["reaction_time_seconds"],
                "action_churn_count": m["action_churn_count"],
            })


def write_analysis_md(result: Dict[str, Any], path: Path) -> None:
    eps = result["episodes"]
    m = result["metrics"]

    def row(name):
        e, mm = eps[name], m[name]
        return (f"| {name} | {e['controller']} | {e['repeat_index']} | {e['episode_status']} | "
                f"{e['observed_decision_steps']}/{e['nominal_decision_steps']} | "
                f"{e['mean_replicas']:.2f} | {e['max_replicas']} | {e['mean_latency_p95_ms']:.1f} | "
                f"{mm['sla_violation_rate']:.3f} | {mm['resource_proxy_mean']:.3f} |")

    full_duration_rows = "\n".join(row(n) for n in ["HPA1", "HPA2", "HPA3", "PPO3"])
    safety_rows = "\n".join(row(n) for n in ["PPO1", "PPO2"])
    f1 = result["hpa2_attempt1_failure"]

    content = f"""# Index39 — Final Evaluation Analysis

OFFLINE analysis only. No live execution occurred to produce this document.

## 1. Training completion

Frozen at `session_019`: 5120 cumulative timesteps, 20 PPO updates, training seed 42.
`session_020` does not exist (verified).

## 2. Final evaluation protocol

Formal sequence executed: PPO1 → HPA1 → PPO2 → HPA2 → PPO3 → HPA3, replaying the
identical hash-pinned final holdout workload (`{result['holdout_hash']}`, seed {result['workload_seed']}, 1500s)
for every episode, per `phase_16u_direct_ppo_protocol.yaml` Section 10 /
`phase_16u_training_and_evaluation_workload_split.json`.

## 3. Episode outcome classification

| Episode | Controller | Repeat | Status | Steps (obs/nom) | Mean Replicas | Max Replicas | Mean P95 (ms) | SLA Viol. Rate | Resource Proxy |
|---|---|---|---|---|---|---|---|---|---|
{full_duration_rows}
{safety_rows}

## 4. Frozen Section-10 metrics

Computed exactly per the frozen definitions (`sla_violation_rate`, `error_rate_mean`,
`resource_proxy_mean`, `reaction_time_seconds`, `action_churn_count`) — see
`task_outputs/index39_final_evaluation_metrics.json` for full per-episode values,
including partial-duration PPO1/PPO2 (reported for completeness, never mixed into
full-duration aggregates).

## 5. Valid full-duration comparison table (HPA1, HPA2, HPA3, PPO3 only)

{full_duration_rows}

## 6. Safety outcomes

{safety_rows}

Both PPO1 and PPO2 terminated via the frozen `consecutive_invalid_steps_limit_reached`
rule (4 consecutive decision steps with Prometheus target unreachable) — a legitimate,
pre-registered safety outcome, not an implementation defect.

## 7. Infrastructure failure provenance

HPA repeat2, physical attempt1: status=`{f1['status']}`. Traffic generator completed its
full {result['episodes']['HPA2']['workload_duration_seconds']:.0f}s run
(`traffic_completed={f1['traffic_completed']}`), but metrics finalization failed
(`metrics_complete={f1['metrics_complete']}`) during post-workload Prometheus drain,
concurrent with observed `NodeNotReady` events. Preserved at
`persistent-data/evaluation/final_session019/{HPA2_ATTEMPT1_FAILURE_FILE}`.
**Excluded from scientific aggregation** — not a scientific HPA result.
HPA repeat2 physical attempt2 (after infrastructure recovery) completed the full
duration and is the scientific HPA2 result used throughout this analysis.

## 8. Descriptive resource-vs-latency findings

Across the three valid full-duration HPA repeats, mean replicas ranged
{min(eps[n]['mean_replicas'] for n in ['HPA1','HPA2','HPA3']):.2f}–{max(eps[n]['mean_replicas'] for n in ['HPA1','HPA2','HPA3']):.2f}
(peaking at {max(eps[n]['max_replicas'] for n in ['HPA1','HPA2','HPA3'])} replicas), with mean P95 latency
{min(eps[n]['mean_latency_p95_ms'] for n in ['HPA1','HPA2','HPA3']):.0f}–{max(eps[n]['mean_latency_p95_ms'] for n in ['HPA1','HPA2','HPA3']):.0f} ms.

The single valid PPO full-duration repeat (PPO3) used substantially fewer replicas
(mean {eps['PPO3']['mean_replicas']:.2f}, max {eps['PPO3']['max_replicas']}) than any HPA repeat, but recorded a
higher mean P95 latency ({eps['PPO3']['mean_latency_p95_ms']:.0f} ms) than all three HPA repeats. This is a
**single-repeat observation**, not a statistically supported trend.

PPO's two safety-terminated repeats (PPO1, PPO2) are themselves evidence of a
robustness limitation under this specific held-out workload: the policy (or the
surrounding live infrastructure under PPO's control pattern) encountered sustained
telemetry/target unavailability severe enough to trigger the frozen safety
termination rule in 2 of 3 attempts, while HPA's control pattern did not trigger
the equivalent failure mode in any of its 3 valid attempts (one HPA attempt did
fail, but from infrastructure/tooling causes unrelated to HPA's own control
actions, not a safety-rule termination).

## 9. Formal decision-rule status

**NOT APPLICABLE.** The frozen Section 10 decision rule requires n=3 valid
full-duration repeats per controller. HPA achieved 3/3; PPO achieved 1/3.

## 10. Limitations

- PPO's valid full-duration sample size (n=1) cannot support the frozen
  decision rule, any inferential statistic, or a directional conclusion.
- PPO1/PPO2's partial-duration data (24/50 and 20/50 decision steps
  respectively) are reported for transparency but are not comparable to a
  full 1500s exposure and are excluded from the primary comparison.
- HPA2 physical attempt1 is excluded entirely (no scientific result).
- Whether PPO's two safety terminations reflect the learned policy's
  behavior, the live infrastructure's independent instability during this
  evaluation window, or an interaction between the two cannot be
  disentangled from this evidence alone.

## 11. Final scientifically defensible conclusion

See `task_outputs/index39_final_scientific_conclusion.md`.
"""
    with open(path, "w") as f:
        f.write(content)


def write_conclusion_md(result: Dict[str, Any], path: Path) -> None:
    content = """# Index39 — Final Scientific Conclusion

The predefined three-repeat PPO-vs-HPA primary comparison could not be
completed as designed because two of the three PPO evaluation repeats
terminated under the frozen safety rule. HPA completed all three held-out
repeats, whereas PPO completed one full repeat. Therefore, the frozen n=3
decision rule is not applicable, and no formal comparative winner is
declared.

## What can be said descriptively

- HPA completed 3/3 held-out repeats of the final evaluation workload.
- PPO completed 1/3 full repeats; the other two terminated early under the
  frozen, pre-registered safety rule (4 consecutive decision steps with the
  Prometheus/application target unreachable).
- In its one full-duration repeat, PPO used markedly fewer replicas
  (mean ~1.20, max 4) than HPA did in any of its three repeats (mean
  ~5.06–5.35, max 7–8).
- HPA achieved lower mean P95 latency in all three of its repeats than PPO
  did in its one full repeat.
- PPO's own safety terminations are themselves a relevant finding: under
  this specific held-out workload and live-cluster conditions, the
  evaluation could not sustain two of three attempts to full duration,
  which is evidence of a robustness limitation in this configuration,
  independent of any latency/resource comparison.

## What cannot be said

- No claim of statistical significance is made or supported (sample sizes
  of n=1 and n=3 do not support one, and the frozen protocol itself
  disclaims inferential testing even at the originally intended n=3).
- No formal "PPO wins," "HPA wins," or "mixed/no-clear-winner under the
  n=3 rule" conclusion is drawn, because the required comparable sample
  count (3 valid full-duration repeats per controller) was not achieved
  for PPO.
- This is not evidence that PPO is categorically worse or better than HPA
  in general -- it reflects this one evaluation run, under this one
  held-out workload, on this one trained checkpoint (`session_019`,
  5120 timesteps), with this specific live-infrastructure behavior during
  the evaluation window.

## Formal status

**NO FORMAL WINNER DECLARED.** The comparison is reported as incomplete
under its own pre-registered decision rule.
"""
    with open(path, "w") as f:
        f.write(content)


def main() -> int:
    result = run_analysis()
    TASK_OUTPUTS.mkdir(parents=True, exist_ok=True)
    write_metrics_json(result, TASK_OUTPUTS / "index39_final_evaluation_metrics.json")
    write_table_csv(result, TASK_OUTPUTS / "index39_final_evaluation_table.csv")
    write_analysis_md(result, TASK_OUTPUTS / "index39_final_evaluation_analysis.md")
    write_conclusion_md(result, TASK_OUTPUTS / "index39_final_scientific_conclusion.md")
    print(json.dumps({
        "hpa_valid_full_repeats": result["hpa_valid_repeats"],
        "ppo_valid_full_repeats": result["ppo_valid_repeats"],
        "formal_n3_rule_applicable": result["formal_n3_applicable"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
