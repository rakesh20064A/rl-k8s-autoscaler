#!/usr/bin/env python3
"""scripts/compile_stage1_calibration.py

Compile all valid empirical single-replica calibration levels (1–6 RPS)
using the canonical summary methodology and print both structured JSON and a canonical Markdown table.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

try:
    from scripts.summarize_calibration_level import analyze_level
except ModuleNotFoundError:
    from summarize_calibration_level import analyze_level

EXPERIMENT_IDS = [
    ("1 RPS", 1.0, "cal_single_r1_i3_20260906T082243Z"),
    ("2 RPS", 2.0, "cal_single_r2_i3_20260906T082448Z"),
    ("3 RPS", 3.0, "cal_single_r3_i3_20260906T082941Z"),
    ("4 RPS", 4.0, "cal_single_r4_i3_20260906T083152Z"),
    ("5 RPS", 5.0, "cal_single_r5_i3_20260906T083402Z"),
    ("6 RPS", 6.0, "cal_single_r6_i3_20260906T084322Z"),
    ("8 RPS", 8.0, "cal_single_r8_i3_20260906T090054Z"),
    ("9 RPS", 9.0, "cal_single_r9_i3_20260906T091609Z"),
    ("10 RPS", 10.0, "cal_single_r10_i3_20260906T090800Z"),
]


def classify_level(res: Dict[str, Any]) -> str:
    """Classify calibration level using canonical multi-signal criteria:

    - Underloaded: Achieved load matches offered, 0 errors/timeouts/drops, p95 < 300ms SLA, healthy CPU headroom.
    - Approaching saturation: Sustained p95 degradation toward/above ~300ms SLA, persistent queueing, or reduced CPU headroom (steady CPU >= 75% of limit).
    - Saturated: Sustained SLA violation, drops, timeouts, or throughput collapse.
    """
    t = res["traffic"]
    s = res["prometheus_steady_window"]

    # Safety checks
    if (
        t.get("success_rate", 1.0) < 0.98
        or t.get("timeouts", 0) > 0
        or t.get("dropped_overload", 0) > 0
        or t.get("conn_errors", 0) > 0
    ):
        return "Saturated"

    client_p95_ms = t["client_latency"]["p95_seconds"] * 1000.0
    server_p95_ms = (s.get("server_p95_seconds") or 0.0) * 1000.0
    cpu_pct = s.get("cpu_limit_utilization_pct") or 0.0

    # Approaching saturation: p95 nearing or exceeding 300ms SLA, or CPU steady usage >= 75% of limit (375m)
    if client_p95_ms >= 270.0 or server_p95_ms >= 285.0 or cpu_pct >= 75.0:
        return "Approaching Saturation"

    return "Underloaded"


def generate_markdown_table(results: List[Dict[str, Any]]) -> str:
    """Generate canonical markdown summary table."""
    lines = [
        "| Offered RPS | Dispatch Rate (Workload Window) | Success Rate | Client p50 | Client p95 | Client p99 | Server Steady p95 | Steady Mean CPU Rate | Max Observed CPU-Rate Sample | 4xx | 5xx | Timeouts | Connection Errors | Generator Drops | Classification |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in results:
        label = r["label"]
        t = r["traffic"]
        s = r["prometheus_steady_window"]
        w = r["prometheus_whole_run"]
        cls = r["classification"]

        dispatch_rate = f"{t.get('dispatch_rate_workload_window', t.get('achieved_send_rps', 0.0)):.2f} RPS"
        success_pct = f"{t['success_rate'] * 100:.1f}%"
        c_p50 = f"{t['client_latency']['p50_seconds'] * 1000:.1f} ms"
        c_p95 = f"{t['client_latency']['p95_seconds'] * 1000:.1f} ms"
        c_p99 = f"{t['client_latency']['p99_seconds'] * 1000:.1f} ms"
        s_p95 = f"{s['server_p95_seconds'] * 1000:.1f} ms" if s["server_p95_seconds"] else "N/A"
        steady_cpu = f"{s['mean_observed_cpu_rate']:.4f} cores ({s['cpu_limit_utilization_pct']:.1f}%)"
        max_cpu = f"{w['max_observed_cpu_rate']:.4f} cores"
        c_4xx = str(t.get("client_4xx", 0))
        s_5xx = str(t.get("server_5xx", 0))
        timeouts = str(t.get("timeouts", 0))
        conn_errors = str(t.get("conn_errors", 0))
        drops = str(t.get("dropped_overload", 0))

        lines.append(
            f"| **{label}** | {dispatch_rate} | {success_pct} | {c_p50} | {c_p95} | {c_p99} | {s_p95} | {steady_cpu} | {max_cpu} | {c_4xx} | {s_5xx} | {timeouts} | {conn_errors} | {drops} | **{cls}** |"
        )
    return "\n".join(lines)


def main():
    root = Path(__file__).resolve().parent.parent
    results = []

    for label, target_rps, exp_id in EXPERIMENT_IDS:
        data = analyze_level(exp_id, project_root=root)
        data["label"] = label
        data["offered_rps"] = target_rps
        data["classification"] = classify_level(data)
        results.append(data)

    if "--json-only" in sys.argv:
        print(json.dumps(results, indent=2))
        return

    md_table = generate_markdown_table(results)
    print("### Canonical Stage-1 Single-Replica Capacity Calibration Summary (1–10 RPS)\n")
    print(md_table)
    print("\n*Notes on Canonical Methodology:*")
    print("1. Client Latencies: Evaluated over all requests in the 90.0s authoritative workload window using standard linear interpolation.")
    print("2. Server Steady p95 & Steady Mean CPU: Evaluated strictly over stabilized Prometheus samples (elapsed_seconds >= 60.0, 1m lookback fully populated).")
    print("3. Max Observed CPU-Rate Sample: Peak 1-minute windowed CPU rate sample observed across the entire run.")
    print("4. Container resource constraints: CPU request 100m, CPU limit 500m (0.5 cores), memory request 64Mi, memory limit 256Mi.")


if __name__ == "__main__":
    main()
