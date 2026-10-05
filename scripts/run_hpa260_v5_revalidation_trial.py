"""scripts/run_hpa260_v5_revalidation_trial.py

Phase 15B.3N.4 (Gate A): NON-OFFICIAL revalidation of the frozen HPA-260%
target under the V5 Docker-network-relay transport (CORR-40), using the
EXACT historical HPA-tuning workload (schedule, profile, generator
invocation) reconstructed from `scripts/run_hpa_tuning_trial.sh` and
`experiments/manifests/hpa_tuning_official_9_trials.yaml` -- never modifying
those historical artifacts.

This is NOT an official calibration attempt: it never touches the campaign
ledger, is never counted as calibration/fitting support, and every output
directory is prefixed `NON_OFFICIAL_PROTOCOL_TRANSITION_VALIDATION_`.

Deliberately narrow, mirroring run_hpa_tuning_trial.sh's own stated scope:
does not know about RL, does not run all trials automatically (call once
per repeat), does not choose a winner, does not alter the live HPA target
or hpa.yaml. The ONLY change from the historical protocol is the transport
mechanism (V5 Docker relay -> NodePort, instead of `minikube service --url`
SSH tunnel -> ClusterIP) -- everything else (schedule bytes, profile,
generator flags, HPA config, collector cadence) is reproduced exactly.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_dynamics_calibration_slot import (  # noqa: E402
    APP_NS,
    HPA_NAME,
    MONITORING_NS,
    VENV_PYTHON,
    _run,
    check_local_listener_present,
    check_relay_container_state,
    resolve_local_listener_owner,
    start_loadtest_transport,
)

CANONICAL_PROFILE = PROJECT_ROOT / "configs/traffic_profiles/hpa_tuning/hpa_tuning_v2.yaml"
CANONICAL_SCHEDULE = PROJECT_ROOT / "configs/traffic_profiles/hpa_tuning/canonical_schedule.csv"
EXPECTED_SCHEDULE_SHA256 = "4f648411e039f9229cd35a06374e6870dc75eca67e385e67f4f52bb0b5ec55ad"

OUTPUT_ROOT = PROJECT_ROOT / "persistent-data/transport_validation/protocol_v5_transition/hpa260_revalidation"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_of(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def kubectl_jsonpath(args) -> str:
    proc = _run(["kubectl"] + args, timeout=15)
    return proc.stdout.strip()


def preflight() -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    checks["docker_ok"] = _run(["docker", "info"], timeout=15).returncode == 0
    mk = _run(["minikube", "status", "-p", "minikube"], timeout=15)
    checks["minikube_running"] = "host: Running" in mk.stdout
    checks["hpa_target"] = kubectl_jsonpath(["get", "hpa", HPA_NAME, "-n", APP_NS, "-o", "jsonpath={.spec.metrics[0].resource.target.averageUtilization}"])
    checks["hpa_min"] = kubectl_jsonpath(["get", "hpa", HPA_NAME, "-n", APP_NS, "-o", "jsonpath={.spec.minReplicas}"])
    checks["hpa_max"] = kubectl_jsonpath(["get", "hpa", HPA_NAME, "-n", APP_NS, "-o", "jsonpath={.spec.maxReplicas}"])
    checks["scaleup_stab"] = kubectl_jsonpath(["get", "hpa", HPA_NAME, "-n", APP_NS, "-o", "jsonpath={.spec.behavior.scaleUp.stabilizationWindowSeconds}"])
    checks["scaledown_stab"] = kubectl_jsonpath(["get", "hpa", HPA_NAME, "-n", APP_NS, "-o", "jsonpath={.spec.behavior.scaleDown.stabilizationWindowSeconds}"])
    checks["cpu_req"] = kubectl_jsonpath(["get", "deployment", "demo-app", "-n", APP_NS, "-o", "jsonpath={.spec.template.spec.containers[0].resources.requests.cpu}"])
    checks["cpu_lim"] = kubectl_jsonpath(["get", "deployment", "demo-app", "-n", APP_NS, "-o", "jsonpath={.spec.template.spec.containers[0].resources.limits.cpu}"])
    checks["desired"] = kubectl_jsonpath(["get", "deployment", "demo-app", "-n", APP_NS, "-o", "jsonpath={.spec.replicas}"])
    checks["current"] = kubectl_jsonpath(["get", "deployment", "demo-app", "-n", APP_NS, "-o", "jsonpath={.status.replicas}"])
    checks["ready"] = kubectl_jsonpath(["get", "deployment", "demo-app", "-n", APP_NS, "-o", "jsonpath={.status.readyReplicas}"])
    checks["schedule_sha256"] = sha256_of(CANONICAL_SCHEDULE)
    checks["schedule_sha256_matches_expected"] = checks["schedule_sha256"] == EXPECTED_SCHEDULE_SHA256
    ok = (
        checks["docker_ok"] and checks["minikube_running"]
        and checks["hpa_target"] == "260" and checks["hpa_min"] == "1" and checks["hpa_max"] == "8"
        and checks["scaleup_stab"] == "0" and checks["scaledown_stab"] == "120"
        and checks["cpu_req"] == "100m" and checks["cpu_lim"] == "500m"
        and checks["desired"] == "1" and checks["current"] == "1" and checks["ready"] == "1"
        and checks["schedule_sha256_matches_expected"]
    )
    checks["preflight_ok"] = ok
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", required=True, type=int)
    args = parser.parse_args()

    experiment_id = f"NON_OFFICIAL_PROTOCOL_TRANSITION_VALIDATION_hpa260_v5_repeat{args.repeat}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = OUTPUT_ROOT / experiment_id
    out_dir.mkdir(parents=True, exist_ok=False)

    result: Dict[str, Any] = {
        "scientific_role": "NON_OFFICIAL_PROTOCOL_TRANSITION_VALIDATION",
        "formal_calibration_eligible": False,
        "campaign_ledger_eligible": False,
        "model_fitting_eligible": False,
        "experiment_id": experiment_id,
        "repeat": args.repeat,
        "gate": "A_hpa260_v5_revalidation",
        "profile": str(CANONICAL_PROFILE.relative_to(PROJECT_ROOT)),
        "schedule": str(CANONICAL_SCHEDULE.relative_to(PROJECT_ROOT)),
        "transport": "docker_network_tcp_relay_v5",
    }

    print("=== Preflight ===")
    checks = preflight()
    result["preflight"] = checks
    if not checks["preflight_ok"]:
        result["status"] = "invalid"
        result["invalid_reason"] = "preflight_failed"
        (out_dir / "trial_metadata.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(checks, indent=2))
        return 2
    print(f"  Preflight OK. Schedule SHA-256 verified: {checks['schedule_sha256']}")

    print("=== V5 load-generator transport (Docker-network raw TCP relay) ===")
    transport = start_loadtest_transport(experiment_id)
    result["transport_lifecycle"] = transport
    if "error" in transport:
        result["status"] = "invalid"
        result["invalid_reason"] = f"wrong_traffic_path_used: {transport['error']}"
        (out_dir / "trial_metadata.json").write_text(json.dumps(result, indent=2))
        return 2
    target_url = transport["target_url"]
    relay_name = transport["container_name"]
    print(f"  Relay established: {target_url} (container {relay_name})")

    prom_ready = _run(["curl", "-s", "-f", "http://localhost:9090/-/ready"], timeout=5).returncode == 0
    started_prom = False
    prom_proc: Optional[subprocess.Popen] = None
    if not prom_ready:
        prom_proc = subprocess.Popen(
            ["kubectl", "port-forward", "-n", MONITORING_NS, "svc/kube-prom-stack-kube-prome-prometheus", "9090:9090"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        started_prom = True
        time.sleep(3)

    total_duration = int(
        subprocess.run(
            [VENV_PYTHON, "-c", f"from loadgen.traffic_generator import parse_traffic_profile; from pathlib import Path; print(int(parse_traffic_profile(Path('{CANONICAL_PROFILE}')).total_duration))"],
            capture_output=True, text=True,
        ).stdout.strip()
    )
    collector_duration = total_duration + 30
    print(f"Profile total duration: {total_duration}s -> collectors run for {collector_duration}s")

    print("=== Starting collectors ===")
    metrics_proc = subprocess.Popen(
        [VENV_PYTHON, "-m", "collectors.metrics_collector", "--experiment-id", experiment_id, "--mode", "test",
         "--traffic-profile", CANONICAL_PROFILE.stem, "--duration-seconds", str(collector_duration), "--seed", "42"],
    )
    hpa_proc = subprocess.Popen(
        [VENV_PYTHON, "-m", "collectors.hpa_status_collector", "--experiment-id", experiment_id,
         "--hpa-name", HPA_NAME, "--namespace", APP_NS, "--sampling-interval-seconds", "5",
         "--duration-seconds", str(collector_duration)],
    )
    time.sleep(1)

    # Passive V5 transport observer, same ~10s cadence as CORR-39, for the
    # duration of the (long) HPA-tuning traffic replay.
    transport_samples = []
    import threading

    stop_sampling = threading.Event()
    tunnel_port = int(target_url.rsplit(":", 1)[1])

    def _sample():
        start_mono = time.monotonic()
        while not stop_sampling.wait(10):
            running, exit_code = check_relay_container_state(relay_name)
            listener = check_local_listener_present(tunnel_port)
            transport_samples.append({
                "elapsed_seconds_since_traffic_start": round(time.monotonic() - start_mono, 1),
                "sampled_at_utc": utc_now(),
                "relay_container_running": running,
                "relay_container_exit_code": exit_code,
                "host_listener_present": listener,
            })

    sampler = threading.Thread(target=_sample, daemon=True)
    sampler.start()

    print("=== Starting traffic generator (schedule REPLAY, historical HPA-tuning invocation, no timeout/max-in-flight override) ===")
    traffic_start_epoch = time.time()
    gen = subprocess.run(
        [VENV_PYTHON, "-m", "loadgen.traffic_generator",
         "--profile", str(CANONICAL_PROFILE), "--experiment-id", experiment_id,
         "--target-url", target_url, "--replay-schedule", str(CANONICAL_SCHEDULE),
         "--arrival-process", "deterministic", "--connection-mode", "fresh"],
    )
    traffic_end_epoch = time.time()
    stop_sampling.set()
    sampler.join(timeout=5)

    result["transport_lifecycle"]["periodic_samples_during_traffic"] = transport_samples
    result["transport_lifecycle"]["traffic_start_epoch"] = traffic_start_epoch
    result["transport_lifecycle"]["traffic_end_epoch"] = traffic_end_epoch
    running_at_end, exit_code_at_end = check_relay_container_state(relay_name)
    result["transport_lifecycle"]["running_at_traffic_end"] = running_at_end
    result["transport_lifecycle"]["host_listener_present_at_traffic_end"] = check_local_listener_present(tunnel_port)
    result["transport_lifecycle"]["listener_owner_at_traffic_end"] = resolve_local_listener_owner(tunnel_port)

    if gen.returncode != 0:
        result["status"] = "invalid"
        result["invalid_reason"] = f"traffic_generator_exit_{gen.returncode}"

    print("=== Waiting for collectors to finalize ===")
    metrics_exit = metrics_proc.wait(timeout=collector_duration + 60)
    hpa_exit = hpa_proc.wait(timeout=collector_duration + 60)
    if metrics_exit != 0:
        result.setdefault("status", "invalid")
        result["invalid_reason"] = result.get("invalid_reason", "") + f" metrics_collector_exit_{metrics_exit}"
    if hpa_exit != 0:
        result.setdefault("status", "invalid")
        result["invalid_reason"] = result.get("invalid_reason", "") + f" hpa_status_collector_exit_{hpa_exit}"

    print("=== Cleanup: relay + Prometheus port-forward ===")
    logs = _run(["docker", "logs", relay_name], timeout=15)
    (out_dir / "relay_docker_logs.txt").write_text((logs.stdout or "") + (logs.stderr or ""))
    _run(["docker", "rm", "-f", relay_name], timeout=15)
    if started_prom and prom_proc is not None:
        prom_proc.terminate()

    result.setdefault("status", "valid")
    result["traffic_dir"] = f"persistent-data/traffic/{experiment_id}"
    result["metrics_dir"] = f"persistent-data/metrics/{experiment_id}"
    result["hpa_dir"] = f"persistent-data/hpa/{experiment_id}"
    result["completed_at_utc"] = utc_now()

    (out_dir / "trial_metadata.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({"status": result["status"], "experiment_id": experiment_id}, indent=2))
    return 0 if result["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
