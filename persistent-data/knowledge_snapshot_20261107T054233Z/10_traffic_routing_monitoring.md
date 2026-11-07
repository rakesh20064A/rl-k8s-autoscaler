# Part 10 — Traffic, Routing, and Monitoring Clients

FROM_CODE unless noted.

## `loadgen/traffic_generator.py` CLI

`--profile` (required), `--experiment-id`/`--run-id` (required),
`--target-url` (default `http://localhost:8080`), `--arrival-process`
(poisson/deterministic/even, default poisson), `--seed` (default None),
`--max-in-flight` (default 50), `--timeout` (default 5.0), `--config`
(default `configs/experiment.yaml`), `--connection-mode`
(keepalive/fresh, default keepalive — `fresh` opens a new TCP connection
per request, required for genuine Service load-balancing per CORR-21),
`--replay-schedule` (default None).

Outcome classes (confirmed from `traffic_generator.py:699-705`): exactly
6 canonical — `success_2xx`, `client_error_4xx`, `server_error_5xx`,
`timeout`, `connection_error`, `dropped_overload` (a 7th de facto
`f"http_{status_code}"` fallback exists for 1xx/3xx responses, not in the
canonical tuple). `dropped_overload` = requests dropped by the generator's
own in-flight concurrency cap before ever being dispatched (never reaches
the network).

## `scripts/traffic_orchestrator.py` — V5 route (the validated mechanism)

`V5TransportManager.establish_route()`: resolves the Minikube Docker
network (`docker inspect minikube`), the node IP (`minikube -p minikube
ip`), and the `demo-app-loadtest` Service's NodePort (`kubectl get svc
... -o jsonpath={.spec.ports[0].nodePort}`); launches a relay container
`docker run -d --rm --name rl-traffic-relay-<run_id> --network <net> -p
8080 alpine/socat TCP-LISTEN:8080,fork,reuseaddr TCP:<node_ip>:<nodeport>`,
publishing the relay's port 8080 to a Docker-assigned EPHEMERAL host port
(never hard-coded). Relay health: `docker inspect --format
{{.State.Running}}|{{.State.ExitCode}}` — independent of any TCP probing.
Route verification: `GET /health` (never `/work`) before trusting the
route. Teardown: `docker rm -f <container>`. Episode synchronization:
`TrafficSyncedWrapper` starts traffic only after the canonical
reset+stabilization completes and stops it at episode end — never "somewhere
in parallel" for a whole multi-episode session.

**Confirmed route**: `Mac localhost:<ephemeral>` → Docker bridge → socat
relay (same Docker network as Minikube) → `TCP:<node_ip>:<nodeport>` →
kube-proxy Service routing (genuine load-balancing across Ready endpoints)
→ a pod. This has been used throughout every real pilot/final-campaign
session without the tunnel-breaking artifact seen in the ad-hoc R4C/R4D
`minikube service --url`/`kubectl port-forward` experiments (neither of
which is this project's own validated mechanism).

## Prometheus URL consistency

`PROMETHEUS_URL = "http://localhost:9090"` is the SINGLE authoritative
constant (`scripts/v2_pilot_safety.py:52`), explicitly injected into
`LiveKubernetesBackend`, `V2TrafficOrchestrator`'s drain query, and the
safety callback's own monitoring — all confirmed identical. This was the
exact root cause of V2 Phase 3A's attempt-1 failure (a stray
`localhost:19090` port-forward vs. every internal client's unmodified
9090 default) and has been a single-source-of-truth constant since Phase
3B. `collectors/metrics_collector.py` and the demo UI are not confirmed to
share this exact constant by direct citation in this pass (not
re-verified this session; UI never references Prometheus directly per
Part 13).

## Per-pod request attribution — confirmed absent, project-wide

No field anywhere (app response body, `CompletedRequest` schema in
`traffic_generator.py`, or any Prometheus metric label actually consumed
by the pipeline) ties an individual HTTP request to the specific pod that
served it. This gap has been independently re-confirmed in Phase 4B-R4C
(forensic audit), 4B-R4F (workload audit), and now again this session —
it is a structural, repo-wide absence, not specific to any one test.
Per-pod resource usage (CPU/memory) IS queryable via Prometheus
(`container_cpu_usage_seconds_total{pod=...}` etc., confirmed live-working
in Phase 4B-R4E) — only REQUEST-level attribution is missing.
