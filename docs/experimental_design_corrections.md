# Experimental Design Corrections
## Standing technical requirements for valid HPA-vs-RL benchmarking

---

## CORR-1 — Minikube Resource-Accounting Mismatch

### What is verified

Docker enforces the Minikube outer-container limits via cgroup:
- **CPU**: `cpu.max = 200000/100000` → **2.0 CPUs** hard enforced
- **Memory**: `memory.max = 3,221,225,472 B` → **3072 MB** hard enforced

However, Kubernetes currently reports:
- **Node Capacity / Allocatable**: ~4 CPU, ~3917 MB (kubelet reads the Docker VM's `/proc`, not the cgroup)

### Why this matters (not harmless)

Kubernetes scheduling uses Node Allocatable. The scheduler may place workloads
believing roughly 4 CPUs and 3917 MB are available. In practice, all pods share
the outer container's 2 CPU / 3072 MB cgroup budget.

Consequences:
- Over-scheduled pods will be CPU-throttled by the cgroup, not by Kubernetes.
- Memory-heavy workloads may trigger OOM inside the cgroup before Kubernetes eviction.
- CPU percentage metrics reported by `kubectl top nodes` are relative to 4 CPUs
  (the VM's capacity), not to the 2-CPU cgroup limit. A reading of "50% CPU"
  actually represents 100% of the usable cgroup budget.

### Required mitigations (apply to all phases)

1. **Use conservative pod resource requests/limits** — keep each pod's request
   well below 500m CPU and 512 Mi memory so the scheduler's placement decisions
   remain reasonable.

2. **Max replica count** — keep `scaling.max_replicas` at 8 (from experiment.yaml)
   but verify during profiling that 8 replicas × pod-request do not exceed
   the 2 CPU / 3072 MB physical budget.

3. **Do not use K8s node CPU percentage as a primary RL observation** — it is
   measured against 4 CPUs (VM capacity) rather than the 2-CPU cgroup. Use
   pod-level CPU (millicores), application request rate, p95 latency, error rate,
   memory, replica count, and temporal trends as the primary RL observation signals.

4. **Measure actual Minikube/Docker cgroup resource usage during each experiment**:
   - `docker stats minikube` for the outer container's real CPU and memory consumption
   - Record alongside pod metrics in persistent-data

5. **Invalidation criterion**: mark an experiment run invalid if the outer Minikube
   container becomes heavily CPU-throttled (sustained cgroup throttle > 30%) or
   memory-constrained (available < 200 MB inside cgroup) during the measurement window.

**Applies to phases**: 10 (Collector), 12 (HPA Baseline), 13 (HPA Experiment Runner),
19 (PPO Training), 22 (HPA vs RL Comparison), 23 (Final Reporting).

---

## CORR-2 — HPA Metric-Resolution Fairness for Benchmark

### Current state (development only)

Metrics Server is running with `--metric-resolution=60s`.
This setting is acceptable during development but must be reviewed before the
final HPA-vs-RL benchmark.

### Correct HPA control-loop characterisation

- **HPA control-loop frequency**: approximately every **15 seconds** (kube-controller-manager default).
- **Scale-up stabilization window**: **0 seconds** (no default delay; HPA can scale up immediately).
- **Scale-down stabilization window**: **300 seconds** (default; overridden in configs/hpa.yaml to 120 s).
- **Metrics Server scrape interval**: currently 60 s → HPA may operate on a metric reading
  up to 60 seconds stale.

### Why fairness requires explicit review

If Metrics Server remains at 60 s resolution while the RL agent receives 10-second
Prometheus observations, the RL agent has a significant information-freshness advantage.
Any superior RL performance in the final benchmark could be partly attributable to
fresher state observations, not policy quality.

### Required action before final benchmark (not now)

Before Phase 22 (HPA vs RL Comparison):
1. Set Metrics Server `--metric-resolution` to approximately **15 seconds** to give
   HPA access to fresher CPU measurements.
2. Set the RL agent's `control_interval_seconds` (experiment.yaml) to the same
   approximate cadence (15–30 s) so that neither controller has a meaningful
   information-freshness advantage.
3. Document the chosen intervals explicitly in the experiment report.
4. Re-run a validation experiment confirming HPA behaviour after resolution change.

**Do not make this change now** — leave Metrics Server at 60 s for development.

**Applies to phases**: 12 (HPA Baseline), 13 (HPA Experiment Runner),
22 (HPA vs RL Comparison), 23 (Final Reporting).

---

## CORR-3 — CPU Limits and maxReplicas (Phase 6A correction)

### Correction

Do not restrict `maxReplicas` to 3 merely because summed CPU limits at maximum scale
exceed the 2-CPU Minikube cgroup budget. **CPU limits are ceilings, not scheduler
reservations.** Kubernetes schedules based on `requests`; limits only cap individual
pod CPU burst. Multiple pods sharing the 2-CPU cgroup will compete for CPU under
load, but this is the expected and observable experimental condition — not a
configuration error.

### Required action

- Keep `maxReplicas` at 8 (unchanged from `experiment.yaml`).
- During HPA/RL experiments, monitor actual CPU saturation and throttling:
  - `docker stats minikube` (outer cgroup)
  - `kubectl top pods -n autoscaler-app`
  - Pod-level CPU throttle ratio (if available via cAdvisor/Prometheus)
- Decide whether to reduce `maxReplicas` only if experiments show sustained,
  severe throttling that distorts results — and only with explicit user approval.

**Applies to phases**: 12, 13, 22, 23.

---

## CORR-4 — RL Agent Metric Source

### Correction

The RL agent must obtain application metrics from **Prometheus**, not by scraping
the application's `/metrics` endpoint directly.

### Rationale

- Direct scraping bypasses the Prometheus time-series pipeline and its staleness/
  interpolation semantics.
- Prometheus provides consistent, rate-computed, and aggregated signals across
  all replica pods (important when replica count > 1).
- Direct scraping from multiple pods requires replica-aware aggregation logic in
  the RL agent itself — duplicating Prometheus's job.
- Using Prometheus as the single metrics source ensures the RL observation space
  reflects the same data that experiment reports and Grafana dashboards use.

### Required action (not yet — Prometheus not installed)

When implementing the RL environment (Phase 15 onwards):
- Query Prometheus HTTP API (`/api/v1/query`) for all RL state observations.
- Do not use `requests.get(pod_ip + "/metrics")` in the RL environment code.
- Prometheus URL is configured in `configs/experiment.yaml:prometheus.url`.

**Applies to phases**: 15 (RL Environment), 16–18 (State/Action/Reward),
19 (PPO Training), 20 (Shadow Mode), 21 (Active Controller).

---

## CORR-5 — Workload Generation Isolation (Phase 6C Diagnostic Run Invalidation)

### What occurred (Invalid Phase 6C Run)

During the initial Phase 6C test, the synthetic load generator script was executed using `kubectl exec` directly inside the `demo-app` container (`autoscaler-app` namespace).

### Status and Classification

- **Result**: Recorded as an **invalid/diagnostic run**.
- **Reason**: "Load generator executed inside target application container, contaminating target pod CPU measurements."

### Why this matters

When load generation (e.g., client threads, HTTP connection handling, serialization) executes inside the target application container, the container cgroup accounts for both the generator overhead and the server overhead. This confounds target pod CPU measurements and prevents isolating FastAPI server CPU consumption from client traffic generator CPU consumption.

### Required mitigation

All synthetic traffic must be generated from a separate container or separate host process:
- In-cluster testing: Deploy a separate temporary pod in the cluster communicating via Kubernetes Service DNS (`http://demo-app/work?...`).
- External testing: Use an external load generator (e.g. Locust) querying the Service via NodePort or ingress/port-forward.
- Never execute workload client processes within the target application container.

**Applies to phases**: 6C (Verification), 8 (Locust), 12 (HPA Baseline), 13 (HPA Experiment Runner), 22 (HPA vs RL Comparison).

---

## CORR-6 — Application Metric Label Collision with Prometheus Target Labels

### What occurred

The initial FastAPI application metric instrumentation used the label name `endpoint="/work"`. However, Prometheus ServiceMonitor target discovery automatically attaches a target label `endpoint` corresponding to the Kubernetes Service port name (`endpoint="http"`). When an ingested metric label collides with an existing target label and `honor_labels` is false, Prometheus renames the application metric label to `exported_<label>` (producing `exported_endpoint="/work"`).

### Why this matters

Relying on `exported_endpoint` produces fragile PromQL expressions across Grafana dashboards, metrics collectors, and RL environment observation queries. Conversely, setting `honor_labels: true` is risky because it allows scraped metrics to overwrite core Prometheus identification labels.

### Required mitigation

The application metric label was renamed cleanly from `endpoint` to `route`:
- `demo_app_requests_total{route="/work", status_code="200"}`
- `demo_app_request_duration_seconds{route="/work"}`
- `demo_app_requests_in_progress{route="/work"}`

All PromQL queries in collectors, dashboards, and the RL environment must query using `{route="/work"}`. Never use `exported_endpoint`.

**Applies to phases**: 7E (Instrumentation), 8 (Traffic), 9 (Grafana Dashboards), 10 (Collector), 15 (RL Environment), 22 (HPA vs RL Benchmark).

---

## CORR-7 — Status of `demo_app_last_work_intensity` Diagnostic Gauge

### What occurred

The application exposes a Prometheus gauge: `demo_app_last_work_intensity`, recording the `intensity` query parameter of the most recently processed `/work` request on each pod.

### Status and Classification

- **Role**: Diagnostic / visualization application gauge only.
- **Limitation**: In a multi-replica deployment, different pods process requests with different intensities at different times. Aggregating or reading this pod-level gauge across replicas can produce stale, conflicting, or ambiguous readings.
- **Authoritative Source**: `demo_app_last_work_intensity` must NOT be treated as the authoritative traffic-intensity source or used as an RL state observation variable. The external traffic generator / experiment runner logs and configuration files serve as the ground truth for scheduled and active workload intensity.

**Applies to phases**: 8A/B/C (Dashboards), 10 (Metrics Collector), 15 (RL Observation Space), 22 (Benchmark Analysis).

---

## CORR-8 — Host Resource Isolation, Pod Lifecycle Resilience, and Cluster Headroom Guidelines

### 1. Verification of Prometheus StatefulSet Lifecycle
The Prometheus StatefulSet pod (`prometheus-kube-prom-stack-kube-prome-prometheus-0`) was confirmed not to be recreated during the Grafana Helm upgrade. Container restart counts (0 for config-reloader, 9 for prometheus) remained identical before and after the upgrade, and pod start timestamp was unchanged. Do not assume or state that containers definitely did not restart unless verified by comparing pre/post restart counts.

### 2. Pod Stability and Troubleshooting Restraint
Avoid deleting or restarting healthy core components (Prometheus Operator, Prometheus Server, Grafana, kube-state-metrics, kube-apiserver, or etcd) during troubleshooting unless concrete diagnostic evidence specifically mandates it. Premature restarts can trigger cascading control-plane thrashing and unnecessary reconciliation loops on resource-constrained nodes.

### 3. Docker Host Contention and Benchmark Isolation
Concurrent Docker Desktop Kubernetes environments (such as Docker Desktop's built-in Kind cluster) introduce severe external CPU and disk I/O contention when running alongside Minikube. The unrelated Kind cluster containers (`desktop-control-plane`, `desktop-worker*`, `kind-cloud-provider`) must remain stopped during capstone benchmark runs to prevent I/O wait latency from confounding autoscaling evaluation.

### 4. Enforced Minikube Memory Headroom
Current baseline Minikube memory usage is approximately 2.1 GiB of the enforced 3.0 GiB cgroup limit (~70% allocation). Future components, metric collectors, and application scaling replicas must be budgeted and monitored carefully to prevent node-level cgroup OOM conditions.

**Applies to phases**: 8B/C (Monitoring), 11–13 (HPA), 15–21 (RL), 22–23 (Benchmarks).

---

## CORR-9 — Scientific Telemetry Collection Integrity, Target Health, and Runtime Provenance (Phase 9A/9B)

### 1. Verified demo-app Resource Configuration
The verified `demo-app` resource specification is:
- **CPU request**: `100m`
- **CPU limit**: `500m`
- **Memory request**: `64Mi`
- **Memory limit**: `256Mi`
Grafana's 50m/200m CPU settings apply strictly to the Grafana pod and must never be used as application metadata.

### 2. Dynamic Runtime Environment Provenance
- Installed Minikube version is `v1.38.1`.
- Kubernetes version is `v1.35.1`.
Neither version must be hardcoded in experiment metadata. The metrics collector dynamically queries runtime provenance (Minikube version, Kubernetes cluster version, git commit) at startup, falling back gracefully if tools are inaccessible.

### 3. Observational Boundaries & Port-Forward Lifecycle
- Accessing `http://localhost:9090` requires a Prometheus port-forward (`kubectl port-forward -n monitoring svc/kube-prom-stack-kube-prome-prometheus 9090:9090`).
- The metrics collector must remain purely observational and must NEVER automatically manipulate Kubernetes resources or start/stop port-forwards internally.
- Port-forward lifecycle management belongs to the higher-level experiment orchestrator or manual testing workflows. For Phase 9B testing, the port-forward is managed externally and terminated after test completion.

### 4. Telemetry Integrity: Distinction of Zeroes vs Missing Data
Scientific queries must NOT blindly use `or vector(0)` fallbacks where missing telemetry could be conflated with genuine zero usage.
The collector verifies Prometheus target health (`/api/v1/targets`) and distinguishes:
- **Real zero**: Application is healthy, target is up, metric exists and evaluates to 0.0 (e.g. idle request rate).
- **Absent metric**: Target is up, but metric has no samples (e.g., latency quantiles when no requests have occurred in the rate window) -> recorded as null/empty.
- **Query failure**: Prometheus returns an error code, timeout, or HTTP failure -> recorded as null with `collection_error` documented.
- **Unhealthy scrape target**: Target health is down or unreachable -> marked `target_up=False`, telemetry recorded as null/NaN, lifecycle event emitted.

### 5. Classification of Last Work Intensity Gauge
`demo_app_last_work_intensity` is strictly a diagnostic application gauge (`last_work_intensity_diagnostic`). Under multiple replicas, pod-level readings can become ambiguous and must never be used as an authoritative workload signal or RL state variable.

### 6. Monotonic Scheduling and Duration Boundary Enforcement (Phase 9C)
- Scheduling uses `time.monotonic()` against fixed planned deadlines (`start_mono + k * sampling_interval`) rather than repeated `sleep(interval)` calls after query completion, eliminating cumulative drift.
- If `--duration-seconds` is set, the scheduler terminates at the exact duration boundary (`start_mono + duration_seconds`) without scheduling samples beyond the deadline or sleeping a full sampling interval past the experiment end.
- Metadata and lifecycle events explicitly record both `requested_duration_seconds` and `actual_duration_seconds`.

### 7. Tightened Scientific Rate Null/Zero Semantics (Phase 9C)
- **Total request rate**: If the `/work` request counter series is absent (empty vector in Prometheus), all request rates (`request_rate_rps`, `success_rate_rps`, `error_rate_rps`) remain null/blank. Zero is never invented.
- **Success / Error rate categories**: If `target_up == True` and the total `/work` request counter series exists, absence of a status-specific series (2xx or 5xx) without query error is interpreted as a genuine `0.0` for that category.
- **Latency**: If no histogram bucket data exists in the range, latency quantiles remain null/blank (`""`) and are never encoded as `0.0s`.
- **Sample Validity**: A sample is scientifically valid (`sample_valid = True`) iff:
  1. `target_up == True`
  2. Essential deployment & container telemetry is present (`replicas_current`, `cpu_cores`, `memory_bytes` are not None)
  3. No Prometheus query or API errors occurred (`query_errors_in_tick == 0`)
  Legitimately idle application metrics (null rates or unobserved latency quantiles) do not invalidate the sample.

**Applies to phases**: 9A/B/C (Metrics Collector), 12–13 (HPA Experiments), 15–21 (RL Training), 22–23 (Benchmarks).

---

## CORR-10 — Workload Computation Reality, Host Sharing, and Planned Schedule Replay (Phase 10A/10B)

### 1. Measured CPU Computation Durations
Actual measured `/work` request durations on this Apple Silicon host environment:
- **Intensity 1**: 5–8 ms
- **Intensity 3**: 46–58 ms
- **Intensity 5**: 127–225+ ms
- **Intensity 10**: ~500 ms
Theoretical 1ms/9ms/16ms estimates from earlier scratch calculations significantly underestimate actual execution times and must not be used for capacity planning.

### 2. RPS Capacity Bounds Require Empirical Calibration
With `demo-app` constrained to a 500m CPU limit and Minikube constrained to an outer 2-CPU cgroup limit, safe sustained and burst RPS limits are not yet established. High rates (such as 50–80 RPS) must not be assumed safe or sustainable prior to empirical capacity calibration.

### 3. Host-Side Generator Physical Sharing
Running the load generator as a host-side Python process avoids consuming Minikube's 2-CPU / 3072 MB container cgroup, but the generator still executes on the same physical Apple Silicon machine. It must not be characterized as "zero contamination"; host CPU and memory footprint must be kept minimal ($<5\%$ CPU, $<50\text{MB}$ RAM).

### 4. Application Port-Forward Mapping and Validation
- The `demo-app` Kubernetes Service exposes port `80` targeting application container port `8080`.
- External port-forwarding to local port 8080 requires mapping `kubectl port-forward -n autoscaler-app svc/demo-app 8080:80`.
- Port-forward latency and throughput overhead must be empirically validated during capacity calibration to ensure it does not become a client-side bottleneck.

### 5. Deterministic Planned Arrival Schedule Replay
Matched HPA vs. RL experiments must not rely on re-running independent pseudo-random Poisson generation loops. The traffic generation system must decouple schedule planning from execution:
1. Generate an authoritative planned arrival schedule from `(profile, seed, arrival_process)`.
2. Persist `traffic_schedule.csv` before dispatching requests.
3. Replay the exact identical planned schedule for both HPA and RL comparative benchmark runs.

**Applies to phases**: 10A/B (Traffic Generator), 11 (Calibration), 12–13 (HPA Experiments), 15–21 (RL Training), 22–23 (Benchmarks).

---

## CORR-11 — Dispatch Lag vs Latency Attribution and Profile Duration Boundaries (Phase 10C)

### 1. Correct Interpretation of Dispatch Lag
`dispatch_lag_seconds = actual_dispatch_elapsed_seconds - planned_elapsed_seconds` measures generator-side scheduling accuracy on the host's monotonic clock. It represents how promptly the asynchronous event loop dispatches requests relative to the planned schedule. It does **not** measure network transit time, TCP handshake latency, or `kubectl port-forward` overhead.

### 2. Client-Side Latency Attribution Restraint
In low-load Poisson smoke tests (e.g. 2 RPS), observed client latency (~143 ms mean, ~293 ms p95) cannot be attributed specifically to port-forward overhead. Key contributors include:
- Stochastic Poisson inter-arrival variance (occasional back-to-back request bursts)
- Application concurrency queuing on a single-replica CPU-bound FastAPI service
- Apple Silicon OS scheduling and thread pool wakeups
- Port-forward local TCP loopback
The smoke test confirms functional end-to-end reliability, not isolated proxy performance.

### 3. Profile-Duration Lifecycle Boundary Semantics
- The authoritative profile duration defines the minimum experiment execution window (`start_mono + profile_duration`).
- In stochastic processes (like Poisson arrivals), the final planned arrival often occurs shortly before the profile deadline (e.g. $t = 24.49\text{s}$ for a $25.0\text{s}$ profile). The generator must **not** exit immediately when the last request completes; it must keep the experiment window open until the planned profile duration has fully elapsed.
- In-flight requests active at the deadline are granted a bounded completion window governed by `timeout_seconds`.
- Metadata explicitly distinguishes:
  - `requested_duration_seconds`: configured profile duration
  - `schedule_end_elapsed_seconds`: planned elapsed time of final scheduled request
  - `actual_runtime_seconds`: total elapsed execution time from start to finalization

**Applies to phases**: 10C (Traffic Generator Hardening), 11 (Calibration), 12–13 (HPA Experiments), 15–21 (RL Training), 22–23 (Benchmarks).

---

## CORR-12 — Stage-1 Calibration Artifact Validity, CPU Rate Terminology, and Throttling Definitions (Phase 10E)

### 1. Prometheus CPU Rate Terminology
Values derived from Prometheus PromQL expressions such as:
```promql
sum(rate(container_cpu_usage_seconds_total{namespace="autoscaler-app", pod=~"demo-app-.*", container="demo-app"}[1m]))
```
must **not** be termed "instantaneous CPU." They represent the sliding time-averaged CPU usage rate over the preceding 1-minute (`[1m]`) lookback window.
In all calibration and benchmark reporting, use rigorous terminology:
* **mean observed CPU rate**: Average of the windowed rate samples over the experiment or steady-state window.
* **maximum observed CPU-rate sample**: Highest single windowed rate sample captured by the collector.

### 2. Precise Definition of CPU Throttling
The `demo-app` container has an enforced CPU limit of `500m` ($0.5$ cores, or $50{,}000\,\mu\text{s}$ CPU quota per $100{,}000\,\mu\text{s}$ CFS period).
* CPU throttling is specifically the kernel-enforced cgroup CFS mechanism that suppresses thread scheduling once container execution exceeds its allotted runtime quota in a period (`container_cpu_cfs_throttled_periods_total`).
* An observed CPU rate sample approaching or exceeding `500m` does **not** by itself constitute throttling. Throttling must not be inferred solely from CPU-rate metrics; it requires direct evidence of throttled time or periods.

### 3. Stage-1 3-RPS Run Audit and Diagnostic Classification
During Phase 10D Stage-1 single-replica capacity calibration, two 3-RPS runs were recorded:
* `cal_single_r3_i3_20260906T082659Z` (Run 1)
* `cal_single_r3_i3_20260906T082941Z` (Run 2)

**Root Cause Investigation**:
In Run 1, deterministic arrival schedule generation used floating-point step increments `t += 1.0 / 3.0` over a 90.0-second profile. Floating-point accumulation resulted in $t \approx 89.99999999999999 < 90.0$, generating an off-by-one 271st planned request scheduled at `planned_elapsed_seconds = round(t, 6) = 90.0`. During execution, the generator's profile boundary check correctly prevented dispatching any request at or past the profile duration ($t \ge 90.0$). As a result:
* `total_offered`: 271 (planned in schedule)
* `total_sent`: 270 (dispatched)
* `success_rate`: $270 / 271 = 99.63\%$, artificially depicting an unsent request as an unfulfilled arrival.

**Remediation and Re-run**:
`loadgen/traffic_generator.py` was hardened to enforce `if t_rounded >= total_duration: break` before appending planned requests. With this fix in place, Run 2 was executed, yielding exactly 270 offered, 270 sent, and 100% success rate.

**Designation**:
* `cal_single_r3_i3_20260906T082659Z` is formally classified as **`invalid_diagnostic`**. Its artifacts are fully preserved for auditability under `persistent-data/` but excluded from official calibration tables.
* `cal_single_r3_i3_20260906T082941Z` is designated as the authoritative Stage-1 3-RPS calibration baseline.

**Applies to phases**: 10D (Stage-1 Calibration), 10E (6 RPS Calibration), 11 (Benchmark Workload Selection).

---

## CORR-13 — Canonical Workload Window and Telemetry Summary Methodology (Phase 10F)

### 1. Scope of Client-Side Latency Metrics
Client HTTP observations (`requests.csv`) must be calculated over **all requests dispatched in the authoritative workload window**.
* Unlike Prometheus rate queries, client-observed latency is an instantaneous measurement of end-to-end request-response time that does not require a moving rate lookback. Discarding early requests ($t < 60\text{s}$) is scientifically incorrect.
* Standard continuous linear interpolation (`numpy.percentile`) is the authoritative estimator for client percentiles ($p_{50}, p_{95}, p_{99}$).
* Rank-based integer indexing (`latencies[int(n * p)]`) is preserved in `metadata.json` for historical traceability. Discrepancies between historical reports (e.g. 1 RPS p95 $164.9\text{ ms}$ vs $166.1\text{ ms}$; 2 RPS p95 $164.4\text{ ms}$ vs $186.4\text{ ms}$) arise from linear interpolation versus integer rank indexing on small samples ($N=90, 180$).

### 2. Prometheus Telemetry Window Partitioning
Prometheus PromQL rate and quantile queries use `rate(...[1m])`. In a 90-second run starting from idle, the lookback window ramps from $t = 0$ to $t = 60\text{s}$.
To prevent conflating warm-up averages with steady-state capacity:
* **`steady_window`**: strictly evaluates samples where `elapsed_seconds >= 60.0` (samples at $t \approx 60\text{s}, 70\text{s}, 80\text{s}$), ensuring the 60-second window is 100% filled with active workload.
  - Reports: `steady_mean_observed_cpu_rate`, `steady_observed_request_rate_rps`, `steady_server_p95_seconds`, `steady_mean_memory_mb`.
* **`whole_run`**: evaluates all valid samples from start to finish, reported separately for operational auditability (e.g., `whole_run_max_observed_cpu_rate`).
* Analysis scripts and comparison tables must never silently mix whole-run and steady-window values in the same column.

### 3. Canonical Saturation Classification Criteria
Saturation is a multi-dimensional system property evaluated against project requirements ($p_{95} \approx 300\text{ ms}$ SLA reference, container limit $500\text{m}$):
* **Underloaded**:
  - Achieved request rate matches offered workload ($100\%$ delivery).
  - Zero drops, timeouts, connection errors, or container restarts.
  - $p_{95}$ latency (both client and server) comfortably satisfies the SLA target ($< 300\text{ ms}$).
  - Observed steady CPU rate leaves healthy headroom ($< 85\%$ of $500\text{m}$ limit).
* **Approaching Saturation**:
  - Sustained $p_{95}$ latency degrades toward or crosses the SLA threshold ($\ge 270\text{ ms}$ client or $\ge 285\text{ ms}$ server), OR
  - Steady CPU rate exceeds $85\%$ of the $500\text{m}$ limit ($> 425\text{m}$), leaving little service headroom, OR
  - Monotonic request queueing emerges, while the offered rate is still largely delivered.
* **Saturated**:
  - Severe and sustained SLA breaches ($p_{95} \gg 300\text{ ms}$).
  - Achieved throughput collapses below offered rate.
  - Elevated 5xx errors, client timeouts, or generator `dropped_overload` drops.
  - Persistent CFS throttling or pod instability/OOMKills.

An isolated outlier in tail latency ($p_{99}$) or CPU crossing the $100\text{m}$ request boundary does **not** constitute saturation.

**Applies to phases**: 10F (Analysis Reconciliation), 11 (Calibration Wrap-up), 12–13 (HPA Experiments), 15–21 (RL Training), 22–23 (Benchmarks).

---

## CORR-14 — Empirical Capacity Boundaries, Workload-Window Dispatch Rate, and cgroup CFS Throttling Deltas (Phase 10I)

### 1. Empirical Single-Replica Capacity Boundaries
Based on empirical calibration over 1–10 RPS at intensity 3 (single pod, 500m CPU limit, 256Mi memory limit):
* **1–6 RPS**: Underloaded (steady CPU 15.2%–53.5%, 0 errors/timeouts/drops, $p_{95} \le 165\text{ ms} < 300\text{ ms}$).
* **8 RPS**: High utilization / Approaching Saturation (steady CPU 76.8%, 7.39% CFS throttled periods, $p_{95} = 107.1\text{ ms}$, 100% throughput).
* **9 RPS**: Upper Sustainable Knee (steady CPU 85.1%, 29.87% CFS throttled periods, client $p_{95} = 211.9\text{ ms} < 300\text{ ms}$, 100% throughput, 0 errors/timeouts/drops).
* **10 RPS**: Saturated (steady CPU 92.5%, max CPU hits 100% limit at 500.1m, 81.41% CFS throttled periods, queue collapse: 141 timeouts, 5 drops, client $p_{95} = 4,709.3\text{ ms} \gg 300\text{ ms}$).
* **Empirical Transition**: Saturation transition lies strictly **between 9 and 10 RPS**. No untested rates (e.g., 7 or 9.5 RPS) may be claimed as verified.

### 2. Workload-Window Dispatch Rate Semantics
Canonical dispatch rate must be calculated across the authoritative workload window:
$$\text{dispatch\_rate\_workload\_window} = \frac{\text{dispatched\_requests}}{\text{requested\_profile\_duration}}$$
Dividing by `actual_runtime_seconds` distorts dispatch rate during overload runs because the process stays alive waiting for hanging in-flight timeout completions (e.g. at 10 RPS, $895 / 90.0\text{s} = 9.94\text{ RPS}$, whereas dividing by $93.21\text{s}$ yielded an artificial $9.60\text{ RPS}$).

### 3. cgroup CFS Throttling Delta Calculation
CFS throttling metrics in Prometheus are cumulative lifetime counters:
* `container_cpu_cfs_throttled_periods_total`
* `container_cpu_cfs_periods_total`
In this environment, these metrics are exposed by cAdvisor at the pod cgroup level (`namespace="autoscaler-app", pod=~"demo-app-.*"`), without an inner `container` label.
Run-specific quota throttling must be computed strictly via pre- and post-run deltas:
$$\Delta_{\text{throttled}} = \text{post\_throttled} - \text{pre\_throttled}$$
$$\Delta_{\text{total}} = \text{post\_total} - \text{pre\_total}$$
$$\text{throttled\_period\_fraction} = \frac{\Delta_{\text{throttled}}}{\Delta_{\text{total}}}$$
Lifetime counter division is prohibited.

**Applies to phases**: 10I (Calibration Conclusion), 11 (Workload Sizing), 12–13 (HPA Experiments), 15–21 (RL Training), 22–23 (Benchmarks).

---

## CORR-15 — Single-Replica Capacity Calibration Closure, Outcome Conservation, and Capacity Statement (Phase 10J)

### 1. Outcome Conservation Invariant
Every planned arrival in a synthetic workload must terminate in exactly one of the six terminal outcome states:
$$\text{total\_offered} = \text{success\_2xx} + \text{client\_error\_4xx} + \text{server\_error\_5xx} + \text{timeout} + \text{connection\_error} + \text{dropped\_overload}$$
The dispatched count is strictly:
$$\text{dispatched} = \text{total\_offered} - \text{dropped\_overload} = \text{success\_2xx} + \text{client\_error\_4xx} + \text{server\_error\_5xx} + \text{timeout} + \text{connection\_error}$$
Analysis tools and reporting tables must never silently omit failure categories (e.g. connection errors) in aggregated column formatting.

### 2. Reconciliation of the 10-RPS Outcome Audit
Auditing `persistent-data/traffic/cal_single_r10_i3_20260906T090800Z/requests.csv` confirms strict adherence to the conservation invariant across all 900 planned requests:
* `success_2xx`: 753
* `timeout`: 141
* `connection_error`: 1
* `dropped_overload`: 5
* `client_error_4xx`: 0
* `server_error_5xx`: 0
Total: $753 + 141 + 1 + 5 = 900$. Dispatched: $895$.
The historical 1-request omission in reporting arose solely from a table column formatter string (`server_5xx / timeouts / drops`) omitting `connection_error`. Raw data was verified 100% complete with zero missing records.

### 3. Final Evidence-Bounded Single-Replica Capacity Statement
* **1–6 RPS**: Empirically tested underloaded region (steady CPU $15.2\%\text{--}53.5\%$, $0$ errors/timeouts/drops, $p_{95} \le 165\text{ ms}$).
* **8 RPS**: Tested high-utilization / approaching-saturation region (steady CPU $76.8\%$, previously observed CFS throttled fraction $\approx 7.39\%$, client $p_{95} = 107.1\text{ ms}$).
* **9 RPS**: Highest tested sustainable rate (steady CPU $85.1\%$, CFS throttled fraction $29.87\%$, client $p_{95} = 211.9\text{ ms} < 300\text{ ms}$, $100\%$ success).
* **10 RPS**: Tested saturated condition (steady CPU $92.5\%$, max CPU hits $100\%$ limit at $500.1\text{m}$, previously observed CFS throttled fraction $\approx 81.41\%$, queue collapse: $141$ timeouts, $1$ connection error, $5$ drops, client $p_{95} = 4,709.3\text{ ms}$).

> **Canonical Finding**: Under the current single-replica configuration (intensity 3, 500m CPU limit, deterministic arrivals), the observed transition from sustainable operation to saturation lies between the tested 9-RPS and 10-RPS operating points. 9 RPS must not be termed an exact maximum capacity.

### 4. Single-Replica Calibration Closure
Single-replica capacity calibration is formally complete and closed. No further single-replica calibration traffic will be generated. The next phase proceeds to HPA configuration and baseline design, where workload sizing will account for both application saturation characteristics and the relationship between HPA CPU target utilization and the 100m CPU request.

**Applies to phases**: 10J (Closure), 11 (Workload Sizing & HPA Design), 12–13 (HPA Experiments), 15–21 (RL Training), 22–23 (Benchmarks).

---

## CORR-16 — Initial HPA Baseline Implementation and Metrics Server Research Configuration (Phase 11A/11A.1/11B)

### 1. CPU-Request Denominator Confirmed and Corrected For
Phase 11A/11A.1 (see `task_outputs/phase_11a_hpa_baseline_design_audit.md` and `task_outputs/phase_11a1_hpa_decision_refinement.md`) confirmed, against the official `autoscaling/v2` API semantics, that HPA CPU utilization is computed relative to the pod's CPU **request** (100m), never the CPU limit (500m). Re-expressed against this denominator, every valid Phase-10 calibration point (1–10 RPS) exceeds a naive 60% (60m) target, including the lightest tested load. The originally configured `target_cpu_utilization_percent: 60` in `configs/hpa.yaml` was therefore not a credible baseline and has been replaced.

### 2. Initial HPA Target Selected for Validation
`target_cpu_utilization_percent` is now **230** (nominal ~230m/pod, request=100m unchanged). This is an **initial Phase-11B validation value**, chosen because — once Kubernetes' default 10% no-action tolerance is correctly applied — it cleanly triggers scale-up at the 6-RPS calibration point (267.5m steady CPU) while leaving the entire empirically-classified 1–5-RPS Underloaded band undisturbed within the tolerance band. **It is NOT the final frozen benchmark target.** Designated tuning candidates for Phase 11C are 220% / 230% / 250%; the final value must be chosen only after empirical validation with dedicated tuning traffic, never on the held-out unseen evaluation profile.

### 3. Initial Scale-Up/Scale-Down Behavior
Scale-up behavior changed from the prior custom `{30s stabilization, +2 pods/30s}` to `{0s stabilization, +2 pods/15s}` — removing an avoidable stacked delay on top of Metrics Server's information staleness, while remaining meaningfully more conservative than the Kubernetes default (`0s stabilization`, up to `+4 pods` or `100%`, whichever is larger, per 15s), which was assessed as a local-cluster-safety risk on this project's constrained 2-CPU Minikube environment. Scale-down behavior is unchanged: `{120s stabilization, -1 pod/60s}`. `min_replicas=1` / `max_replicas=8` are unchanged.

### 4. Metrics Server Research Configuration
Live Metrics Server resolution changed from Minikube's addon-bundled default of `--metric-resolution=60s` to `--metric-resolution=15s`. This is the upstream `kubernetes-sigs/metrics-server` project's own Helm-chart default — a supported, standard value, not an aggressive or unvalidated departure. The change was applied via a targeted `kubectl patch` altering only the `--metric-resolution` argument (image, resources, ports, probes, and all other arguments were left untouched), and is reproducible via `scripts/configure_metrics_server.sh`. **This live patch does not survive `minikube delete` / cluster recreation** — after recreating the cluster, re-run `scripts/configure_metrics_server.sh` (or re-enable the addon and re-run it) to restore 15s. RL's control interval (`configs/experiment.yaml: rl.control_interval_seconds = 30`) is unchanged; it remains slower than HPA's 15s sync loop, so this change does not introduce a new asymmetry in RL's favor.

### 5. Startup/Not-Ready Pod Metric Handling — Expected Semantics for Phase 11C
Kubernetes handles missing or not-yet-reliable CPU metrics from newly-created or not-yet-ready pods conservatively during scale-up evaluation (governed by the default `--horizontal-pod-autoscaler-cpu-initialization-period=5m` and `--horizontal-pod-autoscaler-initial-readiness-delay=30s`, unmodified in this project). This can **dampen, reduce, or postpone** subsequent scale-up recommendations until reliable per-pod metrics become available for newly-scheduled replicas — it does not, by itself, cause *extra* scale-up. No traffic was generated in Phase 11B, so this is recorded here only as expected controller semantics for Phase 11C's interpretation of any observed multi-replica scaling sequence, not as an empirically observed effect.

### 6. Gradual Scale-Down and Replica-Hours
The retained gradual scale-down policy (`-1 pod/60s` after a 120s stabilization window) should not be assumed cost-neutral relative to a faster descale — it trades slower replica-hour recovery after a load spike subsides for reduced flapping risk and avoidance of simultaneous pod-termination herds. Phase 11C should empirically observe post-burst descale duration rather than assume it from this policy's parameters alone.

### 7. Structural Validation Performed (Phase 11B), No Load Generated
The HPA was applied and observed idle over multiple Metrics-Server/HPA sync cycles: `demo-app-hpa` reported `ScalingActive=True` (`ValidMetricFound`, "the HPA was able to successfully calculate a replica count from cpu resource utilization (percentage of request)"), stable `currentReplicas=1`/`desiredReplicas=1`, and observed CPU utilization of 6–8% (6–8m) against the 230% target — consistent with `kubectl top pod`'s independently-observed ~7m idle demo-app CPU. No scaling action occurred, as expected at idle with `minReplicas=1`. **This is an idle structural sanity check only — it does not validate the 230% target's suitability under load**, which remains Phase 11C's task.

**Applies to phases**: 11A/11A.1 (Design), 11B (Initial Implementation, this correction), 11C (Validation/Tuning — not yet performed), 12–13 (HPA Experiments), 22–23 (Benchmarks).

---

## CORR-17 — HPA Tuning Methodology, Exact-Replay Capability, and Precommitted Selection Criteria (Phase 11C.1)

### 1. Canonical Tuning Candidate Set
The three HPA `averageUtilization` values to be empirically compared in Phase 11C are, and remain, **220% / 230% / 250%**. No other value may be substituted during tuning without a documented methodological reason. The live repo HPA (`k8s/hpa/hpa.yaml`, `configs/hpa.yaml`) remains at the Phase-11B initial value (230%) until Phase 11D freezes a final winner — Phase 11C.1 changed neither file.

### 2. Exact-Schedule Replay Is Now Implemented and Required
`loadgen/traffic_generator.py` previously only supported generating-and-persisting a schedule coupled to immediate dispatch (`SCHEDULE_PERSISTENCE_ONLY` — no prior replay capability existed, verified by reading the code, not assumed from prior design descriptions). A `load_planned_schedule()` function and a `TrafficGenerator(..., replay_schedule_path=...)` / `--replay-schedule` CLI path were added, preserving request index, planned timestamp, phase, intensity, and target-RPS metadata exactly, never re-rolling randomness, and failing clearly (`ScheduleValidationError`) on structurally invalid input. **All three HPA-target candidates must be driven by `--replay-schedule` against the single canonical schedule** (`configs/traffic_profiles/hpa_tuning/canonical_schedule.csv`, SHA-256 recorded in the accompanying `.sha256` sidecar and to be re-verified before every candidate run) — never independently regenerated per candidate, even with the same seed.

### 3. Tuning Traffic Is Not Evaluation Traffic
`configs/traffic_profiles/hpa_tuning/hpa_tuning_v1.yaml` is HPA-tuning/design traffic only. It was built using Phase-10 calibration knowledge, which is explicitly permitted for tuning traffic. It must never be described as, or substituted for, the held-out unseen evaluation workload (`configs/traffic_profiles/unseen_eval.yaml`), which was not read, used, or modified by this phase. Findings from Phase 11C tuning must never be used to adjust `unseen_eval.yaml`.

### 4. Per-Phase Steady-Window Semantics (Correcting a Global Misapplication Risk)
For a single constant-RPS calibration run, "steady window = elapsed_seconds ≥ 60.0" was defined relative to the run's own start (CORR-13). For a multi-phase tuning run, this must be reapplied **relative to each phase's own transition boundary**, not globally from run start — i.e., a phase beginning at run-elapsed time `T_phase_start` has its own steady window beginning at `T_phase_start + 60s`, reflecting Prometheus's 1-minute rate lookback repopulating after every step change, not just after the very first one. Client-side (per-request) data is never discarded from the transition window — it remains valuable for measuring autoscaler response — only server-side (Prometheus rate-derived) aggregates require the post-transition offset.

### 5. Precommitted Reaction-Delay Definitions
To avoid collapsing distinct delays into one ambiguous "reaction time" number, Phase 11C separately measures: **decision/recommendation delay** (not directly observable without HPA-internal logs; approximated via `hpa_status.csv` sampling), **desired-replica change delay** (`first observed increase in Deployment desired replicas after a phase's step time, minus that step time` — the preferred primary metric, per instructions), and **ready-capacity delay** (time until the added pod(s) reach `Ready`). Scale-down is measured separately and analogously: time from the recovery phase's start to first scale-down action, time to return to 1 desired replica, time to return to 1 ready replica, and excess replica-seconds during the unwind — gradual scale-down's replica-hour cost is not assumed cost-neutral and must be observed, not inferred from its configured parameters alone.

### 6. Precommitted Candidate-Selection Criteria (Before Any Result Exists)
Frozen before any tuning run executes, to prevent post-hoc cherry-picking: reliability/SLA (success rate, timeouts, connection errors, drops, client p95/p99), responsiveness (desired-replica-increase delay at the 6-RPS and 8-RPS steps, time to Ready), efficiency (replica-seconds/replica-hours, average/peak replicas), stability (scale-up/scale-down action counts, direction reversals, any scaling during the 2/5-RPS phases), and cluster safety (readiness loss, HPA metric errors, invalid collector samples, control-plane/monitoring instability). Selection hierarchy: (1) reject any candidate causing persistent cluster/control-path instability; (2) among stable candidates, prefer those meeting reliability/SLA on the tuning schedule; (3) among reliable candidates, prefer lower replica consumption; (4) use responsiveness and stability as tie-breakers; (5) penalize a target that scales during the clearly-light 2/5-RPS phases without measurable reliability benefit. This rule is not to be altered after seeing results, except to correct a genuine, documented methodological defect.

### 7. Run Validity Criteria (Precommitted)
A future candidate run is **invalid** (result discarded from comparison, artifacts preserved, never deleted) if: the unrelated Kind cluster is running, the wrong HPA target is applied, Metrics Server is not exactly 15s, demo-app does not start at 1/1/1/1, the Prometheus demo-app target is unhealthy, the collector's critical telemetry fails materially, the replayed traffic schedule's hash does not match the canonical schedule's recorded hash, HPA/RL controller exclusivity is violated, the traffic generator fails structurally, or the cluster/control-plane crashes for reasons unrelated to the tested load. A run is **not** invalid merely because the *outcome* is bad — e.g., timeouts caused by genuine application overload under a candidate's scaling behavior are a valid experimental result, not an invalidating condition.

### 8. No Tuning Result Exists Yet
Phase 11C.1 is preparation only. No traffic was generated, no candidate target was applied to the live cluster (it remains at 230%, Phase 11B's initial value), and no finding about 220%, 230%, or 250%'s relative performance is asserted anywhere in this correction or elsewhere in the repository as of this entry.

**Applies to phases**: 11C.1 (Preparation, this correction), 11C.2 (Execution — not yet performed), 11D (Freeze).

---

## CORR-18 — HPA Tuning Methodology Hardening, Repetition/Order Protocol, Numeric SLA, and Integration Smoke Validation (Phase 11C.2)

### 1. The 8-RPS Phase Is Not Guaranteed to Force Scale-Up For Every Candidate
CORR-17's original framing of `scale_demand_8rps` as forcing scale-up "for all three candidates" is corrected. Because 220%/230% may already have scaled 1→2 during `discriminator_6rps`, they can enter the 8-RPS phase with more capacity than a 250% candidate that stayed at 1 replica. Single-replica Phase-10 CPU values cannot be used to assert every candidate scales again at 8 RPS. This phase is redefined as a **higher-load continuation / adequacy test**: does the capacity a candidate already selected remain adequate under higher total load? A candidate that performs no scale action at 8 RPS while preserving SLA/reliability/readiness is not penalized for that — it may indicate efficient provisioning. `discriminator_6rps` remains the primary threshold discriminator (its behavior is grounded in the one-pod starting state, which is a valid Phase-10-calibration-based inference; 8-RPS's behavior is not, once replica count may already differ between candidates). The tuning profile's phase comments and CORR-17's response-metric framing are both updated accordingly — `time to first scale-up after 8 RPS` is no longer a mandatory per-candidate metric; instead, record replicas entering the phase, HPA utilization entering the phase, whether scale-up was recommended, and (if none occurs) evaluate reliability/latency/efficiency as a legitimate outcome.

### 2. Recovery Phase Extended: 300s → 540s
Configured worst case from `maxReplicas=8`: 120s scale-down stabilization + 7 sequential `-1 pod/60s` reductions (8→1) = the 7th and final reduction can fire as late as `120 + 7×60 = 540s` after the recovery phase begins, with the full descent plausible by ~480s and a further ~60s of stable single-replica observation if the unwind completes on schedule. 300s could not guarantee observing a complete return to `minReplicas=1`. `recovery_2rps` is now `540s`; total profile duration is now **1080s (18 minutes)**.

### 3. Canonical Schedule Regenerated — Superseded Hash Recorded, Not Hidden
Because the recovery-phase duration changed, the canonical tuning schedule was regenerated offline (no traffic sent) before any official trial existed:
- **Superseded** (840s profile, never used in any official or smoke run — confirmed by the absence of any `persistent-data/traffic/hpa_tune_*` or `hpa_harness_smoke_*` directory referencing it): SHA-256 `558334bd6f69019299f01243bc2d01153b17f6b25d18527e311a0f11832d697c`, preserved at `configs/traffic_profiles/hpa_tuning/superseded/canonical_schedule_v1_840s_superseded.csv`.
- **Current** (1080s profile): SHA-256 `f5d9a209f19d2ec9704e221481e77ce95cdf39fdcb265f2dfa6ce5afef58c426`, at `configs/traffic_profiles/hpa_tuning/canonical_schedule.csv` — 4,230 planned requests, `t=0.0` to `t=1079.53s`, all five phases present in the expected proportions, reload via `load_planned_schedule()` verified to reproduce all 4,230 rows exactly (offline, no traffic).

### 4. Authoritative vs. Prometheus HPA Timing — Hardened, Not Merely Documented
`collectors/hpa_status_collector.py` previously sourced ALL numeric HPA state from Prometheus/kube-state-metrics (≈10s scrape cadence) while sampling every 5s — false precision, since most 5s ticks would re-read the same cached scrape. It now reads the **authoritative** state directly from the Kubernetes API (`kubectl get hpa/deployment -o json`) at its own sampling cadence for: HPA `status.currentReplicas`, `status.desiredReplicas`, `status.currentMetrics`, `status.conditions`, `status.lastScaleTime`; and Deployment `spec.replicas`, `status.replicas`, `status.readyReplicas`, `status.availableReplicas`. Prometheus/kube-state-metrics values are retained only as clearly-labelled secondary/corroborating `prom_*` fields. **Empirically confirmed** during the Phase 11C.2 integration smoke run (Section 6 of this correction): the `prom_current_cpu_utilization_percent` column visibly lags and clusters more coarsely than the kubectl-sourced `hpa_current_cpu_utilization_percent` column across identical timestamps — direct live evidence the original design's concern was real, not merely theoretical. A further nuance, discovered during that same smoke run and recorded honestly rather than overclaimed: even the kubectl-sourced value only changes as often as the HPA controller's own 15s sync period actually recomputes `status.currentMetrics` — polling via kubectl every 5s does not fabricate sub-15s precision in the underlying object's own update cadence, it only narrows how stale any single fetch can be relative to the last true reconciliation (0–14s reduced to 0–4s in the worst case), which is the correct and honestly-scoped claim.

### 5. HPA Recommendation, Deployment Mutation, and Ready Capacity Kept as Distinct Fields
`hpa_status.csv` now separately persists `hpa_status_desired_replicas` (controller recommendation), `deployment_spec_desired_replicas` (the mutation HPA actually writes), `deployment_status_replicas` (pods actually created), and `deployment_status_ready_replicas` / `deployment_status_available_replicas` (actual serving capacity) — never collapsed into one field, enabling Phase 11C analysis to distinguish these four distinct events in a scaling sequence.

### 6. Kubernetes Event Occurrence Tracking Hardened
UID alone is insufficient to detect a new occurrence of a repeated Event — Kubernetes updates an existing Event object's `count`/`lastTimestamp` (or, on newer API versions, `series.count`/`series.lastObservedTime`) in place rather than creating a new object. `_fetch_new_or_advanced_events()` now records a new observation when a UID is new, OR its occurrence count (top-level or `series.count`) has increased, OR its latest timestamp (`lastTimestamp`, `series.lastObservedTime`, or `eventTime`, in that preference order) has advanced — preserving UID, type, reason, message, count, series_count, first/last timestamps, event_time, and an observation timestamp. Covered by 6 new focused tests (new-UID, unchanged-not-reported-twice, count-advance, series-count-advance, preserved-fields, progress-key precedence), all passing. **This hardening was validated live, not just in unit tests**: the integration smoke run's `hpa_events.jsonl` correctly surfaced exactly one pre-existing `FailedGetScale`/`Unauthorized` warning event (a stale artifact from the HPA's initial RBAC-bootstrap window, `lastTimestamp` well before the smoke run itself) and correctly reported **zero** `SuccessfulRescale` events, consistent with no scaling having occurred.

### 7. Official Tuning Repetition and Counterbalanced Order — Frozen
Replacing Phase 11C.1's implied one-run-per-candidate design: **3 valid repeats per candidate** (220% / 230% / 250%), **9 official valid trials total**, all driven by the identical current canonical schedule (Section 3 hash). Precommitted counterbalanced order (each candidate appears once in first/middle/last position across the three blocks):
```
Block 1: 220 -> 230 -> 250
Block 2: 250 -> 220 -> 230
Block 3: 230 -> 250 -> 220
```
This order is frozen and will not be altered after seeing results. An **invalid** run (per CORR-17 §7's validity criteria) is preserved, never deleted, never counted toward the three valid repeats; its replacement is run for the same candidate immediately after environment recovery and is explicitly marked as a replacement for the invalid trial in its own metadata.

### 8. Repeated-Run Aggregation Rule — Frozen Before Results
For each candidate's 3 valid repeats, report **median, min/max range, and every individual run's value** (never a single mean that could hide one bad run) for: client p95/p99, success rate, timeout/error counts, first 6-RPS scaling delay, average/peak replicas, replica-seconds, scale-up/scale-down counts, recovery time. `n=3` is explicitly acknowledged as too small for high-powered statistical inference — this is a design-tuning decision aid, not a publishable statistical estimate; a stronger statistical protocol is left for the eventual final HPA-vs-RL benchmark, to be defined separately.

### 9. Numeric SLA and Reliability Criteria — Frozen From Authoritative Existing Project Values
Searched `configs/experiment.yaml`, `configs/reward.yaml`, and prior corrections for an existing, authoritative latency SLA before inventing one. **Found, and no conflict between sources**: `configs/experiment.yaml` (`sla.p95_latency_ms: 300`, `sla.error_rate: 0.01`) and `configs/reward.yaml` (`sla.p95_target_ms: 300`, with an explicit `SYNC` comment requiring it to equal `experiment.yaml`'s value) agree exactly. **Adopted as the frozen Phase 11C SLA definition, using the existing project value rather than inventing a new one**: SLA met ⇔ client p95 < 300ms; reliability floor ⇔ success rate ≥ 99% (the direct complement of `error_rate: 0.01`). Any timeout, connection error, or generator-side drop is treated as a reliability flag requiring explicit reporting (zero-tolerance framing consistent with Phase-10's own `classify_level()` precedent, which treats any occurrence of these — not merely a rate threshold — as disqualifying for "Underloaded"/"Approaching Saturation" classification). These criteria are frozen before any official candidate run and were not selected based on any Phase-11C result (none exists yet).

### 10. Candidate-Selection Logic Corrected for No-Scale Outcomes
A candidate is not automatically worse for performing fewer scaling actions, nor automatically better for scaling faster. Per-phase evaluation order is **reliability/SLA first → then efficiency → then responsiveness/stability**. A candidate that does not scale at 8 RPS while preserving SLA, zero errors/timeouts, and pod readiness is evidence of efficient provisioning, not failure to respond. Conversely, fast scaling that produces unnecessary replica-hours with no measurable reliability benefit is not automatically rewarded. This refines (does not reverse) CORR-17 §6's selection hierarchy.

### 11. Test Bookkeeping Correction (Documentation Hygiene Only — No Code Defect)
The Phase 11C.1 report's Section 18 described "24 new tests" for that phase and separately re-mentioned "9 already added in Phase 11B" — both statements are inconsistent with the actual file-by-file additions and are corrected here rather than by editing that prior report. Actual progression, verified by running `pytest tests/unit -q` at each stage: pre-Phase-11B baseline **63 passed**; Phase 11B added `tests/unit/test_hpa_config_consistency.py` (**6** tests) → **69 passed**; Phase 11C.1 added `TestScheduleReplay` in `test_traffic_generator.py` (**9**), `test_hpa_status_collector.py` (**6**), and `test_generate_tuning_schedule.py` (**3**) — **18** new tests, not 24 — → **87 passed** (this total itself was correct in the 11C.1 report; only the prose describing how it was reached was wrong). Phase 11C.2 rewrote `test_hpa_status_collector.py` for the hardened collector (6 → 23 tests, **+17**) → **104 passed**, verified live via `pytest tests/unit -q` during this phase. This is documentation hygiene only; no test failures occurred at any point.

### 12. Specialized Trial Harness Built
`scripts/run_hpa_tuning_trial.sh` (new): runs exactly one trial (preflight → port-forward lifecycle → metrics collector + HPA status collector (background) → traffic generator in schedule-replay mode (foreground) → collector finalization → artifact verification → cleanup → run-level metadata with validity marking). Deliberately narrow: no RL awareness, does not run all 9 trials automatically, does not choose a winner, does not alter `configs/hpa.yaml`/`k8s/hpa/hpa.yaml`/any workload profile, does not change the live HPA target itself (that remains a separate, explicit `scripts/set_hpa_target.sh` step performed before invoking this script).

### 13. Integration Smoke Test Executed — Excluded From Tuning Analysis
One tiny smoke workload (`configs/traffic_profiles/hpa_tuning/smoke/hpa_harness_smoke.yaml`, 1 RPS, 30s, its own separate schedule/hash, explicitly marked harness-integration-only) was run via `scripts/run_hpa_tuning_trial.sh` against the live, unchanged 230% HPA target, under experiment ID `hpa_harness_smoke_20260908T082810Z` (a non-`hpa_tune_*` ID, physically distinguishing it from any official trial). Result: **fully successful** — 30/30 requests succeeded, schedule hash verified, all three components (traffic generator, metrics collector, HPA status collector) produced valid, structurally-correct artifacts, replicas remained at 1 throughout (no scale-up, as expected at 1 RPS against a 230% target), one pre-existing non-scaling Event (`FailedGetScale`/`Unauthorized`, predating this run) was correctly surfaced and zero `SuccessfulRescale` events were recorded, and cleanup left no stale port-forwards or processes. **This run is explicitly excluded from HPA tuning analysis and must never be cited as evidence for or against the 230% target or any other candidate.**

### 14. Official Tuning Has Still Not Begun
No official `hpa_tune_t220_*`, `hpa_tune_t230_*`, or `hpa_tune_t250_*` trial has been executed. The live HPA target remains 230% (Phase 11B's initial value) throughout Phase 11C.1 and 11C.2. No finding about 220%, 230%, or 250%'s relative performance is asserted anywhere in this correction.

**Applies to phases**: 11C.1 (superseded framing corrected here), 11C.2 (hardening + smoke validation, this correction), 11C.3/official tuning (not yet performed), 11D (Freeze).

---

## CORR-19 — Event Baselining, Evidence Hierarchy, and Conditional 6-RPS Interpretation (Phase 11C.3)

### 1. Kubernetes Events Are Supplemental, Not Primary, Timing Evidence
The Kubernetes API itself documents Events as best-effort, limited-retention, supplemental information. The authoritative evidence hierarchy for Phase 11C is therefore, in order: (1) HPA `status.desiredReplicas`, (2) Deployment `spec.replicas`, (3) Deployment current/ready/available replicas, (4) HPA `status.lastScaleTime` — all captured directly in `hpa_status.csv`. Kubernetes Events (e.g. `SuccessfulRescale`) are **corroboration only** and must never be the sole source used to calculate reaction delay or scaling-action counts when the authoritative HPA/Deployment state is available.

### 2. Pre-Existing Events Are Now Baselined at Collector Startup, Not Reported as Trial Occurrences
The Phase 11C.2 integration smoke run surfaced a real defect: a `FailedGetScale`/`Unauthorized` Event that predated the trial by roughly an hour was written into that trial's `hpa_events.jsonl`, because the collector had no concept of "already existed before this run started." `HPAStatusCollector` now performs one baseline read of all currently-visible HPA-related Events **before** entering its sampling loop, initializing its progress-tracking state from them via `_baseline_existing_events()` — these baselined events are recorded separately (`hpa_events_baseline.json`, diagnostic context only, historical cluster Events are never deleted) and are **never** written to `hpa_events.jsonl` as trial occurrences. After the baseline, only a genuinely new Event UID, or a previously-baselined Event whose occurrence count/series count/timestamp has since advanced, is emitted.

### 3. Advanced-Event Detection Remains Hardened
The Phase 11C.2 detection logic (advancement via top-level `count`/`lastTimestamp`, or `series.count`/`series.lastObservedTime`, or `eventTime`) is retained unchanged — baselining only changes *when* the progress-tracking state is first populated (at startup, from a real read, rather than empty), not *how* advancement is detected. 5 new tests confirm: a pre-existing Event is not emitted on startup; an unchanged pre-existing Event stays un-emitted indefinitely; a pre-existing Event with an incremented top-level count is emitted; a pre-existing Event with an advanced `series.count` is emitted; a genuinely new UID appearing after the baseline is emitted. All passing.

### 4. `discriminator_6rps`'s One-Replica Assumption Is Conditional, Not Guaranteed
The schedule is sequential (2→5→6→8→2 RPS); live CPU is not perfectly deterministic. It is therefore **not guaranteed** that every official trial enters `discriminator_6rps` at exactly one replica, even though that is the intended, theoretically-expected state. Corrected framing: `discriminator_6rps` is the primary designed threshold-discrimination phase **when a candidate enters it at one replica** — analysis must always use the actual replica state observed at phase entry, never assume it. If phase-entry desired replicas = 1, the intended tolerance-aware 220/230/250 threshold interpretation applies directly. If phase-entry desired replicas > 1, the single-replica Phase-10 threshold prediction does **not** apply directly to that trial — it must be analyzed as sequential system behavior resulting from the candidate's own earlier decisions, not treated as a violation or reset. Replicas are never reset between phases; earlier scaling is part of a candidate's legitimate observed behavior.

### 5. Scaling During `moderate_5rps` Is a Valid Outcome, Not an Automatic Invalidity
A candidate scaling during the 5-RPS phase is not, by itself, grounds to invalidate that trial — it is a valid behavioral outcome unless caused by one of the existing invalidity criteria (CORR-17 §7: telemetry corruption, wrong target/config, contamination, harness failure, etc.). Candidate-selection logic (CORR-18 §10) may still penalize unnecessary light-load scaling on efficiency grounds, but the *run itself* remains valid data.

### 6. Phase-Entry Snapshot Defined and Implemented as a Tiny Helper
A new, narrowly-scoped pure-function module, `scripts/hpa_phase_entry_snapshot.py`, defines the authoritative phase-entry snapshot: for each phase in a tuning profile, the first `hpa_status.csv` sample at or after that phase's planned start time, reporting `hpa_status_current_replicas`, `hpa_status_desired_replicas`, `deployment_spec_desired_replicas`, `deployment_status_replicas`/`ready_replicas`/`available_replicas`, `hpa_current_cpu_utilization_percent`, and `last_scale_time` — exactly the fields item 4 above requires be recorded for every 6-RPS (and every other) phase. A missing snapshot (no sample found at/after a phase's start) is returned as `None`, never fabricated, and should be treated as a run data-completeness question. 7 new tests, all passing, including one that directly demonstrates the CORR-19 §4 correction (a synthetic run entering "phase_b" at desired=2, not the naively-assumed 1). This is explicitly not the general results-analysis framework — it is one small, reusable primitive for it.

### 7. Canonical Schedule and Protocol Unchanged
No phase RPS value, duration, intensity, or schedule content was altered in this phase. Verified via `shasum -a 256`: `configs/traffic_profiles/hpa_tuning/canonical_schedule.csv` still hashes to exactly `f5d9a209f19d2ec9704e221481e77ce95cdf39fdcb265f2dfa6ce5afef58c426` (unchanged from CORR-18 §3). The 3-valid-repeats-per-candidate protocol, the counterbalanced Block 1/2/3 order (CORR-18 §7), and the numeric SLA/reliability criteria (client p95 < 300ms, success rate ≥ 99%, zero-tolerance reporting of timeouts/connection-errors/drops, reliability-first/efficiency-second/responsiveness-third evaluation order — CORR-18 §9–10) are all unchanged and reconfirmed here, not altered in anticipation of any result.

### 8. Official Tuning Has Still Not Begun
No official `hpa_tune_t220_*`, `hpa_tune_t230_*`, or `hpa_tune_t250_*` trial has been executed. The live HPA target remains 230% throughout Phase 11C.3 (reconfirmed live, unchanged). No new live smoke test was needed or run in this phase — the Phase 11C.2 integration smoke already validated the harness end to end, and the corrections in this entry were validated entirely offline: 12 new tests this phase (5 event-baselining + 7 phase-entry-snapshot), full suite 104 -> 116 passed, no live cluster interaction beyond read-only confirmation that nothing drifted.

**Applies to phases**: 11C.2 (defect identified and corrected here), 11C.3 (this correction), official tuning (not yet performed), 11D (Freeze).

---

## CORR-20 — Corrected HPA Average-Utilization Semantics and Reconciliation of the First Official 220% Trial (Phase 11C.4A)

### 1. `averageUtilization` Does Not Scale With Replica Count
The Phase 11C.4 report incorrectly reasoned that the scale-up percentage threshold multiplies by replica count (e.g. `2 × 220% × 1.10 = 484%` at 2 replicas). This is **incorrect Kubernetes HPA semantics**. For a Resource CPU metric with `target.type: Utilization`, `averageUtilization` is, by definition, an **average per-pod percentage relative to CPU requests** — not a sum across replicas. The tolerance comparison is always `currentAverageUtilization / targetUtilization` versus `1.10`, **independent of replica count**: the percentage threshold stays at **≈242%** (`220% × 1.10`) no matter how many replicas exist. Replica count enters only in the separate, subsequent step `desiredReplicas = ceil(currentReplicas × currentMetric / desiredMetric)`, applied once the percentage tolerance is already breached — never by scaling the threshold itself.

### 2. `hpa_current_cpu_utilization_percent` Is Confirmed Correct — the Error Was a Report-Level Metric Conflation
`collectors/hpa_status_collector.py` reads `hpa_current_cpu_utilization_percent` verbatim from `.status.currentMetrics[].resource.current.averageUtilization` with no arithmetic beyond a `float()` cast — confirmed by code review and by the perfect numerical identity between this field and `hpa_current_cpu_average_value_millicores` holding at every sample of the first official trial (expected exactly because CPU request=100m). **No collector defect exists.** The Phase 11C.4 report's erroneous "335%"/"404%" figures were `metrics_collector.py`'s separate `cpu_cores` field (Prometheus-derived, aggregate, summed-across-replicas, 1-minute rate) misread as if it were the HPA's own per-pod average percentage — a report-interpretation error, not a data-collection or code defect. Future trial analysis must always distinguish `hpa_status.csv`'s `hpa_current_cpu_utilization_percent` (the only field relevant to HPA scaling-decision interpretation) from `metrics.csv`'s `cpu_cores` (a different metric, useful for other purposes, never to be read as an HPA percentage).

### 3. Corrected Findings for `hpa_tune_t220_b1r1_20260908T085324Z`
The true HPA-utilization ranges were: `discriminator_6rps` 146–193%, `scale_demand_8rps` 151–208% — both entirely under the 220% target, let alone 242%, throughout. The **only** above-242% episode in the whole 1080s run was a ~25–30s window (t≈115–140s) immediately surrounding the sole scale-up (1→2), not a recurring condition across the discriminator/8-RPS phases. All SLA/reliability/replica-efficiency conclusions in the original report were unaffected (they were computed from `requests.csv`/`hpa_status.csv` replica-trajectory fields, not the misread field) and are reconfirmed correct.

### 4. One Genuine, Investigated Anomaly: 321% at 2 Replicas Did Not Trigger a Third Pod
At t≈130–140s, a real, correctly-collected reading of 321% average utilization at `currentReplicas=2` did not produce a further scale-up, though the raw formula (`ceil(2×321/220)=3`) implies it should have. A read-only Prometheus per-pod query (against still-retained historical data, temporary port-forward, closed after use) **definitively proved** the two replicas were not receiving proportional traffic: the original pod carried ~153–329m CPU while the newly-created pod did ~1–4m — consistent with HTTP client connection-pinning (Kubernetes Services load-balance per new TCP connection, not per request) leaving the new pod almost idle despite being `Ready`. This imbalance is numerically consistent with the 321% reading being an effectively single-pod-equivalent measurement (~320m/100m≈320%). A documented Kubernetes HPA mechanism — special treatment of a pod's metric when it is not yet past `--horizontal-pod-autoscaler-initial-readiness-delay` (default 30s, unmodified here) or `--horizontal-pod-autoscaler-cpu-initialization-period` (default 300s, unmodified here) — is a well-supported candidate explanation (it would reduce the ratio calculation to `ceil(1×321/220)=2`, matching the observed outcome exactly), but the controller's exact internal per-reconciliation accounting cannot be independently confirmed from any artifact this project currently captures. **Classified `UNRESOLVED_FROM_PRESERVED_EVIDENCE` for the exact mechanism; the underlying per-pod load-imbalance cause is definitively proven, not speculative.**

### 5. Trial Validity Unaffected
`hpa_tune_t220_b1r1_20260908T085324Z` remains **VALID**. Every CORR-17 §7 validity criterion was evaluated from correctly-collected data untouched by this correction. An error in a report's *interpretation* of an already-correctly-collected metric does not retroactively invalidate the underlying experiment; experiment validity and report-interpretation correctness are tracked as separate questions.

### 6. Methodology, Schedule, and Protocol Remain Unchanged
No phase RPS/duration/intensity/schedule, HPA behavior, SLA criteria, validity rules, or the 3-repeat counterbalanced-order protocol required any change as a result of this correction. The original Phase 11C.4 task-output report was not modified (per standing practice, prior task-output files are never rewritten) — this entry, together with `task_outputs/phase_11c4a_t220_b1r1_results_reconciliation.md`, is the authoritative correction for all future reference. **Block 1 Trial 2 (230%) may proceed unchanged whenever it is next authorized.**

**Applies to phases**: 11C.4 (interpretation error identified and corrected here), 11C.4A (this correction), official tuning continuation (Block 1 Trial 2 onward — not yet performed), 11D (Freeze).

---

## CORR-21 — Corrected Multi-Replica Traffic-Delivery Path: `kubectl port-forward` Does Not Load-Balance, First 220% Trial Reclassified (Phase 11C.4B)

### 1. Root Cause Confirmed From Source, Not Report Prose
`scripts/run_hpa_tuning_trial.sh` (pre-correction, line 233) started the load generator's only path to the app with `kubectl port-forward -n autoscaler-app svc/demo-app 8080:80`, and the traffic generator's `--target-url` defaulted to `http://localhost:8080` (line 298) — i.e. every request in every prior trial, including the first official 220% trial, physically traveled through that single port-forward tunnel. Kubernetes documentation confirms `kubectl port-forward service/<name>` selects **one** backing pod at the time the tunnel is established and forwards to that pod for the tunnel's entire lifetime — it never touches the Service's normal kube-proxy load-balancing data plane. This is the exact, sourced mechanism (not merely circumstantial) behind CORR-20 §4's observed 153–329m vs 0.9–3.8m per-pod CPU imbalance after the sole scale-up event in `hpa_tune_t220_b1r1_20260908T085324Z`.

### 2. First Official 220% Trial Reclassified `INVALID_FOR_OFFICIAL_HPA_TUNING`
`hpa_tune_t220_b1r1_20260908T085324Z` is reclassified from VALID (per CORR-20 §5, which addressed only a separate metric-*interpretation* error and correctly found it did not retroactively invalidate the run) to **`INVALID_FOR_OFFICIAL_HPA_TUNING`**, reason code `traffic_delivery_path_pinned_to_single_pod_via_kubectl_port_forward`. CORR-20 §5's finding is not contradicted: an interpretation error and a traffic-delivery defect are independent validity questions, and this trial fails on the second one. No raw artifact (`requests.csv`, `metrics.csv`, `hpa_status.csv`, `traffic_schedule.csv`, `metadata.json`, or either prior report) is deleted or modified — this entry and `task_outputs/phase_11c4b_multi_replica_traffic_delivery_correction.md` are the authoritative record. The trial remains usable as evidence for **single-pod HPA percentage-threshold behavior** (which is what CORR-20 actually analyzed) but must **never** be counted as evidence of multi-replica HPA tuning performance, replica-efficiency, or candidate comparison. **Block 1 Trial 1 (220%) must be re-run under the corrected path before any official 220/230/250 comparison is finalized.**

### 3. Phase-10 Single-Replica Calibration Remains Valid, Unaffected
Phase 10's calibration ran entirely at a fixed single replica (`demo-app`, no HPA present). With exactly one backing pod, `kubectl port-forward svc/demo-app` pinning to "a pod selected by the Service" is definitionally the only pod that exists — there is no second pod to be starved, so the defect described in §1 cannot manifest in single-replica data. Phase-10's RPS→CPU/latency calibration curve (1–6 RPS underloaded 15%→54% of the 500m limit, 8–9 RPS approaching saturation at ~76.8%/85.1%, 10 RPS saturated) is **reconfirmed valid as originally recorded**, unrelated to and unaffected by this correction.

### 4. Corrected Service-Access Architecture
A new, dedicated NodePort Service, `demo-app-loadtest` (`k8s/app/service-loadtest.yaml`), selects the identical pods as `demo-app` (`app: demo-app`, `sessionAffinity: None`) but is reached by the host load generator via `minikube service demo-app-loadtest -n autoscaler-app --url`, a genuine kube-proxy load-balanced path. The primary `demo-app` Service (used by Prometheus/ServiceMonitor scraping and any future in-cluster/RL access) was not modified. Verified reachable for both `/health` and `/work`. Caveat discovered and documented: creating a second Service with the same pod selector causes Prometheus to scrape each pod twice under two different `job` labels (`demo-app` and `demo-app-loadtest`) with identical counter values — any future per-pod query must read one job's series (or filter to one job), never a naive `sum(...) by (pod)` across both, which would silently double-count.

### 5. HTTP Connection-Reuse Audit
The traffic generator's `httpx.AsyncClient` used `httpx.Limits(max_connections=..., max_keepalive_connections=...)`, leaving `keepalive_expiry` at httpx's 5.0s default and never setting `http2=True` (HTTP/1.1) or `Connection: close`. Under continuous traffic at the profile's rates, connections are reused far more often than they idle-expire — i.e. even after fixing the Service path alone, a single reused TCP connection stays pinned to whichever pod kube-proxy assigned it at connection-open time.

### 6. Empirically Measured: Fixing the Service Path Alone Is Insufficient (Run 1)
A non-tuning diagnostic profile (`configs/traffic_profiles/diagnostics/service_distribution_smoke.yaml`; 90s, 4 RPS constant, 360 requests) was run through the corrected NodePort tunnel against a temporarily, manually-scaled 2-replica deployment (HPA temporarily deleted for the duration of this diagnostic only) with the generator's **unmodified, current default** (keepalive) connection behavior — `svc_distribution_smoke_run1_20260908T093700Z`. **Acceptance threshold fixed in advance of observing this result**, per the task's own illustrative guidance: a per-pod split within **30–70%** each is acceptable evidence of genuine load-balancing; anything more skewed fails. Measured via per-pod Prometheus `demo_app_requests_total{route="/work"}` deltas (single `job` series, not summed across the duplicate Services per §4): original pod +2, new pod +358 — a **0.56% / 99.44%** split. This **decisively fails** the pre-declared threshold: severe single-pod stickiness persists even through a correctly load-balanced Service, confirming the root cause is HTTP connection-level pinning (§5), not the Service/routing layer.

### 7. Minimal Fix: Explicit `--connection-mode` Option, Verified to Resolve Distribution (Run 2)
The smallest targeted fix per the task's own guidance: a new `connection_mode` parameter on `TrafficGenerator` (CLI: `--connection-mode {keepalive,fresh}`, default **`keepalive`** — historical Phase-10-onward default behavior is unchanged unless explicitly overridden). `fresh` mode sets `httpx.Limits(max_keepalive_connections=0)`, forcing a new TCP connection (and thus a fresh kube-proxy backend selection) per request. The chosen mode is persisted in each run's `metadata.json` (`"connection_mode"`). A second smoke, `svc_distribution_smoke_run2_20260908T095743Z` (new experiment ID, same profile, same 2-replica/no-HPA state, `--connection-mode fresh`), measured original pod +176, new pod +184 — a **48.9% / 51.1%** split, comfortably inside the pre-declared 30–70% acceptance band. **Outcome: `fresh` mode is required and sufficient** to achieve genuine multi-replica distribution through this generator/Service/kube-proxy stack; `keepalive` must never be used for multi-replica HPA/RL benchmark traffic.

### 8. Phase-10 Recalibration Spot-Check: Attempted, Result Inconclusive (Confounded), Full Recalibration Not Required
Per the task's guidance to check (not necessarily fully re-run) whether the new connection-mode option materially changes single-pod CPU behavior, a small spot-check was attempted at 5 RPS and 9 RPS (both keepalive and fresh, single replica, 60s each: `connmode_spotcheck_{5,9}rps_{keepalive,fresh}_20260908T100159Z`). This spot-check is **methodologically compromised and its numbers are not usable as a clean comparison**: the freshly-reapplied HPA (230% target, canonical) legitimately scaled the deployment 1→2 replicas mid-check (a real, correctly-functioning HPA reaction to the diagnostic load itself, confirmed by `kubectl get hpa` conditions and a new pod's `startTime` of 10:02:24Z, ~25s into the very first sub-run), so 3 of the 4 windows had traffic split across two pods rather than measuring one pod under a controlled, fixed replica count — the raw per-pod CPU deltas measured (approx. 48%/34%/79%/47% of the 500m limit for keepalive-5rps/fresh-5rps/keepalive-9rps/fresh-9rps respectively) cannot be attributed to connection-mode alone and are **not** reported as a validated finding. The HPA scaled back down to 1 replica on its own once the diagnostic load stopped (confirmed organically, no manual intervention needed), which is itself further confirmation the restored HPA is healthy and responsive.

**Decision: a full or corrected-and-repeated calibration campaign is not required before resuming official tuning.** Reasoning: (a) Phase-10's own recorded calibration data is untouched and remains valid under the conditions it was actually collected in (§3); (b) `scripts/run_hpa_tuning_trial.sh` is being updated (§9) to use `--connection-mode fresh` uniformly for the **entire** duration of every future official trial, including its single-replica opening phase — so even if fresh-mode single-pod CPU cost differs somewhat from keepalive-mode (unconfirmed either way by this inconclusive spot-check), that difference would apply **identically across all three candidates (220%/230%/250%)** in the frozen counterbalanced design, and therefore cannot differentially bias the between-candidate comparison that is the actual object of the tuning study. A precise, controlled (HPA paused at a fixed replica count, as was correctly done for §6–7) recalibration curve under `fresh` mode is recommended only if a future phase needs to design a *new* single-replica HPA baseline from scratch, not to safely resume the already-planned candidate comparison.

### 9. Harness Updated for Future Official Trials
`scripts/run_hpa_tuning_trial.sh` was updated: the app-access step now starts `minikube service demo-app-loadtest -n autoscaler-app --url` as a tracked, backgrounded tunnel process (parsed for its printed URL, health-checked, and killed in the existing `cleanup()` trap — no stale tunnel processes), replacing `kubectl port-forward svc/demo-app`. The Prometheus port-forward is unchanged (observer path only). The traffic generator invocation now passes `--target-url "${APP_TUNNEL_URL}" --connection-mode fresh`. A lightweight distribution preflight was added before traffic starts: `demo-app-loadtest`'s selector must equal `demo-app`'s selector, its `sessionAffinity` must be `None`, and its live endpoint count must equal the deployment's Ready replica count — a cheap sanity check, not a per-run distribution test. Verified with `bash -n` (syntax OK).

### 10. Cluster Restored to Canonical State
After all diagnostics: `demo-app-hpa` reapplied from `k8s/hpa/hpa.yaml` (230% target, min/max 1/8, scaleUp stab=0s, scaleDown stab=120s — unchanged) and confirmed healthy; `demo-app` deployment returned to 1/1/1/1 (the original pod, via organic HPA scale-down, not forced); the diagnostic `minikube service demo-app-loadtest --url` tunnel and the temporary Prometheus port-forward (port 9095, used only for this phase's manual Prometheus queries) were both terminated, and no stray tunnel/port-forward processes remain; Metrics Server confirmed still at `--metric-resolution=15s`; Prometheus confirmed `Running`; no RL/custom-controller process found; Kind containers confirmed still exited. The new `demo-app-loadtest` NodePort Service (`k8s/app/service-loadtest.yaml`) intentionally **remains installed** as documented, permanent experiment-access infrastructure — it does not need to be, and was not, removed.

### 11. Verified by Tests, Not Only Manually
`connection_mode` validation (rejects unknown values, defaults to `keepalive`, accepts `fresh`) and the new `--connection-mode` CLI flag (defaults to `keepalive`, accepts `fresh`, rejects invalid choices via `SystemExit`) are covered by new unit tests in `tests/unit/test_traffic_generator.py`. Full suite: 122 passed (up from the CORR-20-era 116, consistent with 6 new tests added, zero regressions). `ruff check` clean on all changed Python files. `bash -n` clean on the updated shell script.

### 12. Official HPA Tuning May Resume
With the corrected load-generator path (`demo-app-loadtest` NodePort + `minikube service --url` + `--connection-mode fresh`) now wired into `scripts/run_hpa_tuning_trial.sh`, official HPA candidate tuning may resume. **Block 1 Trial 1 (220%) must be re-run** under the corrected path (its predecessor is `INVALID_FOR_OFFICIAL_HPA_TUNING` per §2) before the counterbalanced Block 1/2/3 sequence can be considered underway; no candidate targets, repeat count, or block order (CORR-18 §7) were altered by this correction.

**Applies to phases**: 11C.4 (root defect present since this trial), 11C.4A (CORR-20's validity finding refined, not reversed, by this entry), 11C.4B (this correction), official tuning restart (Block 1 Trial 1 onward, under the corrected path — not yet performed), 11D (Freeze).

---

## CORR-22 — Eliminated Duplicate Prometheus Scraping and Found the Fresh-Connection Discriminator Structure Broken (Phase 11C.4C)

### 1. Duplicate-Scrape Root Cause, Confirmed From Manifests and Live Prometheus State
`k8s/app/service-monitor.yaml`'s `spec.selector.matchLabels: {app: demo-app}` selects **Services** by their own `metadata.labels` (not by pod selector, and not by the pods a Service fronts). `k8s/app/service-loadtest.yaml` (added in Phase 11C.4B) carried `metadata.labels: {app: demo-app, ...}` — the same label — purely as descriptive metadata, not realizing it would also satisfy the ServiceMonitor's Service-level selector. Live confirmation before any fix: `kubectl get servicemonitor demo-app -n autoscaler-app -o yaml` showed `selector.matchLabels: {app: demo-app}`; Prometheus's `/api/v1/targets` showed two active targets — `job=demo-app` and `job=demo-app-loadtest` — both at instance `10.244.0.19:8080`, the identical pod.

### 2. Quantitative Proof of Double-Counting (Not Assumed)
Before any fix: `demo_app_requests_total{route="/work"}` read `job=demo-app: 5708` and `job=demo-app-loadtest: 5708` (identical). Sending exactly 20 real requests produced `job=demo-app: 5728` and `job=demo-app-loadtest: 5728` (both advanced by exactly 20, confirming they are the same underlying counter scraped twice). `sum(demo_app_requests_total{route="/work"})` (no job filter) returned **11456 = 2 × 5728** — an exact, unambiguous double-count. The same duplication was confirmed present on every other application metric: `demo_app_request_duration_seconds_count`/`_sum` (histogram), `demo_app_requests_in_progress` (gauge), `demo_app_last_work_intensity` (gauge) — all identical across both `job` labels. **Not affected**: `container_cpu_usage_seconds_total` and other cAdvisor/kubelet-sourced metrics (job=`kubelet`, a completely separate scrape path never touched by the ServiceMonitor), and `kube_deployment_status_replicas`/`kube_state_metrics`-sourced series.

### 3. Which Existing Queries Were Actually Corrupted
Audited every application-metric PromQL in `collectors/metrics_collector.py`, `dashboards/grafana/autoscaling-overview.json`, and `scripts/generate_dashboard_configmap.py`:
- **Materially doubled** (unfiltered `sum(...)` over a doubled-identical-series counter/gauge): `request_rate_rps`, `success_rate_rps`, `error_rate_rps` (all `sum(rate(demo_app_requests_total{...}))`), `requests_in_progress` (`sum(demo_app_requests_in_progress{...})`).
- **Mathematically unaffected despite the duplication**: `latency_p50/p95/p99_seconds` (`histogram_quantile(..., sum(rate(demo_app_request_duration_seconds_bucket{...})) by (le))`) — because both duplicate job series report byte-identical bucket distributions, uniformly doubling every `le` bucket by the same factor leaves the *relative* cumulative proportions histogram_quantile depends on unchanged, so the computed quantile is the same with or without the duplicate. `last_work_intensity_diagnostic` (`max(...)`) is likewise unaffected (max of two identical values). This distinction is important going forward: not every query touching a duplicated metric was actually wrong, only the `sum()`-based rate/gauge aggregates.
- **Never affected**: CPU/memory (cAdvisor via `job=kubelet`) and replica-count (`kube-state-metrics`) queries — a different scrape source entirely, not a different concern about the same source.

### 4. Fix: Prevent Duplicate Discovery, Not a Downstream `job=` Filter
Smallest, most robust fix at the discovery layer rather than patching every current and future PromQL expression: `k8s/app/service.yaml` (the primary `demo-app` Service) gained one new label, `purpose: primary-metrics-source`. `k8s/app/service-monitor.yaml`'s selector was changed from `{app: demo-app}` to `{app: demo-app, purpose: primary-metrics-source}` (matchLabels is AND-of-all-keys). `k8s/app/service-loadtest.yaml` was explicitly documented (comment only, no functional change) as intentionally never carrying this label. No pod selector, application behavior, or Prometheus scrape interval was changed; the primary Service's function is unaffected.

### 5. Fix Verified Live
After `kubectl apply`, the Prometheus Operator reconciled its scrape config within ~50s: `job=demo-app-loadtest` disappeared entirely from `/api/v1/targets` (confirmed via 5 consecutive 10s-spaced polls), leaving exactly one active target (`job=demo-app`, `10.244.0.19:8080`, `up`). Re-running the exact before/after counter-delta test with the fix live: `job=demo-app-loadtest` no longer returns any series at all (correctly stale/gone), `job=demo-app` alone tracked the real delta, and `sum(demo_app_requests_total{route="/work"}) by (job)` now returns exactly **one row**. `request_rate_rps`'s exact production PromQL (`sum(rate(demo_app_requests_total{namespace="autoscaler-app",route="/work"}[1m]))`) was independently re-run and confirmed to return a single, non-doubled value. `demo-app-loadtest`'s own function (host load-generator access) is unaffected — it is a Service, not a ServiceMonitor target, and was never meant to serve metrics.

### 6. Phase-10 Historical Calibration Remains Valid, As Historical Calibration
Phase-10's original single-replica, keep-alive-connection, `kubectl port-forward`-delivered calibration data (`docs/../task_outputs/PROJECT_UNDERSTANDING_PHASE10.md` §11) is **not** invalidated by anything in this entry. It remains the correct historical record of CPU/latency behavior **under the conditions it was actually collected in**. What changed is that official trials now use `--connection-mode fresh` (CORR-21) — a different, later-adopted delivery mechanism — so a new, explicitly-labeled controlled addendum was collected under `fresh` mode for direct comparison, not as a revision of Phase 10 itself.

### 7. Controlled Fresh-Mode Calibration Addendum (2/5/6/8 RPS)
Diagnostic state: `demo-app-hpa` deleted (documented, reversible — reapplied from `k8s/hpa/hpa.yaml` immediately after), `demo-app` fixed at 1/1/1/1 throughout (verified: `replicas=1` in all 9 metrics-collector samples of all 4 runs, 36/36), identical resources to Phase 10 (100m/500m CPU, 64Mi/256Mi memory — unchanged), Metrics Server unchanged (15s), Kind stopped, traffic delivered via the corrected `demo-app-loadtest` NodePort + `minikube service --url` tunnel with `--connection-mode fresh`, intensity=3, deterministic arrivals, 90s per level (first 60s = lookback-population/transition, `elapsed ≥ 60s` = Prometheus steady window — identical semantics to Phase 10's own `compile_stage1_calibration.py`/`summarize_calibration_level.py`, reused directly and unmodified for this analysis). New experiment IDs, distinct from both Phase 10 and the confounded Phase-11C.4B spot-check: `cal_freshaddendum_r{2,5,6,8}_i3_20260908T102529Z`. All 4 runs: offered = sent = successful (180/180, 450/450, 540/540, 720/720), zero timeouts/conn-errors/drops, `connection_mode=fresh` and the corrected tunnel URL confirmed in each run's `metadata.json`.

| RPS | Historical Phase-10 CPU mean | Fresh-mode CPU mean | Difference | Historical client p95 | Fresh client p95 | Historical server p95 | Fresh server p95 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 2 | 146m (29.2%) | 125.5m (25.1%) | −20.5m (−14.0%) | 164.4ms | 144.6ms | 244.8ms | 240.3ms |
| 5 | 229m (45.7%) | 262.8m (52.6%) | +33.8m (+14.8%) | 118.6ms | 117.4ms | 171.9ms | 230.6ms |
| 6 | 268m (53.5%) | 278.6m (55.7%) | +10.6m (+4.0%) | 128.6ms | 114.0ms | 216.7ms | 175.0ms |
| 8 | 384m (76.8%) | 379.5m (75.9%) | −4.5m (−1.2%) | 107.1ms | 292.6ms | 93.3ms | 154.1ms |

The direction of the difference is **not uniform** (down at 2 and 8 RPS, up at 5 and 6 RPS) — this is reported as an **observed difference under final benchmark-delivery semantics**, not attributed solely to TCP connection-setup cost; the access path (NodePort/kube-proxy tunnel vs. direct port-forward) and connection semantics both differ simultaneously, and Minikube's constrained 2-vCPU allocation under concurrent host-side tooling is a plausible contributor to some of this variance. Each level is a single 90s run (n=1 per level), matching Phase-10's own original precedent (also n=1 per level), not a larger repeated study.

### 8. Precommitted Discriminator Rule (Frozen Before Viewing Results) and Its Outcome
Boundaries (10% HPA tolerance applied to each candidate target on a 100m-request pod, exactly as CORR-20 §1 establishes): 220%→**242m**, 230%→**253m**, 250%→**275m**. Rule, fixed in advance: schedule remains valid if 2 RPS stays clearly below all three, 5 RPS stays below the 220 boundary with reasonable margin, 6 RPS falls in the useful discrimination region (crossing 220/230, staying below 250), and 8 RPS stays clearly above all three; redesign is required if 5 RPS crosses a boundary, 6 RPS stops separating the candidates, or 8 RPS falls unexpectedly below a boundary.

**Result — the rule is violated:**
- **2 RPS (125.5m)**: below 242m — assumption **holds**.
- **5 RPS (262.8m)**: exceeds 242m **and** 253m (below only 275m) — the intended "negative control, below all three" role **fails**. 5 RPS now sits exactly where 6 RPS was designed to sit.
- **6 RPS (278.6m)**: exceeds all three, including 275m — the intended "primary discriminator, above 220/230 but below 250" role **fails**; at 6 RPS all three candidates would already be triggered, so 6 RPS no longer separates 250% from 220%/230%.
- **8 RPS (379.5m)**: exceeds all three — assumption **holds**.

Both of Section-9's explicit "redesign required" trigger conditions are met simultaneously (5 RPS crosses boundaries; 6 RPS no longer separates the candidate family). Note this is a **narrowing of an already-thin margin**, not a defect introduced purely by connection mode: even historically, 5 RPS (229m) and 6 RPS (268m) sat much closer to their intended zones than the design narrative implied (229m was already only 13m/5.7% below the 220%-tolerance trigger). The `fresh`-mode shift (+33.8m at 5 RPS, +10.6m at 6 RPS) was enough to tip both over their respective action thresholds.

### 9. Verdict: `HPA_TUNING_METHODOLOGY_REQUIRES_REDESIGN_BEFORE_OFFICIAL_RUNS`
No candidate target, repeat count, or block order was altered in this task, per instruction. **The 220/230/250 candidate set and/or the 2/5/6/8 RPS discriminator schedule require redesign in a separate, dedicated methodology phase before any further official HPA tuning trial (including a replacement Block 1 Trial 1 at 220%) can be run** — running the existing schedule now would no longer test what it was designed to test, because the low/mid discrimination structure has collapsed under the now-mandatory `fresh` connection mode. This finding does not reopen or reverse CORR-21's fix (the multi-replica traffic-delivery correction is separately confirmed correct and necessary); it identifies a second, independent problem that must also be resolved before resuming.

### 10. Cluster Restored to Canonical State
`demo-app-hpa` reapplied from `k8s/hpa/hpa.yaml` (230%, min/max 1/8, unchanged behavior) and confirmed healthy; `demo-app` at 1/1/1/1 throughout and after; the addendum's `minikube service demo-app-loadtest --url` tunnel and both temporary Prometheus port-forwards were terminated; no stray tunnel/port-forward processes remain; Metrics Server confirmed still 15s; Prometheus confirmed healthy with the corrected single-scrape-path monitoring active (`job=demo-app` only); Kind containers confirmed still exited; no RL/custom-controller process found. `demo-app-loadtest` remains installed, as documented, permanent experiment-access infrastructure (labels updated only to exclude it from ServiceMonitor discovery, not removed).

### 11. Tests and Static Checks
Only YAML manifests changed this phase (`k8s/app/service.yaml`, `k8s/app/service-monitor.yaml`, `k8s/app/service-loadtest.yaml` comment-only) — no Python or shell files were modified. All three manifests validated with `yaml.safe_load` and `kubectl apply --dry-run=server`. Full unit suite: 122 passed, unchanged from CORR-21 (no regressions, as expected with no Python changes). No code was changed in order to manufacture a preferred calibration outcome — the addendum used the existing, unmodified `configs/traffic_profiles/calibration/cal_{2,5,6,8}rps_90s.yaml` profiles and the existing, unmodified `scripts/summarize_calibration_level.py` analysis tool.

**Applies to phases**: 11C.4B (the monitoring defect this entry fixes was introduced there), 11C.4C (this correction), official tuning restart (blocked pending a separate candidate/schedule redesign phase — replacement Block 1 Trial 1 at 220% must **not** proceed against the current schedule), 11D (Freeze).

---

## CORR-23 — Fine-Grained Final-Path HPA-Relevant CPU Map: A Robust {220,230}-vs-250 Discriminator Exists, but 220-vs-230 Does Not (Phase 11C.4D)

### 1. Two Distinct CPU Observation Sources, Not Interchangeable
Prometheus's `rate(container_cpu_usage_seconds_total[1m])` (cAdvisor-sourced) is useful for reproducible server-side characterization but is **not** the HPA controller's own input. The HPA reads the Kubernetes resource-metrics path (Metrics Server), which this phase queried directly and read-only via `kubectl get --raw /apis/metrics.k8s.io/v1beta1/namespaces/<ns>/pods/<pod>` — a new minimal helper, `scripts/metrics_api_cpu_sampler.py` (with `tests/unit/test_metrics_api_cpu_sampler.py`, 14 tests, all mocked `subprocess.run`, no live calls). Metrics Server here is configured at `--metric-resolution=15s`; polling faster than that only re-reads the same cached value. The sampler identifies each underlying observation by the API response's own `timestamp` field (not by poll wall-clock time) and explicitly excludes repeated-timestamp polls from summary statistics — verified live before use (a 20s/3s-interval smoke test against the real cluster produced 7 polls but only 2 unique underlying Metrics Server updates, exactly matching the ~15s cadence).

### 2. Controlled State and Fine-Grained Grid
`demo-app-hpa` deleted (documented, reversible, reapplied immediately after); `demo-app` fixed at 1/1/1/1 throughout (independently verified before/after every one of 21 runs — zero replica-count changes, zero invalid runs); identical resources to Phase 10 (100m/500m CPU); Metrics Server unchanged (15s); Kind stopped; traffic via the corrected `demo-app-loadtest` NodePort tunnel with `--connection-mode fresh`; intensity=3; deterministic arrivals; 90s per run. Fractional RPS was confirmed natively supported by the traffic generator (`target_rps: float` throughout, `interval = 1.0 / rate`, no integer casting anywhere) — no code change was needed. Grid: **3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0 RPS**, each with **3 valid repeats** (21 runs total), new profile files under `configs/traffic_profiles/hpa_mapping/` (kept separate from Phase 10's `configs/traffic_profiles/calibration/` and from the Phase 11C.4C addendum). Repeat order was counterbalanced by cyclic rotation, documented before execution: Round 1 (rep1) ascending 3→7; Round 2 (rep2) rotated by 3 positions; Round 3 (rep3) rotated by 5 positions — so no RPS level is confounded with a fixed position in time. Experiment IDs use the `hpa_mapping_r<rps>_rep<n>_<timestamp>` prefix, never `hpa_tune_...` or `cal_...`, and are stored under a new `persistent-data/hpa_mapping/` tree, kept separate from Phase 10, the CORR-22 addendum, and any future official tuning or RL data.

### 3. Measurement Semantics
Each run: full client request set (`requests.csv`); Prometheus steady window `elapsed ≥ 60s` (matching Phase 10/CORR-22 exactly, via the unmodified `scripts/summarize_calibration_level.py`); Metrics-API samples analyzed on their **own** semantics, not Prometheus's rule — specifically, only samples with poll time ≥ 20s after the run's own start (roughly 1.3× one Metrics-Server window) are treated as this run's steady Metrics-API observations, to exclude Metrics-Server windows that still partially overlapped the immediately-preceding run (back-to-back 90s runs with only a 5s gap). This is a deliberately different, smaller cutoff than Prometheus's 60s rule, justified by Metrics Server's own much shorter (~15s) lookback window.

### 4. Results — Fine-Grained CPU Map (median of 3 repeats; full per-repeat values preserved in each run's artifacts)

| RPS | Prometheus steady CPU mean (median across reps, range) | Metrics-API steady CPU (median-of-medians, range) | frac(>242m) per rep | frac(>253m) per rep | frac(>275m) per rep |
|--:|--:|--:|---|---|---|
| 3.0 | 158.9m [131.1–186.2] | 185.4m [156.7–188.5] | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 |
| 4.0 | 197.7m [181.4–230.6] | 229.0m [225.4–233.6] | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 |
| 4.5 | 247.5m [219.0–251.8] | 248.2m [240.5–250.5] | **0.4, 1.0, 1.0** | 0.4, 0, 0.25 | 0, 0, 0 |
| 5.0 | 244.3m [228.0–249.8] | 261.6m [260.5–264.4] | **1.0, 1.0, 1.0** | **1.0, 1.0, 1.0** | 0, 0, 0 |
| 5.5 | 263.1m [244.2–264.4] | 279.5m [279.2–281.2] | 1.0, 1.0, 1.0 | 1.0, 1.0, 1.0 | **1.0, 1.0, 1.0** |
| 6.0 | 267.9m [267.7–277.5] | 297.4m [292.5–306.9] | 1.0, 1.0, 1.0 | 1.0, 1.0, 1.0 | 1.0, 1.0, 1.0 |
| 7.0 | 313.0m [296.1–314.9] | 338.8m [337.8–344.8] | 1.0, 1.0, 1.0 | 1.0, 1.0, 1.0 | 1.0, 1.0, 1.0 |

Boundaries: 220%→242m, 230%→253m, 250%→275m (10% HPA tolerance, per CORR-20 §1). Application context throughout: 100% success rate, 0 errors, client p95 never exceeded ~135ms (well inside Phase 10's own 270ms "approaching saturation" classifier) at every level 3–7 RPS — none of these levels approach saturation; they characterize only the scale-pressure question, not application-capacity limits.

### 5. Discrimination Assessment (Rule Frozen Before Interpretation)
A level counts as a **robust** below- or above-boundary control only if its repeated observations sit on the same side of that boundary with a useful margin across all 3 independent repeats, not merely because a mean happens to fall on one side.

- **3–4 RPS**: robust negative control — `frac(>242m) = 0` in all 3 repeats at both levels, with a real margin (medians 12–17m below 242m even at 4 RPS).
- **4.5 RPS**: the genuine transition zone, and it is **not robust** — `frac(>242m)` swings from 0.4 to 1.0 across three otherwise-identical repeats. The run-level median (240.5–250.5m) sits almost exactly on the 242m boundary, so which side any individual Metrics-Server sample lands on is dominated by measurement noise, not by the workload level itself. This is precisely the failure mode CORR-22/this task warned against: a median close to a boundary is not evidence of reliable discrimination.
- **5.0 RPS**: a **robust, highly reproducible discriminator** — `frac(>242m) = 1.0` and `frac(>253m) = 1.0` in all 3 repeats (220% and 230% both reliably trigger), while `frac(>275m) = 0` in all 3 repeats (250% reliably does not). This cleanly and consistently separates `{220%, 230%}` as a group from `250%` — but it does **not** separate 220% from 230% individually, since both fire together at every repeat.
- **5.5–7 RPS**: robust "all three trigger" / high-scale-pressure region, consistent across all repeats.

### 6. Existing Target Separation
- **220% vs 230% (242m vs 253m, an 11m gap)**: **Not empirically distinguishable.** No tested RPS level produces a reproducible split between crossing 242m but not 253m; the CPU-vs-RPS function transitions from "always below both" (4 RPS) to "always above both" (5 RPS) within roughly half an RPS unit, and the one level whose median falls inside the 11m gap (4.5 RPS) shows unstable, non-reproducible boundary-crossing behavior. The gap is well inside this pipeline's demonstrated run-to-run and within-run noise band.
- **230% vs 250% (253m vs 275m, a 22m gap)**: **Empirically distinguishable.** 5.0 RPS gives a perfectly reproducible 100%-above-253m / 0%-above-275m split across all 3 independent repeats.

Verdict: **`EXISTING_TARGET_SPACING_IS_TOO_NARROW_FOR_ROBUST_TUNING`** — driven specifically by the 220-vs-230 pair; the 230-vs-250 pair alone would be workable.

### 7. Workload-Only Redesign Feasibility
A scientifically valid **multi-step trajectory** (not a single magic discriminator) is achievable with workload levels alone: a robust low/negative-control region (~3–4 RPS), a robust `{220,230}`-vs-`250` discriminator (~5 RPS), and a robust high-scale-pressure region (~5.5–7 RPS, all three trigger). This is sufficient groundwork for the next redesign phase to build a trajectory-based comparison. However, **no workload level can substitute for wider candidate spacing** where individual 220-vs-230 separation is required — that is a target-redesign question, not a workload-redesign question, given the transition's demonstrated steepness relative to measurement noise. A future redesign phase should also consider adding an 8–9 RPS-equivalent level (this map's 3–7 RPS range stayed well inside Phase 10's "Underloaded" classification; a genuine higher-load/reliability-adequacy observation point, as in Phase 10's original schedule, is not covered by this map alone).

### 8. Cluster Restored to Canonical State
`demo-app-hpa` reapplied from `k8s/hpa/hpa.yaml` (230%, min/max 1/8, unchanged behavior), confirmed healthy at 1 replica; the mapping tunnel and the temporary Prometheus port-forward terminated; no stray processes; Metrics Server still 15s; Prometheus healthy with the CORR-22 single-scrape monitoring still active; Kind still exited; no RL controller. `demo-app-loadtest` remains installed. Official HPA tuning remains paused; no new canonical schedule was created in this phase.

### 9. Tests and Static Checks
New: `scripts/metrics_api_cpu_sampler.py` and `tests/unit/test_metrics_api_cpu_sampler.py` (14 tests, mocked subprocess, no live calls). Full unit suite: **136 passed** (up from 122, no regressions). `ruff check` clean on both new files. All 7 new/reused `configs/traffic_profiles/hpa_mapping/*.yaml` profiles validated with `yaml.safe_load`. No application/HPA logic was changed; no traffic-profile schema change was needed (fractional RPS was already supported).

**Applies to phases**: 11C.4C (the discriminator-structure question this entry answers empirically), 11C.4D (this correction), the next methodology-redesign phase (must treat 220-vs-230 as requiring wider target spacing, and may use the ~3–4/~5/~5.5–7 RPS regions identified here as a starting point — no schedule is yet created), official tuning restart (still blocked pending that redesign), 11D (Freeze).

---

## CORR-24 — Redesigned HPA Candidate Family (200/230/260) and Canonical Tuning Schedule v2 (Phase 11C.4E)

### 1. Old Family Superseded, Not Erased
`220% / 230% / 250%` is marked **`SUPERSEDED_BEFORE_ANY_VALID_OFFICIAL_TUNING_RESULT`**. Reason (CORR-23): the 220-vs-230 tolerance-adjusted boundary gap (242m vs 253m, 11m) was smaller than the observed CPU variability at the one workload level (4.5 RPS) whose median fell inside that gap — `fraction of samples crossing 242m` ranged 0.4 to 1.0 across three otherwise-identical repeats, i.e. not reproducible. No valid official result is discarded by this change: the sole 220% attempt (`hpa_tune_t220_b1r1_20260908T085324Z`) was already `INVALID_FOR_OFFICIAL_HPA_TUNING` for an unrelated traffic-delivery defect (CORR-21/22), and no 230% or 250% trial was ever run. `hpa_tuning_v1.yaml` and the 1080s canonical schedule are **preserved**, not deleted, at `configs/traffic_profiles/hpa_tuning/hpa_tuning_v1.yaml` and `configs/traffic_profiles/hpa_tuning/superseded/canonical_schedule_v2_1080s_superseded.csv` (SHA-256 `f5d9a209f19d2ec9704e221481e77ce95cdf39fdcb265f2dfa6ce5afef58c426`, unchanged from CORR-18).

### 2. Candidate-Family Selection Methodology
Selection used the **raw, per-sample** Phase-11C.4D Metrics-API distributions (steady window, elapsed ≥20s per run, 21 runs total) — never medians alone. Precommitted suitability criteria (before comparing families): (1) three trigger regions empirically distinguishable under observed variability; (2) a robust low-load none-trigger region; (3) a region where only the aggressive candidate triggers; (4) a region where aggressive+middle trigger but conservative does not; (5) a higher region where all three experience scale pressure from a one-replica start; (6) operationally reasonable relative to 100m request/500m limit; (7) a meaningful efficiency-vs-responsiveness tradeoff, not three near-identical policies.

### 3. Two Families Evaluated Against Every Raw Sample
**Family A (200/230/260 → 220m/253m/286m tolerance-adjusted)**:
- 3 RPS: `frac(>220m)=0` in all 3 repeats — robust none-trigger.
- 4 RPS: `frac(>220m)=0.75,1.0,1.0`; `frac(>253m)=0` in all 3 — robust "200%-only" region (one repeat at 0.75, not perfectly unanimous but directionally consistent, a near-boundary case: the single missed sample in that repeat was 283.6m against a very different, much-later boundary check, not this one — see raw data).
- 5 RPS: `frac(>220m)=1.0`, `frac(>253m)=1.0`, `frac(>286m)=0.0`, **all three identical across all 3 repeats** — a perfectly robust `{200%,230%}`-vs-`260%` discriminator.
- 6 RPS: `frac(>220m)=1.0`, `frac(>253m)=1.0` all reps; `frac(>286m)=0.75,1.0,1.0` — all three under real pressure, 260% not perfectly unanimous (one repeat's lowest sample, 283.6m, missed 286m by 2.4m).

**Family B (190/230/270 → 209m/253m/297m tolerance-adjusted)**:
- 4 RPS: `frac(>209m)=1.0` in all 3 repeats — cleaner than Family A's 4 RPS region.
- 5 RPS: identical to Family A (`{190%,230%}`-vs-`270%`, perfectly robust, since both families share the 253m boundary and 5 RPS values never approach 286m or 297m).
- 6 RPS: `frac(>297m)=0.25,0.5,1.0` — **not robust**, a wide, unstable spread.
- 7 RPS required for a robust all-trigger region (`frac(>297m)=1.0` in all 3 repeats), one RPS step later than Family A.

**Decision: Family A (200/230/260) selected.** It achieves the full four-tier staircase (none / aggressive-only / aggressive+middle / all-trigger) within the same 3→6 RPS range Phase 11C.4D already measured, one RPS level earlier than Family B needs for a robust all-trigger region — a more evidence-efficient use of the same workload range. Family B's slightly cleaner 4 RPS region did not outweigh its materially less robust 6 RPS region. 230% is retained as the center candidate per the task's preference: it is the Phase-11B canonical baseline, prior design work centers on it, and the raw evidence independently confirms it sits usefully in the middle of Family A's staircase (it fires together with 200% at 5 RPS, stays quiet with 260% at 4 RPS).

### 4. Sequential-Trajectory Principle Unchanged
CORR-19's principle is explicitly reused, not reopened: the one-replica map designs meaningful **initial** trigger opportunities; the actual official schedule is sequential, so a candidate that scales in an earlier phase enters later phases with different capacity than the one-replica map alone would predict. This is expected, legitimate candidate behavior, not a validity violation. Phase-entry replica state must still be recorded (`scripts/hpa_phase_entry_snapshot.py`, unchanged) and every phase after the first interpreted conditionally on it.

### 5. Redesigned Workload Trajectory — `hpa_tuning_v2.yaml`
Six constant-RPS phases, durations justified against unchanged HPA mechanics (Metrics Server 15s, HPA sync ~15s, scaleUp stabilization 0s/+2 pods per 15s, scaleDown stabilization 120s/-1 pod per 60s):

| Phase | RPS | Duration | Purpose |
|---|--:|--:|---|
| `baseline_3rps` | 3 | 90s | Robust none-trigger negative control (unchanged duration from v1's baseline; RPS changed 2→3 per new evidence). |
| `aggressive_probe_4rps` | 4 | 120s | New phase: only 200% expected to trigger. 120s gives margin for detection (~15–30s) + scale-up action + new-pod readiness + metric re-stabilization. |
| `discriminator_5rps` | 5 | 120s | Primary, perfectly-robust `{200%,230%}`-vs-`260%` discriminator (up from v1's 90s `moderate_5rps`, reflecting this phase's increased evidentiary weight). |
| `scale_pressure_6rps` | 6 | 180s | All three expected under real pressure (duration unchanged from v1's `discriminator_6rps`, already justified to accommodate multiple full HPA reconciliation cycles). |
| `adequacy_8rps` | 8 | 180s | Higher-load continuation/adequacy test, unchanged from v1's `scale_demand_8rps` (same RPS, same duration, same non-destructive rationale — final-path fresh-mode data already shows ~379.5m one-replica CPU and client p95 approaching the 300ms SLA here, CORR-22). |
| `recovery_2rps` | 2 | 540s | Unchanged from v1 — worst-case scale-down math (120s stabilization + 7×60s reductions from maxReplicas=8) is a pure function of unchanged `behavior.scaleDown` settings and `maxReplicas=8`, independent of which averageUtilization values are compared. |

Total duration: **1230s** (20.5 min), +150s vs v1's 1080s, entirely attributable to the new `aggressive_probe_4rps` phase plus `discriminator_5rps`'s extended duration. Intensity fixed at 3 throughout (unchanged). Deterministic arrivals (unchanged rationale).

### 6. New Canonical Schedule Generated and Verified Offline
Generated via the existing, unmodified `scripts/generate_tuning_schedule.py` (never opens an HTTP client) from `hpa_tuning_v2.yaml`, deterministic arrivals, seed 42:
- **New**: `configs/traffic_profiles/hpa_tuning/canonical_schedule.csv`, SHA-256 `4f648411e039f9229cd35a06374e6870dc75eca67e385e67f4f52bb0b5ec55ad`, 4950 rows, `t=0.0` to `t=1229.583333s`, phase row counts 271/479/600/1080/1440/1080 (baseline/aggressive/discriminator/scale_pressure/adequacy/recovery) summing to 4950.
- **Old (v2, 1080s)**: preserved at `configs/traffic_profiles/hpa_tuning/superseded/canonical_schedule_v2_1080s_superseded.csv`, SHA-256 `f5d9a209f19d2ec9704e221481e77ce95cdf39fdcb265f2dfa6ce5afef58c426` (unchanged, matches CORR-18).
- **Older (v1, 840s)**: unaffected, remains at `configs/traffic_profiles/hpa_tuning/superseded/canonical_schedule_v1_840s_superseded.csv`.
- Verified offline (no traffic sent): `load_planned_schedule()` reproduces all 4950 rows exactly, all 6 phases present in the expected proportions; regenerating from the same profile/seed a second time to a scratch path produced a byte-identical file and identical SHA-256, confirming determinism.

### 7. Harness and Config Updated
`scripts/set_hpa_target.sh`: `ALLOWED_VALUES` changed from `(220 230 250)` to `(200 230 260)`; usage/error text updated accordingly; still refuses any other value without `--allow-arbitrary`. `scripts/run_hpa_tuning_trial.sh` required **no functional change** — it already reads `--target`/`--profile`/`--schedule` generically and verifies the live HPA target against whatever value is passed, with no candidate value hardcoded (confirmed by inspection). `k8s/hpa/hpa.yaml` and the live cluster HPA **remain at 230%** (Phase-11B baseline), per instruction, until an eventual winner is frozen in Phase 11D. New static guard tests added: `tests/unit/test_hpa_tuning_harness_static.py` (6 tests) assert the harness never contains the old `kubectl port-forward svc/demo-app` traffic pattern, does use the `demo-app-loadtest` NodePort tunnel and `--connection-mode fresh`, still permits the Prometheus observer port-forward, retains its distribution preflight, and that `set_hpa_target.sh` enforces exactly the new `(200 230 260)` family.

### 8. Expected Trajectory (Hypotheses, Not Validity Conditions)
| Phase | 200% expected | 230% expected | 260% expected |
|---|---|---|---|
| `baseline_3rps` | 1 replica | 1 replica | 1 replica |
| `aggressive_probe_4rps` | likely scales (crosses 220m) | likely stays at 1 (below 253m) | likely stays at 1 (below 286m) |
| `discriminator_5rps` | scales/stays scaled (crosses 220m) | scales (crosses 253m) | likely stays at whatever it had (below 286m) |
| `scale_pressure_6rps` | under pressure | under pressure | likely scales too (crosses 286m in most repeats) |
| `adequacy_8rps` | adequacy test at accumulated capacity | adequacy test at accumulated capacity | adequacy test at accumulated capacity |
| `recovery_2rps` | trends toward 1 (subject to accumulated replica count) | trends toward 1 | trends toward 1 |

These are hypotheses from the one-replica map only. Per CORR-19's principle, a live candidate behaving differently does not invalidate a structurally-correct run.

### 9. Counterbalanced Order (9 Official Trials, 3 Repeats Each) — Frozen
LOW=200%, MID=230%, HIGH=260%:
- Block 1: 200 → 230 → 260
- Block 2: 260 → 200 → 230
- Block 3: 230 → 260 → 200

Invalid runs remain preserved and replaced per the existing rule (CORR-17 §7 / CORR-18).

### 10. SLA and Selection Criteria Unchanged
Client p95 < 300ms, success rate ≥99%, all timeouts/connection-errors/generator drops explicitly reported — frozen, unchanged by this candidate-family redesign. Ranking hierarchy unchanged: (1) reject control-path instability, (2) reliability/SLA first, (3) efficiency/resource consumption second, (4) responsiveness/stability as tie-breakers, (5) unnecessary scaling at clearly-light load is a penalty. No winner is selected in this phase.

### 11. No Additional Calibration Required
**`NO_ADDITIONAL_CALIBRATION_REQUIRED_BEFORE_OFFICIAL_TUNING`.** Every phase of the redesigned schedule (3, 4, 5, 6 RPS from the 21-run Phase-11C.4D map; 8 RPS from the CORR-22 fresh-mode addendum; 2 RPS from Phase 10/CORR-22, trivially below every new boundary) already has final-path, repeated, raw-distribution evidence supporting its intended role.

### 12. Tests and Static Checks
New: `tests/unit/test_hpa_tuning_harness_static.py` (6 tests). Full unit suite: **142 passed** (up from 136, no regressions). `ruff check` clean on all changed/new Python. `bash -n` clean on both changed shell scripts. `hpa_tuning_v2.yaml` validated with `yaml.safe_load` and the project's own `parse_traffic_profile()`. `kubectl patch --dry-run=client` confirmed all three new target values (200/230/260) apply cleanly against the live HPA's schema without mutating live state (verified unchanged at 230% afterward). No official candidate was run; no canonical schedule was executed; no RL was run.

**Applies to phases**: 11C.4D (the empirical basis for this redesign), 11C.4E (this correction), official tuning restart (Block 1 Trial 1 at 200% may now proceed under `hpa_tuning_v2.yaml`/the new canonical schedule and the corrected traffic-delivery path from CORR-21/22 — not yet performed), 11D (Freeze, eventual winner selection unaffected in scope by this entry).

---

## CORR-25 — Standardized Exact-Interval Replica-Trajectory Integration and a `hpa_status_current_replicas` Timestamp-Labeling Correction (Phase 11C.4I)

### 1. Purpose
Block 1 (200%/230%/260%, one valid repeat each — phases 11C.4F/G/H) is now complete. This entry documents an offline, read-only audit of all three raw runs using one shared, tested integration method, and the one genuine bookkeeping correction it found. **No experiment validity changes. All three trials remain VALID.** No traffic was sent in this phase.

### 2. New Shared Tool
`scripts/integrate_replica_trajectory.py` (with `tests/unit/test_integrate_replica_trajectory.py`, 13 tests) operates only on an already-persisted `hpa_status.csv` — no Kubernetes/Prometheus/network access. Frozen integration rule for this and all future official trials: integrate every replica field as a right-continuous step function over the exact half-open interval `[0, 1230.0)` seconds; a sample's value applies from its own timestamp until the next sample; the final observed value (at or before 1230s) is carried forward to exactly 1230s; if the first sample is not at t=0, the interval `[0, first_sample)` uses the standardized known starting value (1 replica, per this project's standardized-start-state precondition); a row with a missing/unparseable value for a given field is skipped as a transition point (the previous segment continues through it) rather than corrupting the integral.

### 3. Recomputed, Normalized Block-1 Efficiency Table
All three trials re-integrated with the identical method over the identical `[0, 1230.0)` window:

| Metric | 200% | 230% | 260% |
|---|--:|--:|--:|
| HPA desired replica-seconds | 1935.00 | 1815.00 | 1635.00 |
| Deployment desired replica-seconds | 1935.00 | 1815.00 | 1635.00 |
| Current replica-seconds | 1935.00 | 1815.00 | 1635.00 |
| Ready replica-seconds | 1930.00 | 1810.00 | 1630.00 |
| Available replica-seconds | 1930.00 | 1810.00 | 1630.00 |
| Current replica-hours | 0.5375 | 0.5042 | 0.4542 |
| Average current replicas | 1.5732 | 1.4756 | 1.3293 |
| Peak current replicas | 2 | 2 | 2 |

The 230%/260% values match their individual reports (`phase_11c4g_*`, `phase_11c4h_*`) to sub-second rounding — those reports already used the exact `[0,1230)` convention correctly. The **200% report's** efficiency table (`phase_11c4f_*`) was computed over a truncated ~1225s observer window (1930.0/1925.0 replica-seconds, avg 1.576/1.571) rather than the exact `[0,1230)` window; the normalized values above (1935.00/1930.00, avg 1.5732/1.5691) are the corrected, authoritative Block-1 figures. This is a small windowing/bookkeeping difference (≈0.3% on the affected totals), not a validity issue — the underlying raw `hpa_status.csv` is untouched and `phase_11c4f_*` is **not** rewritten.

### 4. Genuine Correction Found: `hpa_status_current_replicas` Is Not the Same Signal as Deployment Ready/Available
All three individual reports' "Scaling Timeline" tables labeled a "new pod Ready" / "old pod terminated" milestone using the transition of `hpa_status_current_replicas` (the HPA controller's own internal replica-count bookkeeping field). Directly inspecting the raw rows around every transition in all three trials shows this field is **not** synchronized with the Deployment's own `deployment_status_ready_replicas`/`deployment_status_available_replicas` fields — the authoritative physical-pod-state evidence per CORR-20's hierarchy:

- **On scale-up**, `deployment_status_ready_replicas`/`available` reach 2 **one HPA-status-collector sample (~5–10s) earlier** than `hpa_status_current_replicas` does.
- **On scale-down**, `deployment_status_ready_replicas`/`available`/`replicas` drop to 1 **immediately**, at the same sample as the desired-replica decrease itself; `hpa_status_current_replicas` lags by one sample (~5–15s) before catching up.

Corrected timings (raw-timestamp evidence, ±1 HPA-status-collector sampling interval, ~5s, resolution — no sub-sample precision implied):

| Candidate | Trigger phase | Phase start | Desired/current=2 (Δ from phase start) | **Corrected** Ready=2 (Δ) | Originally reported "Ready" time |
|---|---|--:|--:|--:|--:|
| 200% | `aggressive_probe_4rps` | 90s | t=115.007 (+25.0s) | **t=120.001 (+30.0s)** | t=130.001s |
| 230% | `discriminator_5rps` | 210s | t=245.008 (+35.0s) | **t=250.001 (+40.0s)** | t=260.003s |
| 260% | `scale_pressure_6rps` | 330s | t=405.005 (+75.0s) | **t=410.005 (+80.0s)** | t=420.000s |

| Candidate | Recovery start | Desired/current/ready/avail all →1 (Δ from recovery start) | Originally reported "old pod terminated" time | Corrected stable-at-1 duration before t=1230 |
|---|--:|--:|--:|--:|
| 200% | 690s | **t=820.006 (+130.0s)** | t=835.001s | **≈409.99s** (was reported ≈395s) |
| 230% | 690s | **t=830.002 (+140.0s)** | t=845.005s | **≈399.998s** (was reported ≈385s) |
| 260% | 690s | **t=810.001 (+120.0s)** | t=810.001s (already correct — see note) | **≈419.999s** (was reported ≈405s) |

**No quantitative efficiency figure was corrupted by this**: the replica-seconds integrals in all three original reports correctly used `deployment_status_replicas`/`deployment_status_ready_replicas` (never `hpa_status_current_replicas`) as the integration field — confirmed by the shared tool reproducing them almost exactly (§3). Only the narrative "new pod Ready"/"old pod terminated" prose timestamps in the three reports' Scaling Timeline sections used the wrong field and are corrected here. `hpa_status_current_replicas` remains a useful corroborating field (it reflects what the HPA controller itself believed at each reconciliation) but must never again be used as the authoritative timestamp for a physical pod Ready/Available/terminated event — that role belongs to `deployment_status_ready_replicas`/`deployment_status_available_replicas` exclusively.

### 5. Structural Validity Reconfirmed, Independently, for All Three
Re-verified directly from raw artifacts (not from the individual reports' prose): all three trials used the exact frozen schedule (SHA-256 `4f648411e039f9229cd35a06374e6870dc75eca67e385e67f4f52bb0b5ec55ad`, 4950/4950 offered=sent in every trial), `connection_mode: "fresh"` in every trial's `metadata.json`, the `demo-app-loadtest` tunnel URL as `target_url` in every trial (never `localhost:8080`/port-forward), zero `kubectl_error_count` across all 252 `hpa_status.csv` samples in every trial, and exactly one transient (self-resolving, new-pod-not-yet-Ready) `metrics_collector.py` sample per trial, each coinciding with that trial's own corrected Ready transition. No duplicate application scraping evidence in any trial (per each trial's own live verification at the time). Generator dispatch lag (mean 1.2–1.8ms, max 45–53ms) and initial baseline CPU (5–9%) show no meaningful trend across the three trials run in sequence — no evidence of order/time-drift contamination within Block 1; the counterbalanced Block 2/3 design remains available to detect any residual order effect statistically, not because Block 1 itself shows a problem.

### 6. Descriptive Block-1 Summary (Not Statistical Proof, n=1 per Candidate)
200% scaled during `aggressive_probe_4rps` (4 RPS), 230% during `discriminator_5rps` (5 RPS), 260% during `scale_pressure_6rps` (6 RPS) — each candidate's single scale-up landed in the workload phase the Phase-11C.4E redesign specifically intended for it, each entering its trigger phase cleanly at 1 replica, each producing exactly one scale-up and one scale-down with zero oscillation, each reaching 100% request success with p95 well under the 300ms SLA in every phase. This is reported descriptively as a sanity check that the redesigned methodology is producing distinguishable candidate policies — **it is not statistical proof** (n=1 per candidate) and does not constitute or imply a ranking.

### 7. No Methodology Change
Candidate family (200/230/260), canonical schedule, SLA (p95<300ms, success≥99%), HPA behavior, resource settings, replica limits, Metrics Server cadence, and the Block 2/3 counterbalanced order are all unchanged and reconfirmed frozen. Block 1 is complete with one valid repeat per candidate.

**Applies to phases**: 11C.4F/G/H (the narrative timestamp-labeling correction applies to their Scaling Timeline sections; the reports themselves are not rewritten), 11C.4I (this correction), Block 2 (260→200→230, not yet started — may proceed under the same, now-audited methodology and the same shared integration tool for consistency), 11D (Freeze).

---

## CORR-26 — Official HPA Tuning Dataset Closed, 260% Selected as the Frozen Baseline (Phase 11C.4P)

### 1. Full Official Dataset: 9/9 VALID
All three counterbalanced blocks (Block 1: 200→230→260; Block 2: 260→200→230; Block 3: 230→260→200) are complete, giving exactly 3 valid repeats for each of the three CORR-24 candidates (200%/230%/260%), 9 official trials total. Independently re-verified from raw artifacts in this phase (not from prior report prose): every trial used the identical canonical schedule (SHA-256 `4f648411e039f9229cd35a06374e6870dc75eca67e385e67f4f52bb0b5ec55ad`, 4950/4950 offered=sent), `connection_mode: "fresh"`, the corrected `demo-app-loadtest` tunnel path (never `localhost:8080`/port-forward), identical application resources (100m/500m CPU, 64Mi/256Mi memory), identical HPA behavior (scaleUp stabilization=0s/+2 pods per 15s, scaleDown stabilization=120s/-1 pod per 60s), identical min/max replicas (1/8), identical Metrics Server cadence (15s), and zero `kubectl_error_count` samples in any trial's `hpa_status.csv`. No RL/custom controller was present in any trial. This structural re-verification is implemented as reusable code, not one-off manual inspection: `scripts/aggregate_hpa_tuning.py`'s `verify_trial_structural()`, driven by the new frozen manifest `experiments/manifests/hpa_tuning_official_9_trials.yaml`, raises rather than silently excludes on any mismatch — no mismatch was found for any of the 9 trials.

### 2. New Shared Offline Aggregation Tool
`scripts/aggregate_hpa_tuning.py` (with `tests/unit/test_aggregate_hpa_tuning.py`, 22 tests) is a narrow, read-only, offline aggregator: it reads only the frozen manifest and each trial's already-persisted `metadata.json`/`requests.csv`/`hpa_status.csv`/`_trial_metadata.json`, sends no traffic, makes no Kubernetes/Prometheus calls, and delegates all replica-trajectory integration to the existing, unmodified, shared `scripts/integrate_replica_trajectory.py` (CORR-25) rather than reimplementing that logic. Every scored metric (reliability, latency percentiles, tail counts, replica efficiency, responsiveness/stability) was recomputed independently from raw artifacts for this phase — no number was copied from any individual trial's own report. Outputs: `experiments/results/hpa_tuning_final/{per_trial_metrics.csv, candidate_summary.csv, blockwise_comparison.csv, phase_summary.csv, selection_decision.json}` plus four descriptive plots (current replica-seconds, whole-run p95, trigger-to-Ready, >300ms fraction — each showing all 3 individual repeats plus their median, non-truncated axes).

### 3. Latency-Percentile Semantics (Frozen for This Aggregation, Applied Identically to Every Candidate)
A request's client latency is included in latency percentiles / `>300ms` tail counts iff `requests.csv` recorded a parseable `client_latency_seconds` value. A `timeout` outcome *does* record a latency value (the generator's bounded wait, `self.timeout_seconds`, per `loadgen/traffic_generator.py`) and is therefore included; a `connection_error` has no response and no latency value and is excluded from latency percentiles (but always included in reliability counts). Percentiles use `numpy.percentile` (linear interpolation), the CORR-13-frozen authoritative estimator — not the rank-based integer indexing (`latencies[int(n*p)]`) that individual trial `metadata.json` files use for historical traceability. This produces small (sub-1ms to ~1ms at n≈4950) differences from a few individual trial reports' previously-cited whole-run p99 figures; this is expected, documented here, and does not change any SLA verdict or the selection outcome.

### 4. Recomputed Candidate-Level Summary (3 Valid Repeats Each, Median/Min/Max)

| Metric | 200% | 230% | 260% |
|---|--:|--:|--:|
| Valid runs / SLA-passing runs | 3/3 | 3/3 | 3/3 |
| Aggregate success rate (9-trial-family total) | 99.98% | 100% | 100% |
| Total non-success outcomes | 3 (1×5xx, 1×timeout, 1×conn-error) | 0 | 0 |
| Median whole-run p95 | 132.7ms | 133.1ms | 117.0ms |
| Median whole-run p99 | 166.1ms | 147.5ms | 170.8ms |
| Median `>300ms` fraction | 0.20% | 0.32% | 0.53% |
| Median current replica-seconds `[0,1230)` | 1920.00 | 1815.00 | **1680.00** |
| Median Ready replica-seconds | 1915.00 | 1810.00 | 1675.00 |
| Median replica-hours | 0.5333 | 0.5042 | 0.4667 |
| First-scale-up phase (3/3 consistency) | `aggressive_probe_4rps` (3/3) | `discriminator_5rps` (3/3) | `scale_pressure_6rps` (3/3) |
| Median trigger-phase-start→Ready | 40.0s | 40.0s | 40.0s |
| Median recovery-phase-start→Ready=1 | 130.0s | 135.0s | 120.0s |
| Peak replicas / oscillation | 2 / none | 2 / none | 2 / none |

All figures independently recomputed by `scripts/aggregate_hpa_tuning.py` from raw artifacts; full per-trial values in `experiments/results/hpa_tuning_final/per_trial_metrics.csv`.

### 5. Frozen Ranking Hierarchy Applied — No Ad-Hoc Composite Score
Per CORR-17 §6 / CORR-18 §10 (unaltered): (A) structural validity — all 9 trials pass (§1); (B) reliability/SLA (client p95<300ms AND success≥99%) — **all three candidates pass in all 3 repeats**, no candidate is eliminated at this stage; (C) efficiency among Stage-B survivors — **260% has the lowest median current replica-seconds** (1680.00 vs 230%'s 1815.00 and 200%'s 1920.00), a strict, non-tied ranking; (D) responsiveness/stability as tie-breaker — not required (Stage C did not tie); trigger-to-Ready timing is in fact statistically indistinguishable across all three candidates (median 40.0s each) and recovery timing favors 260% as well (120.0s vs 230%'s 135.0s and 200%'s 130.0s), so Stage D would not have changed the outcome even if consulted; (E) unnecessary light-load scaling penalty — not triggered, no candidate scaled during `baseline_3rps` in any of its 3 repeats.

### 6. Selected Target: 260%. Runner-Up: 230%.
**260% is selected** as the frozen HPA baseline. It passed reliability/SLA in all 3 repeats with zero non-success outcomes across all 3 trials, and used **~7.4% fewer replica-seconds** than the 230% runner-up over the exact `[0,1230)` scored window (1680.00 vs 1814.99 median replica-seconds; a resource-consumption proxy, not a monetary cost figure — no cloud-pricing model is applied). 230% (the runner-up) was not selected because, despite also passing reliability/SLA cleanly, it consumed measurably more replica-seconds for comparable trigger/recovery responsiveness, making 260% the more resource-efficient choice at equal reliability under this tuning workload. 200% is descriptively the least favorable candidate here: it is the only candidate that produced any non-success outcome across the 9-trial dataset (one isolated 5xx in Block 2, one connection-error+timeout pair in Block 3, both `SOURCE_UNRESOLVED_FROM_PRESERVED_EVIDENCE` per §7 below) and consumed the most replica-seconds of the three, though it still passed the frozen SLA in all 3 repeats and is not disqualified by the frozen criteria.

### 7. Reliability/Tail Caveats Preserved, Not Elevated Into a New SLA
200%'s two non-2xx events (Block 2's isolated `server_error_5xx`; Block 3's `connection_error`+`timeout` pair, the latter within ~1s of that trial's own scale-down transition) remain `SOURCE_UNRESOLVED_FROM_PRESERVED_EVIDENCE` — reported as preserved reliability outcomes and, where a precise timing relationship exists, as a timing correlation only, never attributed to the HPA target, the application, kube-proxy, the Service tunnel, or pod termination without direct evidence. Scattered `>300ms` tail-latency bursts occur across all three candidates and multiple phases (`CAUSE_UNRESOLVED_FROM_CURRENT_EVIDENCE`); 260% has the highest median `>300ms` fraction of the three candidates (0.53% vs 200%'s 0.20% and 230%'s 0.32%) and the highest median whole-run p99 (170.8ms) — both remain far inside the frozen SLA (p95<300ms) and are recorded here as an honest secondary observation about the selected candidate, not suppressed.

### 8. Candidate Tuning Is Complete
`HPA_TUNING_DATASET_COMPLETE_9_OF_9_VALID`. No further HPA hyperparameter tuning (i.e., no further candidate value beyond 200%/230%/260%, no further official trial against `hpa_tuning_v2.yaml` or any successor tuning schedule) is permitted unless this methodology is explicitly reopened by the user with a new, documented correction. `configs/hpa.yaml` and `k8s/hpa/hpa.yaml` are updated to `averageUtilization: 260`; the live cluster HPA was patched to 260% via `scripts/set_hpa_target.sh 260` as a configuration-finalization step (no traffic sent) and verified live, alongside an unchanged Deployment (1/1/1/1), unchanged min/max (1/8), unchanged behavior, unchanged application resources, Metrics Server still 15s, Prometheus healthy, single-scrape monitoring intact, `demo-app-loadtest` still installed, Kind still exited, and no RL controller present. `scripts/set_hpa_target.sh` continues to support the full historical 200/230/260 candidate set (unchanged) for reproducibility.

### 9. Tuning-vs-Evaluation Separation (Binding Going Forward)
260% is now the **fixed HPA baseline** for all subsequent RL-vs-HPA work. This 9-trial `hpa_tuning_v2.yaml` dataset is HPA hyperparameter **tuning** data and must never be reused or cited as final evidence that HPA outperforms or underperforms RL. Any future HPA-vs-RL comparison must be conducted on the separately frozen, not-yet-used held-out evaluation workload (`configs/traffic_profiles/unseen_eval.yaml`), which this phase did not read, use, or modify.

**Applies to phases**: 11C.4E/F/G/H/I/J/K/L/M/N/O (the tuning campaign this entry closes out), 11C.4P (this correction, final HPA aggregation and selection), 11D (Freeze — superseded/completed by this entry), all future RL environment/training/evaluation phases (260% is now the fixed HPA baseline; the separate held-out evaluation workload governs any final HPA-vs-RL claim).

---

## CORR-27 — RL MDP Contract V2: CPU-Clip, HPA-Coupling, Gating, and Held-Out-Leakage Corrections (Phase 15A.1)

### 1. Context
Phase 15A froze an RL MDP contract v1 (`docs/rl_mdp_spec.md`, `configs/rl.yaml`, `configs/reward.yaml`) before any Gymnasium environment code existed. A pre-implementation review of that v1 draft found seven specification-integrity issues, corrected here as v2 (`rl_contract_version: 1→2`, `reward_contract_version: 1→2`) before any environment/training code is written. No RL code, Gymnasium environment, or PPO training exists at the time of this correction either — this remains a pure specification phase. No traffic was sent, no HPA tuning was reopened, and the frozen HPA baseline (`averageUtilization=260`, CORR-26) was not touched.

### 2. CPU Normalization Was Clipped Below the Physical Ceiling
v1's `cpu_per_replica` feature (and its delta) clipped its raw range at `4.0`× the 100m CPU request. The true physical ceiling is `limit/request = 500m/100m = 5.0`×, and Phase-10's single-replica saturation calibration (10 RPS) recorded a steady-state mean of `4.63`× request (individual samples reaching `5.0`× exactly, the hard cgroup limit) — both above v1's `4.0`× clip. v1 would have silently truncated genuine severe-pressure readings. **v2 freezes the raw clip range at `[0.0, 5.0]`** for both `cpu_per_replica` and `cpu_per_replica_delta` (symmetric `[-5.0, 5.0]`).

### 3. Overclaimed HPA-Equivalence Wording
v1 stated the RL CPU feature is "defined identically to" and "means the same thing as" HPA's own `averageUtilization`. This overclaims: HPA's actual computed value additionally depends on Metrics Server sampling/staleness behavior, eligible-pod selection, not-yet-Ready-pod handling, HPA reconciliation timing, and `kube-controller-manager`'s own conservative recalculation rules — none of which the RL feature reproduces. **v2 corrects the wording to "a nominally comparable CPU-request-relative utilization proxy," not literal equivalence.** The feature's formula itself is unchanged.

### 4. Held-Out Evaluation Leakage, Documented Honestly
Phase 15A's justification for `request_rate_norm`'s `100` RPS raw clip explicitly cited `configs/traffic_profiles/unseen_eval.yaml`'s 80 RPS burst magnitude as supporting evidence ("unseen_eval bursts reaching up to 80 RPS"). This is a genuine violation of the precommitted rule that held-out evaluation traffic must never influence state-feature/normalization design, reward tuning, PPO tuning, environment calibration, or model selection — even though the file itself was never executed, never used as training data, and remains byte-for-byte unmodified.

**Correction, not concealment**: the `100` RPS bound is **retained** (it remains a reasonable, conservative value) but **re-justified using only non-held-out information**: development-candidate profile peaks (`burst.yaml`=70, `ramp.yaml`=60, `step.yaml`=50 RPS — none of which is the held-out file), giving ~30 RPS of headroom above the highest known development profile without reference to `unseen_eval.yaml`'s own magnitude. `configs/rl.yaml`'s comment for this feature was rewritten accordingly and a new test (`test_request_rate_bound_justification_does_not_cite_unseen_eval`) asserts the string `unseen_eval` never appears in `configs/rl.yaml`/`configs/reward.yaml`.

**Consequence for `unseen_eval.yaml`'s status**: this file is **not deleted, not rewritten, not run** — it is preserved exactly as before. But it can **no longer be described as a perfectly untouched formal held-out workload from an information-design standpoint**, since its parameters were inspected and (briefly) used during Phase 15A's normalization design. It is marked, in `docs/rl_mdp_spec.md` §22:

```
NOT_ELIGIBLE_AS_THE_SOLE_FORMAL_FINAL_HELD_OUT_EVALUATION_PROFILE
```

**Required future action**: a dedicated workload-freeze phase, occurring after the v2 contract is implemented and after training/validation workload families are frozen, must define a **new** formal held-out evaluation workload (new file, new seed, new shape, never inspected during any state/reward/PPO design decision). This phase (15A.1) does not create or inspect that replacement file. Once frozen, the new file must never again influence any further model/reward/state/hyperparameter selection.

### 5. Symmetric 60-Second Action Gating Was Too Restrictive
v1 froze a **symmetric** 60-second minimum spacing between all effective scaling actions (both directions). This could unnecessarily delay an urgently-needed additional scale-up during a severe, fast-rising workload increase, and it was in tension with the frozen ±1 discrete action space's own responsiveness rationale. **v2 freezes an asymmetric rule** (`docs/rl_mdp_spec.md` §12):
- **Pending-scale gating** (unchanged concept from v1, both directions): while a previous capacity transition has not yet realized (`deployment_spec_desired_replicas != deployment_status_ready_replicas`), any non-HOLD action is converted to effective HOLD (`action_override_reason="pending_scale_gate"`).
- **Scale-up: no additional cooldown.** Once observations are valid and no transition is pending, `SCALE_UP_ONE` is effective at the very next 30-second decision point — repeated, necessary scale-ups can occur once per decision interval.
- **Scale-down: retains the 60-second minimum spacing** (`configs/experiment.yaml: scaling.cooldown_seconds`, unchanged numeric value, now direction-specific rather than symmetric; `action_override_reason="scale_down_cooldown"`).
- **Emergency reversal is explicitly permitted**: a needed `SCALE_UP_ONE` is never blocked merely because a `SCALE_DOWN_ONE` occurred 30 seconds earlier (provided observations are valid and no transition is pending) — the reward's reversal penalty may still discourage genuine oscillation, but the deterministic guardrail never prevents a needed scale-up.

The previously-stated "1→8 in 210 seconds" full-traverse timing claim is corrected to be stated **relative to the first effective scale-up**, not episode start, and is described as an expected-case estimate (assuming ~10s empirical readiness lag per step, per CORR-25) rather than a hard bound the environment enforces.

### 6. Resource Reward Must Use Current, Not Ready, Replicas
Phase 11C.4P / CORR-26 selected the HPA baseline primarily using **Deployment current replica-seconds** (Ready replica-seconds were explicit secondary corroboration only). v1's resource reward term used `ready_replicas`, conflating resource *consumption* (a newly-created-but-not-yet-Ready pod already consumes real compute) with serving *capacity*. **v2 freezes**: `resource_fraction = (deployment_current_replicas - 1) / 7`. Ready replicas remain used exclusively for per-replica CPU/memory/in-progress normalization, the `ready_replica_norm` observation feature, and pending-scale gating — i.e., serving-capacity questions only, never resource-consumption accounting.

### 7. Efficiency-Bonus Band Decoupled From the Selected HPA Target
v1 set the efficiency-bonus CPU band's upper bound to `2.6`× request specifically because HPA's selected baseline (CORR-26) is `averageUtilization=260`. This directly and unnecessarily coupled the RL reward to an unrelated controller's own hyperparameter-search outcome. **v2 re-derives the upper bound from application-only, no-HPA calibration evidence** (Phase 11C.4D's controlled, HPA-deleted, fixed-1-replica, final-path Metrics-API CPU map, CORR-23 §4): at 6 RPS, steady CPU is `297.4m` ≈ `2.97x` ≈ `3.0x` request, while client p95 stays `<=~135ms` (well under the 300ms SLA) at every tested level 3–7 RPS. **v2 freezes the efficiency band at `[1.0, 3.0]`× request** — deliberately different from `2.6` to make the decoupling auditable rather than merely asserted (`3.0 != 2.6`). The bonus term itself was judged not redundant with the resource term (only CPU-band membership distinguishes "efficiently loaded" from "idle") and is retained, not removed.

### 8. Boundary Penalty Was Too Broad; Action/Reversal Semantics Clarified
v1's boundary penalty fired whenever `requested_replica_delta != effective_replica_delta` — this conflates genuine bound violations with pending-scale gating, scale-down cooldown, and telemetry-failure HOLDs, none of which represent the policy requesting a physically impossible replica count. **v2 freezes an explicit `action_override_reason` enum** (`none`, `replica_bound_clamp`, `pending_scale_gate`, `scale_down_cooldown`, `stale_or_missing_metric`, `prometheus_unavailable`, `policy_inference_failure`, `k8s_api_failure`) and narrows: the boundary penalty fires **only** for `replica_bound_clamp`. The action-churn penalty is now defined on the **requested** action (on any step the policy could meaningfully evaluate, i.e. not a telemetry-failure override) — it fires for `none`, `replica_bound_clamp`, `pending_scale_gate`, and `scale_down_cooldown`, but **never double-counts** alongside the boundary penalty for the same `replica_bound_clamp` event. The reversal penalty is computed only from **effective** (physically realized) scaling actions.

### 9. Zero-Traffic Latency Is a Legitimate Semantic State, Not a Telemetry Failure
v1 did not distinguish "no requests observed" from "latency telemetry unavailable." **v2 freezes**: if `request_rate_rps == 0` (a genuine non-null zero) and the latency histogram has no observations, `latency_p95_seconds` is defined as a genuine `0.0` (`latency_observation_state="no_traffic"`), participating normally in normalization and EWMA updates. If `request_rate_rps > 0` but latency is unavailable/stale, the step is structurally invalid (`latency_observation_state="invalid"`) and triggers the deterministic HOLD/invalid-step machinery — the midpoint-fallback rule never substitutes for a genuine invalid reading here.

### 10. First-Decision and Invalid-Step-Termination Semantics Precisely Frozen
v1 left episode-start observation semantics implicit and used an ambiguous "more than 3 consecutive invalid steps" termination rule. **v2 freezes an exact reset sequence** (Deployment→1 replica, confirm Ready, 60s pre-episode stabilization used only to seed EWMA state, traffic starts at episode `t=0`, first decision at `t=0` uses only stabilization-window data) and an **exact invalid-step counter rule**: `consecutive_invalid_steps` increments on any `sample_valid==False` step, resets to 0 on any single fully-valid step, HOLDs for counts 1–3, and terminates the episode (`terminated=True`) on the 4th consecutive invalid step (120 seconds of persistent decision-time telemetry failure at the 30s decision interval). Zero-Ready-replicas for >120s remains a separate, explicit termination condition.

### 11. Reward on Invalid Steps Must Not Use Fabricated Observations
**v2 freezes**: on a step overridden solely for a telemetry-failure reason, `reward_validity="partial"` and every reward component is zeroed (`R_sla=R_reliability(unless independently valid)=R_action=R_reversal=R_boundary=R_efficiency=0`) **except** `R_resource`, which may still be computed if `deployment_current_replicas` was independently obtained (a separate Kubernetes/kube-state-metrics query from whichever application-level metric failed) — ongoing resource consumption is a real physical fact independent of application-telemetry health.

### 12. Reward Is Not a Live-Inference Input (Clarification, Not a Behavior Change)
v2 makes explicit what v1's prose left implicit: reward is required only for offline training/validation/auditing. A deployed inference controller maps `observation -> action` directly and never computes or consults a reward value as part of live action selection.

### 13. Transition-Log Schema Extended
New fields added to the frozen transition-record schema to distinguish v2's corrected semantics: `deployment_current_replicas_before/after`, `available_replicas_before/after`, `action_override_reason`, `bound_clamped`, `pending_scale`, `consecutive_invalid_steps`, `reward_validity`, `latency_observation_state`. Requested and effective actions continue to be recorded separately, unchanged from v1.

### 14. Versioning
`rl_contract_version: 1→2`, `reward_contract_version: 1→2`. The action encoding (`0=-1, 1=0, 2=+1`) and the observation vector's shape/order (still 11 elements, same names, same positions) are **unchanged** — the version bump reflects changed normalization and reward *semantics* underneath several entries, not a structural change, per the task's own explicit instruction that a semantic-only change still mandates a version bump. `configs/rl.yaml`'s `observation_v1` block was renamed `observation_v2` (its content changed enough that leaving it under the old name would itself be a silent v1 mutation).

### 15. Tests and Static Checks
`tests/unit/test_rl_mdp_contract.py` expanded from 18 to 36 tests (18 new), covering: v2 observation-vector exact match (including the corrected `5.0`× CPU clip), `rl_contract_version==2`/`reward_contract_version==2`, absence of the old `observation_v1` block, the asymmetric gating rule's config-level and documented-prose invariants (no symmetric/scale-up-specific cooldown key; scale-down cooldown exactly 60s from the existing `configs/experiment.yaml` value; pending-scale-gating and emergency-reversal rules documented), resource-reward now sourced from `deployment_current_replicas`, boundary penalty narrowed to `replica_bound_clamp` only, efficiency band changed to `3.0` and explicitly not equal to `2.6`, zero-traffic-vs-invalid-latency semantics, the 4th-consecutive-invalid-step termination rule, invalid-step `reward_validity="partial"` handling, the selected HPA baseline still `260` (unchanged), and `unseen_eval.yaml` marked `NOT_ELIGIBLE_AS_THE_SOLE_FORMAL_FINAL_HELD_OUT_EVALUATION_PROFILE` with zero references to it anywhere in the RL/reward configs. Full suite: **213 passed** (up from 195, no regressions). `ruff check` clean on the changed test file. No environment/controller/training Python code was added — `configs/rl.yaml`/`configs/reward.yaml` are YAML, and `docs/rl_mdp_spec.md` is Markdown.

### 16. HPA Baseline and Held-Out Workload File Contents Unaffected
`configs/hpa.yaml`/`k8s/hpa/hpa.yaml` remain at `averageUtilization=260` (CORR-26), unchanged and reconfirmed by test in this phase. `configs/traffic_profiles/unseen_eval.yaml`'s file content (including its distinguishing `seed: 99` line) is unchanged — only its *documented eligibility status* as a clean sole final held-out workload changed (§4 above). No traffic was sent and no Kubernetes cluster interaction occurred in this phase.

**Applies to phases**: 15A (the v1 draft this corrects), 15A.1 (this correction), all future RL environment-implementation/training/evaluation phases (must implement v2, not v1; must not reuse `unseen_eval.yaml` as a final held-out claim until a new, clean held-out workload is frozen in a dedicated future phase).

---

## CORR-28 — RL Runtime Dependency Compatibility and Pre-Environment Readiness Freeze (Phase 15A.2)

### 1. Context
Before any Gymnasium environment or PPO training code is written, this phase inspected whether `requirements/rl.txt`'s pinned RL stack could actually be installed together without conflict. It could not, as pinned. This entry documents the defect, the correction, and the verified resolved runtime — the RL MDP contract (v2, CORR-27) is unchanged; no environment/controller code was written; no traffic was sent; the HPA baseline (260%, CORR-26) was not touched.

### 2. The Original Incompatibility, Verified From Published Package Metadata (Not Assumed)
`requirements/rl.txt` pinned `stable-baselines3==2.4.1` and `gymnasium==1.1.1` together. Downloading both wheels and reading their `METADATA` directly (not inferring from changelogs or documentation) confirms:
- `stable-baselines3==2.4.1` declares `Requires-Dist: gymnasium<1.1.0,>=0.29.1`.
- `gymnasium==1.1.1` is **not** `<1.1.0` (`1.1.1 > 1.1.0`).

These two pins could never resolve together under normal `pip`/dependency-resolver behavior — a genuine specification defect in `requirements/rl.txt`, not a hypothetical one, confirmed by direct package-metadata inspection before any correction was applied.

### 3. Corrected, Minimal Version Bump — Verified Compatible From the Same Source
`stable-baselines3==2.6.0`'s published `METADATA` declares `Requires-Dist: gymnasium<1.2.0,>=0.29.1` and `Requires-Dist: torch<3.0,>=2.3` — both satisfied by the already-desired `gymnasium==1.1.1` and `torch==2.5.1`. This is the smallest SB3 version bump that restores compatibility; no other package in `requirements/rl.txt` (`tensorboard==2.18.0`, `scikit-learn==1.6.1`, `matplotlib==3.10.9`) required any change, and the shared `requirements/base.txt` pin `numpy==1.26.4` (used across the whole project, including `collectors/`, `loadgen/`, and `scripts/aggregate_hpa_tuning.py`) remained untouched and compatible with both `gymnasium==1.1.1`'s `numpy>=1.21.0` and `stable-baselines3==2.6.0`'s `numpy>=1.20,<3.0` constraints. No newer-than-necessary SB3/PyTorch release was adopted; no alternative RL framework (Ray/RLlib, TensorFlow-based RL, CleanRL, JAX-based stacks, sb3-contrib) was introduced.

### 4. Frozen Runtime Stack
```
stable-baselines3==2.6.0
gymnasium==1.1.1
torch==2.5.1
tensorboard==2.18.0
```
`requirements/rl.txt` updated accordingly (comment block explains the exact incompatibility and its resolution inline, citing the verified METADATA constraints).

### 5. Installed Into the Existing `.venv` — No Second Environment Created
`.venv/bin/python -m pip install -r requirements/rl.txt` was run against the project's existing `.venv` (Python 3.10.14, arm64, macOS 26.6.2, pip 26.2.1) — the same environment used by every other phase of this project. No second/competing Python environment was created. Full resolved environment snapshot (via `pip freeze`) persisted, non-overwriting, at `persistent-data/metadata/rl_runtime_environment_phase15a2.txt` (104 lines). `python -m pip check` reports **"No broken requirements found."** (exit code 0).

### 6. Runtime Import and Version Verification
```
Gymnasium:          1.1.1
Stable-Baselines3:  2.6.0
PyTorch:            2.5.1
TensorBoard:        2.18.0
numpy (unaffected): 1.26.4
```
All four import cleanly. Informational only, no design implication drawn: `torch.backends.mps.is_available()==True`, `torch.backends.mps.is_built()==True` on this Apple Silicon host — MPS is available but this phase does not adopt it; a small MLP PPO policy (`net_arch=[128,128]`, 11-dim input) is cheap enough that CPU-vs-MPS should be benchmarked empirically in a later training phase, not assumed.

### 7. Gymnasium API Smoke Test
`gymnasium.make("CartPole-v1")`, `env.reset(seed=123)`, and 5 random `env.step(action)` calls all returned the standard 5-tuple (`observation, reward, terminated, truncated, info`) with no errors; `env.close()` succeeded. This is a dependency/runtime smoke test only — CartPole is unrelated to the project's actual autoscaling problem and no project-specific environment code was written or exercised.

### 8. Stable-Baselines3 Compatibility Smoke Test
A tiny synthetic `gymnasium.Env` (`Box(shape=(11,))` / `Discrete(3)` — matching only the *shape* of the frozen v2 contract, not its semantics) passed `stable_baselines3.common.env_checker.check_env()`. `PPO("MlpPolicy", env, ...)` constructed successfully using `configs/rl.yaml`'s **exact** hyperparameter values (`learning_rate=3e-4`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.01`, `vf_coef=0.5`, `max_grad_norm=0.5`, `net_arch=[128,128]` via `policy_kwargs`, `n_epochs=10`; only `n_steps`/`batch_size` were reduced to `8` for smoke-test speed, which does not affect field-acceptance). `model.predict()` returned a valid discrete action. **No `.learn()` call was made; no model checkpoint or tensorboard log directory was created** (confirmed by filesystem check: `persistent-data/rl/` and `persistent-data/models/` remained empty after the smoke test).

### 9. PPO Configuration Compatibility Review — All Fields Compatible, None Changed
Every field currently configured in `configs/rl.yaml: ppo` (`policy`, `learning_rate`, `n_steps`, `batch_size`, `n_epochs`, `gamma`, `gae_lambda`, `clip_range`, `ent_coef`, `vf_coef`, `max_grad_norm`) was confirmed, by direct `inspect.signature(PPO.__init__)` introspection against the newly-installed SB3 2.6.0, to be an accepted constructor parameter name — none renamed, deprecated, or removed between the versions this project has ever referenced. `net_arch` is correctly applied via `policy_kwargs={"net_arch": [...]}` (never a top-level PPO kwarg in any SB3 version) — `configs/rl.yaml`'s existing structure already does this correctly (it lives inside the `ppo:` block for human organization, and the future training code must pass it inside `policy_kwargs`, not as a bare kwarg — noted for the implementation phase, not a defect requiring a config change now). **No hyperparameter values were tuned or changed.**

### 10. Documentation Bookkeeping Corrections (Phase 15A.1 Report — Not Rewritten, Clarified Here)

**A. Transition-schema field count.** Phase 15A.1's task-output report (§15/§16) and `docs/rl_mdp_spec.md` describe the v2 transition-log schema as adding "8 new fields." Recounting the actual listed additions: `deployment_current_replicas_before`, `deployment_current_replicas_after`, `available_replicas_before`, `available_replicas_after`, `action_override_reason`, `bound_clamped`, `pending_scale`, `consecutive_invalid_steps`, `reward_validity`, `latency_observation_state` — **10 items**, not 8. This is a documentation bookkeeping correction only; the schema's actual field *set* (verified by re-reading `docs/rl_mdp_spec.md` §14) already correctly lists all 10 fields — only the prose's summary count was wrong. No schema semantic change is made or required.

**B. Kubernetes-interaction wording.** Phase 15A.1's report states, in several places, that "no Kubernetes cluster interaction occurred" / "no Kubernetes interaction occurred" during that phase. This is too broad: that phase's own §20 explicitly performed a live, read-only `kubectl get hpa -n autoscaler-app` (and a pods check) to verify the HPA baseline was still 260% after its config edits. The precise, correct characterization, going forward: **"No workload traffic, scaling experiment, RL controller action, or mutating Kubernetes operation occurred during Phase 15A.1; only a read-only live HPA/pod verification was performed."** Per standing project practice, the original Phase 15A.1 report is **not** rewritten — this entry is the authoritative correction for both bookkeeping points.

### 11. RL MDP Contract, HPA Baseline, and Held-Out Workload — All Unaffected
`rl_contract_version` remains `2`; `reward_contract_version` remains `2`; the 11-element observation vector's order/shape and the `{-1,0,+1}` action encoding are unchanged (independently reconfirmed by new tests in this phase, `tests/unit/test_rl_dependency_compatibility.py::TestRlContractUnchangedByDependencyPhase`). `configs/hpa.yaml`/`k8s/hpa/hpa.yaml` remain `averageUtilization=260`, reconfirmed both by config-file test and by a live, read-only `kubectl get hpa` check at the end of this phase. `configs/traffic_profiles/unseen_eval.yaml` was not inspected, used, or modified in this phase — its CORR-27 `NOT_ELIGIBLE_AS_THE_SOLE_FORMAL_FINAL_HELD_OUT_EVALUATION_PROFILE` status is unaffected and unchanged. No training/validation workload split was created or attempted. `autoscaler/src/` remains empty (verified by test) — no environment/controller code was introduced.

### 12. Tests
`tests/unit/test_rl_dependency_compatibility.py` (new, 20 tests): exact-pin checks against `requirements/rl.txt` (SB3 2.6.0, Gymnasium 1.1.1, Torch 2.5.1, TensorBoard 2.18.0); absence of any disallowed RL framework reference; installed-version checks via `importlib.metadata` matching those pins; a **metadata-driven** (not hardcoded-belief) compatibility check that reads `stable-baselines3`'s own declared `gymnasium`/`torch` requirement specifiers via `importlib.metadata.requires()` and verifies the installed versions satisfy them using `packaging.specifiers`; a regression-documenting test that the *original* `gymnasium>=0.29.1,<1.1.0` constraint (SB3 2.4.1) would **not** have accepted `1.1.1` (proving the defect was real, using an offline-safe literal constraint string rather than a live network call); confirmation that every configured PPO field is a currently-accepted `PPO.__init__` parameter name; and re-confirmation that `rl_contract_version==2`, `reward_contract_version==2`, the observation vector is still 11 elements in order, the action map is unchanged, the HPA baseline is still 260, and no RL environment code was introduced in `autoscaler/src/`. Full suite: **233 passed** (up from 213, no regressions). `ruff check` clean on the new test file.

### 13. No Training, No Environment Implementation, No Traffic
No `.learn()` call was made anywhere in this phase. No model checkpoint, tensorboard log, or training artifact was produced. No Gymnasium `Env` subclass for the actual autoscaling problem was written. No traffic was generated or sent. No mutating Kubernetes operation occurred — only read-only verification (§11).

**Applies to phases**: 15A/15A.1 (the contract this phase's dependency stack must eventually implement, unchanged by this correction), 15A.2 (this correction), the future Gymnasium-environment-implementation phase (must use this now-verified-installable stack; the project's existing `.venv` already has it installed and `pip check`-clean, so no further dependency-resolution work should be needed to begin writing environment code).

---

## CORR-29 — RL Gymnasium Environment Mechanics and Pluggable-Backend Contract (Phase 15B.1)

### 1. Context
This phase implements the project's **first actual RL environment code** — a real `gymnasium.Env` subclass (`autoscaler/src/rl_env/environment.py: KubernetesAutoscalerEnv`) implementing the frozen MDP contract v2 (docs/rl_mdp_spec.md, CORR-27) exactly, obtaining all workload/replica state through an injected, pluggable `DynamicsBackend`. **No empirical CPU/latency/error dynamics model was built or assumed.** No PPO training occurred. No traffic was sent. No mutating Kubernetes operation occurred (only a read-only `kubectl get hpa`/`get pods` check at the end, confirming `averageUtilization=260` unchanged).

### 2. Phase 15A.2 Runtime-Report Correction: `[128,128]` Net-Arch Mapping Was Not Actually Exercised
Phase 15A.2's SB3 compatibility smoke test constructed a `PPO` model and inspected its network, but did **not** pass `policy_kwargs={"net_arch": [...]}` — so the inspected network used SB3's own default `[64, 64]` hidden layers, not the project's configured `[128, 128]`. Reproduced directly in this phase (`PPO("MlpPolicy", env, ...)` with no `policy_kwargs`) confirms the resulting `mlp_extractor.policy_net` hidden layer sizes are `[64, 64]`. This means Phase 15A.2's smoke test proved SB3/Gymnasium/PyTorch import and construct correctly, and that every `configs/rl.yaml: ppo` field name is an accepted `PPO.__init__` parameter — but it did **not** prove the `net_arch: [128, 128]` config value actually reaches the constructed policy network. This is a documentation/runtime-test bookkeeping gap only; it does not invalidate Phase 15A.2's dependency-compatibility, Gymnasium-compatibility, SB3-compatibility, or MDP-v2 findings, all of which remain correct. **Corrected in this phase**: `autoscaler/src/rl_env/ppo_config.py` (`build_ppo_kwargs()`) is a small, tested adapter that maps `configs/rl.yaml: ppo` (including `net_arch` via `policy_kwargs`) to `PPO.__init__` kwargs; `tests/unit/test_rl_env_ppo_config.py` constructs a real `PPO` instance against the new environment using this adapter and asserts the resulting `mlp_extractor.policy_net` hidden layers are exactly `[128, 128]` — passing, confirming the wiring now genuinely works. The original Phase 15A.2 task-output report is not rewritten, per standing project practice; this entry is the authoritative correction.

### 3. Environment/Backend Architecture
```
autoscaler/src/rl_env/
    types.py            -- shared dataclasses (RawMetricSnapshot, ReplicaState,
                            BackendSnapshot, GatingResult, ValidityResult) and the
                            frozen ACTION_MAP / action_override_reason / latency-state enums
    backend.py           -- DynamicsBackend Protocol (reset/current_snapshot/
                             set_desired_replicas/advance) -- the environment
                             never hard-codes Kubernetes or Prometheus operations
    scripted_backend.py   -- ScriptedBackend: TEST_ONLY_SYNTHETIC_METRICS,
                             deterministic, timeline-driven; implements the
                             empirically-grounded replica-transition mechanics
                             (delayed scale-up readiness, immediate scale-down)
    observation.py        -- determine_validity(), EwmaTracker, HistoryState,
                             build_observation() -- the frozen 11-element V2 vector
    gating.py             -- compute_effective_action() -- the frozen V2
                             asymmetric action-gating rule
    reward.py             -- compute_reward() -- the frozen V2 7-term reward formula
    config.py              -- EnvConfig + load_env_config_from_yaml() (reads
                              configs/rl.yaml/reward.yaml/experiment.yaml so
                              normalization/reward numbers are never duplicated
                              magic constants)
    environment.py         -- KubernetesAutoscalerEnv(gymnasium.Env) -- ties all
                              of the above together per the frozen step-order
                              contract
    ppo_config.py           -- build_ppo_kwargs(): the [128,128]-net_arch-wiring
                               adapter (sec.2 above)
```
This separation means a future empirical simulator backend or a future live Kubernetes/Prometheus backend can be added by implementing `DynamicsBackend` alone, without touching observation/gating/reward/environment code.

### 4. No Fabricated Performance Model
`ScriptedBackend` does **not** compute `latency = f(load, replicas)`, `cpu = f(requests)`, or any other invented workload-response formula. It plays back an explicitly supplied, exact-time-keyed dictionary of `RawMetricSnapshot`s (`TEST_ONLY_SYNTHETIC_METRICS`, a module-level constant asserting this explicitly) — every test constructs its own timeline with values chosen for that test's purpose. There is currently insufficient empirical evidence to freeze a general CPU/latency dynamics function suitable for PPO training (a separate, evidence-based calibration phase is required before that can be attempted — see sec.9 below). Replica-*transition mechanics* (as opposed to CPU/latency dynamics) ARE implemented, because CORR-25 already empirically established them: on scale-up, `desired`/`current` update immediately while `ready`/`available` catch up only after a configurable readiness delay (default `10.0s` for deterministic mechanics tests, matching the ~5-15s lag observed across all 9 official HPA tuning trials — explicitly **not** claimed as a final calibrated distribution, only a mechanics-test default, and fully configurable); on scale-down, all four fields update immediately (CORR-25 sec.4).

### 5. V2 Contract Implementation — Verified Exactly, Not Approximately
- **Observation**: `gymnasium.spaces.Box(shape=(11,), dtype=float32)`, bounds `[0,-1,0,0,0,-1,0,-1,0,0,0]` to `[1]*11`, matching docs/rl_mdp_spec.md sec.7 exactly (test-verified: `[0,5]`/`[-5,5]` CPU clip, `[0,4]` memory, `[0,100]` request rate, `[0,2000]`ms latency, `1..8` replica range, `300`s time-since-scale cap — none derived from or coupled to the HPA `260%` baseline).
- **Action**: `gymnasium.spaces.Discrete(3)`, frozen `{0:-1, 1:0, 2:+1}` mapping via a plain dict (`types.ACTION_MAP`), never Python enum ordering.
- **EWMA/history**: exact `alpha = 2/(3+1) = 0.5`, backward-looking only, never updated from an invalid step, correctly updates from a legitimate `no_traffic` `0ms` reading — verified with exact numeric assertions across multi-step sequences (`tests/unit/test_rl_env_observation.py::TestEwmaTracker`), not merely shape checks.
- **Zero-traffic/invalid-latency semantics**: `request_rate==0` + absent histogram ⇒ `latency=0.0ms`, `latency_observation_state="no_traffic"`, sample remains valid; `request_rate>0` + absent/stale latency ⇒ `latency_observation_state="invalid"`, `sample_valid=False`, effective action forced to HOLD — no `1000ms` midpoint fallback is used for either case (verified by test).
- **Action gating**: exact CORR-27 asymmetric rule implemented with the task's own recommended precedence (structural validity → replica-bound clamp → pending-scale gate → scale-down cooldown → execute), verified: pending-scale gating blocks both directions; scale-up has zero additional cooldown and may fire on the very next decision once readiness resolves; scale-down requires ≥60s since the last effective action; a scale-up 30s after a scale-down is **not** blocked (emergency reversal, CORR-27) while another scale-down in that same window **is** blocked; both replica bounds (1 and 8) correctly clamp with `action_override_reason="replica_bound_clamp"`.
- **Reward**: all 7 components implemented independently and exposed in `info["reward_components"]` — quadratic, clipped SLA penalty; live-observable 5xx reliability penalty; resource cost from `deployment_current_replicas` (never Ready); action penalty based on the *requested* action on any step the policy could meaningfully evaluate (excluding telemetry-failure overrides); reversal penalty from *effective* actions only; boundary penalty firing *only* for `replica_bound_clamp` (never double-counted with the action penalty for the same event); efficiency bonus using the independently-derived `[1.0, 3.0]` band (CORR-27 sec.19), never `2.6`.
- **Invalid-step handling**: a step is `reward_validity="partial"` whenever either the state the action was chosen from, or the resulting state, was structurally invalid; in that case SLA/action/reversal/boundary/efficiency are all zeroed, reliability is computed only if independently valid, and resource cost is computed from `deployment_current_replicas` whenever that Kubernetes-sourced field was independently obtainable (verified by test) — no reward term is ever computed from an invented latency value.
- **Episode/termination**: `consecutive_invalid_steps` increments on each invalid observation and resets to `0` on any single valid one; the **4th** consecutive invalid step (120s at the 30s decision interval) terminates the episode (verified with an exact 8-step sequence: counts `[1,2,3,0,1,2,3,4]`, terminating precisely at the last); a separate zero-Ready->120s condition terminates independently (exercised via a minimal test-only backend stub, since the project's real `min_replicas>=1` floor makes a sustained zero-Ready state otherwise unreachable through `ScriptedBackend`'s normal mechanics); truncation fires when the episode's configured duration is reached, independent of termination.

### 6. Transition/Info Schema
`environment.py`'s `_build_info()` returns every field the task and docs/rl_mdp_spec.md sec.14 require: episode id, step index, decision time, raw and normalized observations, requested vs. effective replica delta, desired/current/Ready/Available before and after, total reward and every individual component, terminated/truncated, `action_override_reason`, `bound_clamped`, `pending_scale`, `sample_valid`, `consecutive_invalid_steps`, `reward_validity`, `latency_observation_state`. Verified present on both `reset()` and `step()` by test (`tests/unit/test_rl_env_environment.py::TestTransitionInfoSchema`).

### 7. Gymnasium/SB3 Environment-Check Results
`stable_baselines3.common.env_checker.check_env()` passes against the real `KubernetesAutoscalerEnv` (using `ScriptedBackend`) with no errors. `env.reset(seed=...)`/`env.step(...)` return the standard Gymnasium 5-tuple, verified by test. **No `.learn()` call was made anywhere in this phase; no model checkpoint was produced** — verified by an updated test (`tests/unit/test_rl_dependency_compatibility.py`, corrected from asserting `autoscaler/src/` stays empty — an invariant scoped specifically to the now-completed dependency-only Phase 15A.2 — to asserting `persistent-data/models/`/`persistent-data/rl/` remain empty, which is the invariant that actually matters going forward).

### 8. Tests
85 new tests across `tests/unit/test_rl_env_{observation,gating,reward,scripted_backend,environment,ppo_config}.py`, covering every item in the task's mandatory behavioral-test list (observation shape/clipping/zero-traffic/invalid-latency, EWMA initialization/trend/invalid-skip, HOLD/scale-up/delayed-Ready/pending-gate/cooldown/emergency-reversal/bounds, all 7 reward components including invalid-step partial reward, episode reset/first-decision/termination/truncation, transition-schema completeness) plus the representative scale-up and scale-down timelines from the task's own sec.24/25 examples, verified with exact field-level assertions (not just "it ran"). One pre-existing Phase 15A.2 test was corrected (sec.7 above) rather than left failing. Full suite: **318 passed** (up from 233, no regressions). `ruff check` clean on all new/changed Python files.

### 9. Explicit Scope Boundary: Mechanics, Not Dynamics
This phase validates **environment mechanics** — the Gymnasium API, observation construction, action encoding/gating, reward computation, replica-transition timing, episode/termination semantics, and logging. It does **not** validate, calibrate, or claim any **empirical dynamics** — i.e., how workload and replica count actually produce CPU, latency, error rate, memory, or in-progress-request readings in the real (or a statistically faithful simulated) system. `ScriptedBackend` must never be used to train a PPO policy. The correct status is `ENVIRONMENT_MECHANICS_VALIDATED` / `READY_FOR_EMPIRICAL_DYNAMICS_CALIBRATION` — **not** "ready for PPO training."

### 10. Unaffected by This Phase
`rl_contract_version`/`reward_contract_version` remain `2` (this phase implements v2 exactly; it does not alter it). The HPA baseline remains `averageUtilization=260` (CORR-26), confirmed unchanged both in config and via a live, read-only `kubectl get hpa`/`get pods` check. `configs/traffic_profiles/unseen_eval.yaml` was not inspected, used, or modified. No training/validation workload split was created. No replacement final held-out workload was created. No traffic was sent.

### 11. Next Phase
Empirical dynamics calibration — designing and running the (not-yet-specified) controlled measurement campaign needed to build a statistically faithful workload/replica → CPU/latency/error/memory response model (or to determine that a live-cluster-driven training loop is preferable to any offline simulator), strictly separate from and subsequent to this phase's mechanics validation. PPO training must not begin until that calibration (or an explicit, justified decision to train directly against a live/replayed backend) exists.

**Applies to phases**: 15A/15A.1/15A.2 (the contract and runtime this phase implements; the Phase 15A.2 net-arch bookkeeping gap identified and corrected here), 15B.1 (this correction), the future empirical-dynamics-calibration phase and the future live-Kubernetes-backend phase (both must implement `backend.DynamicsBackend` without changing observation/gating/reward/environment code), the future PPO-training phase (must not begin until empirical dynamics are calibrated or a live-training-loop decision is explicitly justified).

---

## CORR-30 — RL Transition-Reward Validity Decomposition and Deterministic Episode-Reset Control Age (Phase 15B.1A)

### 1. Context
Phase 15B.1's architecture (Gymnasium environment, pluggable `DynamicsBackend` Protocol, `ScriptedBackend`, observation/gating/reward modules) is **not** invalidated or redesigned by this phase. Two implementation defects, both found through pre-training review rather than through training failures, are corrected here: (1) an overly broad reward-validity rule that could erase an already-determined, valid policy action's cost merely because the *next* observation's telemetry failed; (2) an unfrozen (not previously specified) numeric value for the `seconds_since_last_scale_norm` observation feature at episode start. Neither correction required a `rl_contract_version`/`reward_contract_version` bump — inspection of `docs/rl_mdp_spec.md` §17/§18/§20 confirmed the corrected behavior matches what those sections already implied (§17 conditions `R_action`/`R_reversal`/`R_boundary` purely on decision-time quantities; §20 never actually froze a numeric episode-start control-age value) — these are conformance fixes and a gap-fill, not new rules. No traffic was sent, no PPO training occurred, no Kubernetes mutation occurred (only a read-only `kubectl get hpa`/`get pods` check, confirming `averageUtilization=260` unchanged).

### 2. The Exact Reward-Validity Bug
Phase 15B.1's `reward.compute_reward()` took one `transition_fully_valid = s_t_valid AND s_(t+1)_valid` boolean and, when either half was `False`, zeroed `R_sla`, `R_reliability` (unless independently valid), `R_resource` (unless independently valid), `R_action`, `R_reversal`, and `R_boundary` uniformly. This is correct when `s_t` itself is invalid (the policy never had a trustworthy state to act on) but **incorrect** when only `s_(t+1)` is invalid: `R_action`/`R_reversal`/`R_boundary` are determined entirely from decision-time quantities (`action_override_reason`, `effective_replica_delta`, `is_reversal` — all computed *before* the backend is ever advanced to `t+1`) and therefore do not depend on whether the *outcome* telemetry later succeeds. The bug meant a policy's already-real, already-recorded scaling/reversal/boundary cost could vanish purely because of a subsequent telemetry hiccup unrelated to the decision's own quality — a training-time incentive distortion if left uncorrected (a policy could effectively "launder" a costly action through a coincidental telemetry gap).

### 3. Authoritative Interpretation, Confirmed by Re-Reading the Frozen Document (Not Assumed)
`docs/rl_mdp_spec.md` §17 (unchanged text, re-read verbatim in this phase): "`R_action` is evaluated on `requested_replica_delta`... but only on a step where `sample_valid==True`" — referring to the *decision* step's validity — and separately, "`R_reversal` is computed only from effective... scaling actions." §18 (unchanged text): the zeroing rule for `R_action`/`R_reversal`/`R_boundary` is scoped explicitly to steps where `action_override_reason` **is itself** one of the four telemetry-failure reasons — a condition entirely determined at decision time. Neither section, as written, ever conditions these three components on `s_(t+1)`'s validity. **This confirms the fix restores conformance with the already-frozen document; it does not introduce a new rule.**

### 4. The Four-Case Validity Matrix (Now Implemented and Documented Explicitly)

| Case | `s_t` | `s_(t+1)` | SLA / Reliability / Efficiency | Action / Reversal / Boundary | `reward_validity` |
|---|:--:|:--:|---|---|---|
| A | valid | valid | computed normally | computed normally | `full` |
| B | invalid | valid | zeroed (reliability may still be independently valid) | all zero (no real decision occurred) | `partial` |
| **C** | **valid** | **invalid** | zeroed (no genuine outcome data) | **preserved — computed exactly as in Case A** | `partial` |
| D | invalid | invalid | same as B | same as B | `partial` |

`R_resource` may be computed in any case whenever `deployment_current_replicas` was independently obtained (a separate Kubernetes/kube-state-metrics signal, unaffected by whichever application-level metric may have failed) — unchanged from CORR-27's original rule. **Case C is the one Phase 15B.1 mishandled**; it is now correctly implemented in `autoscaler/src/rl_env/reward.py: compute_reward()`, which takes two explicit parameters, `decision_state_valid` and `outcome_state_valid`, rather than one collapsed flag, and computes `R_action`/`R_reversal`/`R_boundary` from a shared helper (`_action_side_components()`) invoked identically whenever `decision_state_valid` is `True` — regardless of `outcome_state_valid`.

### 5. Action-Side Component Preservation, Verified by Test — Not Merely Asserted
`tests/unit/test_rl_env_reward.py::TestValidityMatrixCaseC` and a parallel environment-level test in `tests/unit/test_rl_env_environment.py::TestValidityMatrixAtEnvironmentLevel` verify, with exact numeric assertions: (C1) a valid `SCALE_UP` followed by outcome-telemetry failure still shows its `0.1` action penalty in `reward_components`, while `sla`/`efficiency_bonus` are correctly `0.0` and `resource` is still computed from the independently-known current-replica count; (C2) a valid emergency-reversal `SCALE_UP` (immediately following a prior `SCALE_DOWN`) followed by outcome-telemetry failure preserves **both** its `0.1` action penalty and its `0.5` reversal penalty — neither is erased; (C3) a valid, correctly-clamped bound-violating request followed by outcome-telemetry failure preserves its `0.2` boundary penalty while still correctly showing zero action penalty (no double-count, CORR-27's rule unchanged). A dedicated regression test (`TestRewardCannotIncentivizeTelemetryFailure`) directly compares the **same** valid action event's `action`/`boundary` components under a subsequently-valid vs. subsequently-invalid outcome and confirms they are numerically identical, while the outcome-dependent `sla`/`efficiency_bonus` terms correctly differ (zeroed only when the outcome data is genuinely unavailable).

### 6. Invalid-Decision Safety-HOLD Behavior Unchanged (Cases B/D)
When `decision_state_valid=False` (the policy did not have a trustworthy state to act on), `R_action`/`R_reversal`/`R_boundary`/`R_efficiency`/`R_sla` remain zero regardless of `outcome_state_valid`, exactly as CORR-27 originally specified — verified by `TestValidityMatrixCasesBAndD`, including the independently-valid-resource/reliability carve-outs.

### 7. Episode-Reset Control Age — A Genuine Specification Gap, Not a Changed Value
`docs/rl_mdp_spec.md` §20 froze the reset *sequence* (Deployment→1 replica, confirm Ready, 60s stabilization, episode `t=0`) but never specified what numeric value the `seconds_since_last_scale_norm` observation feature (index 10) should read at the very first decision. Phase 15B.1's implementation defaulted this to the observation cap (`300s`, normalized `1.0`, i.e. "as if no scaling has ever happened") — a reasonable-sounding but arbitrary choice that was never actually frozen anywhere. **Frozen now**: the logical reset-to-`initial_replicas` event is treated as an effective scaling action that occurred exactly `pre_episode_stabilization_s` (60s) before episode `t=0`. At `t=0`: `seconds_since_last_effective_scale = 60s`, normalized `60/300 = 0.2`. This value grows normally thereafter (e.g. `90s` at `t=30s` if no further scaling occurs) exactly as if a real scaling action had occurred at logical time `-60s`. This is implemented in `autoscaler/src/rl_env/environment.py: reset()` by initializing `_last_effective_action_time_s = -pre_episode_stabilization_s` rather than `None`/a cap-returning sentinel — deterministic and reproducible regardless of whatever control history existed before the reset, and equally implementable by a future live backend (which would actually wait out the 60s window in real time and arrive at the same logical value).

### 8. First-Observation Trend Features Confirmed Zero, Independently at the Environment Level
The three EWMA trend features (indices 1, 5, 7 — `cpu_per_replica_delta`, `request_rate_delta`, `p95_latency_delta`) were already correctly zero at the first valid observation per `EwmaTracker`'s first-value-initialization rule (unchanged, Phase 15B.1) — this phase adds an environment-level test (`TestEpisodeResetControlAge::test_trend_features_are_exactly_zero_at_t0`) confirming this holds through the full `env.reset()` path, not merely at the `EwmaTracker` unit level, and a further test (`test_reset_does_not_leak_previous_episode_history`) confirming a second `env.reset()` after an episode with real scaling activity produces identical zero-trend, `0.2`-control-age results — no state leaks across episodes.

### 9. Scale-Down-at-Reset Precedence Confirmed Unaffected
Because the episode-start control age (`60s`) already exceeds the `60s` scale-down cooldown threshold, a naive read might worry the reset-age correction makes a `SCALE_DOWN` request at episode start "cooldown-eligible." It is irrelevant: at `initial_replicas=min_replicas=1`, a `SCALE_DOWN` request's candidate replica count (`0`) is below the minimum bound, so `gating.compute_effective_action()`'s existing precedence (bound-clamp checked before pending-scale gating and before the cooldown check, unchanged since Phase 15B.1) resolves it as `replica_bound_clamp`, never `scale_down_cooldown` — confirmed by a new regression test, `TestEpisodeResetControlAge::test_scale_down_at_t0_resolves_as_bound_clamp_not_cooldown`.

### 10. Documentation Update — Clarification, No Version Bump
`docs/rl_mdp_spec.md` gained two new sections (§24: the explicit four-case reward-validity matrix; §25: the frozen episode-start control-age value) and a corresponding Change Log entry, explicitly labeled "clarification, no version change" — `rl_contract_version`/`reward_contract_version` remain `2`. Neither new section introduces a coefficient, formula, clip range, or eligibility condition that did not already follow from the existing frozen text; §24 makes an already-implied matrix explicit, and §25 fills a genuine prior omission (no value was ever frozen to be "changed").

### 11. Tests and Full Regression
13 new tests: 7 in `tests/unit/test_rl_env_reward.py` (`TestValidityMatrixCasesBAndD` ×4, `TestValidityMatrixCaseC` ×5 collapsed/counted individually — see that file for the exact list — and `TestRewardCannotIncentivizeTelemetryFailure`) and 6 in `tests/unit/test_rl_env_environment.py` (`TestEpisodeResetControlAge` ×4, `TestValidityMatrixAtEnvironmentLevel` ×2). Full suite: **331 passed** (up from 318, no regressions). `ruff check` clean on all changed Python. `stable_baselines3.common.env_checker.check_env()` re-run against the real environment and re-confirmed passing. No `.learn()` call was made; no model was produced.

### 12. Unaffected by This Phase
The 11-element observation order/shape, CPU/memory/request-rate/latency normalization ranges, EWMA `alpha=0.5`, the `{0:-1,1:0,2:+1}` action encoding, pending-scale gating, the asymmetric scale-up/scale-down cooldown rule, replica bounds `1..8`, the `10.0s` scale-up mechanics-test delay, the backend Protocol, all seven reward coefficients, the `[1.0,3.0]` efficiency band, the 4th-consecutive-invalid-step termination rule, the transition-log schema's field set, and the `[128,128]` PPO adapter are all **unchanged** by this phase. No empirical dynamics model was introduced. The HPA baseline remains `averageUtilization=260` (CORR-26), confirmed live and read-only.

**Applies to phases**: 15B.1 (the reward-validity/reset-timing defects identified and corrected here), 15B.1A (this correction), the future empirical-dynamics-calibration phase and any future PPO-training phase (must implement/train against the Case A-D validity matrix and the frozen episode-start control age exactly as specified here, not Phase 15B.1's original overly-broad rule).

---

## CORR-31 — Empirical RL Dynamics Calibration Protocol Freeze (Phase 15B.2)

### 1. Context
This is a **protocol-design and freeze phase only**. No calibration traffic was sent. No Kubernetes object was mutated (a single read-only `kubectl get hpa`/`get pods` check confirmed `averageUtilization=260` unchanged). No empirical dynamics model was implemented or finalized. No PPO training occurred. No training/validation workload split was frozen. No replacement formal held-out HPA-vs-RL evaluation workload was created or inspected. This entry documents the precommitted measurement and model-validation methodology for the future empirical-dynamics-calibration *execution* phase, frozen **before** any new calibration data exists — exactly the sequencing this project's own HPA-tuning campaign (CORR-17 onward) already established as good practice.

### 2. New Authoritative Artifacts
- `docs/rl_dynamics_calibration_protocol.md` — narrative protocol document (v1).
- `experiments/manifests/rl_dynamics_calibration_v1.yaml` — the sole machine-readable source of exact frozen numbers: cell matrix, randomized cell order (fully computed and embedded, not merely "a seed and a method"), timing, repeats, model families, transformations, validation methodology, acceptance criteria, extrapolation policy, invalid-cell criteria, and artifact structure.
- `configs/rl_dynamics.yaml` — a thin, stable entry point mirroring how `configs/rl.yaml`/`reward.yaml` expose the frozen MDP contract; does not duplicate the manifest's content.
- `tests/unit/test_rl_dynamics_calibration_protocol.py` (32 tests) — offline structural validation of all of the above.

### 3. Existing-Data Suitability Audit — the Central Finding
Verified directly from `experiments/results/hpa_tuning_final/per_trial_metrics.csv`: **all 9 official HPA tuning trials peaked at exactly 2 replicas — no exception.** This means no empirical data exists anywhere in this project for replica counts 3 through 8; that entire range is uncalibrated territory the future campaign must measure from scratch. The 9 trials' brief 2-replica windows are explicitly classified **not** directly reusable for fitting (confounded by a continuously-changing tuning-schedule RPS, narrow dwell time, and overlap with the scaling event itself) — only usable as a sanity-check prior for the new dedicated 1→2 transition-calibration cells. CORR-22's fresh-mode addendum and CORR-23's fine-grained final-path CPU map — both already run under HPA-absent, fixed-1-replica, `connection_mode=fresh`, corrected-path, intensity=3 conditions — **are** directly reusable to substitute one repeat each of several replicas=1 fitting cells (2/5/6/8 RPS via CORR-22; 5.0 RPS's full 3-repeat requirement already satisfied via CORR-23). Metrics-Server/Metrics-API CPU readings and HPA's own `averageUtilization` are explicitly excluded as fitting targets in any case (they are not what the RL V2 observation contract actually sources from).

### 4. Dynamics-Backend Target and the Exogenous-Workload/Policy-Observation Boundary
The future backend must produce, at each logical step, the same `RawMetricSnapshot` shape `ScriptedBackend` already produces (request rate, 5xx rate, p95 latency, in-progress count, CPU cores, memory bytes, validity metadata) — no interface change to `backend.DynamicsBackend` is required. The backend may know its own internally-generated offered-workload trajectory (necessary to compute system evolution), but only the already-frozen 11-element V2 observation vector may ever reach a PPO policy — no future-workload-schedule value is exposed as a policy input, unchanged from `docs/rl_mdp_spec.md` §2.

### 5. Frozen Scope and Physical Constraints
`intensity=3` frozen (matches every prior calibration/HPA-tuning artifact in this project; V2's observation vector has no intensity feature, so mixing intensities without one would silently conflate causal variables — no conflict found requiring a STOP). Minikube's outer 2.0-CPU/3072-MiB cgroup ceiling (CORR-1) is carried forward explicitly as the reason performance cannot be modeled as a pure per-replica function — a dedicated fixed-total-aggregate-RPS load level (L5, §7 below) was added specifically to make cluster-wide contention (many "lightly-loaded" pods collectively saturating the shared ceiling) empirically visible, a failure mode invisible to any purely per-replica-RPS view.

### 6. Frozen Matrix
**Fitting anchors**: replicas `{1,2,4,6,8}` × load levels `L1`(2 RPS/replica)/`L2`(5)/`L3`(8)/`L4`(12, deliberately beyond all prior fresh-mode evidence) at every anchor, plus `L5` (fixed 40 RPS total regardless of replica count) at anchors `{1,4,8}` only — 23 cells. **Interpolation-validation**: replicas `{3,5,7}` × `{L1,L2,L3}` — 9 cells. `L1`–`L3` are directly anchored to already-verified fresh-mode, final-path, single-replica evidence (CORR-22/23: 2/5/8 RPS → 25.1%/~49-52%/75.9% of the 500m CPU limit, 100% success in every prior run at those levels); `L4`/`L5` deliberately probe previously-untested territory rather than assuming linear extrapolation from single-replica capacity, which CORR-1's shared-ceiling evidence makes physically implausible above a few replicas.

### 7. Timing, Repeats, and Fully-Precommitted Randomized Order
Per steady-state cell: 60s fixed-replica stabilization → 60s workload warm-up (avoiding `[1m]`-rate-window contamination from the prior cell, CORR-13's steady-window principle reused) → 120s scored steady state (240s total). Repeats: 3 for fitting anchors (matching CORR-18 §7-8's precedent), 2 for validation/transition cells (an explicitly justified tractability asymmetry — validation exists to catch gross interpolation failure, not to independently establish its own noise floor). Cell order within each repeat-block is a deterministic `random.Random(42+repeat_index).shuffle()` permutation (distinct seed bases for fitting/validation/transition blocks) of a canonically-sorted cell list — the exact resulting sequence for every repeat is computed and embedded verbatim in the manifest, not left as "a seed and a method to apply later," so chronological/host-heating drift can never be confounded with replica count or load level after the fact.

### 8. Fixed-Replica Isolation, Traffic Path, and Signal Semantics
The frozen procedure (not executed in this phase): record canonical HPA config → disable HPA for the session only → set fixed replica count via `kubectl scale` → verify desired=current=ready=available before any traffic → run the cell → never mutate replica count mid-window → restore HPA at 260% and natural 1-replica idle afterward, verify no RL controller. Traffic path remains `host → minikube service demo-app-loadtest --url → Service → pods`, `connection_mode=fresh` — `kubectl port-forward svc/demo-app` remains forbidden for workload traffic (CORR-21/22, unchanged). CPU fitting target: Prometheus/cAdvisor aggregate `cpu_cores` (the same source the V2 observation already uses) — Metrics-Server/Metrics-API supplemental only. Latency fitting target: Prometheus server-side p95 — client-side p95 preserved in parallel for diagnostics/relationship-quantification only, never substituted. Reliability fitting target: Prometheus-visible 5xx rate only — client timeouts/connection-errors/drops preserved as diagnostic-only signals, never leaked into the simulated policy observation (docs/rl_mdp_spec.md §3's live-observability limit, reused). Memory/in-progress: a precommitted decision rule (R²<0.10 or CV<10% ⇒ per-replica empirical baseline rather than a dynamic regression) decides, per metric, whether a dynamic model is even warranted — decided by the rule, not by eye after seeing plots.

### 9. Transition Calibration
Two kinds, kept deliberately separate: (a) workload-level step transitions at a *fixed* replica count, determining whether the backend needs one-step lag/inertial state for CPU/latency/in-progress, or whether steady-state lookup plus the already-implemented replica mechanics suffices — not assumed in advance; (b) replica-count transitions `1→2,2→4,4→6,6→8` and reverses, each with 60s pre-stabilization and 180s post-trigger observation, measuring `desired→current/Ready/Available` lag and CPU/latency/in-progress/request-rate settling. Critically: CORR-25's ~5-15s desired→Ready evidence covers **only** 1→2 transitions (no HPA trial ever exceeded 2 replicas) — whether higher-order transitions behave similarly is genuinely unknown, and the current `scale_up_ready_delay_seconds=10.0` mechanics-test default (`autoscaler/src/rl_env/scripted_backend.py`) remains exactly that — a deterministic test default — until this calibration validates or replaces it, possibly with a replica-count-dependent distribution.

### 10. Model Families, Transformations, Validation, Metrics, Acceptance, Extrapolation, Invalid-Cell Policy — All Frozen Before New Data
Candidate families precommitted per metric (monotone-interpolation lookup / piecewise-linear regression surface / tree-based regressor for CPU-memory-in-progress-latency; a regime-classifier-or-threshold model for zero-inflated error rate; deterministic mechanics plus a lagged/exponential-settling model for transitions) — no neural network as a first-line candidate, and no winner selected in this phase. Transformations frozen now (CPU/memory/in-progress raw; p95 latency `log1p`, justified from Phase-10's own already-existing multiplicative-blowup-near-saturation evidence, not chosen after seeing new results; error rate raw probability). Validation strictly at the whole-run/cell-repeat level (never same-run-sample splitting, which would leak autocorrelated observations) across three types: interpolation (replicas 3/5/7, excluded from fitting), repetition (leave-one-repeat-out), and transition (trajectory-level, evaluated separately from steady-state error). Acceptance criteria frozen as a principled noise-relative rule (`model MAE ≤ X=2.0 × observed repeat-to-repeat standard deviation`, applied equally to fitting and interpolation-validation cells — never relaxed for validation cells) plus a concrete `≥90%` error-rate regime-classification accuracy floor, a monotonicity requirement, and hard physical-plausibility constraints (CPU/memory/latency/in-progress ≥0, error rate ∈[0,1]). Extrapolation policy: any query outside the calibrated support region is treated as a structurally invalid observation, reusing (not reinventing) the already-implemented CORR-27/CORR-30 invalid-step safety machinery — a PPO policy can never be trained against a fabricated out-of-envelope value. Invalid-cell criteria mirror the HPA-tuning methodology's validity-before-performance principle (CORR-17 §7 lineage): structural harness/config failures invalidate a cell; genuine bad performance (high latency, 5xx, overload-induced client errors, CPU saturation, low success under legitimate load) remains valid data.

### 11. Campaign Size, Reflecting Honest Planning
23 fitting cells × 3 repeats (69) + 9 validation cells × 2 repeats (18) + 8 transition types × 2 repeats (16) = **103 total runs**, an estimated **≈8.6 hours** of wall-clock time (240s scored/observed per run plus ~60s reconfiguration overhead), expected to span multiple execution sessions — consistent with, not larger in kind than, this project's own HPA-tuning campaign's own multi-session duration. Existing-data reuse (§3) may reduce a small number of replicas=1 fitting runs. No design compromise was made to the full replica-range/interpolation-validation coverage to fit a single session; the honest multi-session estimate is reported instead.

### 12. Tests and Verification
`tests/unit/test_rl_dynamics_calibration_protocol.py` (32 tests, new): protocol/config versioning, intensity frozen at 3, MDP contract still v2, fitting-anchors-plus-validation-cells covering the full 1-8 replica range exactly, no reference to `unseen_eval` anywhere in the new protocol files, HPA tuning trial IDs explicitly excluded from the reusable-fitting-sources list, `connection_mode=fresh`/port-forward-forbidden/loadtest-Service-scrape-exclusion all frozen, HPA/RL controller-isolation requirements present in both the invalid-cell criteria and the required per-run metadata fields, repeat counts and distinct per-block-type randomization seeds, each repeat's frozen cell order matching its declared cell count and being a genuinely distinct permutation, model-family/transformation freezes, acceptance/invalid-cell criteria present and well-formed, the extrapolation policy correctly referencing the existing invalid-step machinery, `configs/rl.yaml`/`configs/reward.yaml`/`configs/hpa.yaml`/`k8s/hpa/hpa.yaml` all unchanged (contract versions still 2, PPO `net_arch` still `[128,128]`, HPA still 260%), and confirmation that no `persistent-data/dynamics_calibration/` result directory or PPO training artifact exists yet. Full suite: **363 passed** (up from 331, no regressions). `ruff check` clean on the new test file. Live cluster re-confirmed via read-only `kubectl get hpa`/`get pods`: `averageUtilization=260`, Deployment naturally 1/1/1/1, unchanged.

### 13. What Remains Explicitly Untouched
The 11-element observation vector, action encoding, gating rules, reward formula/coefficients, replica-transition *mechanics* implementation, `backend.DynamicsBackend` Protocol, `rl_contract_version`/`reward_contract_version` (both still 2), the HPA baseline (260%, CORR-26), `configs/traffic_profiles/unseen_eval.yaml` (not inspected, not modified, still marked `NOT_ELIGIBLE_AS_THE_SOLE_FORMAL_FINAL_HELD_OUT_EVALUATION_PROFILE` per CORR-27), and every development traffic-profile YAML (`burst`/`constant`/`mixed`/`periodic`/`ramp`/`step.yaml`, still unexecuted, still unassigned to any train/validation role).

**Applies to phases**: 15B.1/15B.1A (the mechanics this calibration will eventually feed data into), 15B.2 (this correction, protocol freeze), the future empirical-dynamics-calibration EXECUTION phase (must follow this frozen matrix/order/timing/model-family list/acceptance criteria exactly, or document a new correction before deviating), the future PPO-training phase (must not begin until a dynamics model passes this protocol's acceptance criteria, or an explicit, justified decision to train against a live/replayed backend instead is separately documented).

---

## CORR-32 — RL Empirical-Dynamics Calibration Protocol Integrity Corrections Before First Execution (Phase 15B.2A)

### 1. Context
Pre-execution review of Phase 15B.2's frozen calibration protocol found several genuine preregistration defects — none discovered by collecting new data (none was collected; this remains a protocol-correction phase), all discovered by re-reading the protocol against its own stated goals and the actual constraints of the traffic generator and application. This is a correction of the preregistration **before** its first execution, not a change of methodology after seeing results — the same distinction CORR-27 drew for the RL MDP contract before any environment code existed. No traffic was sent, no Kubernetes object was mutated (a read-only `kubectl get hpa`/`get pods` check confirmed `averageUtilization=260` unchanged), no PPO training occurred, no `unseen_eval.yaml` inspection occurred, and `protocol_version` remains `1` throughout, per explicit instruction.

### 2. Historical-Data Fitting Eligibility, Corrected
Phase 15B.2 classified CORR-23 (Phase 11C.4D) as directly reusable for fitting. This was incorrect for the frozen CPU target: CORR-23 is the **Metrics-API** CPU mapping (built specifically to characterize what HPA itself reads), not the Prometheus/cAdvisor aggregate `cpu_cores` target this protocol actually fits to. Reclassified: `SANITY_CHECK_OR_SUPPLEMENTAL_CPU_CROSS_CHECK_ONLY` — never a direct fitting-label substitute. CORR-22's fresh-mode addendum was independently re-inspected and confirmed to use the *correct* Prometheus/cAdvisor source, path, and connection mode — structurally compatible in isolation — but, for measurement-semantics uniformity (it predates this protocol's newly-frozen decision-aligned extraction convention and canonical-schedule-replay discipline), `historical_run_substitution_for_official_repeats` is frozen `false`. **No historical run substitutes for any official calibration-v1 repeat.** All 102 official runs are newly collected under this one protocol; historical data remains valid only as matrix-design evidence, sanity-check priors, and external consistency checks.

### 3. Duplicate Physical Condition Found and Removed
Audited every `(intensity, replicas, total_offered_rps)` tuple across the entire fitting matrix. Found exactly one genuine duplicate: at replicas=8, `L2_sustainable` (5 RPS/replica × 8 = 40) and `L5_cluster_contention` (fixed 40 total) are the identical physical condition. Merged into one cell, `dyncal_r8_L2_L5`, executed once per repeat and tagged with both analytical roles — never run twice under two names. Every other same-total-RPS coincidence across *different* replica counts was verified to be a genuinely distinct physical condition (same total spread over a different pod count is a different per-pod load), not a duplicate.

### 4. Generator-Capacity Audit and Revised Maximum Workload
Directly inspected `loadgen/traffic_generator.py`: default `timeout_seconds=5.0`, default `max_in_flight=50`. Phase 15B.2's original L4 level (12 RPS/replica scaling to 96 RPS total at replicas=8) implied a worst-case simultaneously-outstanding-request bound of `96×5.0=480` — nearly 10× the generator's default capacity, meaning the harness itself, not the application, would likely have become the effective bottleneck at that extreme. **Corrected**: `max_in_flight` frozen at `400` for the entire campaign (an existing `--max-in-flight` parameter, no code change); L4 restricted to replicas `{1,2,4}` only (totals 12/24/48), removing the 72/96 RPS extremes; the revised matrix's own highest total (64, at replicas=8/L3 — a naturally-arising, already-justified per-replica-near-capacity test, not a new arbitrary extreme) sets the binding worst-case bound (`64×5.0=320 < 400`, verified with a ~25% safety margin). A cell reporting `dropped_overload` under this setting is now an explicit harness-capacity structural defect, never an application-performance observation.

### 5. Canonical Arrival Process Frozen and 37 Schedules Generated Offline
Deterministic, constant, open-loop arrivals frozen as the sole mode. **37 canonical schedules were generated and SHA-256-hashed offline** (`scripts/generate_dynamics_calibration_schedules.py` — new, narrow, network-free, reusing `scripts/generate_tuning_schedule.py`'s existing, tested, pure schedule-generation path rather than a new implementation) — 20 fitting + 9 validation steady-state schedules, 4 replica-transition schedules (each shared across both directions of its pair, since the offered workload during a replica-count transition is direction-independent), and 4 workload-step-transition schedules. All stored under `configs/traffic_profiles/rl_dynamics_calibration/{steady_state,replica_transitions,workload_transitions}/`, indexed in `schedule_registry.json`. No traffic was sent to produce them — verified: no `persistent-data/traffic/dyncal_*` directory exists.

### 6. Decision-Aligned Metric Extraction Precisely Frozen
Steady-state timing restated relative to traffic onset (not the pre-traffic stabilization period): `warmup=[0,60)`, `scored=[60,180)` (180s traffic + 60s no-traffic stabilization = 240s total). Decision-aligned extraction timestamps frozen at exactly `[60, 90, 120, 150]` seconds — for each, the latest valid Prometheus observation at-or-before that timestamp (within the existing 30s staleness tolerance) is used, never a later sample. Whole-run/cell-repeat-level splitting remains mandatory, unchanged in principle from Phase 15B.2 but now unambiguous given the exact timestamps.

### 7. Transition Campaign Corrected: Adjacent ±1 Replica Transitions, Workload-Step Transitions Added
Two independent defects fixed. First, Phase 15B.2's `1→2,2→4,4→6,6→8` replica-transition design used jumps that do not represent one RL action (the policy changes replicas only by ±1) — corrected to four adjacent bidirectional pairs, `1↔2, 3↔4, 5↔6, 7↔8` (8 directional types), with workload frozen at `5.0 RPS × lower_replica_count_of_the_pair` (verified sustainable pre- and post-transition for every pair). Second, Phase 15B.2's own narrative *named* fixed-replica workload-step transitions but never budgeted any runs for them — a real omission, now corrected with 4 explicit types (replicas=1: 2↔8 RPS; replicas=4: 8↔32 RPS, both within the revised 64 RPS ceiling), additional to the replica-transition runs. Both transition kinds now share one precisely frozen timing structure: 60s pre-run replica stabilization (no traffic) → 60s pre-trigger traffic at the "before" condition → trigger at traffic-elapsed=60s → 180s post-trigger observation (300s total per transition run) — Phase 15B.2's "60s pre + 180s post" language had left the traffic-onset point and scored interval undefined.

### 8. Corrected Official Run Count
23→**20** unique fitting cells (deduplication + L4 anchor restriction) × 3 repeats = 60 runs; 9 validation cells × 2 repeats = 18 runs; 8 replica-transition directional types × 2 repeats = 16 runs; 4 workload-transition types × 2 repeats = 8 runs (newly added). **Total: 102 official new calibration runs** (composition substantially changed from Phase 15B.2's 103, not merely re-labeled) — estimated ≈8.9 hours wall-clock, across multiple sessions. Every randomized cell order was regenerated against the corrected cell lists (deterministic `random.Random(seed).shuffle()`, distinct seed bases per cell-type) and verified by test to be an exact permutation of the correct cell set at the correct count, every repeat.

### 9. Acceptance-Criterion Mathematics Corrected
Phase 15B.2's "model MAE ≤ 2× repeat SD" was ambiguous and could demand near-zero error at a near-zero-variance cell. **Corrected, explicit, divide-by-zero-safe formula**: `s_eff = max(within_cell_repeat_sd, 0.5×pooled_within_cell_sd_across_fitting_anchors, epsilon)`; `normalized_error = |predicted_cell_mean − observed_cell_mean| / s_eff`, with fitting-cell predictions computed leave-one-repeat-out (never same-data fitted values). Frozen thresholds: median normalized error ≤2.0, ≥90% of cells ≤2.0, no cell >4.0 — applied identically to fitting and interpolation-validation cells (never relaxed for validation). p95 latency's normalized error is computed in the frozen `log1p` space; millisecond-space error is reported separately, never substituted into the acceptance decision.

### 10. Error-Rate Validation Strengthened
Plain zero/non-zero classification accuracy (Phase 15B.2's original criterion, `≥90%`) could be trivially satisfied by an always-zero classifier in this application's expected low-5xx regime. **Corrected**: `balanced_accuracy ≥0.80` and `nonzero_class_recall ≥0.80` (both classes weighted), plus the existing probability/rate MAE. If the entire dataset contains zero non-zero server-visible 5xx observations, classification quality is explicitly reported `NOT_ESTIMABLE` — never silently asserted as passing from an all-zero dataset.

### 11. Monotonicity Scoped to Physically Justified Axes Only
A generic cross-replica monotonicity requirement was never explicitly frozen in Phase 15B.2 beyond a vague "no more than one violation" note; CORR-32 makes the exact, physically-reasoned scope explicit: increasing offered RPS at *fixed* replicas must not produce a materially lower CPU/p95/in-progress/5xx-rate (enforced); latency improving or CPU falling as replicas increase at fixed *total* RPS is explicitly **not** enforced, since the shared 2-core host may legitimately produce the opposite — exactly what the `L4`/`L5` overload/contention levels exist to detect. A blanket monotonicity rule would have risked masking the campaign's own central research question.

### 12. Dynamics-Out-of-Support Distinguished From Prometheus Unavailability
Phase 15B.2's extrapolation policy proposed mapping an out-of-calibration-support query onto `RawMetricSnapshot`'s existing `target_up=False`/`query_error=True` fields — incorrectly representing simulator-domain ignorance as an infrastructure outage. **Corrected**: a distinct future condition, `DYNAMICS_OUT_OF_SUPPORT`, is frozen conceptually (recommended handling: stop/truncate the episode immediately with an explicit reason, never counted toward Prometheus-downtime statistics) — not implemented in this protocol-correction phase (it requires a small, targeted extension during the later empirical-backend-implementation phase) and does not alter the frozen 11-element policy observation contract.

### 13. Stochastic Residual/Noise Modeling Preregistered
Phase 15B.2's candidate model families addressed conditional means only, risking a perfectly deterministic simulator and consequent PPO overfitting. **Two residual-model candidates preregistered now, before any data exists, no winner selected**: (1) a joint empirical residual bootstrap sampling one `[cpu, latency, memory, in_progress, error_rate]` residual vector at a time (never independent per-metric noise, preserving cross-metric correlation), stratified by nearest supported cell; (2) a parametric multivariate fallback (e.g. Gaussian copula) if empirical sample size proves inadequate. Residuals are frozen to be estimated from fitting runs only, never from interpolation-validation runs.

### 14. Memory/In-Progress Decision Rules Clarified
Memory: evaluated per-Ready-replica first, regressed against `(RPS, replicas)` using fitting-cell data only; below the R²/CV threshold, modeled as `per_replica_empirical_baseline × ready_replicas` (CORR-25-consistent semantics) rather than a raw constant. In-progress: explicitly **not** automatically collapsed to a replica-only constant merely because its coefficient of variation is low — it is logically a workload-tied queueing signal; if simplified, a load-regime-conditioned empirical distribution is preferred over a pure per-replica constant.

### 15. Invalid-Cell Policy Extended
Added explicit generator/harness structural-invalidity criteria: `offered != sent` due to undersized `max_in_flight`; `dropped_overload` caused by harness capacity rather than application response. Clarified as a **valid** performance outcome, not a harness defect: a client timeout occurring after a request was legitimately dispatched to a genuinely overloaded application.

### 16. Tests
`tests/unit/test_rl_dynamics_calibration_protocol.py` was rewritten (61 tests, up from 32) to validate every correction above: historical-substitution disabled and CORR-23/CORR-22 correctly excluded; the exact single duplicate found and merged, and no other duplicate tuples exist; the revised 64 RPS ceiling and generator `max_in_flight=400` sizing verified sufficient for every matrix value including transitions; all 37 canonical schedules exist and their SHA-256 hashes are **independently recomputed and verified** by the test suite itself (not merely trusted from the registry); decision-aligned timestamps `[60,90,120,150]` frozen; replica-transition pairs verified all adjacent and spanning the full 1-8 range; workload-transition runs present and additional to replica-transition runs; the exact 102-run total verified to equal the sum of its components; every randomized order verified to match its cell count and repeat structure; the acceptance-criterion formula's divide-by-zero safety and leave-one-repeat-out prediction source verified; balanced (not plain) error-rate classification metrics verified; monotonicity's justified-axes scoping verified; the dynamics-out-of-support/Prometheus-unavailable distinction verified; residual-model preregistration and its validation-run exclusion verified; the extended invalid-cell criteria verified; and confirmation that `configs/rl.yaml`/`reward.yaml`/`hpa.yaml`/`k8s/hpa/hpa.yaml` remain unchanged, no calibration result directory or PPO training artifact exists, and no traffic directory was created by schedule generation. Full suite: **392 passed** (up from 363, no regressions). `ruff check` clean on all new/changed Python. Live cluster reconfirmed via read-only `kubectl get hpa`/`get pods`: `averageUtilization=260`, unchanged.

### 17. What Remains Unaffected
`rl_contract_version`/`reward_contract_version` (both still 2), the 11-element observation vector, action encoding, gating rules, reward formula, replica-transition mechanics implementation, the HPA baseline (260%), and `configs/traffic_profiles/unseen_eval.yaml` (not inspected, not modified) are all untouched by this correction.

**Applies to phases**: 15B.2 (the preregistration this corrects), 15B.2A (this correction), the future empirical-dynamics-calibration EXECUTION phase (must follow this corrected matrix/schedules/timing/acceptance-criteria exactly — the protocol is now genuinely mechanical to execute, requiring no further design decisions), the future PPO-training phase (must not begin until a dynamics model passes these corrected acceptance criteria, or an explicit, separately-justified decision to train against a live/replayed backend instead is documented).

## CORR-33 — Final RL Dynamics-Calibration Execution-Readiness Clarifications Before First Measurement (Phase 15B.2B)

### 1. Context
CORR-32 fixed genuine preregistration defects but left a small set of execution/model-identification ambiguities unresolved — none of them a matrix redesign. This correction closes those ambiguities before the first calibration request is ever sent. No traffic was sent, no Kubernetes object was mutated (a read-only `kubectl get hpa`/`get pods` check confirmed `averageUtilization=260`, min/max=1/8, Deployment 1/1/1/1, unchanged), no RL/PPO training occurred, `configs/traffic_profiles/unseen_eval.yaml` was not inspected, and `protocol_version` remains `1` throughout, per explicit instruction. The frozen 20/9/8/4/102/37/64/400/5s matrix values, canonical schedule hashes, HPA baseline (260%), and MDP/reward contracts (v2/v2) are all unchanged — this is a clarification pass, not a redesign.

### 2. Request-Rate Realization Frozen as an Explicit Fitting Target
CORR-32 implicitly treated `request_rate_rps` as a passthrough of the exogenous workload schedule. Corrected: `offered_rps` (the internal simulator-driven workload trajectory, never exposed directly to the PPO observation) and `request_rate_rps` (the Prometheus-server-observed rate, one of the 11 frozen observation components) are explicitly distinguished — under overload, a `connection_error` means a request never reaches the application and is never counted by `demo_app_requests_total`, so server-observed rate can legitimately fall below offered rate. `request_rate_rps` is now an explicit fitting target modeled from `(offered_rps, replica_state)` like every other output, never assumed identical to `offered_rps`. Three candidate model families added (bounded lookup/interpolation, piecewise-linear regression surface, tree-based regressor); validation reports MAE in RPS, relative error, the throughput-realization ratio `observed_rps/offered_rps` (never assumed ≡1.0), and overload-regime error separately from the non-overload regime.

### 3. Causal Structure Frozen as Joint-Input, Two-Stage Rejected
A two-stage structure (predict `request_rate_rps` from `offered_rps`, then feed that *prediction* into the CPU/latency/memory/in-progress/error-rate models) was considered and rejected: it creates hidden teacher-forcing at fit time (trained on the true observed rate) that silently becomes propagated error at simulation time (driven by a predicted, noisy rate). **Frozen**: all six steady-state output models share exactly the same exogenous predictors, `(offered_rps, replica_state)`; no output model ever consumes another output model's prediction as an input.

### 4. Traffic Terminology Disambiguated
Four distinct rates are now precisely defined to prevent silent substitution: **offered** (the canonical planned arrivals from the frozen schedule file), **sent** (what the generator actually dispatched — must equal offered for structural validity), **server-observed** (the Prometheus-counted rate, the fitting target), and **client-completed** (the generator's own outcome log — diagnostic-only, never substituted for the RL request-rate observation or for server-observed rate in any model).

### 5. Per-Pod Routing-Distribution Diagnostics Added, With an Exact 3-Tier Criterion
Every run with `ready_replicas ≥ 2` at any point in its scored window now records a per-pod routing diagnostic, using the CORR-22 §4-established PromQL pattern (`job="demo-app"` filtering to avoid the documented double-scrape double-count), grouped by pod, read as a raw counter delta between the scored window's start and end. Three tiers, applied only when a pod's window total is ≥50 requests (below that, diagnostics are recorded but never used to invalidate): **HEALTHY_VARIANCE** (every pod within [0.5×,1.5×] fair share and CV≤0.30 — no action, ordinary load-balancing noise); **ROUTING_PINNING_SUSPECTED** (a pod between 1% and 10% of fair share — flagged for manual review, not automatically invalid); **STRUCTURAL_ROUTING_FAILURE** (a pod below 1% of fair share while others receive substantial traffic — structurally invalid, retried as a new attempt of the same planned slot). This criterion never requires perfectly equal distribution and never penalizes ordinary variance.

### 6. HPA Isolation/Restoration Mechanics Frozen, Reusing Existing Tooling
Rather than inventing new tooling, the exact commands already proven in `scripts/run_rl_experiment.sh` and the CORR-21/22/23 diagnostic sessions are frozen as the mechanical procedure: `kubectl delete hpa demo-app-hpa -n autoscaler-app --ignore-not-found` before a session begins, and `kubectl apply -f k8s/hpa/hpa.yaml` with a live-target=260% verification after a session ends — **including on abort/error**, so a crash mid-campaign never leaves the cluster HPA-less or RL-controller-active. Between cells, only the execution harness sets fixed replica counts directly; HPA is never merely reconfigured (e.g. min=max) as a substitute for deletion.

### 7. Multi-Session Execution and Run/Attempt Identity Frozen
Session semantics: a documented start checklist (cluster/HPA/Prometheus/Metrics-Server health, idle stabilization) and end checklist (stop generators/collectors, restore HPA, verify idle, preserve all artifacts) bound each session; resuming a later session continues from the next not-yet-VALID planned slot in the already-frozen randomized order — the order is never regenerated and already-VALID runs are never rerun. Run/attempt identity: a *planned slot* (e.g. `fit_dyncal_r4_L3_repeat2`) is distinct from an *execution attempt* (e.g. `fit_dyncal_r4_L3_repeat2_attempt1_<timestamp>`) — an invalid attempt is preserved (never deleted or overwritten) and the same slot is retried as the next attempt number; only VALID attempts enter the fitting/validation/transition datasets. Required per-run metadata extended with planned-slot ID, attempt number, session ID, campaign sequence index, wall-clock UTC start/end, and the full chronological/config context (replica count, offered RPS, schedule SHA, intensity, target URL, connection mode, max-in-flight, timeout, HPA/RL absence verified, Kind state, app resource config, Minikube outer limits, Prometheus health, Metrics-Server resolution, pre-run idle context).

### 8. CPU/Memory Transition Semantics and Client/Server Latency Clarified
Memory: the `per_ready_replica_baseline × ready_replicas` simplification applies **only** to steady-state cells (where desired=current=ready=available by construction); during transitions, aggregate memory must depend on `current` replicas and transition/startup state, never silently multiplied by Ready alone — this does not change the frozen V2 observation, which still normalizes by Ready. CPU: aggregate cAdvisor CPU includes not-yet-Ready containers, so `current > ready` may occur during scale-up; aggregate CPU, `current`, and `ready` must be preserved as distinct fields, never assuming a new pod contributes zero CPU pre-Ready. Client/server latency: both Prometheus/server p95 and client-side p95 are preserved for every run; their absolute difference, ratio, cross-cell correlation, and overload-regime divergence are preregistered analyses; client p95 is never a policy input and never silently substitutes for server p95 in model selection.

### 9. All-Zero-5xx Fallback and Precise Dynamics-Out-of-Support Semantics Frozen
All-zero-5xx fallback: if every fitting-and-validation run shows server-visible 5xx==0, the empirical backend is not fit or claimed as a meaningful 5xx classifier — it may output 0 inside the observed calibration domain only, labeled `ZERO_ONLY_WITHIN_OBSERVED_CALIBRATION_DOMAIN`, never represented as predicting overload 5xx behavior outside that domain; this does not by itself block final acceptance but must be reported before any PPO training begins, with the SLA-latency reward term remaining the primary overload-performance signal in that scenario; synthetic 5xx examples are never manufactured. Dynamics-out-of-support (CORR-32 §12, made precise): no fabricated observation is returned; the episode ends via Gymnasium's `truncated` flag (not `terminated` — a simulator-capability boundary, not a genuine MDP terminal state, matching SB3's bootstrapped-value handling of truncation), with `info["termination_reason"]="dynamics_out_of_support"`; this does not increment `consecutive_invalid_steps` and does not set `prometheus_unavailable` — it requires its own new `types.py` constant, confirmed by direct inspection of `autoscaler/src/rl_env/types.py` to not collide with any existing `OVERRIDE_*` value. Implementation remains deferred to the empirical-backend phase; this phase freezes semantics only.

### 10. Generator Safety Margin Restated as an Exact 20%-Headroom Formula
CORR-32's `max_in_flight=400` was justified by an approximate ~25% margin against the matrix's peak (320). Restated precisely: `required_max_in_flight = ceil(max_total_rps_in_matrix × timeout_seconds × 1.20)` — at the matrix's revised peak (64 RPS × 5.0s), this computes to `ceil(64×5.0×1.20)=384`, still comfortably under the frozen `400`. Tests compute this value dynamically from the manifest's own `revised_max_total_rps_ceiling`, `default_timeout_seconds`, and `required_max_in_flight_headroom_factor` fields, never hardcoding `320` or `384` independent of those inputs.

### 11. Residual/Noise Sampling Unit Corrected to Whole-Run Trajectory Blocks
CORR-32's residual-model candidates did not specify the sampling *unit*. Corrected: residuals/noise are sampled as **whole-run trajectory blocks** — the four decision-aligned timestamps `[60,90,120,150]` of one fitting run, sampled together, never as independent per-timestamp rows — to preserve both cross-metric correlation (already frozen) and short-term temporal autocorrelation within a run. Transition residuals are likewise frozen at full-trajectory level relative to the trigger timestamp, never pointwise-shuffled, so settling-time distributions remain reproducible by any future stochastic transition model. Validation runs remain excluded from residual fitting, unchanged from CORR-32.

### 12. Execution-Harness Requirements Specified as a Mechanical Checklist
The future execution harness (not built in this phase) is specified, not implemented: load the frozen manifest; verify all 37 schedule SHA-256 hashes; identify the next not-yet-VALID planned slot; run the session-start preflight; temporarily remove HPA; set the exact fixed replica count and wait for exact readiness; start the NodePort tunnel; start the Prometheus observer/collector; replay the exact canonical schedule byte-for-byte; preserve all artifacts regardless of outcome; validate run structure against every invalid-cell criterion (including the new per-pod-routing and generator-capacity criteria); mark the attempt VALID/INVALID; retry or advance accordingly; and restore canonical cluster state on session exit (normal or aborted). All scientific parameters remain fixed by this frozen protocol — the harness decides none of them.

### 13. Tests
`tests/unit/test_rl_dynamics_calibration_protocol.py` was extended (129 tests in this file, up from 61) to validate every item above: request-rate modeling presence and non-identity-assumption; joint-input causal structure with the two-stage alternative explicitly rejected; all four traffic-terminology terms defined and client-completed excluded from RL use; the 3-tier per-pod routing criterion (including its addition to both the invalid-criteria and valid-outcomes lists); HPA delete/restore commands verified present including the abort/error case; session start/end checklists and the never-regenerate/never-rerun resumption rule; planned-slot-vs-attempt identity and the full required-metadata-field superset (including `artifact_structure.required_metadata_fields` extension); client/server latency preregistration and the never-a-policy-input rule; CPU/memory transition semantics (current vs. ready preserved separately, steady-state-only memory simplification); the all-zero-5xx fallback's domain label, non-blocking acceptance impact, and pre-PPO reporting requirement; dynamics-out-of-support's `truncated`/`termination_reason`/non-collision-with-`OVERRIDE_*` semantics (directly inspecting `autoscaler/src/rl_env/types.py`); the 20%-headroom formula computed dynamically from the manifest's own fields (never hardcoded); the trajectory-level residual sampling unit and its renamed candidate; and an explicit "no matrix redesign" invariant class re-verifying 20/9/8/4/102/37/64/400/5s, the HPA=260 baseline, and both frozen contract versions are byte-for-byte unchanged. Full unit suite: **460 passed** (up from the 392-passing baseline noted in CORR-32, +68 new tests in this file; no regressions). `ruff check` clean on the changed test file. Live cluster reconfirmed via read-only `kubectl get hpa`/`get pods`: `averageUtilization=260`, min/max=1/8, Deployment 1/1/1/1, unchanged.

### 14. What Remains Unaffected
The 20 fitting cells, 9 interpolation cells, 8 replica-transition directional types, 4 workload-transition types, 102 official runs, 37 canonical schedules and their SHA-256 hashes, the 64 RPS ceiling, 400 max-in-flight, 5s timeout, decision timestamps `[60,90,120,150]`, repetition counts, and randomized orders are all unchanged from CORR-32. `rl_contract_version`/`reward_contract_version` (both still 2), the HPA baseline (260%), and `configs/traffic_profiles/unseen_eval.yaml` (not inspected, not modified) remain untouched.

**Applies to phases**: 15B.2/15B.2A (the preregistration this further clarifies), 15B.2B (this correction), the future empirical-dynamics-calibration EXECUTION phase (must follow this clarified protocol exactly — request-rate modeling, traffic terminology, per-pod routing diagnostics, HPA mechanics, session/attempt semantics, and the execution-harness checklist are all now precisely specified, requiring no further design decisions from the execution agent), the future empirical-backend-implementation phase (must implement `DYNAMICS_OUT_OF_SUPPORT` truncation semantics and the new `types.py` constant exactly as frozen here), the future PPO-training phase (must respect the all-zero-5xx fallback's reporting requirement if it applies).

## CORR-34 — Global Calibration Run Order and Exhaustive Routing-Integrity Classification (Phase 15B.2C)

### 1. Context
CORR-33 closed the modeling/terminology/mechanics ambiguities but left two remaining execution-manifest gaps: (1) no single, explicit, indexed list existed for the whole 102-run campaign — only four separate per-block randomized orders (fitting/validation/replica-transition/workload-transition) — leaving a future execution agent to decide, on its own, how those four orders interleave; (2) the CORR-33 per-pod routing classifier's 3 tiers left real gaps (e.g. a pod at 20%, 40%, or 170% of fair share, or a run with CV>0.30 while every individual share still looked near-fair, mapped to no tier at all). Both are closed here, purely additively — no cell, repeat count, schedule, model family, acceptance criterion, residual/noise design, HPA isolation rule, or session-semantics rule was touched. No traffic was sent, no Kubernetes object was mutated (a read-only `kubectl get hpa`/`get pods` check confirmed `averageUtilization=260`, min/max=1/8, Deployment 1/1/1/1, unchanged), no RL/PPO training occurred, `unseen_eval.yaml` was not inspected, and `protocol_version` remains `1` throughout.

### 2. Global 102-Slot Campaign Order Did Not Previously Exist
Direct inspection of `experiments/manifests/rl_dynamics_calibration_v1.yaml` confirmed no single global execution-order list existed prior to this phase — `campaign_sequence_index` appeared only as a *required per-run metadata field name* (CORR-33's `run_identity`), never as an actual populated ordered sequence. It was frozen now, before any data collection, per the task's explicit instruction not to introduce a new randomization layer: `scripts/generate_global_campaign_sequence.py` (new, pure, network-free, no `random`/`shuffle` call anywhere in its own code, verified idempotent by test) deterministically concatenates the already-frozen `randomization.fitting_order` / `validation_order` / `replica_transition_order` / `workload_transition_order` sequences (CORR-32) in one precommitted block order: fitting-repeat-1 (20) → validation-repeat-1 (9) → replica-transition-repeat-1 (8) → workload-transition-repeat-1 (4) → fitting-repeat-2 (20) → validation-repeat-2 (9) → replica-transition-repeat-2 (8) → workload-transition-repeat-2 (4) → fitting-repeat-3 (20) = 102. Output persisted at `configs/traffic_profiles/rl_dynamics_calibration/global_campaign_sequence.json`, SHA-256-hashed and referenced from the manifest's new `global_campaign_order` section. Each block's own internal order is preserved byte-for-byte from CORR-32's frozen seeded shuffle — verified by test to match exactly.

### 3. Resume Rule Restated Against the Global Index
Resume always means: the lowest `campaign_sequence_index` whose planned slot has no VALID attempt yet — never a new shuffle, never "pick an easy run first." A session boundary never alters this list or its order; the list is immutable, identical in spirit to `schedule_registry.json`'s already-frozen schedule hashes.

### 4. Exhaustive Four-State Routing Classification
CORR-33's 3-tier classifier (`HEALTHY_VARIANCE` / `ROUTING_PINNING_SUSPECTED` / `STRUCTURAL_ROUTING_FAILURE`) left gaps. **Corrected to four tiers, checked in this exact precedence order** so a severely starved pod is never masked by an otherwise-acceptable aggregate CV: (1) `INSUFFICIENT_ROUTING_SAMPLE` — total traffic below the sample-sufficiency threshold, or the per-pod count unreconstructable; recorded, not a failure, routing not used for structural validity; (2) `STRUCTURAL_ROUTING_FAILURE` — at least one eligible pod's `fair_share_multiple < 0.01` while another receives substantial traffic; structurally invalid, retried as a new attempt; (3) `HEALTHY_VARIANCE` — every eligible pod within `[0.5,1.5]×` fair share and CV≤0.30; no action; (4) `ROUTING_PINNING_SUSPECTED` — the explicit catch-all for every other sufficiently-sampled, non-structural case (1%-49.999% of fair share, >150% of fair share, CV>0.30 even with individually near-fair shares, or any other substantial imbalance) — never automatically invalidates, always preserved and surfaced for review. Every eligible run now maps to exactly one tier.

### 5. Fair-Share Mathematics Frozen
`fair_fraction = 1/N_eligible`; `observed_fraction_i = pod_request_count_i / total_request_count_across_eligible_pods`; `fair_share_multiple_i = observed_fraction_i / fair_fraction`. Coefficient of variation is frozen to be computed over **raw per-pod request counts** (not shares — the two scale equivalently, but exactly one must be chosen; count-CV is the frozen choice).

### 6. Sample-Sufficiency Threshold Corrected to Total-Window, Not Per-Pod
CORR-33's "applied only when a pod's window total is ≥50 requests" was ambiguous and, read as a per-pod requirement, would have hidden exactly the severe-starvation pattern the diagnostic exists to detect (a starved pod's own count is precisely what may be near zero). **Corrected**: `total_request_count_across_eligible_pods ≥ 50 × N_eligible`, sized so there is enough traffic for ~50 requests/pod under a fair distribution. Below threshold: `INSUFFICIENT_ROUTING_SAMPLE`, not a failure. Still permits detecting a 0-or-1-request pod inside a sufficiently large total window.

### 7. Ready-Serving Eligibility, Counter History, and Reset Semantics
Eligibility is now frozen as Ready-serving exposure, not desired/current membership: a pod counts toward `N_eligible` only for the portion of the window it was actually Ready and capable of serving traffic; per-pod first/last Ready timestamp, eligible serving duration, and request-count-while-eligible are recorded; a newly-starting or terminating pod is never classified as starved merely for partial-window presence, and a pod never Ready during the window is excluded entirely. Transition-run routing is never computed as a bare `counter_at_final_timestamp − counter_at_start_timestamp` (a scaled-down pod may have no live series at the window's end) — the full raw counter-sample history over the window is queried instead, using each pod's first and last valid sample while its own series exists, retaining terminated pods' historical series from Prometheus whenever still available. Counter resets: a non-negative diff is summed normally; a negative diff is treated as a reset and the running total continues from the new value (never subtracted); `counter_reset_count` is recorded per pod; a reset alone never invalidates a run if the count remains reconstructable this way. An unreconstructable count classifies the run `INSUFFICIENT_ROUTING_SAMPLE` unless an independent structural failure is separately established — missing counts are never invented or interpolated. The routing PromQL continues to filter `job="demo-app"` only, never `job="demo-app-loadtest"` (CORR-22 §4).

### 8. Routing Remains Diagnostic-Only; Planned-Slot State Ledger Specified
Per-pod routing distribution remains a structural/path diagnostic only — never added to the 11-element RL observation or to any dynamics-output contract, and causes no MDP version change (verified by direct inspection of `autoscaler/src/rl_env/observation.py`: no `per_pod` reference exists). A machine-readable campaign-state ledger format is specified, not created: three states (`NOT_ATTEMPTED` / `VALID` / `INVALID_RETRY_REQUIRED`), reconstructable from persisted per-attempt metadata alone if the ledger file is ever lost; the frozen manifest and `global_campaign_sequence.json` remain immutable, only the ledger and per-attempt artifacts change during execution.

### 9. Tests
`tests/unit/test_rl_dynamics_calibration_protocol.py` was extended (182 tests, up from 129) to validate every item above: the global-order section's presence, the sequence file's existence and hash match, exactly 102 entries with indices `1..102` contiguous and unique, composition `60/18/16/8`, every planned-slot ID unique, every schedule hash independently recomputed and matched against the registry, the exact precommitted block-kind sequence, within-block order proven identical to the CORR-32 frozen randomized order (fitting-repeat-1 spot-checked in full), byte-identical regeneration from the generator function (no additional randomness), the generator's own code verified to contain no `random`/`shuffle` call, and the resume rule's exact wording; the ledger specification's three states and reconstructability; the four-tier classifier's precedence order and exhaustiveness note; a reference-implementation classifier (mirroring the frozen numeric thresholds) exercised against concrete examples — 20%-of-fair and 170%-of-fair share both correctly resolve to `ROUTING_PINNING_SUSPECTED`, a high-CV/near-fair-shares construction also resolves to `ROUTING_PINNING_SUSPECTED` (with the CV threshold itself verified load-bearing), a severe <1%-of-fair-share construction resolves to `STRUCTURAL_ROUTING_FAILURE`, a low-CV/near-fair construction resolves to `HEALTHY_VARIANCE`, and a too-small-total construction resolves to `INSUFFICIENT_ROUTING_SAMPLE`; the fair-share and CV formulas verified present with count-CV (not share-CV) explicitly frozen; the total-window (not per-pod) sample threshold verified; Ready-serving eligibility fields and the not-yet-Ready-pods-not-starved rule verified; the never-bare-endpoint-delta and first/last-valid-sample-while-series-exists rules verified; counter-reset handling and the unreconstructable-count fallback verified; the diagnostic-only reaffirmation and untouched observation-builder verified; and a full "no matrix redesign" invariant class re-verifying 20/9/8/4/102/37/64/400/5s, the HPA=260 baseline, and both frozen contract versions are byte-for-byte unchanged. Full unit suite: **513 passed** (up from 460, +53 new tests; no regressions). `ruff check` clean on both changed/new Python files. Live cluster reconfirmed via read-only `kubectl get hpa`/`get pods`: `averageUtilization=260`, min/max=1/8, Deployment 1/1/1/1, unchanged.

### 10. What Remains Unaffected
The 20 fitting cells, 9 interpolation cells, 8 replica-transition directional types, 4 workload-transition types, 102 official runs, 37 canonical schedules and their SHA-256 hashes, the 64 RPS ceiling, 400 max-in-flight, 5s timeout, decision timestamps `[60,90,120,150]`, repetition counts, randomized orders, model families, acceptance criteria, the preregistered residual/noise design, HPA isolation rules, and session semantics are all unchanged from CORR-32/CORR-33. `rl_contract_version`/`reward_contract_version` (both still 2), the HPA baseline (260%), and `configs/traffic_profiles/unseen_eval.yaml` (not inspected, not modified) remain untouched.

**Applies to phases**: 15B.2/15B.2A/15B.2B (the preregistration this closes for execution), 15B.2C (this correction), the future empirical-dynamics-calibration EXECUTION phase (must consult `global_campaign_sequence.json` and this global index exclusively for run ordering — it decides no scientific parameter itself; must apply the exhaustive 4-tier routing classifier exactly as frozen here), the future empirical-backend-implementation phase (unaffected by this correction — routing remains diagnostic-only).

## CORR-35 — Exposure-Weighted Routing Opportunity and Transition-Safe Per-Pod Integrity Analysis (Phase 15B.2D)

### 1. Context
CORR-34 froze the global campaign order (accepted, unchanged by this correction) and an exhaustive 4-tier routing classifier — but that classifier's fair-share mathematics, `fair_fraction = 1/N_eligible` over the whole routing-analysis window, silently assumes every eligible pod had *equal opportunity* to serve traffic across the entire window. This is true for an ordinary steady-state cell (all eligible pods Ready for the whole window) but false for any replica-transition run, where one pod may be Ready for 180s and another only for the final 20s after scale-up — a 50/50 whole-window expectation between such pods is not a meaningful fairness baseline. This correction replaces the whole-window expectation with an interval-level, traffic-weighted expected-routing-opportunity calculation, and removes a remaining vague "substantial traffic" clause from the structural-failure condition. No traffic was sent, no Kubernetes object was mutated (a read-only `kubectl get hpa`/`get pods` check confirmed `averageUtilization=260`, min/max=1/8, Deployment 1/1/1/1, unchanged), no RL/PPO training occurred, `unseen_eval.yaml` was not inspected, `protocol_version` remains `1`, and the global 102-slot campaign order (including `campaign_sequence_index=1` = `fit_dyncal_r2_L4_repeat1`) is confirmed byte-identical to Phase 15B.2C — this correction touches ONLY routing mathematics and wording, nothing else.

### 2. Why CORR-34's Whole-Window `1/N_eligible` Was Insufficient
CORR-34's Ready-serving-exposure eligibility (first/last Ready timestamp, eligible serving duration) already recognized that pods can have unequal presence in a window — but its fair-share *comparison* still divided total traffic by `N_eligible` as if every eligible pod were present for the whole window. The two ideas conflicted: eligibility correctly excludes non-Ready time from a pod's own accounting, but the whole-window `1/N_eligible` baseline it was compared against implicitly assumed uniform exposure across all eligible pods simultaneously. Direct inspection of `docs/rl_dynamics_calibration_protocol.md`, `experiments/manifests/rl_dynamics_calibration_v1.yaml`, CORR-33/CORR-34, `global_campaign_sequence.json`, the schedule registry, `collectors/metrics_collector.py` (confirming aggregate request-rate queries use `sum(rate(demo_app_requests_total{...}))`, i.e. per-pod series summed), and the Kubernetes Ready-condition observation pattern already established by `hpa_status_collector.py` (authoritative Kubernetes API state preferred over coarser Prometheus/kube-state-metrics polling) confirmed the exact defect and grounded its fix.

### 3. Interval-Level Expected Routing Opportunity
For each reconstruction interval `j`, aligned to the frozen 10-second Prometheus/collector cadence (`configs/experiment.yaml: prometheus.scrape_interval_seconds`, never a coarser 1-minute `rate()`): `delta_total_requests_j` is the reconstructed, reset-aware, positive delta of the aggregate legitimate request counter (`demo_app_requests_total{job="demo-app",route="/work"}`, never `job="demo-app-loadtest"`) over that interval; `ready_pod_set_j` is the set of eligible pods actually Ready at interval `j`'s start timestamp (sourced from the Kubernetes API's per-pod Ready condition); `N_ready_j = len(ready_pod_set_j)`. Then `expected_increment_i_j = delta_total_requests_j / N_ready_j` if pod `i` is in `ready_pod_set_j`, else `0`; `expected_request_count_i = sum_j(expected_increment_i_j)`. This is deliberately **not** weighted by wall-clock Ready duration alone — traffic-volume weighting automatically incorporates unequal exposure, changing pod membership, and changing workload intensity (workload-step transitions, overload regimes) in one formula. No sub-interval precision is invented: a Ready/NotReady transition observed between two interval boundaries is attributed to whichever boundary sample first reflects it.

### 4. The Canonical Routing-Opportunity Ratio and Ratio-CV
`actual_request_count_i` reuses CORR-34's counter-history and reset mechanics unchanged. The canonical comparison quantity is now `routing_opportunity_ratio_i = actual_request_count_i / expected_request_count_i`, with `1.0` representing exactly-expected routing (e.g. `0.2` = 20% of expected opportunity received, `1.7` = 170%). For a steady-state run where every eligible pod shares equal exposure, `N_ready_j` is constant and equal to `N_eligible` for every interval, so `expected_request_count_i` reduces exactly to CORR-34's `1/N_eligible` result — this is a correction/generalization of the same diagnostic, not a change of its purpose. Coefficient of variation is now computed over the **normalized ratios** of assessable pods, replacing CORR-34's raw-request-count CV (invalid once pods can have unequal exposure, since a short-lived pod's naturally smaller raw count would inflate count-CV even under perfectly proportional routing).

### 5. Assessability Corrected to Expected Count, Not Actual or Total
CORR-34's whole-window `total_request_count_across_eligible_pods ≥ 50×N_eligible` threshold is replaced: a pod is **assessable** if `expected_request_count_i ≥ 50` (frozen `routing_min_expected_requests_per_pod = 50`) — evaluated against EXPECTED count, never actual count, so a genuinely starved pod (expected=100, actual=0) remains assessable and can still trigger structural failure, while a short-lived pod with little Ready-serving opportunity (expected=7, actual=1) is correctly excluded rather than falsely flagged as starved. Routing structural classification requires at least **2 assessable pods** (`routing_run_level_minimum_assessable_pods = 2`); fewer classifies `INSUFFICIENT_ROUTING_SAMPLE`, not a failure.

### 6. Reconstruction Consistency Check
A new check verifies that summing all eligible pods' independently-reconstructed `actual_request_count_i` matches the aggregate `delta_total_requests_j` reconstruction (computed as the reset-aware reconstruction of the *summed* per-pod raw values at each shared interval boundary). Frozen tolerance: **0 requests** — `demo_app_requests_total` is an exact integer Counter, and both reconstructions apply identical reset-aware arithmetic to the same raw samples with no smoothing or extrapolation, so they must match exactly whenever every eligible pod has a joint sample at every interval boundary used. A joint-observability gap (a scrape miss for one pod at a boundary) excludes that boundary consistently from both sides rather than fabricating a value; a persistent gap falls back to the existing unreconstructable-count rule. A discrepancy beyond tolerance classifies `INSUFFICIENT_ROUTING_SAMPLE` unless an independent structural failure is separately established — the missing/discrepant difference is never fabricated.

### 7. Structural-Failure Ambiguity Removed
CORR-34's `STRUCTURAL_ROUTING_FAILURE` required "at least one eligible pod's fair_share_multiple < 0.01 AND at least one other eligible pod receives substantial traffic" — the second clause was vague and unnecessary. **Corrected**: at least 2 assessable pods exist AND at least one assessable pod's `routing_opportunity_ratio_i < 0.01` is now sufficient on its own — if a pod had ≥50 *expected* requests (meaning the legitimate traffic that occurred during its own Ready exposure was itself substantial) yet received <1% of that expected opportunity, starvation is already established numerically by the ratio alone; no secondary undefined threshold is needed.

### 8. Classification Remains Exhaustive
The same four tiers, same precedence order (`INSUFFICIENT_ROUTING_SAMPLE` → `STRUCTURAL_ROUTING_FAILURE` → `HEALTHY_VARIANCE` → `ROUTING_PINNING_SUSPECTED`), now expressed in ratio terms: `HEALTHY_VARIANCE` requires every assessable pod's ratio within `[0.5,1.5]` and ratio-CV ≤0.30; `ROUTING_PINNING_SUSPECTED` remains the exhaustive catch-all for every other sufficiently-sampled, non-structural case. Every eligible run still maps to exactly one tier — verified by an independent reference-implementation classifier operating on (expected, actual) pairs against 10+ concrete constructed examples, including the manifest's own three preregistered worked examples.

### 9. Routing Remains Diagnostic-Only; No Contract Change
Reaffirmed and re-verified: per-pod routing distribution is a structural/path diagnostic only. No pod-share/ratio state was added to the PPO observation (still 11 elements), no field was added to `RawMetricSnapshot`, `DynamicsBackend` was not altered, the reward formula was not changed, and `rl_contract_version`/`reward_contract_version` remain `2`. `autoscaler/src/rl_env/observation.py` was directly re-inspected and confirmed to contain no `per_pod` reference.

### 10. Global Campaign Order and First Slot Verified Unchanged
`global_campaign_sequence.json` was recomputed and its SHA-256 confirmed byte-identical to the value frozen in Phase 15B.2C; `campaign_sequence_index=1` remains exactly `fit_dyncal_r2_L4_repeat1` (replicas=2, offered RPS=24, repeat 1, unchanged schedule path and SHA-256). This correction did not touch the generator, the sequence file, or any cell/schedule value — only the manifest's `per_pod_routing_diagnostics` block was rewritten in place, superseded with an explicit `superseded_by_corr35` pointer documenting exactly what changed and why (consistent with how CORR-34 itself superseded CORR-33's 3-tier classifier in place).

### 11. Worked Examples
Three deterministic protocol-level examples are frozen, directly relevant to `campaign_sequence_index=1`: (1) 2 Ready pods for the whole window, 2880 total legitimate requests → expected 1440/pod, actual 1450/1430 → `HEALTHY_VARIANCE`; (2) the same exposure, actual 0/2880 → `STRUCTURAL_ROUTING_FAILURE`; (3) a replica-transition example (Pod A Ready for the full high-traffic interval, Pod B Ready only after scale-up) demonstrating that expected opportunity is traffic-weighted, not duration-weighted, and does not default to a naive 50/50 whole-window split.

### 12. Tests
`tests/unit/test_rl_dynamics_calibration_protocol.py` was extended to 217 tests in this file (up from 182), including: a generic correction-chain-integrity test (prior-correction-reference is always exactly one CORR number below the current one, replacing brittle hardcoded version-pointer tests that would otherwise break every future phase) and content-preservation checks proving CORR-33/CORR-34 material was superseded-in-place with an explicit pointer, never silently dropped; the interval-level expected-opportunity formula, its rejection of duration-only weighting, and the routing-opportunity-ratio/ratio-CV formulas (replacing the now-removed `fair_share_mathematics` tests); the assessability rule using EXPECTED not actual count, with the exact expected=100/actual=0 (remains assessable, flags structural) and expected=7/actual=1 (not assessable, not falsely flagged) cases from the task's own specification; the run-level minimum-2-assessable-pods rule; the reconstruction-consistency check's zero-tolerance and joint-observability-gap handling; the corrected `STRUCTURAL_ROUTING_FAILURE` condition with no secondary "substantial traffic" clause; an independent reference-implementation classifier (rewritten to operate on expected/actual pairs) exercised against 10+ examples including the manifest's own three worked examples and a concrete traffic-weighted-vs-duration-weighted transition regression guard; the Ready-set interval-reconstruction rules (Kubernetes-API-sourced, no invented sub-sample precision, 10s cadence matching `configs/experiment.yaml`); and a full "no matrix/global-order/MDP redesign" invariant class re-verifying 20/9/8/4/102/37/64/400/5s, the global sequence's SHA-256 and first-slot identity, HPA=260, and both frozen contract versions are byte-for-byte unchanged. Full unit suite: **548 passed** (up from 513, +35 net new tests; no regressions). `ruff check` clean on the changed test file. Live cluster reconfirmed via read-only `kubectl get hpa`/`get pods`: `averageUtilization=260`, min/max=1/8, Deployment 1/1/1/1, unchanged.

### 13. What Remains Unaffected
The 20 fitting cells, 9 interpolation cells, 8 replica-transition directional types, 4 workload-transition types, 102 official runs, 37 canonical schedules and their SHA-256 hashes, the 64 RPS ceiling, 400 max-in-flight, 5s timeout, the global 102-slot campaign order and its SHA-256 (byte-identical to Phase 15B.2C), the first planned slot (`fit_dyncal_r2_L4_repeat1`), model families, acceptance criteria, the preregistered residual/noise design, HPA isolation rules, and session semantics are all unchanged from CORR-32/CORR-33/CORR-34. `rl_contract_version`/`reward_contract_version` (both still 2), the HPA baseline (260%), and `configs/traffic_profiles/unseen_eval.yaml` (not inspected, not modified) remain untouched.

**Applies to phases**: 15B.2/15B.2A/15B.2B/15B.2C (the preregistration this further corrects), 15B.2D (this correction), the future empirical-dynamics-calibration EXECUTION phase (must implement the interval-level expected-routing-opportunity calculation exactly as frozen here — it decides no scientific parameter itself), the future empirical-backend-implementation phase (unaffected — routing remains diagnostic-only).

## CORR-36 — Steady-State Ready-Count Invariant Can Be Violated by Overload-Induced Readiness/Restart Behavior (Phase 15B.3A, DOCUMENTATION ONLY — no resolution frozen yet)

### 1. Context
Phase 15B.3A built the empirical dynamics-calibration execution harness (`scripts/run_dynamics_calibration_slot.py` and helpers) and executed the campaign's first official run: `campaign_sequence_index=1` (`fit_dyncal_r2_L4_repeat1` — replicas=2, offered 24 RPS, L4/overload level, repeat 1 of 3), attempt `fit_dyncal_r2_L4_repeat1_attempt1_20260909T140146Z`. That execution surfaced a genuine specification gap in the frozen protocol, not a harness or methodology-application defect. Per explicit instruction, this correction **documents the gap only** — it does not freeze a resolution. No traffic was sent, no Kubernetes mutation was left dangling, and no further calibration run was executed while this gap remains open, per instruction.

### 2. The Finding
Active, per-pod, Kubernetes-API-sourced Ready-condition history (`cluster/pod_readiness_history.csv`, a NEW artifact this phase introduced specifically because the API cannot reconstruct historical per-pod readiness after the fact) shows that at **all four** frozen decision-aligned timestamps (60s, 90s, 120s, 150s) of this run, only **1 of the intended 2 replicas** was actually Ready — the two pods were never simultaneously Ready at any of the four exact decision points. One pod (`demo-app-647f665b49-rpztw`) also incurred a genuine container restart (`restartCount` 6→7) within the scored window (between elapsed 160s and 170s), confirmed by the same authoritative source. Deployment `spec.replicas`/`status.replicas` (current) remained exactly 2 throughout the entire run — there was no scaling event and no pod replacement (pod identities captured pre-stabilization and post-traffic are identical). This is pure application-level readiness-probe/container-crash behavior under the L4 (12 RPS/replica) overload condition, corroborated independently by Prometheus/kube-state-metrics's own `replicas_ready` column and by repeated `target_unhealthy` events in the metrics collector's `events.jsonl` (`"context deadline exceeded"` scraping the app's own `/metrics` endpoint under load).

### 3. Why This Is Not Covered by Any Currently-Frozen Criterion
None of the enumerated `invalid_cell_structural_criteria` in `experiments/manifests/rl_dynamics_calibration_v1.yaml` addresses a steady-state cell where Ready count falls below the intended fixed replica count with no scaling/replacement event. CORR-33 §14/15 already distinguishes current-vs-Ready replicas explicitly for **replica transitions**, but the steady-state simplification (`signal_semantics.memory.dynamic_vs_baseline_decision_rule`, and the general framing throughout CORR-31 through CORR-35) is documented as valid "ONLY because desired=current=ready=available by construction in steady state." This run is direct empirical evidence that this construction can be violated by sufficiently extreme, application-level overload behavior even with zero Kubernetes-level scaling events — something the protocol's designers anticipated only in the weaker forms of "cpu_saturation" / "very_high_latency" (already frozen as valid, non-invalidating outcomes), not in the specific form of "the fixed-replica premise itself breaks down at the exact decision-aligned timestamps."

### 4. Three Separate, Already-Fixed Harness Bugs (Not Protocol Defects)
This same run's raw artifacts exposed three genuine implementation bugs in the NEW harness code written this same phase — none is a defect in the already-frozen protocol itself, and none required a CORR entry on its own; they are noted here only for completeness, since all three surfaced in the same run:
1. **Decision-aligned extraction ignored sample validity.** `scripts/dynamics_calibration_lib.py`'s `select_decision_sample()` did not filter candidate Prometheus samples by their own `sample_valid` column before selecting the latest one at-or-before a decision timestamp — several samples near the decision timestamps were `target_unhealthy`/invalid (the same overload-induced scrape failures described above), and the original implementation could have silently selected one of them. The already-frozen protocol's own extraction rule has always correctly specified "the latest **valid** Prometheus observation" (`docs/rl_dynamics_calibration_protocol.md` §9) — the bug was in the harness's implementation of that rule, not the rule itself. Fixed via `select_decision_sample(..., validity_key="sample_valid")` (now the default), with new regression tests. This attempt's `decision_aligned_observations.json` was regenerated from the already-collected raw `metrics.csv` (no new traffic).
2. **Routing reconstruction matched independent collector clocks by exact-float equality.** `scripts/run_dynamics_calibration_slot.py`'s `build_routing_inputs()` aligned the (separately-launched-subprocess) pod-readiness and per-pod-counter collectors' rows by requiring their `elapsed_seconds` values to be EXACTLY equal — since each subprocess has its own independent monotonic clock reference, this exact match essentially never succeeded, silently producing zero aligned samples and a resulting `expected_request_count_i=0.0` for every pod (masked as a spurious `INSUFFICIENT_ROUTING_SAMPLE` classification rather than a visible error). Fixed by aligning on shared wall-clock `timestamp_epoch` with a tolerance (half the 10s cadence), anchored to the readiness collector's own tick grid, with new regression tests. `routing_summary.json` was regenerated from the already-collected raw CSVs (no new traffic); the corrected result is `ROUTING_PINNING_SUSPECTED` (both pods' `routing_opportunity_ratio_i` ≈2.2–2.5×, low CV ≈0.054 — plausibly explained by kube-proxy Endpoints-list propagation lag relative to this harness's own 10s discrete Ready-state polling cadence during the same severe readiness-flapping episode described in §2) — diagnostic-only, not itself a structural-invalidity trigger under the currently-frozen classifier.
3. **Ledger correction left a stale `valid_attempt_id` pointer.** `scripts/dynamics_calibration_lib.py`'s `update_slot_state()` only ever SET `valid_attempt_id` (when `mark_valid=True`) and never cleared it, so correcting a slot from `VALID` back to `INVALID_RETRY_REQUIRED` (exactly the correction applied to this attempt, §5) left `valid_attempt_id` dangling on an attempt no longer considered valid. Fixed so any transition without `mark_valid=True` explicitly clears `valid_attempt_id` to `null`, with a new regression test.

All three fixes are covered by `tests/unit/test_dynamics_calibration_harness.py` (65 tests, up from 58 at the end of the live run's initial pass).

### 5. Disposition of This Attempt
The attempt's raw artifacts are fully preserved and scientifically valuable: this is the first direct evidence that the `dyncal_r2_L4` cell (replicas=2, L4/overload) induces reliability degradation severe enough to violate the steady-state Ready-count assumption at every one of its four decision-aligned timestamps. Per explicit instruction not to silently modify methodology or invent a new invalidity criterion ad hoc, **this attempt is NOT asserted as clean steady-state fitting data**, and it is also not silently accepted as passing. The attempt's own `metadata.json` was extended (never overwritten) with a `post_execution_review` block recording this finding in full. The campaign ledger's entry for `campaign_sequence_index=1` was corrected from the harness's original, criteria-incomplete `VALID` determination (computed inline during execution, before this review, against only the currently-enumerated criteria) to `INVALID_RETRY_REQUIRED`, with the attempt ID preserved in `all_attempt_ids` and `valid_attempt_id` correctly cleared to `null` (a related ledger bug — `update_slot_state()` previously left a stale `valid_attempt_id` pointer when a slot was corrected away from `VALID` — was found and fixed in the same pass, with a regression test).

### 6. What Must Be Decided Before This Gap Is Closed (NOT decided here)
This correction deliberately freezes **no** resolution. A future phase must explicitly choose and freeze one of (at minimum) these options, or another better-justified one, before campaign execution resumes past this point:
- **(A)** Define a new, explicit structural-invalidity criterion (e.g. `steady_state_readiness_instability`): if fewer than the intended replica count is Ready at N or more of the 4 decision-aligned timestamps (a threshold to be chosen and justified), the run is structurally invalid and retried.
- **(B)** Extend CORR-33's current-vs-Ready distinction (already frozen for transitions) to steady-state cells: compute memory/CPU-per-Ready-replica quantities using the **actually-observed** Ready count at each specific decision timestamp, never the cell's nominal/intended count, and accept the resulting decision-aligned sample as valid data under that actual condition.
- **(C)** Introduce a new decision-aligned validity state distinct from both "observed" and "invalid" (e.g. `steady_state_readiness_degraded`) that is recorded and reported but excluded from steady-state model fitting, while still being preserved as a first-class reliability-under-overload finding.
Any of these is a legitimate, additive, non-matrix-redesigning correction consistent with the pattern of CORR-32 through CORR-35 — the point of this entry is only to document that the choice has not yet been made, so it is never made silently.

### 7. Matrix, Global Order, HPA, and Contract Integrity
No cell, repeat count, schedule, model family, acceptance criterion, residual/noise design element, HPA isolation rule, session semantics rule, or global campaign order value was touched. The global 102-slot sequence and its SHA-256 remain byte-identical to Phase 15B.2C; `campaign_sequence_index=1` remains exactly `fit_dyncal_r2_L4_repeat1`. HPA was restored to `averageUtilization=260%`/min=1/max=8 immediately after this attempt (confirmed live), the Deployment returned to natural 1/1/1/1 idle, no RL/PPO process ran, and no stale generator/collector/tunnel process remained. `protocol_version` remains `1`.

### 8. Tests
`tests/unit/test_dynamics_calibration_harness.py` gained 3 tests for the `select_decision_sample` validity-filter fix, 3 tests for the `build_routing_inputs` cross-process timestamp-alignment fix, and 1 regression test for the `update_slot_state` stale-`valid_attempt_id` fix (65 tests total, up from 58 at the end of the live run's first pass). `tests/unit/test_rl_dynamics_calibration_protocol.py` had 6 now-permanently-obsolete "no calibration result directory exists yet" assertions (correct through Phase 15B.2D, necessarily false once official execution legitimately begins) replaced with assertions that remain permanently true: PPO/RL training artifacts must never exist, any `persistent-data/dynamics_calibration/` attempt directory must correspond to a real frozen `planned_slot_id`, any campaign ledger must conform exactly to the frozen three-state spec, and the schedule generator script's own source must never contain a traffic/HTTP call. Full unit suite: **613 passed** (up from 606 immediately post-15B.2D, net +7 new harness tests, +0 net change in protocol-test count; zero regressions). `ruff check` clean on all new/changed Python.

### 9. What Remains Unaffected
Everything listed in CORR-35 §13 remains unchanged. Additionally: the empirical-backend-implementation and PPO-training phases are unaffected by this entry (no dynamics model exists yet; none was fitted from this or any other run).

**Applies to phases**: 15B.3A (this execution and finding), the future phase that must explicitly choose and freeze a resolution from §6 (or a better-justified alternative) before any further official calibration run in the `dyncal_r2_L4` cell or any other cell showing similar overload-induced readiness instability is accepted as valid, the future empirical-backend-implementation phase (must not treat this attempt's decision-aligned data as clean steady-state ground truth until the gap is resolved).

## CORR-37 — Post-First-Run Separation of Execution Validity, Steady-State Fitting Eligibility, Health-Instability Outcomes, and Routing-Reconstruction Conservation (Phase 15B.3B)

### 1. Context
CORR-36 documented a genuine specification gap discovered after observing `campaign_sequence_index=1`'s real evidence, deliberately freezing **no** resolution. This correction resolves it, and is itself the **first post-data** methodology correction in the project's history — unlike CORR-31 through CORR-35 (all pre-execution preregistration corrections), CORR-37's rules were written after seeing attempt1's real behavior. For that reason it is versioned as a genuinely new file, `experiments/manifests/rl_dynamics_calibration_v2.yaml` and `docs/rl_dynamics_calibration_protocol_v2.md` (`protocol_version: 2`), rather than edited in place into v1 — v1 remains byte-for-byte intact as the historical record of what rules governed attempt1 when it actually ran (SHA-256 `2e6011495307245bcc85832d0b0e1aa6cbc6ea63e4655914eeb9c942c15fc2bf`, pinned in v2's `inherits_from`). No traffic was sent, no Kubernetes object was mutated (a read-only `kubectl get hpa`/`get pods`/`get events`/`get deployment` check was used only as supplemental forensic evidence), no RL/PPO training occurred, no dynamics model was fitted, and no sequence index other than 1 was touched.

### 2. Restart/Readiness Forensic Audit
Preserved artifacts (`cluster/pod_readiness_history.csv`, pod-identity snapshots, `routing/per_pod_request_counters.csv`) established: `demo-app-647f665b49-rpztw` first lost Ready at elapsed=30s (a brief blip), then repeatedly through the scored window; its `restartCount` transitioned 6→7 between elapsed=160s (NotReady) and 170s (NotReady, post-restart), recovering Ready by elapsed=180s; its own request counter reset from 48688 to 139 in that same window, consistent with a fresh process. `demo-app-647f665b49-xtzgk` also lost Ready repeatedly but never restarted (`restartCount` stayed 0). `desired`/`current` replicas remained exactly 2 throughout (no scaling event); pod identities before stabilization and after traffic were identical (no replacement). Supplemental **live** `kubectl get events` (still within the cluster's default retention window, ~35 minutes after the run) provided **definitive, non-speculative** root-cause evidence: `Warning Unhealthy ... Liveness probe failed: Get "http://10.244.0.19:8080/health": context deadline exceeded` (the app's own 3-second-timeout `/health` liveness probe, `periodSeconds=10`, `failureThreshold=5`, timed out repeatedly under CPU/request saturation) followed by `Normal Killing ... Container demo-app failed liveness probe, will be restarted`. Restart cause classification: **`LIVENESS_PROBE_TIMEOUT_UNDER_LOAD`** — not OOMKilled, not an application exit code, not eviction, not unresolved. This is a genuine, load-induced reliability finding, not a harness or Kubernetes-orchestration defect.

### 3. Two-Axis Validity Semantics Frozen (Protocol V2)
**Axis A — `execution_structural_validity`**: whether the harness mechanically executed the planned experiment correctly. The v1 `invalid_cell_structural_criteria` enumeration is retained unchanged. Explicitly reaffirmed as **never** structural invalidity on their own: high latency, client timeouts after legitimate dispatch, connection errors from an overloaded service, CPU saturation, application-induced readiness loss, a workload-caused container restart, or the tested application's own `/metrics` scrape target becoming unresponsive. **Axis B — `steady_state_fit_eligibility`**: independently, whether a run (or a specific decision-aligned sample within it) may enter the fixed-replica steady-state fitting dataset. A decision point is fit-ineligible whenever, at that exact timestamp, desired/current/ready/available do not all equal the cell's intended replica count — `current=2, ready=1` is never treated as `2/2/2/2` — with two frozen, mutually exclusive exclusion reasons: `LOAD_INDUCED_HEALTH_INSTABILITY` and `APPLICATION_TARGET_UNOBSERVABLE_UNDER_LOAD`. Neither implies structural invalidity. A correctly-executed overload attempt that legitimately degrades is `execution_structural_validity=VALID` **and** may simultaneously be `steady_state_fit_eligibility=STEADY_STATE_FIT_INELIGIBLE` — a real, storable, non-contradictory outcome, precisely so the campaign is never incentivized to selectively retry away inconvenient-but-legitimate evidence.

### 4. Decision-Point Eligibility Bookkeeping
All four frozen decision-aligned targets (60/90/120/150s) are always represented in the per-run table, each carrying its own `decision_point_observable` and `steady_state_fit_eligible` booleans plus an `exclusion_reason` — never imputed, interpolated, or silently dropped. Run-level classification (`STEADY_STATE_FIT_ELIGIBLE`/`_PARTIAL`/`_INELIGIBLE`) is derived from the count of fit-eligible points (4/some/0) — no final cell-level model-support decision is made from one attempt alone (§5).

### 5. Cell-Level Stable-Calibration-Support Rule (Frozen Before Repeats 2/3)
Precommitted now, before any repeat of `dyncal_r2_L4` or any other cell is observed: a 3-repeat fitting cell reaches `STABLE_CALIBRATION_SUPPORT` only if at least 2 of its 3 official repeats each provide ≥2 of 4 fit-eligible decision points; a 2-repeat interpolation-validation cell needs ≥1 of 2 repeats meeting that same bar. Below threshold: `OUTSIDE_STABLE_CALIBRATION_SUPPORT` — no simulator output is invented for that cell. Freezing this now, rather than after seeing repeats 2/3, is the entire point of the correction: a support threshold chosen after seeing which repeats look favorable would let performance decide methodology.

### 6. Root Cause and Correction of the 591-vs-1427 Routing Discrepancy
Attempt1's original routing summary (`routing/routing_summary.json`) reported `expected=591` vs `actual=1427`, marked `SUPERSEDED_PENDING_FORENSIC_RECONSTRUCTION` (preserved unmodified). Forensic diagnosis, confirmed by hand-deriving the exact interval-by-interval arithmetic from the raw preserved CSVs before trusting any code fix: `compute_expected_and_actual()` restricted the interval aggregate to Ready-**and**-observable pods only, silently dropping real traffic whenever the Ready pod for an interval lacked a joint counter sample there (common under scrape gaps); separately, `actual` was reconstructed as an unconstrained whole-span total per pod, never scoped to the same interval set as `expected`, capturing strictly more traffic. The two totals had no principled relationship. **Fix**: the interval aggregate is now built from every pod with a valid joint sample at that interval's two boundaries (Ready-set membership determines only who shares the resulting expected credit, never whose traffic counts toward the aggregate); `actual` is now the sum of each pod's own per-interval delta over the identical interval set — which conserves (`aggregate_total == sum(actual.values())`) by construction. Re-run against the same preserved raw data: `conservation_error == 0.0` exactly (corrected `expected`: rpztw=406.0, xtzgk=628.0; corrected `actual`: rpztw=361.0, xtzgk=673.0; both totals 1034.0). **No tolerance was relaxed or changed** — v1's zero-tolerance conservation requirement is retained unchanged and is now correctly achievable with the fixed reference implementation. Corrected classification: `HEALTHY_VARIANCE` (ratios 0.889/1.072, CV≈0.093) — written to a **new** artifact, `routing/routing_summary_forensic_v2.json`, never overwriting the original.

### 7. Attempt1 Final Multi-Axis Classification
`execution_structural_validity = VALID` (no Axis-A criterion violated; no evidence-collection failure — all required artifacts were successfully captured). `steady_state_fit_eligibility = STEADY_STATE_FIT_INELIGIBLE` (0 of 4 decision points fit-eligible: 3 `LOAD_INDUCED_HEALTH_INSTABILITY`, 1 `APPLICATION_TARGET_UNOBSERVABLE_UNDER_LOAD`). `performance_outcome`: legitimate, valid, severe overload degradation (high timeout/connection-error rate, near-saturated CPU, readiness/liveness instability, one liveness-triggered restart). `routing_diagnostic_status = HEALTHY_VARIANCE` (corrected). This repeat does not by itself satisfy `dyncal_r2_L4`'s cell-level stable-support requirement; repeats 2 and 3 remain to be executed in a future phase.

### 8. Ledger Disposition
Per `ledger_semantics_v2` (Axis A alone determines ledger state), slot 1 was corrected from its Phase 15B.3A provisional `INVALID_RETRY_REQUIRED` (pending this audit) to **`VALID`**, with `valid_attempt_id` set to the original attempt — the attempt is preserved in `all_attempt_ids`, and the full provisional-then-corrected history is recorded in the attempt's own `metadata.json` (`corr37_two_axis_reclassification` addendum), never silently erased. `next_resumable_index()` now correctly returns **2**. A transcription error in Phase 15B.3A's own report (which had stated index 2 as `val_dyncal_r7_L3_repeat1`) was caught and corrected during this audit by direct re-resolution against the authoritative `global_campaign_sequence.json`: index 2 is actually **`fit_dyncal_r4_L5_repeat1`** (fitting, `dyncal_r4_L5`, replicas=4, offered_rps=40.0).

### 9. Tests
`tests/unit/test_dynamics_calibration_harness.py` gained 22 tests (91 total, up from 69 immediately post-Phase-15B.3A): 4 for exact-zero routing conservation (including a direct reproduction of attempt1's real interval structure, hand-verified against the raw CSVs), 4 for two-axis representability (execution-valid-but-fit-ineligible, ledger reflecting Axis A only), 8 for decision-point fit-eligibility (current-2/ready-1 exclusion, unobservable-vs-health-instability distinction, never-imputed guarantee, all-four-targets-always-represented, run-level classification), 5 for the cell-level stable-support rule (fitting 2-of-3, validation 1-of-2, threshold value), and 4 for V2 versioning/inheritance (v1 hash pinned and matched, V2 explicitly post-data, global sequence/schedules inherited unchanged, v1 doc untouched). Full unit suite: **639 passed** (up from 617 immediately post-Phase-15B.3A; zero regressions). `ruff check` clean on all changed Python.

### 10. What Remains Unaffected
The 20 fitting cells, 9 interpolation cells, 8 replica-transition types, 4 workload-transition types, 102 global planned slots, 37 canonical schedules and their SHA-256 hashes, the global campaign sequence and its order, `campaign_sequence_index=1`'s identity, the 64 RPS ceiling, 400 max-in-flight, 5s timeout, all traffic shapes, all candidate model families, and every acceptance criterion not explicitly superseded above are unchanged from v1/CORR-32 through CORR-35. `rl_contract_version`/`reward_contract_version` (both still 2) and the HPA baseline (260%) remain untouched, confirmed live. No traffic was sent in this phase; no second calibration run was executed.

**Applies to phases**: 15B.3A (the execution this resolves), 15B.3B (this correction), the future empirical-dynamics-calibration EXECUTION phase (must apply the two-axis model, the frozen exclusion reasons, and the cell-level support rule exactly as frozen here for every subsequent run), the future empirical-backend-implementation and PPO-training phases (must consult `steady_state_fit_eligibility` per decision point, never raw execution validity alone, when deciding what data may enter fitting).

## CORR-38 — Post-Data Expansion From Ready-Eligible Routing Completeness to Complete Relevant-Pod Observability (Phase 15B.3C.3)

### 1. Context
Phase 15B.3C.1 fixed a domain-mismatch bug in `compute_expected_and_actual()` (expected scoped to allocatable intervals only, actual scoped to the whole jointly-observable domain) without a new CORR, since it was a straightforward implementation-conformance fix against already-frozen semantics. Phase 15B.3C.2 went further and closed a second loophole: an interval could still pass exact-zero conservation while being reconstructed from an INCOMPLETE subset of the pods relevant to it (a Ready pod missing one boundary sample was silently dropped from the aggregate rather than excluding the interval). Phase 15B.3C.2's own report framed this, too, as "an implementation-conformance tightening, not a methodology change," and its appendix answered "Was this compatible with CORR-35/37?" with a single "Yes." That framing conflated two distinct claims (see §5 below) and is corrected here.

### 2. Why Phase 15B.3C.2's Change Is Not Merely an Implementation Detail
Three facts, taken together, mean the population-completeness rule Phase 15B.3C.2 introduced cannot be silently treated as a pre-existing implementation requirement:
1. It changes which intervals are admissible for routing classification (index1: 8 of 11 intervals newly excluded; index2: 11 of 11).
2. It changes the routing evidence retained from real, already-observed runs — not from a future run designed with the new rule in mind, but from data collected under a materially different (Ready-serving-only) completeness assumption.
3. It changed BOTH index1's and index2's routing conclusions (index1: `HEALTHY_VARIANCE` → `INSUFFICIENT_ROUTING_SAMPLE`; index2: `ROUTING_PINNING_SUSPECTED` → `INSUFFICIENT_ROUTING_SAMPLE`).
A rule that is chosen — however scientifically sound and conservative — after observing that it changes the conclusions drawn from already-collected data is a post-data methodology decision by this project's own established precedent (CORR-27's MDP-contract-before-code discipline; CORR-32's pre-vs-post-data distinction; CORR-37's own explicit self-description as "the first post-data correction"). It must be versioned transparently, exactly as CORR-37 was, rather than folded into an unversioned bug-fix label.

### 3. What v1 Literally Said
`experiments/manifests/rl_dynamics_calibration_v1.yaml`'s `per_pod_routing_diagnostics.eligibility.basis` reads: "Ready-serving exposure: a pod is eligible only for the portion of the analysis window during which it was actually Ready and capable of serving traffic." Read literally, this scopes "eligible pod" (the population referenced by `reconstruction_consistency_check.compares`, "sum(actual_request_count_i across all reconstructed eligible pod series)") to Ready-serving pods, not to every pod that might exist. Phase 15B.3C.2's broadened `relevant_pod_set_j` — including NotReady-but-existing pods and pods known only via counter-series presence — goes beyond this literal wording.

### 4. Empirical Motivation
Real preserved data from both executed slots shows pods that were confirmed `Running` in Kubernetes (via the authoritative `pod_readiness_history.csv`, sourced directly from `kubectl get pods -o json`, per this project's established preference for authoritative object state) for the entire scored window, yet whose Prometheus counter series was absent for large stretches of that window: `demo-app-647f665b49-rpztw` in index1 (correlated with its own confirmed `LIVENESS_PROBE_TIMEOUT_UNDER_LOAD` restart storm, CORR-36) and `demo-app-647f665b49-6lld4` in index2 (continuously `Running`, `restartCount` 0→1, its own `Ready` status flapping, counter series present at only 2 of 12 boundaries). Both pods could legitimately have carried real, unknown traffic during their counter-absent windows even though neither was continuously Ready. A completeness rule scoped only to Ready-serving pods would silently permit exactly the incomplete-population false-conservation this correction closes.

### 5. The Precise Distinction (Correcting Phase 15B.3C.2's Appendix)
Two separate claims must not be conflated:
- **The OLD subset-based implementation** — checking each pod's own two-boundary presence independently and silently dropping a missing pod from the aggregate rather than excluding the interval — was **NOT compatible** with the trustworthy-reconstruction principle CORR-35/v1 already intended. The frozen manifest text (`joint_observability_gap_handling`: "excluded consistently from BOTH... never estimated or fabricated"; `on_discrepancy_beyond_tolerance`: "Classify INSUFFICIENT_ROUTING_SAMPLE... never fabricated") already implied whole-interval exclusion on any gap, not silent per-pod dropping. This part is a genuine, pre-existing implementation defect — correctly fixed in Phase 15B.3C.2 without a new CORR, and remains so.
- **The NEW complete-`relevant_pod_set_j` population** — extending completeness beyond the literal Ready-serving-only wording of v1's `eligibility.basis` to also include NotReady-but-existing and counter-series-only-known pods — is **NOT** something the literal prior text required. It is a post-data conservative methodology EXTENSION: scientifically well-motivated (§4), strictly more conservative (it can only exclude more, never fabricate less, never relax a threshold), but a genuinely new rule chosen after seeing its effect on retained evidence. This part requires CORR-38 and the version bump to `protocol_version: 3`.

Phase 15B.3C.2's appendix answer "Yes, compatible with CORR-35/37" was correct for the first claim and imprecise for conflating it with the second. This correction resolves that imprecision. Phase 15B.3C.2's CODE was correct and is not modified by this correction; only its methodological status is now versioned rather than implied.

### 6. What Is Frozen (v3 / CORR-38)
`experiments/manifests/rl_dynamics_calibration_v3.yaml` and `docs/rl_dynamics_calibration_protocol_v3.md` freeze: (a) `relevant_pod_set_j` as the union of Kubernetes-known-any-status-at-either-boundary, Ready-at-either-boundary, and counter-series-known-at-either-boundary pod identities; (b) Kubernetes pod UID as the canonical identity key (pod name is the human-readable join label against Prometheus's own `pod` metric label, which carries no UID); (c) the four-way interval partition `ALLOCATABLE`/`ZERO_READY`/`EXCLUDED_INCOMPLETE_POD_COUNTER_OBSERVABILITY`/`EXCLUDED_OTHER_NO_RELEVANT_PODS`; (d) a requirement that future counter collection record the authoritative Kubernetes-known pod population per tick (not merely a previous-tick-relative gap check) and distinguish whole-query failure from single-series absence from Kubernetes-listing failure. **Routing classification thresholds themselves are UNCHANGED**: structural `<0.01`; healthy `[0.5,1.5]` with CV`<=0.30`; suspected catch-all; minimum 2 assessable pods with `expected>=50`; exact-zero conservation tolerance. Only the population admitted into the reconstruction changes — never the criteria applied to it once admitted.

### 7. No New Traffic Collected
No traffic was sent, no Kubernetes object was mutated, no dynamics model was fitted, and no PPO training occurred while freezing this correction. `configs/traffic_profiles/unseen_eval.yaml` was not inspected. Sequence index 3 was not executed. v3 was frozen purely by re-analyzing already-preserved raw artifacts from index1 and index2 with the (already Phase-15B.3C.2-implemented, now formally versioned) stricter gate.

### 8. Historical Chain Preserved, Nothing Erased
Index1's routing status progressed `ROUTING_PINNING_SUSPECTED` (original, buggy) → `HEALTHY_VARIANCE` (Phase 15B.3C.1 domain-partition fix) → `INSUFFICIENT_ROUTING_SAMPLE` (Phase 15B.3C.2/CORR-38 complete relevant-pod gate). Index2's progressed `ROUTING_PINNING_SUSPECTED` (original, buggy) → `ROUTING_PINNING_SUSPECTED` (Phase 15B.3C.1, corrected numbers but same label) → `INSUFFICIENT_ROUTING_SAMPLE` (Phase 15B.3C.2/CORR-38). Every original and intermediate artifact (`routing_summary.json`, `routing_summary_forensic_v2.json`, `routing_summary_interval_domain_forensic_v2.json`, `routing_intervals_interval_domain_forensic_v2.csv`, `routing_summary_complete_pod_observability_v3.json`, `routing_intervals_complete_pod_observability_v3.csv`) is preserved unmodified; each attempt's `metadata.json` carries a full, non-destructively-appended addendum chain (`post_execution_review` → `corr37_two_axis_reclassification` → `corr37_interval_domain_reconciliation` → `corr37_complete_observability_gate_reconciliation`).

### 9. Axis-A / Axis-B Unaffected
`execution_structural_validity` (Axis A) and `steady_state_fit_eligibility` (Axis B) are unchanged by this correction — `INSUFFICIENT_ROUTING_SAMPLE` has never been an Axis-A trigger in this harness (only `STRUCTURAL_ROUTING_FAILURE` is), and nothing in v2's ledger semantics requires a downgrade for a routing-evidence-insufficiency outcome traceable to genuine application/infrastructure observability limitations rather than a harness/collector defect. Both slots remain ledger `VALID`, execution-structural-validity `VALID`, fit-eligibility `STEADY_STATE_FIT_INELIGIBLE`. Neither slot is retried.

### 10. Tests
`tests/unit/test_dynamics_calibration_harness.py` gained coverage for: V2 preservation/hash-pinning and V3 versioning metadata; Kubernetes-aware missing-pod-series detection on the first tick, for a continuously-missing pod, and distinguishing whole-query failure from Kubernetes-listing failure; pod-identity (UID) consistency checks, including that a reused name with a changed UID is never silently merged; and V3-code regression recomputation confirming both index1 and index2 resolve to `INSUFFICIENT_ROUTING_SAMPLE`. See `task_outputs/phase_15b3c3_routing_relevance_protocol_v3_freeze.md` §15 for exact counts.

### 11. What Remains Unaffected
The 20 fitting cells, 9 interpolation cells, 8 replica-transition types, 4 workload-transition types, 102 global planned slots, 37 canonical schedules and their SHA-256 hashes, the global campaign sequence and its order, decision-aligned timestamps (60/90/120/150), the 64 RPS ceiling, 400 max-in-flight, 5s timeout, all traffic shapes, all candidate model families, the two-axis validity model and ledger semantics (v2/CORR-37), the cell-level stable-support rule, and every routing classification THRESHOLD are unchanged. `rl_contract_version`/`reward_contract_version` (both still 2) and the HPA baseline (260%) remain untouched, confirmed via direct file inspection. `configs/traffic_profiles/unseen_eval.yaml` was not inspected or modified.

**Applies to phases**: 15B.3C.1/15B.3C.2 (the implementation work this formally versions), 15B.3C.3 (this correction), the future empirical-dynamics-calibration EXECUTION phase starting at `campaign_sequence_index=3` (must apply the v3 `relevant_pod_set_j` definition, UID-aware identity checks, and the future counter-observability collection requirements to every subsequent run), the future empirical-backend-implementation and PPO-training phases (unaffected — no dynamics model has been fitted from any run yet, and routing status remains diagnostic-only, never itself a fitting-eligibility criterion).

## CORR-39 — Unverified External Workload-Delivery-Path Integrity During a Sustained Connection-Error Block (Phase 15B.3M.1)

### 1. What Was Found
Index12 (`fit_dyncal_r8_L3_repeat1`, campaign_sequence_index=12, replicas=8, offered_rps=64.0, Phase 15B.3M) exhibited a **121-second unbroken block of 100% `connection_error`** (elapsed 59–179s of the 180s traffic window, reconstructed directly from `traffic_summary.csv` — Phase 15B.3M's report had understated this as "65 consecutive seconds, elapsed≈115–179s," which this correction supersedes as the accurate figure). For the final ~60 seconds of that block (elapsed≥120s), Kubernetes Ready was confirmed at the full intended 8/8 and aggregate application CPU was falling toward idle (1.08→0.23 cores between t=120 and t=150) — i.e., the pods were not overloaded during that sub-window, yet 100% of client connections still failed.

Forensic audit (Phase 15B.3M.1, `task_outputs/phase_15b3m1_index12_external_load_path_attribution_audit.md`) established, by direct source inspection, that `wrong_traffic_path_used` — the one existing `invalid_cell_structural_criteria` entry whose name plausibly covers this scenario — is implemented (`scripts/run_dynamics_calibration_slot.py:start_loadtest_tunnel`/`main()`) to check ONLY whether the Minikube-service tunnel's URL resolves once at launch (a 30-second polling window). No code anywhere checks the tunnel's, the `demo-app-loadtest` Service's, or the kube-proxy dataplane's continued liveness/correctness for the remainder of an official run. No historical Endpoints/EndpointSlice snapshot for either Service has ever been captured by any script in this repository. The tunnel's own stdout/stderr log for this attempt survived by chance (an OS temp file, not a tracked artifact) and showed only two lines, both written within 2 seconds of launch, with total silence for the rest of the ~13-minute attempt — evidentially ambiguous (equally consistent with a healthy, silently-relaying process and with an unlogged death).

Separately, Phase 15B.3M's report cited a non-invalidating-outcome string, `connection_errors_caused_by_overloaded_service`, to justify treating the block as legitimate performance evidence. That exact string does not exist anywhere in this repository. `experiments/manifests/rl_dynamics_calibration_v1.yaml`'s actual `valid_performance_outcomes_not_grounds_for_invalidity` list contains no entry naming connection errors at all; the closest entry, `client_errors_from_legitimate_overload`, presupposes the overload is genuinely occurring in the studied system — which the Ready=8/8, falling-CPU evidence for elapsed≥120s directly contradicts. This citation was inaccurate and has been retracted (Phase 15B.3M.1 §12).

### 2. Why This Is a Genuine Gap, Not Merely a Reporting Correction
Unlike prior post-data reporting corrections in this project (CORR-37's Axis separation, Phase 15B.3J.1's restart-evidence taxonomy, Phase 15B.3L.1's SLA/health-dimension separation, Phase 15B.3K's CSV-fieldnames bug), the underlying FACT this gap turns on — did the official external workload-delivery path actually keep functioning throughout index12's traffic window — is not recoverable from any already-preserved evidence, and cannot become recoverable by re-reading or re-deriving from existing artifacts more carefully. The evidence needed to resolve it was simply never collected, by design of the pre-15B.3M.1 harness. This is a genuine specification/coverage gap in `invalid_cell_structural_criteria`'s effective reach, discovered post-data, exactly analogous in character (though not in resolution) to CORR-38's post-data discovery that Ready-serving-only routing eligibility was too narrow.

### 3. What Is NOT Decided Here
This correction does **not**:
- Retroactively reclassify index12's `execution_structural_validity` (remains `VALID` under every currently-implemented, currently-checked criterion — none of which this gap violates, since none of them ever attempted to check sustained tunnel/Service integrity in the first place).
- Retroactively reclassify index12's ledger state (slot12 remains `VALID`).
- Change `steady_state_fit_eligibility` for t=120/t=150 (both remain `steady_state_fit_eligible=true` under the unchanged, currently-implemented rule: sample observable + Ready count matches intended replicas — no new eligibility criterion was invented).
- Invent a new structural-invalidity criterion, a new eligibility exclusion, a new SLA definition, a new routing threshold, or any change to the 20/9/8/4/102/37/64/400/5s matrix, the HPA baseline, or the RL/reward contracts.
- Require or perform a retry of index12.

Choosing any of those outcomes now, based on this one attempt's ambiguous evidence, would itself be exactly the kind of "post-data decision that changes conclusions from already-collected data" this project's own established discipline (CORR-27, CORR-32, CORR-37, CORR-38) requires be made transparently and deliberately — not silently folded into a bug-fix label, and not decided unilaterally by whichever single agent happens to be auditing one attempt.

### 4. What IS Frozen Now (Prospective Hardening Only)
Passive, non-invasive tunnel-lifecycle evidence capture (Phase 15B.3M.1, `scripts/dynamics_calibration_lib.py:classify_tunnel_termination_action`, `scripts/run_dynamics_calibration_slot.py`'s `tunnel_lifecycle` tracking): every future official attempt now records the tunnel's launch command, PID, launch timestamp, destination URL, a passive `Popen.poll()` liveness check immediately after traffic completes, and whether cleanup found the process already dead versus explicitly terminated it — persisted to `cluster/loadtest_tunnel_lifecycle.json`. This makes no new network call and sends no probe traffic through the tunnel. This closes the evidence-collection gap for FUTURE runs; it does not and cannot retroactively resolve index12's own unverifiable window.

### 5. What Remains an Open Decision (Explicitly Deferred, Not Resolved Silently)
Whether, and how, a future run exhibiting this same ambiguous pattern (sustained 100% connection_error with no corroborating overload evidence, and no tunnel-liveness proof either way) should be treated — e.g., whether `wrong_traffic_path_used`'s scope should be formally extended to include a post-hoc liveness-evidence check once the Phase 15B.3M.1 instrumentation exists, whether such a run should default to `INVALID_RETRY_REQUIRED` pending confirmed tunnel health, or whether the existing `valid_performance_outcomes_not_grounds_for_invalidity` list should gain an explicit, narrowly-scoped entry for this exact evidentiary state — is **not decided by this correction**. It requires deliberate methodology review before being adopted, not an ad hoc choice made while auditing one already-collected attempt under time pressure to reach a preferred conclusion.

### 6. What Remains Unaffected
The 20 fitting cells, 9 interpolation cells, 8 replica-transition types, 4 workload-transition types, 102 global planned slots, 37 canonical schedules and their SHA-256 hashes, the global campaign sequence and its order, decision-aligned timestamps (60/90/120/150), the 64 RPS ceiling, 400 max-in-flight, 5s timeout, the two-axis validity model, all routing classification thresholds, the CPU-normalization definitions (Phase 15B.3F.1), the restart-evidence taxonomy (Phase 15B.3J.1), and the SLA thresholds/dual-metric-source discipline (Phase 15B.3L.1) are all unchanged. `rl_contract_version`/`reward_contract_version` (both still 2) and the HPA baseline (260%) remain untouched, confirmed live. No traffic was sent, no Kubernetes object was mutated, and no dynamics model was fitted while freezing this correction. `configs/traffic_profiles/unseen_eval.yaml` was not inspected. Sequence index 13 was not executed pending this methodology review.

**Applies to phases**: 15B.3M (the report this corrects, causal wording and citation only — its raw measurements are unaffected and unmodified), 15B.3M.1 (this correction), every future official dynamics-calibration attempt (must carry the new passive tunnel-lifecycle evidence), and any future methodology-review phase that decides how to treat this class of evidentiary gap going forward (not yet scheduled, not resolved here).

### 7. Resolution (Phase 15B.3M.2) — The Open Decision From Sec.5 Is Now Answered
Sec.5's deferred policy question is resolved, versioned as a new protocol layer rather than an in-place edit: `experiments/manifests/rl_dynamics_calibration_v4.yaml` and `docs/rl_dynamics_calibration_protocol_v4.md` freeze the distinction between **generator schedule execution integrity** (`offered==sent`, `dropped_overload==0` — proves only that the generator attempted its schedule) and **official workload-route integrity** (did the schedule actually traverse the Minikube-service tunnel → `demo-app-loadtest` Service → pods path), plus a `CONFIRMED`/`FAILED`/`UNVERIFIED` classification gated by an explicit materiality criterion (zero-success interval, overlapping the scored window, covering ≥25% of total traffic duration). One new `invalid_cell_structural_criteria` entry, `official_workload_route_integrity_unverified`, covers the material-`UNVERIFIED` case; `wrong_traffic_path_used` (existing, reused) covers confirmed harness-side failure. Applying this to index12's real numbers (121s zero-success interval, 67% of traffic duration, overlapping the scored window, no tunnel evidence) yields `UNVERIFIED` → structurally invalidating: `fit_dyncal_r8_L3_repeat1_attempt1_20260916T130416Z` is reclassified `execution_structural_validity=INVALID`, ledger slot12 `INVALID_RETRY_REQUIRED`. All raw evidence remains fully preserved (not deleted, not rewritten); the attempt does not contribute to formal calibration fitting or `dyncal_r8_L3` cell support (0 of 3 planned repeats now formally usable). `build_protocol_provenance()` emits `active_calibration_protocol_version=4`, `active_correction_reference="CORR-39"` for every attempt from this point forward. See `task_outputs/phase_15b3m2_corr39_route_integrity_methodology_resolution.md`.

**Correction of a documentation error from Phase 15B.3M.2's own report**: that report's prose said the new evidence contract would apply "from index13 onward" — inaccurate, since `build_protocol_provenance()` takes no `campaign_sequence_index` argument at all and structurally cannot special-case which slot runs next. The contract applies to **the very next official attempt**, whichever slot that is — concretely, the index12 retry (attempt2), not index13. Corrected in Phase 15B.3M.3.

### 8. Proof-Sufficiency Hardening (Phase 15B.3M.3) — Process Liveness Is Not Forwarding Proof
Sec.7's resolution, as first implemented, allowed a material zero-success interval to be classified `CONFIRMED` on nothing stronger than `tunnel_confirmed_dead_during_interval=False` — i.e. a bare `Popen.poll() is None` reading. This is a real gap: process liveness proves only that the helper has not exited; it proves nothing about whether its local listener is still bound, whether it is still forwarding, or whether any request actually crossed the harness boundary into the tested workload. Left uncorrected, this would have silently let "tunnel process alive" become equivalent to "official workload route confirmed" for the very next live attempt — reintroducing an index12-shaped ambiguity as a false `CONFIRMED` instead of an honest `UNVERIFIED`.

**Corrected contract** (`scripts/dynamics_calibration_lib.py: classify_route_integrity()`, now returning a full evidence dict, never a bare string): a material interval consults four separate evidence layers, none collapsed into one boolean:
- **Layer B (tunnel process state)**: PID, `poll()`, exit code — necessary but insufficient. `tunnel_process_alive_throughout is False` → `FAILED`.
- **Layer C (local listener state)**: read-only `lsof -nP -iTCP:<port> -sTCP:LISTEN` (`scripts/run_dynamics_calibration_slot.py: check_local_listener_present()`) — necessary but insufficient. `local_listener_present_throughout is False` → `FAILED`.
- **Layer D (loadtest-Service endpoint state)**: `demo-app-loadtest` Endpoints ready-count, sampled periodically — supporting evidence only, never sufficient alone.
- **Layer E (delivered-request evidence)**: positive proof that at least one request crossed the harness boundary into the tested application during the interval — any HTTP response (2xx/4xx/5xx — success is never required), a non-zero server-observed `/work` RPS, or a per-pod request-counter increment (`scripts/dynamics_calibration_lib.py: compute_delivered_request_evidence()`, checked against `metrics.csv`/`per_pod_request_counters.csv` by wall-clock epoch). **Layer E present is the only path to `CONFIRMED` for a material interval** — Layers B/C/D being individually healthy, even all three at once, is explicitly insufficient (this is precisely the corrected index12 regression case: process alive, listener present, endpoints valid, zero server RPS, 100% client connection_error → `UNVERIFIED`, never `CONFIRMED`).

Tunnel/helper process liveness is necessary evidence when that helper is required, but **process liveness alone is not sufficient proof of forwarding integrity during a material zero-success interval.**

**Versioning decision**: Protocol V4 is retained as the active protocol — a new V5 was deliberately NOT created. V4 had not yet governed any live attempt when this proof-sufficiency gap was found and closed (index12 attempt1 predates V4 entirely and is unaffected by this section), so this is a pre-first-use implementation correction to a just-introduced mechanism, not a post-data methodology reversal of the kind that required V2/V3/V4's own version bumps (CORR-37/38/39 sec.7) — those each changed a rule's substance after observing what it did to already-collected data; this closes a proof-standard gap in CORR-39's own not-yet-exercised implementation before it ever ran live. `experiments/manifests/rl_dynamics_calibration_v4.yaml` and `docs/rl_dynamics_calibration_protocol_v4.md` are amended in place to reflect the corrected evidence layers; their `protocol_version: 4`/`correction_reference: CORR-39` identity is unchanged.

The 25% materiality threshold, the scored-window-overlap requirement, index12's own retrospective disposition (`INVALID_RETRY_REQUIRED`, unresolved, not re-audited), and every item in sec.6 remain unchanged by this hardening. See `task_outputs/phase_15b3m3_corr39_route_integrity_proof_sufficiency_hardening.md`.

### 9. Temporal Coverage Hardening (Phase 15B.3M.4) — One Delivered Request Does Not Confirm an Entire Interval

Sec.8's Layer E (`delivered_request_evidence_present`) resolved a proof-*sufficiency* gap but left a proof-*sufficiency-in-time* gap unaddressed: `compute_delivered_request_evidence()` asked only whether at least one request crossed the harness boundary **anywhere** inside a material zero-success interval. For a short interval this is adequate, but for a long one (index12's own interval was 121 of 180 seconds) it allowed a single delivered request near the start of the interval to `CONFIRM` the entire remainder, even if the route then silently stopped forwarding for good and the rest of the interval carried zero further delivery evidence, with the tunnel process, local listener, and Service endpoints all still nominally healthy throughout. Left uncorrected, this would have let a route that failed at second 60 of a 180-second interval reach `CONFIRMED` on the strength of one response observed at second 61 — the temporal analogue of sec.8's process-liveness gap, one level deeper.

**Core principle now frozen**: for a material zero-success interval, route-integrity confirmation requires temporally sufficient delivery evidence across that interval, not merely evidence that at least one request crossed the route somewhere inside it. **Positive route-delivery evidence is temporally scoped. Evidence that the route forwarded one request at one point in a material zero-success interval does not establish forwarding integrity for the remainder of that interval. A material interval is `CONFIRMED` only when every evidence window covering that interval has positive delivered-request evidence and no direct harness-route failure.** `CONFIRMED` at one point in time does not imply `CONFIRMED` at a later point in the same interval — a persistent tunnel/dataplane path can fail partway through and never recover, and this hardening ensures that failure is no longer masked by an earlier, unrelated success.

**Corrected contract** (`scripts/dynamics_calibration_lib.py`):
- `ROUTE_INTEGRITY_EVIDENCE_WINDOW_SECONDS = 10` (frozen) — the temporal resolution at which delivery evidence is now evaluated, matching the pre-existing tunnel/listener/endpoint observer cadence and the metrics/counter collector cadence; no new active monitoring cadence is introduced, and the partition is fully reproducible offline from already-persisted files.
- `find_all_zero_success_runs()` (new) supersedes relying solely on `longest_zero_success_run()` for classification purposes — a single attempt can contain more than one independently material zero-success interval, and every one of them is now evaluated (`longest_zero_success_run()` is retained only as a convenience/reporting wrapper).
- `is_interval_material()` applies the **unchanged** sec.5 materiality gate (≥25% of traffic duration, overlapping the scored window) to each candidate interval independently.
- `partition_into_evidence_windows()` splits each material interval into deterministic, half-open `[start, end)` windows aligned to the interval's own start (not absolute wall-clock), including a shorter final partial window that is never dropped — e.g. interval `[59,180)` → `[59,69), [69,79), ..., [169,179), [179,180)`.
- `compute_delivered_request_evidence_windows()` (new; the interval-wide `compute_delivered_request_evidence()` is retained for backward-compatible reporting only, never used for classification going forward) computes delivery evidence **per window** from three sources, any one sufficient: an actual HTTP response (2xx/4xx/5xx, or any `http_*` outcome; `timeout`/`connection_error` are not proof) observed in `requests.csv` within that window's traffic-elapsed bounds; a positive server-side `/work` RPS in `metrics.csv` within that window's wall-clock (epoch) bounds; or a per-pod request-counter increase in `per_pod_request_counters.csv` whose **both** bounding samples localize to that same window — an increase whose bounding samples straddle two windows is credited to **neither** window (conservative, deliberate, never duplicated).
- `aggregate_observer_evidence_for_window()` (new) maps the already-captured ~10s periodic tunnel/listener/endpoint observer samples into whichever single window contains each sample's own timestamp — a window with no sample reports `None`/unknown, never fabricated healthy.
- `classify_route_integrity_window()` (new) applies sec.8's same three-way evidentiary rule (`FAILED` on a confirmed harness-component failure; `CONFIRMED` only on positive delivered-request evidence with no such failure; `UNVERIFIED` otherwise) **per window** rather than per interval.
- `classify_route_integrity()` (redesigned) now takes `material_intervals` (all materially-qualifying runs) and a flattened `evidence_windows` list (every window's classification across every material interval), and aggregates at the attempt level: `FAILED` if **any** window is `FAILED` (a single confirmed failure anywhere is decisive, never downgraded by other healthy windows); `CONFIRMED` only if **every** window across **every** material interval is `CONFIRMED` (full temporal coverage); `UNVERIFIED` otherwise — explicitly, positive evidence in one window never upgrades a different, evidence-lacking window. The result carries `material_intervals`, `evidence_windows`, `confirmed_window_count`, `unverified_window_count`, `failed_window_count`, and `full_temporal_coverage`, so the classification is never divorced from the windowed evidence that produced it.

**Regression proof**: the exact index12-shaped case — a 120s material interval partitioned into 12 windows, window1 with a server-side counter increase (`CONFIRMED`) and windows 2–12 with tunnel/listener/endpoints all nominally healthy but zero delivery evidence (`UNVERIFIED` each) — now correctly resolves attempt-level `UNVERIFIED`, not `CONFIRMED`. A full-coverage case (every window independently `CONFIRMED`, however catastrophic client-side performance is) still resolves `CONFIRMED`. A mid-run harness failure (an early run of `CONFIRMED` windows followed by a `FAILED` window) resolves attempt-level `FAILED`, never downgraded by the earlier confirmed windows.

**Artifacts**: the full per-window temporal audit is persisted to `cluster/loadtest_route_integrity_timeline.json` for every future official attempt; `metadata.json`'s `route_integrity` key carries only the summary (`classification`, `reason`, `material_intervals`, `full_temporal_coverage`, `confirmed_window_count`, `unverified_window_count`, `failed_window_count`) — the full per-window timeline is never duplicated wholesale into `metadata.json`.

**Versioning decision**: Protocol V4 is retained — a new V5 was deliberately NOT created. Exactly as with sec.8, V4 had not yet governed any live attempt when this temporal-coverage gap was found and closed (index12 attempt1 predates V4 entirely, remains `INVALID_RETRY_REQUIRED`, and is not re-audited by this section), so this is a second pre-first-use implementation correction to a just-introduced mechanism, not a post-data methodology reversal. `experiments/manifests/rl_dynamics_calibration_v4.yaml` and `docs/rl_dynamics_calibration_protocol_v4.md` are amended in place to document the windowed evidence contract; their `protocol_version: 4`/`correction_reference: CORR-39` identity is unchanged.

The 25% materiality threshold, the scored-window-overlap requirement, sec.8's four evidence-layer taxonomy (now applied per window rather than per interval), index12's own retrospective disposition, and every item in sec.6 remain unchanged by this hardening. No traffic was sent, no Kubernetes object was mutated, and the index12 retry (attempt2) was not executed while freezing this correction. See `task_outputs/phase_15b3m4_corr39_temporal_route_coverage_hardening.md`.

### 10. Window-Local Delivered-Request Attribution Hardening (Phase 15B.3M.5) — An Observation Timestamp Is Not an Event-Attribution Timestamp

Sec.9's per-window evidence, as first implemented, established WHEN a window's evidence was *observed* but not always WHEN the underlying delivery *event* actually occurred relative to that window's own boundaries. Two concrete gaps were found and closed, both discovered by direct source audit before this contract ever governed a live attempt:

**A. HTTP-response attribution used completion time, not dispatch time.** A request dispatched at t=58.5 (before a `[59,69)` window even begins) whose response happened to complete at t=61.0 (inside that window) was being credited to `[59,69)` — but that request's route-entry attempt began before the window, so its eventual response proves nothing about what happened specifically during `[59,69)`. **Fixed**: `compute_delivered_request_evidence_windows()` now attributes an HTTP response by `actual_dispatch_elapsed_seconds` — the `time.monotonic()` timestamp captured as the literal first statement inside `loadgen/traffic_generator.py:_dispatch_single_request()`, before URL construction or the `httpx` call itself (confirmed by direct source read; the scheduling gap between `asyncio.create_task()` and this timestamp is a single event-loop tick, not materially different from true attempt-start). A request counts as window-local delivery evidence for window W only when its dispatch falls in W AND it eventually produced a genuine HTTP response (2xx/4xx/5xx/`http_*`); its completion time is retained for provenance only, never for window-membership.

**B. Server-observed `request_rate_rps` is a 60-second rolling rate, not a point-in-time count.** Direct audit of `collectors/metrics_collector.py` confirmed the exact PromQL: `sum(rate(demo_app_requests_total{namespace="...",route="..."}[1m]))`, sampled at the collector's own ~10s cadence (`sampling_interval_seconds`). A positive value at t=65 reflects Prometheus's 60-second lookback window (roughly t=5–65), not proof that a request specifically entered the route during `[59,69)` or any other single 10-second evidence window. Treating a positive sample inside a window as decisive per-window proof — as sec.9's first implementation did — conflated *observation timestamp* with *event-attribution timestamp*. **Fixed**: `request_rate_rps` is now recorded per window as `rolling_server_rps_observed`, explicitly flagged `rolling_server_rps_supporting_only=True`, and is **never** included in `delivered_request_evidence`. The canonical `request_rate_rps` metric itself, and its use in dynamics calibration/RL observation/throughput analysis, is entirely unchanged — this correction concerns only its use as forensic per-window route-integrity proof.

**Raw per-pod counters remain the preferred server-side evidence, with stricter provenance.** A counter increase is credited to a window only when BOTH of its bounding samples' `timestamp_epoch` fall fully inside that window's own epoch bounds (unchanged principle from sec.9); an interval that only partially overlaps a window is now explicitly recorded as a `counter_intervals_rejected_as_cross_window` entry for that window, never silently dropped and never smeared across two windows. Given ~10-second counter-sampling cadence against ~10-second evidence windows, many real counter intervals will straddle a boundary and be usable for neither adjacent window — this is accepted as correct, conservative behavior (an `UNVERIFIED` window from a straddling interval is preferable to fabricated temporal precision), not loosened by duplicating the increment.

**A genuine pre-existing defect was also found and fixed while auditing this evidence path**: both `compute_delivered_request_evidence()` (sec.8) and the first implementation of `compute_delivered_request_evidence_windows()` (sec.9) read `row.get("pod_name")` from `per_pod_request_counters.csv` rows — but the real column, written by `scripts/per_pod_counter_collector.py`'s own `CSV_FIELDNAMES`, is `pod`. This meant the counter-increment evidence source had **never actually fired** in either function, silently, since its introduction — every counter row was skipped by the `pod is None` guard. Corrected to `row.get("pod")` in both functions; both had never yet governed a live attempt, so this is a pre-first-use implementation-correctness fix, amended in place without a version bump, exactly like sec.8's and sec.9's own corrections.

**Core principle now frozen**: *Evidence is window-local only when the event or count change can be attributed to requests whose route-entry/delivery occurred within that window. A metric observed inside a window is not automatically evidence generated inside that window. Do not confuse an observation timestamp with an event-attribution timestamp.*

**What did NOT change**: the `ROUTE_INTEGRITY_EVIDENCE_WINDOW_SECONDS=10` resolution, the half-open window-boundary convention, the 25% materiality threshold, the scored-window-overlap requirement, and sec.9's attempt-level aggregation rule (`FAILED` if any window `FAILED`; `CONFIRMED` only if every window is `CONFIRMED`; `UNVERIFIED` otherwise) are all unchanged — this hardening corrects *how* `delivered_request_evidence` is established inside a window, not the windowing or aggregation machinery itself. `classify_route_integrity_window()` and `classify_route_integrity()` required no changes.

**Versioning decision**: Protocol V4 is retained — a third pre-first-use hardening pass, not a V5. Index12 attempt1 remains `INVALID_RETRY_REQUIRED`, unaffected and not re-audited. See `task_outputs/phase_15b3m5_corr39_window_local_delivery_attribution_hardening.md`.

## CORR-40 — Stable Host-to-Kubernetes Workload Transport Transition (Phase 15B.3N.3)

### Problem

The V4/CORR-39-governed SSH-based host-forwarding mechanism (`minikube service --url`, a local SSH tunnel from the Mac host to the Service's ClusterIP) repeatedly fails under this project's actual sustained fresh-connection workload (`connection_mode=fresh`, ~64 new TCP connections/second for 180 seconds). Index12 attempt2 — the first live attempt V4's hardened route-integrity classifier ever governed — exhibited this directly: the tunnel's local listener disappeared at elapsed ~60s and never returned, while the wrapping process remained alive throughout, and CORR-39's classifier correctly returned `FAILED`. A dedicated engineering investigation (Phase 15B.3N.1: root-caused the failure to an unsupervised child SSH process; Phase 15B.3N.2: found that direct harness ownership of the SSH process does not fix it, and found and provisionally validated a raw-TCP Docker-network relay as a stable alternative; Phase 15B.3N.2A: durably reconstructed and preserved all decisive evidence, reproducing the SSH failure a third independent time in the same ~60–67s window, reconfirming the relay's stability across two further full-length runs, and correcting an earlier overclaimed latency penalty for the relay that did not survive proper replication) established that this failure is **structural**, not an evidentiary gap — CORR-39's own classification logic is not at fault and is not being changed.

### Evidence

Durable, permanently preserved raw evidence for every claim below lives under `persistent-data/transport_validation/` (never part of the official campaign ledger, fitting support, or model-fitting dataset — see that directory's own `README.md` for the structural exclusion argument). Summary: the SSH-tunnel mechanism failed the exact frozen `dyncal_r8_L3` workload three independent times (attempt2 itself, elapsed ~60s; a harness-owned direct-SSH reconstruction, elapsed 66.8s twice across two phases) regardless of process-ownership model; a raw TCP relay container (`alpine/socat`, `--restart=no`, attached to the Minikube Docker network, forwarding a Docker-published loopback port to the Minikube node's `demo-app-loadtest` NodePort) survived the identical exact-workload replay twice (once clean, once with injected mid-run pod churn), with zero container/listener exceptions across 121 combined observations, and reduced `connection_error` by roughly two orders of magnitude versus the SSH mechanism (183–880 vs. 9,000+), with the shortfall properly absorbed by legitimate `timeout`s against a CPU-starved backend rather than transport-level refusals. Multi-pod Service-level routing through the relay was independently confirmed via per-pod request counts.

### Decision

Future official workload traffic uses the Docker-network raw-TCP-relay-to-NodePort transport (`socat TCP-LISTEN:<port>,fork,reuseaddr TCP:<node-ip>:<nodeport>`), frozen as Protocol V5 (`experiments/manifests/rl_dynamics_calibration_v5.yaml`, `docs/rl_dynamics_calibration_protocol_v5.md`). This is implemented deterministically in the calibration harness with no runtime fallback to the SSH mechanism.

### Scientific Impact

The concrete workload-transport access path structurally changes: `Service ClusterIP via SSH tunnel` becomes `Service NodePort via Docker-network raw TCP relay`. This is **not** claimed to be an identical path — it is a different Kubernetes Service access mode and a different process/network topology — which is why this is versioned as a new protocol (V5/CORR-40) rather than silently edited into V4. A previously-reported latency difference between the SSH mechanism and the relay (Phase 15B.3N.2) did **not** survive Phase 15B.3N.2A's properly counterbalanced replication (the delta flipped sign across blocks) and is explicitly not relied upon as part of this decision's rationale.

### Preserved Invariants

Host-based client (traffic generator remains on the Mac host, never moved in-cluster); fresh TCP connections (`connection_mode=fresh`, one inbound relay connection mapped to exactly one outbound connection); the frozen offered schedules and their SHA-256 hashes (including index12's own, `4594f8ce7f813f39588a362360f17e596797e0b8cf579990e08d3bfed33c4dda`); the application itself and its `/work?intensity=` endpoint; Kubernetes Service-level backend selection (kube-proxy remains solely responsible — the relay never load-balances, retries, pools, or caches); `sessionAffinity=None`; replica counts and resource requests/limits; route-integrity failure observability (at least as strong as the SSH mechanism's, now via container/listener primitives instead of process/listener primitives); and CORR-39's entire validity-before-performance classification contract (materiality gate, 10-second evidence windows, window-local delivery attribution, FAILED/CONFIRMED/UNVERIFIED semantics, Axis-A consequence) — none of it weakened or redesigned by this correction.

### Pending Revalidation (Explicitly NOT Resolved Here)

**HPA-260 transport-transition validity**: the 260% HPA target was selected using measurements taken through the pre-V5 SSH-transport path; whether it remains valid, transport-independent evidence now that the measurement path has structurally changed is an open question for a future validation phase (`analysis/transport_epoch_compatibility.json: hpa_260_status = PENDING_TRANSPORT_TRANSITION_REVALIDATION`). Neither retuning nor no-retuning is concluded here. **Pre-V5/V5 calibration cross-epoch compatibility**: whether slots 1–11's already-accepted pre-V5 (SSH-epoch) measurements can coexist scientifically with future V5 (relay-epoch) measurements in the same dynamics-model fitting dataset is likewise unresolved (`cross_epoch_model_fitting_status = PENDING_BRIDGE_VALIDATION`). Slots 1–11 are **not** reclassified, invalidated, or declared cross-epoch-interchangeable by this correction — both outcomes remain open pending a dedicated bridge-validation phase. Index12 attempt3 is **not** authorized by this correction; it remains blocked pending that same future review.

**Applies to phases**: 15B.3N (attempt2, the failure this corrects responds to), 15B.3N.1/15B.3N.2/15B.3N.2A (the investigation and durable evidence underlying this decision), 15B.3N.3 (this correction, freezing Protocol V5), and a not-yet-scheduled future phase that must resolve the two pending revalidation questions above before index12 attempt3 may be authorized.

### Phase 15B.3N.4 Transition-Validation Result (Additive — Resolves Both Pending Questions Above)

Both questions left open above were resolved by a dedicated, strictly non-official validation phase (every run tagged `scientific_role=NON_OFFICIAL_PROTOCOL_TRANSITION_VALIDATION`, `formal_calibration_eligible=false`, `campaign_ledger_eligible=false`, `model_fitting_eligible=false`; the campaign ledger was never written to; index12 attempt3 and index13 were explicitly **not** executed). Full evidence: `persistent-data/transport_validation/protocol_v5_transition/` and `task_outputs/phase_15b3n4_v5_hpa260_revalidation_and_cross_epoch_bridge.md`.

**Gate A — HPA-260 transport-transition validity: `HPA260_V5_REVALIDATED`.** Three V5-transport repeats of the exact frozen historical HPA-tuning workload (`experiments/manifests/hpa_tuning_official_9_trials.yaml`, canonical schedule SHA-256 `4f648411e039f9229cd35a06374e6870dc75eca67e385e67f4f52bb0b5ec55ad`, unmodified) were run. All 3 were structurally valid, the V5 transport was `CONFIRMED` throughout all 3 (zero relay/listener anomalies across 366 combined periodic samples), all 3 met the SLA (100% success in all 3; p95 = 0.1219/0.1936/0.1195s, all ≪ 0.300s), the first scale-up trigger occurred in the `scale_pressure_6rps` phase in all 3 repeats (matching the historical 3/3 finding for the 260% candidate exactly), and no repeat showed oscillation or a scale-direction reversal. Current-replica-seconds were 1725.009 / 1960.0 / 1769.999 (median ≈ 1770) against the historical 260% median of 1680.004 — a modest (≈2.7–16.7%) descriptive increase, not judged disqualifying since it does not violate any of the stated revalidation criteria (SLA, trigger phase, absence of new oscillation/reversal, no evidence of a changed load/CPU relationship). HPA-200%/230% were explicitly not retested, per phase scope. `analysis/transport_epoch_compatibility.json: hpa_260_status.validity_status_for_v5` set to `HPA260_V5_REVALIDATED`.

**Gate B — Pre-V5/V5 cross-epoch calibration compatibility: `CROSS_EPOCH_COMPATIBLE_FOR_MODEL_FITTING`.** 24 non-official bridge runs (4 historically-valid steady-state cells — `dyncal_r1_L1`/`dyncal_r1_L3`/`dyncal_r2_L2`/`dyncal_r8_L1`, campaign_sequence_indices 8/4/5/7, all `HISTORICALLY_VALID_UNDER_ORIGINAL_PROTOCOL` and unreclassified — × 2 transports × 3 repeats, counterbalanced starting-transport order per cell) were run using the frozen steady-state dynamics-calibration timing (180s trial, 60s stabilization, decision timestamps [60,90,120,150]). All 24 runs were structurally valid with `route_integrity=CONFIRMED`; zero exclusions were needed. Decision-point **medians** for server RPS, aggregate CPU, server p95 latency, and memory were comparable between transports in every cell, with no consistent cross-cell directional shift (differences were small and sign-inconsistent across cells, consistent with ordinary repeat-to-repeat variability). Zero pod restarts occurred in any of the 24 runs under either transport. **One genuine, explicitly-flagged residual finding**: 2 of the 12 V5-transport repeats (`dyncal_r1_L3` repeat1 and `dyncal_r8_L1` repeat2) showed a transient right-tail latency/reliability degradation not reproduced in any of the 12 old-transport repeats — elevated server p95 (2.306s and 4.040s respectively, vs. 0.12–0.25s in all other runs of either transport) and depressed decision-point success (79.2% and 93.0% at t=150) — while still classifying `route_integrity=CONFIRMED` (no material zero-success interval ever formed). Root-cause investigation of both runs' full transport-lifecycle periodic samples found the V5 relay container alive and the host listener present for the entire duration of both runs with zero anomalous samples, ruling out a transport-layer failure; `dyncal_r8_L1` repeat2 additionally showed a single transient single-pod readiness drop (1 of 8 pods, at elapsed 122.6s) unrelated to transport, plausibly explaining that run's timeout cluster as ordinary Kubernetes-scheduling/probe variability under host load; `dyncal_r1_L3` repeat1 showed no readiness correlate at all. Because (a) decision-point medians are unaffected in all 4 cells, (b) the transport-level primitives showed zero anomalies in both flagged runs, (c) a plausible non-transport (application/scheduling-level) explanation exists for at least one of the two flagged runs, and (d) this is 2 of 24 total runs — the evidence does not meet the bar for "a consistent material transport bias" required for `CROSS_EPOCH_INCOMPATIBLE_FOR_MODEL_FITTING`. It is explicitly **not** strong enough, at n=3 per cell, to be dismissed as certainly transport-independent either; it is recorded here as an open, flagged item for any future phase collecting additional V5 repeats of `dyncal_r1_L3` or `dyncal_r8_L1` to keep in view, not as a resolved non-issue. `analysis/transport_epoch_compatibility.json: pre_v5_slots_1_11.cross_epoch_model_fitting_status` set to `COMPATIBLE_WITH_V5`.

**Campaign-progression consequence**: `V5_OFFICIAL_CALIBRATION_COLLECTION_MAY_RESUME_AFTER_REVIEW`. Index12 attempt3 remains explicitly **not** authorized by this phase and was not executed; the ledger (`slot12=INVALID_RETRY_REQUIRED`, `valid_attempt_id=None`, `all_attempt_ids=[attempt1,attempt2]`, `next_resumable_index()==12`) and `slot13=NOT_ATTEMPTED` are unchanged. Authorization to execute index12 attempt3 under Protocol V5 is a decision for the review this phase's report is submitted for, not for this phase itself.

**Versioning decision**: Protocol V5 itself is unchanged by this phase (no manifest/prose edits were required beyond documentation of this result) — this is a validation-result record, additive to CORR-40, not a new correction or a V6. See `task_outputs/phase_15b3n4_v5_hpa260_revalidation_and_cross_epoch_bridge.md`.

### Phase 15B.3N.4A — Focused Resolution of the V5 Cross-Epoch Tail-Latency/Reliability Anomaly (Additive)

Phase 15B.3N.4's Gate B classification (`CROSS_EPOCH_COMPATIBLE_FOR_MODEL_FITTING`) carried one explicitly-flagged, non-dismissed residual finding: 2 of 12 V5 bridge repeats (`dyncal_r1_L3` repeat1, `dyncal_r8_L1` repeat2) showed a severe tail-latency/reliability degradation with zero old-transport counterpart in that n=3-per-cell sample. Before accepting that call as final, this focused, strictly non-official follow-up phase (`scientific_role=NON_OFFICIAL_PROTOCOL_TRANSITION_VALIDATION` throughout; the bridge status was provisionally downgraded to `CROSS_EPOCH_BRIDGE_INCONCLUSIVE_PENDING_TAIL_ANOMALY_RESOLUTION` in `analysis/transport_epoch_compatibility.json` for the duration of this phase) expanded exactly the two affected cells — `dyncal_r1_L3` (1 replica, offered_rps 8.0, schedule SHA-256 `8b55d656397d75610ab99cb896c6baf3c0ee44b6fff24a72cf5a3a7c3768fdc2`, recomputed and reconfirmed) and `dyncal_r8_L1` (8 replicas, offered_rps 16.0, schedule SHA-256 `7318a98570eb719d0bec1a955d33b2157a475d2910d87353a14d5a13338f556d`, recomputed and reconfirmed) — from 3 to 8 structurally-valid repeats per transport per cell (20 new runs: 2 cells × 2 transports × 5 additional repeats, counterbalanced execution order, all `route_integrity=CONFIRMED`, zero exclusions). `dyncal_r1_L1` and `dyncal_r2_L2` were correctly not retested — neither showed the anomaly.

**Result**: using the prospectively-declared, frozen SLA criterion (overall-run `client_p95 < 0.300s` AND `success_rate ≥ 0.99` — not decided post hoc), the combined 8-repeat-per-transport-per-cell dataset shows SLA pass/fail incidence **exactly symmetric between transports in both cells**: `dyncal_r1_L3` old 6/8 pass, V5 6/8 pass; `dyncal_r8_L1` old 4/8 pass, V5 4/8 pass (Fisher exact p=1.0 in both cells, reported as descriptive context only, not as the basis for the decision). `dyncal_r1_L3`'s original anomaly did not recur in any of the 10 new repeats (5 old, 5 V5) for that cell — it now reads as an isolated, non-reproducing event. `dyncal_r8_L1` produced further tail episodes under **both** transports, including the single most severe episode of the entire investigation (old-transport repeat13: 5 pods restarted, 41.6% success, near-total readiness collapse). Direct inspection of Kubernetes Events for that run and for the earlier V5 `dyncal_r8_L1` episodes shows the same underlying signature in both transports: `Unhealthy: Readiness/Liveness probe failed ... context deadline exceeded`, followed by `Killing: Container demo-app failed liveness probe, will be restarted`. This is a kubelet probe timing out against the pod's own IP directly — a path that never touches the external load transport (SSH tunnel or Docker relay) at all — under host CPU contention (`dyncal_r8_L1`'s Minikube VM context: `outer_cpu_cores=2.0` for 8 replicas at 16 RPS of CPU-bound synthetic work). This mechanism is transport-independent by construction, and it occurred under **old** transport in this phase's evidence with greater severity (6 total pod-restart events across 2 old-transport episodes) than under V5 (0 pod-restart events across V5's episodes in this phase, though earlier V5 episodes in Phase 15B.3N.4 and this phase did show severe latency/timeout degradation without restarts).

**Conclusion**: the tail/reliability variability observed in `dyncal_r8_L1` is a property of that cell's aggregate CPU demand relative to this specific under-provisioned host (2 CPU cores for 8 replicas), reproducible under both transports with comparable-or-worse severity under the historically-accepted old transport, not a V5-specific defect. `dyncal_r1_L3`'s single original anomaly is now best explained as an isolated, non-reproducing event given 10/10 clean recollections. Neither cell shows a transport-attributable systematic bias. **Gate B classification is updated: `CROSS_EPOCH_COMPATIBLE_FOR_MODEL_FITTING`** (superseding the provisional inconclusive-pending status; the underlying finding is unchanged from Phase 15B.3N.4's own call, now confirmed rather than merely provisional). `analysis/transport_epoch_compatibility.json: pre_v5_slots_1_11.cross_epoch_model_fitting_status` set to `COMPATIBLE_WITH_V5`. `hpa_260_status.validity_status_for_v5` (`HPA260_V5_REVALIDATED`) is unchanged — this phase did not touch HPA-260, HPA-200%, or HPA-230%.

**Campaign-progression consequence (unchanged)**: `V5_OFFICIAL_CALIBRATION_COLLECTION_MAY_RESUME_AFTER_REVIEW`. Index12 attempt3 and index13 were explicitly **not** executed by this phase; the ledger (`slot12=INVALID_RETRY_REQUIRED`, `valid_attempt_id=None`, `all_attempt_ids=[attempt1,attempt2]`, `next_resumable_index()==12`, `slot13=NOT_ATTEMPTED`) is unchanged.

**Versioning decision**: no new protocol version. The V5 transport definition itself is unchanged; this phase found and documented a host-resource-driven, transport-independent readiness/liveness-probe mechanism specific to the heaviest bridge cell, not a defect in the V5 transport implementation. See `task_outputs/phase_15b3n4a_cross_epoch_tail_anomaly_resolution.md` for full evidence, including the raw Kubernetes Events, transport-lifecycle periodic samples, and per-run decision-point tables for all 20 new runs plus the combined 16-run-per-cell analysis.

**Causal-scope clarification (Phase 15B.3O, additive)**: the expanded bridge evidence above supports the narrower claim that *no V5-relay-specific tail/reliability mechanism was demonstrated* — it does not prove that every anomalous repeat across both this phase and Phase 15B.3N.4 shares the exact same host-CPU-contention cause. Kubernetes readiness/liveness-probe timeouts under host resource pressure are directly documented, via Kubernetes Events, for the severe `dyncal_r8_L1` cases under both transports (§ "Phase 15B.3N.4A" above); this is a demonstrated, transport-independent mechanism for those specific cases. The original, non-recurring `dyncal_r1_L3` V5 anomaly (Phase 15B.3N.4's repeat1) was never directly tied to a host-CPU-contention signature in its own evidence (its Kubernetes Events showed no probe-timeout/restart pattern) — its cause remains formally unresolved, and this document does not claim otherwise. Future analysis should describe isolated, non-reproducing tail events as "cause unresolved" or "consistent with, but not proven to be caused by, resource contention," and reserve "host CPU contention" as a claim only for cases where the supporting Kubernetes-Events/host-resource evidence for that specific attempt actually establishes it. This clarification does **not** change `HPA260_V5_REVALIDATED` or `CROSS_EPOCH_COMPATIBLE_FOR_MODEL_FITTING`, and does not warrant a new correction or protocol version — it is a wording-precision refinement of Phase 15B.3N.4A's own causal language.

## CORR-41 — Kubernetes-Dataplane Workload-Ingress Evidence (Phase 15B.3O.1)

### Problem

CORR-39's route-integrity classifier has exactly two decisive positive evidence sources — an HTTP response attributable by dispatch time, and a raw per-pod application request-counter increment — and both require the application itself to be alive, scraped, and counting. Campaign_sequence_index=12 attempt3 (V5/CORR-40 transport, `fit_dyncal_r8_L3_repeat1_attempt3_20260919T032529Z`) exposed the resulting methodology gap directly and durably: the relay container and host listener remained healthy throughout, the generator kept dispatching, but application readiness collapsed, Prometheus scraping and HTTP successes nearly disappeared, and per-pod counter evidence became unavailable — so all 14 evidence windows across both material zero-success intervals ( [1.0, 68.0], [101.0, 166.0] ) were correctly, but unhelpfully, classified `UNVERIFIED`, forcing `execution_structural_validity=INVALID` for a run in which the harness/Kubernetes workload-ingress path may well have remained healthy. This creates an undesirable coupling: the application must remain observable/responding merely to prove the *experimental route* is structurally sound, even though application/backend collapse under load is itself one of this campaign's legitimate scientific outcomes.

### Evidence

A dedicated, strictly non-official investigation and validation phase (`persistent-data/transport_validation/protocol_v6_route_evidence/NON_OFFICIAL_ROUTE_EVIDENCE_VALIDATION_corr41_nodeport_dataplane_20260919T054000Z.json`; no official traffic, no ledger writes, no attempt4/index13 execution) audited this Minikube cluster's live kube-proxy dataplane (iptables backed by the `nf_tables` kernel subsystem, `mode: ""` i.e. default/iptables in the kube-proxy ConfigMap) and found two mutually-exclusive, Service-specific, endpoint-independent counters for `autoscaler-app/demo-app-loadtest`:

- **Endpoint-backed regime**: one `nat`-table `KUBE-SEP-<hash> ... -j DNAT --to-destination <podIP>:8080` chain per Ready endpoint, each uniquely commented `"autoscaler-app/demo-app-loadtest:http"`. Empirically verified to increment exactly with real traffic (5 curls -> 5 packets/300 bytes) and to correctly distribute and sum across multiple Ready endpoints (2 endpoints, 10 curls -> 7+3=10 packets).
- **No-endpoints regime**: a single `filter`-table `KUBE-EXTERNAL-SERVICES ... --dport 31891 -j REJECT --reject-with icmp-port-unreachable` rule, commented `"autoscaler-app/demo-app-loadtest:http has no endpoints"`, programmed by kube-proxy ONLY while the Service has zero Ready endpoints. Empirically verified live: scaling `demo-app` to 0 replicas made every endpoint-backed rule disappear entirely (not merely stop incrementing) while this REJECT rule appeared and incremented exactly with test traffic sent to the NodePort (1 curl -> 1/60; 4 more -> 5/300 total), even though every such request was refused at the kernel/dataplane level with no application involvement whatsoever.

Also empirically established: (a) rule *identity* (the exact `KUBE-SEP-<hash>` chain name) is **not** stable across pod recreation — a scale-to-0/scale-to-1 cycle changed `KUBE-SEP-ENIF66D2G3466LGQ` to a new `KUBE-SEP-OGXRNUUKR756PIKN` for the same Service, requiring reset-safe, identity-tracked window attribution, never a raw cross-sample delta; (b) sampling overhead (`docker exec minikube iptables-save -t nat -c`) measured 0.05–0.14s per call, safely supporting a 3-second observer cadence (faster than the existing 10-second evidence window) on the 2-core Minikube VM; (c) a contamination audit of `k8s/app/service.yaml`/`service-monitor.yaml`/`service-loadtest.yaml` (pre-existing design from Phase 11C.4B/CORR-21 and Phase 11C.4C/CORR-22) confirms Prometheus/Grafana/metrics-server scrape exclusively via the separate `demo-app` ClusterIP Service, never `demo-app-loadtest`'s NodePort — the dedicated load-generator Service and its dataplane counters are structurally isolated from all other in-cluster traffic.

### Decision

Adopt the Kubernetes-dataplane NodePort ingress counter as a THIRD decisive positive route-evidence source, alongside CORR-39's existing two, frozen as Protocol V6 (`experiments/manifests/rl_dynamics_calibration_v6.yaml`, `docs/rl_dynamics_calibration_protocol_v6.md`). A route-integrity evidence window is `CONFIRMED` when no direct harness-path failure is proven AND at least one of the three sources (HTTP response, application counter, **or** NodePort dataplane ingress) shows a positive, window-local, reset-safe, non-duplicated delta — implemented in `classify_route_integrity_window()`'s new `nodeport_dataplane_ingress_evidence` parameter (default `None`, exactly reproducing pre-CORR-41 behavior when omitted) and `compute_dataplane_ingress_evidence_windows()` (`scripts/dynamics_calibration_lib.py`). Crucially, dataplane ingress evidence is available and decisive in **both** regimes above — including `no_endpoints`, meaning a window can now be `CONFIRMED` (route healthy) simultaneously with a documented application/backend outage, instead of forcing `UNVERIFIED` for both questions at once. CORR-39's FAILED precedence (`relay_container_running=false` or `host_listener_present=false`) is entirely unchanged and remains decisive over any positive dataplane evidence.

### Scientific Impact

This does **not** change the V5/CORR-40 workload-transport mechanism (`transport_protocol_version` remains 5 — still the Docker-network raw-TCP relay to the `demo-app-loadtest` NodePort) — only the route-integrity evidence/classification layer changes (`route_integrity_protocol_version` 4 → 6). It separates two previously-conflated scientific questions: "did workload traffic reach the Kubernetes Service/NodePort dataplane" (now independently provable) from "did the application successfully process it" (still governed entirely by existing SLA/performance/restart evidence, untouched by this correction). Timeout rate, p95 latency, zero-Ready-endpoints, restarts, scrape failures, and SLA failure must never, by themselves, force `UNVERIFIED` when dataplane ingress evidence directly proves workload traffic continued reaching the workload path — that coupling is exactly what this correction removes.

### Preserved Invariants

The 102-slot global campaign sequence and its SHA-256; all frozen schedule hashes; CPU normalization; the SLA definition; the RL reward/observation/action contracts; `HPA260_V5_REVALIDATED` and `CROSS_EPOCH_COMPATIBLE_FOR_MODEL_FITTING` (HPA-260 was not rerun, the pre-V5/V5 bridge was not rerun — workload transport, application, resources, and schedules are all unchanged by this correction); the V5 transport mechanism itself (socat, NodePort, connection mode, client location — none changed); CORR-39's materiality gate, 10-second evidence windows, and FAILED-precedence semantics.

### Attempt3 Non-Retroactivity

Campaign_sequence_index=12 attempt3 did **not** collect the NodePort dataplane-ingress counter timeline this correction defines — Protocol V6 was frozen strictly after, and as a direct consequence of, reviewing attempt3's own evidence, not before it. Per sec.18 of this phase's own governing instructions, what those counters "would have shown" during attempt3's two material intervals is never inferred, estimated, or backfilled. Attempt3 remains, permanently: `RL_DYNAMICS_CALIBRATION_SEQUENCE_INDEX_12_ATTEMPT3_STRUCTURALLY_INVALID`, `ROUTE_INTEGRITY_UNVERIFIED`. `slot12.state=INVALID_RETRY_REQUIRED`, `slot12.valid_attempt_id=None`, `slot12.all_attempt_ids=[attempt1, attempt2, attempt3]`, `next_resumable_index()==12`, `slot13=NOT_ATTEMPTED` — all unchanged by this phase. CORR-41/Protocol V6 applies prospectively only, starting with a future, separately-authorized attempt4.

**Applies to phases**: 15B.3M.1–15B.3M.5 (CORR-39, the classification contract this correction extends rather than replaces), 15B.3N.3 (CORR-40, the transport this correction does not touch), 15B.3O.1 (this correction, freezing Protocol V6). Index12 attempt4 authorization is a decision for the review this phase's report is submitted for, not for this phase itself.


## CORR-42 — Temporal Dataplane-Evidence Acquisition Coverage (Phase 15B.3P.1)

### Problem

Campaign_sequence_index=12 attempt4 (V6/CORR-41, `fit_dyncal_r8_L3_repeat1_attempt4_20260919T091715Z`) exposed a genuine, structural observability gap in Protocol V6's dataplane-ingress *acquisition* schedule, distinct from anything CORR-41's classification rules got wrong. Of 18 route-integrity evidence windows, 16 were `CONFIRMED` exclusively via `nodeport_dataplane_ingress_evidence` (HTTP and application-counter evidence were false in every material window — the same application collapse seen in attempt3), but exactly 2 windows — `[115.0,125.0)` and `[175.0,179.0)` — were `UNVERIFIED`, forcing overall `route_integrity=UNVERIFIED` and `execution_structural_validity=INVALID` under CORR-39's unmodified attempt-level "every window must be CONFIRMED" aggregation rule.

### Evidence

A non-official investigation (Phase 15B.3P.1; no official traffic, no sequence-index-12 attempt5, no index13, no ledger mutation) reconstructed both windows directly from attempt4's preserved raw `cluster/loadtest_dataplane_ingress_timeline.json` (52 periodic samples) and found BOTH windows' `UNVERIFIED` status reflects a measurement-locality limitation, not evidence of route/delivery failure — packet counts are continuous and monotonically rising on both sides of each window in the raw data:

- **`[115.0,125.0)`**: exactly two interior samples (118.02s, 121.44s). The only interior-to-interior candidate pair is same-regime (`endpoint_backed`) but different `rule_identity` (8 SEPs → 7 SEPs, one endpoint dropped between samples) — correctly rejected as a RESET by CORR-41's existing, unmodified rules. There is no other candidate pair available inside the window.
- **`[175.0,179.0)`**: a 4-second trailing remainder window (the material interval's length, 74s, is not a multiple of 10: `7×10 + 4`). Exactly ONE interior sample (176.07s) exists — structurally insufficient to form the two-sample pair the classifier requires, regardless of dataplane health.

A formal, general (not attempt4-timestamp-specific) synthetic timing analysis, modeling the sampler's REAL (not nominal) inter-sample cadence — empirically 3.31s–4.31s, mean 3.60s, measured directly from attempt4's own 52 samples, versus the nominal `DATAPLANE_INGRESS_SAMPLING_INTERVAL_SECONDS = 3.0` constant's own justification, which assumed idealized zero-jitter timing — confirmed this is a **structural** defect, not an attempt4-specific coincidence: at realistic cadences, a meaningful fraction of possible sampler/window phase alignments (23%–68%, depending on realized cadence) leave a full 10-second window with only 2 interior samples instead of 3+, and trailing windows narrower than roughly one cadence period are sometimes or always insufficient depending on exact width. `compute_dataplane_ingress_evidence_windows()` was confirmed, by direct code reading, to be completely agnostic to how or when its input samples were collected — the defect is entirely in the acquisition SCHEDULE, not the classification semantics.

### Decision

Adopt three combined, acquisition-only corrections, frozen as Protocol V7 (`experiments/manifests/rl_dynamics_calibration_v7.yaml`, `docs/rl_dynamics_calibration_protocol_v7.md`):

1. **Event-triggered reactive re-sampling**: `should_trigger_immediate_resample()` (`scripts/dynamics_calibration_lib.py`) is a pure function comparing consecutive samples' `regime`/`rule_identity`; the orchestrator (`scripts/run_dynamics_calibration_slot.py`) takes one additional sample `DATAPLANE_INGRESS_TRANSITION_RESAMPLE_DELAY_SECONDS` (1.0s) after any detected transition, maximizing the chance a legitimate, same-identity confirming pair still forms under the new baseline soon after a reset — never fabricating a delta across the transition itself.
2. **End-of-run anchor sample**: one additional dataplane sample is taken the instant the traffic-generator subprocess exits, before the periodic sampler is stopped — directly targeting trailing windows whose material interval ends near the end of a run (the `[175,179)` shape). This is only possible because generator-exit is a harness-controlled, real-time-knowable event; the alternative literal reading of "sample at every future route-window boundary" was evaluated and rejected, because route-window boundaries are discovered only POST-HOC from `traffic_summary.csv` after traffic ends (`find_all_zero_success_runs()` / `partition_into_evidence_windows()`) — the harness cannot know them live.
3. **A modest baseline cadence reduction**: `DATAPLANE_INGRESS_SAMPLING_INTERVAL_SECONDS_V7 = 2.0` (from 3.0), evaluated as a fallback-only lever per this phase's own instructions, quantitatively reducing (not eliminating) the fraction of phase-space producing sparse windows. No jump to 1s was made without further justification.

All three changes only ever ADD samples to the same list `compute_dataplane_ingress_evidence_windows()` already consumes — that function, `classify_route_integrity_window()`, and `classify_route_integrity()` are byte-for-byte unchanged. This is why `route_integrity_protocol_version` stays 6: V7 changes only evidence ACQUISITION, never evidence CLASSIFICATION. A new, independent `dataplane_acquisition_protocol_version` field (1 → 2) tracks the acquisition-only change in `build_protocol_provenance()`.

### Scientific Impact

This does not weaken validity: no positive evidence is borrowed from neighboring windows, no healthy-relay-implies-NodePort-delivery assumption is introduced, no counter increment is smeared across a reset, no cross-identity interval is accepted, no short window is automatically marked `CONFIRMED`, FAILED precedence is completely unchanged, and the full-temporal-coverage requirement (every window must be individually `CONFIRMED`) is not reduced. The correction only improves the observer's chance of collecting genuine, window-local, reset-safe evidence that was always physically available but was not necessarily sampled at the right instant. It separates "the acquisition schedule under-sampled this window" from "the dataplane failed to deliver traffic" — a distinction CORR-41 itself established for the HTTP/counter-vs-dataplane question, now extended to the acquisition layer.

### Preserved Invariants

Everything CORR-39/CORR-40/CORR-41 preserved (see their own entries) remains preserved: the 102-slot global campaign sequence and its SHA-256; all frozen schedule hashes; CPU normalization; the SLA definition; the RL reward/observation/action contracts; `HPA260_V5_REVALIDATED` and `CROSS_EPOCH_COMPATIBLE_FOR_MODEL_FITTING`; the V5 transport mechanism; CORR-39's materiality gate, 10-second evidence windows, and FAILED-precedence semantics; CORR-41's three-source evidence rule and its `route_integrity_protocol_version=6` classification contract, entirely unchanged by this correction.

### Attempt 1–4 Non-Retroactivity

Attempts 1–4 did not collect samples under this acquisition schedule — Protocol V7 was frozen strictly after, and as a direct consequence of, reviewing attempt4's own raw evidence, not before it. What the corrected schedule "would have shown" during attempt4's two UNVERIFIED windows is never inferred, estimated, or backfilled into attempt4's stored, official result. Attempt4 remains, permanently: `SEQUENCE_INDEX_12_ATTEMPT4_INVALID`, `route_integrity=UNVERIFIED`, Axis A `INVALID`. `slot12.state=INVALID_RETRY_REQUIRED`, `slot12.valid_attempt_id=None`, `slot12.all_attempt_ids=[attempt1, attempt2, attempt3, attempt4]`, `next_resumable_index()==12`, `slot13=NOT_ATTEMPTED` — all unchanged by this phase. A separate, clearly labeled `NON_OFFICIAL_COUNTERFACTUAL_ONLY` offline/synthetic replay using attempt4's preserved raw timing structure is permitted to assess whether the corrected schedule would likely have resolved its two windows, but this never changes attempt4's official stored classification. CORR-42/Protocol V7 applies prospectively only, starting with a future, separately-authorized attempt5.

**Applies to phases**: 15B.3O.1 (CORR-41, the classification contract this correction extends rather than replaces), 15B.3P (attempt4, the evidence this correction is a direct response to), 15B.3P.1 (this correction, freezing Protocol V7). Index12 attempt5 authorization is a decision for the review this phase's report is submitted for, not for this phase itself.


## CORR-43 — Deterministic Short-Trailing-Window Evidence Partitioning (Phase 15B.3P.2)

### Problem

Phase 15B.3P.1's own report was explicit that CORR-42/Protocol V7's fix (reduced baseline cadence, reactive re-sampling, an end-of-run anchor sample) was a **probabilistic** improvement, not a deterministic guarantee, and that "a future attempt5 executed under V7 could still, in principle, exhibit an isolated short-trailing-window `UNVERIFIED` result." A deeper, deterministic (exhaustive fine-grid enumeration, not Monte-Carlo sampling) re-analysis of the same timing geometry confirms this precisely: for a periodic sampler with worst-case realized period `T`, a window of width `r` is guaranteed *exactly* `floor(r/T)` interior samples at its worst possible phase offset. A guarantee of `>=2` samples therefore strictly requires `r >= 2T`; no finite periodic cadence can guarantee even one interior sample once the window width can be adversarially placed below one full cadence period. This is a structural property of the **window-partitioning rule** (`partition_into_evidence_windows()`, frozen since Phase 15B.3M.4/CORR-39), which can construct a trailing window of arbitrarily small width whenever a material zero-success interval's length is not an exact multiple of the 10-second evidence-window size — not a property of the sampling cadence, so no amount of further cadence tuning inside V7's own acquisition layer could ever fully close it for the narrowest possible remainders.

### Evidence

A non-official investigation (Phase 15B.3P.2; no official traffic, no sequence-index-12 attempt5, no index13, no ledger mutation) re-audited attempt4's `[175.0,179.0)` window directly from its raw `cluster/loadtest_dataplane_ingress_timeline.json` (unchanged since Phase 15B.3P.1) and confirmed it is a 4-second trailing remainder window (the material interval `[105,179]` has length 74s = 7×10+4). An exhaustive (not random) phase-offset sweep proved the exact closed-form `floor(r/T)` worst-case formula, and derived two thresholds from a conservative, measured-with-margin worst-case realized dataplane-sampler cadence `T=3.0s` (V7's 2.0s nominal sleep plus command-duration/scheduling jitter — Phase 15B.3P.1 measured a live mean of 0.191s and worst-observed 0.306s over 7 real invocations; Phase 15B.3P.2 re-measured with a larger live sample during its own U1–U6 non-official validation, reconfirming the bound): `2T=6.0s` guarantees `>=2` interior samples for any phase offset (sufficient for a confirming pair absent any identity/regime transition); `3T=9.0s` guarantees `>=3` interior samples (resilient to a single identity/regime transition landing anywhere inside the window, independent of whether CORR-42's own reactive resample completes before the window's own end). Candidates for a live boundary-anchored sample at exact window/material-interval boundaries (Candidate B) and live event-derived material-boundary anchors (Candidate C) were both evaluated and rejected: material-interval and interior-window boundaries remain discoverable only post-hoc (`find_all_zero_success_runs()` analyzing `traffic_summary.csv` after traffic ends), the same architectural constraint CORR-42 already established — no new real-time-knowable trigger exists beyond what CORR-42's generator-exit anchor already exploits. A further cadence increase (Candidate D) was evaluated and rejected as insufficient in principle: since the tail width can approach zero, no finite practical cadence can deterministically guarantee even one interior sample for an arbitrarily short remainder.

### Decision

Adopt window-partitioning repartitioning (Candidate A) as the minimal, deterministic correction, frozen as Protocol V8 (`experiments/manifests/rl_dynamics_calibration_v8.yaml`, `docs/rl_dynamics_calibration_protocol_v8.md`). `partition_into_evidence_windows()` gains an optional `min_window_seconds` parameter (`None` by default, reproducing pre-CORR-43 behavior byte-for-byte): when a trailing remainder window's width would be strictly less than `min_window_seconds` (set to `MIN_EVIDENCE_WINDOW_SECONDS_V8=9.0`, the stricter of the two derived thresholds) AND at least one other window already exists, the remainder is merged into the immediately preceding window instead of being constructed as its own short, potentially unconfirmable window. This never applies when the interval length is an exact multiple of the window size, when the remainder is already `>=min_window_seconds` (left standalone to preserve temporal resolution), or when the whole material interval is itself shorter than one window (the sole irreducible case, honestly retained — architecturally rare at official scale, since the existing 25% materiality gate combined with realistic traffic durations implies a qualifying material interval is at least 45 seconds long). `classify_route_integrity_window()`, `classify_route_integrity()`, `compute_dataplane_ingress_evidence_windows()`, and CORR-42's dataplane-acquisition schedule are all byte-for-byte unchanged — this correction only changes which window boundaries are fed into them, never how those windows are classified or how samples are acquired within them. Coverage invariants (`union(windows)==[interval_start,interval_end)` exactly, no gap, no duplicated duration, chronological ordering) are proven for both the default and merged case.

### Scientific Impact

This does **not** change the V7/CORR-42 acquisition schedule (`dataplane_acquisition_protocol_version` remains 2) nor the V6/CORR-41 classification rules (`route_integrity_protocol_version` remains 6) — only the window-PARTITIONING layer changes, tracked by a new, independent `window_partitioning_protocol_version` field (1→2). It closes the deterministic gap CORR-42 left open: rather than merely reducing the probability that a short trailing window lacks sufficient evidence, it prevents such a pathologically short window from ever being constructed in the first place (for any material interval at or above the 45-second realistic official minimum), while still covering every second of the material interval exactly once. The full-temporal-coverage requirement, FAILED precedence, materiality gate, and reset/identity-rejection rules are entirely untouched — no validity standard is weakened to make a future attempt pass.

### Preserved Invariants

Everything CORR-39/CORR-40/CORR-41/CORR-42 preserved (see their own entries) remains preserved: the 102-slot global campaign sequence and its SHA-256; all frozen schedule hashes; CPU normalization; the SLA definition; the RL reward/observation/action contracts; `HPA260_V5_REVALIDATED` and `CROSS_EPOCH_COMPATIBLE_FOR_MODEL_FITTING`; the V5 transport mechanism; CORR-39's materiality gate and FAILED-precedence semantics; CORR-41's three-source evidence rule and its `route_integrity_protocol_version=6`; CORR-42's dataplane-acquisition schedule and its `dataplane_acquisition_protocol_version=2`, entirely unchanged by this correction.

### Attempt 1–4 Non-Retroactivity

Attempt4 was not re-partitioned or re-classified under this rule — Protocol V8 was frozen strictly after, and as a direct consequence of, a deterministic re-analysis of attempt4's already-collected evidence, not before it. What the merged partitioning "would have shown" for attempt4's own `[175,179)` window is assessed only via a separate, clearly labeled `NON_OFFICIAL_COUNTERFACTUAL_ONLY` analysis (this phase's own report) that never changes attempt4's official stored classification or its already-persisted `evidence_windows`. Attempt4 remains, permanently: `SEQUENCE_INDEX_12_ATTEMPT4_INVALID`, `route_integrity=UNVERIFIED`, Axis A `INVALID`. `slot12.state=INVALID_RETRY_REQUIRED`, `slot12.valid_attempt_id=None`, `slot12.all_attempt_ids=[attempt1, attempt2, attempt3, attempt4]`, `next_resumable_index()==12`, `slot13=NOT_ATTEMPTED` — all unchanged by this phase. CORR-43/Protocol V8 applies prospectively only, starting with a future, separately-authorized attempt5.

### Phase 15B.3P.3 Audit Amendment (Pre-First-Use, Amended In Place — Not A New Correction)

V8 had not yet been used by any official attempt when Phase 15B.3P.3 audited its own deterministic-guarantee claim, so this amendment is made in place, per this project's own standing pre-first-use convention, rather than as a new CORR entry. **Finding**: the `floor(r/T)` derivation is a sound mathematical implication (if every real gap `<= T`, a window of width `>= k·T` is guaranteed `>= k` interior samples), but `T=3.0s` was never a hard, OS/Docker-enforced real-time bound — the production sampler sleeps a fixed 2.0s *after* each sample completes (not against an absolute deadline), and each `docker exec` carries only a 5-second timeout, so a slow-but-successful command could in principle exceed 3.0s. This project does not claim "a guaranteed maximum inter-sample gap" as a real-time property; `T=3.0s` is retained as an empirically well-supported (not hard-guaranteed) value, now paired with explicit per-sample gap recording (`actual_gap_seconds_from_previous_sample`, `acquisition_gap_exceeded_assumed_bound`) for post-hoc audit — a large gap can only ever reduce available confirming pairs (honest `UNVERIFIED`), never fabricate a `CONFIRMED`. A related, previously undocumented finding: a valid confirming pair requires only that *some* consecutive sample pair share an identity, regardless of the total number of distinct identities or transitions in a window — a window churning on every single sample (e.g. `A→B→C→D`) yields zero confirming pairs regardless of sample count, an inherent limit of interval/delta-based evidence, honestly reflected as `UNVERIFIED`. Implementation hardening (no version bump): a previously-uncaught `subprocess.TimeoutExpired` (capable of silently killing the entire dataplane sampler thread) is now caught as an ordinary observer failure in `sample_dataplane_ingress_rules()`. No declared acquisition-schedule parameter, window-partitioning threshold, or classification rule changed — `route_integrity_protocol_version`, `dataplane_acquisition_protocol_version`, and `window_partitioning_protocol_version` are all unchanged.

**Applies to phases**: 15B.3O.1 (CORR-41, the classification contract this correction does not touch), 15B.3P.1 (CORR-42, the acquisition schedule this correction extends rather than replaces), 15B.3P (attempt4, the evidence re-analyzed here), 15B.3P.2 (this correction, freezing Protocol V8), 15B.3P.3 (this amendment). Index12 attempt5 authorization is a decision for the review this phase's report is submitted for, not for this phase itself.

## Implementation Note — Replica-Transition Execution Path Added (Phase 15B.5B, Not A Correction)

**Scientific design unchanged; execution implementation added.** This is an engineering-log entry, not a CORR: no scientific parameter, threshold, timing value, or classification rule described anywhere in this document or in `experiments/manifests/rl_dynamics_calibration_v1.yaml`'s `replica_transitions:` block was altered by this phase.

**Problem (found in Phase 15B.5A)**: the `replica_transition` `run_kind` (16 of the 102 frozen global-sequence slots) had a fully specified scientific design — `pairs`, `directional_types`, `workload_by_pair`, `timing` (`pre_run_replica_stabilization_seconds=60`, `pre_transition_traffic_steady_seconds=60`, `trigger_at_traffic_elapsed_seconds=60`, `post_transition_observation_seconds=180`, `total_traffic_seconds=240`, `total_seconds_per_run=300`), and `measured_fields` — but zero execution implementation. The single existing harness flow (`scripts/run_dynamics_calibration_slot.py`) supported only a one-time, fixed-replica-count steady-state run keyed on `slot["replicas"]`, a key that resolved `replica_transition` slots never carry (`replicas_before`/`replicas_after` instead); attempting to execute one live would delete the HPA and then crash with `KeyError: 'replicas'` before any scaling or traffic occurred.

**Implementation (Phase 15B.5B)**: a new, self-contained `execute_replica_transition_slot()` execution path was added — kept entirely separate from the existing steady-state body of `main()` so the flow already exercised by 29 official attempts is provably untouched. It implements the frozen timing plan exactly (initial scale to `replicas_before`, 60s no-traffic stabilization, traffic starts, precise monotonic-clock-anchored trigger at the frozen `trigger_at_traffic_elapsed_seconds` offset while traffic continues uninterrupted, 180s post-transition observation, then stop), reusing the existing V5 transport/dataplane-acquisition/route-integrity pure functions byte-for-byte (`route_integrity_protocol_version`, `dataplane_acquisition_protocol_version`, `window_partitioning_protocol_version`, `active_calibration_protocol_version`, `active_correction_reference` all unchanged). Supports both scale-up and scale-down directions generically (validated live for 5→6 and read-only-resolved for the 2→1 directional slot).

Of the manifest's seven `measured_fields`, three (`desired_to_current_lag_seconds`, `desired_to_ready_lag_seconds`, `desired_to_available_lag_seconds`) have an objectively-defined start/end (the trigger timestamp; a Kubernetes-API-observed replica count crossing the transition's own target) and are computed from a new high-resolution `replica_transition_timeline.csv` trajectory artifact. The remaining four (`cpu_settling_time_seconds`, `latency_settling_time_seconds`, `in_progress_settling_time_seconds`, `request_rate_realization_lag_seconds`) have no settling threshold/band/window defined anywhere in this project's manifest or protocol documentation — rather than inventing one inside this implementation task, they are explicitly reported as `NOT_FORMALLY_OPERATIONALIZED`, with their full raw time series preserved unabridged in the attempt's own artifacts pending a future, separately-frozen definition.

Live-cluster validation (a single `NON_OFFICIAL` smoke run, `attempt_id=NONOFFICIAL_transrep_5-6_5to6_repeat1_smoke_20260920T105441Z`, artifacts under `persistent-data/transport_validation/replica_transition_smoke/`, official campaign ledger never read or written) confirmed the mechanism end-to-end: the trigger fired 3ms from its planned monotonic target, the Deployment's `current_replicas` genuinely reached 6 at `elapsed_from_trigger=2.912s` (matching the computed `desired_to_current_lag_seconds`), the new pod's identity appeared in the same trajectory sample, and `desired_to_ready_lag_seconds`/`desired_to_available_lag_seconds` correctly resolved to `null` (never fabricated) because this smoke run's own application-level instability meant the target was never actually observed to be reached within the run.

**Official use**: `campaign_sequence_index=30` (and every other `replica_transition` slot) remains `NOT_ATTEMPTED` — this phase performed implementation and non-official validation only, per its own governing instructions; it did not execute, and was explicitly forbidden from executing, any official replica-transition attempt.

**Applies to phases**: 15B.5A (the implementation-gap finding this phase resolves), 15B.5B (this implementation).

## CORR-44 — Replica-Transition Materiality/Scored-Window Operationalization (Phase 15B.5C)

**Documentation-gap disclosure (found during Phase 15B.5E)**: this entry was decided, implemented, and reported as adopted by Phase 15B.5C's own report (`task_outputs/phase_15b5c_replica_transition_materiality_freeze.md`) — including live code (`scripts/dynamics_calibration_lib.py`'s `REPLICA_TRANSITION_MATERIALITY_FRACTION`/`replica_transition_route_scored_window()`/`replica_transition_materiality_threshold_seconds()`, `build_protocol_provenance()`'s `active_correction_reference` bump, and the persisted `scored_window_correction_reference: "CORR-44"` field in every subsequent attempt's artifacts, including official index30's own) — but the corresponding formal doc entry itself was never actually appended to this file at that time, a genuine oversight found while preparing this file's next entry (CORR-45). It is added retroactively now, verbatim to what Phase 15B.5C's own report already describes and what the code has already been doing since that phase, so that this document's own correction sequence matches the code's actual, already-live behavior. No behavior changes as a result of adding this entry — it documents a decision and implementation that were already in force.

### Problem

Phase 15B.5B's implementation of official `replica_transition` execution needed a route-integrity materiality/scored-window rule for its own 240-second traffic runs, but none had ever been frozen — steady-state's own `[60,180)`/45-seconds-of-180 rule had no established transition-run analogue. Phase 15B.5B adopted `[0, total_traffic_seconds)` as an explicitly disclosed, non-frozen placeholder (`scored_window_frozen: false` in every artifact it produced) pending deliberate methodology review.

### Evidence

Direct inspection of `scripts/dynamics_calibration_lib.py`'s `DEFAULT_ROUTE_INTEGRITY_MATERIALITY_FRACTION = 0.25` and `is_interval_material()`'s own `fraction = duration / traffic_duration_seconds` arithmetic, together with this document's own CORR-39 sec.7 founding prose ("a materiality criterion... covering ≥25% of total traffic duration"), proves the original steady-state 45-second threshold was always a **duration-relative fraction** (`0.25 × 180s`), never an absolute, duration-independent invariant. Applying the identical, unchanged 25% fraction to a 240-second transition run yields `0.25 × 240s = 60.0` seconds. Separately, the manifest's `pre_transition_traffic_steady_seconds` is a genuine, load-bearing traffic period (traffic is already running), never named "warmup" anywhere — unlike steady-state's own explicit `warmup_window_seconds` — supporting scoring the transition's full `[0, total_traffic_seconds)` rather than excluding its first 60 seconds by analogy with steady-state's own discard-only warmup.

### Decision

Freeze, for `run_kind=replica_transition` only:

```text
replica_transition_route_scored_window(total_traffic_seconds) = (0.0, total_traffic_seconds)
REPLICA_TRANSITION_MATERIALITY_FRACTION = DEFAULT_ROUTE_INTEGRITY_MATERIALITY_FRACTION = 0.25 (the SAME fraction steady-state already uses)
replica_transition_materiality_threshold_seconds(240.0) = 60.0 seconds  (for the frozen 240s official transition schedules)
```

`is_interval_material()` itself required zero code changes (it was already fully generic in both `traffic_duration_seconds` and `scored_window`) — this correction only specifies, for the first time, which values of its own already-generic parameters apply to a `run_kind` that had never been given any before.

### Scientific Impact

Steady-state's own `[60,180)`/45-seconds-of-180 behavior (slots12–29, every `fit_*`/`val_*` attempt) is completely untouched — no historical result is reclassified. No V8 partitioning mechanic, route-classification rule, or same-identity evidence rule changed. `route_integrity_protocol_version` remains 6.

### Preserved Invariants

Everything CORR-39 through CORR-43 preserved remains preserved (see their own entries): the 102-slot global campaign sequence and its SHA-256; all frozen schedule hashes; the two-axis validity model; the V5 transport mechanism; the 25%-fraction materiality concept itself (now explicitly reused, not reinvented, for transitions); CORR-41's three-source evidence rule; CORR-42's dataplane-acquisition schedule; CORR-43's window-partitioning merge rule.

### Non-Retroactivity

`replica_transition` had zero official attempts when this correction was frozen (Phase 15B.5C) — there is no historical official transition result to reclassify. The Phase 15B.5B non-official smoke run's own artifacts were re-analyzed (never mutated) under this frozen rule and reproduced their already-persisted result exactly, with zero behavior change.

**Protocol-version decision**: no `protocol_version` field bump. By direct analogy to this project's own established pre-first-use-hardening convention (CORR-39 sec.8/9/10), this correction specifies parameter *values* for an already-generic, unchanged mechanism, applied to a `run_kind` that had never used it officially — not a post-data reversal of an already-exercised mechanism's actual behavior. `active_correction_reference` is updated globally (via `build_protocol_provenance()`, applied to every future attempt regardless of `run_kind`, exactly as every prior CORR bump has been) to `"CORR-44"` (`prior_correction_reference: "CORR-43"`).

**Applies to phases**: 15B.5B (the disclosed placeholder this correction replaces), 15B.5C (this correction, originally decided/implemented here), 15B.5D (official index30, the first attempt to carry `active_correction_reference=CORR-44` and the persisted `scored_window_correction_reference: "CORR-44"` field), 15B.5E (the phase that found and fixed this entry's own missing documentation). See `task_outputs/phase_15b5c_replica_transition_materiality_freeze.md`.

## CORR-45 — Replica-Transition Scale-Down Lag Target-State Operationalization (Phase 15B.5E)

### Problem

`compute_transition_replica_lags()` (Phase 15B.5B) computed `desired_to_ready_lag_seconds`/`desired_to_available_lag_seconds` for a scale-down transition by checking `ready_replicas`/`available_replicas` against the target replica count **completely independently** of whether `current_replicas` had itself already reached that target. For a 2→1 scale-down, this meant an unrelated readiness or availability loss on a pod that had **not yet been removed** (`current_replicas` still 2, one of the two surviving pods merely becoming unready) would satisfy `ready_replicas <= 1` and be misreported as "the 2→1 scale-down has completed and its surviving pod is Ready" — an artificially short, **false** lag for a transition that had not actually occurred yet. Direct reproduction of the original algorithm against a concrete worked example (2 pods healthy → one becomes unready while `current` is still 2 → `current` genuinely drops to 1 two seconds later → the surviving pod becomes Ready two seconds after that) confirmed the original implementation returned `desired_to_ready_lag_seconds=2.0` (the moment of the unrelated readiness loss) instead of the correct `6.0` (the moment readiness was actually achieved at the post-scale-down target count).

### Evidence

Direct inspection of `compute_transition_replica_lags()`'s pre-correction source and a literal re-execution of its exact original algorithm against the scenario above (preserved in `tests/unit/test_dynamics_calibration_harness.py::TestCorr45ScaleDownFalsePositiveFix::test_old_buggy_logic_would_have_produced_a_false_positive`) reproduces the false-positive `2.0` result byte-for-byte, proving this is a real, demonstrable defect and not a hypothetical concern. Official index30 (the campaign's only official `replica_transition` attempt to date) is a **scale-up**, whose `ready`/`available` lags never resolved to a non-`null` value at all (the target was never reached within the observation period) — this defect was therefore never actually triggered by any official evidence, discovered instead through deliberate pre-first-use review before the first official scale-down slot (index31).

### Decision

`compute_transition_replica_lags()` is hardened: `desired_to_ready_lag_seconds`/`desired_to_available_lag_seconds` now **additionally require `current_replicas` to have already reached `target_replicas` at that same sample** before that sample's `ready`/`available` value is even considered — the replica-COUNT transition itself must be observed complete before readiness/availability *at the target count* can be credited. `desired_to_current_lag_seconds` is unaffected (it already checked `current_replicas` alone). Direction-specific "reached" semantics are also made explicit and frozen:

```text
scale-up   (target > pre-trigger count): value >= target_replicas   (tolerates transient overshoot)
scale-down (target < pre-trigger count): value == target_replicas EXACTLY (a <= test would wrongly
                                          credit an INTERMEDIATE count, e.g. 0 mid-restart en route
                                          to a final target of 1, as having "reached" that target)
```

Two new measurement-quality helpers are added: `summarize_replica_transition_trajectory_gaps()` (mirroring `summarize_dataplane_acquisition_gaps()`'s own pattern — a nominal 2.0s trajectory-sampling cadence is explicitly documented as NOT a guaranteed acquisition cadence, backed by index30's own real trajectory averaging ~3.86s/sample) and `compute_transition_replica_lag_uncertainty()` (reports, per lag, the elapsed-from-trigger time of the sample immediately before the first satisfying one, so the sampling-cadence slack around each reported lag is visible). Every lag is now explicitly documented, in code and in every attempt's own persisted `transition_metrics.lag_measurement_semantic` field, as a **FIRST-OBSERVED TRAJECTORY LAG** — the elapsed time of the first sampled observation satisfying a target condition, never an exact physical Kubernetes-event timestamp, since the true underlying state transition could have occurred anywhere in the gap since the previous sample.

### Scientific Impact

The transition's own frozen scientific design (manifest `pairs`/`timing`/`workload_by_pair`) is entirely unchanged. `route_integrity_protocol_version`, `dataplane_acquisition_protocol_version`, and `window_partitioning_protocol_version` are all unaffected — this correction concerns only the replica-count-lag measurement layer. Scale-up semantics are unchanged in substance (verified via dedicated regression tests reproducing the exact lag values a scale-up scenario would have produced both before and after this fix) — this correction only closes a gap in the scale-down direction, which had never yet been officially exercised.

### Preserved Invariants

Everything CORR-39 through CORR-44 preserved remains preserved (see their own entries). The three replica-count lags remain the ONLY formally-operationalized `measured_fields`; the four settling/realization-lag metrics remain `NOT_FORMALLY_OPERATIONALIZED`, with no arbitrary threshold invented by this correction.

### Non-Retroactivity

**Official index30 (`transrep_5-6_5to6_repeat1_attempt1_20260920T114902Z`) is NOT modified, recomputed, or reclassified by this correction.** It remains permanently `SEQUENCE_INDEX_30_ATTEMPT1_VALID`, `slot30.state=VALID`. Its own already-persisted `desired_to_current_lag_seconds=1.469` is confirmed, by direct regression test against this correction's own hardened implementation, to be produced identically by the corrected code (index30 is a scale-up, and its `current` lag computation path is unaffected by this fix in either direction) — see `tests/unit/test_dynamics_calibration_harness.py::TestCorr45ScaleUpRegressionUnchanged::test_index30_real_current_lag_reproduced_by_current_implementation`. Its own `desired_to_ready_lag_seconds`/`desired_to_available_lag_seconds=null` remain correctly `null` under the hardened implementation as well (the scale-up target was never observed reached at all, independent of this fix). This correction applies prospectively to the first official scale-down attempt (index31, not yet executed) and onward.

**Protocol-version decision**: no `protocol_version` field bump — pre-first-use hardening of the scale-down lag path specifically (never yet officially exercised), exactly analogous to CORR-39 sec.8/9/10's and CORR-44's own precedent. `active_correction_reference` is updated globally to `"CORR-45"` (`prior_correction_reference: "CORR-44"`).

**Applies to phases**: 15B.5B (the original, now-corrected implementation), 15B.5D (official index30, confirmed unaffected by this fix), 15B.5E (this correction). Index31 was not executed while freezing this correction. See `task_outputs/phase_15b5e_scale_down_transition_lag_hardening.md`.

## CORR-46 — Transition Collector Transient-Timeout Resilience (Phase 15B.5H)

### Problem

Official index33 attempt1 (`transrep_3-4_3to4_repeat1_attempt1_20260920T130955Z`, the campaign's next scheduled replica-transition slot) was correctly classified `execution_structural_validity: INVALID` (`collector_or_generator_structural_failure: pod readiness collector non-zero exit`). Root-cause inspection found three independent, previously-unguarded `subprocess.TimeoutExpired` failure points on the replica-transition execution path, all triggered by transient `kubectl`/`docker` command slowness at elapsed traffic time ≈141-143s:

1. `pod_readiness_collector.py`'s `fetch_pods_json()` let a `kubectl get pods` timeout propagate all the way out of `main()`, crashing the entire collector **subprocess** (non-zero exit) — the exact failure `readiness_proc.wait()`'s own `!= 0` check (unchanged by this correction) detected.
2. `run_dynamics_calibration_slot.py`'s `_sample_replica_trajectory_periodically()` had no per-iteration exception handling around `sample_replica_transition_state()`'s four `kubectl_jsonpath()` calls; the one timeout it hit permanently killed that background **thread** (an uncaught exception in a Python thread does not crash the process, but does end that thread's loop forever) — confirmed directly from the raw `replica_transition_timeline.csv`: clean 2s-cadence samples from t=0 to t=141.2s, then a single 106.6s gap, then one final sample from the separate FINAL_CAPTURE call site.
3. `_sample_route_integrity_periodically()` had the same unguarded-loop defect around `fetch_loadtest_endpoints_json()`/`check_relay_container_state()`, independently confirmed via its own "Exception in thread Thread-1" traceback in the same run's log.

`sample_replica_transition_state()` and `check_relay_container_state()` are each also called directly from the **main thread** (the one-off pre-traffic baseline sample and the STOP_TRAFFIC/FINAL CAPTURE step) — an unguarded timeout hitting either of those call sites specifically would have crashed the entire attempt outright, a strictly worse failure mode than a background thread quietly dying, and had not yet been triggered only by chance.

Whether index32's own extremely punishing immediately-preceding run (35 RPS/8 pods, 0.79% success, 7 pod restarts) caused this residual API-server/cluster pressure is **not established** and is not claimed here — index33's own preflight passed cleanly before launch, and equally plausible causes include index33's own load, transient Kubernetes API latency, or unrelated environmental scheduling effects. No causal claim is required to fix the underlying robustness gap.

### Evidence

Direct code inspection: `pod_readiness_collector.fetch_pods_json()` (pre-fix) had no try/except at all, unlike this same file's own `fetch_docker_ps_all()`, which already caught `(subprocess.TimeoutExpired, OSError)` — establishing that the recoverable-observation-error pattern was already this codebase's own local convention, simply not applied consistently. `sample_replica_transition_state()` and `_sample_route_integrity_periodically()`'s inline body were confirmed, by direct reading, to have zero exception handling around their respective `kubectl`/`docker` calls. `check_relay_container_state()` had no exception handling at all (unlike `check_local_listener_present()`/`resolve_local_listener_owner()`, which already caught `(FileNotFoundError, OSError)` but not `TimeoutExpired`). The already-existing `sample_dataplane_ingress_rules()` (CORR-41/Phase 15B.3O.1, hardened against exactly this failure mode in Phase 15B.3P.3) was confirmed to already implement the correct pattern and required no change — it served as the template for this correction.

Independently, index32's own V8 material-window partition (`[3.0, 239.0]`, 236.0s) was audited exactly against its raw `loadtest_route_integrity_timeline.json`: 23 evidence windows, 22 ordinary 10.0s windows (`[3.0,13.0)` through `[213.0,223.0)`) plus one final merged 16.0s window (`[223.0,239.0)`), first window start=3.0, last window end=239.0, total duration=236.0, no gap, no overlap — confirmed byte-for-byte correct. Index32 itself is entirely unaffected by, and unrelated to, this correction; it is documented here only because its audit was performed as a precondition of this same phase.

### Decision

Each of the three failure points is hardened to treat a single transient `subprocess.TimeoutExpired` as a **recoverable observation error**, never a fabricated value and never a fatal exception:

- `pod_readiness_collector.fetch_pods_json()`: wrapped in `try/except (subprocess.TimeoutExpired, OSError)`, returning `{"items": [], "_error": str(exc)}` — mirrors `fetch_docker_ps_all()`'s own pre-existing pattern exactly. A new `pod_readiness_observation_errors.csv` artifact persists, per occurrence, `timestamp_utc`/`elapsed_seconds`/`collector`/`error_type`/`timeout_seconds`/`message`/`continued_after_error` for audit.
- `sample_replica_transition_state()`: its four `kubectl_jsonpath()` calls, `get_pod_names()`, and `fetch_loadtest_endpoints_json()` are wrapped in one atomic `try/except subprocess.TimeoutExpired`, mirroring `sample_dataplane_ingress_rules()`'s own all-or-nothing observer-failure convention. On timeout, every replica/endpoint field is `None`/empty (never fabricated from a stale prior value) and a new `replica_state_observation_error` field records the failure. Hardening this ONE function protects all three of its call sites (pre-traffic baseline, periodic thread, final capture) at once.
- `_sample_route_integrity_periodically()`'s per-iteration body is wrapped in `try/except subprocess.TimeoutExpired`, appending one error sample (`endpoint_query_error` set, all other fields `None`) and continuing to the next scheduled iteration, rather than letting the thread die.
- `check_relay_container_state()` is additionally hardened directly (`try/except subprocess.TimeoutExpired: return None, None`) since it is called from both the periodic sampler AND the main thread's final-capture step; `check_local_listener_present()`/`resolve_local_listener_owner()`'s existing `except (FileNotFoundError, OSError)` clauses are extended to also catch `subprocess.TimeoutExpired`, for the same reason.

The pre-existing `collector_or_generator_structural_failure` structural-validity gate (`readiness_proc.wait()` exit-code check) is **explicitly unchanged** and untouched by this correction — it still fires on a genuine, unrecovered collector/generator failure. This correction narrows what counts as "genuine failure" to exclude a single recoverable sampling-command timeout; it does not weaken, remove, or bypass the gate itself (regression-tested: a collector hitting only recoverable per-tick errors still exits 0; a genuinely unhandled, non-timeout exception still propagates uncaught).

No arbitrary error-count pass/fail threshold (e.g. "max 3 timeouts") is introduced. Sampling errors are simply made observable, persisted, and honestly reported; if a collector's evidence becomes structurally incomplete for a genuine reason, the existing validity logic still applies unchanged.

### Scientific Impact

None. No classification rule, materiality window, scale-up/scale-down lag-target-state definition, or evidence-window-partitioning behavior changed. `route_integrity_protocol_version` (6), `dataplane_acquisition_protocol_version` (2), `window_partitioning_protocol_version` (2), and `transport_protocol_version` (5) are all unaffected — this correction concerns only observer/collector execution-robustness, not any measured scientific quantity. New fields (`replica_state_observation_error`, `pod_readiness_observation_errors.csv`) are purely additive; all existing consumers already read these dicts via `.get()`-style access and are unaffected.

### Preserved Invariants

Everything CORR-39 through CORR-45 preserved remains preserved (see their own entries). CORR-45's scale-down exact-target lag semantics, CORR-44's `[0,240)`/0.25/60.0s materiality parameters, and Protocol V8's window-partitioning rules are all verified byte-for-byte unchanged by direct regression test.

### Non-Retroactivity

**Official index32 (`transrep_7-8_8to7_repeat1_attempt1_20260920T125257Z`) is NOT modified, recomputed, or reclassified.** It remains permanently `VALID`; its V8 partition (23 windows, 22×10s + 1×16s, 236.0s total) is independently re-verified correct, not altered. **Official index33 attempt1 (`transrep_3-4_3to4_repeat1_attempt1_20260920T130955Z`) is NOT modified, recomputed, or reclassified.** It remains permanently `INVALID_RETRY_REQUIRED` at the ledger level and `execution_structural_validity: INVALID` in its own immutable `metadata.json`; it is retained as genuine negative execution evidence and is not deleted, overwritten, or removed from `slot33.all_attempt_ids`. This correction applies prospectively only: a future index33 attempt2 (if separately authorized) would be the first attempt to run under this hardening.

**Protocol-version decision**: no `protocol_version` field bump — this is an execution-harness robustness change with no effect on any scientific/methodology output schema or computation, exactly analogous to how CORR-44 and CORR-45 themselves introduced no protocol_version bump for their own (methodology-adjacent, not execution-adjacent) changes. `active_correction_reference` is updated globally to `"CORR-46"` (`prior_correction_reference: "CORR-45"`).

**Applies to phases**: 15B.5F/15B.5G (the attempts, index31/index32, that exercised the pre-hardening code without incident), 15B.5G (index33 attempt1, which exposed this defect), 15B.5H (this correction). See `task_outputs/phase_15b5h_transition_collector_timeout_hardening.md`.

## CORR-47 — Workload-Transition Pre-First-Use Operationalization (Phase 15B.6B)

### Problem

Phase 15B.6A's readiness audit, performed before the campaign's first `run_kind=workload_transition` slot (index38) could be officially attempted, found the execution harness had never implemented this run kind at all: `RUN_KIND_WORKLOAD_TRANSITION` was a bare string constant with no `is_..._slot()` gate and no dedicated execution path, so `dry_run_report()`/`main()` silently fell through to the STEADY-STATE body — reporting a factually wrong 180s traffic duration and `[60,180)` warmup/scored window for a run whose own manifest section (frozen since v1) specifies a 240s/60s-step plan (`workload_transitions.timing`). Additionally, unlike `replica_transitions`, the manifest names no `measured_fields` for `workload_transitions` at all, and no route-materiality/scored-window rule had ever been frozen for this run kind (CORR-44 is explicitly scoped to `replica_transition` only).

### Evidence

Direct source inspection (Phase 15B.6A, reconfirmed here): `grep "workload_transition" scripts/*.py` returned exactly one occurrence, the bare constant itself. Live dry-run reproduction confirmed the wrong output (`traffic_duration_seconds: 180`, `offered_rps: null`, no `rps_before`/`rps_after`, no trigger-time field) before this correction, and the correct output (`traffic_duration_seconds: 240`, `rps_before/rps_after`, `expected_requests` matching the schedule's own row count, `scientific_role`) after it. The manifest's own `workload_transitions.timing` block was confirmed present and frozen since v1, simply never read by any code path. `is_interval_material()` was confirmed, by direct source review, to have always been a generic, duration-relative gate (`fraction = duration / traffic_duration_seconds`) never specifically justified only for replica-count transitions — CORR-39 sec.7's own prose already stated the rule in general terms ("a materiality criterion ... covering >=25% of total traffic duration").

### Decision

Freezes, for `run_kind=workload_transition`, the same class of previously-unspecified details CORR-44/45 already froze for `replica_transition`:

- **Scientific role**: `scientific_role = "WORKLOAD_TRANSITION_TRAJECTORY_VALIDATION"` (`WORKLOAD_TRANSITION_SCIENTIFIC_ROLE`), explicitly excluded from steady-state fitting/interpolation-validation dataset construction (regression-tested: `execute_workload_transition_slot()`'s own source never constructs a `steady_state_fit_eligibility` field, unlike the steady-state body it is completely separate from).
- **Timing**: `resolve_workload_transition_timing_plan(manifest)` is a pure passthrough of the manifest's own `workload_transitions.timing` block (`pre_run_replica_stabilization_seconds=60`, `pre_step_traffic_seconds=60`, `trigger_at_traffic_elapsed_seconds=60`, `post_step_observation_seconds=180`, `total_traffic_seconds=240`, `total_seconds_per_run=300`) — no value invented, no steady-state 180s shape borrowed.
- **Workload-step trigger**: schedule-owned, not harness-owned. The schedule CSV itself already encodes both RPS regimes (`phase_name=pre_step`/`post_step`, verified for `dyncal_transwork_r4_8to32`: 8.0 RPS through `planned_elapsed_seconds<60.0`, 32.0 RPS from exactly `60.0` onward) — no new `kubectl`/trigger-command mechanism was implemented or is needed; `execute_workload_transition_slot()` launches the traffic generator with `--replay-schedule` exactly as `execute_replica_transition_slot()` does, and persists `workload_step_owner="schedule"`, `planned_workload_step_elapsed_seconds=60.0`, and (purely descriptively, per sec.14 of the governing task — never reinterpreted as a derived application-response lag) `first_post_step_request_elapsed_seconds`.
- **Materiality**: `workload_transition_route_scored_window(total_traffic_seconds) = (0.0, total_traffic_seconds)` and `WORKLOAD_TRANSITION_MATERIALITY_FRACTION = DEFAULT_ROUTE_INTEGRITY_MATERIALITY_FRACTION = 0.25`, yielding `workload_transition_materiality_threshold_seconds(240.0) = 60.0` — the same numeric threshold as replica-transition's own 240s-run value, because both run kinds share the same frozen 240s `total_traffic_seconds` and the same duration-relative fraction, not because one was silently copied from the other. A dedicated `scored_window_correction_reference: "CORR-47"` is persisted per attempt, distinct from CORR-44's own reference.
- **Derived metrics**: since the manifest names no `measured_fields` for this run kind, no metric name or settling threshold is invented. A single honest `WORKLOAD_TRANSITION_DERIVED_METRICS_STATUS = NOT_FORMALLY_OPERATIONALIZED` marker is persisted instead, with the full raw trajectory (target/realized RPS, latency p50/p95/p99, requests-in-progress, CPU, memory, fixed-replica desired/current/ready/available state) preserved via a new `cluster/workload_transition_timeline.csv` artifact for future analysis.
- **Execution path**: `execute_workload_transition_slot()` — a new, self-contained function mirroring `execute_replica_transition_slot()`'s own structure and reusing its V5 transport, collector-startup, route-integrity/dataplane sampler (both already CORR-46-hardened), and cleanup/restoration infrastructure verbatim. It sets up `slot["replicas"]` as a FIXED replica count (never a before/after pair) and issues no replica-scale command of any kind. The workload-transition trajectory artifact is ASSEMBLED post-hoc from already-collected raw streams (`metrics.csv`, produced by the same `collectors/metrics_collector.py` invocation every other run kind already uses unchanged) rather than a new expensive live Kubernetes/Prometheus poller, per the governing task's own explicit preference. A fixed-replica invariant (`evaluate_fixed_replica_invariant()`) is computed and persisted honestly; an observed drift is reported, not hidden, and is not turned into a new, invented invalidity rule (per the governing task's own explicit instruction) — only pre-existing, unrelated rules (e.g. `hpa_unexpectedly_active`) can invalidate an attempt.
- **Dry-run**: `dry_run_report()` gains a dedicated `is_workload_transition_slot(slot)` branch (mirroring the existing `is_replica_transition_slot(slot)` branch) reporting the correct 240s/60s-step/`rps_before`/`rps_after`/`expected_requests`/`scientific_role` fields; the steady-state fallback below it is untouched and unreachable for this run kind.

### Scientific Impact

None to any other run kind. `route_integrity_protocol_version` (6), `dataplane_acquisition_protocol_version` (2), `window_partitioning_protocol_version` (2), and `transport_protocol_version` (5) are all unaffected — Protocol V8's generic 10s-window/`<9s`-merge/`==9s`-standalone partitioning logic is reused completely unchanged, only now exercisable for a workload-transition material interval for the first time. No `replica_transition`, `fitting`, or `interpolation_validation` code path was modified.

### Preserved Invariants

Everything CORR-39 through CORR-46 preserved remains preserved (see their own entries). CORR-44's `[0,240)`/0.25/60.0s replica-transition materiality parameters and CORR-45's scale-up/scale-down lag semantics are unaffected and verified byte-for-byte unchanged by direct regression test.

### Non-Retroactivity

This is a pre-first-use operationalization: no official `workload_transition` attempt existed before this correction, so nothing is reclassified. The global campaign sequence, its schedules, and their hashes are completely unchanged (`ce1de372e92de0460e53f1ca36d31ec402712d5cd638738200fcedf70403e901`). No official index38 (or any other workload-transition slot) attempt was executed by this correction itself — it is prospective only, closing the implementation/methodology gap before the first live attempt.

**Protocol-version decision**: no `protocol_version` field bump — this correction only extends already-generic, already-proven mechanisms (materiality fraction, V8 partitioning, V5 transport, CORR-46 collector resilience) to a run kind that had never yet used them; it does not change any existing mechanism's own behavior. Exactly analogous to CORR-44/45/46's own precedent. `active_correction_reference` is updated globally to `"CORR-47"` (`prior_correction_reference: "CORR-46"`).

**Applies to phases**: 15B.6A (the readiness audit that discovered this gap), 15B.6B (this correction). See `task_outputs/phase_15b6b_workload_transition_harness_implementation.md`.

## CORR-48 — Workload-Transition Trajectory Traffic-Window Boundary (Phase 15B.6D)

### Problem

Index38 attempt1 (the campaign's first official `workload_transition` execution) was audited immediately after execution and found to have a suspicious row count in its derived `cluster/workload_transition_timeline.csv`: 27 rows against an official 240-second traffic window, with the collector processes deliberately sized for `total_traffic_seconds + 30 = 270` seconds (observation/cleanup robustness margin, unchanged from `replica_transition`'s own established pattern). Direct inspection of `build_workload_transition_trajectory_rows()` (CORR-47, Phase 15B.6B) confirmed it applied no boundary filter at all -- it classified every row in the raw `metrics.csv` as `pre_step`/`post_step` purely by comparing `elapsed_seconds` against the 60s workload-step boundary, with no upper bound. Three of index38's own 27 rows (`elapsed_seconds` = 240.0, 250.0, 260.01 -- all at or past the official 240s traffic boundary) were consequently admitted into the SCIENTIFIC trajectory artifact and mislabeled `post_step`, `target_rps=32.0`, even though they were genuinely sampled during the collector's own post-traffic observation buffer, not official traffic.

### Evidence

Direct raw-timestamp audit of index38's immutable `metrics.csv` (27 rows, elapsed 0.20s to 260.01s) and `traffic_summary.csv` (exactly 240 rows, elapsed 0-239, one per second) confirmed: the official traffic window is genuinely `[0, 240)` seconds (`requested_duration_seconds: 240.0` in the traffic generator's own metadata), and the raw metrics collector legitimately continued sampling for ~20 additional seconds beyond that into its own cleanup-margin buffer -- this is CORRECT, EXPECTED raw-collector behavior, not itself a defect. The defect was entirely in the downstream trajectory-assembly function's own missing upper-bound filter.

### Decision

`build_workload_transition_trajectory_rows()` gains a new required `total_traffic_seconds` parameter and now excludes any row whose `elapsed_seconds` does not satisfy `0.0 <= elapsed_seconds < total_traffic_seconds` -- the same half-open interval already used by `workload_transition_route_scored_window()` (CORR-47), so the derived trajectory and the route-materiality scored window now agree on exactly which interval counts as official traffic. A sample landing exactly at `total_traffic_seconds` (e.g. `240.0`) is excluded, not included, per the governing task's own explicit instruction. Raw `metrics.csv` itself is never truncated, filtered, or altered -- it remains full, authoritative collector-lifecycle evidence; only the DERIVED scientific artifact (`workload_transition_timeline.csv`) is now boundary-restricted. The call site in `execute_workload_transition_slot()` was updated to pass `timing["total_traffic_seconds"]` through.

### Scientific Impact

**None to index38's already-official result.** Direct code-path tracing confirms `workload_trajectory_rows` (the buggy, unfiltered list) fed only into (a) the `workload_transition_timeline.csv` artifact write and (b) `evaluate_fixed_replica_invariant()` -- it was never consulted by generator totals, latency statistics, success-rate computation, zero-success-interval detection, CORR-47 materiality, Protocol V8, route classification, `execution_structural_validity`, or the SLA evaluation, all of which are independently derived from `traffic_summary.csv`/`requests.csv`/`pod_readiness_history.csv`/`loadtest_route_integrity_timeline.json`. The one artifact that DID consume the buggy row set, `fixed_replica_invariant`, was independently re-evaluated over the corrected `[0,240)`-only row set and produced a byte-identical result (`desired_replicas_observed_distinct_values: [4.0]`, `current_replicas_observed_distinct_values: [4.0]`, `fixed_replica_invariant_holds: true`, `drift_sample_count: 0`) -- the three excluded post-traffic-buffer rows happened to also show `replicas_desired=4`/`replicas_current=4` (HPA restoration had not yet reduced the count by that point), so no retroactive correction to index38's own already-persisted `fixed_replica_invariant` field is warranted or made.

### Preserved Invariants

CORR-47's own frozen scientific role, schedule-owned step semantics, fixed-replica semantics, `[0, total_traffic_seconds)` route-materiality scored window (0.25 fraction, 60.0s threshold for 240s schedules), and Protocol V8 windowing are all completely unchanged. This correction concerns only the boundary of one derived, non-authoritative diagnostic artifact.

### Non-Retroactivity

**Official index38 (`transwork_r4_8to32_repeat1_attempt1_20260921T042932Z`) is NOT modified, recomputed, or reclassified.** Its `metadata.json`, its official `workload_transition_timeline.csv` (still the original 27-row file), and its ledger entry (`slot38.state=VALID`) all remain byte-identical and untouched. A separate, clearly-labeled read-only reanalysis artifact (`task_outputs/phase_15b6d_index38_workload_transition_trajectory_reanalysis.csv`, 24 rows: 6 `pre_step` + 18 `post_step`, generated from the same immutable raw `metrics.csv` under the corrected boundary logic) was produced outside the official attempt directory for reference. This correction is prospective only: a future workload-transition attempt (index39 onward, if separately authorized) will be the first to have its official trajectory artifact generated under this hardened boundary.

**Protocol-version decision**: no `protocol_version` field bump -- this is an implementation-robustness fix to one derived diagnostic artifact's own row-inclusion boundary, with no effect on any classification rule, materiality window, or scientific-role definition. `active_correction_reference` is updated globally to `"CORR-48"` (`prior_correction_reference: "CORR-47"`).

**Applies to phases**: 15B.6C (official index38, which exposed this defect via its own suspicious row count), 15B.6D (this correction). See `task_outputs/phase_15b6d_index38_workload_trajectory_boundary_audit.md`.

## CORR-55 — V2 Replica Floor Raised 1 -> 2 for Both V2 Controllers (Phase 4D-B)

### Problem

Both of the V2 multi-metric final-training campaign's live attempts at scientific session index 3 failed, and both root causes trace to the SAME single-replica vulnerability: `session_003` terminated on a sustained `zero_ready_replicas` safety trip, and the retry `session_003__attempt2` terminated on a pod liveness-probe restart -- in both cases, a single pod's own saturation or restart was sufficient to drive the deployment's entire Ready-replica count to zero, because the V2 floor (`min_replicas`) was 1, identical to V1's. The Phase 4B-R4F audit (R4F sec.11, Prioritised roadmap) listed this as its only P0 finding: with a floor of 1, PPO (and the planned CPU+memory HPA comparison arm) have no structural redundancy against a single pod's transient unhealthiness, unlike any realistic production deployment.

### Decision (user-authorised, 2026-11-07)

Raise the replica floor from 1 to 2 for **both** V2 controllers -- PPO (`autoscaler.src.rl_env.environment_v2`/`config_v2`) and the V2 CPU+memory HPA (`k8s/hpa/hpa_v2_cpu_memory.yaml`) -- so a single saturated or restarting pod can never alone reduce the deployment to zero Ready replicas. Upper bound (`max_replicas=8`) is unchanged. Nothing else in the V2 design changes: the 11-feature observation vector's order/names, the reward formula and every weight, the 3-action space, probes, pod resource requests/limits, the 30s control interval, the SLA target, the training seed, and every PPO hyperparameter are all byte-identical to before this correction.

**V1 is completely unaffected** -- `configs/experiment.yaml` (the file V1's own `min_replicas=1`/`max_replicas=8`/`initial_replicas=1` scaling block lives in, and which V2's shared-loader path also reads by default) was never edited. Instead, a new, additive, V2-only `scaling_v2` block was added to `configs/rl_v2.yaml` (a file that does not exist for V1 and is never read by V1 code), and `autoscaler/src/rl_env/config_v2.py`'s `load_env_config_v2_from_yaml()` was extended to read this block and override the V1-shaped base config's `gating.min_replicas`/`initial_replicas` (and `reward_v2`'s own `min_replicas`/`max_replicas`, which `load_reward_config_v2_from_yaml()` had never sourced from any YAML at all -- it silently used `RewardConfigV2`'s own Python-level default of 1 regardless of what `configs/reward_v2.yaml` contained, a second, previously-latent instance of the same floor=1 assumption). `configs/rl_v2.yaml`'s own `observation_v2` entry for `ready_replica_norm` (`raw_min`) was also changed from `1.0` to `2.0`, since this is the one place the V2 observation-normalization floor is encoded; `raw_max` (8.0) is unchanged. `k8s/hpa/hpa_v2_cpu_memory.yaml`'s `minReplicas` was changed from `1` to `2`; nothing else in that manifest (metrics, 260% CPU / 120% memory targets, behavior policies) changed. `k8s/hpa/hpa.yaml` (V1) and `k8s/app/deployment.yaml` (shared, `replicas: 1`) were NOT edited.

Every live-cluster-facing V2 script (`train_ppo_final_v2.py`, `train_ppo_live_v2.py`, `evaluate_ppo_vs_hpa_v2.py`) previously constructed `LiveKubernetesBackend` with no explicit `min_replicas`/`initial_replicas` at all, silently inheriting that class's own V1-identical defaults (1/1) regardless of what the V2 config said -- a third, independently-discovered instance of the same latent floor=1 assumption, now fixed by passing `min_replicas=env_config_v2.gating.min_replicas, initial_replicas=env_config_v2.initial_replicas` explicitly at every V2 construction site.

### Consequence: fresh campaign namespace, old campaign preserved as evidence

The 768-step campaign under `persistent-data/rl_training_v2/final/` (`session_000`-`session_002` verified, `session_003`/`session_003__attempt2` failed) was trained entirely under the floor=1 contract and can never be validly resumed under the floor=2 contract -- its checkpoints' own recorded `governing_hashes_snapshot` now provably mismatches the current `configs/rl_v2.yaml` hash (confirmed directly: `verify_resume_provenance()` now raises `CheckpointProvenanceError` -- "governing hash drift detected for 'configs/rl_v2.yaml'" -- when given `session_002`'s own real metadata against the post-correction file). This campaign's directory is **preserved untouched, byte-for-byte, as evidence** -- it is never written to again, and `train_ppo_final_v2.py` now fails closed (`FinalNamespaceViolation`) against any attempt to write into it. A fresh namespace, `persistent-data/rl_training_v2/final_min2/`, is prepared (directory constants only -- no session, ledger, or metadata file created by this correction) as the sole destination for every future V2 final-campaign write; a matching reserved final-model path, `models/ppo_v2/ppo_model_v2_min2.zip`, is also now protected by the same namespace-isolation guard, alongside (not instead of) the original `models/ppo_v2/ppo_model_v2.zip`. Campaign budget is unchanged: 5,120 total timesteps, 20 sessions of 256 steps each, same schedule logic.

### Scientific Impact

**None to V1.** V1's `configs/experiment.yaml`, `configs/rl.yaml`, `configs/reward.yaml`, `k8s/hpa/hpa.yaml`, `k8s/app/deployment.yaml`, `scripts/evaluate_ppo_vs_hpa.py`, every V1 training script, `models/ppo/*`, and `results/*` are all byte-identical before and after this correction (verified by SHA-256). **The entire V2 768-step campaign's scientific standing is unaffected in the sense that nothing about it is retroactively altered or reclassified** -- it remains exactly what it always was: a verified-valid partial campaign (`session_000`-`session_002`) that stopped due to a now-understood, now-fixed structural defect (floor=1), preserved as the evidence base for this very correction. No V2 result under the floor=1 contract was ever used as a final/official PPO-vs-HPA comparison (the formal evaluation has never been run), so there is no official result to invalidate.

### Preserved Invariants

The 11-feature observation vector's order, names, and every bound except `ready_replica_norm`'s `raw_min` are unchanged. The reward formula and every weight (`reliability`, `sla`, `pressure`, `waste`, `action`, `reversal`, `boundary`) are unchanged -- only `waste`'s own `min_replicas`/`max_replicas` comparison operands now correctly reflect 2/8 instead of silently using 1/8. The 3-action discrete action space, the near-tie-epsilon safe-selector mechanism (still 0.05, still `provisional_unvalidated`, untouched by this correction), the 30s control interval, probes, pod resources, the training seed, and every PPO hyperparameter are unchanged. `environment_v2.py` and `observation.py` required NO code changes at all -- both are already fully config-driven (gating/observation bounds flow through from `config_v2.EnvConfigV2` with no hardcoded replica literals), so the floor change propagates through them purely via the new config values.

### Non-Retroactivity

This correction is prospective only. It does not reclassify, recompute, or modify any file under `persistent-data/rl_training_v2/final/` or `persistent-data/rl_training_v2/pilot/` (both preserved, read-only, as evidence), any ledger, or any V1 result. `active_correction_reference` is updated globally to `"CORR-55"` (`prior_correction_reference`: the highest reference already in use in the codebase at the time of this correction, `"CORR-54"`, found via a codebase-wide, not merely doc-wide, scan -- `CORR-51`-`CORR-54` exist in code, e.g. `scripts/run_dynamics_calibration_slot.py`/`tests/unit/test_traffic_generator.py`, but are not yet reflected in this document; that pre-existing documentation gap is independent of this correction and not addressed here).

**Applies to phases**: 4B-R2, 4B-R3, 4B-R4F (prior phases that discovered and diagnosed the single-saturated-pod failure mode as the campaign's only P0), 4D-B (this correction). See `persistent-data/traffic/v2_phase4db_min2_contract_report.md`.

## CORR-56 — Idle (Zero-Traffic) Samples Incorrectly Classified Invalid (Phase 4D-D3)

### Problem

Phase 4D-D2's live validation (`S1 idle_baseline`, 120s, zero traffic) produced 0/4 valid samples -- every sample's `latency_observation_state` was `invalid`, contrary to the frozen contract (`observation.py`'s `determine_validity()`): `request_rate_rps == 0.0` is explicitly documented as a VALID `no_traffic` state, never a failure.

### Root Cause

`app/src/metrics.py`'s `REQUEST_COUNT`/`REQUESTS_IN_PROGRESS` (both `prometheus_client.Counter`/`Gauge` instances declared with label names but never pre-registered) are LAZILY labeled: `.labels(route=..., status_code=...).inc()`/`.labels(route=...).dec()` are only ever called from inside `app/src/main.py`'s request handler (lines 118, 139, 141), never at startup. Consequently, a pod that has never served a single request on a given route since its last restart exposes **no Prometheus time series at all** for `demo_app_requests_total{route=...}` or `demo_app_requests_in_progress{route=...}` -- not a stale-but-present zero, a genuinely absent series. `autoscaler/src/rl_env/live_backend.py`'s `_snapshot_metrics()` (prior to this fix) treated ANY absent query result uniformly as `None`, regardless of whether the query itself errored or simply matched nothing -- so a genuinely-idle-since-restart pod produced `request_rate_rps=None`/`requests_in_progress=None`, which `determine_validity()` correctly (per its OWN contract) classifies as `invalid` (`STALE_OR_MISSING_METRIC`), since `None` is indistinguishable, at that layer, from "Prometheus/the exporter is actually broken."

Confirmed directly, live, read-only: as of this phase, both pods' `demo_app_requests_in_progress{route="/work"}` and `rate(demo_app_requests_total{route="/work"}[1m])` now correctly return `0` (not absent) -- because Phase 4D-D1's own earlier traffic scenarios already touched these labels on both pods, "unlocking" their lazy registration. This is consistent with, and does not contradict, the root-cause diagnosis: the defect is specific to a pod that has NEVER served the route in question, which was true at the start of Phase 4D-D1 (pods had been idle since their last restart, 8-9 hours earlier) and is no longer reproducible live without a pod restart (not performed, per this phase's own constraints).

### Decision

Fixed in `autoscaler/src/rl_env/live_backend.py`'s `_snapshot_metrics()` (NOT the frozen `observation.py`, per this phase's explicit preference for non-governing files): an ABSENT (not errored) result for `request_rate_rps` or `requests_in_progress` now defaults to `0.0` instead of `None`, matching the project's own established null-vs-zero convention (CORR-9) and the frozen validity contract's own intent. This defaulting is applied PER-QUERY, keyed off that specific query's own error flag -- a genuine query error (not mere absence) on either metric still leaves it `None` and `query_error=True`, preserving the existing "an errored request-rate query forces error_rate_rps None too" contract byte-for-byte. `cpu_cores`/`memory_bytes` (cAdvisor container-level metrics, unrelated to the app's own lazy label registration, exist as soon as the container starts regardless of traffic) are deliberately left unchanged -- their absence remains a genuine failure signal, never defaulted.

### Scientific Impact

**None to any already-recorded evidence.** This is a live-cluster-query interpretation fix in a non-frozen module; it does not alter `observation.py`, the reward formula, the 11-feature contract, or any V1 behavior. It changes ONLY what a live idle (zero-traffic-since-restart) window reports: previously `invalid` (discarded, prior-value-held), now correctly `valid`/`no_traffic` with `request_rate_rps=0.0`. A live re-check post-fix (this phase) confirms `determine_validity()` now returns `sample_valid=True, latency_observation_state="no_traffic"` for the current idle state (though, as noted above, the pods were no longer in the exact pre-first-request condition that originally exposed the bug, so this re-check is a non-regression confirmation, not a reproduction of the original failure).

### V1 Investigation (evidence only)

Investigated whether this same root cause explains V1's PPO1/PPO2 evaluation terminations (`results/evaluation/ledgers/ppo_final_evaluation_ledger.jsonl`: `termination_reason: consecutive_invalid_steps_limit_reached`, PPO1 at 23/50 decisions, PPO2 at 20/50). **Evidence found: NO, this is not the same mechanism.** PPO1/PPO2 terminated roughly 10-11.5 minutes INTO their respective ~25-minute episodes, well after real traffic had already been flowing continuously (the lazy-registration gap this correction fixes can only manifest on or near a pod's very FIRST request since restart, which these mid-episode terminations are not consistent with). `results/evaluation/analysis.md`'s own existing text attributes these to genuine "telemetry/target unavailability," consistent with the project's independently-documented CORR-1 cluster-resource-contention finding, not this lazy-metric-registration defect. `v1_invalid_terminations_explained = NO` (evidence-based, not merely absence of proof).

### Preserved Invariants

`target_up=False` and genuine `query_error=True` paths are completely unaffected -- both still force every metric to `None` (target-down case) or still propagate `query_error=True` (per-query error case) exactly as before. No change to `observation.py`, `environment_v2.py`, `reward_v2.py`, or any reward weight/threshold.

### Non-Retroactivity

No historical telemetry, ledger, or evaluation result is reclassified or recomputed. This is prospective only, affecting live queries from this phase onward. `active_correction_reference` is updated globally to `"CORR-56"` (`prior_correction_reference: "CORR-55"`).

**Applies to phases**: 4D-D1 (discovered the defect via the idle_baseline scenario), 4D-D2 (diagnosed it as an open anomaly), 4D-D3 (this correction). See `persistent-data/traffic/v2_phase4dd3_prefreeze_fixes_report.md`.

## CORR-57 — V2 SLA Target and Waste Thresholds Recalibrated (Phase 4D-D3)

### Problem

Phase 4D-D2's live validation found that `constant_memory`/`constant_mixed` steady-state traffic consistently produces `p95_latency_norm ≈ 975ms` (a `histogram_quantile(0.95)` bucket-interpolation artifact of `app/src/metrics.py`'s frozen `(..., 0.5, 1.0, ...)` histogram buckets, given these profiles' own ~500ms memory-hold-per-request design -- nearly all requests land just inside the `(0.5s, 1.0s]` bucket, and Prometheus linearly interpolates within that single coarse bucket). Against the ORIGINAL 300ms SLA target, this made the SLA term permanently, uninformatively penalized for these two profiles (99.2-99.6% of historical steps penalized) regardless of actual system health, and simultaneously made the `waste` term's own `p95 < target` eligibility condition essentially unreachable whenever a step was otherwise eligible for it -- confirmed directly: **0 of 1,310 historical telemetry steps, and 0 of 30 new Phase 4D-D2 validation samples, ever triggered `waste`**, even under a read-only counterfactual loosening both thresholds up to `cpu_waste_threshold=1.3`/`memory_waste_threshold=1.2` (only 1.14% of eligible historical steps fired even then).

### Evidence Gates (all evaluated BEFORE any change, all passed)

Proposed values: `sla.p95_target_ms` 300 -> 1000, `waste.cpu_waste_threshold` 0.3 -> 1.0, `waste.memory_waste_threshold` 0.7 -> 1.0 (floor=2, per CORR-55, unchanged).

- **G1** (S2 `fixed2_cpu`, p95≈2.2s, still SLA-penalized under 1000ms target): all 9 valid samples (1758-2333ms) exceed 1000ms -- **PASS**.
- **G2** (memory/mixed steady-state p95≈975ms NOT penalized under 1000ms): all 17 qualifying samples (900-1000ms range) fall below 1000ms -- **PASS**.
- **G3** (waste fires in (0%, 40%] of eligible historical steps, `replicas > 2`): 167/792 eligible steps fire (21.09%) -- **PASS**.
- **G4** (zero sanity violations -- waste never co-occurs with `pressure > 0`, `p95 >= target`, `memory_per_replica >= 1.0`, or `cpu_per_replica >= 1.0`, in either historical or new validation data): 0 violations in both -- **PASS**.

All 4 gates passed; the recalibration was applied.

### Decision

Changed ONLY `configs/reward_v2.yaml`'s three values (`sla.p95_target_ms`, `waste.cpu_waste_threshold`, `waste.memory_waste_threshold`) as proposed above, with adjacent comments explaining the change and citing this correction. `configs/experiment.yaml`'s own shared `sla.p95_latency_ms=300` (V1's value, and the OLD "must stay in sync" convention `reward_v2.yaml` previously documented) is **unchanged** -- V2 now deliberately diverges from it. V1's own `configs/reward.yaml`/`reward.py` are completely untouched and confirmed still `300.0` by direct load.

**Consumers of the V2 SLA target traced** (file:line): `configs/reward_v2.yaml`'s own `sla.p95_target_ms` (the source); `autoscaler/src/rl_env/reward_v2.py:160`'s `load_reward_config_v2_from_yaml()` (reads it into `RewardConfigV2.sla_p95_target_ms`); `reward_v2.py:240` (the SLA excess calculation) and `reward_v2.py:252` (the waste eligibility condition's own `p95_ms < config.sla_p95_target_ms` check) -- both consume the loaded config value dynamically, requiring no code change. No V2-aware SLA-violation metric exists yet in `scripts/evaluate_ppo_vs_hpa_v2.py` (nothing to trace there); `decision_record_v2.py` stores only the raw p95 value, no baked-in threshold; `demo_ui/` was confirmed to have zero V2 awareness (no reference to `reward_v2`/`rl_v2`/`config_v2`/`KubernetesAutoscalerEnvV2` anywhere in that package) -- its own hardcoded `300.0` (`demo_ui/fixtures.py`) is V1's value and correctly untouched.

### Scientific Impact

Changes the V2 reward's own numeric behavior going forward (training has not yet started under either the old or new values, so nothing is retroactively invalidated). Pre-existing V2 tests that hardcoded a "violation" latency value shared between V1 and V2 calls (`tests/unit/test_environment_v2.py`'s `test_c_sla_violation_state`, `test_severe_sla_violation_dominates_over_healthy_waste_free_state`) were updated to use distinct, still-clearly-violating latency values per env version (V1 stays at 600ms against its unchanged 300ms target; V2 now uses 1500ms against its new 1000ms target) -- not a weakening of either test's intent.

### Preserved Invariants

Reward weights (`w_reliability`, `w_sla`, `w_pressure`, `w_waste`, `w_action`, `w_reversal`, `w_boundary`), the `pressure` block's own thresholds, the 11-feature observation contract, the action space, and every V1 value are completely unchanged. The `waste`-never-fires-during-an-SLA-violation invariant (the reward's own core design claim) is preserved and re-verified under the new thresholds (G4, and a new direct unit test).

### Non-Retroactivity

No historical telemetry or evaluation result is recomputed or reclassified; this applies prospectively to training/validation from this phase onward. `active_correction_reference` is updated globally to `"CORR-57"` (`prior_correction_reference: "CORR-56"`).

**Applies to phases**: 4D-D2 (discovered the SLA/waste defect and produced the counterfactual evidence base), 4D-D3 (this correction, gate evaluation and application). See `persistent-data/traffic/v2_phase4dd3_prefreeze_fixes_report.md`.

## CORR-58 — Safety-Aware Asymmetric Tie-Breaking for V2 Deterministic Inference (Phase 4D-E)

### Problem

The existing V2 near-tie-safe selector (`action_selection.select_safe_deterministic_action()`) resolves EVERY tie (two or more actions within `epsilon` of the maximum probability) to HOLD, regardless of which actions are tied. This is a reasonable default against `model.predict(obs, deterministic=True)`'s own index-0-biased tie-break (plain `argmax` silently favors SCALE_DOWN purely because of its index), but it is scientifically uninformative about WHICH kind of tie occurred and treats a HOLD/SCALE_UP tie identically to a SCALE_DOWN-involving tie -- there is no reason an uncertain policy should default to inaction with EQUAL conservatism in both directions when the live system is already observably under stress.

### Decision

Added `action_selection.select_safety_aware_action()` (and its `determine_decision_time_stress()` helper), used ONLY by the V2 deterministic-inference path (`safe_inference.py` -> `safe_ppo_controller.py` -> `scripts/evaluate_ppo_vs_hpa_v2.py`) -- the original `select_safe_deterministic_action()` is left completely UNCHANGED and remains available for any existing caller/test. Rule (tie set = the within-epsilon actions, by `ACTION_MAP` index order [SCALE_DOWN_ONE, HOLD, SCALE_UP_ONE]):

- **R1** exactly one action within epsilon of the max -> that action (a clear winner).
- **R2** SCALE_DOWN_ONE never wins any tie -- structurally guaranteed: every tie set containing SCALE_DOWN_ONE resolves via R4/R5 below, both HOLD, never SCALE_DOWN_ONE itself.
- **R3** tie set == `{HOLD, SCALE_UP_ONE}` -> SCALE_UP_ONE if the decision-time state is STRESSED, else HOLD.
- **R4** tie set == `{SCALE_DOWN_ONE, SCALE_UP_ONE}` or the full `{SCALE_DOWN_ONE, HOLD, SCALE_UP_ONE}` -> HOLD.
- **R5** tie set == `{SCALE_DOWN_ONE, HOLD}` -> HOLD.
- **R6** STRESSED could not be determined (all four raw stress inputs missing) for an otherwise-R3-eligible tie, OR any policy-inference failure (model None, invalid observation, probability-extraction failure, malformed probabilities) -> HOLD -- the same fail-safe default `safe_inference.py` already used, unchanged in spirit.

STRESSED (decision-time RAW state -- the observation that PRODUCED the current decision, carried forward from the previous step's/`env.reset()`'s own `info["raw_observation"]`, never the post-action state): `p95_latency_ms >= sla.p95_target_ms` OR `error_rate > 0` OR `cpu_per_replica > pressure.cpu_safe_high` OR `memory_per_replica > pressure.memory_safe_high`. All four thresholds are read from the LIVE, loaded `configs/reward_v2.yaml` (`RewardConfigV2`, via `load_reward_config_v2_from_yaml()` or a caller-supplied instance) -- never hardcoded. A missing individual value counts as not-stressed for that term alone; all four missing makes stress undeterminable (R6).

**Rationale**: availability first -- scale up fast under genuine stress (mirroring conventional HPA behavior, which reacts quickly to exceeded targets), scale down only on a clear signal, never as a coin-flip resolution of policy uncertainty. SCALE_DOWN must never be the "accidental" outcome of an uncertain policy.

### Scope

**V2 deterministic inference ONLY.** Training is completely unaffected: `model.learn()` samples stochastically from the policy's own categorical distribution (SB3 library internals), never through `action_selection.py`/`safe_inference.py`/`safe_ppo_controller.py` at all -- verified by `tests/unit/test_training_excludes_safe_tie_override.py`'s structural import-graph guard, extended this phase to also cover `scripts/train_ppo_final_v2.py`/`scripts/train_ppo_live_v2.py` (previously only the V1 trainers were guarded). V1's own evaluation/demo path (`scripts/evaluate_ppo_vs_hpa.py`, `scripts/run_demo_episode.py`) is completely untouched -- both files are confirmed byte-identical by hash. `action_selection.py`'s frozen `ACTION_MAP`/`types.py`'s `ACTION_NAMES` (action IDs 0=SCALE_DOWN_ONE, 1=HOLD, 2=SCALE_UP_ONE) are unchanged.

`near_tie_epsilon` remains at its existing PROVISIONAL value (0.05, `epsilon_status: provisional_unvalidated` in `configs/rl_v2.yaml`'s `safe_action` block) -- this correction changes WHAT HAPPENS when a tie occurs, not the epsilon threshold that defines a tie, which still awaits post-training calibration (R4F's own finding: 91% of real session_002 decisions would be forced into the tie band at this epsilon) exactly as before.

### Scientific Impact

None to any already-recorded evidence or trained model (`--execute` has never been invoked against this path). Additive decision-record fields (`tie_set`, `tie_rule_applied`, `stressed`, `stress_reasons`) let a future V2 evaluation run audit exactly which rule fired on every tied decision -- no existing field's meaning or value changes.

### Preserved Invariants

`select_safe_deterministic_action()`, `compute_margin_stats()`, `extract_action_probabilities()`, the 11-feature observation contract, the reward formula/weights, and every V1 behavior are unchanged.

### Non-Retroactivity

Prospective only -- no historical telemetry, ledger, or evaluation result is reclassified. `active_correction_reference` is updated globally to `"CORR-58"` (`prior_correction_reference: "CORR-57"`).

**Applies to phases**: 4D-E (this correction). See `persistent-data/traffic/v2_phase4de_tie_break_report.md`.

## CORR-59 — Pre-Registered Budget Amendment 5,120 → 12,288 Timesteps (Phase 4H-A)

### Problem

The frozen `final_min2` campaign contract (`FREEZE_MANIFEST.json`, Phase 4E) fixed the final training campaign at 5,120 total timesteps (20 sessions x 256 steps, 20 PPO updates). With the live campaign (PID 26490, `--max-sessions 19`) running toward that original budget, mid `session_001-019` at decision time, it was decided that 20 PPO updates may be too few for a stable converged policy given the project's remaining ~10-day window, and that the campaign should continue to 48 sessions (12,288 timesteps, 48 updates) once the currently-running process finishes its own bounded 19-session invocation.

### Decision

Raised `FINAL_TOTAL_TIMESTEPS_BUDGET` from 5,120 to 12,288 and `FINAL_PLANNED_SESSIONS` from 20 to 48 in `scripts/train_ppo_final_v2.py`. `FINAL_N_STEPS` (256), `FINAL_TRAINING_SEED` (42), and every hyperparameter, reward term, observation feature, and governing file (`configs/rl_v2.yaml`, `configs/reward_v2.yaml`, `autoscaler/src/rl_env/environment_v2.py`, `autoscaler/src/rl_env/observation.py`, `autoscaler/src/rl_env/reward_v2.py`) are UNCHANGED -- confirmed byte-identical by SHA-256 re-verification against `FREEZE_MANIFEST.json`.

**This decision was made and recorded BEFORE any training result from session_020 onward existed** -- the then-running campaign was bounded by `--max-sessions 19` to stop exactly at the original 5,120-step budget, and no session beyond `session_000` (reviewed in Phase 4F, prior to this decision) had been inspected. This preserves the project's own no-outcome-aware-change discipline, the same discipline already applied to V1's PPO1/PPO2 evaluation repeats.

**Rationale**:
- More PPO updates (48 instead of 20) give the policy more opportunity to converge to a stable behaviour within the project's remaining window.
- No hyperparameter, reward, or observation change is justified by any evidence -- this amendment changes ONLY the number of sessions/timesteps.
- The learning rate is constant and the training schedule (`training_schedule_v2.build_training_profile_rotation`) is fully deterministic given the same seed, so the first 5,120 steps of a 12,288-step run are schedule-identical to the already-completed/already-running 5,120-step run -- verified directly: `build_training_profile_rotation(42, 193) == build_training_profile_rotation(42, 470)[:193]`.

No new "continuation handling" code was required beyond the two constants: `run_final_training_campaign()`'s own `already_complete` early-return is keyed to the LIVE `FINAL_TOTAL_TIMESTEPS_BUDGET` constant (not a stored value), so it automatically un-blocks once the constant is raised; and `campaign_state["status"] = "in_progress"` is set unconditionally on every campaign start, so a stale `"completed"` value from the old 5,120-timestep run is overwritten via the existing, normal code path with no hand-editing of any JSON file. Separately confirmed (by direct `grep`) that no code anywhere exports a model to `V2_MODEL_PATH`/`FINAL_MODEL_PATH_MIN2` -- these are reserved-path guards only -- so the task's "preserve the 5,120 export" contingency is moot; no new preservation code was added.

### Scope

`scripts/train_ppo_final_v2.py` (budget constants and docstring only) and its tests (`tests/unit/test_v2_final_campaign.py`, `tests/unit/test_v2_final_campaign_resume.py` -- updated to reference the live budget constants dynamically instead of hardcoding 5,120/20). No change to learning, rewards, observations, actions, safety, or schedule logic. The running process (PID 26490) was never signaled, read-modified-written, or otherwise interacted with; no file under `persistent-data/rl_training_v2/final_min2/` was touched except the new `FREEZE_MANIFEST_AMENDMENT_001.json`.

### Scientific Impact

None to sessions 000-019 (schedule-identical, proven above) or to any already-recorded evidence. The campaign will now continue to session_047 (12,288 timesteps) instead of stopping at session_019, under the Tuesday continuation command `.venv/bin/python scripts/train_ppo_final_v2.py --execute-campaign --i-confirm-live-cluster-interaction --max-sessions 28`.

### Preserved Invariants

`FINAL_N_STEPS`, `FINAL_TRAINING_SEED`, all 5 governing files, the resume/provenance-verification path, and the `already_complete`/status-transition code paths are unchanged in behavior (only the constant each is keyed against changed).

### Non-Retroactivity

Prospective only -- no historical telemetry, ledger, or session result is reclassified. `active_correction_reference` is updated globally to `"CORR-59"` (`prior_correction_reference: "CORR-58"`).

**Applies to phases**: 4H-A (this correction). See `persistent-data/rl_training_v2/final_min2/FREEZE_MANIFEST_AMENDMENT_001.json` and `persistent-data/traffic/v2_phase4ha_budget_amendment_report.md`.

## CORR-60 — V2 Evaluation Pre-Registration (Phase 5A)

### Problem

No pre-registered record existed yet for how the final V2 evaluation (PPO-V2 vs. HPA baselines) will be designed and judged. Fixing the design only after a trained model and its results exist risks (even unintentionally) choosing workloads, thresholds, or a decision rule that happen to favor whichever arm looks better in a first look -- the same outcome-aware-design risk this project's other pre-registration decisions (the budget amendment, CORR-59; the tie-break rule, CORR-58) are built to avoid.

### Decision

Created `persistent-data/rl_training_v2/final_min2/EVALUATION_PREREGISTRATION.json` (machine-readable) and `docs/evaluation_v2_preregistration.md` (readable copy), **before** any evaluation run and before `session_047` (the final, 48-update checkpoint under the amended CORR-59 budget) exists:

- **Model under test**: the 48-update final checkpoint (session_047) only; intermediate checkpoints are for training-curve analysis only.
- **Three arms**: PPO-V2 (deterministic + safety-aware tie rule R1-R6, CORR-58), HPA-V2 CPU+memory (`demo-app-hpa-v2`), and a new third arm, HPA-CPU-only with the V2 replica floor (`demo-app-hpa-v2-cpu`, `k8s/hpa/hpa_v2_cpu_only_min2.yaml`, manifest-only this phase) -- isolating "does the memory metric matter" from "does the higher replica floor matter." All arms: bounds 2-8, same image, Metrics Server 15s verified live before every HPA run, Grafana 0.
- **Workloads**: 3 new held-out profiles (`configs/traffic_profiles/v2_eval/`: `eval_cpu_step`, `eval_memory_ramp`, `eval_mixed_burst`), seeds 9101-9103 (none used in training), none a member of `V2_TRAINING_PROFILE_ALLOWLIST` -- 3 repeats each, order rotated per repeat via a Latin square over the 3 arms.
- **Epsilon rule**: 10th percentile of the final model's top1-top2 probability margin over the last 5 training sessions' telemetry, capped at 0.05, floored at 0.0, computed once before evaluation (value not yet knowable -- training is still live).
- **Primary metrics**: SLA-violation rate at 1000ms (CORR-57's target) and replica-minutes; a full secondary list plus a 300ms reference-only SLA figure for historical comparability.
- **Validity**: tooling failure -> one retry, no scientific outcome; safety termination -> a counted outcome, never retried; every attempt preserved.
- **Decision rule**: mean +/- sd over 3 repeats; "better" requires winning all 3 repeats, not merely on average; no aggregate winner across conflicting metrics -- trade-offs are reported as such.

### Scope

Offline preparation only. No file under `persistent-data/rl_training_v2/` was touched except this new pre-registration record. `scripts/evaluate_ppo_vs_hpa_v2.py` and every `autoscaler/src/rl_env/` module are UNCHANGED this phase (that is Phase 5B's scope). Also prepared this phase, also unapplied/unrun: `k8s/hpa/hpa_v2_cpu_only_min2.yaml` (the third arm's manifest) and `k8s/metrics-server/metrics-server.yaml` + `scripts/apply_metrics_server_permanent.sh` (a prepared, not-yet-applied self-managed replacement for the Minikube metrics-server addon, which otherwise resets to its own 60s packaged default on every cluster recreation) -- the manifest's container image is a clearly marked placeholder (`PLACEHOLDER_IMAGE_TAG_UNVERIFIED_OFFLINE`), since the live addon's exact image tag could not be determined offline this phase.

### Scientific Impact

None to training (unaffected; all 43 `FREEZE_MANIFEST.json` files plus the AMENDMENT_001 trainer hash re-verified unchanged this phase). Establishes the fixed, pre-registered terms under which Phase 5B's evaluation-code implementation and the eventual live evaluation run will be judged.

### Preserved Invariants

The training contract (governing files, budget, schedule, safety) is completely untouched. `demo-app-hpa-v2`'s own manifest is untouched; the new CPU-only manifest is a separate, additional file.

### Non-Retroactivity

Prospective only. `active_correction_reference` is updated globally to `"CORR-60"` (`prior_correction_reference: "CORR-59"`).

**Applies to phases**: 5A (this correction). See `persistent-data/rl_training_v2/final_min2/EVALUATION_PREREGISTRATION.json`, `docs/evaluation_v2_preregistration.md`, and `persistent-data/traffic/v2_phase5a_eval_prep_report.md`.

## CORR-61 — Evaluation Safety/Termination Clarification (Phase 5C-A)

### Problem

`EVALUATION_PREREGISTRATION.json` (CORR-60) states a `validity_rules` policy (tooling failure -> one retry; safety termination -> counted, never retried) but did not specify WHICH live-cluster conditions count as a safety termination for the evaluation's real per-arm episode functions (`scripts/eval_v2_episodes.py`, built this phase) -- an omission that, left unresolved until evaluation results existed, could have invited an outcome-aware choice of thresholds.

### Decision

Stated here, BEFORE any evaluation run and before any further training result beyond what already existed at Phase 5B's close: the evaluation's safety-termination thresholds are the SAME ones training already uses, applied identically to EVERY arm (PPO and both HPA arms):

- **Per-pod memory >= 230Mi (`scripts.v2_pilot_safety.MEMORY_SAFETY_THRESHOLD_MIB`) for >= 2 consecutive samples (`SUSTAINED_BREACH_SAMPLES`)** -> `SafetyTerminationError`, for ANY arm.
- **A pod restart, or sustained zero-ready replicas** (the same two conditions `scripts.v2_pilot_safety.PilotSafetyCallback` already checks during training) -> `SafetyTerminationError`, for ANY arm.
- **The ppo_v2 arm additionally** ends on its own environment's existing 4-consecutive-invalid-step termination (`EnvConfigV2.max_consecutive_invalid_steps`, unchanged, already enforced by `KubernetesAutoscalerEnvV2`/`KubernetesAutoscalerEnv` -- this arm alone can reach a state where it has no valid observation to act on; the HPA arms do not depend on PPO's own observation pipeline to keep scaling).
- Any other failure (route/Prometheus/generator/Kubernetes-API error) -> `ToolingFailureError` (one retry, per the pre-registration).
- Invalid-sample counts and pod-restart counts are recorded identically in every arm's own per-decision `EpisodeSample`, regardless of whether that arm's run subsequently terminates.

No new threshold was invented; this correction only makes explicit, before evaluation, that the ALREADY-FROZEN training-time thresholds also govern evaluation-time termination, applied uniformly across arms (an HPA arm is not held to a looser safety standard than PPO, nor the reverse).

### Scope

`scripts/eval_v2_episodes.py` only (a new file). No training file, no `FREEZE_MANIFEST.json` file, and no governing-hash-tracked module is touched. `scripts/v2_pilot_safety.py`'s `MEMORY_SAFETY_THRESHOLD_MIB`/`SUSTAINED_BREACH_SAMPLES`/`get_pod_health`/`get_max_per_pod_memory_mib` are imported and reused, never modified.

### Scientific Impact

None to training. Fixes, before any evaluation result exists, exactly which live conditions terminate an evaluation run and how they are classified (safety vs. tooling) -- a prerequisite the pre-registration (CORR-60) left implicit.

### Preserved Invariants

The training-time safety thresholds themselves are unchanged (230Mi / 2 samples, pod-restart/zero-ready detection, 4-consecutive-invalid-step cap) -- this correction only extends their application to the evaluation's own, separate episode functions.

### Non-Retroactivity

Prospective only. `active_correction_reference` is updated globally to `"CORR-61"` (`prior_correction_reference: "CORR-60"`).

**Applies to phases**: 5C-A (this correction). See `persistent-data/traffic/v2_phase5ca_episode_recorders_report.md`.

## CORR-62 — CONTROLLER_CPU_ONLY_HPA_V2 Added (Post-Training, Phase 5D)

### Problem

Phase 5C-A's `hpa_v2_cpu_only` evaluation arm (the third arm, isolating "does the memory metric matter" from "does the higher replica floor matter") failed closed via `scripts/eval_v2_episodes.resolve_cpu_only_controller_identity()`, because `autoscaler/src/rl_env/controller_exclusivity.py` had no registered controller identity for it. That report named the exact one-line fix needed and deferred it to a dedicated post-training phase, per this project's own discipline of never making a mid-campaign change to a `FREEZE_MANIFEST.json` file without a separate, deliberate amendment.

### Decision

Made only AFTER the `final_min2` training campaign had already completed in full (48/48 sessions, 12,288/12,288 timesteps) -- added, additively only, to `autoscaler/src/rl_env/controller_exclusivity.py`:

```python
CONTROLLER_CPU_ONLY_HPA_V2 = "cpu_only_hpa_v2"
ALL_CONTROLLERS = (CONTROLLER_PPO, CONTROLLER_CPU_HPA, CONTROLLER_CPU_MEMORY_HPA, CONTROLLER_CPU_ONLY_HPA_V2)
HPA_NAME_BY_CONTROLLER[CONTROLLER_CPU_ONLY_HPA_V2] = "demo-app-hpa-v2-cpu"
```

No existing constant, function, or behavior in this file was modified or removed. Recorded as `persistent-data/rl_training_v2/final_min2/FREEZE_MANIFEST_AMENDMENT_002.json`.

### Scope

`autoscaler/src/rl_env/controller_exclusivity.py` only (plus its own tests). This file is one of the 43 `FREEZE_MANIFEST.json` files, but is **not** one of the 5 files `scripts/train_ppo_final_v2.py`'s `V2_GOVERNING_HASH_PATHS` checks for resume-provenance verification -- combined with training having already fully completed, this amendment has zero effect on training or on any already-verified session's resume provenance. `CONTROLLER_CPU_ONLY_HPA_V2` is never imported by any training-side module -- only by the evaluation-side `scripts/eval_v2_episodes.py`/`scripts/evaluate_v2_campaign.py` (both non-frozen, evaluation-only files from Phase 5B/5C-A).

### Scientific Impact

None to training (already complete; re-verified: 41 of 43 manifest files unchanged, the trainer still at its AMENDMENT_001 hash, only `controller_exclusivity.py` itself differs). Unblocks the `hpa_v2_cpu_only` evaluation arm for the upcoming live evaluation campaign.

### Preserved Invariants

`CONTROLLER_PPO`, `CONTROLLER_CPU_HPA`, `CONTROLLER_CPU_MEMORY_HPA` and their existing `HPA_NAME_BY_CONTROLLER` entries are unchanged. `acquire_controller`/`release_controller`/`detect_conflicting_hpas`/`current_status` behavior for all pre-existing controllers is unchanged (verified: all pre-existing `test_controller_exclusivity.py` tests pass unmodified).

### Non-Retroactivity

Prospective only -- applies to the evaluation campaign about to launch, never to any already-completed training session. `active_correction_reference` is updated globally to `"CORR-62"` (`prior_correction_reference: "CORR-61"`).

**Applies to phases**: 5D (this correction). See `persistent-data/rl_training_v2/final_min2/FREEZE_MANIFEST_AMENDMENT_002.json`.

