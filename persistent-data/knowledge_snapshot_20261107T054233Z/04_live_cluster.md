# Part 04 — Live Cluster State

**API server reachability**: the cluster WAS reachable for the first ~2-3
minutes of this snapshot session (all data below captured then), after
which `kubectl` began failing with `net/http: TLS handshake timeout`, then
later `EOF` — sustained and unrecovered for 5+ minutes of repeated,
reasonably-spaced checks (see 03_minikube_k8s.md's CORR-1 section for the
`docker stats` evidence: the `minikube` container was sustaining 229-483%
CPU against its 200%-of-cap hard limit throughout). Per this phase's own
instruction, items that could not be gathered during the unreachable window
are marked **UNKNOWN (API server unreachable)** below, not guessed.

## Namespaces and pods (VERIFIED, captured before unreachability)

Namespaces: `autoscaler-app`, `default`, `kube-node-lease`, `kube-public`,
`kube-system`, `monitoring` — all `Active`, age 62d.

`kubectl get pods -A -o wide`:

| Namespace | Pod | Ready | Restarts | Age |
|---|---|---|---|---|
| autoscaler-app | demo-app-6579f6d9b-96fjb | 1/1 | 6 (3m13s ago) | 25d |
| autoscaler-app | demo-app-6579f6d9b-jhdpm | 1/1 | 9 (3m13s ago) | 25d |
| kube-system | coredns-7d764666f9-27pfd | 1/1 | 29 (3m13s ago) | 62d |
| kube-system | coredns-7d764666f9-pvn68 | 1/1 | 31 (3m13s ago) | 62d |
| kube-system | etcd-minikube | 1/1 | 27 (3m13s ago) | 62d |
| kube-system | kube-apiserver-minikube | 1/1 | 132 (3m13s ago) | 62d |
| kube-system | kube-controller-manager-minikube | 1/1 | 70 (3m13s ago) | 62d |
| kube-system | kube-proxy-nq6z6 | 1/1 | 22 (3m13s ago) | 62d |
| kube-system | kube-scheduler-minikube | 1/1 | 29 (3m13s ago) | 62d |
| kube-system | metrics-server-9d74bb658-7vwgr | **0/1** | 2 (31s ago) | 2m49s |
| kube-system | storage-provisioner | 1/1 | 232 (70s ago) | 62d |
| monitoring | kube-prom-stack-grafana-59cdf9bbd8-264lx | 2/3 | 206 (3m13s ago) | 61d |
| monitoring | kube-prom-stack-kube-prome-operator-...-b24vt | 1/1 | 259 (2m22s ago) | 61d |
| monitoring | kube-prom-stack-kube-state-metrics-...-jmgmt | 1/1 | 280 (2m22s ago) | 62d |
| monitoring | prometheus-...-prometheus-0 | 2/2 | 146 (3m13s ago) | 62d |

**Every single pod in the cluster shows a restart timestamped ~"3m13s
ago"** (relative to that check) — i.e. the ENTIRE cluster's pod set
restarted together at the same moment, consistent with a `minikube stop`
→ `minikube start` cycle occurring shortly before this snapshot began (not
performed by this session — this session never ran `minikube start/stop`).
`metrics-server`'s brand-new pod (age 2m49s, a fresh ReplicaSet-created pod,
not merely a container restart) was **actively crash-looping (0/1 Ready, 2
restarts in 31s)** at the moment of capture — see probe-failure detail
below.

## demo-app Deployment — full spec (VERIFIED)

- `replicas: 2` (live `spec.replicas`), `readyReplicas: 2`,
  `availableReplicas: 2` — **does NOT match canonical `replicas: 1`**
  (`k8s/app/deployment.yaml`'s own baked-in value, confirmed still `1` in
  the `kubectl.kubernetes.io/last-applied-configuration` annotation — the
  live replica count was changed by an out-of-band `kubectl scale`, not by
  re-applying the manifest). This exact drift (2 replicas, HPA absent) was
  already flagged in the Phase 4B-R4C/D/E reports from 2026-11-03 and
  **has not been restored in the 4 days since**.
- Image: `rl-autoscaler-demo-app:local`, `imagePullPolicy: Never`.
  `imageID` of the running pods was NOT captured before unreachability —
  **UNKNOWN (not captured before API server became unreachable)**; see
  02_docker.md for the host-side image ID discrepancy already found
  (current `75911ce139a5` vs. the `da89...` ID recorded in Phase 4B-R3).
- Resources: requests cpu=100m/memory=64Mi, limits cpu=500m/memory=256Mi
  (matches `k8s/app/deployment.yaml` exactly).
- Probes: readiness HTTP `/health` (5s/5s/3-fail/3s-timeout), liveness HTTP
  `/health` (10s/10s/5-fail/3s-timeout) — matches the manifest (the TCP-
  liveness experiment from Phase 4B-R3 is confirmed reverted; only its
  documentation comment remains).
- `terminationGracePeriodSeconds: 30`. `envFrom: demo-app-config`.
  `deployment.kubernetes.io/revision: "7"`, `generation: 4108` (a very high
  generation count given only 7 revisions — consistent with 4108 individual
  spec/status updates having occurred over this Deployment's 62-day life,
  e.g. every HPA/PPO replica-count change increments `.status`, not
  `.metadata.generation`... actually `generation` increments only on
  `.spec` changes — 4108 spec-level changes is itself notable and
  consistent with the very large number of `kubectl scale`/manifest-apply
  cycles this Deployment has been through across the whole project).

## Services / EndpointSlices (VERIFIED)

- `demo-app` (ClusterIP, port 80→8080, label `purpose: primary-metrics-source`).
- `demo-app-loadtest` (NodePort 31891, port 80→8080, `sessionAffinity: None`,
  label `purpose: experiment-load-generator-access`).
- EndpointSlices for both list the SAME 2 pod IPs as Ready endpoints
  (consistent with the live 2-replica Deployment).

## HPA (VERIFIED)

`kubectl get hpa -n autoscaler-app` → **No resources found.** The canonical
`demo-app-hpa` is absent — consistent with the same, still-unresolved drift
first found in Phase 4B-R4C/D/E.

## ConfigMaps / Events

ConfigMap `demo-app-config`: `LOG_LEVEL=INFO, LOG_HEALTH_REQUESTS=false,
MAX_WORK_DURATION_SECONDS=10` (matches `k8s/app/configmap.yaml`). Recent
events were captured but showed only routine historical scaling/probe
events predating this session (not reproduced here; available in
`raw/04_cluster_raw.txt`).

## Metrics Server (VERIFIED — partial, then blocked)

- Live args (captured before unreachability):
  `["--cert-dir=/tmp","--secure-port=4443","--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname","--kubelet-use-node-status-port","--metric-resolution=60s","--kubelet-insecure-tls"]`
  — **`--metric-resolution=60s`, NOT the 15s value set in Phase 4B-R4D**.
  **Confirmed regression**: the 15s patch was a live, one-off
  `kubectl patch` with no persistence mechanism; it did not survive the
  subsequent Minikube stop/start cycle, which caused minikube's own addon
  manager to reconcile the Deployment back to the addon's hardcoded
  60s default. This is a genuine, newly-confirmed finding: **the 15s
  Metrics Server resolution must be re-applied after every Minikube
  restart; it is not currently durable in any committed manifest/config.**
- Pod was actively crash-looping (`0/1 Ready`, 2 restarts in 31s) with
  `kubectl describe pod` events showing `Unhealthy` readiness/liveness
  probes (`connect: connection refused`, `context deadline exceeded`,
  `HTTP probe failed with statuscode: 500`) — the identical cold-start-
  race signature documented in Phase 4B-R4D/E, now recurring again after
  this latest restart.
- **5 timestamped Metrics API samples**: **UNKNOWN (API server became
  unreachable before this could be attempted)**.

## Monitoring namespace / Helm / Prometheus (UNKNOWN — API server unreachable)

All of the following were attempted and **blocked by the sustained API-
server unreachability** documented above:
- `kubectl get pods -n monitoring -o wide` (beyond the one earlier capture
  already shown in the pods-A table above).
- `helm list -A`, `helm get values kube-prom-stack -n monitoring`.
- Prometheus StatefulSet resources/retention/scrapeInterval/evaluationInterval
  live re-verification (the `k8s/monitoring/values.yaml` FROM_CODE values —
  retention=6h, retentionSize=2GB, scrapeInterval=10s, evaluationInterval=10s
  — are documented in 08_configs.md, but could not be re-confirmed as
  actually applied to the live StatefulSet this session).
- `kubectl exec` into the Prometheus pod for `/-/ready`, active targets,
  or the observation-pipeline PromQL queries.
- Running host processes related to the project: **separately confirmed
  VERIFIED via plain `ps aux`** (does not need the K8s API) — **none found**
  (no `kubectl port-forward`, `traffic_generator`, `train_ppo*`,
  `evaluate_*`, `caffeinate`, `streamlit`, `socat`, or `minikube
  service/tunnel` process running).
- Controller lock file (`persistent-data/controller/active_controller.json`,
  a local file, independent of the K8s API): **VERIFIED** —
  `{"active_controller": "none", "updated_at_utc":
  "2026-10-12T07:53:39.620859+00:00"}` — unchanged since Phase 4B-R2.

## Live-vs-manifest field-by-field comparison (what could be compared)

| Field | Manifest (`k8s/app/deployment.yaml`) | Live | Match |
|---|---|---|---|
| replicas | 1 | 2 | **NO** |
| image | `rl-autoscaler-demo-app:local` | same | YES |
| imagePullPolicy | Never | Never | YES |
| cpu/memory requests/limits | 100m/64Mi, 500m/256Mi | same | YES |
| readinessProbe | HTTP /health 5/5/3/3 | same | YES |
| livenessProbe | HTTP /health 10/10/5/3 | same | YES |
| HPA (`k8s/hpa/hpa.yaml`, `demo-app-hpa`) | should exist | **absent** | **NO** |
| Metrics Server resolution (Phase 4B-R4D target) | 15s (not a committed manifest — only a prior live patch) | 60s (reverted) | **NO** |

**Top risk from this part**: the live cluster was already known (as of
2026-11-03) to be non-canonical, and 4 days later it is STILL non-canonical
AND additionally exhibits a live control-plane-level instability (API
server unresponsive under real CPU contention) that was not previously
documented as occurring absent any training/traffic workload. This should
be resolved (canonical restoration + a documented, repeatable way to
reapply the 15s Metrics Server patch, or an acceptance that 60s is the
realistic default) before any further live diagnostic or training work.
