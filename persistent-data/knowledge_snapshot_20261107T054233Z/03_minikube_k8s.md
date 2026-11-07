# Part 03 — Minikube and Kubernetes Platform

All VERIFIED directly, 2026-11-07.

- **minikube version**: v1.38.1.
- **Profile**: `minikube` (only profile), status `OK`/Running at check time
  (apiserver/kubelet/kubeconfig all "Running"/"Configured" per `minikube
  status`), though see 04_live_cluster.md — the apiserver became
  unreachable to `kubectl` shortly after this check due to sustained CPU
  saturation.
- **Profile config** (`~/.minikube/profiles/minikube/config.json`):
  `Driver=docker`, `Memory=3072` (MB), `CPUs=2`, `DiskSize=20000` (MB),
  `KubernetesVersion=v1.35.1`, `ContainerRuntime=docker`, node IP
  `192.168.49.2`, `Addons: {default-storageclass: true, metrics-server:
  true, storage-provisioner: true}`.
- **`docker inspect minikube` actual cgroup limits**: `NanoCpus=2000000000`
  (= 2 CPUs), `Memory=3221225472` bytes (= exactly 3072 MiB) — **matches**
  the profile config exactly; no mismatch between the profile's declared
  intent and the actual Docker-enforced container limit.
- **kubectl**: client/server both `v1.35.1`. Current context: `minikube`.
- **Node capacity/allocatable** (`kubectl describe node minikube`):
  `cpu: 4`, `memory: 4010880Ki` (≈3.92GiB) for BOTH Capacity and
  Allocatable — i.e. the kubelet believes it has the full Docker-VM-level
  4 CPU / ~4GB available to schedule against.
- **Node conditions**: `Ready=True`, `MemoryPressure=False`,
  `DiskPressure=False`, `PIDPressure=False` (as last read while the API
  server was still reachable).
- **Addons** (`minikube addons list`): `default-storageclass`,
  `metrics-server`, `storage-provisioner` enabled; everything else
  (ingress, dashboard, registry, gvisor, etc.) disabled.

## CORR-1 mismatch — confirmed, with exact numbers, including LIVE evidence

**The documented CORR-1 mismatch is real and independently re-confirmed
this session, with a live demonstration of its consequence:**

- Kubernetes' own node object advertises **4 CPU / ~4GB** allocatable
  (what the scheduler and any resource-aware logic would reason about).
- The ACTUAL Docker-enforced hard cgroup limit on the `minikube` container
  is **2 CPU / 3GB** (`NanoCpus=2000000000`, `Memory=3221225472`) — i.e.
  kubelet is scheduling against a resource picture that is **2x the real
  CPU ceiling and ~33% more memory than actually exists.**
- **Live, direct proof of consequence (VERIFIED, this session)**: shortly
  after this snapshot began, `docker stats --no-stream minikube` showed the
  container sustaining **229–483% CPU** (against its hard 200% cgroup cap)
  for several consecutive minutes, memory at 56–71% of its 3GB cap, while
  `kubectl get nodes` failed repeatedly with `TLS handshake timeout` — the
  control plane itself became unresponsive under this real CPU contention,
  not merely a workload pod. See 04_live_cluster.md for the full timeline.
  This is a live, reproduced instance of exactly the structural risk
  CORR-1 describes, occurring independently of any training/traffic
  workload (no PPO, no traffic generator, no relay was running at the
  time) — the restart of all ~15 cluster pods simultaneously (consistent
  with a very recent `minikube start`) was itself enough to saturate the
  real 2-CPU ceiling for several minutes.

**No other UNKNOWNs in this part.**
