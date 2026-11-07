# Part 02 — Docker Desktop

All VERIFIED directly, 2026-11-07.

- **Server version**: 29.7.2. **Context**: `desktop-linux` (active).
- **VM resources** (from `docker info` + `settings-store.json`):
  CPUs=4, MemoryMiB=4096 (`docker info` reports "Total Memory: 3.825GiB",
  i.e. ~4096MiB minus VM overhead), DiskSizeMiB=65536 (64GB).
  `KubernetesEnabled = False` (Docker Desktop's own built-in
  Kubernetes/Kind is correctly disabled — satisfies the CORR-8.3
  requirement). No `Swap`/`Rosetta`/`UseVirtualizationFramework` keys were
  found under those exact names in the settings-store schema for this
  Docker Desktop version — reported as **UNKNOWN** (not found) rather than
  guessed.
- **VM internals**: Kernel `7.0.12-linuxkit`, OSType `linux`, Architecture
  `aarch64`, Storage Driver `overlayfs`/containerd snapshotter, Cgroup
  Driver `cgroupfs`, Cgroup Version `2`.
- **Containers**: exactly 1 running — `minikube`
  (`gcr.io/k8s-minikube/kicbase:v0.0.50`), status "Up About a minute" at
  time of first check (i.e. the whole node was restarted very recently,
  shortly before this snapshot began — not caused by this session). No
  `rl-traffic-relay*` leftover containers found (clean).
- **Images** (host Docker, `docker images`):
  - `rl-autoscaler-demo-app:local` — image ID `75911ce139a5`, 268MB.
  - `rl-autoscaler-demo-app:phase5e` — image ID `dec88d9a305a`, 268MB
    (an older tagged build, retained).
  - Plus `alpine/socat` (the V5 relay image), `gcr.io/k8s-minikube/kicbase`
    (both tagged and digest-pinned forms), `docker/desktop-cloud-provider-kind`,
    `docker/desktop-containerd-registry-mirror`, `envoyproxy/envoy`,
    `jenkins/jenkins`, `kindest/node` — these last few appear to be
    Docker-Desktop-internal or unrelated leftover images, not part of this
    project's own pipeline.
  - **Discrepancy flagged**: `rl-autoscaler-demo-app:local`'s current image
    ID (`75911ce139a5`) does **not** match the image ID recorded in Phase
    4B-R3 (`sha256:da89cdff88c9075ab7a6421eeb4036138408257e7a1984880fc9b61f00173101`)
    — **the image has been rebuilt** at some point between 2026-10-12 and
    now. This phase could NOT re-verify the in-pod source hash against the
    current repo source (the live API server was unreachable throughout
    this snapshot — see 04_live_cluster.md) — **UNKNOWN** whether the
    current running pods' container actually match this new image ID or
    the current repo source byte-for-byte.
