# Part 01 — Host Machine

All facts VERIFIED directly on 2026-11-07 ~05:42 UTC unless marked otherwise.

- **macOS**: ProductVersion 26.6.2, BuildVersion 25G83. VERIFIED.
- **Chip**: Apple M1. VERIFIED (`sysctl -n machdep.cpu.brand_string`).
- **CPU cores**: 8 total (`hw.ncpu=8`); 4 performance cores
  (`hw.perflevel0.physicalcpu=4`), 4 efficiency cores
  (`hw.perflevel1.physicalcpu=4`). VERIFIED.
- **RAM**: 8 GB total (`hw.memsize` = 8,589,934,592 bytes). VERIFIED.
  **Risk note**: this is a small total for running Docker Desktop (4GB
  allocated, see 02_docker.md) + Minikube + a training workload + the host
  OS simultaneously — directly relevant to the CORR-1 mismatch and the
  repeated CPU-pressure findings across Phases 4B-R2 through R4F.
- **Disk free**: `/System/Volumes/Data` — 228Gi total, 148Gi used, **53Gi
  available**, 74% capacity. VERIFIED.
- **Power**: drawing from AC Power, battery 46% and charging. VERIFIED.
- **pmset settings** (`pmset -g`): `sleep 1` but currently **sleep is being
  prevented** by `powerd`, `coreaudiod` (x4), `runningboardd` (x3) —
  i.e. something is actively holding a no-sleep assertion right now.
  `displaysleep 0` (disabled, held by "Wispr Flow"). `hibernatemode 3`,
  `disksleep 10`, `standby 1`, `powernap 1`. VERIFIED. Historical note (not
  re-verified this phase, carried from Phase 4B-R2's investigation): a
  BatteryHealth-critical event was directly correlated with one training
  failure on 2026-10-12 — sleep/power state remains a credible risk factor
  for any future long-running training session on this specific machine.
- **Shell**: `/bin/zsh`. VERIFIED.
- **pyenv versions installed**: `system`, `3.10.14` (active, set by this
  repo's own `.python-version`), `3.10.14/envs/ml-env` (a separate pyenv
  virtualenv named `ml-env`). VERIFIED.
- **Global `python`/`python3`**: both resolve to pyenv shims
  (`~/.pyenv/shims/python[3]`), reporting `Python 3.10.14`. VERIFIED.

**No UNKNOWNs in this part.**
