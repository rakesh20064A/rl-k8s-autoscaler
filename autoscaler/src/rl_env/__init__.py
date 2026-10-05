"""autoscaler.src.rl_env -- Phase 15B.1: RL Gymnasium environment MECHANICS.

Implements the frozen RL MDP contract v2 (docs/rl_mdp_spec.md, CORR-27,
CORR-28) as a real `gymnasium.Env` subclass (`environment.KubernetesAutoscalerEnv`)
that obtains all workload/replica state through an injected, pluggable
`backend.DynamicsBackend`.

IMPORTANT SCOPE LIMIT (Phase 15B.1): this package validates environment
MECHANICS only -- the Gymnasium API, the frozen 11-element observation
vector, the discrete action encoding, the V2 action-gating rules, the V2
reward formula, replica-transition timing (scale-up readiness delay,
scale-down immediacy), episode/termination semantics, and the transition
logging schema. It does NOT contain, and must never be used as, an
empirical model of how workload/replicas actually produce CPU, latency,
memory, or error-rate readings -- `scripted_backend.ScriptedBackend` is a
deterministic, test-only backend that consumes an explicitly supplied
timeline of synthetic metric snapshots (`TEST_ONLY_SYNTHETIC_METRICS`) and
must never be used to train a PPO policy. A future, separately-calibrated
empirical simulator backend and a live Kubernetes/Prometheus backend can
both be added later by implementing the same `DynamicsBackend` protocol,
without changing anything in this package's environment/observation/
reward/gating code.
"""
