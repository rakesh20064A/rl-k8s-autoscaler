"""demo_ui: additive, offline demo/UI layer for the RL-vs-HPA autoscaler project.

This package is fully isolated from the governed scientific core
(scripts/train_ppo_live_session.py, scripts/traffic_orchestrator.py, PPO
reward/observation/action logic, Kubernetes/HPA manifests). It must never
import from, invoke, or modify any of those at import time.
"""
