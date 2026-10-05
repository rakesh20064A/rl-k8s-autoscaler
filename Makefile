PROJECT_ROOT := $(shell pwd)
VENV         := $(PROJECT_ROOT)/.venv
PYTHON       := $(VENV)/bin/python
PIP          := $(VENV)/bin/pip
PYTEST       := $(VENV)/bin/pytest

.PHONY: help bootstrap venv install install-rl install-dev lint test \
        start-minikube install-monitoring deploy-app health-check \
        collect-metrics backup-results clean-runtime

help:
	@echo "RL-K8s-Autoscaler project targets"
	@echo ""
	@echo "  bootstrap           Check prerequisites and create Python venv"
	@echo "  venv                Create .venv using python3.10"
	@echo "  install             Install base Python dependencies"
	@echo "  install-rl          Install RL dependencies (SB3, Gymnasium)"
	@echo "  install-dev         Install development/lint dependencies"
	@echo "  lint                Run ruff linter"
	@echo "  test                Run unit and smoke tests"
	@echo "  start-minikube      Start Minikube cluster (requires Docker Desktop)"
	@echo "  install-monitoring  Install Prometheus + Grafana via Helm"
	@echo "  deploy-app          Build and deploy the FastAPI application"
	@echo "  health-check        Verify full stack is operational"
	@echo "  collect-metrics     Start the metrics collector process"
	@echo "  backup-results      Copy experiment results to persistent-data/backups"
	@echo "  clean-runtime       Remove replaceable runtime resources (NOT persistent-data)"

bootstrap:
	@scripts/bootstrap.sh

venv:
	@echo "Creating Python 3.10 virtual environment..."
	@python3.10 -m venv $(VENV)
	@$(PIP) install --upgrade pip setuptools wheel
	@echo "Virtual environment ready at $(VENV)"

install: venv
	@$(PIP) install -r requirements/base.txt

install-rl: install
	@$(PIP) install -r requirements/rl.txt

install-dev: install
	@$(PIP) install -r requirements/dev.txt

lint:
	@$(VENV)/bin/ruff check .

test:
	@$(PYTEST) tests/unit tests/smoke -q

start-minikube:
	@scripts/start_minikube.sh

install-monitoring:
	@scripts/install_monitoring.sh

deploy-app:
	@scripts/deploy_app.sh

health-check:
	@scripts/health_check.sh

collect-metrics:
	@scripts/collect_metrics.sh

backup-results:
	@scripts/backup_results.sh

clean-runtime:
	@scripts/cleanup_runtime_only.sh
