# Multi-stage Dockerfile for the RL autoscaler demo application.
# Base: python:3.10-slim — small, arm64-native, no emulation needed on Apple Silicon.
# Build stage installs dependencies; runtime stage has only what is needed.

FROM python:3.10-slim AS builder

WORKDIR /build

# All packages (fastapi, uvicorn[standard], prometheus-client) have prebuilt arm64
# manylinux wheels — no C compiler needed in the builder stage.
COPY requirements.txt .
RUN pip install --upgrade pip --quiet --no-cache-dir && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.10-slim AS runtime

# Non-root user for least-privilege execution
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy only application source — not tests, not local venv
COPY src/ .

# Kubernetes will send SIGTERM before SIGKILL — uvicorn respects it by default.
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV LOG_HEALTH_REQUESTS=false
ENV MAX_WORK_DURATION_SECONDS=10

USER appuser

EXPOSE 8080

# Use uvicorn directly; gunicorn is not needed for a single-process benchmark pod.
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
