#!/usr/bin/env python3
"""
scripts/generate_dashboard_configmap.py
Generates the canonical Grafana dashboard JSON and the corresponding
declarative Kubernetes ConfigMap manifest for the dashboard sidecar.
Uses only standard library modules (no third-party dependencies required).
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_JSON_PATH = PROJECT_ROOT / "dashboards" / "grafana" / "autoscaling-overview.json"
CONFIGMAP_YAML_PATH = PROJECT_ROOT / "k8s" / "monitoring" / "grafana-dashboard.yaml"

def create_dashboard() -> dict:
    datasource = {
        "type": "prometheus",
        "uid": "prometheus"
    }

    panels = [
        # 1. Request Rate
        {
            "id": 1,
            "title": "Request Rate",
            "type": "timeseries",
            "description": "Total requests per second handled by demo-app on route /work",
            "datasource": datasource,
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "smooth",
                        "lineWidth": 2,
                        "fillOpacity": 15,
                        "gradientMode": "opacity"
                    },
                    "unit": "reqps",
                    "min": 0
                },
                "overrides": []
            },
            "options": {
                "tooltip": {"mode": "single", "sort": "none"},
                "legend": {"displayMode": "list", "placement": "bottom", "calcs": ["mean", "max", "lastNotNull"]}
            },
            "targets": [
                {
                    "datasource": datasource,
                    "editorMode": "code",
                    "expr": "sum(rate(demo_app_requests_total{namespace=\"autoscaler-app\",route=\"/work\"}[1m]))",
                    "legendFormat": "Requests/sec",
                    "range": True,
                    "refId": "A"
                }
            ]
        },
        # 2. p50 Latency
        {
            "id": 2,
            "title": "p50 Latency",
            "type": "timeseries",
            "description": "Median (p50) request latency for route /work",
            "datasource": datasource,
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "smooth",
                        "lineWidth": 2,
                        "fillOpacity": 10
                    },
                    "unit": "s",
                    "min": 0
                },
                "overrides": []
            },
            "options": {
                "tooltip": {"mode": "single", "sort": "none"},
                "legend": {"displayMode": "list", "placement": "bottom", "calcs": ["mean", "max", "lastNotNull"]}
            },
            "targets": [
                {
                    "datasource": datasource,
                    "editorMode": "code",
                    "expr": "histogram_quantile(0.50, sum(rate(demo_app_request_duration_seconds_bucket{namespace=\"autoscaler-app\",route=\"/work\"}[1m])) by (le))",
                    "legendFormat": "p50 latency",
                    "range": True,
                    "refId": "A"
                }
            ]
        },
        # 3. p95 Latency
        {
            "id": 3,
            "title": "p95 Latency",
            "type": "timeseries",
            "description": "95th percentile request latency for route /work",
            "datasource": datasource,
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "smooth",
                        "lineWidth": 2,
                        "fillOpacity": 10
                    },
                    "unit": "s",
                    "min": 0
                },
                "overrides": []
            },
            "options": {
                "tooltip": {"mode": "single", "sort": "none"},
                "legend": {"displayMode": "list", "placement": "bottom", "calcs": ["mean", "max", "lastNotNull"]}
            },
            "targets": [
                {
                    "datasource": datasource,
                    "editorMode": "code",
                    "expr": "histogram_quantile(0.95, sum(rate(demo_app_request_duration_seconds_bucket{namespace=\"autoscaler-app\",route=\"/work\"}[1m])) by (le))",
                    "legendFormat": "p95 latency",
                    "range": True,
                    "refId": "A"
                }
            ]
        },
        # 4. p99 Latency
        {
            "id": 4,
            "title": "p99 Latency",
            "type": "timeseries",
            "description": "99th percentile request latency for route /work",
            "datasource": datasource,
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "smooth",
                        "lineWidth": 2,
                        "fillOpacity": 10
                    },
                    "unit": "s",
                    "min": 0
                },
                "overrides": []
            },
            "options": {
                "tooltip": {"mode": "single", "sort": "none"},
                "legend": {"displayMode": "list", "placement": "bottom", "calcs": ["mean", "max", "lastNotNull"]}
            },
            "targets": [
                {
                    "datasource": datasource,
                    "editorMode": "code",
                    "expr": "histogram_quantile(0.99, sum(rate(demo_app_request_duration_seconds_bucket{namespace=\"autoscaler-app\",route=\"/work\"}[1m])) by (le))",
                    "legendFormat": "p99 latency",
                    "range": True,
                    "refId": "A"
                }
            ]
        },
        # 5. Application CPU
        {
            "id": 5,
            "title": "Application CPU",
            "type": "timeseries",
            "description": "Total CPU cores used across demo-app pods",
            "datasource": datasource,
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "smooth",
                        "lineWidth": 2,
                        "fillOpacity": 15,
                        "gradientMode": "opacity"
                    },
                    "unit": "short",
                    "decimals": 3,
                    "min": 0
                },
                "overrides": []
            },
            "options": {
                "tooltip": {"mode": "single", "sort": "none"},
                "legend": {"displayMode": "list", "placement": "bottom", "calcs": ["mean", "max", "lastNotNull"]}
            },
            "targets": [
                {
                    "datasource": datasource,
                    "editorMode": "code",
                    "expr": "sum(rate(container_cpu_usage_seconds_total{namespace=\"autoscaler-app\",pod=~\"demo-app-.*\"}[1m]))",
                    "legendFormat": "CPU cores",
                    "range": True,
                    "refId": "A"
                }
            ]
        },
        # 6. Application Memory
        {
            "id": 6,
            "title": "Application Memory",
            "type": "timeseries",
            "description": "Total working set memory consumed by demo-app pods",
            "datasource": datasource,
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "smooth",
                        "lineWidth": 2,
                        "fillOpacity": 15,
                        "gradientMode": "opacity"
                    },
                    "unit": "bytes",
                    "min": 0
                },
                "overrides": []
            },
            "options": {
                "tooltip": {"mode": "single", "sort": "none"},
                "legend": {"displayMode": "list", "placement": "bottom", "calcs": ["mean", "max", "lastNotNull"]}
            },
            "targets": [
                {
                    "datasource": datasource,
                    "editorMode": "code",
                    "expr": "sum(container_memory_working_set_bytes{namespace=\"autoscaler-app\",pod=~\"demo-app-.*\"})",
                    "legendFormat": "Working set bytes",
                    "range": True,
                    "refId": "A"
                }
            ]
        },
        # 7. Replica Count
        {
            "id": 7,
            "title": "Replica Count",
            "type": "timeseries",
            "description": "Active demo-app deployment replicas",
            "datasource": datasource,
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 24},
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "stepAfter",
                        "lineWidth": 2,
                        "fillOpacity": 10
                    },
                    "unit": "short",
                    "decimals": 0,
                    "min": 0
                },
                "overrides": []
            },
            "options": {
                "tooltip": {"mode": "single", "sort": "none"},
                "legend": {"displayMode": "list", "placement": "bottom", "calcs": ["max", "lastNotNull"]}
            },
            "targets": [
                {
                    "datasource": datasource,
                    "editorMode": "code",
                    "expr": "kube_deployment_status_replicas{namespace=\"autoscaler-app\",deployment=\"demo-app\"}",
                    "legendFormat": "Replicas",
                    "range": True,
                    "refId": "A"
                }
            ]
        },
        # 8. HTTP 5xx Error Rate
        {
            "id": 8,
            "title": "HTTP 5xx Error Rate",
            "type": "timeseries",
            "description": "Rate of 5xx server errors on route /work",
            "datasource": datasource,
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 24},
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "smooth",
                        "lineWidth": 2,
                        "fillOpacity": 10
                    },
                    "unit": "reqps",
                    "min": 0
                },
                "overrides": []
            },
            "options": {
                "tooltip": {"mode": "single", "sort": "none"},
                "legend": {"displayMode": "list", "placement": "bottom", "calcs": ["mean", "max", "lastNotNull"]}
            },
            "targets": [
                {
                    "datasource": datasource,
                    "editorMode": "code",
                    "expr": "sum(rate(demo_app_requests_total{namespace=\"autoscaler-app\",route=\"/work\",status_code=~\"5..\"}[1m])) or on() vector(0)",
                    "legendFormat": "5xx errors/sec",
                    "range": True,
                    "refId": "A"
                }
            ]
        },
        # 9. Requests In Progress
        {
            "id": 9,
            "title": "Requests In Progress",
            "type": "stat",
            "description": "Current concurrent requests active on /work endpoint",
            "datasource": datasource,
            "gridPos": {"h": 6, "w": 12, "x": 0, "y": 32},
            "fieldConfig": {
                "defaults": {
                    "unit": "short",
                    "decimals": 0,
                    "min": 0,
                    "color": {"mode": "palette-classic"}
                },
                "overrides": []
            },
            "options": {
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"],
                    "fields": ""
                },
                "orientation": "auto",
                "textMode": "auto",
                "colorMode": "value",
                "graphMode": "area",
                "justifyMode": "auto"
            },
            "targets": [
                {
                    "datasource": datasource,
                    "editorMode": "code",
                    "expr": "sum(demo_app_requests_in_progress{namespace=\"autoscaler-app\",route=\"/work\"})",
                    "legendFormat": "In Progress",
                    "range": True,
                    "refId": "A"
                }
            ]
        },
        # 10. Last Observed Work Intensity (Diagnostic Only)
        {
            "id": 10,
            "title": "Last Observed Work Intensity (Diagnostic Only)",
            "type": "stat",
            "description": "Diagnostic application gauge only. With multiple replicas this value may be stale or ambiguous and is not the authoritative workload-intensity source.",
            "datasource": datasource,
            "gridPos": {"h": 6, "w": 12, "x": 12, "y": 32},
            "fieldConfig": {
                "defaults": {
                    "unit": "short",
                    "decimals": 1,
                    "min": 0,
                    "color": {"mode": "palette-classic"}
                },
                "overrides": []
            },
            "options": {
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"],
                    "fields": ""
                },
                "orientation": "auto",
                "textMode": "auto",
                "colorMode": "value",
                "graphMode": "area",
                "justifyMode": "auto"
            },
            "targets": [
                {
                    "datasource": datasource,
                    "editorMode": "code",
                    "expr": "max(demo_app_last_work_intensity{namespace=\"autoscaler-app\"})",
                    "legendFormat": "Last Intensity",
                    "range": True,
                    "refId": "A"
                }
            ]
        }
    ]

    dashboard = {
        "annotations": {"list": []},
        "editable": True,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 1,
        "id": None,
        "links": [],
        "liveNow": False,
        "panels": panels,
        "refresh": "10s",
        "schemaVersion": 39,
        "tags": ["kubernetes", "autoscaling", "hpa", "rl"],
        "templating": {"list": []},
        "time": {"from": "now-15m", "to": "now"},
        "timepicker": {
            "refresh_intervals": ["5s", "10s", "30s", "1m", "5m"]
        },
        "timezone": "browser",
        "title": "RL Kubernetes Autoscaling - Overview",
        "uid": "rl-k8s-autoscaling-overview",
        "version": 1,
        "weekStart": ""
    }

    return dashboard

def main():
    DASHBOARD_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIGMAP_YAML_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. Generate canonical JSON
    dashboard_dict = create_dashboard()
    json_str = json.dumps(dashboard_dict, indent=2)
    DASHBOARD_JSON_PATH.write_text(json_str + "\n")
    print(f"Wrote canonical dashboard JSON to: {DASHBOARD_JSON_PATH}")

    # 2. Format ConfigMap YAML using standard library
    # Indent JSON by 4 spaces for multiline YAML literal block
    indented_json = "\n".join("    " + line if line else "" for line in json_str.splitlines())

    configmap_yaml = f"""# k8s/monitoring/grafana-dashboard.yaml
# Declarative ConfigMap for the Grafana dashboard sidecar.
# Generated deterministically from dashboards/grafana/autoscaling-overview.json.
# Label 'grafana_dashboard: "1"' is required for discovery by the sidecar.
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboard-autoscaling-overview
  namespace: monitoring
  labels:
    grafana_dashboard: "1"
data:
  autoscaling-overview.json: |
{indented_json}
"""
    CONFIGMAP_YAML_PATH.write_text(configmap_yaml)
    print(f"Wrote ConfigMap manifest to: {CONFIGMAP_YAML_PATH}")

if __name__ == "__main__":
    main()
