"""Traffic generator package for Kubernetes autoscaling experiments."""

__all__ = ["TrafficGenerator", "TrafficProfile", "generate_arrival_schedule"]


def __getattr__(name: str):
    if name in ("TrafficGenerator", "TrafficProfile", "generate_arrival_schedule"):
        from loadgen.traffic_generator import (
            TrafficGenerator,
            TrafficProfile,
            generate_arrival_schedule,
        )
        return {
            "TrafficGenerator": TrafficGenerator,
            "TrafficProfile": TrafficProfile,
            "generate_arrival_schedule": generate_arrival_schedule,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
