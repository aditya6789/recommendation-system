"""Prometheus metrics for recommendation APIs."""

from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter(
    "recsys_http_requests_total",
    "Total number of API requests",
    ["endpoint", "method", "status"],
)
REQUEST_LATENCY = Histogram(
    "recsys_http_request_latency_seconds",
    "API request latency in seconds",
    ["endpoint", "method"],
)
INTERACTION_EVENTS = Counter(
    "recsys_interaction_events_total",
    "Total interaction events by type",
    ["event_type"],
)
RECOMMENDATION_IMPRESSIONS = Counter(
    "recsys_recommendation_impressions_total",
    "Recommendation impressions by experiment variant",
    ["variant"],
)


def render_metrics() -> bytes:
    """Expose Prometheus metrics payload."""
    return generate_latest()
