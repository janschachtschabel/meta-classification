"""Request-level counters for ``/metrics`` — what the six gauges could not answer.

The ServiceMonitor was wired up and had nothing to alert on: "the API is returning 500s"
and "requests are taking twenty seconds" were both invisible, because every gauge described
the process rather than the traffic (audit OPS-9).

Hand-rolled like the gauges beside it, for the same reason: a client library would be a new
runtime dependency, and this is three dictionaries. Process-local, like the rate limiter and
the job runner — the chart pins one replica, so a single process's counters *are* the
service's.

**Cardinality is the thing to get right.** Keying on the request PATH would let anyone mint
unbounded series by scanning URLs, in a process that never restarts. So the key is the
matched route TEMPLATE (``/models/{model_name}``), of which there are as many as the app has
endpoints, and anything that matched no route is counted once under ``<unmatched>``.
"""

from __future__ import annotations

import threading

UNMATCHED = "<unmatched>"

# (method, template, status) -> how many. The status is an int, so a 500 shows up as a
# series and not as a missing one.
_requests: dict[tuple[str, str, int], int] = {}
# (method, template) -> (count, total seconds). A Prometheus summary's _sum/_count pair:
# rate(sum)/rate(count) is the average latency over the scrape interval, which is the
# question an alert actually asks.
_latency: dict[tuple[str, str], tuple[int, float]] = {}

# The counters are touched from the event loop and read by a scrape on that same loop, but
# a `def` route runs in a worker thread, so the increments are guarded rather than assumed
# atomic. Uncontended, this costs nothing measurable.
_lock = threading.Lock()


def record(method: str, template: str, status: int, seconds: float) -> None:
    with _lock:
        key = (method, template, status)
        _requests[key] = _requests.get(key, 0) + 1
        count, total = _latency.get((method, template), (0, 0.0))
        _latency[(method, template)] = (count + 1, total + seconds)


def reset() -> None:
    """Drop every counter. For tests — nothing in the app calls it."""
    with _lock:
        _requests.clear()
        _latency.clear()


def _label(value: str) -> str:
    """Escape a label value as the Prometheus text format requires."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def render() -> list[str]:
    """The counter lines for the ``/metrics`` body."""
    with _lock:
        requests = sorted(_requests.items())
        latency = sorted(_latency.items())
    lines = [
        "# HELP apiv3_requests_total Requests by method, route template and status code.",
        "# TYPE apiv3_requests_total counter",
    ]
    lines += [
        f'apiv3_requests_total{{method="{_label(method)}",route="{_label(route)}",'
        f'status="{status}"}} {count}'
        for (method, route, status), count in requests
    ]
    lines += [
        "# HELP apiv3_request_duration_seconds Request duration by method and route template.",
        "# TYPE apiv3_request_duration_seconds summary",
    ]
    for (method, route), (count, total) in latency:
        labels = f'method="{_label(method)}",route="{_label(route)}"'
        lines.append(f"apiv3_request_duration_seconds_count{{{labels}}} {count}")
        lines.append(f"apiv3_request_duration_seconds_sum{{{labels}}} {total:.6f}")
    return lines
