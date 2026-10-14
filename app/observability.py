"""Logging, request IDs and metrics.

- Every HTTP request gets an ID (taken from an incoming ``X-Request-ID`` header or generated), returned in
  the response header and attached to every log line written while handling it.
- Logs are plain text by default or one JSON object per line (``RAG_LOG_FORMAT=json``) for log shippers.
- Metrics are kept in-process and exposed in the Prometheus text format at ``/metrics``. A tiny built-in
  registry avoids a dependency; it covers counters and histograms, which is all this service needs.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from collections import defaultdict
from contextvars import ContextVar
from datetime import datetime, timezone

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

_access_logger = logging.getLogger("app.access")


# ------------------------------------------------------------------------------------------------ logging
class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "fields", None) or {})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(fmt: str = "text", level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.addFilter(_RequestIdFilter())
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per model call is noise; we log stages
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)  # replaced by the "request" log line


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields) -> None:
    """Log ``event key=value ...`` as text, with the same fields as structured keys in JSON mode."""
    text = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.log(level, f"{event} {text}".strip(), extra={"fields": {"event": event, **fields}})


# ------------------------------------------------------------------------------------------------ metrics
_DEFAULT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80)


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._help: dict[str, tuple[str, str]] = {}
        self._counters: dict[str, dict[tuple, float]] = defaultdict(lambda: defaultdict(float))
        self._histograms: dict[str, dict[tuple, list]] = defaultdict(dict)

    def describe(self, name: str, kind: str, help_text: str) -> None:
        self._help[name] = (kind, help_text)

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        with self._lock:
            self._counters[name][tuple(sorted(labels.items()))] += value

    def observe(self, name: str, seconds: float, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            series = self._histograms[name].setdefault(key, [[0] * len(_DEFAULT_BUCKETS), 0.0, 0])
            for i, bound in enumerate(_DEFAULT_BUCKETS):
                if seconds <= bound:
                    series[0][i] += 1
            series[1] += seconds
            series[2] += 1

    def render(self, gauges: dict[str, list[tuple[dict, float]]] | None = None) -> str:
        lines: list[str] = []

        def header(name: str, default_kind: str) -> None:
            kind, help_text = self._help.get(name, (default_kind, name))
            lines.extend([f"# HELP {name} {help_text}", f"# TYPE {name} {kind}"])

        with self._lock:
            for name, series in sorted(self._counters.items()):
                header(name, "counter")
                for key, value in sorted(series.items()):
                    lines.append(f"{name}{_labels(dict(key))} {value:g}")
            for name, series in sorted(self._histograms.items()):
                header(name, "histogram")
                for key, (buckets, total, count) in sorted(series.items()):
                    labels = dict(key)
                    for bound, bucket_count in zip(_DEFAULT_BUCKETS, buckets):
                        lines.append(f"{name}_bucket{_labels({**labels, 'le': f'{bound:g}'})} {bucket_count}")
                    lines.append(f"{name}_bucket{_labels({**labels, 'le': '+Inf'})} {count}")
                    lines.append(f"{name}_sum{_labels(labels)} {total:.6f}")
                    lines.append(f"{name}_count{_labels(labels)} {count}")
        for name, samples in sorted((gauges or {}).items()):
            header(name, "gauge")
            for labels, value in samples:
                lines.append(f"{name}{_labels(labels)} {value:g}")
        return "\n".join(lines) + "\n"


def _labels(labels: dict) -> str:
    if not labels:
        return ""
    escaped = (f'{k}="{str(v).replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"' for k, v in labels.items())
    return "{" + ",".join(escaped) + "}"


METRICS = Metrics()
METRICS.describe("http_requests_total", "counter", "HTTP requests by method, route template and status.")
METRICS.describe("http_request_duration_seconds", "histogram", "HTTP request duration (full body, incl. streaming).")
METRICS.describe("rag_answers_total", "counter", "Answers by status and abstention reason.")
METRICS.describe("rag_stage_duration_seconds", "histogram", "Duration of pipeline stages (retrieval, generation, ingest_*).")
METRICS.describe("rag_ingest_jobs_total", "counter", "Finished background ingestion jobs by outcome.")


class Timer:
    """``with Timer() as t: ...`` then ``t.ms``; also records a stage histogram when ``stage`` is given."""

    def __init__(self, stage: str | None = None) -> None:
        self.stage = stage
        self.seconds = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self._start
        if self.stage:
            METRICS.observe("rag_stage_duration_seconds", self.seconds, stage=self.stage)

    @property
    def ms(self) -> int:
        return int(self.seconds * 1000)

    @property
    def elapsed_ms(self) -> int:
        """Time so far, while the block is still running."""
        return int((time.perf_counter() - self._start) * 1000)


# --------------------------------------------------------------------------------------------- middleware
_QUIET_ROUTES = {"/metrics", "/health", "/health/ready", "/jobs/{job_id}", "/"}


class ObservabilityMiddleware:
    """Request ID + access log + HTTP metrics. Pure ASGI, so streamed responses are timed to their last byte."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = dict(scope.get("headers") or []).get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex[:16]
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status = 500

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                MutableHeaders(scope=message).append("X-Request-ID", request_id)
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            seconds = time.perf_counter() - started
            route = getattr(scope.get("route"), "path", None) or "unmatched"
            METRICS.inc("http_requests_total", method=scope["method"], route=route, status=str(status))
            METRICS.observe("http_request_duration_seconds", seconds, route=route)
            if route not in _QUIET_ROUTES or status >= 500:
                log_event(
                    _access_logger, "request", method=scope["method"], path=scope["path"], route=route,
                    status=status, duration_ms=int(seconds * 1000),
                )
            request_id_var.reset(token)
