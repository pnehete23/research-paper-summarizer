from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator, Optional, Dict, Any, Tuple
import logging
import contextvars
import json

try:
    from langfuse import Langfuse
except Exception:  # pragma: no cover
    Langfuse = None  # type: ignore

from ..settings import settings


_client: Optional["Langfuse"] = None

# Per-request context
_request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)


def set_request_id(request_id: Optional[str]) -> None:
    _request_id_var.set(request_id)


def get_request_id() -> Optional[str]:
    return _request_id_var.get()


def log_info(message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    payload: Dict[str, Any] = {
        "level": "INFO",
        "message": message,
        "request_id": get_request_id(),
    }
    if metadata:
        payload.update(metadata)
    logging.getLogger("app").info(json.dumps(payload))


def get_client() -> Optional["Langfuse"]:
    global _client
    if not settings.observability or Langfuse is None:
        return None
    if _client is None:
        try:
            _client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
            )
        except Exception:
            _client = None
    return _client


@contextmanager
def trace(name: str, metadata: Optional[Dict[str, Any]] = None) -> Iterator[None]:
    start = time.time()
    client = get_client()
    obs_span = None
    if client is not None:
        try:
            obs_span = client.trace(name=name, metadata=metadata or {})
        except Exception:
            obs_span = None
    try:
        try:
            log_info("trace_start", {"name": name, **(metadata or {})})
        except Exception:
            pass
        yield None
    finally:
        if obs_span is not None:
            try:
                obs_span.end(metadata={"duration_ms": int((time.time() - start) * 1000)})
            except Exception:
                pass
        try:
            log_info("trace_end", {"name": name, "duration_ms": int((time.time() - start) * 1000)})
        except Exception:
            pass


# --- Lightweight in-process metrics ---
_counters: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}
_histograms: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], list[float]] = {}


def inc_counter(name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
    if not settings.metrics_enabled:
        return
    key = (name, tuple(sorted((labels or {}).items())))
    _counters[key] = _counters.get(key, 0.0) + value


def observe_histogram(name: str, value_ms: float, labels: Optional[Dict[str, str]] = None) -> None:
    if not settings.metrics_enabled:
        return
    key = (name, tuple(sorted((labels or {}).items())))
    _histograms.setdefault(key, []).append(value_ms)


def export_metrics() -> str:
    if not settings.metrics_enabled:
        return "# metrics disabled\n"
    lines: list[str] = []
    # Counters
    for (metric, labels), val in _counters.items():
        label_str = "" if not labels else "{" + ",".join([f"{k}=\"{v}\"" for k, v in labels]) + "}"
        lines.append(f"{metric}{label_str} {val}")
    # Histograms: export count and avg in ms
    for (metric, labels), values in _histograms.items():
        count = len(values)
        avg = (sum(values) / count) if count else 0.0
        label_str = "" if not labels else "{" + ",".join([f"{k}=\"{v}\"" for k, v in labels]) + "}"
        lines.append(f"{metric}_count{label_str} {count}")
        lines.append(f"{metric}_avg_ms{label_str} {avg:.2f}")
    return "\n".join(lines) + "\n"

