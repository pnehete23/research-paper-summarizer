from __future__ import annotations

from fastapi import Depends, Header, HTTPException, Request
import time

from ..settings import settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not settings.require_api_key:
        return
    if not settings.api_key or x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


_RATE_BUCKETS: dict[str, list[float]] = {}


def rate_limiter(request: Request) -> None:
    if not settings.rate_limit_enabled:
        return
    # Use API key if present, else client host as key
    key = request.headers.get("x-api-key") or request.client.host or "anon"
    now = time.time()
    window = 60.0
    limit = max(1, settings.rate_limit_per_minute)
    burst = max(1, settings.rate_limit_burst)
    bucket = _RATE_BUCKETS.setdefault(key, [])
    # Drop old entries
    cutoff = now - window
    i = 0
    for t in bucket:
        if t >= cutoff:
            break
        i += 1
    if i:
        del bucket[:i]
    # Enforce burst
    if len(bucket) >= burst:
        raise HTTPException(status_code=429, detail="Too Many Requests (burst limit)")
    # Enforce rate over window
    if len(bucket) >= limit:
        raise HTTPException(status_code=429, detail="Too Many Requests")
    bucket.append(now)
