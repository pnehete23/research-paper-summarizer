from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import time
import uuid
import logging

from .settings import settings
from .database import init_db
from .api.health import router as health_router
from .api.ingest import router as ingest_router
from .api.query import router as query_router
from .api.datasets import router as datasets_router
from .api.datasets import datasets_router as datasets_alias_router
from .api.review import router as review_router
from .api.eval import router as eval_router
from .services.observability import set_request_id, log_info, inc_counter, observe_histogram


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name)
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_request_id_and_logging(request: Request, call_next):
        req_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        set_request_id(req_id)
        start = time.time()
        try:
            log_info("request_start", {"path": request.url.path, "method": request.method})
        except Exception:
            pass
        response = await call_next(request)
        duration_ms = int((time.time() - start) * 1000)
        response.headers["X-Request-ID"] = req_id
        try:
            log_info("request_end", {"path": request.url.path, "status": response.status_code, "duration_ms": duration_ms})
        except Exception:
            pass
        try:
            inc_counter("http_requests_total", labels={"path": request.url.path, "method": request.method, "status": str(response.status_code)})
            observe_histogram("http_request_latency_ms", value_ms=duration_ms, labels={"path": request.url.path, "method": request.method})
        except Exception:
            pass
        return response

    @app.on_event("startup")
    def on_startup() -> None:
        try:
            init_db()
        except Exception:
            try:
                log_info("init_db_failed", {"message": "continuing in degraded mode"})
            except Exception:
                pass

    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(query_router)
    app.include_router(datasets_router)
    app.include_router(datasets_alias_router)
    app.include_router(review_router)
    app.include_router(eval_router)
    return app


app = create_app()

