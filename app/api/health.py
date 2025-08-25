from fastapi import APIRouter, Response

from ..database import db_healthcheck
from ..settings import settings
from ..services.observability import export_metrics


router = APIRouter()


@router.get("/health")
def health() -> dict:
    db_ok = db_healthcheck()
    return {"status": "ok" if db_ok else "degraded", "db": db_ok, "app": settings.app_name}


@router.get("/metrics")
def metrics() -> Response:
    content = export_metrics()
    return Response(content=content, media_type="text/plain; version=0.0.4")

