from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel

from ..database import session_scope
from ..models import Dataset
from ..services.datasets import profile_csv
from ..services.observability import trace
from . import require_api_key, rate_limiter


router = APIRouter(prefix="/ingest", dependencies=[Depends(require_api_key), Depends(rate_limiter)])


class IngestDatasetResponse(BaseModel):
    datasetId: int
    name: str
    columns: list[str]


@router.post("/dataset", response_model=IngestDatasetResponse)
async def ingest_dataset(file: UploadFile = File(...)) -> IngestDatasetResponse:
    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Expected a CSV file")
    if getattr(file, "size", None) is not None:
        from ..settings import settings
        max_bytes = settings.max_upload_mb * 1024 * 1024
        if file.size > max_bytes:
            raise HTTPException(status_code=413, detail="File too large")
    try:
        data = await file.read()
        name, schema, stats = profile_csv(data, name=file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to profile CSV: {str(e)}")
    with trace("ingest_dataset", {"name": name, "cols": len(schema)}):
        with session_scope() as session:
            ds = Dataset(name=name, schema=schema, stats=stats, file_path=None)
            session.add(ds)
            session.flush()
            return IngestDatasetResponse(datasetId=ds.id, name=ds.name, columns=list(schema.keys()))


# Backwards-compatible alias under /datasets
datasets_router = APIRouter(prefix="/datasets", dependencies=[Depends(require_api_key), Depends(rate_limiter)])


@datasets_router.post("")
async def datasets_post(file: UploadFile = File(...)) -> dict:
    return await ingest_dataset(file)

