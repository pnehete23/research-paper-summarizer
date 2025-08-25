from __future__ import annotations

from typing import Optional, List, Optional as Opt, Any
import time

from fastapi import APIRouter, Body, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..database import session_scope
from ..services.retrieval import search_hybrid
from ..services.llm import generate_answer
from ..services.safety import redact_pii
from ..services.observability import trace
from ..models import Dataset, QueryLog
from . import require_api_key, rate_limiter
from ..settings import settings


class Citation(BaseModel):
    index: int
    docId: Opt[str | int] = None
    page: Opt[int] = None


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    model: Opt[str] = None
    usage: Opt[dict] = None


router = APIRouter(dependencies=[Depends(require_api_key), Depends(rate_limiter)])


@router.post("/query", response_model=QueryResponse)
async def query(payload: Any = Body(...)) -> dict:
    # Support flexible bodies: string or object
    question: Optional[str]
    k: int = 8
    model: Optional[str] = None
    dataset_id = None
    stream: bool = False
    if isinstance(payload, str):
        question = payload
    elif isinstance(payload, dict):
        question = payload.get("question")
        k = int(payload.get("k", 8))
        model = payload.get("llm")
        dataset_id = payload.get("datasetId")
        stream = bool(payload.get("stream", False))
    else:
        question = None
    if not question:
        return {"error": "question is required"}

    redacted_q = redact_pii(question)

    with trace("query", {"k": k}):
        contexts = []
        try:
            with session_scope() as session:
                if dataset_id:
                    ds = session.get(Dataset, int(dataset_id))
                    if ds:
                        contexts.append({
                            "docId": f"dataset:{ds.id}",
                            "page": None,
                            "text": f"Dataset {ds.name} profile: columns={list((ds.schema or {}).keys())} stats_summary_keys={list((ds.stats or {}).keys())[:10]}",
                        })
                chunks = search_hybrid(session, redacted_q, k=k * 5)
                contexts.extend([
                    {"docId": c.document_id, "page": c.page, "text": c.text}
                    for c in chunks[:k]
                ])
        except Exception:
            contexts = []
        if stream:
            from ..services.llm import generate_answer_stream
            gen = generate_answer_stream(redacted_q, contexts, model=model)
            # Minimal logging for streaming (response/token usage not available)
            with session_scope() as session:
                qlog = QueryLog(
                    question=redacted_q,
                    response=None,
                    citations={"contexts": contexts[: min(6, len(contexts))]},
                    latency_ms=None,
                    token_usage=None,
                    model=(model or settings.prefer_llm),
                )
                session.add(qlog)
            return StreamingResponse(gen, media_type="text/plain")
        else:
            start = time.time()
            result = generate_answer(redacted_q, contexts, model=model)
            latency_ms = int((time.time() - start) * 1000)
            # Persist query log
            try:
                with session_scope() as session:
                    qlog = QueryLog(
                        question=redacted_q,
                        response=result.get("answer"),
                        citations={"contexts": result.get("citations", [])},
                        latency_ms=latency_ms,
                        token_usage=result.get("usage"),
                        model=result.get("model"),
                    )
                    session.add(qlog)
            except Exception:
                pass
            return result


@router.get("/query/stream")
async def query_stream(question: str, k: int = 8, llm: Opt[str] = None, datasetId: Opt[int] = None):
    if not question:
        return {"error": "question is required"}
    redacted_q = redact_pii(question)
    with trace("query_stream", {"k": k}):
        contexts = []
        try:
            with session_scope() as session:
                if datasetId:
                    ds = session.get(Dataset, int(datasetId))
                    if ds:
                        contexts.append({
                            "docId": f"dataset:{ds.id}",
                            "page": None,
                            "text": f"Dataset {ds.name} profile: columns={list((ds.schema or {}).keys())} stats_summary_keys={list((ds.stats or {}).keys())[:10]}",
                        })
                chunks = search_hybrid(session, redacted_q, k=k * 5)
                contexts.extend([
                    {"docId": c.document_id, "page": c.page, "text": c.text}
                    for c in chunks[:k]
                ])
        except Exception:
            contexts = []
        from ..services.llm import generate_answer_stream
        gen = generate_answer_stream(redacted_q, contexts, model=llm)
        return StreamingResponse(gen, media_type="text/plain")
