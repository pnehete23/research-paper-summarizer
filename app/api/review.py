from __future__ import annotations

from typing import Optional, List

from fastapi import APIRouter, Body, Depends
from pydantic import BaseModel

from ..database import session_scope
from ..models import Document, Chunk
from ..services.retrieval import search_hybrid
from ..services.llm import generate_answer
from . import require_api_key, rate_limiter


router = APIRouter(dependencies=[Depends(require_api_key), Depends(rate_limiter)])


class ReviewRequest(BaseModel):
    topic: str
    targetWords: int = 800
    k: int = 10


class ReviewResponse(BaseModel):
    markdown: str
    references: List[dict]


@router.post("/review/generate", response_model=ReviewResponse)
async def generate_review(payload: ReviewRequest = Body(...)) -> ReviewResponse:
    topic: str = payload.topic
    target_words: int = int(payload.targetWords)
    k: int = int(payload.k)
    if not topic:
        return {"error": "topic is required"}
    with session_scope() as session:
        chunks = search_hybrid(session, topic, k=k * 8)
        contexts = [
            {"docId": c.document_id, "page": c.page, "text": c.text}
            for c in chunks[:k]
        ]
    system = f"Draft a literature review (~{target_words} words), with inline citations [n] tied to sources. Include a brief references list."
    result = generate_answer(system + "\nQuestion: " + topic, contexts)
    return ReviewResponse(
        markdown=result.get("answer", ""),
        references=[{"docId": c["docId"], "page": c.get("page")} for c in contexts],
    )

