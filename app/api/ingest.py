from __future__ import annotations

from typing import Optional, List

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi import Path
from fastapi import Body
from pydantic import BaseModel

from ..database import session_scope
from ..models import Document, Chunk
from ..services.parsing import parse_pdf_bytes, fetch_and_parse_url
from ..services.chunking import chunk_pages
from ..services.embeddings import embed_texts
from ..services.observability import trace, inc_counter
from . import require_api_key, rate_limiter


router = APIRouter(prefix="/ingest", dependencies=[Depends(require_api_key), Depends(rate_limiter)])


class IngestPdfResponse(BaseModel):
    documentId: int
    pages: int
    chunks: int


class IngestUrlRequest(BaseModel):
    url: str


class IngestUrlResponse(BaseModel):
    documentId: int
    pages: int
    chunks: int


class DocumentResponse(BaseModel):
    id: int
    title: str
    source_url: Optional[str] = None
    meta: Optional[dict] = None
    created_at: str


@router.post("/pdf", response_model=IngestPdfResponse)
async def ingest_pdf(file: UploadFile = File(...)) -> IngestPdfResponse:
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Expected a PDF file")
    # Size cap
    if getattr(file, "size", None) is not None:
        from ..settings import settings
        max_bytes = settings.max_upload_mb * 1024 * 1024
        if file.size > max_bytes:
            raise HTTPException(status_code=413, detail="File too large")
    try:
        data = await file.read()
        parsed = parse_pdf_bytes(data, title=file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {str(e)}")

    with trace("ingest_pdf", {"pages": len(parsed.pages)}):
        with session_scope() as session:
            doc = Document(title=parsed.title, source_url=None, meta={})
            session.add(doc)
            session.flush()

            pages = [(p.page_number, p.text) for p in parsed.pages]
            chunks = chunk_pages(pages)
            vectors = embed_texts([c.text for c in chunks])
            for c, v in zip(chunks, vectors):
                session.add(Chunk(document_id=doc.id, section=c.section, page=c.page, text=c.text, embedding=v, tokens=len(c.text.split())))

            inc_counter("ingest_pdf_total")
            return IngestPdfResponse(documentId=doc.id, pages=len(parsed.pages), chunks=len(chunks))


@router.post("/url", response_model=IngestUrlResponse)
async def ingest_url(payload: IngestUrlRequest = Body(...)) -> IngestUrlResponse:
    url: str = payload.url
    try:
        parsed = fetch_and_parse_url(url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch or parse URL: {str(e)}")

    with trace("ingest_url"):
        with session_scope() as session:
            doc = Document(title=parsed.title, source_url=url, meta={})
            session.add(doc)
            session.flush()

            pages = [(p.page_number, p.text) for p in parsed.pages]
            chunks = chunk_pages(pages)
            vectors = embed_texts([c.text for c in chunks])
            for c, v in zip(chunks, vectors):
                session.add(Chunk(document_id=doc.id, section=c.section, page=c.page, text=c.text, embedding=v, tokens=len(c.text.split())))

            inc_counter("ingest_url_total")
            return IngestUrlResponse(documentId=doc.id, pages=len(parsed.pages), chunks=len(chunks))


@router.get("/docs/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: int = Path(..., ge=1)) -> DocumentResponse:
    with session_scope() as session:
        doc = session.get(Document, doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return DocumentResponse(
            id=doc.id,
            title=doc.title,
            source_url=doc.source_url,
            meta=doc.meta,
            created_at=doc.created_at.isoformat(),
        )

