from __future__ import annotations

from typing import List, Dict

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from ..models import Chunk
from .embeddings import embed_texts
from .observability import trace, log_info
from ..settings import settings
try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover
    BM25Okapi = None  # type: ignore


def search_hybrid(session: Session, query: str, k: int = 50) -> List[Chunk]:
    # Vector search using pgvector
    query_emb = embed_texts([query])[0]
    vec_rows = []
    with trace("retrieval.vector", {"k": k}):
        try:
            # Tune ANN probes to balance recall/perf (SESSION-local)
            try:
                session.execute(text("SET LOCAL ivfflat.probes = :p"), {"p": max(1, min(16, k // 5 or 1))})
            except Exception:
                pass
            stmt_vec = text(
                """
                SELECT id, document_id, section, page, text, tokens, created_at,
                       (embedding <=> :qv) AS distance
                FROM chunks
                ORDER BY embedding <=> :qv
                LIMIT :k
                """
            )
            vec_rows = session.execute(stmt_vec, {"qv": query_emb, "k": k}).mappings().all()
        except Exception:
            # If pgvector/ANN errors (e.g., extension/index not ready), fall back gracefully
            try:
                session.rollback()
            except Exception:
                pass
            # Fallback: lexical only
            vec_rows = []
    vec_rank: Dict[int, float] = {r["id"]: float(r["distance"]) for r in vec_rows}

    # Lexical search (configurable)
    lex_rank: Dict[int, float] = {}
    method = (settings.lexical_method or "trgm").lower()
    with trace("retrieval.lexical", {"k": k, "method": method}):
        if method == "trgm":
            try:
                stmt_lex = text(
                    """
                    SELECT id, similarity(text, :q) AS sim
                    FROM chunks
                    WHERE similarity(text, :q) > 0.1
                    ORDER BY sim DESC
                    LIMIT :k
                    """
                )
                rows = session.execute(stmt_lex, {"q": query, "k": k}).mappings().all()
                lex_rank = {r["id"]: float(r["sim"]) for r in rows}
            except Exception:
                lex_rank = {}
        elif method == "tsvector":
            try:
                stmt_ts = text(
                    """
                    SELECT id,
                           ts_rank_cd(to_tsvector('english', text), plainto_tsquery('english', :q)) AS rank
                    FROM chunks
                    WHERE to_tsvector('english', text) @@ plainto_tsquery('english', :q)
                    ORDER BY rank DESC
                    LIMIT :k
                    """
                )
                rows = session.execute(stmt_ts, {"q": query, "k": k}).mappings().all()
                lex_rank = {r["id"]: float(r["rank"]) for r in rows}
            except Exception:
                lex_rank = {}
        elif method == "bm25":
            try:
                # Pull a candidate pool and score in-memory
                candidates = session.execute(
                    text("SELECT id, text FROM chunks ORDER BY created_at DESC LIMIT :n"),
                    {"n": max(k * 20, 200)}
                ).mappings().all()
                if candidates and BM25Okapi is not None:
                    corpus = [c["text"].split() for c in candidates]
                    bm25 = BM25Okapi(corpus)
                    scores = bm25.get_scores(query.split())
                    ids = [c["id"] for c in candidates]
                    pairs = list(zip(ids, scores))
                    pairs.sort(key=lambda x: x[1], reverse=True)
                    lex_rank = {i: float(s) for i, s in pairs[:k]}
                else:
                    lex_rank = {}
            except Exception:
                lex_rank = {}
        else:
            lex_rank = {}

    # Merge scores: lower distance is better, higher sim is better
    ids = set(vec_rank.keys()) | set(lex_rank.keys())
    if not ids:
        return []
    # Normalize and combine
    if vec_rank:
        max_d = max(vec_rank.values())
        min_d = min(vec_rank.values())
    else:
        max_d = 1.0
        min_d = 0.0
    def norm_d(x: float) -> float:
        return 0.0 if max_d == min_d else (x - min_d) / (max_d - min_d)

    max_s = max(lex_rank.values()) if lex_rank else 1.0
    def norm_s(x: float) -> float:
        return x / max_s if max_s else 0.0

    combined = []
    for id_ in ids:
        d = norm_d(vec_rank.get(id_, max_d))  # 0..1 (lower is better)
        s = norm_s(lex_rank.get(id_, 0.0))    # 0..1 (higher is better)
        # configurable weights
        vw = max(0.0, min(1.0, settings.vec_weight))
        lw = max(0.0, min(1.0, settings.lex_weight))
        wsum = vw + lw if (vw + lw) > 0 else 1.0
        vw = vw / wsum
        lw = lw / wsum
        score = (1.0 - d) * vw + s * lw
        combined.append((id_, score))
    combined.sort(key=lambda x: x[1], reverse=True)
    top_ids = [i for i, _ in combined[:k]]
    objs = session.scalars(select(Chunk).where(Chunk.id.in_(top_ids))).all()
    # Preserve the combined order
    id_to_obj = {c.id: c for c in objs}
    results = [id_to_obj[i] for i in top_ids if i in id_to_obj]
    try:
        log_info("retrieval_merge", {
            "k": k,
            "vec_weight": settings.vec_weight,
            "lex_weight": settings.lex_weight,
            "vec_candidates": len(vec_rank),
            "lex_candidates": len(lex_rank),
            "returned": len(results),
        })
    except Exception:
        pass
    return results

