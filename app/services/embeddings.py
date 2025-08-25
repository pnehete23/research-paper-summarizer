from __future__ import annotations

from typing import Iterable, List
import math

from sentence_transformers import SentenceTransformer

from ..models import EMBEDDING_DIMENSIONS
from ..services.observability import log_info


_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        # Lightweight, good quality general model; dim may differ, pad/trim as needed
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def embed_texts(texts: Iterable[str]) -> List[list[float]]:
    model = _get_model()
    vectors = model.encode(list(texts), normalize_embeddings=True).tolist()
    # Adjust dimension to EMBEDDING_DIMENSIONS by tiling/trimming and re-normalizing
    adjusted: List[list[float]] = []
    for v in vectors:
        if len(v) == EMBEDDING_DIMENSIONS:
            adjusted_vec = v
        elif len(v) < EMBEDDING_DIMENSIONS:
            # Tile the vector to reach target dimension, then trim
            times = (EMBEDDING_DIMENSIONS + len(v) - 1) // len(v)
            tiled = (v * times)[:EMBEDDING_DIMENSIONS]
            adjusted_vec = tiled
        else:
            adjusted_vec = v[:EMBEDDING_DIMENSIONS]
        # Re-normalize to unit length for cosine distance stability
        norm = math.sqrt(sum(x * x for x in adjusted_vec)) or 1.0
        adjusted.append([x / norm for x in adjusted_vec])
    # Emit a single warning once when tiling occurs (best-effort, no global flag)
    try:
        if vectors and len(vectors[0]) != EMBEDDING_DIMENSIONS:
            log_info("embedding_dim_mismatch", {
                "model_dim": len(vectors[0]),
                "db_dim": EMBEDDING_DIMENSIONS,
                "action": "tiled_or_trimmed",
            })
    except Exception:
        pass
    return adjusted

