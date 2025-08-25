from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Body
from pydantic import BaseModel

from ..database import session_scope
from ..models import EvalRun, Document, Chunk
from ..services.retrieval import search_hybrid
from ..services.embeddings import embed_texts
from ..services.llm import generate_answer
from ..services.observability import trace, log_info
from . import require_api_key

try:
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas import evaluate
    from datasets import Dataset as HFDataset
except Exception:  # pragma: no cover
    faithfulness = answer_relevancy = context_precision = context_recall = None  # type: ignore
    evaluate = None  # type: ignore
    HFDataset = None  # type: ignore


router = APIRouter(dependencies=[Depends(require_api_key)])


# --- Tiny curated corpus and QA set for demo ---
_SAMPLE_DOCS = [
    {
        "title": "Claude Overview",
        "text": (
            "Claude is a family of large language models by Anthropic. "
            "Claude 3.5 Sonnet excels at reasoning and tool use."
        ),
    },
    {
        "title": "PgVector Notes",
        "text": (
            "pgvector is a PostgreSQL extension for vector similarity search. "
            "It supports ivfflat and HNSW indexing in recent versions."
        ),
    },
    {
        "title": "Hybrid Retrieval",
        "text": (
            "Hybrid retrieval combines vector search with lexical signals like trigram similarity or BM25. "
            "Weights can be tuned to balance recall and precision."
        ),
    },
]

_QA_SET = [
    {
        "question": "Which Claude model is strong at reasoning and tool use?",
        "ground_truth": "Claude 3.5 Sonnet",
    },
    {
        "question": "What is pgvector used for?",
        "ground_truth": "vector similarity search in PostgreSQL",
    },
    {
        "question": "What does hybrid retrieval combine?",
        "ground_truth": "vector search and lexical signals",
    },
]


class SeedResponse(BaseModel):
    documents: int
    chunks: int


@router.post("/eval/seed", response_model=SeedResponse)
async def seed_sample_corpus() -> SeedResponse:
    inserted_docs = 0
    inserted_chunks = 0
    with trace("eval_seed"):
        with session_scope() as session:
            for d in _SAMPLE_DOCS:
                doc = Document(title=d["title"], source_url=None, meta={})
                session.add(doc)
                session.flush()
                # Single chunk per doc sufficient for demo; reuse chunker if preferred
                text = d["text"]
                vectors = embed_texts([text])
                session.add(Chunk(document_id=doc.id, section=None, page=None, text=text, embedding=vectors[0], tokens=len(text.split())))
                inserted_docs += 1
                inserted_chunks += 1
    return SeedResponse(documents=inserted_docs, chunks=inserted_chunks)


class EvalRequest(BaseModel):
    k: int = 6
    llm: Optional[str] = None
    save_dir: str = "eval_runs"


class EvalResult(BaseModel):
    runId: int
    size: int
    metrics: dict
    path: Optional[str] = None


@router.post("/eval/run", response_model=EvalResult)
async def run_eval(payload: EvalRequest = Body(default=EvalRequest())) -> EvalResult:
    k = int(payload.k)
    llm = payload.llm
    save_dir = payload.save_dir

    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    jsonl_path = os.path.join(save_dir, f"eval_{ts}.jsonl")
    csv_path = os.path.join(save_dir, f"eval_{ts}.csv")

    rows: List[dict] = []
    metric_summary: dict = {}

    with trace("eval_run", {"k": k}):
        for item in _QA_SET:
            q = item["question"]
            gt = item["ground_truth"]
            with session_scope() as session:
                chunks = search_hybrid(session, q, k=k)
                contexts = [
                    {"docId": c.document_id, "page": c.page, "text": c.text}
                    for c in chunks
                ]
            result = generate_answer(q, contexts, model=llm)
            answer = result.get("answer", "")
            usage = result.get("usage") or {}
            # Basic inline metrics
            exact_contains = int(gt.lower() in answer.lower())
            length = len(answer)
            rows.append({
                "question": q,
                "ground_truth": gt,
                "answer": answer,
                "contexts": [c["text"] for c in contexts],
                "model": result.get("model"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "exact_contains": exact_contains,
                "answer_len": length,
            })

        # Optional Ragas if available
        if evaluate and HFDataset and faithfulness is not None:
            try:
                ds = HFDataset.from_list([
                    {
                        "question": r["question"],
                        "answers": [r["ground_truth"]],
                        "contexts": r["contexts"],
                        "response": r["answer"],
                    }
                    for r in rows
                ])
                ragas_res = evaluate(
                    ds,
                    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                )
                metric_summary = {m: float(ragas_res[m]) for m in ragas_res.column_names if m in ragas_res.features}
            except Exception:
                metric_summary = {}

        # Compute deterministic rollups
        if rows:
            acc = sum(r["exact_contains"] for r in rows) / len(rows)
            avg_len = sum(r["answer_len"] for r in rows) / len(rows)
            metric_summary.update({
                "contains_acc": round(acc, 4),
                "avg_answer_len": round(avg_len, 2),
            })

        # Persist logs
        try:
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["question", "answer"])
                writer.writeheader()
                writer.writerows(rows)
        except Exception:
            pass

        with session_scope() as session:
            run = EvalRun(name=f"eval-{ts}", metrics=metric_summary)
            session.add(run)
            session.flush()
            try:
                log_info("eval_completed", {"run_id": run.id, **metric_summary})
            except Exception:
                pass
            return EvalResult(runId=run.id, size=len(rows), metrics=metric_summary, path=save_dir)

