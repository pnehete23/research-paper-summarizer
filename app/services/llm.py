from __future__ import annotations

from typing import List, Optional, Iterable
import time

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore
try:
    from anthropic import Anthropic
except Exception:  # pragma: no cover
    Anthropic = None  # type: ignore

from ..settings import settings
from typing import Tuple
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception


def _estimate_tokens(text: str) -> int:
    # Rough heuristic: ~1 token per word
    return max(1, len(text.split()))


def _trim_contexts(contexts: List[dict], max_context_tokens: int, per_snippet_cap: int = 300) -> Tuple[List[dict], int]:
    trimmed: List[dict] = []
    used = 0
    for c in contexts:
        snippet = c.get("text", "")
        words = snippet.split()
        if len(words) > per_snippet_cap:
            snippet = " ".join(words[:per_snippet_cap])
        cost = _estimate_tokens(snippet) + 12  # header overhead
        if used + cost > max_context_tokens:
            break
        used += cost
        trimmed.append({"docId": c.get("docId"), "page": c.get("page"), "text": snippet})
    return trimmed, used


def _should_retry_exception(exc: Exception) -> bool:
    """Return True if the exception looks transient (overload/timeout)."""
    msg = str(exc).lower()
    return (
        "overload" in msg
        or "overloaded" in msg
        or "timeout" in msg
        or "timed out" in msg
        or "temporarily unavailable" in msg
        or " 529" in msg
        or " 503" in msg
    )


@retry(reraise=True, stop=stop_after_attempt(4), wait=wait_exponential_jitter(initial=0.5, max=6.0), retry=retry_if_exception(_should_retry_exception))
def _anthropic_create_with_retry(client: "Anthropic", **kwargs):
    return client.messages.create(**kwargs)


def generate_answer(question: str, contexts: List[dict], model: Optional[str] = None) -> dict:
    """Generate an answer using the configured LLM provider.

    Provider/model selection precedence:
    - If `model` includes a provider prefix like "anthropic:claude-3-5-sonnet", it overrides settings.
    - Else provider comes from `settings.prefer_llm` prefix, and model is the remainder or `model` if provided.
    """

    # Resolve provider and model
    if ":" in (settings.prefer_llm or ""):
        default_provider, default_model = settings.prefer_llm.split(":", 1)
    else:
        default_provider, default_model = "anthropic", (settings.prefer_llm or "claude-3-5-sonnet-latest")

    provider = default_provider
    chosen = model or default_model
    if model and ":" in model:
        provider, chosen = model.split(":", 1)

    # Trim contexts to stay within prompt budget
    budget = max(512, settings.max_context_tokens)
    trimmed_contexts, _ = _trim_contexts(contexts, budget)
    prompt = _build_prompt(question, trimmed_contexts)
    system_text = (
        "You are a careful research assistant.\n"
        "- Use ONLY the provided context.\n"
        "- If the answer is not in context, say: 'I don't know based on the provided sources.'\n"
        "- Cite evidence inline as [n] tied to the numbered sources. Include page numbers if present.\n"
        "- End with a short References list enumerating sources [n]: docId and page.\n"
        "- Be concise and factual."
    )

    # Route to provider with graceful fallbacks
    if provider == "anthropic":
        if not settings.anthropic_api_key or Anthropic is None:
            return {
                "answer": "[offline mode] Provide your ANTHROPIC_API_KEY to enable generation.",
                "citations": contexts[:3],
                "model": None,
                "usage": None,
            }
        try:
            client = Anthropic(api_key=settings.anthropic_api_key, timeout=settings.llm_timeout_s)
            response = _anthropic_create_with_retry(
                client,
                model=chosen,
                system=system_text,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=max(1, min(settings.max_output_tokens, 4000)),
            )
            parts = getattr(response, "content", [])
            text = "".join(
                getattr(p, "text", "") for p in parts if getattr(p, "type", None) == "text"
            )
            usage = getattr(response, "usage", None)
            usage_dict = (
                {
                    "prompt_tokens": getattr(usage, "input_tokens", None),
                    "completion_tokens": getattr(usage, "output_tokens", None),
                    "total_tokens": (
                        (getattr(usage, "input_tokens", 0) or 0)
                        + (getattr(usage, "output_tokens", 0) or 0)
                    ),
                }
                if usage is not None
                else None
            )
            citations = [
                {"index": i + 1, "docId": c.get("docId"), "page": c.get("page")}
                for i, c in enumerate(trimmed_contexts[: min(6, len(trimmed_contexts))])
            ]
            return {
                "answer": text,
                "citations": citations,
                "model": chosen,
                "usage": usage_dict,
            }
        except Exception as e:
            # Try OpenAI as secondary if available
            if settings.openai_api_key and OpenAI is not None:
                try:
                    client = OpenAI(api_key=settings.openai_api_key)
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_text},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.2,
                    )
                    text = response.choices[0].message.content
                    usage = getattr(response, "usage", None)
                    usage_dict = (
                        {
                            "prompt_tokens": getattr(usage, "prompt_tokens", None),
                            "completion_tokens": getattr(usage, "completion_tokens", None),
                            "total_tokens": getattr(usage, "total_tokens", None),
                        }
                        if usage is not None
                        else None
                    )
                    citations = [
                        {"index": i + 1, "docId": c.get("docId"), "page": c.get("page")}
                        for i, c in enumerate(trimmed_contexts[: min(6, len(trimmed_contexts))])
                    ]
                    return {
                        "answer": text,
                        "citations": citations,
                        "model": "openai:gpt-4o-mini",
                        "usage": usage_dict,
                    }
                except Exception:
                    pass
            # Final offline message
            return {
                "answer": f"[llm unavailable] Provider error: {str(e)}",
                "citations": contexts[:3],
                "model": None,
                "usage": None,
            }
    elif provider == "openai":
        if not settings.openai_api_key or OpenAI is None:
            return {
                "answer": "[offline mode] Provide your OPENAI_API_KEY to enable generation.",
                "citations": contexts[:3],
                "model": None,
                "usage": None,
            }
        try:
            client = OpenAI(api_key=settings.openai_api_key)
            response = client.chat.completions.create(
                model=chosen,
                messages=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            text = response.choices[0].message.content
            usage = getattr(response, "usage", None)
            usage_dict = (
                {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
                if usage is not None
                else None
            )
            citations = [
                {"index": i + 1, "docId": c.get("docId"), "page": c.get("page")}
                for i, c in enumerate(trimmed_contexts[: min(6, len(trimmed_contexts))])
            ]
            return {
                "answer": text,
                "citations": citations,
                "model": chosen,
                "usage": usage_dict,
            }
        except Exception as e:
            return {
                "answer": f"[llm unavailable] Provider error: {str(e)}",
                "citations": contexts[:3],
                "model": None,
                "usage": None,
            }
    else:
        return {
            "answer": f"[offline mode] Unsupported LLM provider: {provider}",
            "citations": contexts[:3],
            "model": None,
            "usage": None,
        }


def generate_answer_stream(question: str, contexts: List[dict], model: Optional[str] = None) -> Iterable[str]:
    """Stream an answer using the configured LLM provider. Falls back to non-streaming if unsupported."""

    # Resolve provider and model
    if ":" in (settings.prefer_llm or ""):
        default_provider, default_model = settings.prefer_llm.split(":", 1)
    else:
        default_provider, default_model = "anthropic", (settings.prefer_llm or "claude-3-5-sonnet-latest")
    provider = default_provider
    chosen = model or default_model
    if model and ":" in model:
        provider, chosen = model.split(":", 1)

    budget = max(512, settings.max_context_tokens)
    trimmed_contexts, _ = _trim_contexts(contexts, budget)
    prompt = _build_prompt(question, trimmed_contexts)
    system_text = (
        "You are a careful research assistant.\n"
        "- Use ONLY the provided context.\n"
        "- If the answer is not in context, say: 'I don't know based on the provided sources.'\n"
        "- Cite evidence inline as [n] tied to the numbered sources. Include page numbers if present.\n"
        "- End with a short References list enumerating sources [n]: docId and page.\n"
        "- Be concise and factual."
    )

    if provider == "openai":
        if not settings.openai_api_key or OpenAI is None:
            yield "[offline mode] Provide your OPENAI_API_KEY to enable streaming."
            return
        client = OpenAI(api_key=settings.openai_api_key)
        stream = client.chat.completions.create(
            model=chosen,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            stream=True,
        )
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content  # type: ignore[attr-defined]
            except Exception:
                delta = None
            if delta:
                yield delta
        return

    elif provider == "anthropic":
        if not settings.anthropic_api_key or Anthropic is None:
            yield "[offline mode] Provide your ANTHROPIC_API_KEY to enable streaming."
            return
        client = Anthropic(api_key=settings.anthropic_api_key)
        try:
            # Streaming with retries is complex; attempt once, then fallback to non-streaming with retry
            with client.messages.stream(
                model=chosen,
                system=system_text,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=max(1, min(settings.max_output_tokens, 4000)),
            ) as stream:
                for event in stream:
                    try:
                        if getattr(event, "type", None) == "content_block_delta":
                            delta = getattr(event, "delta", None)
                            if delta is not None and getattr(delta, "type", None) == "text_delta":
                                text_piece = getattr(delta, "text", None)
                                if text_piece:
                                    yield text_piece
                    except Exception:
                        continue
        except Exception:
            # Fallback to non-streaming with retry/backoff
            response = _anthropic_create_with_retry(
                client,
                model=chosen,
                system=system_text,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=max(1, min(settings.max_output_tokens, 4000)),
            )
            parts = getattr(response, "content", [])
            text = "".join(getattr(p, "text", "") for p in parts if getattr(p, "type", None) == "text")
            step = 512
            for i in range(0, len(text), step):
                yield text[i:i + step]
        return

    else:
        yield "[offline mode] Unsupported LLM provider for streaming."
        return


def _build_prompt(question: str, contexts: List[dict]) -> str:
    blocks = []
    for i, c in enumerate(contexts, start=1):
        header = f"[Source {i}] docId={c.get('docId')} page={c.get('page')}"
        snippet = c.get("text", "")
        blocks.append(f"{header}\n{snippet}")
    joined = "\n\n".join(blocks)
    return f"Question: {question}\n\nContext:\n{joined}\n\nInstructions: Answer concisely with citations like [1], [2] tied to sources above."

