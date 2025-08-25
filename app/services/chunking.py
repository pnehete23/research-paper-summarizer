from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class Chunk:
    section: str | None
    page: int | None
    text: str


def _split_by_headers(text: str) -> List[tuple[str | None, str]]:
    lines = text.splitlines()
    sections: List[tuple[str | None, str]] = []
    current_header: str | None = None
    buffer: list[str] = []
    for line in lines:
        if line.strip().endswith(":") or (len(line.strip()) < 120 and line.isupper()):
            if buffer:
                sections.append((current_header, "\n".join(buffer).strip()))
                buffer = []
            current_header = line.strip().strip(":")
        else:
            buffer.append(line)
    if buffer:
        sections.append((current_header, "\n".join(buffer).strip()))
    return sections


def _semantic_split(text: str, max_tokens: int = 512, overlap: int = 64) -> list[str]:
    # Approximate tokenization by words; for MVP, a rough heuristic is fine
    words = text.split()
    window = max_tokens
    step = max(1, window - overlap)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        piece = " ".join(words[start:start + window])
        if piece:
            chunks.append(piece)
        if start + window >= len(words):
            break
    return chunks


def chunk_pages(pages: Iterable[tuple[int, str]]) -> list[Chunk]:
    results: list[Chunk] = []
    for page_number, text in pages:
        for header, body in _split_by_headers(text):
            for piece in _semantic_split(body):
                results.append(Chunk(section=header, page=page_number, text=piece))
    return results

