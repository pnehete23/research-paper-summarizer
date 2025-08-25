from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import fitz  # PyMuPDF
import trafilatura


@dataclass
class ParsedPage:
    page_number: int
    text: str


@dataclass
class ParsedDocument:
    title: str
    pages: list[ParsedPage]
    source_url: Optional[str] = None


def parse_pdf_bytes(data: bytes, title: str | None = None) -> ParsedDocument:
    doc = fitz.open(stream=data, filetype="pdf")
    pages: list[ParsedPage] = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append(ParsedPage(page_number=i + 1, text=text))
    return ParsedDocument(title=title or "PDF Document", pages=pages)


def fetch_and_parse_url(url: str) -> ParsedDocument:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError("Failed to fetch URL")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=True) or ""
    title = trafilatura.extract_metadata(downloaded).title if trafilatura.extract_metadata(downloaded) else url
    pages = [ParsedPage(page_number=1, text=text)]
    return ParsedDocument(title=title, pages=pages, source_url=url)

