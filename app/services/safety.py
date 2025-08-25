from __future__ import annotations

import re

from ..settings import settings


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")


def redact_pii(text: str) -> str:
    if not settings.redact_pii:
        return text
    redacted = EMAIL_RE.sub("[EMAIL]", text)
    redacted = PHONE_RE.sub("[PHONE]", redacted)
    return redacted

