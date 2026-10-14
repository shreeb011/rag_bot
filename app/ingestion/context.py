"""Document-level context for retrieval.

Documents rarely say what kind of document they are: a CV never contains the word "resume", an offer letter
doesn't call itself one. At ingestion the LLM writes a two-line catalogue entry for each document (its type
with synonyms, and who/what it is about). The entry is indexed (embedded + BM25) with the document's first
chunk only, so "the name in the resume" finds the CV's opening, while every other chunk keeps an embedding
of just its own content (adding the entry to all chunks blurred them and lowered retrieval recall in eval).

The entry is only used to *find* chunks. It is never shown to the answering LLM as a source, so generated
text can't become evidence.
"""

from __future__ import annotations

import logging
import re

from app.llm import LLM
from app.models import Section

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You write short catalogue entries that help a search engine find documents.
Use only what the provided text shows. Do not invent names, dates or facts."""

_LINE_RE = re.compile(r"^\s*\**\s*(type|about)\s*\**\s*:\s*\**\s*(.+)$", re.IGNORECASE | re.MULTILINE)
MAX_CONTEXT_CHARS = 400


def build_prompt(filename: str, sections: list[Section], preview_chars: int = 1500) -> str:
    headings = list(dict.fromkeys(s.heading for s in sections if s.heading))[:15]
    preview = "\n\n".join(s.text for s in sections)[:preview_chars]
    return (
        f"File name: {filename}\n"
        + (f"Section headings: {'; '.join(headings)}\n" if headings else "")
        + f"Beginning of the document:\n\"\"\"\n{preview}\n\"\"\"\n\n"
        "Write exactly two lines:\n"
        "Type: <the kind of document, then 3-6 other words people use for that kind of document, comma-separated "
        "(for example: invoice, bill, receipt, payment request)>\n"
        "About: <one sentence: who or what the document is about>"
    )


def parse_context(raw: str) -> str | None:
    fields = {key.lower(): value.strip() for key, value in _LINE_RE.findall(raw)}
    if "about" not in fields:
        return None
    lines = [f"{key.capitalize()}: {fields[key]}" for key in ("type", "about") if fields.get(key)]
    return "\n".join(lines)[:MAX_CONTEXT_CHARS]


def describe_document(llm: LLM, filename: str, sections: list[Section]) -> str | None:
    """Best effort: returns None (document is indexed without context) if the LLM is unavailable or rambles."""
    try:
        context = parse_context(llm.complete(SYSTEM_PROMPT, build_prompt(filename, sections)))
    except Exception as exc:
        logger.warning("Could not describe %s, indexing without document context: %s", filename, exc)
        return None
    if context is None:
        logger.warning("Unusable document description for %s, indexing without document context", filename)
    return context
