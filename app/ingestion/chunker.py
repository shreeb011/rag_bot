"""Structure-aware chunking.

Each section is split into paragraph-sized units (falling back to lines, sentences and finally words for
oversized units), which are then packed greedily into chunks of at most ``chunk_size`` characters.
Consecutive chunks share up to ``chunk_overlap`` characters of trailing units so facts that straddle a
boundary are still retrievable. Chunks never cross section boundaries, so every chunk has exactly one
page number / heading for citation.
"""

from __future__ import annotations

import re

from app.models import Chunk, Section

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_sections(
    sections: list[Section],
    doc_id: str,
    doc_name: str,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    doc_context: str | None = None,
) -> list[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    for section in sections:
        for text in split_text(section.text, chunk_size, chunk_overlap):
            chunks.append(
                Chunk(
                    id=f"{doc_id}:{len(chunks)}",
                    doc_id=doc_id,
                    doc_name=doc_name,
                    chunk_index=len(chunks),
                    text=text,
                    page=section.page,
                    heading=section.heading,
                    doc_context=doc_context if not chunks else None,
                )
            )
    return chunks


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    units = _split_units(text.strip(), chunk_size)
    chunks: list[str] = []
    current: list[str] = []

    def length(parts: list[str]) -> int:
        return sum(len(p) for p in parts) + max(len(parts) - 1, 0)

    for unit in units:
        if current and length(current + [unit]) > chunk_size:
            chunks.append("\n".join(current))
            overlap: list[str] = []
            for previous in reversed(current):
                if length([previous] + overlap) > chunk_overlap:
                    break
                overlap.insert(0, previous)
            while overlap and length(overlap + [unit]) > chunk_size:
                overlap.pop(0)
            current = overlap
        current.append(unit)
    if current:
        chunks.append("\n".join(current))
    return chunks


def _split_units(text: str, chunk_size: int) -> list[str]:
    """Split into the largest natural units (paragraph > line > sentence > words) that fit the chunk size."""
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
            continue
        for line in paragraph.split("\n"):
            line = line.strip()
            if not line:
                continue
            if len(line) <= chunk_size:
                units.append(line)
                continue
            for sentence in _SENTENCE_RE.split(line):
                if len(sentence) <= chunk_size:
                    units.append(sentence)
                else:
                    units.extend(_split_words(sentence, chunk_size))
    return units


def _split_words(text: str, chunk_size: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for word in text.split():
        while len(word) > chunk_size:  # pathological token (e.g. a base64 blob)
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:chunk_size])
            word = word[chunk_size:]
        candidate = f"{current} {word}".strip()
        if len(candidate) > chunk_size:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces
