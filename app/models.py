"""Core domain types shared across ingestion, retrieval and answering."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Section:
    """A contiguous piece of a source document (a PDF page, a Markdown section, a whole TXT file)."""

    text: str
    page: int | None = None
    heading: str | None = None


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    doc_name: str
    chunk_index: int
    text: str
    page: int | None = None
    heading: str | None = None
    # Generated document description, set on a document's first chunk only. Indexed for retrieval, never
    # shown to the answering LLM as evidence.
    doc_context: str | None = None

    def embedding_text(self) -> str:
        """Text that is embedded/indexed: the chunk prefixed with its document and section context."""
        location = self.doc_name if not self.heading else f"{self.doc_name} > {self.heading}"
        prefix = f"{location}\n{self.doc_context}" if self.doc_context else location
        return f"{prefix}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DocumentInfo:
    doc_id: str
    name: str
    content_hash: str
    num_chunks: int
    ingested_at: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # fused ranking score (RRF when hybrid, cosine otherwise)
    dense_score: float  # cosine similarity between query and chunk
    lexical_score: float = 0.0  # BM25 score (0 when not matched / hybrid disabled)


@dataclass
class Source:
    id: str
    document: str
    doc_id: str
    chunk_id: str
    page: int | None
    section: str | None
    excerpt: str
    score: float


@dataclass
class Answer:
    question: str
    answer: str
    status: str  # "answered" | "unavailable"
    reason: str | None = None
    sources: list[Source] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    max_similarity: float | None = None
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    timings: dict[str, int] = field(default_factory=dict)  # milliseconds per pipeline stage
    cached: bool = False
