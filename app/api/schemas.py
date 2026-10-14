from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, examples=["How many days of annual leave do I get?"])
    top_k: int | None = Field(None, ge=1, le=20, description="Number of chunks to retrieve (defaults to RAG_TOP_K).")
    history: list[Annotated[str, Field(max_length=2000)]] = Field(
        default_factory=list,
        max_length=10,
        description="Earlier questions in this conversation, oldest first. Used to resolve follow-ups such as "
        "'what is his email?'.",
    )

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if len(value.strip()) < 3:
            raise ValueError("question must contain at least 3 non-whitespace characters")
        return value.strip()


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=50)


class SourceOut(BaseModel):
    id: str = Field(description="Citation id used in the answer text, e.g. S1")
    document: str
    doc_id: str
    chunk_id: str
    page: int | None
    section: str | None
    excerpt: str = Field(description="Verbatim excerpt from the source chunk")
    score: float = Field(description="Cosine similarity between question and chunk")


class QueryResponse(BaseModel):
    question: str
    answer: str
    status: Literal["answered", "unavailable"]
    reason: Literal["no_documents", "low_relevance", "not_in_documents", "ungrounded_answer"] | None = None
    sources: list[SourceOut]
    warnings: list[str]
    max_similarity: float | None
    latency_ms: int
    timings: dict[str, int] = Field(
        default_factory=dict, description="Milliseconds per stage: retrieval_ms, generation_ms, first_token_ms."
    )
    cached: bool = Field(False, description="Served from the answer cache (same question, unchanged index).")


class SearchHit(BaseModel):
    document: str
    doc_id: str
    chunk_id: str
    page: int | None
    section: str | None
    text: str
    score: float
    dense_score: float
    lexical_score: float


class SearchResponse(BaseModel):
    query: str
    max_similarity: float
    hits: list[SearchHit]


class DocumentOut(BaseModel):
    doc_id: str
    name: str
    content_hash: str
    num_chunks: int
    ingested_at: str


class IngestItem(BaseModel):
    status: Literal["created", "updated", "unchanged"]
    document: DocumentOut


class IngestResponse(BaseModel):
    results: list[IngestItem]


class JobOut(BaseModel):
    id: str
    workspace: str
    filename: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    stage: str | None = Field(None, description="parsing | describing | embedding | saving, while running")
    progress: float | None = Field(None, description="Fraction of chunks embedded (0-1) during 'embedding'")
    cancel_requested: bool = False
    result: Literal["created", "updated", "unchanged"] | None = None
    document: DocumentOut | None = None
    error: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None


class JobsResponse(BaseModel):
    jobs: list[JobOut]


class WorkspaceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=40, examples=["hr-team"])


class WorkspaceOut(BaseModel):
    name: str
    documents: int
    chunks: int


class HealthResponse(BaseModel):
    status: str
    documents: int
    chunks: int
    workspaces: int
    ingest_queue: int
    max_upload_mb: float
    max_chunks_per_document: int
    embedding_model: str
    llm_model: str


class ReadinessResponse(BaseModel):
    ready: bool
    checks: dict[str, str]
