"""Application settings, loaded from environment variables (prefix ``RAG_``) or a ``.env`` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    # Storage
    data_dir: Path = Path("data")

    # Embeddings. "openai" means any OpenAI-compatible /embeddings endpoint (OpenAI, Ollama, vLLM, ...).
    # "hash" is a dependency-free lexical embedder used for tests and offline smoke runs only.
    embedding_provider: Literal["openai", "hash"] = "openai"
    embedding_model: str = "nomic-embed-text"
    embedding_base_url: str = "http://localhost:11434/v1"
    embedding_api_key: SecretStr = SecretStr("ollama")
    # nomic-embed-text is trained with task prefixes; set both to "" for OpenAI embedding models.
    embedding_query_prefix: str = "search_query: "
    embedding_document_prefix: str = "search_document: "
    embedding_batch_size: int = Field(32, ge=1)
    embedding_timeout_s: float = 60.0

    # LLM (any OpenAI-compatible /chat/completions endpoint)
    llm_model: str = "qwen3:4b-instruct-2507-q4_K_M"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: SecretStr = SecretStr("ollama")
    llm_temperature: float = 0.0
    llm_max_tokens: int = 350
    llm_timeout_s: float = 180.0

    # Speed
    # Ollama-only: how long models stay loaded after a request (avoids multi-second reloads).
    # Set to "" for providers that reject unknown parameters (e.g. OpenAI).
    keep_alive: str = "30m"
    warmup: bool = True  # load the embedding model and LLM in the background at startup
    answer_cache_size: int = Field(256, ge=0)  # identical questions on an unchanged index are answered instantly

    # Chunking (characters)
    chunk_size: int = Field(500, ge=200)
    chunk_overlap: int = Field(80, ge=0)
    # At ingestion, have the LLM describe each document (type + synonyms, subject) and index that with its
    # first chunk, so questions can refer to a document by what it is ("the resume", "the offer letter").
    document_context: bool = True

    # Retrieval
    top_k: int = Field(6, ge=1, le=20)
    candidate_k: int = Field(20, ge=1)
    hybrid_search: bool = True
    # Also pass the first chunk (title, names, contact details) of up to this many of the retrieved documents.
    document_lead_chunks: int = Field(1, ge=0, le=5)
    # Questions whose best dense cosine similarity is below this are answered as "unavailable"
    # without calling the LLM. Calibrated for nomic-embed-text on the sample corpus (see README).
    min_relevance: float = Field(0.55, ge=-1.0, le=1.0)

    # Upload limits. Indexing time grows with the number of chunks, not the file size (images in a PDF cost
    # nothing, a text-only CSV of the same size can be 100k chunks), so both are capped.
    max_upload_mb: float = Field(50.0, gt=0)
    max_chunks_per_document: int = Field(20000, ge=1)

    # Logging: "text" for humans, "json" (one object per line, with request IDs) for log shippers.
    log_format: Literal["text", "json"] = "text"
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
