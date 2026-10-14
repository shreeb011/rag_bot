from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.embeddings import HashingEmbedder
from app.main import create_app
from app.service import RAGService

SAMPLE_DOCS = Path(__file__).resolve().parents[1] / "sample_docs"


class FakeLLM:
    """Scriptable LLM double: returns a fixed string, or the result of a function of the user prompt."""

    def __init__(self, response: str | Callable[[str], str] = "Grounded answer [S1].") -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        return "fake-llm"

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.response(user) if callable(self.response) else self.response

    def stream(self, system: str, user: str):
        text = self.complete(system, user)
        for start in range(0, len(text), 5):
            yield text[start : start + 5]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # The hashing embedder is lexical, so its similarity scale is much lower than a real embedding model's.
    return Settings(
        _env_file=None,
        data_dir=tmp_path,
        embedding_provider="hash",
        min_relevance=0.05,
        chunk_size=500,
        chunk_overlap=80,
        top_k=5,
        document_context=False,
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def service(settings: Settings, fake_llm: FakeLLM) -> RAGService:
    return RAGService(settings, embedder=HashingEmbedder(), llm=fake_llm)


@pytest.fixture
def loaded_service(service: RAGService) -> RAGService:
    service.ingest_directory(SAMPLE_DOCS)
    return service


@pytest.fixture
def client(service: RAGService):
    with TestClient(create_app(service)) as test_client:
        yield test_client
