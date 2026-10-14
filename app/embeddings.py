"""Embedding backends. All return L2-normalised float32 matrices so dot product == cosine similarity."""

from __future__ import annotations

import hashlib
from typing import Protocol

import httpx
import numpy as np

from app.config import Settings
from app.errors import UpstreamModelError
from app.text import tokenize


class Embedder(Protocol):
    @property
    def identity(self) -> str:
        """Identifies the embedding space; indexes built with a different identity are incompatible."""
        ...

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalise(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class OpenAICompatibleEmbedder:
    """Calls ``POST {base_url}/embeddings`` — works with OpenAI, Ollama, vLLM, LM Studio, etc."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        query_prefix: str = "",
        document_prefix: str = "",
        batch_size: int = 32,
        timeout_s: float = 60.0,
        keep_alive: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.batch_size = batch_size
        self.keep_alive = keep_alive
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
        )

    @property
    def identity(self) -> str:
        return f"openai:{self.model}:{self.document_prefix!r}"

    def ping(self) -> None:
        """Cheap reachability check for readiness probes (lists models, doesn't run one)."""
        self._client.get("/models", timeout=3.0).raise_for_status()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [self.document_prefix + t for t in texts[start : start + self.batch_size]]
            vectors.extend(self._embed(batch))
        return _normalise(np.array(vectors))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalise(np.array(self._embed([self.query_prefix + text])))[0]

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        payload: dict = {"model": self.model, "input": inputs}
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        try:
            response = self._client.post("/embeddings", json=payload)
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda item: item["index"])
            vectors = [item["embedding"] for item in data]
        except httpx.HTTPError as exc:
            raise UpstreamModelError(f"Embedding request failed: {exc}") from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise UpstreamModelError(f"Malformed embedding response: {exc}") from exc
        if len(vectors) != len(inputs):
            raise UpstreamModelError(f"Expected {len(inputs)} embeddings, got {len(vectors)}")
        return vectors


class HashingEmbedder:
    """Deterministic feature-hashing embedder (word unigrams + bigrams).

    It is lexical, not semantic: use it for tests and offline smoke runs, never as the production default.
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    @property
    def identity(self) -> str:
        return f"hash:{self.dim}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.stack([self._vector(t) for t in texts])) if texts else np.zeros((0, self.dim))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalise(self._vector(text)[None, :])[0]

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text)
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        for feature in features:
            digest = hashlib.md5(feature.encode()).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        return vector


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "hash":
        return HashingEmbedder()
    return OpenAICompatibleEmbedder(
        model=settings.embedding_model,
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key.get_secret_value(),
        query_prefix=settings.embedding_query_prefix,
        document_prefix=settings.embedding_document_prefix,
        batch_size=settings.embedding_batch_size,
        timeout_s=settings.embedding_timeout_s,
        keep_alive=settings.keep_alive or None,
    )
