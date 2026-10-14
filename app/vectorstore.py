"""A small persistent vector store backed by NumPy.

Exact (brute-force) cosine search over a dense matrix is simple, has zero infrastructure, and is fast
for the scale of this assistant (a 50k x 768 float32 matrix is ~150 MB and searches in milliseconds).
The interface is deliberately narrow so it can be swapped for Qdrant/pgvector/Chroma later.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np

from app.errors import IndexCompatibilityError
from app.models import Chunk, DocumentInfo

_META_FILE = "index.json"
_VECTORS_FILE = "vectors.npy"


class VectorStore:
    def __init__(self, directory: Path | None, embedder_identity: str) -> None:
        self.directory = directory
        self.embedder_identity = embedder_identity
        self._lock = threading.RLock()
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None
        self._documents: dict[str, DocumentInfo] = {}
        self.version = 0  # bumped on every mutation; lets derived indexes (BM25) know when to rebuild
        if directory is not None:
            self._load()

    # ----------------------------------------------------------------- queries
    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    def documents(self) -> list[DocumentInfo]:
        with self._lock:
            return sorted(self._documents.values(), key=lambda d: d.name.lower())

    def get_document(self, doc_id: str) -> DocumentInfo | None:
        return self._documents.get(doc_id)

    def find_by_name(self, name: str) -> DocumentInfo | None:
        with self._lock:
            return next((d for d in self._documents.values() if d.name == name), None)

    def chunks(self) -> list[Chunk]:
        with self._lock:
            return list(self._chunks)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        with self._lock:
            if self._vectors is None or not self._chunks:
                return []
            scores = self._vectors @ np.asarray(query_vector, dtype=np.float32)
            k = min(k, len(scores))
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top])]
            return [(self._chunks[i], float(scores[i])) for i in top]

    def similarities(self, query_vector: np.ndarray) -> dict[str, float]:
        """Cosine similarity of the query against every chunk, keyed by chunk id."""
        with self._lock:
            if self._vectors is None:
                return {}
            scores = self._vectors @ np.asarray(query_vector, dtype=np.float32)
            return {chunk.id: float(s) for chunk, s in zip(self._chunks, scores)}

    # --------------------------------------------------------------- mutations
    def add_document(self, info: DocumentInfo, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        vectors = np.asarray(vectors, dtype=np.float32)
        with self._lock:
            if self._vectors is not None and vectors.shape[1] != self._vectors.shape[1]:
                raise IndexCompatibilityError(
                    f"Embedding dimension {vectors.shape[1]} does not match index dimension {self._vectors.shape[1]}"
                )
            self._remove(info.doc_id)
            self._chunks.extend(chunks)
            self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])
            self._documents[info.doc_id] = info
            self._commit()

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id not in self._documents:
                return False
            self._remove(doc_id)
            self._commit()
            return True

    def _remove(self, doc_id: str) -> None:
        if doc_id not in self._documents:
            return
        keep = [i for i, c in enumerate(self._chunks) if c.doc_id != doc_id]
        self._chunks = [self._chunks[i] for i in keep]
        self._vectors = self._vectors[keep] if self._vectors is not None and keep else None
        del self._documents[doc_id]

    def _commit(self) -> None:
        self.version += 1
        self._persist()

    # ------------------------------------------------------------- persistence
    def _persist(self) -> None:
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "embedder": self.embedder_identity,
            "documents": [d.to_dict() for d in self._documents.values()],
            "chunks": [c.to_dict() for c in self._chunks],
        }
        vectors = self._vectors if self._vectors is not None else np.zeros((0, 0), dtype=np.float32)
        # Write to temp files then atomically replace, so a crash never leaves a half-written index.
        tmp_vectors = self.directory / (_VECTORS_FILE + ".tmp")
        with open(tmp_vectors, "wb") as fh:
            np.save(fh, vectors)
        tmp_meta = self.directory / (_META_FILE + ".tmp")
        tmp_meta.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_vectors, self.directory / _VECTORS_FILE)
        os.replace(tmp_meta, self.directory / _META_FILE)

    def _load(self) -> None:
        meta_path = self.directory / _META_FILE
        vectors_path = self.directory / _VECTORS_FILE
        if not meta_path.exists() or not vectors_path.exists():
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("embedder") != self.embedder_identity:
            raise IndexCompatibilityError(
                f"Index in '{self.directory}' was built with embedder '{meta.get('embedder')}', but the current "
                f"embedder is '{self.embedder_identity}'. Delete the directory and re-ingest your documents."
            )
        self._documents = {d["doc_id"]: DocumentInfo(**d) for d in meta["documents"]}
        self._chunks = [Chunk(**c) for c in meta["chunks"]]
        vectors = np.load(vectors_path)
        self._vectors = vectors if len(self._chunks) else None
        if self._vectors is not None and len(self._vectors) != len(self._chunks):
            raise IndexCompatibilityError(f"Index in '{self.directory}' is corrupt (chunk/vector count mismatch).")
