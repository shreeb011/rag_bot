"""Composition root: wires settings, embedder, LLM, workspaces (vector store + retriever + QA) and the ingestion
queue into one service object used by the API, the CLI and the evaluation.

Workspaces are independent document libraries. Each has its own index and its own folder of original files;
the embedding model and LLM are shared. The "default" workspace lives directly in ``RAG_DATA_DIR`` (so indexes
created before workspaces existed keep working); others live in ``RAG_DATA_DIR/workspaces/<name>``.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

import numpy as np

from app.config import Settings
from app.embeddings import Embedder, build_embedder
from app.errors import (
    FileTooLargeError,
    InvalidWorkspaceError,
    TooManyChunksError,
    WorkspaceExistsError,
    WorkspaceNotFoundError,
)
from app.ingestion.chunker import chunk_sections
from app.ingestion.context import describe_document
from app.ingestion.loaders import PARSER_VERSION, SUPPORTED_EXTENSIONS, load_document
from app.jobs import IngestJob, IngestJobQueue
from app.llm import LLM, build_llm
from app.models import Answer, DocumentInfo
from app.observability import Timer, log_event
from app.qa import QAEngine
from app.retrieval import RetrievalResult, Retriever
from app.vectorstore import VectorStore

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE = "default"
_WORKSPACE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
EMBED_SLICE = 256  # chunks embedded between progress updates (and cancellation checks)

# Called with the stage name and, while embedding, the fraction done. It may raise to abort ingestion.
StageCallback = Callable[[str, "float | None"], None]


@dataclass
class IngestResult:
    document: DocumentInfo
    status: str  # "created" | "updated" | "unchanged"


@dataclass
class Workspace:
    name: str
    root: Path
    store: VectorStore
    retriever: Retriever
    qa: QAEngine

    @property
    def files_dir(self) -> Path:
        return self.root / "files"


def document_id(name: str) -> str:
    return hashlib.sha1(name.encode()).hexdigest()[:12]


class RAGService:
    def __init__(self, settings: Settings, embedder: Embedder | None = None, llm: LLM | None = None) -> None:
        self.settings = settings
        self.embedder = embedder or build_embedder(settings)
        self.llm = llm or build_llm(settings)
        self._workspaces: dict[str, Workspace] = {}
        self._workspaces_lock = threading.RLock()
        self.jobs = IngestJobQueue(self)
        self.workspace(DEFAULT_WORKSPACE)

    # Shortcuts to the default workspace (CLI, evaluation, tests).
    @property
    def store(self) -> VectorStore:
        return self.workspace().store

    @property
    def retriever(self) -> Retriever:
        return self.workspace().retriever

    @property
    def qa(self) -> QAEngine:
        return self.workspace().qa

    # ------------------------------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Start the ingestion worker and re-queue files whose indexing never finished (or is out of date)."""
        self.jobs.start()
        queued = sum(len(self.sync_workspace(ws.name)) for ws in self.list_workspaces())
        if queued:
            log_event(logger, "resume_ingestion", files=queued)

    def stop(self) -> None:
        self.jobs.stop()

    def warm_up(self) -> None:
        """Load the embedding model and LLM into memory so the first real question isn't slowed by model loading."""
        try:
            self.embedder.embed_query("warm up")
            self.llm.complete("Reply with OK.", "OK")
            logger.info("Models warmed up")
        except Exception as exc:  # warm-up is best effort; real requests report errors properly
            logger.warning("Model warm-up failed: %s", exc)

    def readiness(self) -> dict[str, str]:
        """"ok" / "skipped" / an error message per dependency; the service is ready when nothing failed."""
        checks = {"index": "ok", "ingest_worker": "ok" if self.jobs.alive else "not running"}
        for name, component in (("embeddings", self.embedder), ("llm", self.llm)):
            ping = getattr(component, "ping", None)
            if ping is None:
                checks[name] = "skipped"
                continue
            try:
                ping()
                checks[name] = "ok"
            except Exception as exc:
                checks[name] = f"unreachable: {type(exc).__name__}"
        return checks

    # ----------------------------------------------------------------------------------------- workspaces
    def _root(self, name: str) -> Path:
        return self.settings.data_dir if name == DEFAULT_WORKSPACE else self.settings.data_dir / "workspaces" / name

    def _open(self, name: str) -> Workspace:
        root = self._root(name)
        store = VectorStore(root / "index", self.embedder.identity)
        s = self.settings
        retriever = Retriever(store, self.embedder, s.candidate_k, s.hybrid_search, s.document_lead_chunks)
        qa = QAEngine(retriever, self.llm, s.top_k, s.min_relevance, s.answer_cache_size)
        return Workspace(name, root, store, retriever, qa)

    @staticmethod
    def validate_workspace_name(name: str) -> str:
        if not _WORKSPACE_NAME_RE.match(name or ""):
            raise InvalidWorkspaceError(
                "Workspace names use 1-40 lowercase letters, digits, '-' or '_', starting with a letter or digit."
            )
        return name

    def workspace(self, name: str = DEFAULT_WORKSPACE) -> Workspace:
        self.validate_workspace_name(name)
        with self._workspaces_lock:
            if name not in self._workspaces:
                if name != DEFAULT_WORKSPACE and not self._root(name).is_dir():
                    raise WorkspaceNotFoundError(f"Workspace '{name}' does not exist.")
                self._workspaces[name] = self._open(name)
            return self._workspaces[name]

    def create_workspace(self, name: str) -> Workspace:
        self.validate_workspace_name(name)
        with self._workspaces_lock:
            if name == DEFAULT_WORKSPACE or self._root(name).is_dir():
                raise WorkspaceExistsError(f"Workspace '{name}' already exists.")
            self._root(name).mkdir(parents=True)
            log_event(logger, "workspace_created", workspace=name)
            return self.workspace(name)

    def delete_workspace(self, name: str) -> None:
        if name == DEFAULT_WORKSPACE:
            raise InvalidWorkspaceError("The default workspace can't be deleted.")
        with self._workspaces_lock:
            self.workspace(name)  # raises if it doesn't exist
            self._workspaces.pop(name)
            shutil.rmtree(self._root(name), ignore_errors=True)
        log_event(logger, "workspace_deleted", workspace=name)

    def list_workspaces(self) -> list[Workspace]:
        names = {DEFAULT_WORKSPACE}
        parent = self.settings.data_dir / "workspaces"
        if parent.is_dir():
            names.update(p.name for p in parent.iterdir() if p.is_dir() and _WORKSPACE_NAME_RE.match(p.name))
        with self._workspaces_lock:
            names.update(self._workspaces)
        return [self.workspace(n) for n in sorted(names, key=lambda n: (n != DEFAULT_WORKSPACE, n))]

    # ---------------------------------------------------------------------------------- original files
    def _original_path(self, ws: Workspace, name: str) -> Path:
        return ws.files_dir / document_id(name) / name

    def save_original(self, workspace: str, filename: str, data: bytes) -> str:
        """Persist the uploaded bytes (atomically) and return the sanitised file name."""
        ws = self.workspace(workspace)
        name = self._check_upload(filename, data)
        path = self._original_path(ws, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        return name

    def discard_original(self, workspace: str, filename: str) -> None:
        """Remove a stored original that never made it into the index (so it isn't retried at startup)."""
        try:
            ws = self.workspace(workspace)
        except WorkspaceNotFoundError:
            return
        if ws.store.find_by_name(filename) is None:
            shutil.rmtree(ws.files_dir / document_id(filename), ignore_errors=True)

    def read_original(self, workspace: str, filename: str) -> bytes | None:
        path = self._original_path(self.workspace(workspace), filename)
        return path.read_bytes() if path.is_file() else None

    def stored_originals(self, workspace: str) -> list[str]:
        files_dir = self.workspace(workspace).files_dir
        if not files_dir.is_dir():
            return []
        return sorted(p.name for p in files_dir.glob("*/*") if p.is_file() and not p.name.endswith(".tmp"))

    # ----------------------------------------------------------------------------------------- ingestion
    def _check_upload(self, filename: str, data: bytes) -> str:
        name = PureWindowsPath(filename).name  # strips any client-supplied directories ("/" and "\")
        if len(data) > int(self.settings.max_upload_mb * 1024 * 1024):
            raise FileTooLargeError(f"'{name}' exceeds the {self.settings.max_upload_mb:g} MB upload limit.")
        return name

    def _check_chunk_count(self, name: str, count: int) -> None:
        limit = self.settings.max_chunks_per_document
        if count > limit:
            raise TooManyChunksError(
                f"'{name}' would create {count:,} chunks, more than the limit of {limit:,} per document "
                f"(RAG_MAX_CHUNKS_PER_DOCUMENT). Split it into smaller files, or raise the limit if you can "
                f"wait for it to be embedded."
            )

    def check_document(self, filename: str, data: bytes) -> int:
        """Everything that can reject a file without the models: type, size, parseable text, chunk count."""
        name = self._check_upload(filename, data)
        s = self.settings
        chunks = chunk_sections(load_document(name, data), "check", name, s.chunk_size, s.chunk_overlap)
        self._check_chunk_count(name, len(chunks))
        return len(chunks)

    def fingerprint(self, data: bytes) -> str:
        # Covers the parsing/chunking settings too, so an upgrade re-indexes documents that were indexed before.
        s = self.settings
        pipeline = f"parser={PARSER_VERSION};chunk={s.chunk_size}/{s.chunk_overlap};context={s.document_context};"
        return hashlib.sha256(pipeline.encode() + data).hexdigest()

    def ingest(
        self,
        filename: str,
        data: bytes,
        workspace: str = DEFAULT_WORKSPACE,
        on_stage: StageCallback | None = None,
        force: bool = False,
        keep_original: bool = True,
    ) -> IngestResult:
        ws = self.workspace(workspace)
        name = self._check_upload(filename, data)
        stage = on_stage or (lambda _stage, _progress=None: None)
        content_hash = self.fingerprint(data)
        existing = ws.store.find_by_name(name)
        if existing and existing.content_hash == content_hash and not force:
            return IngestResult(existing, "unchanged")

        s = self.settings
        stage("parsing", None)
        with Timer("ingest_parse") as parse_t:
            sections = load_document(name, data)
            self._check_chunk_count(name, len(chunk_sections(sections, "check", name, s.chunk_size, s.chunk_overlap)))
        if keep_original:
            self.save_original(workspace, name, data)
        stage("describing", None)
        with Timer("ingest_describe") as describe_t:
            doc_context = describe_document(self.llm, name, sections) if s.document_context else None
        doc_id = existing.doc_id if existing else document_id(name)
        chunks = chunk_sections(sections, doc_id, name, s.chunk_size, s.chunk_overlap, doc_context)
        texts = [c.embedding_text() for c in chunks]
        parts = []
        with Timer("ingest_embed") as embed_t:
            for start in range(0, len(texts), EMBED_SLICE):
                stage("embedding", start / len(texts))
                parts.append(self.embedder.embed_documents(texts[start : start + EMBED_SLICE]))
        vectors = np.vstack(parts)
        stage("saving", None)
        info = DocumentInfo(
            doc_id=doc_id,
            name=name,
            content_hash=content_hash,
            num_chunks=len(chunks),
            ingested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        ws.store.add_document(info, chunks, vectors)
        status = "updated" if existing else "created"
        log_event(
            logger, "document_indexed", workspace=workspace, file=name, status=status, chunks=len(chunks),
            parse_ms=parse_t.ms, describe_ms=describe_t.ms, embed_ms=embed_t.ms,
            context=(doc_context or "-").replace("\n", " | "),
        )
        return IngestResult(info, status)

    def ingest_directory(self, directory: Path, workspace: str = DEFAULT_WORKSPACE) -> list[IngestResult]:
        files = sorted(p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
        return [self.ingest(p.name, p.read_bytes(), workspace) for p in files]

    def submit(self, workspace: str, filename: str, data: bytes) -> IngestJob:
        """Save the original, then index it in the background. The caller has already validated the file."""
        name = self.save_original(workspace, filename, data)
        return self.jobs.submit(workspace, name)

    def sync_workspace(self, workspace: str) -> list[IngestJob]:
        """Queue stored files that aren't indexed, or were indexed from other content or an older pipeline."""
        ws = self.workspace(workspace)
        jobs = []
        for name in self.stored_originals(workspace):
            existing = ws.store.find_by_name(name)
            data = self.read_original(workspace, name)
            if data is not None and (existing is None or existing.content_hash != self.fingerprint(data)):
                jobs.append(self.jobs.submit(workspace, name))
        return jobs

    def reindex(self, workspace: str) -> list[IngestJob]:
        """Re-index every stored original from scratch (e.g. after changing the description prompt)."""
        return [self.jobs.submit(workspace, name, force=True) for name in self.stored_originals(workspace)]

    # ------------------------------------------------------------------------------------------- queries
    def list_documents(self, workspace: str = DEFAULT_WORKSPACE) -> list[DocumentInfo]:
        return self.workspace(workspace).store.documents()

    def delete_document(self, doc_id: str, workspace: str = DEFAULT_WORKSPACE) -> bool:
        ws = self.workspace(workspace)
        deleted = ws.store.delete_document(doc_id)
        original_dir = ws.files_dir / doc_id
        if original_dir.is_dir():
            shutil.rmtree(original_dir, ignore_errors=True)
            deleted = True
        if deleted:
            log_event(logger, "document_deleted", workspace=workspace, doc_id=doc_id)
        return deleted

    def search(self, query: str, top_k: int | None = None, workspace: str = DEFAULT_WORKSPACE) -> RetrievalResult:
        return self.workspace(workspace).retriever.retrieve(query, top_k or self.settings.top_k)

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        history: list[str] | None = None,
        workspace: str = DEFAULT_WORKSPACE,
    ) -> Answer:
        return self.workspace(workspace).qa.answer(question, top_k, history)

    def ask_stream(
        self,
        question: str,
        top_k: int | None = None,
        history: list[str] | None = None,
        workspace: str = DEFAULT_WORKSPACE,
    ) -> Iterator[str | Answer]:
        return self.workspace(workspace).qa.answer_stream(question, top_k, history)
