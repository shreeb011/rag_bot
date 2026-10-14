"""Background ingestion.

Uploads are validated and their original bytes written to disk by the request handler, then indexed here
by a single worker thread. One worker is deliberate: describing and embedding a document keeps the local
model busy, and running several at once would only make every upload (and every question) slower.

Because the original file is on disk before the job is queued, nothing is lost if the server stops before
indexing finishes: on startup the service re-queues any stored file whose index entry is missing or stale.
"""

from __future__ import annotations

import logging
import queue
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

from app.errors import RAGError, UpstreamModelError
from app.models import DocumentInfo
from app.observability import METRICS, log_event, request_id_var

if TYPE_CHECKING:
    from app.service import RAGService

logger = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
UPSTREAM_JOB_ERROR = "A model backend was unavailable. The file is saved and will be indexed on the next restart or reindex."
MAX_JOBS_KEPT = 500


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class IngestJob:
    workspace: str
    filename: str
    force: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: JobStatus = "queued"
    stage: str | None = None
    progress: float | None = None  # 0..1 while embedding
    cancel_requested: bool = False
    result: str | None = None  # "created" | "updated" | "unchanged"
    document: DocumentInfo | None = None
    error: str | None = None
    request_id: str = "-"
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None


class IngestJobQueue:
    def __init__(self, service: RAGService) -> None:
        self.service = service
        self._queue: queue.Queue[IngestJob | None] = queue.Queue()
        self._jobs: OrderedDict[str, IngestJob] = OrderedDict()
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._pending = 0
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------------------------------ control
    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name="ingest-worker", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._queue.put(None)
            self._thread.join(timeout)

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def depth(self) -> int:
        return self._pending

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Block until every queued job has finished (used by tests and the CLI)."""
        with self._idle:
            return self._idle.wait_for(lambda: self._pending == 0, timeout)

    # ------------------------------------------------------------------------------------------ jobs
    def submit(self, workspace: str, filename: str, force: bool = False) -> IngestJob:
        with self._lock:
            for job in reversed(self._jobs.values()):  # same file already waiting: don't index it twice
                if job.status == "queued" and job.workspace == workspace and job.filename == filename:
                    job.force = job.force or force
                    return job
            job = IngestJob(workspace, filename, force, request_id=request_id_var.get())
            self._jobs[job.id] = job
            while len(self._jobs) > MAX_JOBS_KEPT:
                self._jobs.popitem(last=False)
            self._pending += 1
        self._queue.put(job)
        return job

    def get(self, job_id: str) -> IngestJob | None:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> IngestJob | None:
        """Cancel a queued job at once, or a running one at its next stage / embedding slice."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status not in ("queued", "running"):
                return job
            job.cancel_requested = True
            if job.status == "queued":
                job.status, job.finished_at = "cancelled", _now()
        if job.status == "cancelled":
            self.service.discard_original(job.workspace, job.filename)
            METRICS.inc("rag_ingest_jobs_total", status="cancelled")
            log_event(logger, "ingest_cancelled", job=job.id, file=job.filename)
        return job

    def list(self, workspace: str | None = None, limit: int = 50) -> list[IngestJob]:
        with self._lock:
            jobs = [j for j in reversed(self._jobs.values()) if workspace is None or j.workspace == workspace]
        return jobs[:limit]

    # ------------------------------------------------------------------------------------------ worker
    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            token = request_id_var.set(job.request_id)
            try:
                self._process(job)
            except Exception:  # never let one job kill the worker
                logger.exception("Ingestion worker error on job %s", job.id)
            finally:
                request_id_var.reset(token)
                with self._idle:
                    self._pending -= 1
                    self._idle.notify_all()

    def _process(self, job: IngestJob) -> None:
        with self._lock:
            if job.status == "cancelled":
                return
            job.status, job.started_at = "running", _now()

        def on_stage(stage: str, progress: float | None = None) -> None:
            if job.cancel_requested:
                raise _Cancelled
            job.stage, job.progress = stage, progress

        try:
            data = self.service.read_original(job.workspace, job.filename)
            if data is None:
                raise RAGError("The file was deleted before it could be indexed.")
            result = self.service.ingest(
                job.filename, data, workspace=job.workspace, on_stage=on_stage, force=job.force
            )
            job.status, job.result, job.document = "succeeded", result.status, result.document
        except _Cancelled:
            job.status = "cancelled"
            self.service.discard_original(job.workspace, job.filename)
            log_event(logger, "ingest_cancelled", job=job.id, file=job.filename)
        except UpstreamModelError as exc:
            # The original is kept, so the file is retried on the next restart or reindex.
            job.status, job.error = "failed", UPSTREAM_JOB_ERROR
            log_event(logger, "ingest_failed", logging.WARNING, job=job.id, file=job.filename, error=str(exc))
        except RAGError as exc:
            # Unparseable or too large: retrying can't help, so don't keep the file around.
            job.status, job.error = "failed", str(exc)
            self.service.discard_original(job.workspace, job.filename)
            log_event(logger, "ingest_failed", logging.WARNING, job=job.id, file=job.filename, error=str(exc))
        except Exception:
            job.status, job.error = "failed", "Internal error while indexing this file."
            logger.exception("Unexpected error in ingestion job %s (%s)", job.id, job.filename)
        finally:
            job.finished_at, job.stage, job.progress = _now(), None, None
            METRICS.inc("rag_ingest_jobs_total", status=job.status)


class _Cancelled(Exception):
    pass
