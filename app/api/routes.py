from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse

from app.api.schemas import (
    DocumentOut,
    HealthResponse,
    IngestItem,
    IngestResponse,
    JobOut,
    JobsResponse,
    QueryRequest,
    QueryResponse,
    ReadinessResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    SourceOut,
    WorkspaceCreate,
    WorkspaceOut,
)
from app.errors import FileTooLargeError, UpstreamModelError
from app.ingestion.loaders import SUPPORTED_EXTENSIONS, file_extension
from app.jobs import IngestJob
from app.models import Answer
from app.observability import METRICS
from app.service import DEFAULT_WORKSPACE, RAGService, Workspace

logger = logging.getLogger(__name__)
router = APIRouter()

UPSTREAM_ERROR_MESSAGE = "A model backend (embeddings or LLM) is unavailable. Please retry later."

UI_PAGE = Path(__file__).resolve().parents[1] / "static" / "index.html"


def get_service(request: Request) -> RAGService:
    return request.app.state.service


def get_workspace(
    service: RAGService = Depends(get_service),
    x_workspace: str = Header(
        DEFAULT_WORKSPACE, alias="X-Workspace", description="Document library to use (see /workspaces)."
    ),
) -> str:
    service.workspace(x_workspace)  # 400 for a bad name, 404 if it doesn't exist
    return x_workspace


# --------------------------------------------------------------------------------------------- system
@router.get("/", include_in_schema=False)
def ui() -> FileResponse:
    return FileResponse(UI_PAGE, media_type="text/html", headers={"Cache-Control": "no-cache"})


@router.get("/health", response_model=HealthResponse, tags=["system"], summary="Liveness and index size")
def health(service: RAGService = Depends(get_service)) -> HealthResponse:
    workspaces = service.list_workspaces()
    return HealthResponse(
        status="ok",
        documents=sum(len(ws.store.documents()) for ws in workspaces),
        chunks=sum(ws.store.num_chunks for ws in workspaces),
        workspaces=len(workspaces),
        ingest_queue=service.jobs.depth,
        max_upload_mb=service.settings.max_upload_mb,
        max_chunks_per_document=service.settings.max_chunks_per_document,
        embedding_model=service.embedder.identity,
        llm_model=service.llm.name,
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    tags=["system"],
    summary="Readiness: model backends reachable and ingestion worker running (503 if not)",
)
def ready(service: RAGService = Depends(get_service)) -> JSONResponse:
    checks = service.readiness()
    is_ready = all(v in ("ok", "skipped") for v in checks.values())
    body = ReadinessResponse(ready=is_ready, checks=checks).model_dump()
    return JSONResponse(body, status_code=status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE)


@router.get("/metrics", tags=["system"], summary="Prometheus metrics", response_class=PlainTextResponse)
def metrics(service: RAGService = Depends(get_service)) -> PlainTextResponse:
    workspaces = service.list_workspaces()
    gauges = {
        "rag_documents": [({"workspace": ws.name}, len(ws.store.documents())) for ws in workspaces],
        "rag_chunks": [({"workspace": ws.name}, ws.store.num_chunks) for ws in workspaces],
        "rag_ingest_queue_depth": [({}, service.jobs.depth)],
    }
    return PlainTextResponse(METRICS.render(gauges), media_type="text/plain; version=0.0.4")


# ------------------------------------------------------------------------------------------ workspaces
def _workspace_out(ws: Workspace) -> WorkspaceOut:
    return WorkspaceOut(name=ws.name, documents=len(ws.store.documents()), chunks=ws.store.num_chunks)


@router.get("/workspaces", response_model=list[WorkspaceOut], tags=["workspaces"])
def list_workspaces(service: RAGService = Depends(get_service)) -> list[WorkspaceOut]:
    return [_workspace_out(ws) for ws in service.list_workspaces()]


@router.post("/workspaces", response_model=WorkspaceOut, status_code=status.HTTP_201_CREATED, tags=["workspaces"])
def create_workspace(body: WorkspaceCreate, service: RAGService = Depends(get_service)) -> WorkspaceOut:
    return _workspace_out(service.create_workspace(body.name))


@router.delete(
    "/workspaces/{name}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["workspaces"],
    summary="Delete a workspace with all its documents and files",
)
def delete_workspace(name: str, service: RAGService = Depends(get_service)) -> Response:
    service.delete_workspace(name)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ------------------------------------------------------------------------------------------- documents
def _job_out(job: IngestJob) -> JobOut:
    document = DocumentOut(**asdict(job.document)) if job.document else None
    return JobOut(**{**asdict(job), "document": document})


@router.post(
    "/documents",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
    summary="Upload and index one or more PDF / Word / Excel / CSV / TXT / Markdown files",
    responses={202: {"model": JobsResponse, "description": "Accepted for background indexing (`background=true`)."}},
)
async def upload_documents(
    files: list[UploadFile] = File(...),
    background: bool = Query(
        False, description="Return 202 immediately and index in the background; poll /jobs/{id} for progress."
    ),
    workspace: str = Depends(get_workspace),
    service: RAGService = Depends(get_service),
):
    max_bytes = int(service.settings.max_upload_mb * 1024 * 1024)
    payloads: list[tuple[str, bytes]] = []
    # Validate every file before indexing any of them, so a bad batch is rejected atomically.
    for upload in files:
        name = upload.filename or ""
        if file_extension(name) not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                f"Unsupported file type for '{name}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            )
        data = await upload.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise FileTooLargeError(f"'{name}' exceeds the {service.settings.max_upload_mb:g} MB upload limit.")
        await run_in_threadpool(service.check_document, name, data)  # 422 unreadable, 413 too many chunks
        payloads.append((name, data))

    if background:
        jobs = [await run_in_threadpool(service.submit, workspace, name, data) for name, data in payloads]
        body = JobsResponse(jobs=[_job_out(j) for j in jobs]).model_dump()
        return JSONResponse(body, status_code=status.HTTP_202_ACCEPTED)

    results = []
    for name, data in payloads:
        result = await run_in_threadpool(service.ingest, name, data, workspace)
        results.append(IngestItem(status=result.status, document=DocumentOut(**asdict(result.document))))
    return IngestResponse(results=results)


@router.get("/documents", response_model=list[DocumentOut], tags=["documents"])
def list_documents(
    workspace: str = Depends(get_workspace), service: RAGService = Depends(get_service)
) -> list[DocumentOut]:
    return [DocumentOut(**asdict(d)) for d in service.list_documents(workspace)]


@router.post(
    "/documents/reindex",
    response_model=JobsResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["documents"],
    summary="Re-index every stored original file in the workspace, in the background",
)
def reindex_documents(
    workspace: str = Depends(get_workspace), service: RAGService = Depends(get_service)
) -> JobsResponse:
    return JobsResponse(jobs=[_job_out(j) for j in service.reindex(workspace)])


@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["documents"])
def delete_document(
    doc_id: str, workspace: str = Depends(get_workspace), service: RAGService = Depends(get_service)
) -> Response:
    if not service.delete_document(doc_id, workspace):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Document '{doc_id}' not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/jobs", response_model=JobsResponse, tags=["documents"], summary="Recent ingestion jobs, newest first")
def list_jobs(workspace: str = Depends(get_workspace), service: RAGService = Depends(get_service)) -> JobsResponse:
    return JobsResponse(jobs=[_job_out(j) for j in service.jobs.list(workspace)])


@router.get("/jobs/{job_id}", response_model=JobOut, tags=["documents"])
def get_job(job_id: str, service: RAGService = Depends(get_service)) -> JobOut:
    job = service.jobs.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Job '{job_id}' not found")
    return _job_out(job)


@router.delete(
    "/jobs/{job_id}",
    response_model=JobOut,
    tags=["documents"],
    summary="Cancel a queued or running ingestion job (a running one stops at its next embedding batch)",
)
def cancel_job(job_id: str, service: RAGService = Depends(get_service)) -> JobOut:
    job = service.jobs.cancel(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Job '{job_id}' not found")
    if job.status in ("succeeded", "failed"):
        raise HTTPException(status.HTTP_409_CONFLICT, f"Job '{job_id}' has already finished")
    return _job_out(job)


# ------------------------------------------------------------------------------------------------ query
@router.post("/search", response_model=SearchResponse, tags=["query"], summary="Retrieval only (no LLM)")
def search(
    request: SearchRequest, workspace: str = Depends(get_workspace), service: RAGService = Depends(get_service)
) -> SearchResponse:
    result = service.search(request.query, request.top_k, workspace)
    hits = [
        SearchHit(
            document=h.chunk.doc_name,
            doc_id=h.chunk.doc_id,
            chunk_id=h.chunk.id,
            page=h.chunk.page,
            section=h.chunk.heading,
            text=h.chunk.text,
            score=round(h.score, 6),
            dense_score=round(h.dense_score, 4),
            lexical_score=round(h.lexical_score, 4),
        )
        for h in result.hits
    ]
    return SearchResponse(query=request.query, max_similarity=round(result.max_similarity, 4), hits=hits)


def _query_response(answer: Answer, started: float) -> QueryResponse:
    return QueryResponse(
        question=answer.question,
        answer=answer.answer,
        status=answer.status,
        reason=answer.reason,
        sources=[SourceOut(**asdict(s)) for s in answer.sources],
        warnings=answer.warnings,
        max_similarity=None if answer.max_similarity is None else round(answer.max_similarity, 4),
        latency_ms=int((time.perf_counter() - started) * 1000),
        timings=answer.timings,
        cached=answer.cached,
    )


@router.post("/query", response_model=QueryResponse, tags=["query"], summary="Answer a question with citations")
def query(
    request: QueryRequest, workspace: str = Depends(get_workspace), service: RAGService = Depends(get_service)
) -> QueryResponse:
    started = time.perf_counter()
    return _query_response(service.ask(request.question, request.top_k, request.history, workspace), started)


@router.post(
    "/query/stream",
    tags=["query"],
    summary="Same as /query, streamed as newline-delimited JSON",
    description=(
        'Emits `{"type": "token", "text": ...}` lines while the model writes (an unvalidated draft), then one '
        '`{"type": "final", ...}` line with the validated QueryResponse, or `{"type": "error", "detail": ...}`.'
    ),
)
def query_stream(
    request: QueryRequest, workspace: str = Depends(get_workspace), service: RAGService = Depends(get_service)
) -> StreamingResponse:
    started = time.perf_counter()

    def events() -> Iterator[str]:
        try:
            for item in service.ask_stream(request.question, request.top_k, request.history, workspace):
                if isinstance(item, str):
                    yield json.dumps({"type": "token", "text": item}) + "\n"
                else:
                    final = _query_response(item, started).model_dump()
                    yield json.dumps({"type": "final", **final}) + "\n"
        except UpstreamModelError as exc:
            logger.warning("Upstream failure while streaming: %s", exc)
            yield json.dumps({"type": "error", "detail": UPSTREAM_ERROR_MESSAGE}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")
