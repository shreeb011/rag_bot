"""Workspaces, background ingestion jobs, original-file storage and observability."""

from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

from app.embeddings import HashingEmbedder
from app.errors import UpstreamModelError
from app.main import create_app
from app.observability import JsonFormatter, Metrics, log_event, request_id_var
from app.service import RAGService
from tests.conftest import SAMPLE_DOCS

HANDBOOK = (SAMPLE_DOCS / "employee_handbook.md").read_bytes()
FAQ = (SAMPLE_DOCS / "product_faq.txt").read_bytes()


def _upload(client, name, data, workspace=None, background=False):
    headers = {"X-Workspace": workspace} if workspace else {}
    params = {"background": "true"} if background else {}
    return client.post("/documents", files=[("files", (name, data))], headers=headers, params=params)


# ----------------------------------------------------------------------------------------- workspaces
def test_workspace_crud(client):
    assert [w["name"] for w in client.get("/workspaces").json()] == ["default"]
    created = client.post("/workspaces", json={"name": "hr-team"})
    assert created.status_code == 201 and created.json() == {"name": "hr-team", "documents": 0, "chunks": 0}
    assert client.post("/workspaces", json={"name": "hr-team"}).status_code == 409
    assert [w["name"] for w in client.get("/workspaces").json()] == ["default", "hr-team"]
    assert client.delete("/workspaces/hr-team").status_code == 204
    assert [w["name"] for w in client.get("/workspaces").json()] == ["default"]


@pytest.mark.parametrize("name", ["Bad Name", "../etc", "-leading", "x" * 41, ""])
def test_invalid_workspace_names_are_rejected(client, name):
    assert client.post("/workspaces", json={"name": name}).status_code in (400, 422)


def test_default_workspace_cannot_be_deleted(client):
    assert client.delete("/workspaces/default").status_code == 400


def test_unknown_or_invalid_workspace_header(client):
    assert client.get("/documents", headers={"X-Workspace": "missing"}).status_code == 404
    assert client.get("/documents", headers={"X-Workspace": "../x"}).status_code == 400


def test_workspaces_are_isolated(client):
    client.post("/workspaces", json={"name": "support"})
    assert _upload(client, "employee_handbook.md", HANDBOOK).status_code == 201
    assert _upload(client, "product_faq.txt", FAQ, workspace="support").status_code == 201

    default_docs = [d["name"] for d in client.get("/documents").json()]
    support_docs = [d["name"] for d in client.get("/documents", headers={"X-Workspace": "support"}).json()]
    assert default_docs == ["employee_handbook.md"] and support_docs == ["product_faq.txt"]

    hits = client.post("/search", json={"query": "annual leave days"}, headers={"X-Workspace": "support"}).json()
    assert all(h["document"] == "product_faq.txt" for h in hits["hits"])
    answer = client.post("/query", json={"question": "How many days of annual leave?"}).json()
    assert all(s["document"] == "employee_handbook.md" for s in answer["sources"])


def test_deleting_a_workspace_removes_its_files(client, service):
    client.post("/workspaces", json={"name": "tmp"})
    _upload(client, "product_faq.txt", FAQ, workspace="tmp")
    root = service.workspace("tmp").root
    assert (root / "index").exists() and service.stored_originals("tmp") == ["product_faq.txt"]
    client.delete("/workspaces/tmp")
    assert not root.exists()


def test_workspaces_persist_across_restarts(settings, fake_llm):
    first = RAGService(settings, embedder=HashingEmbedder(), llm=fake_llm)
    first.create_workspace("legal")
    first.ingest("product_faq.txt", FAQ, workspace="legal")
    second = RAGService(settings, embedder=HashingEmbedder(), llm=fake_llm)
    assert [w.name for w in second.list_workspaces()] == ["default", "legal"]
    assert [d.name for d in second.list_documents("legal")] == ["product_faq.txt"]


# ------------------------------------------------------------------------------------- background jobs
def test_background_upload_returns_job_and_indexes(client, service):
    response = _upload(client, "employee_handbook.md", HANDBOOK, background=True)
    assert response.status_code == 202
    (job,) = response.json()["jobs"]
    assert job["filename"] == "employee_handbook.md" and job["status"] in ("queued", "running", "succeeded")

    assert service.jobs.wait_idle(10)
    done = client.get(f"/jobs/{job['id']}").json()
    assert done["status"] == "succeeded" and done["result"] == "created" and done["document"]["num_chunks"] > 0
    assert done["stage"] is None and done["finished_at"]
    assert [d["name"] for d in client.get("/documents").json()] == ["employee_handbook.md"]
    assert [j["id"] for j in client.get("/jobs").json()["jobs"]] == [job["id"]]


def test_background_upload_still_validates_up_front(client, service):
    assert _upload(client, "notes.exe", b"x", background=True).status_code == 415
    assert _upload(client, "empty.txt", b"   ", background=True).status_code == 422
    assert service.stored_originals("default") == []


def test_unknown_job_is_404(client):
    assert client.get("/jobs/nope").status_code == 404


def test_failed_job_reports_error_and_keeps_original(client, service, monkeypatch):
    def broken(_texts):
        raise UpstreamModelError("embedding backend down")

    monkeypatch.setattr(service.embedder, "embed_documents", broken)
    job = _upload(client, "product_faq.txt", FAQ, background=True).json()["jobs"][0]
    assert service.jobs.wait_idle(10)
    failed = client.get(f"/jobs/{job['id']}").json()
    assert failed["status"] == "failed" and "saved" in failed["error"]
    assert service.stored_originals("default") == ["product_faq.txt"]
    assert client.get("/documents").json() == []


def test_unfinished_uploads_are_indexed_on_startup(settings, fake_llm):
    first = RAGService(settings, embedder=HashingEmbedder(), llm=fake_llm)
    first.save_original("default", "product_faq.txt", FAQ)  # server stopped before the worker ran

    second = RAGService(settings, embedder=HashingEmbedder(), llm=fake_llm)
    with TestClient(create_app(second)):
        assert second.jobs.wait_idle(10)
        assert [d.name for d in second.list_documents()] == ["product_faq.txt"]


def test_reindex_reprocesses_every_stored_original(client, service):
    _upload(client, "employee_handbook.md", HANDBOOK)
    _upload(client, "product_faq.txt", FAQ)
    response = client.post("/documents/reindex")
    assert response.status_code == 202 and len(response.json()["jobs"]) == 2
    assert service.jobs.wait_idle(10)
    results = {j["filename"]: j["result"] for j in client.get("/jobs").json()["jobs"]}
    assert results == {"employee_handbook.md": "updated", "product_faq.txt": "updated"}


def test_synchronous_upload_stores_original_and_delete_removes_it(client, service):
    doc = _upload(client, "product_faq.txt", FAQ).json()["results"][0]["document"]
    assert service.read_original("default", "product_faq.txt") == FAQ
    assert client.delete(f"/documents/{doc['doc_id']}").status_code == 204
    assert service.stored_originals("default") == []


# ---------------------------------------------------------------------------------------- large files
def test_documents_over_the_chunk_limit_are_rejected_up_front(client, service):
    service.settings.max_chunks_per_document = 3
    for background in (False, True):
        response = _upload(client, "product_faq.txt", FAQ, background=background)
        assert response.status_code == 413 and "RAG_MAX_CHUNKS_PER_DOCUMENT" in response.json()["detail"]
    assert service.stored_originals("default") == [] and client.get("/documents").json() == []


def test_embedding_reports_progress(service, monkeypatch):
    monkeypatch.setattr("app.service.EMBED_SLICE", 3)
    events = []
    result = service.ingest("product_faq.txt", FAQ, on_stage=lambda stage, progress=None: events.append((stage, progress)))
    stages = [s for s, _ in events]
    assert stages[:2] == ["parsing", "describing"] and stages[-1] == "saving"
    progress = [p for s, p in events if s == "embedding"]
    assert progress[0] == 0 and progress == sorted(progress) and all(0 <= p < 1 for p in progress)
    assert len(progress) == -(-result.document.num_chunks // 3)


def test_cancel_queued_job(service):
    job = service.submit("default", "product_faq.txt", FAQ)  # worker not started yet
    assert service.jobs.cancel(job.id).status == "cancelled"
    assert service.stored_originals("default") == []
    service.start()
    try:
        assert service.jobs.wait_idle(10)
    finally:
        service.stop()
    assert job.status == "cancelled" and service.list_documents() == []


def test_cancel_running_job_stops_at_next_embedding_slice(client, service, monkeypatch):
    import threading

    monkeypatch.setattr("app.service.EMBED_SLICE", 2)
    started, release = threading.Event(), threading.Event()
    real_embed = service.embedder.embed_documents

    def slow_embed(texts):
        started.set()
        release.wait(5)
        return real_embed(texts)

    monkeypatch.setattr(service.embedder, "embed_documents", slow_embed)
    job = _upload(client, "product_faq.txt", FAQ, background=True).json()["jobs"][0]
    assert started.wait(5)
    cancelled = client.delete(f"/jobs/{job['id']}").json()
    assert cancelled["cancel_requested"] is True and cancelled["status"] == "running"
    release.set()
    assert service.jobs.wait_idle(10)
    assert client.get(f"/jobs/{job['id']}").json()["status"] == "cancelled"
    assert client.get("/documents").json() == [] and service.stored_originals("default") == []


def test_cancel_finished_or_unknown_job(client, service):
    job = _upload(client, "product_faq.txt", FAQ, background=True).json()["jobs"][0]
    assert service.jobs.wait_idle(10)
    assert client.delete(f"/jobs/{job['id']}").status_code == 409
    assert client.delete("/jobs/nope").status_code == 404


def test_unparseable_stored_file_is_not_retried_forever(settings, fake_llm):
    service = RAGService(settings, embedder=HashingEmbedder(), llm=fake_llm)
    service.save_original("default", "broken.pdf", b"%PDF-1.4 not really")
    with TestClient(create_app(service)):
        assert service.jobs.wait_idle(10)
        assert service.jobs.list()[0].status == "failed"
    assert service.stored_originals("default") == []


def test_health_reports_upload_limits(client, service):
    body = client.get("/health").json()
    assert body["max_upload_mb"] == service.settings.max_upload_mb
    assert body["max_chunks_per_document"] == service.settings.max_chunks_per_document


# --------------------------------------------------------------------------------------- observability
def test_request_id_is_generated_or_echoed(client):
    generated = client.get("/health").headers["X-Request-ID"]
    assert len(generated) == 16
    assert client.get("/health", headers={"X-Request-ID": "abc-123"}).headers["X-Request-ID"] == "abc-123"
    assert client.get("/health", headers={"X-Request-ID": "bad id!"}).headers["X-Request-ID"] != "bad id!"


def test_metrics_endpoint(client):
    _upload(client, "product_faq.txt", FAQ)
    client.post("/query", json={"question": "What does the Pro plan cost?"})
    body = client.get("/metrics").text
    assert 'http_requests_total{method="POST",route="/query",status="200"}' in body
    assert 'rag_documents{workspace="default"} 1' in body
    assert "rag_answers_total{" in body
    assert 'rag_stage_duration_seconds_count{stage="retrieval"}' in body
    assert "rag_ingest_queue_depth 0" in body


def test_metrics_use_route_templates_not_raw_paths(client):
    client.get("/jobs/abc123")
    body = client.get("/metrics").text
    assert 'route="/jobs/{job_id}"' in body and "abc123" not in body


def test_readiness(client):
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {
        "ready": True,
        "checks": {"index": "ok", "ingest_worker": "ok", "embeddings": "skipped", "llm": "skipped"},
    }


def test_readiness_fails_when_llm_unreachable(client, service, monkeypatch):
    def ping():
        raise ConnectionError("refused")

    monkeypatch.setattr(service.llm, "ping", ping, raising=False)
    response = client.get("/health/ready")
    assert response.status_code == 503 and response.json()["checks"]["llm"].startswith("unreachable")


def test_health_reports_workspaces_and_queue(client):
    client.post("/workspaces", json={"name": "a"})
    body = client.get("/health").json()
    assert body["workspaces"] == 2 and body["ingest_queue"] == 0


def test_query_response_includes_timings_and_cache_flag(client):
    _upload(client, "employee_handbook.md", HANDBOOK)
    question = {"question": "How many days of annual leave do employees get?"}
    first = client.post("/query", json=question).json()
    assert first["cached"] is False and {"retrieval_ms", "generation_ms"} <= first["timings"].keys()
    second = client.post("/query", json=question).json()
    assert second["cached"] is True and second["answer"] == first["answer"]


def test_json_log_format_includes_request_id_and_fields():
    record = logging.LogRecord("app.test", logging.INFO, __file__, 1, "msg", None, None)
    token = request_id_var.set("req-1")
    try:
        record.request_id = request_id_var.get()
        record.fields = {"event": "document_indexed", "chunks": 3}
        payload = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert payload["request_id"] == "req-1" and payload["event"] == "document_indexed" and payload["chunks"] == 3


def test_log_event_text_and_fields(caplog):
    with caplog.at_level(logging.INFO, logger="app.test"):
        log_event(logging.getLogger("app.test"), "answered", status="answered", sources=2)
    (record,) = caplog.records
    assert record.getMessage() == "answered status=answered sources=2"
    assert record.fields == {"event": "answered", "status": "answered", "sources": 2}


def test_metrics_registry_renders_prometheus_text():
    metrics = Metrics()
    metrics.describe("jobs_total", "counter", "Jobs.")
    metrics.inc("jobs_total", status="ok")
    metrics.inc("jobs_total", status="ok")
    metrics.observe("latency_seconds", 0.3, stage="x")
    text = metrics.render({"queue": [({}, 4)]})
    assert '# TYPE jobs_total counter\njobs_total{status="ok"} 2' in text
    assert 'latency_seconds_bucket{stage="x",le="0.25"} 0' in text
    assert 'latency_seconds_bucket{stage="x",le="0.5"} 1' in text
    assert 'latency_seconds_count{stage="x"} 1' in text
    assert "# TYPE queue gauge\nqueue 4" in text
