import json

from app.errors import UpstreamModelError
from app.qa import NOT_FOUND_TOKEN
from tests.conftest import SAMPLE_DOCS, FakeLLM


def _upload(client, *names):
    files = [("files", (n, (SAMPLE_DOCS / n).read_bytes())) for n in names]
    return client.post("/documents", files=files)


def test_ui_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Document Assistant" in response.text


def test_health_reports_empty_index(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["documents"] == 0 and body["chunks"] == 0


def test_upload_pdf_txt_and_markdown(client):
    response = _upload(client, "information_security_policy.pdf", "product_faq.txt", "employee_handbook.md")
    assert response.status_code == 201
    results = response.json()["results"]
    assert [r["status"] for r in results] == ["created"] * 3
    assert all(r["document"]["num_chunks"] > 0 for r in results)
    assert len(client.get("/documents").json()) == 3


def test_reupload_identical_file_is_idempotent(client):
    _upload(client, "product_faq.txt")
    chunks_before = client.get("/health").json()["chunks"]
    response = _upload(client, "product_faq.txt")
    assert response.json()["results"][0]["status"] == "unchanged"
    assert client.get("/health").json()["chunks"] == chunks_before


def test_reupload_after_parser_upgrade_reindexes(client, monkeypatch):
    _upload(client, "product_faq.txt")
    monkeypatch.setattr("app.service.PARSER_VERSION", 999)
    assert _upload(client, "product_faq.txt").json()["results"][0]["status"] == "updated"


def test_reupload_changed_file_replaces_previous_version(client):
    client.post("/documents", files=[("files", ("notes.txt", b"The office plant is called Gerald."))])
    response = client.post("/documents", files=[("files", ("notes.txt", b"The office plant is called Fern."))])
    assert response.json()["results"][0]["status"] == "updated"
    hits = client.post("/search", json={"query": "office plant name"}).json()["hits"]
    assert [h["text"] for h in hits] == ["The office plant is called Fern."]


def test_unsupported_file_type_returns_415(client):
    response = client.post("/documents", files=[("files", ("malware.exe", b"MZ..."))])
    assert response.status_code == 415


def test_oversized_upload_returns_413(client, service):
    service.settings.max_upload_mb = 0.001  # ~1 KB
    response = client.post("/documents", files=[("files", ("big.txt", b"a " * 2000))])
    assert response.status_code == 413


def test_empty_and_corrupt_files_return_422(client):
    assert client.post("/documents", files=[("files", ("empty.txt", b""))]).status_code == 422
    assert client.post("/documents", files=[("files", ("bad.pdf", b"%PDF-1.7 not really"))]).status_code == 422


def test_batch_with_one_bad_file_ingests_nothing(client):
    files = [
        ("files", ("good.txt", b"Useful content about leave.")),
        ("files", ("bad.doc", b"binary")),
    ]
    assert client.post("/documents", files=files).status_code == 415
    assert client.get("/documents").json() == []


def test_delete_document(client):
    doc_id = _upload(client, "product_faq.txt").json()["results"][0]["document"]["doc_id"]
    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert client.delete(f"/documents/{doc_id}").status_code == 404
    assert client.get("/documents").json() == []


def test_query_returns_answer_with_cited_evidence(client, service):
    _upload(client, "employee_handbook.md")
    service.qa.llm = FakeLLM("Full-time employees receive 24 days of paid annual leave [S1].")
    response = client.post("/query", json={"question": "How many days of paid annual leave do I get?"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["sources"][0]["id"] == "S1"
    assert body["sources"][0]["document"] == "employee_handbook.md"
    assert body["sources"][0]["excerpt"]
    assert body["latency_ms"] >= 0


def test_query_unavailable_when_llm_finds_nothing(client, service):
    _upload(client, "employee_handbook.md")
    service.qa.llm = FakeLLM(NOT_FOUND_TOKEN)
    body = client.post("/query", json={"question": "What is the parental leave policy?"}).json()
    assert body["status"] == "unavailable"
    assert body["reason"] == "not_in_documents"
    assert body["sources"] == []
    assert "could not find" in body["answer"].lower()


def test_query_with_no_documents(client):
    body = client.post("/query", json={"question": "What is the leave policy?"}).json()
    assert body["status"] == "unavailable" and body["reason"] == "no_documents"


def test_query_validation_errors(client):
    assert client.post("/query", json={}).status_code == 422
    assert client.post("/query", json={"question": "   "}).status_code == 422
    assert client.post("/query", json={"question": "x" * 2001}).status_code == 422
    assert client.post("/query", json={"question": "valid question", "top_k": 0}).status_code == 422


def test_llm_outage_returns_503_without_leaking_details(client, service):
    _upload(client, "employee_handbook.md")

    def boom(_prompt):
        raise UpstreamModelError("connect to http://secret-host:11434 refused")

    service.qa.llm = FakeLLM(boom)
    response = client.post("/query", json={"question": "How many days of annual leave do I get?"})
    assert response.status_code == 503
    assert "secret-host" not in response.text


def _ndjson(response):
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def test_query_stream_emits_tokens_then_final_answer(client, service):
    _upload(client, "employee_handbook.md")
    service.qa.llm = FakeLLM("Full-time employees receive 24 days of paid annual leave [S1].")
    response = client.post("/query/stream", json={"question": "How many days of paid annual leave do I get?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = _ndjson(response)
    assert {e["type"] for e in events[:-1]} == {"token"}
    assert "".join(e["text"] for e in events[:-1]) == "Full-time employees receive 24 days of paid annual leave [S1]."
    final = events[-1]
    assert final["type"] == "final" and final["status"] == "answered"
    assert final["sources"][0]["document"] == "employee_handbook.md"


def test_query_stream_reports_backend_failure_as_error_event(client, service):
    _upload(client, "employee_handbook.md")

    def boom(_prompt):
        raise UpstreamModelError("connect to http://secret-host:11434 refused")

    service.qa.llm = FakeLLM(boom)
    events = _ndjson(client.post("/query/stream", json={"question": "How many days of annual leave do I get?"}))
    assert events[-1]["type"] == "error"
    assert "secret-host" not in json.dumps(events)


def test_query_stream_validates_input(client):
    assert client.post("/query/stream", json={"question": " "}).status_code == 422


def test_search_endpoint_returns_scored_hits(client):
    _upload(client, "product_faq.txt")
    body = client.post("/search", json={"query": "SAML single sign-on", "top_k": 3}).json()
    assert len(body["hits"]) == 3
    assert body["hits"][0]["document"] == "product_faq.txt"
    assert "SAML" in body["hits"][0]["text"]


def test_path_components_in_filename_are_stripped(client):
    response = client.post("/documents", files=[("files", ("../../etc/notes.txt", b"Some harmless notes."))])
    assert response.status_code == 201
    assert response.json()["results"][0]["document"]["name"] == "notes.txt"
