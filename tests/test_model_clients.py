"""HTTP clients for the embedding and LLM backends, exercised against a mocked transport."""

import httpx
import numpy as np
import pytest

from app.embeddings import OpenAICompatibleEmbedder
from app.errors import UpstreamModelError
from app.llm import OpenAICompatibleLLM


def _client(handler) -> httpx.Client:
    return httpx.Client(base_url="http://backend", transport=httpx.MockTransport(handler))


def test_llm_strips_think_blocks():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "<think>hmm</think>Answer [S1]."}}]})

    llm = OpenAICompatibleLLM("m", "http://backend", "k", client=_client(handler))
    assert llm.complete("sys", "user") == "Answer [S1]."


def test_llm_sends_deterministic_chat_request():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(path=request.url.path, body=request.read())
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    OpenAICompatibleLLM("m", "http://backend", "k", temperature=0.0, client=_client(handler)).complete("s", "u")
    assert seen["path"] == "/chat/completions"
    assert b'"temperature":0.0' in seen["body"].replace(b" ", b"")


def test_llm_stream_parses_server_sent_events():
    sse = (
        'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"24 "}}]}\n\n'
        ': keep-alive comment\n\n'
        'data: {"choices":[{"delta":{"content":"days [S1]."}}]}\n\n'
        "data: [DONE]\n\n"
    )
    handler = lambda r: httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})  # noqa: E731
    llm = OpenAICompatibleLLM("m", "http://backend", "k", client=_client(handler))
    assert list(llm.stream("s", "u")) == ["24 ", "days [S1]."]


def test_llm_stream_wraps_backend_errors():
    llm = OpenAICompatibleLLM("m", "http://backend", "k", client=_client(lambda r: httpx.Response(500)))
    with pytest.raises(UpstreamModelError):
        list(llm.stream("s", "u"))


@pytest.mark.parametrize("keep_alive,expected", [("30m", b'"keep_alive":"30m"'), (None, None)])
def test_keep_alive_is_sent_only_when_configured(keep_alive, expected):
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.read().replace(b" ", b""))
        if request.url.path == "/embeddings":
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 0]}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    OpenAICompatibleLLM("m", "http://backend", "k", keep_alive=keep_alive, client=_client(handler)).complete("s", "u")
    OpenAICompatibleEmbedder("m", "http://backend", "k", keep_alive=keep_alive, client=_client(handler)).embed_query("q")
    for body in bodies:
        assert (expected in body) if expected else (b"keep_alive" not in body)


@pytest.mark.parametrize("status,body", [(500, {"error": "boom"}), (200, {"unexpected": True}), (200, {"choices": []})])
def test_llm_wraps_backend_errors(status, body):
    llm = OpenAICompatibleLLM("m", "http://backend", "k", client=_client(lambda r: httpx.Response(status, json=body)))
    with pytest.raises(UpstreamModelError):
        llm.complete("s", "u")


def test_llm_wraps_connection_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(UpstreamModelError):
        OpenAICompatibleLLM("m", "http://backend", "k", client=_client(handler)).complete("s", "u")


def test_embedder_wraps_backend_errors():
    embedder = OpenAICompatibleEmbedder("m", "http://backend", "k", client=_client(lambda r: httpx.Response(503)))
    with pytest.raises(UpstreamModelError):
        embedder.embed_query("hello")


def test_embedder_normalises_orders_and_prefixes():
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.read())
        return httpx.Response(200, json={"data": [{"index": 1, "embedding": [0, 2]}, {"index": 0, "embedding": [3, 4]}]})

    embedder = OpenAICompatibleEmbedder("m", "http://backend", "k", document_prefix="doc: ", client=_client(handler))
    vectors = embedder.embed_documents(["a", "b"])
    assert np.allclose(vectors, [[0.6, 0.8], [0.0, 1.0]])
    assert b'"doc: a"' in sent[0]


def test_embedder_detects_count_mismatch():
    handler = lambda r: httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 0]}]})  # noqa: E731
    embedder = OpenAICompatibleEmbedder("m", "http://backend", "k", client=_client(handler))
    with pytest.raises(UpstreamModelError):
        embedder.embed_documents(["a", "b"])
