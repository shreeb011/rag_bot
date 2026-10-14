import numpy as np
import pytest

from app.embeddings import HashingEmbedder
from app.errors import IndexCompatibilityError
from app.models import Chunk, DocumentInfo
from app.retrieval import BM25Index, Retriever
from app.text import tokenize
from app.vectorstore import VectorStore


def _doc(doc_id: str, texts: list[str]) -> tuple[DocumentInfo, list[Chunk]]:
    chunks = [Chunk(f"{doc_id}:{i}", doc_id, f"{doc_id}.txt", i, t) for i, t in enumerate(texts)]
    return DocumentInfo(doc_id, f"{doc_id}.txt", "hash", len(chunks), "now"), chunks


def _store(tmp_path, embedder=None) -> tuple[VectorStore, HashingEmbedder]:
    embedder = embedder or HashingEmbedder()
    return VectorStore(tmp_path, embedder.identity), embedder


def _add(store, embedder, doc_id, texts):
    info, chunks = _doc(doc_id, texts)
    store.add_document(info, chunks, embedder.embed_documents([c.embedding_text() for c in chunks]))


def test_vector_search_ranks_most_similar_first(tmp_path):
    store, embedder = _store(tmp_path)
    _add(store, embedder, "a", ["annual leave is 24 days", "laptops are refreshed every 3 years", "per diem is 60 USD"])
    hits = store.search(embedder.embed_query("how many days of annual leave"), k=2)
    assert hits[0][0].text == "annual leave is 24 days"
    assert hits[0][1] > hits[1][1]


def test_store_persists_and_reloads(tmp_path):
    store, embedder = _store(tmp_path)
    _add(store, embedder, "a", ["alpha text", "beta text"])
    reloaded = VectorStore(tmp_path, embedder.identity)
    assert reloaded.num_chunks == 2
    assert [d.doc_id for d in reloaded.documents()] == ["a"]
    assert reloaded.search(embedder.embed_query("beta"), 1)[0][0].text == "beta text"


def test_reingesting_document_replaces_its_chunks(tmp_path):
    store, embedder = _store(tmp_path)
    _add(store, embedder, "a", ["old one", "old two"])
    _add(store, embedder, "b", ["other doc"])
    _add(store, embedder, "a", ["new content"])
    assert sorted(c.text for c in store.chunks()) == ["new content", "other doc"]


def test_delete_document(tmp_path):
    store, embedder = _store(tmp_path)
    _add(store, embedder, "a", ["one"])
    _add(store, embedder, "b", ["two"])
    assert store.delete_document("a") is True
    assert store.delete_document("a") is False
    assert [c.doc_id for c in store.chunks()] == ["b"]
    assert VectorStore(tmp_path, embedder.identity).num_chunks == 1


def test_index_built_with_other_embedder_is_rejected(tmp_path):
    store, embedder = _store(tmp_path)
    _add(store, embedder, "a", ["text"])
    with pytest.raises(IndexCompatibilityError):
        VectorStore(tmp_path, "openai:some-other-model")


def test_dimension_mismatch_is_rejected(tmp_path):
    store, embedder = _store(tmp_path)
    _add(store, embedder, "a", ["text"])
    info, chunks = _doc("b", ["x"])
    with pytest.raises(IndexCompatibilityError):
        store.add_document(info, chunks, np.ones((1, 3), dtype=np.float32))


def test_empty_store_returns_no_results(tmp_path):
    store, embedder = _store(tmp_path)
    assert store.search(embedder.embed_query("anything"), 5) == []
    result = Retriever(store, embedder).retrieve("anything", 5)
    assert result.hits == [] and result.max_similarity == -1.0


@pytest.mark.parametrize("lead_chunks,expected", [(0, ["cv:2"]), (2, ["cv:2", "cv:0"])])
def test_document_lead_chunk_is_appended(tmp_path, lead_chunks, expected):
    store, embedder = _store(tmp_path)
    _add(store, embedder, "cv", ["JANE DOE data engineer", "summary of skills", "education chemistry degree"])
    result = Retriever(store, embedder, hybrid=False, document_lead_chunks=lead_chunks).retrieve("chemistry degree", 1)
    assert [h.chunk.id for h in result.hits] == expected


def test_tokenizer_stems_common_inflections():
    assert tokenize("Reported devices") == tokenize("report device")
    assert tokenize("policies charges") == tokenize("policy charge")
    assert tokenize("the of and") == []


def test_bm25_prefers_rare_exact_terms():
    chunks = [Chunk(str(i), "d", "d.txt", i, t) for i, t in enumerate(
        ["SAML single sign-on is Enterprise only", "sign-in with Google on Pro", "plans and pricing overview"]
    )]
    scores = BM25Index(chunks).scores("Is SAML supported?")
    assert scores.index(max(scores)) == 0
    assert scores[2] == 0


@pytest.mark.parametrize(
    "question,expected_doc",
    [
        ("How many days of paid annual leave do employees get?", "employee_handbook.md"),
        ("What is the API rate limit on the Pro plan?", "product_faq.txt"),
        ("minimum password length characters", "information_security_policy.pdf"),
        ("international per diem meal allowance", "travel_expense_policy.md"),
    ],
)
def test_retrieval_finds_expected_document_in_sample_corpus(loaded_service, question, expected_doc):
    result = loaded_service.search(question, top_k=3)
    assert result.hits[0].chunk.doc_name == expected_doc


def test_hybrid_retrieval_covers_multiple_documents(loaded_service):
    result = loaded_service.search("lost stolen laptop reported within hours security team charged", top_k=6)
    docs = {h.chunk.doc_name for h in result.hits}
    assert {"information_security_policy.pdf", "employee_handbook.md"} <= docs


def test_hybrid_scores_are_reported(loaded_service):
    hit = loaded_service.search("SAML single sign-on", top_k=1).hits[0]
    assert hit.lexical_score > 0 and -1.0 <= hit.dense_score <= 1.0
