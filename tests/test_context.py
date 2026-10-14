import pytest

from app.config import Settings
from app.embeddings import HashingEmbedder
from app.errors import UpstreamModelError
from app.ingestion.context import describe_document, parse_context
from app.models import Section
from app.qa import build_user_prompt, previous_question_for
from app.service import RAGService
from tests.conftest import FakeLLM

CV_CONTEXT = "Type: resume, CV, curriculum vitae\nAbout: Resume of Jane Doe, a data engineer."


def test_parse_context_keeps_known_fields_and_tolerates_markdown():
    raw = "Sure!\n**Type:** resume, CV\n**About:** Resume of Jane Doe.\nKeywords: python, sql\nThanks"
    assert parse_context(raw) == "Type: resume, CV\nAbout: Resume of Jane Doe."


def test_context_is_indexed_with_the_first_chunk_only(context_service):
    chunks = [c for c in context_service.store.chunks() if c.doc_name == "jane.txt"]
    assert chunks[0].doc_context == CV_CONTEXT and all(c.doc_context is None for c in chunks[1:])


def test_parse_context_rejects_rambling_output():
    assert parse_context("I think this is some kind of document.") is None


def test_describe_document_falls_back_to_none_when_llm_fails():
    def boom(_prompt: str) -> str:
        raise UpstreamModelError("down")

    assert describe_document(FakeLLM(boom), "cv.docx", [Section("JANE DOE")]) is None


@pytest.fixture
def context_service(settings: Settings) -> RAGService:
    llm = FakeLLM(lambda prompt: CV_CONTEXT if "File name: jane.txt\n" in prompt else "About: Travel rules.")
    service = RAGService(settings.model_copy(update={"document_context": True}), embedder=HashingEmbedder(), llm=llm)
    service.ingest("travel.txt", b"Travel rules: book economy flights and keep your receipts.")
    service.ingest("jane.txt", b"JANE DOE\nEmail: jane@example.com")
    return service


def test_document_context_makes_type_words_searchable(context_service):
    top = context_service.search("whose curriculum vitae is this", top_k=1).hits[0]
    assert top.chunk.doc_name == "jane.txt"
    assert top.chunk.doc_context == CV_CONTEXT


def test_generated_context_is_never_shown_to_the_answering_llm(context_service):
    labels = {"S1": context_service.search("curriculum vitae", top_k=1).hits[0]}
    prompt = build_user_prompt("Whose CV is it?", labels)
    assert "JANE DOE" in prompt and "curriculum vitae" not in prompt


@pytest.mark.parametrize(
    "question,history,expected",
    [
        ("what is his email?", ["name in the resume?"], "name in the resume?"),
        ("and his phone?", ["name in the resume?", "what is his email?"], "name in the resume? / what is his email?"),
        ("email?", ["name in the resume?"], "name in the resume?"),
        ("what about the skills section?", ["name in the resume?"], "name in the resume?"),
        ("How many days of annual leave?", ["name in the resume?"], None),
        ("what is his email?", [], None),
    ],
)
def test_follow_up_questions_pick_up_earlier_questions(question, history, expected):
    assert previous_question_for(question, history) == expected


def test_follow_up_retrieval_uses_the_earlier_question(loaded_service, fake_llm):
    loaded_service.ask("Does it apply to contractors?", history=["What is the minimum password length?"])
    prompt = fake_llm.calls[-1][1]
    assert "Earlier question in this conversation" in prompt
    assert "information_security_policy.pdf" in prompt
