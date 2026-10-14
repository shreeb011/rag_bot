import re

import pytest

from app.errors import UpstreamModelError
from app.qa import NOT_FOUND_TOKEN, SYSTEM_PROMPT, UNAVAILABLE_MESSAGE, select_excerpt
from tests.conftest import FakeLLM


def _source_id_for(prompt: str, needle: str) -> str:
    """Find the [Sn] label the prompt assigned to the chunk containing ``needle``."""
    for match in re.finditer(r"\[(S\d+)\] \([^)]*\)\n(.*?)(?=\n\n\[S\d+\]|\n\n---)", prompt, re.DOTALL):
        if needle in match.group(2):
            return match.group(1)
    raise AssertionError(f"{needle!r} not in prompt")


def test_answer_with_valid_citation_returns_sources_and_excerpts(loaded_service):
    loaded_service.qa.llm = FakeLLM(lambda p: f"Employees get 24 days of paid annual leave [{_source_id_for(p, '24 days')}].")
    answer = loaded_service.ask("How many days of paid annual leave do employees get?")
    assert answer.status == "answered" and answer.reason is None
    assert len(answer.sources) == 1
    source = answer.sources[0]
    assert source.document == "employee_handbook.md"
    assert "24 days of paid annual leave" in source.excerpt
    assert source.excerpt in loaded_service.store.chunks()[[c.id for c in loaded_service.store.chunks()].index(source.chunk_id)].text
    assert answer.warnings == []


def test_prompt_contains_rules_question_and_numbered_sources(loaded_service, fake_llm):
    loaded_service.ask("What is the API rate limit on the Pro plan?")
    system, user = fake_llm.calls[0]
    assert system == SYSTEM_PROMPT and NOT_FOUND_TOKEN in system
    assert "Question: What is the API rate limit on the Pro plan?" in user
    assert "[S1] (document:" in user


def test_sentinel_response_means_unavailable(loaded_service):
    loaded_service.qa.llm = FakeLLM(NOT_FOUND_TOKEN)
    answer = loaded_service.ask("What is the parental leave policy?")
    assert answer.status == "unavailable"
    assert answer.reason == "not_in_documents"
    assert answer.answer == UNAVAILABLE_MESSAGE and answer.sources == []


def test_sentinel_appended_to_cited_answer_keeps_the_answer(loaded_service):
    loaded_service.qa.llm = FakeLLM(f"Pro retains data for 1 year [S1]. {NOT_FOUND_TOKEN} about the Starter SLA.")
    answer = loaded_service.ask("Compare data retention of the Pro and Enterprise plans.")
    assert answer.status == "answered"
    assert answer.answer == "Pro retains data for 1 year [S1]."
    assert any("not be covered" in w for w in answer.warnings)


def test_uncited_answer_is_rejected_as_ungrounded(loaded_service):
    loaded_service.qa.llm = FakeLLM("Employees get 30 days of leave.")
    answer = loaded_service.ask("How many days of annual leave do employees get?")
    assert answer.status == "unavailable" and answer.reason == "ungrounded_answer"
    assert "30 days" not in answer.answer


def test_uncited_refusal_is_classified_as_not_in_documents(loaded_service):
    loaded_service.qa.llm = FakeLLM("The documents do not mention parental leave.")
    answer = loaded_service.ask("What is the parental leave policy?")
    assert answer.reason == "not_in_documents"


def test_citations_to_unknown_sources_are_stripped(loaded_service):
    loaded_service.qa.llm = FakeLLM("Leave is 24 days [S1][S99].")
    answer = loaded_service.ask("How many days of annual leave do employees get?")
    assert "[S99]" not in answer.answer and "[S1]" in answer.answer
    assert [s.id for s in answer.sources] == ["S1"]
    assert any("S99" in w for w in answer.warnings)


def test_grouped_citations_are_normalised(loaded_service):
    loaded_service.qa.llm = FakeLLM("Fact one and fact two [S1, S2].")
    answer = loaded_service.ask("How many days of annual leave do employees get?")
    assert "[S1][S2]" in answer.answer
    assert [s.id for s in answer.sources] == ["S1", "S2"]


def test_only_cited_sources_are_returned(loaded_service):
    loaded_service.qa.llm = FakeLLM("Answer [S2].")
    answer = loaded_service.ask("How many days of annual leave do employees get?")
    assert [s.id for s in answer.sources] == ["S2"]
    assert len(answer.retrieved) > 1


def test_numbers_missing_from_cited_sources_are_flagged(loaded_service):
    loaded_service.qa.llm = FakeLLM(lambda p: f"Employees get 37 days of leave [{_source_id_for(p, '24 days')}].")
    answer = loaded_service.ask("How many days of annual leave do employees get?")
    assert answer.status == "answered"
    assert any("'37'" in w for w in answer.warnings)


def test_low_relevance_question_abstains_without_calling_llm(loaded_service, fake_llm):
    loaded_service.qa.min_relevance = 0.99
    answer = loaded_service.ask("Who won the 2022 FIFA World Cup?")
    assert answer.status == "unavailable" and answer.reason == "low_relevance"
    assert fake_llm.calls == []


def test_no_documents_abstains_without_calling_llm(service, fake_llm):
    answer = service.ask("How many days of annual leave do employees get?")
    assert answer.status == "unavailable" and answer.reason == "no_documents"
    assert fake_llm.calls == []


def test_llm_failure_propagates_as_upstream_error(loaded_service):
    def boom(_prompt):
        raise UpstreamModelError("timeout")

    loaded_service.qa.llm = FakeLLM(boom)
    with pytest.raises(UpstreamModelError):
        loaded_service.ask("How many days of annual leave do employees get?")


def test_stream_yields_draft_tokens_then_validated_answer(loaded_service):
    loaded_service.qa.llm = FakeLLM("Leave is 24 days [S1][S99].")
    items = list(loaded_service.ask_stream("How many days of annual leave do employees get?"))
    tokens, final = items[:-1], items[-1]
    assert all(isinstance(t, str) for t in tokens) and "".join(tokens) == "Leave is 24 days [S1][S99]."
    assert final.status == "answered" and "[S99]" not in final.answer


def test_stream_without_llm_call_yields_only_final_answer(service, fake_llm):
    items = list(service.ask_stream("How many days of annual leave do employees get?"))
    assert len(items) == 1 and items[0].reason == "no_documents"
    assert fake_llm.calls == []


def test_repeated_question_is_served_from_cache(loaded_service, fake_llm):
    first = loaded_service.ask("How many days of annual leave do employees get?")
    second = loaded_service.ask("  how many days of annual leave do employees GET? ")
    assert second.answer == first.answer and len(fake_llm.calls) == 1
    assert second.cached and not first.cached


def test_cache_is_invalidated_when_documents_change(loaded_service, fake_llm):
    loaded_service.ask("How many days of annual leave do employees get?")
    loaded_service.ingest("new.txt", b"Annual leave rules were updated.")
    loaded_service.ask("How many days of annual leave do employees get?")
    assert len(fake_llm.calls) == 2


def test_excerpt_is_verbatim_and_focused():
    text = (
        "The standard week is 40 hours. Core hours are 10:00 to 16:00. "
        "Full-time employees receive 24 days of paid annual leave per calendar year. "
        "Laptops are refreshed every 3 years."
    )
    excerpt = select_excerpt(text, "How many days of annual leave?", max_chars=90)
    assert "24 days of paid annual leave" in excerpt
    assert excerpt.rstrip("\u2026") in text
    assert len(excerpt) <= 90
