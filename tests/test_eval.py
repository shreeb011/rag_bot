import json
from collections import Counter
from pathlib import Path

from eval.run_eval import evaluate, fact_present, summarise

QUESTIONS = json.loads((Path(__file__).resolve().parents[1] / "eval" / "questions.json").read_text(encoding="utf-8"))


def test_question_set_covers_all_categories():
    counts = Counter(q["category"] for q in QUESTIONS)
    assert len(QUESTIONS) >= 10
    assert counts["answerable"] >= 3 and counts["unanswerable"] >= 3 and counts["multi_evidence"] >= 3
    assert len({q["id"] for q in QUESTIONS}) == len(QUESTIONS)
    for q in QUESTIONS:
        if q["category"] == "unanswerable":
            assert not q["expected_sources"] and not q["expected_facts"]
        else:
            assert q["expected_sources"] and q["expected_facts"]


def test_fact_matching_handles_alternatives_and_number_formatting():
    assert fact_present("4 hours|four hours", "within Four Hours of discovery")
    assert fact_present("USD 4000", "USD 4,000")
    assert not fact_present("CFO", "the manager approves")


def test_expected_facts_exist_in_sample_documents(loaded_service):
    corpus = "\n".join(c.text for c in loaded_service.store.chunks())
    for q in QUESTIONS:
        for fact in q["expected_facts"]:
            if q["id"] == "M1" and "charged" in fact:
                continue  # answer phrasing fact ("not charged"), checked against the generated answer
            assert fact_present(fact, corpus), f"{q['id']}: '{fact}' not found in sample docs"


def test_offline_retrieval_eval_runs(loaded_service):
    rows = evaluate(loaded_service, QUESTIONS, top_k=8, retrieval_only=True)
    summary = summarise(rows, retrieval_only=True)
    # Even the lexical hashing embedder + BM25 should find the right documents for most questions.
    assert summary["retrieval"]["source_recall"] >= 0.8
