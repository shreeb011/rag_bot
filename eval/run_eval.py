"""Evaluate retrieval and answer quality on eval/questions.json.

    python -m eval.run_eval                      # full pipeline (needs the configured embedding + LLM backends)
    python -m eval.run_eval --retrieval-only     # retrieval metrics only, no LLM calls
    python -m eval.run_eval --embedder hash --retrieval-only   # fully offline smoke run

Metrics
- source_recall     fraction of expected source documents present in the top-k retrieved chunks
- mrr               reciprocal rank of the first chunk from an expected source
- context_recall    fraction of expected facts that appear verbatim in the retrieved chunks
- fact_coverage     fraction of expected facts that appear in the generated answer
- citation_precision fraction of cited sources that come from an expected source document
- abstention        unanswerable questions must come back with status "unavailable";
                    answerable ones must not
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from app.config import Settings
from app.service import RAGService

ROOT = Path(__file__).resolve().parents[1]


def _normalise(text: str) -> str:
    return " ".join(text.lower().replace(",", "").replace("’", "'").split())


def fact_present(fact: str, text: str) -> bool:
    haystack = _normalise(text)
    return any(_normalise(alt) in haystack for alt in fact.split("|"))


def evaluate(service: RAGService, questions: list[dict], top_k: int, retrieval_only: bool) -> list[dict]:
    rows = []
    for q in questions:
        expected_sources = set(q["expected_sources"])
        facts = q["expected_facts"]
        row: dict = {"id": q["id"], "category": q["category"], "question": q["question"]}

        retrieval = service.search(q["question"], top_k)
        retrieved_docs = [h.chunk.doc_name for h in retrieval.hits]
        context = "\n".join(h.chunk.text for h in retrieval.hits)
        row["max_similarity"] = round(retrieval.max_similarity, 4)
        row["retrieved_docs"] = retrieved_docs
        if expected_sources:
            row["source_recall"] = len(expected_sources & set(retrieved_docs)) / len(expected_sources)
            first = next((i for i, d in enumerate(retrieved_docs) if d in expected_sources), None)
            row["mrr"] = 0.0 if first is None else 1.0 / (first + 1)
            row["context_recall"] = sum(fact_present(f, context) for f in facts) / len(facts) if facts else 1.0

        if not retrieval_only:
            started = time.perf_counter()
            answer = service.ask(q["question"], top_k)
            row["latency_s"] = round(time.perf_counter() - started, 2)
            row["status"] = answer.status
            row["reason"] = answer.reason
            row["answer"] = answer.answer
            row["cited_docs"] = [s.document for s in answer.sources]
            row["warnings"] = answer.warnings
            if q["category"] == "unanswerable":
                row["passed"] = answer.status == "unavailable"
            else:
                row["fact_coverage"] = sum(fact_present(f, answer.answer) for f in facts) / len(facts)
                cited = row["cited_docs"]
                row["citation_precision"] = (
                    sum(d in expected_sources for d in cited) / len(cited) if cited else 0.0
                )
                row["passed"] = (
                    answer.status == "answered" and row["fact_coverage"] == 1.0 and row["citation_precision"] >= 0.5
                )
        rows.append(row)
        status = row.get("status", "-")
        print(f"[{row['id']}] sim={row['max_similarity']:.3f} status={status} passed={row.get('passed', '-')}",
              flush=True)
    return rows


def summarise(rows: list[dict], retrieval_only: bool) -> dict:
    def mean(key: str, subset: list[dict]) -> float | None:
        values = [r[key] for r in subset if key in r]
        return round(statistics.mean(values), 3) if values else None

    answerable = [r for r in rows if r["category"] != "unanswerable"]
    unanswerable = [r for r in rows if r["category"] == "unanswerable"]
    summary = {
        "questions": len(rows),
        "retrieval": {
            "source_recall": mean("source_recall", answerable),
            "mrr": mean("mrr", answerable),
            "context_recall": mean("context_recall", answerable),
            "context_recall_multi_evidence": mean("context_recall", [r for r in answerable if r["category"] == "multi_evidence"]),
        },
    }
    if not retrieval_only:
        summary["answers"] = {
            "pass_rate": round(sum(r["passed"] for r in rows) / len(rows), 3),
            "answerable_pass_rate": round(sum(r["passed"] for r in answerable) / len(answerable), 3),
            "fact_coverage": mean("fact_coverage", answerable),
            "citation_precision": mean("citation_precision", answerable),
            "false_abstentions": sum(r["status"] == "unavailable" for r in answerable),
            "abstention_accuracy_unanswerable": round(
                sum(r["status"] == "unavailable" for r in unanswerable) / len(unanswerable), 3
            ) if unanswerable else None,
            "hallucinated_answers_to_unanswerable": sum(r["status"] == "answered" for r in unanswerable),
            "median_latency_s": round(statistics.median(r["latency_s"] for r in rows), 2),
        }
    return summary


def write_report(path: Path, rows: list[dict], summary: dict, settings: Settings, retrieval_only: bool) -> None:
    lines = [
        "# Evaluation report",
        "",
        f"Generated {datetime.now().isoformat(timespec='seconds')}  ",
        f"Embeddings: `{settings.embedding_provider}:{settings.embedding_model}` | LLM: `{settings.llm_model}` | "
        f"chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap}, top_k={settings.top_k}, "
        f"hybrid={settings.hybrid_search}, min_relevance={settings.min_relevance}",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "## Per question",
        "",
    ]
    if retrieval_only:
        lines += ["| id | category | max sim | source recall | MRR | context recall |", "|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(
                f"| {r['id']} | {r['category']} | {r['max_similarity']:.3f} | {r.get('source_recall', '-')} | "
                f"{r.get('mrr', '-')} | {r.get('context_recall', '-')} |"
            )
    else:
        lines += [
            "| id | category | passed | status / reason | max sim | context recall | fact coverage | citation precision | latency s |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            status = r["status"] + (f" / {r['reason']}" if r["reason"] else "")
            lines.append(
                f"| {r['id']} | {r['category']} | {'yes' if r['passed'] else '**no**'} | {status} | "
                f"{r['max_similarity']:.3f} | {r.get('context_recall', '-')} | {r.get('fact_coverage', '-')} | "
                f"{r.get('citation_precision', '-')} | {r['latency_s']} |"
            )
        lines += ["", "## Answers", ""]
        for r in rows:
            lines += [f"### {r['id']} - {r['question']}", "", r["answer"], ""]
            if r["cited_docs"]:
                lines.append(f"Cited: {', '.join(r['cited_docs'])}  ")
            for w in r["warnings"]:
                lines.append(f"Warning: {w}  ")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.json")
    parser.add_argument("--docs", type=Path, default=ROOT / "sample_docs")
    parser.add_argument("--out", type=Path, default=ROOT / "eval" / "results")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--embedder", choices=["openai", "hash"], default=None)
    parser.add_argument("--no-hybrid", action="store_true", help="Dense retrieval only (ablation)")
    args = parser.parse_args(argv)

    overrides: dict = {"data_dir": Path(tempfile.mkdtemp(prefix="rag_eval_"))}
    if args.embedder:
        overrides["embedding_provider"] = args.embedder
    if args.no_hybrid:
        overrides["hybrid_search"] = False
    settings = Settings(**overrides)
    top_k = args.top_k or settings.top_k

    service = RAGService(settings)
    for result in service.ingest_directory(args.docs):
        print(f"ingested {result.document.name} ({result.document.num_chunks} chunks)")

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    rows = evaluate(service, questions, top_k, args.retrieval_only)
    summary = summarise(rows, args.retrieval_only)
    print(json.dumps(summary, indent=2))

    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "retrieval" if args.retrieval_only else "full"
    if args.no_hybrid:
        suffix += "_dense_only"
    if args.embedder == "hash":
        suffix += "_hash"
    (args.out / f"results_{suffix}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(args.out / f"report_{suffix}.md", rows, summary, settings, args.retrieval_only)
    print(f"Wrote {args.out / f'report_{suffix}.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
