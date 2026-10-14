# Evaluation report

Generated 2026-10-13T10:13:15  
Embeddings: `openai:nomic-embed-text` | LLM: `qwen3:4b-instruct-2507-q4_K_M` | chunk_size=500, overlap=80, top_k=8, hybrid=True, min_relevance=0.55

## Summary

```json
{
  "questions": 19,
  "retrieval": {
    "source_recall": 0.964,
    "mrr": 1.0,
    "context_recall": 0.964,
    "context_recall_multi_evidence": 0.9
  }
}
```

## Per question

| id | category | max sim | source recall | MRR | context recall |
|---|---|---|---|---|---|
| A1 | answerable | 0.831 | 1.0 | 1.0 | 1.0 |
| A2 | answerable | 0.794 | 1.0 | 1.0 | 1.0 |
| A3 | answerable | 0.822 | 1.0 | 1.0 | 1.0 |
| A4 | answerable | 0.791 | 1.0 | 1.0 | 1.0 |
| A5 | answerable | 0.882 | 1.0 | 1.0 | 1.0 |
| A6 | answerable | 0.841 | 1.0 | 1.0 | 1.0 |
| A7 | answerable | 0.781 | 1.0 | 1.0 | 1.0 |
| A8 | answerable | 0.712 | 1.0 | 1.0 | 1.0 |
| A9 | answerable | 0.764 | 1.0 | 1.0 | 1.0 |
| M1 | multi_evidence | 0.746 | 0.5 | 1.0 | 0.5 |
| M2 | multi_evidence | 0.723 | 1.0 | 1.0 | 1.0 |
| M3 | multi_evidence | 0.802 | 1.0 | 1.0 | 1.0 |
| M4 | multi_evidence | 0.741 | 1.0 | 1.0 | 1.0 |
| M5 | multi_evidence | 0.765 | 1.0 | 1.0 | 1.0 |
| U1 | unanswerable | 0.722 | - | - | - |
| U2 | unanswerable | 0.704 | - | - | - |
| U3 | unanswerable | 0.483 | - | - | - |
| U4 | unanswerable | 0.630 | - | - | - |
| U5 | unanswerable | 0.753 | - | - | - |