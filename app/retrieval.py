"""Hybrid retrieval: dense cosine similarity + BM25, fused with Reciprocal Rank Fusion (RRF).

Dense embeddings capture paraphrases ("time off" ~ "annual leave"); BM25 rescues exact tokens that small
embedding models blur together (plan names, numbers, acronyms such as "SAML" or "TLS"). RRF combines the two
rankings without having to calibrate their very different score scales.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from app.embeddings import Embedder
from app.models import Chunk, RetrievedChunk
from app.text import tokenize
from app.vectorstore import VectorStore

RRF_K = 60


@dataclass
class RetrievalResult:
    hits: list[RetrievedChunk]
    max_similarity: float  # best dense cosine similarity over the whole corpus (used for abstention)


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.term_freqs = [Counter(tokenize(c.embedding_text())) for c in chunks]
        self.lengths = [sum(tf.values()) for tf in self.term_freqs]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        doc_freq: Counter[str] = Counter()
        for tf in self.term_freqs:
            doc_freq.update(tf.keys())
        n = len(chunks)
        self.idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()}

    def scores(self, query: str) -> list[float]:
        terms = set(tokenize(query))
        results = []
        for tf, length in zip(self.term_freqs, self.lengths):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * length / (self.avg_length or 1.0))
            for term in terms:
                freq = tf.get(term)
                if freq:
                    score += self.idf[term] * freq * (self.k1 + 1) / (freq + norm)
            results.append(score)
        return results


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        candidate_k: int = 20,
        hybrid: bool = True,
        document_lead_chunks: int = 0,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.candidate_k = candidate_k
        self.hybrid = hybrid
        self.document_lead_chunks = document_lead_chunks
        self._bm25: BM25Index | None = None
        self._bm25_version = -1

    def retrieve(self, query: str, top_k: int) -> RetrievalResult:
        version = self.store.version
        chunks = self.store.chunks()
        if not chunks:
            return RetrievalResult([], max_similarity=-1.0)
        by_id = {c.id: c for c in chunks}
        dense = self.store.similarities(self.embedder.embed_query(query))
        dense_ranking = sorted(chunks, key=lambda c: dense.get(c.id, -1.0), reverse=True)[: self.candidate_k]
        max_similarity = dense.get(dense_ranking[0].id, -1.0)

        if not self.hybrid:
            hits = [RetrievedChunk(c, score=dense[c.id], dense_score=dense[c.id]) for c in dense_ranking[:top_k]]
            return RetrievalResult(self._add_document_leads(hits, by_id, dense, {}), max_similarity)

        bm25 = self._bm25_index(chunks, version)
        lexical_scores = dict(zip((c.id for c in chunks), bm25.scores(query)))
        lexical_ranking = [
            c for c in sorted(chunks, key=lambda c: lexical_scores[c.id], reverse=True) if lexical_scores[c.id] > 0
        ][: self.candidate_k]

        fused: dict[str, float] = {}
        for ranking in (dense_ranking, lexical_ranking):
            for rank, chunk in enumerate(ranking):
                fused[chunk.id] = fused.get(chunk.id, 0.0) + 1.0 / (RRF_K + rank + 1)

        ordered = sorted(fused, key=lambda cid: (fused[cid], dense.get(cid, -1.0)), reverse=True)[:top_k]
        hits = [
            RetrievedChunk(by_id[cid], score=fused[cid], dense_score=dense.get(cid, -1.0), lexical_score=lexical_scores[cid])
            for cid in ordered
        ]
        return RetrievalResult(self._add_document_leads(hits, by_id, dense, lexical_scores), max_similarity)

    def _add_document_leads(
        self, hits: list[RetrievedChunk], by_id: dict[str, Chunk], dense: dict[str, float], lexical: dict[str, float]
    ) -> list[RetrievedChunk]:
        """Append the first chunk of the top-ranked documents among the hits.

        A document's opening (title, author, a CV's name and contact details, the parties to a contract) is what
        questions like "what is the name in the resume?" are about, yet it rarely shares words with them.
        """
        present = {h.chunk.id for h in hits}
        leads: list[RetrievedChunk] = []
        for doc_id in dict.fromkeys(h.chunk.doc_id for h in hits):
            if len(leads) >= self.document_lead_chunks:
                break
            lead = by_id.get(f"{doc_id}:0")
            if lead is not None and lead.id not in present:
                leads.append(RetrievedChunk(lead, score=0.0, dense_score=dense.get(lead.id, -1.0),
                                            lexical_score=lexical.get(lead.id, 0.0)))
        return hits + leads

    def _bm25_index(self, chunks: list[Chunk], version: int) -> BM25Index:
        if self._bm25 is None or self._bm25_version != version:
            self._bm25 = BM25Index(chunks)
            self._bm25_version = version
        return self._bm25
