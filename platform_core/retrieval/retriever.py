"""The single, flag-driven retriever.

One retrieve path. Dense BGE text retrieval always runs; each enhancement is
gated by its own flag and *adds* to the pool. There is deliberately no
baseline-vs-enhanced branch — behaviour is a pure function of RetrievalFlags.
Results merge by chunk_id and carry comma-joined provenance. ``last_trace`` records
the sources hit and the predicates walked (for the eval per-question log).
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from platform_core.retrieval.base import Retriever
from platform_core.types import Chunk, RetrievedChunk


@dataclass
class _Cand:
    chunk: Chunk
    score: float
    sources: set = field(default_factory=set)
    evidence: Optional[str] = None


class FlagDrivenRetriever(Retriever):
    def __init__(self, flags, params, store, text_index, text_embedder,
                 image_index=None, clip_embedder=None, graph=None,
                 chunk_nodes=None, reranker=None, unit_sources=None) -> None:
        self.flags = flags
        self.params = params
        self.store = store
        self.text_index = text_index
        self.text_embedder = text_embedder
        self.image_index = image_index
        self.clip_embedder = clip_embedder
        self.graph = graph
        self.chunk_nodes = chunk_nodes or {}
        self.reranker = reranker
        self.unit_sources = unit_sources or {}  # unit_id -> source region chunk_ids
        self.last_trace: dict = {}

    def candidate_pool(self, query: str) -> list[RetrievedChunk]:
        """Full merged, provenance-tagged candidate set (before rerank/truncation)."""
        qvec = self.text_embedder.embed_query(query)
        pool: dict[str, _Cand] = {}
        walked_predicates: set = set()

        # dense BGE text retrieval — always on; this block alone IS the baseline.
        # A retrieval unit resolves to its source region chunk_ids (naive windows
        # span several regions); scoring against gold is thus region-coverage.
        text_ids: list[str] = []
        seen: set = set()
        for unit_id, score in self.text_index.search(qvec, self.params.top_k):
            for cid in self.unit_sources.get(unit_id) or [unit_id.split("#", 1)[0]]:
                self._add(pool, cid, score, "text")
                if cid not in seen:
                    seen.add(cid)
                    text_ids.append(cid)

        if self.flags.use_caption_anchor:
            self._add_caption_hits(pool, text_ids)
        if getattr(self.flags, "use_page_crop_expansion", False):
            self._add_page_crop_hits(pool, text_ids)
        if self.flags.use_clip_images:
            self._add_clip_hits(pool, query)
        if self.flags.use_kg:
            walked_predicates = self._add_graph_hits(pool, qvec, text_ids)

        results = [RetrievedChunk(chunk=c.chunk, score=c.score,
                                  source=",".join(sorted(c.sources)), evidence=c.evidence)
                   for c in pool.values()]
        self.last_trace = {
            "sources": dict(collections.Counter(s for c in pool.values() for s in c.sources)),
            "graph_predicates": sorted(walked_predicates),
        }
        return results

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        results = self.candidate_pool(query)
        if self.flags.use_rerank and self.reranker is not None:
            return self.reranker.rerank(query, results, self.params.rerank_top_k,
                                        text_of=self._rerank_text)
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:self.params.top_k]

    def _add(self, pool: dict, chunk_id: str, score: float, source: str,
             evidence: Optional[str] = None) -> str:
        cid = chunk_id.split("#", 1)[0]  # collapse split sub-chunks to their region
        c = pool.get(cid)
        if c is None:
            c = _Cand(chunk=self.store.get(cid), score=score)
            pool[cid] = c
        c.sources.add(source)
        if score > c.score:
            c.score = score
        if evidence and not c.evidence:
            c.evidence = evidence
        return cid

    def _add_caption_hits(self, pool: dict, text_ids: list[str]) -> None:
        # PRIMARY figure route: a retrieved caption -> its crop, inheriting the caption's score
        for crop in self.store.crops_for_captions(text_ids):
            cap_score = pool[crop.caption_id].score if crop.caption_id in pool else 0.0
            self._add(pool, crop.chunk_id, cap_score, "caption-link")

    def _add_page_crop_hits(self, pool: dict, text_ids: list[str]) -> None:
        # A table/figure is retrievable through the PAGE whose prose the question
        # matched, even when the crop itself is not matchable: its values carry no
        # linking vocabulary. Stricter than it looks — page == paper in this corpus and
        # gold questions are within-page, so this stays inside the answering document.
        # Inherits the page's best text score, so a crop never outranks its own page.
        pages: dict[str, float] = {}
        for cid in text_ids:
            c = pool.get(cid)
            if c is not None:
                pages[c.chunk.page_id] = max(pages.get(c.chunk.page_id, 0.0), c.score)
        for page_id, score in pages.items():
            for chunk in self.store.get_by_page(page_id):
                if chunk.image_path:
                    self._add(pool, chunk.chunk_id, score, "page-crop")

    def _add_clip_hits(self, pool: dict, query: str) -> None:
        # SECONDARY: CLIP image similarity (weak on out-of-domain scientific figures)
        if self.image_index is not None and self.clip_embedder is not None:
            qv = self.clip_embedder.embed_text([query])[0]
            for cid, score in self.image_index.search(qv, self.params.image_k):
                self._add(pool, cid, score, "clip-image")

    def _add_graph_hits(self, pool: dict, qvec, text_ids: list[str]) -> set:
        if self.graph is None:
            return set()
        seeds: set = set()  # entry into the graph: chunk_id -> its extracted entity nodes
        for cid in text_ids:
            seeds.update(self.chunk_nodes.get(cid, []))
        if not seeds:
            return set()
        edges = self.graph.subgraph(sorted(seeds), hops=self.params.kg_hops)
        rep: dict = {}
        for e in edges:
            rep.setdefault(e.chunk_id, e)  # one representative edge per evidence chunk
        chunks = self.store.get_many(list(rep))
        if not chunks:
            return {e.predicate for e in edges}
        vecs = self.text_embedder.embed([c.text for c in chunks])
        qv = np.asarray(qvec, dtype="float32")
        for c, v in zip(chunks, vecs):
            score = float(np.dot(qv, np.asarray(v, dtype="float32")))
            self._add(pool, c.chunk_id, score, "graph-hop", evidence=rep[c.chunk_id].evidence)
        return {e.predicate for e in edges}

    def _rerank_text(self, r: RetrievedChunk) -> str:
        # A crop is reranked on its caption, and a table on its caption + transcription
        # when one exists: reranking a table on its linearised OCR demoted the very
        # crops page-expansion had just recovered (phase 9).
        parts = []
        if r.chunk.caption_id:
            try:
                parts.append(self.store.get(r.chunk.caption_id).text)
            except KeyError:
                pass
        body = r.chunk.retrieval_text(
            prefer_vqa=getattr(self.flags, "use_table_vqa_text", False))
        if body.strip():
            parts.append(body)
        return "\n\n".join(p for p in parts if p.strip()) or r.chunk.region_type.value
