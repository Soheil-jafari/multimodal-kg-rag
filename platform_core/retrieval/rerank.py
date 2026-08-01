"""Cross-encoder reranker (final stage when use_rerank).

``text_of`` lets the caller supply the text to score for text-less crops (their
caption), so figure/table hits are reranked on their caption rather than "".
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Optional

from platform_core.retrieval.base import Reranker
from platform_core.types import RetrievedChunk


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 device: str = "cuda") -> None:
        import torch
        from sentence_transformers import CrossEncoder

        dev = "cpu" if (device == "cuda" and not torch.cuda.is_available()) else device
        self.model = CrossEncoder(model_name, device=dev)

    def rerank(self, query: str, candidates: Sequence[RetrievedChunk], top_k: int,
               text_of: Optional[Callable[[RetrievedChunk], str]] = None) -> list[RetrievedChunk]:
        if not candidates:
            return []
        text_of = text_of or (lambda r: r.chunk.text)
        scores = self.model.predict([(query, text_of(r) or "") for r in candidates])
        order = sorted(zip(candidates, scores), key=lambda t: float(t[1]), reverse=True)
        return [RetrievedChunk(chunk=r.chunk, score=float(s), source=r.source, evidence=r.evidence)
                for r, s in order[:top_k]]
