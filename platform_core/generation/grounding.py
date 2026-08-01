"""Grounding trust layer: verify each answer sentence against its cited chunk.

Cheap cosine (sentence vs cited chunk text) decides support; sentences within a
band of the threshold optionally go to an LLM yes/no verify.
"""
from __future__ import annotations

import re

import numpy as np

from platform_core.generation.generator import passage_text
from platform_core.types import RetrievedChunk, SentenceGrounding

_CITE = re.compile(r"\[([^\]]+)\]")
_SENT = re.compile(r"(?<=[.!?])\s+")


class GroundingChecker:
    def __init__(self, embedder, min_sim: float = 0.45, border: float = 0.08, llm=None) -> None:
        self.embedder = embedder
        self.min_sim = min_sim
        self.border = border
        self.llm = llm

    def check(self, answer: str, context, store) -> list[SentenceGrounding]:
        id2text = {c.chunk.chunk_id: passage_text(c.chunk, store) for c in context}
        sentences = [s for s in _SENT.split(answer.strip()) if s.strip()]
        cleaned = [_CITE.sub("", s).strip() for s in sentences]
        svecs = self.embedder.embed(cleaned) if cleaned else []

        out: list[SentenceGrounding] = []
        for sent, clean, sv in zip(sentences, cleaned, svecs):
            cited = [c for c in _CITE.findall(sent) if c in id2text]
            texts = [id2text[c] for c in cited if id2text[c].strip()]
            if not texts:
                out.append(SentenceGrounding(sent, cited, None, False, "uncited"))
                continue
            sim = max(float(np.dot(sv, cv)) for cv in self.embedder.embed(texts))
            supported, method = sim >= self.min_sim, "similarity"
            if self.llm is not None and abs(sim - self.min_sim) <= self.border:
                supported, method = self._llm_verify(clean, texts), "llm"
            out.append(SentenceGrounding(sent, cited, sim, supported, method))
        return out

    def _llm_verify(self, sentence: str, texts: list[str]) -> bool:
        passage = " ".join(texts)[:1500]
        reply = self.llm.complete(
            "Reply with only 'yes' or 'no'.",
            f"Passage:\n{passage}\n\nStatement: {sentence}\n\n"
            "Is the statement supported by the passage? Answer yes or no.",
        )
        return reply.strip().lower().startswith("y")
