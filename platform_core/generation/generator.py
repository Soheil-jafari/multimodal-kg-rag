"""Grounded answer generation with flag-gated abstention.

One path: dense answering always; abstention (code gate + prompt clause) is gated
by allow_abstain. No baseline/enhanced fork.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from platform_core.generation.base import AnswerGenerator
from platform_core.generation.prompts import (
    ABSTAIN_CLAUSE, ABSTENTION_TEXT, ANSWER_SYSTEM, ANSWER_USER, COT_ANSWER_MARK,
    COT_CLAUSE, COT_REASONING_MARK, VQA_SYSTEM, VQA_USER,
)
from platform_core.types import GenerationResult, RetrievedChunk

_CITE = re.compile(r"\[([^\]]+)\]")


def split_cot(raw: str) -> tuple[str, str]:
    """(answer, reasoning) from a chain-of-thought response.

    Everything downstream scores the ANSWER alone — citations, the abstention match and
    per-sentence grounding — so the reasoning has to come off before any of them run.
    Falls back to treating the whole response as the answer when the model does not emit
    the marker, which keeps a malformed response scoreable instead of empty.
    """
    i = raw.rfind(COT_ANSWER_MARK)
    if i < 0:
        return raw.strip(), ""
    answer = raw[i + len(COT_ANSWER_MARK):].strip()
    head = raw[:i]
    j = head.find(COT_REASONING_MARK)
    reasoning = head[j + len(COT_REASONING_MARK):] if j >= 0 else head
    return (answer or raw.strip()), reasoning.strip()


def passage_text(chunk, store, prefer_vqa: bool = False) -> str:
    """Text used to represent a chunk to the model — a crop falls back to its caption."""
    body = chunk.retrieval_text(prefer_vqa=prefer_vqa)
    if body.strip():
        return body
    if chunk.caption_id:
        try:
            return store.get(chunk.caption_id).text
        except KeyError:
            pass
    return ""


class GroundedAnswerGenerator(AnswerGenerator):
    def __init__(self, llm, flags, params, embedder, store, vqa_llm=None) -> None:
        self.llm = llm
        self.flags = flags
        self.params = params
        self.embedder = embedder
        self.store = store
        # separate client so the vision model is config-selectable independently of
        # the text answer model; falls back to the text client when not supplied
        self.vqa_llm = vqa_llm or llm
        self.last_vqa: dict = {}  # per-call trace: which crops were sent (eval logging)
        self.last_reasoning: str = ""  # use_cot: the step-by-step trace, for explainability

    @property
    def _prefer_vqa_text(self) -> bool:
        """Whether a table is represented by its transcription rather than its OCR.

        Scored/indexed on: yes. Quoted as evidence: NO — see :meth:`_block`.
        """
        return bool(getattr(self.flags, "use_table_vqa_text", False))

    def relevance(self, query: str, context: Sequence[RetrievedChunk]) -> float:
        """Top query-context cosine, recomputed so the gate is rerank-independent."""
        texts = [t for t in (passage_text(c.chunk, self.store, self._prefer_vqa_text)
                             for c in context) if t.strip()]
        if not texts:
            return 0.0
        qv = self.embedder.embed_query(query)
        return max(float(np.dot(qv, v)) for v in self.embedder.embed(texts))

    def _block(self, context: Sequence[RetrievedChunk]) -> str:
        # Deliberately OCR text, never the transcription, even when the flag is on.
        # Measured in phase 9: the transcription gets a table's STRUCTURE right (which
        # is what makes it retrievable) but misreads digits in some rows. It is
        # therefore trusted to FIND and SCORE a table, and never quoted as the source
        # of a value — the attached image is that source.
        return "\n".join(
            f"[{c.chunk.chunk_id}] {passage_text(c.chunk, self.store) or '(' + c.chunk.region_type.value + ', no text)'}"
            for c in context
        )

    def _gated_crops(self, query: str, context: Sequence[RetrievedChunk]) -> tuple[list, list]:
        """Crops worth reading, plus the gate's per-crop scores for the trace.

        THE SAFETY GATE (phase 8 finding #3). Retrieval always returns *something*, and
        a vision model handed an off-topic crop will read a real number off it and cite
        it — turning a retrieval miss into a confident wrong answer instead of an
        abstention. So each crop must independently clear
        ``vqa_min_crop_score`` against the question before it is read. Scored on the
        crop's own content (transcription when present, else its caption), recomputed
        here so the gate is independent of rerank scores.
        """
        crops = [c for c in context if c.chunk.image_path]
        if not crops:
            return [], []
        if self.params.vqa_min_crop_score <= 0:  # gate disabled: read whatever came back
            return crops[:self.params.vqa_max_images], [
                {"chunk_id": c.chunk.chunk_id, "score": None,
                 "on_supported_page": None, "passed": True} for c in crops]

        # PROVENANCE is the gate, not the score. Measured on q061: the WRONG crop (a
        # similar table from another paper) scores 0.5534 while the RIGHT one scores
        # 0.4589 — the ordering is inverted, so no threshold can separate them. What
        # does separate them is whether the crop's page is one whose PROSE the question
        # matched: a table states values but not the concepts that make it relevant, so
        # its page is the evidence of relevance. Score is then used only to pick among
        # eligible crops, with a low floor to skip obvious junk.
        supported_pages = {c.chunk.page_id for c in context if "text" in (c.source or "")}
        texts = [passage_text(c.chunk, self.store, prefer_vqa=True) or
                 c.chunk.region_type.value for c in crops]
        qv = self.embedder.embed_query(query)
        sims = [float(np.dot(qv, v)) for v in self.embedder.embed(texts)]
        scored, eligible = [], []
        for c, s in zip(crops, sims):
            on_page = c.chunk.page_id in supported_pages
            ok = on_page and s >= self.params.vqa_min_crop_score
            scored.append({"chunk_id": c.chunk.chunk_id, "score": round(s, 4),
                           "on_supported_page": on_page, "passed": ok})
            if ok:
                eligible.append((s, c))
        eligible.sort(key=lambda t: t[0], reverse=True)
        return [c for _, c in eligible[:self.params.vqa_max_images]], scored

    @property
    def _cot(self) -> str:
        return COT_CLAUSE if getattr(self.flags, "use_cot", False) else ""

    def generate(self, query: str, context: Sequence[RetrievedChunk]) -> GenerationResult:
        self.last_vqa = {"used": False, "crops": []}
        self.last_reasoning = ""
        # code-gate abstention: thin/weak context -> abstain without spending an LLM call
        if self.flags.allow_abstain and (
            not context or self.relevance(query, context) < self.params.abstain_min_score
        ):
            return GenerationResult(answer=ABSTENTION_TEXT, cited_chunk_ids=[], abstained=True)

        # VQA path: a RELEVANT retrieved crop is in context and the LLM can see images.
        # No question-category test — the pipeline does not know the gold label at
        # inference time, so "a crop was retrieved and cleared the gate" is the only
        # honest trigger. Crops that fail the gate are not read at all; the answer
        # falls through to the text path, which keeps its own abstention gate. That is
        # deliberately narrower than "abstain whenever a crop fails": a text question
        # that happens to retrieve an off-topic crop must still be answerable.
        crops: list = []
        if self.flags.use_vqa:
            crops, gate = self._gated_crops(query, context)
            self.last_vqa["gate"] = gate
            self.last_vqa["gate_threshold"] = self.params.vqa_min_crop_score
        if crops and getattr(self.vqa_llm, "supports_images", False):
            ids = [c.chunk.chunk_id for c in crops]
            system = VQA_SYSTEM + (ABSTAIN_CLAUSE if self.flags.allow_abstain else "") + self._cot
            user = (VQA_USER.replace("{Q}", query)
                    .replace("{BLOCK}", self._block(context))
                    .replace("{IMGS}", ", ".join(ids)))
            try:
                answer = self.vqa_llm.answer_with_images(
                    system, user, [c.chunk.image_path for c in crops])
                # update, never rebind: rebinding dropped the gate trace on exactly the
                # calls where the gate ADMITTED a crop, so the eval log recorded which
                # crops were blocked only when nothing was read
                self.last_vqa.update({"used": True, "crops": ids})
            except NotImplementedError:  # vision-less backend -> text path
                crops = []
        if not self.last_vqa["used"]:
            system = ANSWER_SYSTEM + (ABSTAIN_CLAUSE if self.flags.allow_abstain else "") + self._cot
            answer = self.llm.complete(
                system,
                ANSWER_USER.replace("{Q}", query).replace("{BLOCK}", self._block(context)),
            )

        # Split BEFORE anything reads the answer: the abstention match, citation
        # extraction and (downstream) grounding must all see the final answer only, never
        # the reasoning that led to it.
        if getattr(self.flags, "use_cot", False):
            answer, self.last_reasoning = split_cot(answer)

        # prompt-gate abstention (belt-and-braces with the code gate above)
        if self.flags.allow_abstain and ABSTENTION_TEXT in answer.lower():
            return GenerationResult(answer=ABSTENTION_TEXT, cited_chunk_ids=[], abstained=True)

        in_ctx = {c.chunk.chunk_id for c in context}
        cited: list[str] = []
        for cid in _CITE.findall(answer):
            if cid in in_ctx and cid not in cited:
                cited.append(cid)
        return GenerationResult(answer=answer, cited_chunk_ids=cited, abstained=False)
