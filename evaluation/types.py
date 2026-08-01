"""Evaluation value types."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class QuestionCategory(str, Enum):
    """Gold-question categories (coverage-driven)."""

    SINGLE_FACT = "single_fact"      # one triple
    MULTI_HOP = "multi_hop"          # 2+ connected triples sharing a node (needs graph)
    FIGURE = "figure"                # figure + its caption link
    FIGURE_VALUE = "figure_value"    # answer lives inside a table/figure (for VQA)
    TEXT_DERIVED = "text_derived"    # bias guard: from source text, not the graph
    UNANSWERABLE = "unanswerable"    # answer not in the corpus (tests abstention)


@dataclass
class GoldQuestion:
    """One gold record. Answer + evidence are known from the source, not invented."""

    qid: str
    question: str
    category: str
    expected_answer: str
    supporting_chunk_ids: list[str] = field(default_factory=list)
    source_predicates: list[str] = field(default_factory=list)

    @property
    def is_answerable(self) -> bool:
        return self.category != QuestionCategory.UNANSWERABLE.value


@dataclass
class RetrievalScores:
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    mrr: float
    ndcg: float


@dataclass
class AnswerScores:
    correctness: float
    faithfulness: float
    abstention_accuracy: float


@dataclass
class CategoryReport:
    category: str
    n: int
    retrieval: RetrievalScores
    answer: AnswerScores
