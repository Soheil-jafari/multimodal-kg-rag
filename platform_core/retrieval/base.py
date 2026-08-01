"""Retrieval interfaces (ABCs).

There is exactly ONE retriever. Its behaviour is fully determined by
:class:`~platform_core.config.RetrievalFlags`; there is no separate "baseline
retriever". The reranker is an optional stage gated by ``use_rerank``.
"""
from __future__ import annotations

import abc
from collections.abc import Sequence

from platform_core.types import RetrievedChunk


class Retriever(abc.ABC):
    """Return canonical chunks ranked for a query, honouring the config flags."""

    @abc.abstractmethod
    def retrieve(self, query: str) -> list[RetrievedChunk]:
        raise NotImplementedError


class Reranker(abc.ABC):
    """Re-order candidate chunks with a stronger (cross-encoder) scorer."""

    @abc.abstractmethod
    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        raise NotImplementedError
