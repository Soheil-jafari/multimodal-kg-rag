"""Knowledge-graph interfaces (ABCs).

:class:`GraphBackend` is one interface over a local NetworkX backend and a
server-based Neo4j backend — selected in config, never hard-coded. Every edge is
a grounded :class:`~platform_core.types.GraphEdge` (carries its source
``chunk_id`` + evidence sentence), so graph hits always trace back to canonical
chunks.

:class:`TripleExtractor` extracts edges from chunk text INTO a closed predicate
schema (see :mod:`platform_core.graph.schema`); it must drop any triple whose
predicate is not in the schema.
"""
from __future__ import annotations

import abc
from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, Optional

from platform_core.types import Chunk, GraphEdge

if TYPE_CHECKING:
    from platform_core.graph.schema import PredicateSchema


class GraphBackend(abc.ABC):
    """Persist and query grounded KG edges."""

    @abc.abstractmethod
    def add_edge(self, edge: GraphEdge) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def neighbors(
        self, entity: str, predicates: Optional[Collection[str]] = None
    ) -> list[GraphEdge]:
        """Edges incident to ``entity``, optionally filtered by predicate."""
        raise NotImplementedError

    @abc.abstractmethod
    def subgraph(self, entities: Sequence[str], hops: int = 1) -> list[GraphEdge]:
        """The n-hop neighbourhood around a seed set (used for KG expansion)."""
        raise NotImplementedError

    @abc.abstractmethod
    def chunk_ids_for(self, edges: Sequence[GraphEdge]) -> list[str]:
        """Map edges back to the canonical chunk_ids they were extracted from."""
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError


class TripleExtractor(abc.ABC):
    """Extract grounded edges from a chunk, constrained to the closed schema."""

    @abc.abstractmethod
    def extract(self, chunk: Chunk, schema: "PredicateSchema") -> list[GraphEdge]:
        raise NotImplementedError
