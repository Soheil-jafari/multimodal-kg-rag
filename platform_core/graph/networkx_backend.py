"""NetworkX graph backend (local, zero-infrastructure default).

Stores grounded KG edges on a MultiDiGraph: nodes are entity strings, each edge
carries predicate/chunk_id/evidence. Parallel edges (same triple from different
chunks) are kept distinct via integer edge keys.
"""
from __future__ import annotations

import os
import pickle
from collections.abc import Collection, Sequence
from typing import Optional

import networkx as nx

from platform_core.graph.base import GraphBackend
from platform_core.types import GraphEdge


class NetworkXGraphBackend(GraphBackend):
    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()

    def add_edge(self, edge: GraphEdge) -> None:
        self.g.add_edge(edge.subject, edge.object, predicate=edge.predicate,
                        chunk_id=edge.chunk_id, evidence=edge.evidence)

    @staticmethod
    def _edge(u: str, v: str, data: dict) -> GraphEdge:
        return GraphEdge(subject=u, object=v, predicate=data["predicate"],
                         chunk_id=data["chunk_id"], evidence=data["evidence"])

    def neighbors(self, entity: str, predicates: Optional[Collection[str]] = None) -> list[GraphEdge]:
        if entity not in self.g:
            return []
        out = []
        for u, v, data in list(self.g.out_edges(entity, data=True)) + list(self.g.in_edges(entity, data=True)):
            if predicates is None or data["predicate"] in predicates:
                out.append(self._edge(u, v, data))
        return out

    def subgraph(self, entities: Sequence[str], hops: int = 1) -> list[GraphEdge]:
        seen = {e for e in entities if e in self.g}
        frontier = set(seen)
        for _ in range(hops):
            nxt = set()
            for e in frontier:
                nxt.update(v for _, v in self.g.out_edges(e))
                nxt.update(u for u, _ in self.g.in_edges(e))
            frontier = nxt - seen
            seen |= nxt
        return [self._edge(u, v, d) for u, v, d in self.g.edges(data=True) if u in seen and v in seen]

    def chunk_ids_for(self, edges: Sequence[GraphEdge]) -> list[str]:
        return [e.chunk_id for e in edges]

    def all_edges(self) -> list[GraphEdge]:
        return [self._edge(u, v, d) for u, v, d in self.g.edges(data=True)]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.g, f)

    @classmethod
    def load(cls, path: str) -> "NetworkXGraphBackend":
        obj = cls()
        with open(path, "rb") as f:
            obj.g = pickle.load(f)
        return obj

    def __len__(self) -> int:
        return self.g.number_of_edges()
