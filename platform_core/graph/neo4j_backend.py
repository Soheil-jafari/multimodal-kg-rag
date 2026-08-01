"""Neo4j graph backend (server-based, same interface as NetworkX).

Implements :class:`~platform_core.graph.base.GraphBackend` over the Neo4j Python
driver. Connection details come from environment variables (NEO4J_URI /
NEO4J_USERNAME / NEO4J_PASSWORD), never from committed config. Demonstrates that
the graph store is a swappable backend, not a hard-coded dependency.

Not implemented. The interface exists to keep the graph store swappable and to
prove nothing in the retriever depends on NetworkX; every result in this project
was produced on the NetworkX backend. Completing it means Cypher MERGE for edges
and parameterized neighbourhood queries.
"""
from __future__ import annotations

from platform_core.graph.base import GraphBackend


class Neo4jGraphBackend(GraphBackend):
    """Neo4j-backed KG. Credentials from env only. Not implemented."""
