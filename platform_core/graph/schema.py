"""The closed KG predicate schema and its enforcement.

The predicate list is CLOSED: the LLM extracts triples INTO these predicates and
never invents new ones. The concrete list is domain-specific and lives in the
domain pack (``domain_packs/biomed/predicates.yaml``); this module loads it and
enforces membership.
"""
from __future__ import annotations

from dataclasses import dataclass


def load_predicates(path: str) -> list[dict]:
    """Load the predicate definitions ``[{name, description}, ...]`` from YAML."""
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return list(data.get("predicates", []))


@dataclass(frozen=True)
class PredicateSchema:
    """An immutable, closed set of allowed predicate names."""

    predicates: frozenset[str]

    def is_allowed(self, predicate: str) -> bool:
        """True iff ``predicate`` is in the closed schema."""
        return predicate in self.predicates

    @classmethod
    def from_yaml(cls, path: str) -> "PredicateSchema":
        return cls(frozenset(p["name"] for p in load_predicates(path)))
