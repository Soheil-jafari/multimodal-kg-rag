"""platform_core — reusable, domain-agnostic building blocks.

Subpackages: ingestion, stores, graph, llm, retrieval, generation.
Cross-cutting modules: types (shared dataclasses), config (typed config schema),
pipeline (assembles a run from a config).

Domain-specific content (predicate lists, gold sets) lives in ``domain_packs/``.
"""
