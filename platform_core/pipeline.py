"""End-to-end pipeline assembly — the embodiment of "one codebase, flag-driven".

Given an :class:`~platform_core.config.AppConfig`, this module constructs the
concrete backends selected in config (LLM, vector store, graph, embedders,
reranker), wires them into the single :class:`~platform_core.retrieval.base.Retriever`
and :class:`~platform_core.generation.base.AnswerGenerator`, and exposes a simple
``ask(question) -> GenerationResult`` surface used by both the app and the
evaluation harness.

There is exactly one construction path. Baseline vs enhanced is purely a matter
of which flags the config carries — this module contains no ``if baseline`` code.

Not yet implemented: the evaluation harness builds the same objects directly
(see ``scripts.ablate.build_harness``), so this factory is the unused half of the
surface. Consolidating the two is outstanding work.
"""
from __future__ import annotations
