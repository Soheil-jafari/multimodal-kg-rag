"""Generation interface (ABC).

The generator turns retrieved context into a grounded answer. When
``allow_abstain`` is set and the context does not support an answer,
implementations MUST return ``GenerationResult(abstained=True, answer=
"insufficient evidence in the corpus")`` rather than guessing. Answers cite the
``chunk_id``s they rely on so faithfulness can be scored.
"""
from __future__ import annotations

import abc
from collections.abc import Sequence

from platform_core.types import GenerationResult, RetrievedChunk


class AnswerGenerator(abc.ABC):
    """Produce a grounded, optionally-abstaining answer from context."""

    @abc.abstractmethod
    def generate(
        self, query: str, context: Sequence[RetrievedChunk]
    ) -> GenerationResult:
        raise NotImplementedError
