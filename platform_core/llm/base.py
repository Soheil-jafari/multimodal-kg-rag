"""LLM and embedding interfaces (ABCs).

Text and image embedders are separate interfaces because they load different
models. On a 4GB GPU, implementations load ONE model at a time and embed in
batches (batch size from config). Vectors are typed ``Any`` to keep numpy out of
the interface; implementations return ``numpy.ndarray`` of shape ``(n, dim)``.
"""
from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any, Optional


class LLMClient(abc.ABC):
    """Chat/completion + structured-output surface (answers and KG extraction)."""

    @abc.abstractmethod
    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def complete_json(
        self, system: str, user: str, schema: dict, temperature: float = 0.0
    ) -> dict:
        """Constrained/structured output — used for closed-schema KG extraction."""
        raise NotImplementedError

    def answer_with_images(self, system: str, user: str, image_paths: Sequence[str],
                           temperature: float = 0.0) -> str:
        """Answer with figure/table crops attached (VQA path, gated by ``use_vqa``).

        Reads values OUT of the rendered image, which is the only way to recover a
        table's grid: OCR linearises it and loses structure (see Phase 1 caveat).
        Not abstract — a text-only backend may legitimately not support vision, so
        the default raises and callers must fall back to the text path.
        """
        raise NotImplementedError(f"{type(self).__name__} has no vision support")

    @property
    def supports_images(self) -> bool:
        """Whether ``answer_with_images`` is usable — lets callers degrade cleanly."""
        return type(self).answer_with_images is not LLMClient.answer_with_images


class TextEmbeddingModel(abc.ABC):
    """Embeds text (queries and chunk text). Batched; one model at a time."""

    @abc.abstractmethod
    def embed(self, texts: Sequence[str]) -> Any:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        raise NotImplementedError


class ImageEmbeddingModel(abc.ABC):
    """CLIP-style dual encoder: image tower for crops, text tower for queries."""

    @abc.abstractmethod
    def embed_images(self, image_paths: Sequence[str]) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def embed_text(self, texts: Sequence[str]) -> Any:
        """Query-side embedding into the shared image/text space."""
        raise NotImplementedError
