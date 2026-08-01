"""Storage interfaces (ABCs).

:class:`ChunkStore` is the SINGLE SOURCE OF TRUTH. :class:`VectorIndex` (used for
both text and CLIP-image embeddings) is *derived* from it and stores only
``chunk_id`` + vector — never a second copy of the text. Everything joins back
to the chunk store by ``chunk_id``.

Vectors are typed as ``Any`` here to keep numpy out of the interface layer;
implementations use ``numpy.ndarray`` of shape ``(n, dim)``.
"""
from __future__ import annotations

import abc
from collections.abc import Collection, Iterator, Sequence
from typing import Any, Optional

from platform_core.types import Chunk, RegionType


class ChunkStore(abc.ABC):
    """Canonical persistent store of chunks, keyed by ``chunk_id``."""

    @abc.abstractmethod
    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, chunk_id: str) -> Chunk:
        raise NotImplementedError

    @abc.abstractmethod
    def get_many(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        raise NotImplementedError

    @abc.abstractmethod
    def iter_chunks(
        self, region_types: Optional[Collection[RegionType]] = None
    ) -> Iterator[Chunk]:
        """Iterate all chunks, optionally filtered by region type."""
        raise NotImplementedError

    @abc.abstractmethod
    def paper_ids(self) -> list[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_by_page(self, page_id: str) -> list[Chunk]:
        """Every chunk on ONE page — keyed by ``page_id``, never ``paper_id``.

        Page-keyed on purpose. ``use_page_crop_expansion`` reaches a crop through the
        page whose prose was retrieved; keying that on ``paper_id`` is identical today
        (page == paper) but would pull in every crop of every page of a paper once real
        multi-page grouping is swapped in, flooding the pool. The retriever depends on
        this method, so it belongs on the interface.
        """
        raise NotImplementedError


class VectorIndex(abc.ABC):
    """A derived nearest-neighbour index over chunk embeddings.

    The same interface backs the text index and the CLIP image index.
    """

    @abc.abstractmethod
    def add(self, chunk_ids: Sequence[str], vectors: Any) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def search(self, query_vector: Any, k: int) -> list[tuple[str, float]]:
        """Return ``[(chunk_id, score), ...]`` sorted by descending similarity."""
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def load(cls, path: str) -> "VectorIndex":
        raise NotImplementedError
