"""Ingestion interfaces (ABCs).

Ingestion turns PubLayNet (page images + region bounding boxes/labels — NOT
text) into rows of the canonical chunk store::

    page image + region bbox  --OCR-->  region text
    region text               --store--> Chunk rows (chunk_id, paper_id, ...)

Pages are grouped into papers by a :class:`PaperGrouper` (identity mapping for
the PubLayNet parquet). A :class:`Chunker` then derives retrieval units from the
canonical region rows at index-build time (layout vs naive — the
``use_layout_chunking`` toggle).
"""
from __future__ import annotations

import abc
from collections.abc import Iterator, Sequence
from typing import Any, Optional

from platform_core.types import BBox, Chunk, Page, RegionType, RetrievalUnit


class LayoutSource(abc.ABC):
    """Yields pages with their labelled regions from a PubLayNet-style corpus."""

    @abc.abstractmethod
    def iter_pages(self, limit: Optional[int] = None) -> Iterator[Page]:
        """Stream pages (in-memory image + labelled regions). ``limit`` caps the
        number of pages for quick dev runs."""
        raise NotImplementedError


class PaperGrouper(abc.ABC):
    """Maps a page to the ``paper_id`` it belongs to."""

    @abc.abstractmethod
    def paper_id_for(self, page: Page) -> str:
        raise NotImplementedError


class RegionOCR(abc.ABC):
    """Extracts text from a single region crop of a page image.

    Takes the in-memory page image (PIL) and the region bbox; the implementation
    crops and OCRs. FIGURE regions return "" (kept as crops for CLIP).
    """

    @abc.abstractmethod
    def ocr_region(self, page_image: Any, bbox: BBox, region_type: RegionType) -> str:
        raise NotImplementedError


class Chunker(abc.ABC):
    """Derives retrieval units from a page's canonical region chunks.

    Two implementations back the ``use_layout_chunking`` flag:

    * layout-aware — retrieval unit = region (over-long regions split with
      overlap, section title prepended for context);
    * naive — concatenate the page's text and re-split into fixed-size windows,
      recording which region chunk_ids each window covers (baseline behaviour).
    """

    @abc.abstractmethod
    def chunk_page(self, regions: Sequence[Chunk]) -> list[RetrievalUnit]:
        raise NotImplementedError
