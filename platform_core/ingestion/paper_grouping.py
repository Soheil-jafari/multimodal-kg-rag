"""Paper grouping: assign each page to the paper it belongs to.

Implements :class:`~platform_core.ingestion.base.PaperGrouper`.

The PubLayNet parquet carries no document identifier (``image.path`` is null and
``id`` is a per-page COCO image_id), so we do NOT invent cross-page papers.
:class:`PageAsPaperGrouper` treats each page as its own document
(``paper_id == page_uid``). This is an honest dataset limitation; because the
rest of the pipeline is ``paper_id``-agnostic, a real multi-page PMC grouper can
replace this class with no other code change.
"""
from __future__ import annotations

from platform_core.ingestion.base import PaperGrouper
from platform_core.types import Page


class PageAsPaperGrouper(PaperGrouper):
    """Each page is its own paper: ``paper_id = page_uid``."""

    def paper_id_for(self, page: Page) -> str:
        return page.page_uid
