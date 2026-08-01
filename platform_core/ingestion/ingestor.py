"""Ingestor: wires loader + grouper + OCR + crop-saving into the chunk store.

For each page: group to a paper, then for each region emit one canonical
:class:`~platform_core.types.Chunk` row. text/title/list/table regions are OCR'd;
figure/table regions additionally have their crop saved to ``crops_dir`` (keyed
by chunk_id) for CLIP in phase 2. Figures are crop-only (no OCR text).

This populates the single source of truth. Deriving retrieval units (layout vs
naive chunking) happens later, at index-build time.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from platform_core.ingestion.base import LayoutSource, PaperGrouper, RegionOCR
from platform_core.ingestion.ocr import crop_bbox
from platform_core.stores.chunk_store import SQLiteChunkStore
from platform_core.types import BBox, Chunk, RegionType


def _safe_name(chunk_id: str) -> str:
    """chunk_id -> filesystem-safe filename stem (':' and '/' are invalid on Windows)."""
    return chunk_id.replace(":", "_").replace("/", "_")


class Ingestor:
    def __init__(
        self,
        loader: LayoutSource,
        grouper: PaperGrouper,
        ocr: RegionOCR,
        store: SQLiteChunkStore,
        crops_dir: str,
    ) -> None:
        self.loader = loader
        self.grouper = grouper
        self.ocr = ocr
        self.store = store
        self.crops_dir = crops_dir
        os.makedirs(crops_dir, exist_ok=True)

    def _save_crop(self, page_image: Any, bbox: BBox, chunk_id: str) -> Optional[str]:
        crop = crop_bbox(page_image, bbox)
        if crop is None:
            return None
        path = os.path.join(self.crops_dir, f"{_safe_name(chunk_id)}.png")
        crop.save(path)
        return path

    def run(self, limit: Optional[int] = None, skip_page_ids: Optional[set] = None,
            progress_every: int = 0) -> dict:
        """OCR and store ``limit`` pages from the loader.

        ``skip_page_ids`` makes the run resumable: pages already in the store are
        stepped over but still counted against ``limit``, so "the first N pages of this
        shard" means the same set whether it is ingested in one pass or resumed after
        an interruption. Ingest is the one genuinely slow, non-idempotent-in-cost step
        (OCR is CPU-bound, minutes per hundred pages), so a crash must not mean redoing
        the whole corpus.
        """
        self.store.init_schema()
        skip = skip_page_ids or set()
        pages = 0
        skipped = 0
        chunks_written = 0
        for page in self.loader.iter_pages(limit=limit):
            if page.page_uid in skip:
                skipped += 1
                continue
            pages += 1
            if progress_every and pages % progress_every == 0:
                print(f"    ... {pages} pages OCR'd ({chunks_written} chunks)", flush=True)
            paper_id = self.grouper.paper_id_for(page)
            chunks: list[Chunk] = []
            for k, region in enumerate(page.regions):
                chunk_id = f"{page.page_uid}:r{k}"
                rtype = region.region_type
                image_path = None
                if rtype in (RegionType.FIGURE, RegionType.TABLE):
                    image_path = self._save_crop(page.image, region.bbox, chunk_id)
                text = "" if rtype == RegionType.FIGURE else self.ocr.ocr_region(
                    page.image, region.bbox, rtype
                )
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        paper_id=paper_id,
                        page_id=page.page_uid,
                        page=page.page_index,
                        region_type=rtype,
                        bbox=region.bbox,
                        text=text,
                        image_path=image_path,
                        page_width=page.width,
                        page_height=page.height,
                    )
                )
            self.store.add_chunks(chunks)
            chunks_written += len(chunks)
        return {"pages": pages, "chunks": chunks_written, "skipped": skipped}
