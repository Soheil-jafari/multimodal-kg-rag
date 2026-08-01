"""PubLayNet loader: reads a parquet shard's rows into Pages.

Implements :class:`~platform_core.ingestion.base.LayoutSource`. Each parquet row
is one page: ``image`` (struct<bytes, path>), ``id`` (per-page COCO image_id),
and ``annotations`` (COCO list with ``bbox`` = [x, y, w, h] px and an integer
``category_id``). We read directly with pyarrow (no ``datasets`` dependency),
convert bboxes to corner form, and yield in-memory pages.

The shard has no document identifier, so ``page_uid = "<shard_tag>:<id>"`` and
page grouping is handled separately (see paper_grouping).
"""
from __future__ import annotations

import io
from collections.abc import Iterator
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

from platform_core.ingestion.base import LayoutSource
from platform_core.types import BBox, Page, Region, RegionType

# Verified against the shard's region-type distribution (text-dominated).
CATEGORY_MAP = {
    1: RegionType.TEXT,
    2: RegionType.TITLE,
    3: RegionType.LIST,
    4: RegionType.TABLE,
    5: RegionType.FIGURE,
}


class PubLayNetLoader(LayoutSource):
    """Streams PubLayNet pages (with labelled regions) from one parquet shard."""

    def __init__(
        self,
        repo: str = "jordanparker6/publaynet",
        shard: str = "data/validation-00000-of-00008-a39835b959a7216c.parquet",
        shard_tag: str = "val00000",
        batch_size: int = 64,
    ) -> None:
        self.repo = repo
        self.shard = shard
        self.shard_tag = shard_tag
        self.batch_size = batch_size

    def local_path(self) -> str:
        """Download (cached) the shard parquet and return its local path."""
        return hf_hub_download(self.repo, self.shard, repo_type="dataset")

    def iter_pages(self, limit: Optional[int] = None) -> Iterator[Page]:
        pf = pq.ParquetFile(self.local_path())
        seen = 0
        for batch in pf.iter_batches(
            batch_size=self.batch_size, columns=["image", "id", "annotations"]
        ):
            for row in pa.Table.from_batches([batch]).to_pylist():
                img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                width, height = img.size
                regions: list[Region] = []
                for ann in row["annotations"]:
                    rtype = CATEGORY_MAP.get(ann["category_id"])
                    if rtype is None:
                        continue
                    x, y, w, h = ann["bbox"]
                    regions.append(
                        Region(
                            region_type=rtype,
                            bbox=BBox(x, y, x + w, y + h),
                            source_ann_id=ann.get("id"),
                        )
                    )
                yield Page(
                    page_uid=f"{self.shard_tag}:{row['id']}",
                    page_index=0,
                    width=width,
                    height=height,
                    regions=regions,
                    image=img,
                )
                seen += 1
                if limit is not None and seen >= limit:
                    return
