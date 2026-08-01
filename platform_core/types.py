"""Core domain types shared across the platform.

Declarations only (dataclasses + enums); no behaviour lives here. Every store,
index, and graph edge is keyed by ``chunk_id`` so the vector index, image index,
and knowledge graph can all be joined back to the single canonical chunk store
(see :mod:`platform_core.stores.chunk_store`).

Paper/page identity: ``paper_id`` and ``page_id`` are stored separately even
though, for the PubLayNet parquet (which carries no document id), they are equal
today (page == paper). Keeping both means real multi-page PMC grouping can be
swapped in later with no schema change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class RegionType(str, Enum):
    """PubLayNet region labels. Region text is obtained by OCR-ing the crop."""

    TEXT = "text"
    TITLE = "title"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in page-image pixel coordinates ``[x0,y0,x1,y1]``.

    PubLayNet stores COCO ``[x, y, width, height]``; the loader converts to this
    corner form at ingest time.
    """

    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class Region:
    """A labelled layout region on a page (geometry + label, pre-OCR)."""

    region_type: RegionType
    bbox: BBox
    source_ann_id: Optional[int] = None  # original COCO annotation id (traceability)


@dataclass
class Page:
    """A single page image with its labelled regions."""

    page_uid: str          # unique page id, e.g. "val00000:409598"
    page_index: int        # within-paper page index (0 while page == paper)
    width: int
    height: int
    regions: list[Region] = field(default_factory=list)
    image: Any = None      # in-memory PIL page image; not persisted on the dataclass


@dataclass
class Chunk:
    """A canonical retrieval unit — one row of the single-source-of-truth store.

    ``chunk_id`` is the bridge key used by every derived index. For FIGURE/TABLE
    regions, ``image_path`` points at the cropped region image used for CLIP.
    """

    chunk_id: str
    paper_id: str
    page_id: str
    page: int
    region_type: RegionType
    bbox: BBox
    text: str
    image_path: Optional[str] = None
    caption_id: Optional[str] = None
    page_width: Optional[int] = None
    page_height: Optional[int] = None
    # vision-read transcription of a TABLE crop (Phase 9). OCR linearises a table and
    # loses the grid, which made table crops unfindable; this is the clean version,
    # computed once at ingest and used as the region's retrieval text when enabled.
    vqa_text: Optional[str] = None

    def retrieval_text(self, prefer_vqa: bool = False) -> str:
        """Text this region is indexed and scored by."""
        if prefer_vqa and self.vqa_text and self.vqa_text.strip():
            return self.vqa_text
        return self.text


@dataclass
class RetrievalUnit:
    """A derived retrieval unit produced by a Chunker (index-build time).

    ``unit_id`` is stamped into the vector index. ``source_chunk_ids`` are the
    canonical region chunk_ids this unit's text came from, preserving the
    chunk_id bridge for evaluation even for naive windows that span regions.
    """

    unit_id: str
    text: str
    source_chunk_ids: list[str]
    page_id: str
    paper_id: str
    region_type: Optional[str] = None


@dataclass
class RetrievedChunk:
    """A chunk returned by the retriever, with its score and provenance."""

    chunk: Chunk
    score: float
    source: str  # comma-joined provenance: text | clip-image | caption-link | graph-hop
    evidence: Optional[str] = None  # graph-hop: the edge's evidence sentence


@dataclass
class GraphEdge:
    """A knowledge-graph edge extracted INTO the closed predicate schema.

    Every edge is grounded: it records the ``chunk_id`` it came from and the
    verbatim ``evidence`` sentence, so KG hits trace back to source text.
    """

    subject: str
    predicate: str  # MUST be a member of the closed schema
    object: str
    chunk_id: str
    evidence: str


@dataclass
class GenerationResult:
    """Output of the answer generator."""

    answer: str
    cited_chunk_ids: list[str] = field(default_factory=list)
    abstained: bool = False


@dataclass
class SentenceGrounding:
    """One answer sentence checked against its cited chunk(s)."""

    sentence: str
    cited_chunk_ids: list[str]
    similarity: Optional[float]
    supported: bool
    method: str  # "similarity" | "llm" | "uncited"
