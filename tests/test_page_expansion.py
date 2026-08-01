"""``use_page_crop_expansion`` must key on PAGE, not paper.

The two are identical in this corpus (the PubLayNet parquet carries no document id,
so page == paper), which means no real gold question can catch a paper-keyed
regression. This builds the corpus we do *not* have yet — ONE paper, TWO pages, a
crop on each — and asserts the expansion stays on the page whose prose matched.

Why it matters: the flag exists because a table states values but not the concepts
linking them to a question, so it is reached through its page's prose. Keyed on
paper, one prose hit would instead drag in every crop of every page of the document
— flooding the pool with crops nothing matched, which is exactly the failure the
provenance gate downstream is meant to prevent.
"""
from __future__ import annotations

import numpy as np

from platform_core.config import RetrievalFlags, RetrievalParams
from platform_core.retrieval.retriever import FlagDrivenRetriever
from platform_core.types import BBox, Chunk, RegionType

PAPER = "P1"
PAGE1, PAGE2 = "P1:pg1", "P1:pg2"


def _chunk(cid: str, page_id: str, rtype: RegionType, text: str = "", crop: str = None) -> Chunk:
    return Chunk(chunk_id=cid, paper_id=PAPER, page_id=page_id, page=0, region_type=rtype,
                 bbox=BBox(0, 0, 10, 10), text=text, image_path=crop)


PROSE1 = _chunk("P1:pg1:r0", PAGE1, RegionType.TEXT, "lung cancer risk among exposed workers")
CROP1 = _chunk("P1:pg1:r1", PAGE1, RegionType.TABLE, "1.20 1.11 1.29", crop="/crops/pg1.png")
PROSE2 = _chunk("P1:pg2:r0", PAGE2, RegionType.TEXT, "unrelated methods paragraph")
CROP2 = _chunk("P1:pg2:r1", PAGE2, RegionType.TABLE, "4320 26.0 8.6", crop="/crops/pg2.png")
ALL = [PROSE1, CROP1, PROSE2, CROP2]


class _Store:
    """Answers a lookup by page id OR by paper id, so a paper-keyed retriever would
    silently succeed here and leak page 2 — which is what the assertions catch."""

    def __init__(self) -> None:
        self.by_id = {c.chunk_id: c for c in ALL}
        self.by_key = {PAGE1: [PROSE1, CROP1], PAGE2: [PROSE2, CROP2], PAPER: ALL}
        self.lookups: list[str] = []

    def get(self, chunk_id: str) -> Chunk:
        return self.by_id[chunk_id]

    def get_by_page(self, key: str) -> list[Chunk]:
        self.lookups.append(key)
        return self.by_key.get(key, [])


class _Index:
    """Only page 1's prose is retrieved; page 2 is never matched by the query."""

    def search(self, qvec, k):
        return [(PROSE1.chunk_id, 0.71)]


class _Embedder:
    def embed_query(self, q):
        return np.zeros(4, dtype="float32")


def _pool(**flag_overrides):
    store = _Store()
    r = FlagDrivenRetriever(
        flags=RetrievalFlags(**flag_overrides), params=RetrievalParams(top_k=5),
        store=store, text_index=_Index(), text_embedder=_Embedder(),
    )
    return store, {c.chunk.chunk_id: c for c in r.candidate_pool("lung cancer relative risk")}


def test_expansion_stays_on_the_matched_page():
    store, pool = _pool(use_page_crop_expansion=True)
    assert CROP1.chunk_id in pool, "the matched page's crop must enter the pool"
    assert CROP2.chunk_id not in pool, (
        "a crop from another page of the same paper leaked in — expansion is "
        "paper-keyed, and would flood the pool on a real multi-page corpus"
    )
    assert store.lookups == [PAGE1], f"looked up {store.lookups}, expected page ids only"


def test_expanded_crop_never_outranks_its_own_page():
    _, pool = _pool(use_page_crop_expansion=True)
    assert pool[CROP1.chunk_id].score <= pool[PROSE1.chunk_id].score
    assert pool[CROP1.chunk_id].source == "page-crop"


def test_flag_off_means_no_crops_at_all():
    _, pool = _pool()
    assert set(pool) == {PROSE1.chunk_id}, "baseline must be pure text — no crops"
