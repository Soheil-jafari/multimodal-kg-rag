"""Chunkers: the two strategies behind the ``use_layout_chunking`` flag.

* :class:`LayoutChunker` (enhanced) — retrieval unit = region; over-long regions
  are split into overlapping windows, and each unit is prefixed with its section
  title for context.
* :class:`NaiveChunker` (baseline) — concatenate the page's text in reading order
  and cut fixed ~N-token windows, ignoring layout; each window records the region
  chunk_ids it covers.

Both consume canonical region :class:`~platform_core.types.Chunk` rows and emit
:class:`~platform_core.types.RetrievalUnit`. Token counting is pluggable via
``count_tokens`` (defaults to whitespace words; the build script passes the BGE
tokenizer for accuracy). Window sizing uses per-word token lengths (cached) so we
never re-tokenize growing strings.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Optional

from platform_core.ingestion.base import Chunker
from platform_core.types import Chunk, RegionType, RetrievalUnit

TEXT_TYPES = {RegionType.TEXT, RegionType.TITLE, RegionType.LIST, RegionType.TABLE}


def _words_count(text: str) -> int:
    return len(text.split())


def _reading_order(regions: Sequence[Chunk], page_width: Optional[int]) -> list[Chunk]:
    """Column-aware reading order: left column top-to-bottom, then right column."""
    mid = (page_width / 2) if page_width else None

    def key(c: Chunk):
        xc = (c.bbox.x0 + c.bbox.x1) / 2
        col = 0 if (mid is None or xc < mid) else 1
        return (col, c.bbox.y0, c.bbox.x0)

    return sorted(regions, key=key)


def _word_token_lengths(words: list[str], count_tokens: Callable[[str], int]) -> list[int]:
    cache: dict[str, int] = {}
    out = []
    for w in words:
        n = cache.get(w)
        if n is None:
            n = max(1, count_tokens(w))
            cache[w] = n
        out.append(n)
    return out


def _window_spans(wlens: list[int], max_tokens: int, overlap_tokens: int) -> list[tuple[int, int]]:
    """Sliding [start,end) word spans, each <= max_tokens, overlapping by ~overlap_tokens."""
    n = len(wlens)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        tot, end = 0, start
        while end < n and (end == start or tot + wlens[end] <= max_tokens):
            tot += wlens[end]
            end += 1
        spans.append((start, end))
        if end >= n:
            break
        back, s = 0, end
        while s > start + 1 and back < overlap_tokens:
            s -= 1
            back += wlens[s]
        start = s
    return spans


class LayoutChunker(Chunker):
    def __init__(self, max_tokens: int = 350, overlap_tokens: int = 50,
                 count_tokens: Optional[Callable[[str], int]] = None,
                 text_of: Optional[Callable[[Chunk], str]] = None) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.count_tokens = count_tokens or _words_count
        # which text a region is indexed by — lets the caller substitute a table's
        # vision-read transcription for its linearised OCR (use_table_vqa_text)
        self.text_of = text_of or (lambda c: c.text)

    def _section_title(self, region: Chunk, titles: list[Chunk], mid: Optional[float]) -> str:
        best, best_y = "", None
        rc = (region.bbox.x0 + region.bbox.x1) / 2
        rcol = 0 if (mid is None or rc < mid) else 1
        for t in titles:
            tc = (t.bbox.x0 + t.bbox.x1) / 2
            tcol = 0 if (mid is None or tc < mid) else 1
            if t.bbox.y0 < region.bbox.y0 and tcol == rcol:
                if best_y is None or t.bbox.y0 > best_y:
                    best, best_y = t.text.strip(), t.bbox.y0
        return best

    def chunk_page(self, regions: Sequence[Chunk]) -> list[RetrievalUnit]:
        if not regions:
            return []
        page_width = regions[0].page_width
        mid = (page_width / 2) if page_width else None
        ordered = _reading_order(regions, page_width)
        titles = [r for r in ordered if r.region_type == RegionType.TITLE and r.text.strip()]
        units: list[RetrievalUnit] = []
        for r in ordered:
            body = (self.text_of(r) or "").strip()
            if r.region_type == RegionType.FIGURE or not body:
                continue
            section = "" if r.region_type == RegionType.TITLE else self._section_title(r, titles, mid)
            prefix = f"{section}\n\n" if section else ""
            budget = self.max_tokens - (self.count_tokens(prefix) if prefix else 0)
            if self.count_tokens(body) <= budget:
                pieces = [body]
            else:
                words = body.split()
                wlens = _word_token_lengths(words, self.count_tokens)
                spans = _window_spans(wlens, max(budget, 50), self.overlap_tokens)
                pieces = [" ".join(words[s:e]) for s, e in spans]
            for j, piece in enumerate(pieces):
                uid = r.chunk_id if len(pieces) == 1 else f"{r.chunk_id}#s{j}"
                units.append(RetrievalUnit(
                    unit_id=uid, text=prefix + piece, source_chunk_ids=[r.chunk_id],
                    page_id=r.page_id, paper_id=r.paper_id, region_type=r.region_type.value,
                ))
        return units


class NaiveChunker(Chunker):
    def __init__(self, window_tokens: int = 400, overlap_tokens: int = 50,
                 count_tokens: Optional[Callable[[str], int]] = None) -> None:
        self.window_tokens = window_tokens
        self.overlap_tokens = overlap_tokens
        self.count_tokens = count_tokens or _words_count

    def chunk_page(self, regions: Sequence[Chunk]) -> list[RetrievalUnit]:
        text_regions = [r for r in regions if r.region_type in TEXT_TYPES and r.text.strip()]
        if not text_regions:
            return []
        page_width = regions[0].page_width
        ordered = _reading_order(text_regions, page_width)
        pairs: list[tuple[str, str]] = []  # (word, source chunk_id)
        for r in ordered:
            for w in r.text.split():
                pairs.append((w, r.chunk_id))
        words = [w for w, _ in pairs]
        wlens = _word_token_lengths(words, self.count_tokens)
        page_id, paper_id = ordered[0].page_id, ordered[0].paper_id
        units: list[RetrievalUnit] = []
        for i, (s, e) in enumerate(_window_spans(wlens, self.window_tokens, self.overlap_tokens)):
            text = " ".join(words[s:e])
            srcs = list(dict.fromkeys(cid for _, cid in pairs[s:e]))
            units.append(RetrievalUnit(
                unit_id=f"{page_id}:w{i}", text=text, source_chunk_ids=srcs,
                page_id=page_id, paper_id=paper_id, region_type=None,
            ))
        return units
