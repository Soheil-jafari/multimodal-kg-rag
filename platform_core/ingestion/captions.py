"""Caption -> figure/table linking.

For each figure/table crop, find the nearest caption text region on the same page
— a text chunk whose OCR begins "Figure N" / "Table N" — and record its chunk_id
as ``caption_id`` on the crop's row. Figures caption below, tables caption above,
so we score candidates by side + vertical gap + horizontal overlap and pick the
best. This is the hook that makes figure/table questions groundable (crop + caption).
"""
from __future__ import annotations

import difflib
import re

from platform_core.stores.chunk_store import SQLiteChunkStore
from platform_core.types import BBox, Chunk, RegionType

_FIG_RE = re.compile(r"^\s*fig(\.|ure|\b)", re.IGNORECASE)
_TAB_RE = re.compile(r"^\s*tab(\.|le|\b)", re.IGNORECASE)
_MAX_PENALTY = 400.0  # reject links worse than this (px-ish)

# --- fuzzy label matching (fallback only) ---------------------------------------
# RapidOCR mangles the small-caps caption label used by one journal style that is
# common from val00001 on: "FiGurz", "FiGukz", "FiGuke", "FiGvre", "Tablle". The strict
# prefixes above miss every one of them, which cost 23 of the 500-page corpus's 119
# unlinked crops. Deliberately used ONLY when the strict match finds no candidate for
# that crop type on that page, so pages that already link are bit-for-bit unaffected.
_FIG_WORDS = ("figure", "fig")
_TAB_WORDS = ("table", "tab")
_FIRST_TOKEN = re.compile(r"^\s*[\(\[]?\s*([A-Za-z]{2,9})")
#: 0.65 admits "figukz"->figure (0.67) while "table"/"figure" stay far apart (0.36), so
#: the fallback cannot confuse the two types. No trailing number is required: the SARS
#: caption OCRs as "Figure Spatial clusters ..." with the numeral destroyed, and it is a
#: known-good link that a number requirement would break.
_FUZZ_MIN = 0.65


def _fuzzy_label(text: str, words: tuple) -> bool:
    m = _FIRST_TOKEN.match(text)
    if not m:
        return False
    tok = m.group(1).lower()
    return any(difflib.SequenceMatcher(None, tok, w).ratio() >= _FUZZ_MIN for w in words)


def _h_overlap(a: BBox, b: BBox) -> float:
    return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))


def _h_gap(a: BBox, b: BBox) -> float:
    return max(0.0, max(a.x0, b.x0) - min(a.x1, b.x1))


def _penalty(crop: Chunk, cap: Chunk, prefer_below: bool) -> float:
    gap = (cap.bbox.y0 - crop.bbox.y1) if prefer_below else (crop.bbox.y0 - cap.bbox.y1)
    wrong_side = 0.0 if gap >= -5 else 1000.0
    # A caption sitting in the ADJACENT COLUMN of a two-column page has zero horizontal
    # overlap with its crop but is still its caption. The old flat 500 exceeded
    # _MAX_PENALTY on its own, so such a pair could never link even at zero vertical gap
    # — 20 of the 500-page corpus's unlinked crops were rejected for exactly that. Scaled
    # by how far away it actually is, so a caption across the page still loses.
    no_overlap = 0.0 if _h_overlap(crop.bbox, cap.bbox) > 0 else 250.0 + _h_gap(crop.bbox, cap.bbox)
    return wrong_side + no_overlap + abs(gap)


class CaptionLinker:
    def __init__(self, store: SQLiteChunkStore) -> None:
        self.store = store

    def link(self) -> dict:
        stats = {"figure": [0, 0], "table": [0, 0]}  # [linked, total]
        fuzzy_used = 0
        updates: list[tuple[str, str]] = []
        self.store.clear_captions()  # idempotent: a re-run must not keep stale links
        for page_id in self.store.page_ids():
            chunks = self.store.get_by_page(page_id)
            texts = [c for c in chunks if c.region_type == RegionType.TEXT and c.text.strip()]
            fig_caps = [c for c in texts if _FIG_RE.match(c.text.strip())]
            tab_caps = [c for c in texts if _TAB_RE.match(c.text.strip())]
            # fallback ONLY where the strict prefix found nothing, so existing links
            # are untouched and the change can only add
            if not fig_caps:
                fig_caps = [c for c in texts if _fuzzy_label(c.text.strip(), _FIG_WORDS)]
                fuzzy_used += bool(fig_caps)
            if not tab_caps:
                tab_caps = [c for c in texts if _fuzzy_label(c.text.strip(), _TAB_WORDS)]
                fuzzy_used += bool(tab_caps)
            for crop in chunks:
                if crop.region_type not in (RegionType.FIGURE, RegionType.TABLE):
                    continue
                is_fig = crop.region_type == RegionType.FIGURE
                key = "figure" if is_fig else "table"
                stats[key][1] += 1
                cands = fig_caps if is_fig else tab_caps
                if not cands:
                    continue
                scored = sorted(cands, key=lambda c: _penalty(crop, c, prefer_below=is_fig))
                best = scored[0]
                if _penalty(crop, best, prefer_below=is_fig) <= _MAX_PENALTY:
                    updates.append((crop.chunk_id, best.chunk_id))
                    stats[key][0] += 1
        self.store.set_captions(updates)
        return {
            "figure_linked": stats["figure"][0], "figure_total": stats["figure"][1],
            "table_linked": stats["table"][0], "table_total": stats["table"][1],
            "pages_using_fuzzy_label": fuzzy_used,
        }
