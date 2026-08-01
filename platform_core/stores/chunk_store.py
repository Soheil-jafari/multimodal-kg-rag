"""SQLite implementation of the canonical chunk store — THE source of truth.

Implements :class:`~platform_core.stores.base.ChunkStore`. One table, one row per
OCR'd layout region. ``bbox`` is stored as four REAL columns (x0,y0,x1,y1) so the
evaluation layer can do region-coverage overlap in SQL. ``paper_id`` and
``page_id`` are stored separately (equal today). ``caption_id`` links a
figure/table crop to its caption text region (populated in the index phase).
"""
from __future__ import annotations

import os
import sqlite3
from collections.abc import Collection, Iterator, Sequence
from typing import Optional

from platform_core.stores.base import ChunkStore
from platform_core.types import BBox, Chunk, RegionType

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    paper_id    TEXT NOT NULL,
    page_id     TEXT NOT NULL,
    page        INTEGER NOT NULL,
    region_type TEXT NOT NULL,
    x0 REAL NOT NULL, y0 REAL NOT NULL, x1 REAL NOT NULL, y1 REAL NOT NULL,
    page_w INTEGER, page_h INTEGER,
    text        TEXT,
    image_path  TEXT,
    caption_id  TEXT,
    vqa_text    TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);
CREATE INDEX IF NOT EXISTS idx_chunks_page  ON chunks(page_id);
CREATE INDEX IF NOT EXISTS idx_chunks_rtype ON chunks(region_type);
"""


def _row_to_chunk(r: sqlite3.Row) -> Chunk:
    keys = r.keys()
    return Chunk(
        chunk_id=r["chunk_id"],
        paper_id=r["paper_id"],
        page_id=r["page_id"],
        page=r["page"],
        region_type=RegionType(r["region_type"]),
        bbox=BBox(r["x0"], r["y0"], r["x1"], r["y1"]),
        text=r["text"] or "",
        image_path=r["image_path"],
        caption_id=r["caption_id"] if "caption_id" in keys else None,
        page_width=r["page_w"],
        page_height=r["page_h"],
        vqa_text=r["vqa_text"] if "vqa_text" in keys else None,
    )


class SQLiteChunkStore(ChunkStore):
    """Canonical chunks in SQLite."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def init_schema(self) -> None:
        self.conn.executescript(_SCHEMA)
        # migrate older DBs (created before caption_id existed)
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(chunks)")}
        if "caption_id" not in cols:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN caption_id TEXT")
        if "vqa_text" not in cols:  # added in phase 9 (table transcription)
            self.conn.execute("ALTER TABLE chunks ADD COLUMN vqa_text TEXT")
        self.conn.commit()

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        self.conn.executemany(
            """INSERT OR REPLACE INTO chunks
               (chunk_id, paper_id, page_id, page, region_type,
                x0, y0, x1, y1, page_w, page_h, text, image_path, caption_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    c.chunk_id, c.paper_id, c.page_id, c.page, c.region_type.value,
                    c.bbox.x0, c.bbox.y0, c.bbox.x1, c.bbox.y1,
                    c.page_width, c.page_height, c.text, c.image_path, c.caption_id,
                )
                for c in chunks
            ],
        )
        self.conn.commit()

    def get(self, chunk_id: str) -> Chunk:
        row = self.conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            raise KeyError(chunk_id)
        return _row_to_chunk(row)

    def get_many(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self.conn.execute(
            f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", tuple(chunk_ids)
        ).fetchall()
        by_id = {r["chunk_id"]: _row_to_chunk(r) for r in rows}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def iter_chunks(
        self, region_types: Optional[Collection[RegionType]] = None
    ) -> Iterator[Chunk]:
        if region_types:
            vals = [rt.value for rt in region_types]
            placeholders = ",".join("?" * len(vals))
            cur = self.conn.execute(
                f"SELECT * FROM chunks WHERE region_type IN ({placeholders}) "
                "ORDER BY chunk_id",
                tuple(vals),
            )
        else:
            cur = self.conn.execute("SELECT * FROM chunks ORDER BY chunk_id")
        for row in cur:
            yield _row_to_chunk(row)

    def paper_ids(self) -> list[str]:
        return [r["paper_id"] for r in self.conn.execute(
            "SELECT DISTINCT paper_id FROM chunks ORDER BY paper_id")]

    # --- convenience helpers (beyond the ABC) ---
    def page_ids(self) -> list[str]:
        return [r["page_id"] for r in self.conn.execute(
            "SELECT DISTINCT page_id FROM chunks ORDER BY page_id")]

    def get_by_page(self, page_id: str) -> list[Chunk]:
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE page_id = ? ORDER BY chunk_id", (page_id,)
        ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def crops_for_captions(self, caption_ids: Sequence[str]) -> list[Chunk]:
        """Figure/table crops whose caption_id is in the given set (caption -> crop)."""
        if not caption_ids:
            return []
        ph = ",".join("?" * len(caption_ids))
        rows = self.conn.execute(
            f"SELECT * FROM chunks WHERE caption_id IN ({ph})", tuple(caption_ids)
        ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def count_by_region_type(self) -> dict[str, int]:
        return {r["region_type"]: r["n"] for r in self.conn.execute(
            "SELECT region_type, COUNT(*) AS n FROM chunks GROUP BY region_type")}

    def set_vqa_text(self, pairs: Sequence[tuple[str, str]]) -> None:
        """Batch-store vision-read transcriptions for (chunk_id, text) pairs."""
        self.conn.executemany(
            "UPDATE chunks SET vqa_text = ? WHERE chunk_id = ?",
            [(txt, cid) for cid, txt in pairs],
        )
        self.conn.commit()

    def count_vqa_text(self) -> dict[str, int]:
        """How many rows per region type carry a transcription (build-report line)."""
        return {r["region_type"]: r["n"] for r in self.conn.execute(
            "SELECT region_type, COUNT(*) AS n FROM chunks "
            "WHERE vqa_text IS NOT NULL AND TRIM(vqa_text) <> '' GROUP BY region_type")}

    def clear_captions(self) -> None:
        """Drop every caption link, so re-linking is idempotent rather than additive.

        ``set_captions`` only writes the pairs it is given, so without this a crop that
        the current matcher would NOT link keeps a link an earlier matcher gave it, and
        the reported coverage describes no matcher that exists.
        """
        self.conn.execute("UPDATE chunks SET caption_id = NULL")
        self.conn.commit()

    def set_captions(self, pairs: Sequence[tuple[str, str]]) -> None:
        """Batch-set caption_id for (crop_chunk_id, caption_chunk_id) pairs."""
        self.conn.executemany(
            "UPDATE chunks SET caption_id = ? WHERE chunk_id = ?",
            [(caption_id, crop_id) for crop_id, caption_id in pairs],
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
