"""Entry point: vision-read every TABLE crop once into clean retrieval text.

    python -m scripts.transcribe_tables --config configs/enhanced_vqa.yaml

Phase 8 finding #1: table crops were unfindable because their OCR is linearised and
loses the grid, so the region's embedding is weak (gold table at cosine 0.536 while
plain prose on the same topic scored 0.693 — the table lost to the surrounding text).
This pass stores a clean transcription in `chunks.vqa_text`; `build_indices.py` then
embeds THAT as the table's retrieval text, so a table becomes findable by its own
contents.

Ingest-time and cached in SQLite: paid once per corpus, not per query. Idempotent —
already-transcribed rows are skipped unless --force. Figures are not transcribed (a
plot has no rows to read); only TABLE regions.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from platform_core.config import AppConfig
from platform_core.generation.prompts import TRANSCRIBE_SYSTEM, TRANSCRIBE_USER
from platform_core.llm.openai_client import OpenAIClient
from platform_core.stores.chunk_store import SQLiteChunkStore
from platform_core.types import RegionType

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
NO_TABLE = "NO_TABLE"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=R + "/configs/enhanced_vqa.yaml")
    ap.add_argument("--force", action="store_true", help="re-transcribe rows that already have text")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    print(cfg.describe())

    store = SQLiteChunkStore(cfg.paths.chunk_db)
    store.init_schema()  # adds the vqa_text column on pre-phase-9 DBs
    tables = [c for c in store.iter_chunks([RegionType.TABLE]) if c.image_path]
    todo = tables if args.force else [c for c in tables if not (c.vqa_text or "").strip()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"\ntable crops: {len(tables)}   to transcribe: {len(todo)}"
          f"{' (--force)' if args.force else ' (skipping already-done)'}")

    llm = OpenAIClient(model=cfg.generation.vqa_model)
    pairs, failed = [], []
    for i, c in enumerate(todo, 1):
        try:
            txt = llm.answer_with_images(TRANSCRIBE_SYSTEM, TRANSCRIBE_USER, [c.image_path]).strip()
        except Exception as e:  # keep going; report at the end rather than half-writing
            failed.append((c.chunk_id, f"{type(e).__name__}: {e}"))
            continue
        if txt.upper().startswith(NO_TABLE) or not txt:
            failed.append((c.chunk_id, NO_TABLE))
            continue
        pairs.append((c.chunk_id, txt))
        if i % 10 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} …")
    if pairs:
        store.set_vqa_text(pairs)

    print(f"\ntranscribed: {len(pairs)}   failed/NO_TABLE: {len(failed)}")
    for cid, why in failed:
        print(f"  ! {cid}: {why}")
    print(f"rows with transcription now: {store.count_vqa_text()}")

    if pairs:
        cid, txt = pairs[0]
        print(f"\nsample — {cid}\n  OCR : {' '.join((store.get(cid).text or '').split())[:200]}")
        print(f"  VQA : {' '.join(txt.split())[:200]}")

    u = llm.total_usage
    print(f"\nACTUAL API cost: ${u['prompt_tokens'] * 2.5 / 1e6 + u['completion_tokens'] * 10 / 1e6:.3f}")
    store.close()


if __name__ == "__main__":
    main()
