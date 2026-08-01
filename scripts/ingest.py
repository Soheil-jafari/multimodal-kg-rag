"""Entry point: PubLayNet shards -> OCR -> canonical chunk store.

Usage:
    python -m scripts.ingest --config configs/scale500.yaml

Which shards and how many pages each come from the config's `ingest` section, not
from argv — the corpus is part of the experiment, so it is declared alongside the
flags that will run over it (see EXPERIMENT_LOG Phase 7.5 on config-driven).

Wires loader + grouper + OCR into the chunk store, then prints a verification
report: per-shard and per-region-type counts, an OCR empty/low-quality rate, one
sample page's rows, and a mix of OCR text samples (text / title / table) to judge
quality. Resumable — pages already in the store are skipped.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")  # OCR text may contain non-cp1252 chars
except Exception:
    pass

from platform_core.config import AppConfig
from platform_core.ingestion.ingestor import Ingestor
from platform_core.ingestion.ocr import RapidOCRRegionOCR
from platform_core.ingestion.paper_grouping import PageAsPaperGrouper
from platform_core.ingestion.publaynet_loader import PubLayNetLoader
from platform_core.stores.chunk_store import SQLiteChunkStore

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
DEFAULT_CONFIG = R + "/configs/default.yaml"
OCR_TYPES = ["text", "title", "list", "table"]  # figure = crop-only, not OCR'd


def looks_garbled(text: str) -> bool:
    """Rough low-quality proxy: too few letters, or symbol/number soup."""
    s = text.strip()
    dense = s.replace(" ", "").replace("\n", "")
    if not dense:
        return True
    letters = sum(ch.isalpha() for ch in dense)
    alnum = sum(ch.isalnum() for ch in dense)
    return letters < 3 or (alnum / len(dense)) < 0.5


def preview(text: str, n: int = 70) -> str:
    one = " ".join(text.split())
    return one[:n] + ("…" if len(one) > n else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--limit", type=int, default=None,
                    help="override ingest.pages_per_shard (smoke runs)")
    args = ap.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    print(cfg.describe())
    per_shard = args.limit or cfg.ingest.pages_per_shard

    os.makedirs(os.path.dirname(cfg.paths.chunk_db), exist_ok=True)
    store = SQLiteChunkStore(cfg.paths.chunk_db)
    store.init_schema()
    done = set(store.page_ids())  # resume: never re-OCR a page already stored
    if done:
        print(f"\nresuming — {len(done)} pages already in {cfg.paths.chunk_db}")

    ocr = RapidOCRRegionOCR()  # one engine, reused across shards
    grouper = PageAsPaperGrouper()
    t0 = time.time()
    totals = collections.Counter()
    print(f"\ningesting {len(cfg.ingest.shards)} shard(s) x {per_shard} pages ...")
    for s in cfg.ingest.shards:
        loader = PubLayNetLoader(cfg.ingest.repo, s["file"], s["tag"])
        ingestor = Ingestor(loader, grouper, ocr, store, cfg.paths.crops_dir)
        t1 = time.time()
        st = ingestor.run(limit=per_shard, skip_page_ids=done, progress_every=25)
        totals.update(st)
        print(f"  {s['tag']}: pages={st['pages']} chunks={st['chunks']} "
              f"skipped={st['skipped']} ({time.time() - t1:.0f}s)")
    mins = (time.time() - t0) / 60
    print(f"  TOTAL new: pages={totals['pages']} chunks={totals['chunks']} "
          f"skipped={totals['skipped']}  [{mins:.1f} min]")

    chunks = list(store.iter_chunks())
    by_type: dict[str, list] = {}
    for c in chunks:
        by_type.setdefault(c.region_type.value, []).append(c)

    print("\n== corpus totals ==")
    print(f"  pages in store : {len(store.page_ids())}")
    print(f"  chunks in store: {len(chunks)}")
    print("\n== pages per shard ==")
    for tag, k in sorted(collections.Counter(
            p.split(":", 1)[0] for p in store.page_ids()).items()):
        print(f"  {tag:10s} {k}")

    print("\n== region-type counts ==")
    for t, cs in sorted(by_type.items()):
        print(f"  {t:7s} {len(cs)}")

    print("\n== OCR quality (per type) ==")
    print(f"  {'type':7s} {'n':>5s} {'empty%':>7s} {'lowqual%':>9s}")
    tot_n = tot_empty = tot_low = 0
    for t in OCR_TYPES:
        cs = by_type.get(t, [])
        n = len(cs)
        empty = sum(1 for c in cs if not c.text.strip())
        low = sum(1 for c in cs if c.text.strip() and looks_garbled(c.text))
        tot_n += n; tot_empty += empty; tot_low += low
        if n:
            print(f"  {t:7s} {n:5d} {100*empty/n:6.1f}% {100*low/n:8.1f}%")
    if tot_n:
        print(f"  {'ALL':7s} {tot_n:5d} {100*tot_empty/tot_n:6.1f}% {100*tot_low/tot_n:8.1f}%")
    figs = by_type.get("figure", [])
    print(f"  figure  {len(figs)} regions: crop-only (not OCR'd); "
          f"{sum(1 for c in figs if c.image_path)} crops saved")

    # sample page: prefer one with a table (and a figure), else first page
    pages = store.page_ids()

    def has(pid, t):
        return any(c.region_type.value == t for c in store.get_by_page(pid))

    sample = (next((p for p in pages if has(p, "table") and has(p, "figure")), None)
              or next((p for p in pages if has(p, "table")), None)
              or (pages[0] if pages else None))
    if sample:
        print(f"\n== chunk-store rows for sample page: {sample} ==")
        print(f"  {'chunk_id':24s} {'type':7s} {'bbox(x0,y0,x1,y1)':26s} {'img':4s} text")
        for c in store.get_by_page(sample):
            b = c.bbox
            bbox = f"({b.x0:.0f},{b.y0:.0f},{b.x1:.0f},{b.y1:.0f})"
            img = "yes" if c.image_path else "-"
            print(f"  {c.chunk_id:24s} {c.region_type.value:7s} {bbox:26s} {img:4s} {preview(c.text, 46)}")

    print("\n== OCR samples (mix, judge quality per type) ==")
    samples = []
    texts = sorted((c for c in by_type.get("text", []) if c.text.strip()),
                   key=lambda c: len(c.text), reverse=True)
    samples += [("text", c) for c in texts[:2]]
    title = next((c for c in by_type.get("title", []) if c.text.strip()), None)
    if title:
        samples.append(("title", title))
    tables = [c for c in by_type.get("table", []) if c.text.strip()][:2]
    samples += [("table", c) for c in tables]
    for label, c in samples:
        print(f"\n  [{label}] {c.chunk_id}")
        body = " ".join(c.text.split())
        print("   " + (body[:500] + ("…" if len(body) > 500 else "")))

    store.close()


if __name__ == "__main__":
    main()
