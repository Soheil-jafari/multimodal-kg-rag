"""Entry point: build derived indices from the canonical chunk store.

Usage:
    python -m scripts.build_indices --query "How did SARS spread geographically?"

Steps (BGE and CLIP are loaded ONE AT A TIME to fit a 4GB GPU):
  0. link figure/table crops to their caption text region (caption_id).
  1. BGE text embeddings -> FAISS text index for BOTH chunkers (layout + naive).
  2. text retrieval smoke (while BGE is loaded).
  3. free BGE; CLIP image embeddings for figure+table crops -> FAISS image index.
  4. image retrieval smoke (while CLIP is loaded).
Reports index sizes, caption-link coverage, and the smoke results.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from platform_core.config import AppConfig
from platform_core.ingestion.captions import CaptionLinker
from platform_core.ingestion.chunking import LayoutChunker, NaiveChunker
from platform_core.llm.embeddings import SentenceTransformerEmbedder, make_image_embedder
from platform_core.stores.chunk_store import SQLiteChunkStore
from platform_core.stores.vector_store import FaissVectorIndex
from platform_core.types import RegionType

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
DEFAULT_CONFIG = R + "/configs/enhanced.yaml"


def build_units(store, chunker):
    units = []
    for page_id in store.page_ids():
        units.extend(chunker.chunk_page(store.get_by_page(page_id)))
    return units


def write_manifest(path, units):
    with open(path + ".units.jsonl", "w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps({
                "unit_id": u.unit_id, "source_chunk_ids": u.source_chunk_ids,
                "page_id": u.page_id, "region_type": u.region_type,
            }) + "\n")


def free(*objs):
    for o in objs:
        del o
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def preview(text, n=64):
    return " ".join(text.split())[:n] + ("…" if len(" ".join(text.split())) > n else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--query", default="How did SARS spread geographically over time?")
    ap.add_argument("--image-only", action="store_true",
                    help="rebuild just this encoder's image index (text indices untouched)")
    ap.add_argument("--captions-only", action="store_true",
                    help="re-run caption linking and report coverage, then exit. The "
                         "chunkers never read caption_id, so the FAISS indices do not "
                         "depend on it and stay valid.")
    args = ap.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    print(cfg.describe())
    m, out = cfg.models, cfg.paths.index_dir
    os.makedirs(out, exist_ok=True)

    store = SQLiteChunkStore(cfg.paths.chunk_db)
    store.init_schema()  # also migrates caption_id column onto phase-1 DBs

    # --- 0. caption linking (no models) ---
    cov = CaptionLinker(store).link()
    if args.captions_only:
        ft, fl = cov["figure_total"], cov["figure_linked"]
        tt, tl = cov["table_total"], cov["table_linked"]
        tot, lin = ft + tt, fl + tl
        print("\ncaption-link coverage (crop -> caption text region):")
        print(f"  figures: {fl}/{ft} linked ({100*fl/ft:.0f}%)" if ft else "  figures: none")
        print(f"  tables : {tl}/{tt} linked ({100*tl/tt:.0f}%)" if tt else "  tables : none")
        print(f"  overall: {lin}/{tot} linked ({100*lin/tot:.0f}%)" if tot else "  overall: none")
        print(f"  pages where the fuzzy label fallback supplied the candidates: "
              f"{cov['pages_using_fuzzy_label']}")
        store.close()
        return

    text_index = naive_index = None
    units_layout: list = []
    units_naive: list = []
    top_text: list = []
    text_dim = 0
    n_tablevqa = 0
    if not args.image_only:
        # --- 1. BGE text indices (layout + naive) ---
        bge = SentenceTransformerEmbedder(m.text_embedding_model, m.device, m.embed_batch_size)
        layout = LayoutChunker(count_tokens=bge.token_len)
        naive = NaiveChunker(count_tokens=bge.token_len)
        units_layout = build_units(store, layout)
        units_naive = build_units(store, naive)

        layout_vecs = bge.embed([u.text for u in units_layout])
        text_index = FaissVectorIndex(int(layout_vecs.shape[1]))
        text_index.add([u.unit_id for u in units_layout], layout_vecs)
        text_index.save(os.path.join(out, "text_layout"))
        write_manifest(os.path.join(out, "text_layout"), units_layout)

        naive_vecs = bge.embed([u.text for u in units_naive])
        naive_index = FaissVectorIndex(int(naive_vecs.shape[1]))
        naive_index.add([u.unit_id for u in units_naive], naive_vecs)
        naive_index.save(os.path.join(out, "text_naive"))
        write_manifest(os.path.join(out, "text_naive"), units_naive)

        # --- 1b. layout variant indexing tables by their transcription, when any
        # exist. Built as a SEPARATE index so text_layout stays the published
        # baseline and use_table_vqa_text is a pure selection.
        n_tablevqa = store.count_vqa_text().get(RegionType.TABLE.value, 0)
        if n_tablevqa:
            def table_text(c):
                # Caption FIRST: measured in phase 9, the caption carries the query's
                # vocabulary and the transcription carries only values, so caption +
                # transcription beats either alone (q061 0.579 vs OCR 0.536 / trans 0.459).
                body = c.retrieval_text(prefer_vqa=True)
                if c.caption_id and c.image_path:
                    try:
                        return store.get(c.caption_id).text + "\n\n" + body
                    except KeyError:
                        pass
                return body

            tvqa = LayoutChunker(count_tokens=bge.token_len, text_of=table_text)
            units_tvqa = build_units(store, tvqa)
            tvqa_vecs = bge.embed([u.text for u in units_tvqa])
            tvqa_index = FaissVectorIndex(int(tvqa_vecs.shape[1]))
            tvqa_index.add([u.unit_id for u in units_tvqa], tvqa_vecs)
            tvqa_index.save(os.path.join(out, "text_layout_tablevqa"))
            write_manifest(os.path.join(out, "text_layout_tablevqa"), units_tvqa)

        # --- 2. text smoke (BGE loaded) ---
        top_text = text_index.search(bge.embed_query(args.query), 5)
        text_dim = int(layout_vecs.shape[1])
        free(bge)

    # --- 3. image index (figure + table crops), encoder chosen by config ---
    crops = [c for c in store.iter_chunks([RegionType.FIGURE, RegionType.TABLE]) if c.image_path]
    img_enc = make_image_embedder(m.image_embedding_model, m.device, m.embed_batch_size)
    img_base = cfg.image_index_path()
    image_index = None
    image_dim = 0
    top_img: list = []
    if crops:
        img_vecs = img_enc.embed_images([c.image_path for c in crops])
        image_dim = int(img_vecs.shape[1])
        image_index = FaissVectorIndex(image_dim)
        image_index.add([c.chunk_id for c in crops], img_vecs)
        image_index.save(img_base)
        # --- 4. image smoke (encoder loaded) ---
        top_img = image_index.search(img_enc.embed_text([args.query])[0], 3)
    free(img_enc)

    # ================= report =================
    print("\n================ PHASE 2: DERIVED INDICES ================")
    print("index sizes (vectors):")
    if text_index is not None and naive_index is not None:
        print(f"  text_layout (enhanced, BGE dim {text_dim}): {len(text_index)}  units")
        print(f"  text_naive  (baseline, BGE dim {text_dim}): {len(naive_index)}  units")
        print(f"  [chunker effect] layout={len(units_layout)} units vs naive={len(units_naive)} windows "
              f"over the same corpus")
        if n_tablevqa:
            print(f"  text_layout_tablevqa (tables indexed by transcription, "
                  f"{n_tablevqa} tables): built")
    else:
        print("  text indices: SKIPPED (--image-only)")
    print(f"  {os.path.basename(img_base):17s} ({m.image_embedding_model}, dim {image_dim}, fig+table crops):"
          f" {len(image_index) if image_index else 0}  vectors")

    ft, fl = cov["figure_total"], cov["figure_linked"]
    tt, tl = cov["table_total"], cov["table_linked"]
    tot, lin = ft + tt, fl + tl
    print("\ncaption-link coverage (crop -> caption text region):")
    print(f"  figures: {fl}/{ft} linked ({100*fl/ft:.0f}%)" if ft else "  figures: none")
    print(f"  tables : {tl}/{tt} linked ({100*tl/tt:.0f}%)" if tt else "  tables : none")
    print(f"  overall: {lin}/{tot} linked ({100*lin/tot:.0f}%)" if tot else "  overall: none")

    print(f"\nretrieval smoke test — query: {args.query!r}")
    if top_text:
        print("  top-5 TEXT chunk_ids (BGE):")
        for uid, score in top_text:
            base = uid.split("#")[0]
            try:
                c = store.get(base)
                rt, tx = c.region_type.value, preview(c.text)
            except KeyError:
                rt, tx = "?", ""
            print(f"    {score:5.3f}  {uid:24s} [{rt}] {tx}")
    print(f"  top-3 IMAGE chunk_ids ({m.image_embedding_model}):")
    for cid, score in top_img:
        c = store.get(cid)
        cap = ""
        if c.caption_id:
            try:
                cap = "caption=" + preview(store.get(c.caption_id).text, 48)
            except KeyError:
                cap = "caption=<missing>"
        print(f"    {score:5.3f}  {cid:24s} [{c.region_type.value}] {cap}")

    store.close()


if __name__ == "__main__":
    main()
