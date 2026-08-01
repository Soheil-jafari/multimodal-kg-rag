"""Entry point: the Phase-2 figure smoke test, run against a chosen image encoder.

    python -m scripts.figure_smoke --image-model clip-ViT-B-32
    python -m scripts.figure_smoke --image-model microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

Scores ONE text query against ALL figure+table crops with the encoder's text tower
and reports the full similarity distribution — the Phase-2 finding was that general
CLIP's distribution is so tight (0.19-0.25, sigma 0.013) that ranking #1 is a coin
toss, so the spread and the #1-vs-#2 margin are the numbers that matter, not the
top-1 hit alone.

Embeds the crops live rather than reading the FAISS index, so the query and the
crops are guaranteed to come from the same encoder.
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

import numpy as np

from platform_core.llm.embeddings import make_image_embedder
from platform_core.stores.chunk_store import SQLiteChunkStore
from platform_core.types import RegionType

# The Phase-2 query and the crop that truly answers it (SARS spatial-spread figure).
QUERY = "How did SARS spread geographically over time?"
GOLD_CROP = "val00000:415624:r11"


def preview(text: str, n: int = 48) -> str:
    flat = " ".join(text.split())
    return flat[:n] + ("…" if len(flat) > n else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="artifacts/dev/chunks.sqlite")
    ap.add_argument("--image-model", default="clip-ViT-B-32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--query", default=QUERY)
    ap.add_argument("--gold", default=GOLD_CROP)
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    store = SQLiteChunkStore(args.db)
    crops = [c for c in store.iter_chunks([RegionType.FIGURE, RegionType.TABLE])
             if c.image_path]

    enc = make_image_embedder(args.image_model, args.device, args.batch)
    img = np.asarray(enc.embed_images([c.image_path for c in crops]), dtype="float32")
    qv = np.asarray(enc.embed_text([args.query])[0], dtype="float32")
    sims = img @ qv

    order = np.argsort(-sims)
    ranked = [(crops[i].chunk_id, float(sims[i])) for i in order]
    rank_of_gold = next(r for r, (cid, _) in enumerate(ranked, 1) if cid == args.gold)
    top1_id, top1 = ranked[0]
    _, top2 = ranked[1]

    print(f"\n================ FIGURE SMOKE — {args.image_model} ================")
    print(f"query : {args.query!r}")
    print(f"crops : {len(crops)}  (figure+table, dim {img.shape[1]})")
    print(f"\ndistribution over all {len(crops)} crops:")
    print(f"  min {sims.min():.4f}  mean {sims.mean():.4f}  max {sims.max():.4f}  "
          f"sigma {sims.std():.4f}  range {sims.max() - sims.min():.4f}")
    print(f"  #1 vs #2 margin: {top1 - top2:.4f}")
    z = (sims.max() - sims.mean()) / sims.std() if sims.std() else 0.0
    print(f"  top-1 z-score (how far #1 stands out): {z:.2f} sigma")

    print(f"\ngold crop {args.gold}: rank {rank_of_gold}/{len(crops)}  "
          f"score {dict(ranked)[args.gold]:.4f}  "
          f"{'WINS' if ranked[0][0] == args.gold else 'LOSES to ' + top1_id}")

    print(f"\ntop-{args.show}:")
    for r, (cid, s) in enumerate(ranked[:args.show], 1):
        c = store.get(cid)
        cap = ""
        if c.caption_id:
            try:
                cap = preview(store.get(c.caption_id).text)
            except KeyError:
                cap = "<caption missing>"
        mark = " <-- GOLD" if cid == args.gold else ""
        print(f"  {r:2d}. {s:7.4f}  {cid:22s} [{c.region_type.value:6s}] {cap}{mark}")

    store.close()


if __name__ == "__main__":
    main()
