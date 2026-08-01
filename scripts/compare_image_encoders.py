"""Entry point: before/after comparison of two image encoders on the same crops.

    python -m scripts.compare_image_encoders --device cpu

Runs the Phase-2 figure smoke query against every figure+table crop under each
encoder in turn (loaded ONE AT A TIME, freed between — 4GB-GPU discipline) and
prints the score distributions side by side.

Phase 2 found general CLIP's distribution so tight that top-1 was a coin toss, so
the comparison reports spread (sigma, range), the #1-vs-#2 margin, and the top-1
z-score — not just whether the gold crop ranked first. It also ranks every crop on
the gold crop's OWN page, because "did the right page win" and "did the exact
labelled crop win" are different questions and can disagree.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from platform_core.llm.embeddings import BIOMEDCLIP_MODEL, CLIP_MODEL, make_image_embedder
from platform_core.stores.chunk_store import SQLiteChunkStore
from platform_core.types import RegionType

QUERY = "How did SARS spread geographically over time?"
GOLD_CROP = "val00000:415624:r11"


def preview(text: str, n: int = 44) -> str:
    flat = " ".join(text.split())
    return flat[:n] + ("…" if len(flat) > n else "")


def score_all(model_name, crops, queries, device, batch):
    """Load one encoder, score every query vs all crops, free it.

    Returns (n_queries, n_crops). Crops are embedded once per encoder, so adding
    queries is nearly free.
    """
    enc = make_image_embedder(model_name, device, batch)
    img = np.asarray(enc.embed_images([c.image_path for c in crops]), dtype="float32")
    qv = np.asarray(enc.embed_text(list(queries)), dtype="float32")
    sims = qv @ img.T
    del enc
    gc.collect()
    return sims


def load_figure_questions(path, crop_ids):
    """Gold questions whose answer lives in a crop — the image path's actual job."""
    import json

    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("category") not in ("figure", "figure_value"):
                continue
            golds = [c for c in (r.get("supporting_chunk_ids") or []) if c in crop_ids]
            if golds:  # some figure questions are answered by the caption region only
                out.append({"qid": r["qid"], "category": r["category"],
                            "question": r["question"], "golds": golds})
    return out


def stats(sims, crops, gold):
    order = np.argsort(-sims)
    ranked = [(crops[i].chunk_id, float(sims[i])) for i in order]
    by_id = dict(ranked)
    rank = {cid: r for r, (cid, _) in enumerate(ranked, 1)}
    return {
        "min": float(sims.min()), "mean": float(sims.mean()), "max": float(sims.max()),
        "sigma": float(sims.std()), "range": float(sims.max() - sims.min()),
        "margin": ranked[0][1] - ranked[1][1],
        "z": float((sims.max() - sims.mean()) / sims.std()) if sims.std() else 0.0,
        "top1_id": ranked[0][0], "gold_rank": rank[gold], "gold_score": by_id[gold],
        "ranked": ranked, "rank": rank, "by_id": by_id,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="artifacts/dev/chunks.sqlite")
    ap.add_argument("--before", default=CLIP_MODEL)
    ap.add_argument("--after", default=BIOMEDCLIP_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--query", default=QUERY)
    ap.add_argument("--gold", default=GOLD_CROP)
    ap.add_argument("--show", type=int, default=10)
    ap.add_argument("--gold-file",
                    default="domain_packs/biomed/gold/gold_set.jsonl",
                    help="also score every image-dependent gold question")
    args = ap.parse_args()

    store = SQLiteChunkStore(args.db)
    crops = [c for c in store.iter_chunks([RegionType.FIGURE, RegionType.TABLE])
             if c.image_path]
    gold_page = store.get(args.gold).page_id
    crop_ids = {c.chunk_id for c in crops}

    # image-dependent gold questions, each with the crop(s) it truly answers
    gold_qs = load_figure_questions(args.gold_file, crop_ids)
    queries = [args.query] + [q["question"] for q in gold_qs]

    runs, gold_runs = {}, {}
    for label, name in (("BEFORE", args.before), ("AFTER", args.after)):
        sims = score_all(name, crops, queries, args.device, args.batch)
        runs[label] = stats(sims[0], crops, args.gold)
        gold_runs[label] = sims[1:]

    b, a = runs["BEFORE"], runs["AFTER"]
    print(f"\n=========== IMAGE ENCODER BEFORE/AFTER — {len(crops)} crops ===========")
    print(f"query: {args.query!r}")
    print(f"BEFORE: {args.before}")
    print(f"AFTER : {args.after}")

    print("\n| metric | BEFORE | AFTER | change |")
    print("|---|---|---|---|")
    rows = [
        ("min", "min", "{:.4f}"), ("mean", "mean", "{:.4f}"), ("max", "max", "{:.4f}"),
        ("sigma (spread)", "sigma", "{:.4f}"), ("range (max-min)", "range", "{:.4f}"),
        ("#1 vs #2 margin", "margin", "{:.4f}"), ("top-1 z-score", "z", "{:.2f}"),
    ]
    for label, key, fmt in rows:
        bv, av = b[key], a[key]
        mult = f"{av / bv:.1f}x" if bv else "-"
        print(f"| {label} | {fmt.format(bv)} | {fmt.format(av)} | {mult} |")
    print(f"| gold crop rank | {b['gold_rank']}/{len(crops)} | {a['gold_rank']}/{len(crops)} | |")
    print(f"| gold crop score | {b['gold_score']:.4f} | {a['gold_score']:.4f} | |")
    print(f"| top-1 crop | {b['top1_id']} | {a['top1_id']} | |")

    for label in ("BEFORE", "AFTER"):
        r = runs[label]
        print(f"\n{label} top-{args.show}:")
        for i, (cid, s) in enumerate(r["ranked"][:args.show], 1):
            c = store.get(cid)
            cap = ""
            if c.caption_id:
                try:
                    cap = preview(store.get(c.caption_id).text)
                except KeyError:
                    cap = "<caption missing>"
            mark = "  <-- GOLD" if cid == args.gold else ""
            print(f"  {i:2d}. {s:7.4f}  {cid:22s} [{c.region_type.value:6s}] {cap}{mark}")

    # page-level diagnostic: same-page crops are all candidate answers
    page_crops = [c for c in crops if c.page_id == gold_page]
    print(f"\nall crops on the gold page ({gold_page}) — rank / score under each encoder:")
    print("| crop | type | caption | BEFORE | AFTER |")
    print("|---|---|---|---|---|")
    for c in sorted(page_crops, key=lambda c: a["rank"][c.chunk_id]):
        cap = ""
        if c.caption_id:
            try:
                cap = preview(store.get(c.caption_id).text, 40)
            except KeyError:
                cap = "<missing>"
        mark = " **(gold)**" if c.chunk_id == args.gold else ""
        print(f"| {c.chunk_id}{mark} | {c.region_type.value} | {cap} "
              f"| #{b['rank'][c.chunk_id]} ({b['by_id'][c.chunk_id]:.4f}) "
              f"| #{a['rank'][c.chunk_id]} ({a['by_id'][c.chunk_id]:.4f}) |")

    n_page_top5 = {lbl: sum(1 for c in page_crops if runs[lbl]["rank"][c.chunk_id] <= 5)
                   for lbl in ("BEFORE", "AFTER")}
    print(f"\ngold-page crops in top-5: BEFORE {n_page_top5['BEFORE']}/{len(page_crops)}"
          f"  AFTER {n_page_top5['AFTER']}/{len(page_crops)}")
    print(f"top-1 on the correct page? BEFORE "
          f"{store.get(b['top1_id']).page_id == gold_page}  AFTER "
          f"{store.get(a['top1_id']).page_id == gold_page}")

    # ---- image path alone, over the gold questions that depend on a crop ----
    # One probe query cannot settle an encoder swap; this is the same comparison
    # over every gold question whose answer is IN a crop. Image path only — no
    # caption anchoring, no text index, no LLM.
    if gold_qs:
        ids = [c.chunk_id for c in crops]
        print(f"\n=== image path alone — {len(gold_qs)} image-dependent gold questions ===")
        print("| qid | category | gold crop rank BEFORE | AFTER |")
        print("|---|---|---|---|")
        agg = {"BEFORE": [], "AFTER": []}
        for i, q in enumerate(gold_qs):
            cells = {}
            for lbl in ("BEFORE", "AFTER"):
                order = np.argsort(-gold_runs[lbl][i])
                rank = {ids[j]: r for r, j in enumerate(order, 1)}
                best = min(rank[g] for g in q["golds"])
                agg[lbl].append(best)
                cells[lbl] = best
            print(f"| {q['qid']} | {q['category']} | {cells['BEFORE']} | {cells['AFTER']} |")
        print("\n| metric (image path only) | BEFORE | AFTER |")
        print("|---|---|---|")
        for name, fn in (
            ("recall@1", lambda rs: sum(r <= 1 for r in rs) / len(rs)),
            ("recall@5", lambda rs: sum(r <= 5 for r in rs) / len(rs)),
            ("MRR", lambda rs: sum(1 / r for r in rs) / len(rs)),
            ("median rank", lambda rs: float(np.median(rs))),
        ):
            print(f"| {name} | {fn(agg['BEFORE']):.2f} | {fn(agg['AFTER']):.2f} |")

    store.close()


if __name__ == "__main__":
    main()
