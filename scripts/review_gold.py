"""Render a stratified sample of the gold set for human spot-checking.

    python -m scripts.review_gold --config configs/scale500.yaml --per-category 4

The gold builder's filters are mechanical; the defects that actually matter are the
ones only a reader catches — a reversed direction ("height increases age"), an
over-generalised qualifier (the q061 class), an answer that is true of the evidence
but not of the question. So every item is printed WITH the source it came from:
the evidence sentence, the source region's own text, and for crop-backed items the
path to the image, since a table value cannot be verified from text at all.

Writes <reports_dir>/gold_review.md and prints the same to stdout. Read-only — it
never edits the gold set.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from platform_core.config import AppConfig
from platform_core.stores.chunk_store import SQLiteChunkStore

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
DEFAULT_CONFIG = R + "/configs/scale500.yaml"
ORDER = ["single_fact", "multi_hop", "figure", "figure_value", "text_derived", "unanswerable"]


def flat(s: str, n: int) -> str:
    one = " ".join((s or "").split())
    return one[:n] + ("…" if len(one) > n else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--per-category", type=int, default=4)
    ap.add_argument("--all-figure-values", action="store_true", default=True,
                    help="show every figure_value item (their values need eyeballing)")
    args = ap.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    store = SQLiteChunkStore(cfg.paths.chunk_db)
    gold = [json.loads(l) for l in open(cfg.paths.gold_set, encoding="utf-8")]
    edges_by_chunk: dict = collections.defaultdict(list)
    ep = cfg.paths.graph_store + ".edges.jsonl"
    if os.path.exists(ep):
        for line in open(ep, encoding="utf-8"):
            e = json.loads(line)
            edges_by_chunk[e["chunk_id"]].append(e)

    by_cat: dict = collections.defaultdict(list)
    for r in gold:
        by_cat[r["category"]].append(r)

    out: list[str] = [
        f"# Gold-set spot-check — {cfg.name}",
        "",
        f"`{cfg.paths.gold_set}` — {len(gold)} questions. Automated filters only; "
        "nothing here is human-verified yet.",
        "",
        "For each item: the question, the answer the builder derived from the graph, and "
        "the SOURCE it was derived from. Check that the question is answerable from that "
        "source, that the direction is not reversed, and that no qualifier in the question "
        "is absent from the source (the q061 defect class).",
        "",
        "| category | n |", "|---|---|",
    ]
    for c in ORDER:
        out.append(f"| {c} | {len(by_cat.get(c, []))} |")

    for cat in ORDER:
        items = by_cat.get(cat, [])
        if not items:
            continue
        show = items if (cat == "figure_value" and args.all_figure_values) else items[:args.per_category]
        stride = max(1, len(items) // len(show)) if show else 1
        if show is not items:
            show = items[::stride][:args.per_category]
        out += ["", f"## {cat}  ({len(show)} of {len(items)} shown)"]
        for r in show:
            out += ["", f"**{r['qid']}** — {r['question']}", "",
                    f"- **expected answer:** `{r['expected_answer']}`"]
            if r.get("source_predicates"):
                out.append(f"- **from predicate(s):** {', '.join(r['source_predicates'])}")
            if r.get("needs_value_verification"):
                out.append("- ⚠ **value NOT verified** — derived from table OCR, which phase 9 "
                           "showed misplaces digits. Verify against the crop before use.")
            if r.get("crop_path"):
                out.append(f"- **crop:** `{r['crop_path']}`")
            if r.get("absence_check_top_cosine") is not None:
                out.append(f"- **absence check:** top cosine vs corpus = "
                           f"{r['absence_check_top_cosine']} (lower = more clearly absent)")
            for cid in r["supporting_chunk_ids"]:
                try:
                    c = store.get(cid)
                except KeyError:
                    out.append(f"- **{cid}** — ⚠ MISSING from the chunk store")
                    continue
                out.append(f"- **{cid}** [{c.region_type.value}]"
                           + (f" crop=`{c.image_path}`" if c.image_path else ""))
                ev = [e["evidence"] for e in edges_by_chunk.get(cid, [])
                      if e["object"].strip() == str(r["expected_answer"]).strip()]
                if ev:
                    out.append(f"    - evidence: *{flat(ev[0], 300)}*")
                body = c.text or ""
                out.append(f"    - region text: {flat(body, 320) or '(no OCR text)'}")
                if c.vqa_text:
                    out.append(f"    - transcription (NOT authoritative for digits): "
                               f"{flat(c.vqa_text, 220)}")

    md = "\n".join(out)
    os.makedirs(cfg.paths.reports_dir, exist_ok=True)
    path = os.path.join(cfg.paths.reports_dir, "gold_review.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)
    print(f"\n\nwrote {path}")
    store.close()


if __name__ == "__main__":
    main()
