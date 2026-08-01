"""Entry point: the two Phase-9 re-tests, each a before/after on one change.

    python -m scripts.phase9_retests

PART A (retrievability, fixes Phase 8 finding #1) — for every figure_value gold
question, the gold table crop's rank in the text index and whether it enters the
candidate pool, with tables indexed by OCR vs by their vision-read transcription.

PART B (safety, fixes Phase 8 finding #3) — q061 answered with the crop-relevance
gate OFF and ON. Gate off reproduces the Phase-8 regression (a confident value read
off the wrong paper's table); gate on must refuse to read it.

Both parts vary exactly one thing and hold retrieval otherwise identical.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from evaluation.gold_set import load_gold
from platform_core.config import AppConfig
from platform_core.generation.generator import GroundedAnswerGenerator
from platform_core.llm.openai_client import OpenAIClient
from scripts.config_demo import build_retriever

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
# Phase-8 pool ranks, for a like-for-like before/after in the printed table.
PHASE8_TEXT_RANK = {"q060": 31, "q061": 19, "q062": 1, "q063": 1}


def text_rank(retriever, store, question: str, gold: str, k: int = 50):
    """Rank of the gold chunk in the raw text index (before flags add to the pool)."""
    hits = retriever.text_index.search(retriever.text_embedder.embed_query(question), k)
    ids = [u.split("#")[0] for u, _ in hits]
    return (ids.index(gold) + 1) if gold in ids else None


#: The three arms that attribute the fix. Each adds exactly one thing.
PART_A_ARMS = ("phase8 (OCR, no expansion)", "caption+transcription", "+ page expansion")


def part_a(cfgs: dict, gold) -> list[dict]:
    runs = {}
    for label, cfg in cfgs.items():
        retriever, store = build_retriever(cfg)
        runs[label] = {}
        for g in gold:
            gc = g.supporting_chunk_ids[0]
            pool = sorted(retriever.candidate_pool(g.question), key=lambda r: r.score, reverse=True)
            pool_ids = [r.chunk.chunk_id for r in pool]
            top = [r.chunk.chunk_id for r in retriever.retrieve(g.question)]
            runs[label][g.qid] = {
                "text_rank": text_rank(retriever, store, g.question, gc),
                "pool_rank": (pool_ids.index(gc) + 1) if gc in pool_ids else None,
                "topk_rank": (top.index(gc) + 1) if gc in top else None,
            }
        store.close()
    return [{"qid": g.qid, "gold": g.supporting_chunk_ids[0],
             **{label: runs[label][g.qid] for label in cfgs}} for g in gold]


def part_b(cfg: AppConfig, question, ) -> dict:
    """Same retrieval, same VQA model; only the gate threshold differs."""
    retriever, store = build_retriever(cfg)
    results = retriever.retrieve(question.question)
    crops = [r.chunk.chunk_id for r in results if r.chunk.image_path]
    gold = question.supporting_chunk_ids[0]

    gen_llm = OpenAIClient(model=cfg.generation.answer_model)
    vqa_llm = OpenAIClient(model=cfg.generation.vqa_model)
    out = {"qid": question.qid, "question": question.question,
           "expected": question.expected_answer, "gold_crop": gold,
           "gold_crop_retrieved": gold in [r.chunk.chunk_id for r in results],
           "crops_in_context": crops, "runs": {}}
    # gate OFF = threshold 0.0 (nothing can fail) -> reproduces phase 8
    for label, thr in (("gate_off", 0.0), ("gate_on", cfg.generation.vqa_min_crop_score)):
        params = dataclasses.replace(cfg.generation, vqa_min_crop_score=thr)
        gen = GroundedAnswerGenerator(gen_llm, cfg.flags, params, retriever.text_embedder,
                                      store, vqa_llm=vqa_llm)
        res = gen.generate(question.question, results)
        out["runs"][label] = {"threshold": thr, "answer": res.answer,
                              "abstained": res.abstained, "vqa": gen.last_vqa}
    store.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=R + "/configs/enhanced_vqa.yaml")
    ap.add_argument("--qid", default="q061", help="question for the part-B safety re-test")
    args = ap.parse_args()

    full = AppConfig.from_yaml(args.config)          # both part-A changes on
    F = full.flags
    cfgs = {
        PART_A_ARMS[0]: dataclasses.replace(full, flags=dataclasses.replace(
            F, use_table_vqa_text=False, use_page_crop_expansion=False)),
        PART_A_ARMS[1]: dataclasses.replace(full, flags=dataclasses.replace(
            F, use_table_vqa_text=True, use_page_crop_expansion=False)),
        PART_A_ARMS[2]: full,
    }
    print(full.describe())
    print(f"vqa_min_crop_score: {full.generation.vqa_min_crop_score}\n")

    gold_all = load_gold(full.paths.gold_set)
    fv = [g for g in gold_all if g.category == "figure_value"]

    print("=" * 78)
    print("PART A — do the gold TABLE crops become retrievable?")
    print("  each arm adds exactly one thing to the one before it\n")
    rows = part_a(cfgs, fv)
    f = lambda v: str(v) if v else "—"               # noqa: E731
    for metric, title in (("text_rank", "rank in the TEXT index"),
                          ("pool_rank", "rank in the CANDIDATE POOL"),
                          ("topk_rank", "rank in the FINAL top-k (what the answerer sees)")):
        print(f"  {title}")
        print("  | qid | " + " | ".join(PART_A_ARMS) + " |")
        print("  |" + "---|" * (len(PART_A_ARMS) + 1))
        for r in rows:
            print(f"  | {r['qid']} | " + " | ".join(f(r[a][metric]) for a in PART_A_ARMS) + " |")
        hit = {a: sum(1 for r in rows if r[a][metric]) for a in PART_A_ARMS}
        print(f"  found: " + "   ".join(f"{a} {hit[a]}/{len(rows)}" for a in PART_A_ARMS) + "\n")

    print("\n" + "=" * 78)
    q = next(g for g in gold_all if g.qid == args.qid)
    print(f"PART B — crop-relevance gate, {q.qid}")
    # Run under BOTH index variants. The OCR variant is the exact phase-8 situation
    # (gold crop missing), which is what the gate exists for; the transcription
    # variant shows what the gate does once part A has fixed retrieval.
    bs = {}
    for label, cfg in (("phase-8 retrieval (gold crop missing)", cfgs[PART_A_ARMS[0]]),
                       ("phase-9 retrieval (after part A)", full)):
        b = bs[label] = part_b(cfg, q)
        print(f"\n--- {label} ---")
        print(f"  expected : {b['expected']}")
        print(f"  gold crop: {b['gold_crop']}  "
              f"{'RETRIEVED' if b['gold_crop_retrieved'] else '*** NOT RETRIEVED ***'}")
        print(f"  crops in context: {b['crops_in_context']}")
        for run_label in ("gate_off", "gate_on"):
            r = b["runs"][run_label]
            print(f"    {run_label} (threshold {r['threshold']}):")
            for gsc in (r["vqa"].get("gate") or []):
                sc = "  n/a " if gsc["score"] is None else f"{gsc['score']:.4f}"
                pg = {True: "same page as retrieved prose", False: "OTHER PAPER",
                      None: "not checked"}[gsc["on_supported_page"]]
                print(f"      crop {gsc['chunk_id']:22s} score {sc}  {pg:28s} "
                      f"{'PASS -> read' if gsc['passed'] else 'BLOCKED'}")
            print(f"      vqa used={r['vqa'].get('used')}  read: {r['vqa'].get('crops') or '(none)'}")
            print(f"      answer : {'[ABSTAINED] ' if r['abstained'] else ''}"
                  f"{' '.join(r['answer'].split())[:170]}")

    out = os.path.join(full.paths.reports_dir, "phase9_retests.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"part_a": rows, "part_b": bs}, f, indent=2, ensure_ascii=False)
    print(f"\n(n={len(fv)} — capability + safety demonstration, NOT a statistical claim)")
    print(f"log: {out}")


if __name__ == "__main__":
    main()
