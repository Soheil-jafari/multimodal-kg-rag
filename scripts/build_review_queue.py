"""Precompute the human review queue from a LOCKED evaluation run.

    python -m scripts.build_review_queue --config configs/scale500.yaml --step +rerank

Read-only with respect to every frozen artifact. It reads `perq_<step>.jsonl`, the gold
set and the chunk store, and writes ONE new sidecar — `review_queue.jsonl`. No metric in
`ablation.md` or any `perq_*.jsonl` is recomputed or rewritten.

## The confidence score

Built only from signals the system has **at inference time**. This matters: the obvious
move is to reuse the retrieval metrics already in the per-question log, but those
(recall@k, MRR, nDCG) are computed against gold supporting_chunk_ids. A "confidence"
derived from them would be a correctness proxy in disguise — it would rank items by how
right they are, which a deployed reviewer queue cannot know and which would make the
dashboard look far better than it is.

Two label-free signals, both already produced by the pipeline:

* **retrieval strength** `R` — the top query-context cosine, recomputed here exactly as
  `GroundedAnswerGenerator.relevance()` computes it. This is the same number the
  abstention code gate thresholds against `generation.abstain_min_score`.
* **grounding** `G` — the per-sentence grounded fraction from the locked run
  (`faithfulness`): the share of answer sentences whose cosine to their cited chunk cleared
  `grounding_min_sim`, with the LLM verifier deciding the borderline band.

`R` is normalised against the config's own abstention floor, since below it the system
declines to answer at all:

    Rn = clip((R - abstain_min_score) / (0.85 - abstain_min_score), 0, 1)

**Answered items:** `confidence = 0.5*Rn + 0.5*G`, and `priority = 1 - confidence`, so the
least-supported answers surface first.

**Abstained items:** there is no answer to grade, so confidence is undefined and
`priority = Rn`. That inversion is deliberate — an abstention on *weak* retrieval is
probably correct and needs no review, while an abstention on *strong* retrieval is the
false-abstention failure mode Phase 6D was built to catch, and is the most valuable thing a
reviewer can look at.

The weighting is a flat 0.5/0.5 by choice, not by fitting: tuning it against the locked
correctness column would fit the score to this run's answers and quietly turn it back into
a correctness proxy.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from evaluation.ablation import ablation_flags
from evaluation.gold_set import load_gold
from platform_core.config import AppConfig
from platform_core.generation.generator import passage_text
from scripts.ablate import build_harness

R_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
DEFAULT_CONFIG = R_ROOT + "/configs/scale500.yaml"
_CITE = re.compile(r"\[([^\]]+)\]")
R_CEIL = 0.85  # BGE tops out near here on a strong match; the floor comes from the config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--step", default="+rerank", help="which ablation column to review")
    ap.add_argument("--cot", default=None,
                    help="cot_compare.jsonl for reasoning traces (default: alongside reports)")
    args = ap.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    print(cfg.describe())
    steps = dict(ablation_flags("classic"))
    if args.step not in steps:
        raise SystemExit(f"unknown step {args.step!r}; known: {sorted(steps)}")
    flags = steps[args.step]

    perq_path = os.path.join(cfg.paths.reports_dir, f"perq_{args.step}.jsonl")
    if not os.path.exists(perq_path):
        raise SystemExit(f"no locked run at {perq_path} — run the ablation first")
    perq = {r["qid"]: r for r in (json.loads(l) for l in open(perq_path, encoding="utf-8"))}
    gold = {g.qid: g for g in load_gold(cfg.paths.gold_set)}

    cot_path = args.cot or os.path.join(cfg.paths.reports_dir, "cot_compare.jsonl")
    traces: dict = {}
    if os.path.exists(cot_path):
        for line in open(cot_path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("reasoning"):
                traces[r["qid"]] = {"reasoning": r["reasoning"],
                                    "cot_answer": r.get("on_answer", "")}

    harness = build_harness(cfg)
    retriever = harness._retriever(flags)
    store = harness.store
    floor = cfg.generation.abstain_min_score

    def text_of(cid: str) -> str:
        try:
            c = store.get(cid)
        except KeyError:
            return ""
        return passage_text(c, store) or f"({c.region_type.value}, no OCR text)"

    rows: list[dict] = []
    print(f"\nscoring {len(perq)} questions from perq_{args.step}.jsonl "
          f"(retrieval re-run locally; no API calls, no frozen file touched)")
    for i, (qid, rec) in enumerate(sorted(perq.items()), 1):
        g = gold.get(qid)
        if g is None:
            continue
        results = retriever.retrieve(g.question)
        # exactly GroundedAnswerGenerator.relevance(): top query-context cosine
        texts = [t for t in (passage_text(r.chunk, store) for r in results) if t.strip()]
        if texts:
            qv = np.asarray(harness.bge.embed_query(g.question), dtype="float32")
            strength = float(np.max(np.asarray(harness.bge.embed(texts),
                                               dtype="float32") @ qv))
        else:
            strength = 0.0
        rn = min(1.0, max(0.0, (strength - floor) / (R_CEIL - floor)))

        abstained = bool(rec["abstained"])
        grounding = rec.get("faithfulness")
        if abstained:
            confidence = None
            priority = rn
            reason = ("abstained despite strong retrieval" if rn >= 0.5
                      else "abstained on weak retrieval")
        else:
            grounding = 0.0 if grounding is None else float(grounding)
            confidence = 0.5 * rn + 0.5 * grounding
            priority = 1.0 - confidence
            weak = []
            if rn < 0.5:
                weak.append("weak retrieval")
            if grounding < 0.8:
                weak.append("ungrounded sentences")
            reason = " + ".join(weak) if weak else "well supported"

        cited = []
        for cid in dict.fromkeys(_CITE.findall(rec.get("answer") or "")):
            cited.append({"chunk_id": cid, "text": text_of(cid),
                          "in_store": bool(text_of(cid))})

        rows.append({
            "qid": qid, "category": rec["category"], "question": g.question,
            "answer": rec.get("answer", ""), "abstained": abstained,
            "answerable": rec["answerable"],
            "retrieval_strength": round(strength, 4), "retrieval_norm": round(rn, 4),
            "grounding": None if grounding is None else round(float(grounding), 4),
            "confidence": None if confidence is None else round(confidence, 4),
            "priority": round(priority, 4), "flag_reason": reason,
            "provenance": rec.get("provenance", {}),
            "graph_predicates": rec.get("graph_predicates", []),
            "vqa": rec.get("vqa", {}),
            "citations": cited,
            "retrieved_top": [{"chunk_id": r.chunk.chunk_id, "source": r.source,
                               "region": r.chunk.region_type.value}
                              for r in results[:8]],
            "reasoning": traces.get(qid, {}).get("reasoning", ""),
            # revealed behind a toggle in the dashboard so the reviewer judges from the
            # evidence first, then checks themselves against the gold and the judge
            "expected_answer": g.expected_answer,
            "gold_chunk_ids": g.supporting_chunk_ids,
            "judge_correctness": rec.get("correctness"),
            "source_step": args.step, "source_config": cfg.name,
        })
        if i % 25 == 0 or i == len(perq):
            print(f"  [{i}/{len(perq)}]")

    rows.sort(key=lambda r: -r["priority"])
    out = os.path.join(cfg.paths.reports_dir, "review_queue.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ab = [r for r in rows if r["abstained"]]
    ans = [r for r in rows if not r["abstained"]]
    low = [r for r in ans if r["confidence"] is not None and r["confidence"] < 0.5]
    print(f"\nwrote {len(rows)} items -> {out}")
    print(f"  answered {len(ans)}  |  abstained {len(ab)}")
    print(f"  answered with confidence < 0.50: {len(low)}")
    print(f"  abstained despite strong retrieval (Rn >= 0.5): "
          f"{sum(1 for r in ab if r['retrieval_norm'] >= 0.5)}")
    print(f"  reasoning traces attached: {sum(1 for r in rows if r['reasoning'])}")
    print("\ntop of the queue:")
    for r in rows[:8]:
        c = "abstained" if r["abstained"] else f"conf {r['confidence']:.2f}"
        print(f"  {r['priority']:.2f}  {r['qid']:6s} {r['category']:13s} {c:12s} {r['flag_reason']}")
    store.close()


if __name__ == "__main__":
    main()
