"""Entry point: end-to-end VQA check on the figure_value gold questions.

    python -m scripts.vqa_figure_values --config configs/enhanced_vqa.yaml

The point is the REALISTIC path: retrieve normally (text + caption-anchor + BiomedCLIP
+ KG + rerank, per the config), then let the generator answer from whatever crops
retrieval actually returned. Crops are never hand-fed — if retrieval misses the gold
crop, that shows up here as a retrieval failure, which is part of the honest picture.

Runs each question twice under the same retrieval: once with use_vqa OFF (text/OCR
only) and once ON (crop image attached), so the delta is attributable to VQA alone.
Reports expected value vs the value actually read, per question.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
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


def norm(s: str) -> str:
    """Loose numeric comparison: strip everything but digits, dots and signs."""
    return re.sub(r"[^0-9.\-]", "", (s or "").replace("–", "-").replace("—", "-"))


def _contains_in_order(nums, answer: str) -> bool:
    hay = norm(answer)
    pos = 0
    for n in nums:
        i = hay.find(norm(n), pos)
        if i < 0:
            return False
        pos = i + len(norm(n))
    return True


def value_present(expected: str, answer: str) -> bool:
    """STRICT: every number in the expected value, in order (point estimate + CI)."""
    return _contains_in_order(re.findall(r"\d+\.?\d*", expected), answer)


def primary_value_present(expected: str, answer: str) -> bool:
    """LENIENT: the point estimate only.

    Reported alongside the strict score because several gold `expected_answer`
    strings carry a 95% CI that the question does not actually ask for — an answer
    giving the right point estimate and no CI is right about what was asked, and
    conflating that with a misread would overstate VQA's error rate.
    """
    nums = re.findall(r"\d+\.?\d*", expected)
    return bool(nums) and _contains_in_order(nums[:1], answer)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=R + "/configs/enhanced_vqa.yaml")
    ap.add_argument("--category", default="figure_value")
    args = ap.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    print(cfg.describe())
    print(f"vqa_max_images  : {cfg.generation.vqa_max_images}\n")

    gold = [g for g in load_gold(cfg.paths.gold_set) if g.category == args.category]
    retriever, store = build_retriever(cfg)

    gen_llm = OpenAIClient(model=cfg.generation.answer_model)
    vqa_llm = OpenAIClient(model=cfg.generation.vqa_model)
    off = dataclasses.replace(cfg.flags, use_vqa=False)
    gen_text = GroundedAnswerGenerator(gen_llm, off, cfg.generation, retriever.text_embedder, store)
    gen_vqa = GroundedAnswerGenerator(gen_llm, cfg.flags, cfg.generation,
                                      retriever.text_embedder, store, vqa_llm=vqa_llm)

    rows = []
    for g in gold:
        results = retriever.retrieve(g.question)          # ONE retrieval, shared
        got = [r.chunk.chunk_id for r in results]
        crops = [r.chunk.chunk_id for r in results if r.chunk.image_path]
        gold_crop = g.supporting_chunk_ids[0]
        rank = got.index(gold_crop) + 1 if gold_crop in got else None

        pool = sorted(retriever.candidate_pool(g.question), key=lambda r: r.score, reverse=True)
        pool_ids = [r.chunk.chunk_id for r in pool]
        pool_rank = pool_ids.index(gold_crop) + 1 if gold_crop in pool_ids else None

        a_text = gen_text.generate(g.question, results)
        a_vqa = gen_vqa.generate(g.question, results)
        rows.append({
            "qid": g.qid, "question": g.question, "expected": g.expected_answer,
            "gold_crop": gold_crop, "gold_crop_rank": rank,
            "gold_crop_retrieved": rank is not None,
            "crops_in_context": crops,
            "crops_sent_to_vqa": gen_vqa.last_vqa.get("crops", []),
            "vqa_used": gen_vqa.last_vqa.get("used", False),
            "text_answer": a_text.answer, "text_abstained": a_text.abstained,
            "vqa_answer": a_vqa.answer, "vqa_abstained": a_vqa.abstained,
            "text_ok": (not a_text.abstained) and value_present(g.expected_answer, a_text.answer),
            "vqa_ok": (not a_vqa.abstained) and value_present(g.expected_answer, a_vqa.answer),
            "text_ok_primary": (not a_text.abstained)
            and primary_value_present(g.expected_answer, a_text.answer),
            "vqa_ok_primary": (not a_vqa.abstained)
            and primary_value_present(g.expected_answer, a_vqa.answer),
            # where the gold crop sat in the FULL pool — separates "retrieval missed
            # it entirely" from "it was found but truncated below top_k"
            "gold_crop_pool_rank": pool_rank,
        })

    out = os.path.join(cfg.paths.reports_dir, "vqa_figure_values.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=" * 78)
    for r in rows:
        print(f"\n{r['qid']}  {r['question']}")
        print(f"  expected value      : {r['expected']}")
        print(f"  gold crop           : {r['gold_crop']}  "
              f"{'RETRIEVED at rank ' + str(r['gold_crop_rank']) if r['gold_crop_retrieved'] else '*** NOT RETRIEVED ***'}")
        print(f"  crops sent to VQA   : {r['crops_sent_to_vqa'] or '(none)'}  (vqa_used={r['vqa_used']})")
        print(f"  TEXT-only answer    : {'[ABSTAINED] ' if r['text_abstained'] else ''}"
              f"{' '.join(r['text_answer'].split())[:200]}")
        print(f"    -> {'CORRECT' if r['text_ok'] else 'INCORRECT'}")
        print(f"  VQA answer          : {'[ABSTAINED] ' if r['vqa_abstained'] else ''}"
              f"{' '.join(r['vqa_answer'].split())[:200]}")
        print(f"    -> {'CORRECT' if r['vqa_ok'] else 'INCORRECT'}")

    n = len(rows)
    ret = [r for r in rows if r["gold_crop_retrieved"]]
    print("\n" + "=" * 78)
    print("| qid | expected | VQA read | strict | point est. |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['qid']} | {r['expected']} | {' '.join(r['vqa_answer'].split())[:52]} "
              f"| {'YES' if r['vqa_ok'] else 'NO'} | {'YES' if r['vqa_ok_primary'] else 'NO'} |")
    print(f"\ngold crop retrieved        : {len(ret)}/{n}   "
          f"(pool ranks: {[r['gold_crop_pool_rank'] for r in rows]})")
    print(f"TEXT-only correct (strict) : {sum(r['text_ok'] for r in rows)}/{n}"
          f"   point est.: {sum(r['text_ok_primary'] for r in rows)}/{n}")
    print(f"VQA correct       (strict) : {sum(r['vqa_ok'] for r in rows)}/{n}"
          f"   point est.: {sum(r['vqa_ok_primary'] for r in rows)}/{n}")
    if ret:
        print(f"VQA, gold crop retrieved   : strict {sum(r['vqa_ok'] for r in ret)}/{len(ret)}"
              f"   point est. {sum(r['vqa_ok_primary'] for r in ret)}/{len(ret)}")
    print(f"(n={n} — capability demonstration, NOT a statistical claim)")

    IN, OUT = 2.5 / 1e6, 10 / 1e6
    u = {k: gen_llm.total_usage[k] + vqa_llm.total_usage[k] for k in gen_llm.total_usage}
    print(f"\nACTUAL API cost: ${u['prompt_tokens'] * IN + u['completion_tokens'] * OUT:.3f}")
    print(f"per-question log: {out}")
    store.close()


if __name__ == "__main__":
    main()
