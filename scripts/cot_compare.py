"""Paired chain-of-thought comparison on the reasoning-heavy subset.

    python -m scripts.cot_compare --config configs/scale500.yaml --resume

Deliberately NOT a seventh ablation config. The six-config results in
reports/scale500/ are frozen; re-running them to add CoT would cost another ~$5 and
put a locked result at risk. CoT is a generation-side change — it cannot affect
retrieval — so it is measured where it could plausibly matter and nowhere else.

Design, and why each part is what it is:

* **Flags come from the ablation's own enhanced step**, `ablation_flags("classic")[-1]`,
  not from the config's `flags` block. That guarantees the CoT-off arm is bit-identical
  to the `+rerank` column of the frozen table, so the comparison sits against a number
  already published rather than against a fresh baseline.
* **Retrieval runs ONCE per question** and both arms answer from the same result, so the
  delta is attributable to the prompt and nothing else.
* **Both arms are judged by the same judge**, on the ANSWER section only — the generator
  strips the reasoning before anything scores it.
* **multi_hop is the hypothesis, single_fact is the control.** If CoT helps by making the
  model combine passages, multi_hop should move and single_fact should not. A control
  that also moves would mean something other than reasoning changed.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from evaluation.ablation import ablation_flags
from evaluation.answer_metrics import judge_correctness
from evaluation.gold_set import load_gold
from platform_core.config import AppConfig
from platform_core.generation.generator import GroundedAnswerGenerator
from platform_core.llm.openai_client import OpenAIClient
from scripts.ablate import build_harness

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
DEFAULT_CONFIG = R + "/configs/scale500.yaml"
IN, OUT = 2.5 / 1e6, 10 / 1e6  # gpt-4o


def subset(gold, n_single: int):
    """All multi_hop (the hypothesis) + a strided single_fact sample (the control).

    Strided rather than head-sliced so the control spans the corpus instead of
    clustering in whichever predicate the builder happened to emit first.
    """
    multi = [g for g in gold if g.category == "multi_hop"]
    singles = [g for g in gold if g.category == "single_fact"]
    stride = max(1, len(singles) // n_single)
    return multi + singles[::stride][:n_single]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--single", type=int, default=20, help="single_fact controls")
    ap.add_argument("--resume", action="store_true",
                    help="reuse a complete cot_compare.jsonl instead of re-buying it")
    args = ap.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    print(cfg.describe())

    gold = load_gold(cfg.paths.gold_set)
    items = subset(gold, args.single)
    out_path = os.path.join(cfg.paths.reports_dir, "cot_compare.jsonl")

    rows: list[dict] = []
    if args.resume and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if len(rows) == len(items):
            print(f"\n[resume] reusing {len(rows)} logged questions")
        else:
            print(f"[resume] log has {len(rows)}/{len(items)} — re-running")
            rows = []

    cost = 0.0
    if not rows:
        # the frozen enhanced config, taken from the same source the ablation used
        enhanced = dict(ablation_flags("classic"))["+rerank"]
        off = dataclasses.replace(enhanced, use_cot=False)
        on = dataclasses.replace(enhanced, use_cot=True)
        print("\nCoT OFF flags == the frozen '+rerank' column; ON differs only in use_cot")

        harness = build_harness(cfg)
        retriever = harness._retriever(enhanced)
        gen_off = GroundedAnswerGenerator(harness.gen_llm, off, cfg.generation,
                                          harness.bge, harness.store)
        gen_on = GroundedAnswerGenerator(harness.gen_llm, on, cfg.generation,
                                         harness.bge, harness.store)

        print(f"\n{len(items)} questions "
              f"({sum(1 for g in items if g.category == 'multi_hop')} multi_hop, "
              f"{sum(1 for g in items if g.category == 'single_fact')} single_fact)\n")
        for i, g in enumerate(items, 1):
            results = retriever.retrieve(g.question)      # ONE retrieval, shared
            a_off = gen_off.generate(g.question, results)
            a_on = gen_on.generate(g.question, results)
            reasoning = gen_on.last_reasoning
            rows.append({
                "qid": g.qid, "category": g.category, "question": g.question,
                "expected": g.expected_answer,
                "off_answer": a_off.answer, "off_abstained": a_off.abstained,
                "off_correct": 0.0 if a_off.abstained else judge_correctness(
                    harness.judge_llm, g.question, g.expected_answer, a_off.answer),
                "on_answer": a_on.answer, "on_abstained": a_on.abstained,
                "on_correct": 0.0 if a_on.abstained else judge_correctness(
                    harness.judge_llm, g.question, g.expected_answer, a_on.answer),
                "reasoning": reasoning,
                "reasoning_chars": len(reasoning),
                "off_citations": a_off.cited_chunk_ids, "on_citations": a_on.cited_chunk_ids,
            })
            if i % 5 == 0 or i == len(items):
                print(f"  [{i}/{len(items)}]")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        u = {k: harness.gen_llm.total_usage[k] + harness.judge_llm.total_usage[k]
             for k in harness.gen_llm.total_usage}
        cost = u["prompt_tokens"] * IN + u["completion_tokens"] * OUT

    # ---------- report ----------
    def agg(cat=None):
        sel = [r for r in rows if cat is None or r["category"] == cat]
        if not sel:
            return None
        n = len(sel)
        return {
            "n": n,
            "off": sum(r["off_correct"] for r in sel) / n,
            "on": sum(r["on_correct"] for r in sel) / n,
            "off_abst": sum(r["off_abstained"] for r in sel) / n,
            "on_abst": sum(r["on_abstained"] for r in sel) / n,
            "flip_up": sum(1 for r in sel if r["on_correct"] > r["off_correct"]),
            "flip_dn": sum(1 for r in sel if r["on_correct"] < r["off_correct"]),
        }

    md = ["# Chain-of-thought — paired comparison", "",
          f"Enhanced retrieval (`+rerank`, the frozen configuration), one retrieval per "
          f"question answered twice: CoT off vs on. {len(rows)} questions.", "",
          "| subset | n | Correct OFF | Correct ON | delta | improved | regressed |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for label, cat in (("multi_hop (hypothesis)", "multi_hop"),
                       ("single_fact (control)", "single_fact"),
                       ("**combined**", None)):
        a = agg(cat)
        if a:
            md.append(f"| {label} | {a['n']} | {a['off']:.2f} | {a['on']:.2f} | "
                      f"{a['on'] - a['off']:+.2f} | {a['flip_up']} | {a['flip_dn']} |")
    md += ["", "Abstention (a CoT prompt can make the model more cautious, which would "
           "show up as correctness loss without any reasoning failure):", "",
           "| subset | abstained OFF | abstained ON |", "|---|---:|---:|"]
    for label, cat in (("multi_hop", "multi_hop"), ("single_fact", "single_fact")):
        a = agg(cat)
        if a:
            md.append(f"| {label} | {a['off_abst']:.2f} | {a['on_abst']:.2f} |")

    flips = [r for r in rows if r["on_correct"] != r["off_correct"]]
    md += ["", f"## Questions whose verdict changed ({len(flips)})", ""]
    for r in flips:
        d = "improved" if r["on_correct"] > r["off_correct"] else "regressed"
        md += [f"**{r['qid']}** ({r['category']}, {d}) — {r['question']}", "",
               f"- expected: `{r['expected'][:160]}`",
               f"- OFF: {'*abstained*' if r['off_abstained'] else ' '.join(r['off_answer'].split())[:220]}",
               f"- ON: {'*abstained*' if r['on_abstained'] else ' '.join(r['on_answer'].split())[:220]}", ""]

    ex = next((r for r in flips if r["category"] == "multi_hop" and r["reasoning"]), None) \
        or next((r for r in rows if r["reasoning"]), None)
    if ex:
        md += ["## Example reasoning trace", "",
               f"**{ex['qid']}** ({ex['category']}) — {ex['question']}", "",
               "```", ex["reasoning"][:1600], "```", "",
               f"Answer: {' '.join(ex['on_answer'].split())[:300]}", ""]
    lens = [r["reasoning_chars"] for r in rows if r["reasoning_chars"]]
    if lens:
        md += [f"Reasoning traces captured for {len(lens)}/{len(rows)} answers, "
               f"median {sorted(lens)[len(lens)//2]} characters.", ""]

    doc = "\n".join(md)
    p = os.path.join(cfg.paths.reports_dir, "cot_compare.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write(doc)
    print("\n" + doc)
    if cost:
        print(f"\nACTUAL API cost: ${cost:.2f}")
    print(f"wrote {p}  and  {out_path}")


if __name__ == "__main__":
    main()
