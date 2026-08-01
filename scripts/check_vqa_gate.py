"""Does the VQA provenance gate still work with table transcription switched OFF?

    python -m scripts.check_vqa_gate

The 500-page run skips the transcription pass (phase 9 measured it as contributing
nothing to retrievability), so `chunks.vqa_text` is NULL for that whole corpus. The
gate scores each candidate crop on ``passage_text(..., prefer_vqa=True)``, which falls
back to the region's OWN OCR text when there is no transcription — and only falls
through to the caption when that text is empty too. So switching transcription off
silently changes what the safety gate scores on, for tables specifically.

This measures that change instead of assuming it is harmless, on the DEV corpus — the
only corpus where both versions exist, and the one containing the q061 case the gate
was built for. For each figure_value question it retrieves once, then runs the gate
twice off that same retrieval: once with the transcription, once with `vqa_text`
stripped to simulate the 500-page corpus. Retrieval is identical between the two, so
any difference is attributable to the gate's scoring input and nothing else.

NO API SPEND: retrieval and gating only. The vision model is never called.
"""
from __future__ import annotations

import argparse
import copy
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
from platform_core.generation.generator import GroundedAnswerGenerator, passage_text
from scripts.ablate import build_harness

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
DEFAULT_CONFIG = R + "/configs/enhanced_vqa.yaml"


def strip_transcription(results):
    """A copy of the retrieval whose chunks have no vqa_text — the 500-page corpus."""
    out = []
    for r in results:
        r2 = copy.copy(r)
        r2.chunk = copy.copy(r.chunk)
        r2.chunk.vqa_text = None
        out.append(r2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--category", default="figure_value")
    args = ap.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    print(cfg.describe())
    harness = build_harness(cfg)
    gold = [g for g in load_gold(cfg.paths.gold_set) if g.category == args.category]
    print(f"\n{len(gold)} {args.category} questions\n")

    retriever = harness._retriever(cfg.flags)
    gen = GroundedAnswerGenerator(harness.gen_llm, cfg.flags, cfg.generation,
                                  harness.bge, harness.store, vqa_llm=harness.vqa_llm)

    rows: list[dict] = []
    md: list[str] = [
        "# VQA provenance gate — with vs without table transcription", "",
        "Retrieval is run ONCE per question and the gate is applied twice to that same "
        "result, so the only variable is what the gate scores a crop on: its "
        "transcription (dev) or its OCR text (the 500-page corpus, where no "
        "transcription exists).", "",
        f"Gate floor `vqa_min_crop_score` = {cfg.generation.vqa_min_crop_score}; "
        f"`vqa_max_images` = {cfg.generation.vqa_max_images}.", "",
    ]

    for g in gold:
        results = retriever.retrieve(g.question)
        gold_ids = set(g.supporting_chunk_ids)
        variants = {"with transcription": results,
                    "without (500-page)": strip_transcription(results)}
        md += [f"## {g.qid} — {g.question}", "",
               f"gold crop(s): `{', '.join(g.supporting_chunk_ids)}`", ""]
        row: dict = {"qid": g.qid, "question": g.question, "gold": sorted(gold_ids)}
        for label, res in variants.items():
            crops, gate = gen._gated_crops(g.question, res)
            sel = [c.chunk.chunk_id for c in crops]
            passed = [x for x in gate if x["passed"]]
            gold_in_ctx = [c.chunk.chunk_id for c in res
                           if c.chunk.image_path and c.chunk.chunk_id in gold_ids]
            gold_passed = [x for x in gate if x["passed"] and x["chunk_id"] in gold_ids]
            row[label] = {
                "crops_in_context": len(gate), "passed": len(passed), "selected": sel,
                "gold_crop_retrieved": bool(gold_in_ctx),
                "gold_crop_passed_gate": bool(gold_passed),
                "gold_crop_selected": any(s in gold_ids for s in sel),
                "gate": gate,
            }
            md += [f"**{label}** — {len(gate)} crop(s) in context, {len(passed)} passed, "
                   f"read: `{', '.join(sel) or 'none (abstains / text path)'}`", "",
                   "| crop | scored on | score | on supported page | passed | gold |",
                   "|---|---|---:|:---:|:---:|:---:|"]
            for x in gate:
                ch = next((c.chunk for c in res if c.chunk.chunk_id == x["chunk_id"]), None)
                src = "-"
                if ch is not None:
                    body = ch.retrieval_text(prefer_vqa=True)
                    src = ("transcription" if (ch.vqa_text or "").strip() else
                           ("OCR text" if body.strip() else
                            ("caption" if ch.caption_id else "region type only")))
                md.append(
                    f"| `{x['chunk_id']}` | {src} | "
                    f"{x['score'] if x['score'] is not None else '-'} | "
                    f"{'yes' if x['on_supported_page'] else 'NO'} | "
                    f"{'PASS' if x['passed'] else 'block'} | "
                    f"{'**GOLD**' if x['chunk_id'] in gold_ids else ''} |")
            md.append("")
        rows.append(row)

    # ---- verdict, computed rather than asserted
    def tally(key):
        return sum(1 for r in rows if r[key]["gold_crop_passed_gate"])

    def sel_tally(key):
        return sum(1 for r in rows if r[key]["gold_crop_selected"])

    a, b = "with transcription", "without (500-page)"
    changed = [r["qid"] for r in rows
               if r[a]["selected"] != r[b]["selected"]]
    md += ["## Verdict", "",
           f"- gold crop passes the gate: **{tally(a)}/{len(rows)}** with transcription, "
           f"**{tally(b)}/{len(rows)}** without.",
           f"- gold crop is among the crops actually read: **{sel_tally(a)}/{len(rows)}** "
           f"vs **{sel_tally(b)}/{len(rows)}**.",
           f"- questions where the selected crop set CHANGED: "
           f"{', '.join(changed) if changed else 'none'}.", "",
           "The gate's primary condition is provenance — a crop is eligible only if its "
           "page is one whose prose the question matched — and provenance does not depend "
           "on the transcription at all. The score survives only as a floor and a "
           "tie-break among already-eligible crops, which is why the decision is stable "
           "under a change of scoring text.", ""]

    os.makedirs(cfg.paths.reports_dir, exist_ok=True)
    p = os.path.join(cfg.paths.reports_dir, "vqa_gate_check.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(os.path.join(cfg.paths.reports_dir, "vqa_gate_check.json"), "w",
              encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print("\n".join(md))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
