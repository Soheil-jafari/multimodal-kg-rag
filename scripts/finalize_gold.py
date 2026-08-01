"""Apply the human figure_value verification to produce the final gold set.

    python -m scripts.finalize_gold --config configs/scale500.yaml

The figure_value category cannot be certified mechanically: its answers are numbers in
a grid, and every text route to them is known-unreliable (OCR has the right digits in
the wrong structure; the transcription has the right structure with some wrong digits;
the vision model is the system under test and cannot author its own answer key). So the
17 candidates were rendered beside their crops and checked by a person against the
images. This script records that verdict — verbatim, with the reason for every drop —
and writes the set the evaluation actually runs on.

The pre-verification set is preserved alongside it. The drop rate is not bookkeeping:
it is the measurement of how unreliable table-OCR-derived values are, and it is
reported as a finding.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from platform_core.config import AppConfig

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
DEFAULT_CONFIG = R + "/configs/scale500.yaml"

#: Human verdict, read off the crops (reports/scale500/gold_review.html, slots V01-V17).
#: V01-V03 are the graph-selected items; V04-V17 the table-OCR proposals.
KEEP = {"V02", "V03", "V04", "V06", "V08", "V10", "V14", "V16"}
DROP_REASON = {
    "V01": "circular — 'CYP1A inhibits CYP1A protein expression' is self-referential",
    "V05": "wrong value AND q061-class defect: the table reports NO2, the question says NO",
    "V07": "wrong value",
    "V09": "unverifiable — crop too poor to read",
    "V11": "unreadable crop",
    "V12": "value misplaced: 26 is not e432's maximum (81 is)",
    "V13": "wrong value",
    "V15": "wrong value — should be 8.03 +/- 0.19",
    "V17": "wrong value — drops the x10^-4 exponent",
}
#: Corrections applied to kept items, as given by the reviewer.
FIXES = {"V14": {"replace": [("Fkelihood", "Likelihood")]}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    args = ap.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    gold_path = cfg.paths.gold_set
    gold = [json.loads(l) for l in open(gold_path, encoding="utf-8")]
    cands = [json.loads(l) for l in open(
        os.path.join(cfg.paths.reports_dir, "figure_value_candidates.jsonl"),
        encoding="utf-8")]

    fv = [g for g in gold if g["category"] == "figure_value"]
    if len(fv) != 3 or len(cands) != 14:
        raise SystemExit(f"expected 3 graph-derived + 14 candidates, got {len(fv)} + {len(cands)}")
    slot_of = {id(g): f"V{i:02d}" for i, g in enumerate(fv, 1)}
    cand_slot = {i: f"V{i + 4:02d}" for i in range(len(cands))}

    def fix(rec: dict, slot: str) -> dict:
        for old, new in FIXES.get(slot, {}).get("replace", []):
            rec["question"] = rec["question"].replace(old, new)
            if rec.get("row_label"):
                rec["row_label"] = rec["row_label"].replace(old, new)
        return rec

    out = [g for g in gold if g["category"] != "figure_value"]
    kept_fv: list = []
    for g in fv:
        slot = slot_of[id(g)]
        if slot in KEEP:
            g = fix(dict(g), slot)
            g.pop("needs_value_verification", None)
            g.update({"review_slot": slot, "value_verified": True,
                      "value_source": "graph edge, checked against the crop by a human"})
            kept_fv.append(g)

    next_n = max(int(r["qid"][1:]) for r in gold) + 1
    for i, c in enumerate(cands):
        slot = cand_slot[i]
        if slot not in KEEP:
            continue
        c = fix(dict(c), slot)
        kept_fv.append({
            "qid": f"q{next_n:03d}", "question": c["question"].strip(),
            "category": "figure_value", "expected_answer": c["proposed_answer"].strip(),
            "supporting_chunk_ids": c["supporting_chunk_ids"], "source_predicates": [],
            "review_slot": slot, "value_verified": True, "crop_path": c.get("crop_path"),
            "value_source": "read from the table crop by a human (never the vision model)",
        })
        next_n += 1
    out += kept_fv

    backup = gold_path.replace(".jsonl", "_prereview.jsonl")
    if not os.path.exists(backup):
        shutil.copy2(gold_path, backup)
    with open(gold_path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_drop = len(DROP_REASON)
    n_tot = len(fv) + len(cands)
    graph_drop = sum(1 for s in DROP_REASON if s in {"V01", "V02", "V03"})
    print(f"final gold set: {len(out)} questions -> {gold_path}")
    print(f"pre-verification set preserved -> {backup}\n")
    for cat, k in collections.Counter(r["category"] for r in out).most_common():
        print(f"  {cat:14s} {k}")
    print(f"\nfigure_value verification: {len(kept_fv)}/{n_tot} kept, "
          f"{n_drop}/{n_tot} rejected ({100*n_drop/n_tot:.0f}%)")
    print(f"  of the 14 table-OCR proposals : {n_drop - graph_drop} rejected "
          f"({100*(n_drop-graph_drop)/len(cands):.0f}%)")
    print(f"  of the 3 graph-selected items : {graph_drop} rejected")
    print("\nrejections:")
    for s, why in sorted(DROP_REASON.items()):
        print(f"  {s}  {why}")
    low = [c for c, k in collections.Counter(r["category"] for r in out).items() if k < 5]
    print(f"\nlow-sample categories (n<5): {low or 'none'}")


if __name__ == "__main__":
    main()
