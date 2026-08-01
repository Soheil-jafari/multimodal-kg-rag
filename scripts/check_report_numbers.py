"""Trace every figure in the final report back to a locked artifact.

    python -m scripts.check_report_numbers

A report that quotes a number no artifact contains is the failure mode this guards
against. Primary source is `reports/scale500/`; anything not found there is reported with
where it actually comes from, rather than passed over.
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "reports/FINAL_REPORT.md")
PRIMARY = os.path.join(ROOT, "reports/scale500")
SEARCH = ["reports", "artifacts", "domain_packs"]
TEXTY = (".md", ".json", ".jsonl", ".log", ".txt")
#: citation years and identifiers are not results
CITATION = {"2020", "2022", "2024", "2025", "2026", "16130", "15552", "9459", "9474"}


def read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def corpus(paths) -> str:
    out = []
    for p in paths:
        for dp, _, fns in os.walk(p):
            if "__pycache__" in dp:
                continue
            for fn in fns:
                if fn.endswith(TEXTY):
                    out.append(read(os.path.join(dp, fn)))
    return "\n".join(out)


def locate(needle: str) -> list:
    hits = []
    for p in SEARCH:
        for dp, _, fns in os.walk(os.path.join(ROOT, p)):
            if "__pycache__" in dp:
                continue
            for fn in fns:
                if fn.endswith(TEXTY) and needle in read(os.path.join(dp, fn)):
                    hits.append(os.path.relpath(os.path.join(dp, fn), ROOT).replace("\\", "/"))
    return sorted(set(hits))


def main() -> None:
    md = read(REPORT)
    body = md.split("**References.**")[0]  # citation years/ids are not results
    primary = corpus([PRIMARY])

    decimals = sorted(set(re.findall(r"\d+\.\d+", body)))
    ints = sorted({n for n in re.findall(r"(?<![\w.@:/-])(\d{2,5})(?![\w.%])", body)}
                  - CITATION)
    miss_d = [n for n in decimals if n not in primary]
    miss_i = [n for n in ints if n not in primary]

    print(f"report: {os.path.relpath(REPORT, ROOT)}")
    print(f"primary source: reports/scale500/  "
          f"({sum(1 for f in os.listdir(PRIMARY) if f.endswith(TEXTY))} text files)\n")
    print(f"decimal figures : {len(decimals):3d}   traced to primary: "
          f"{len(decimals) - len(miss_d)}   elsewhere: {len(miss_d)}")
    print(f"integer figures : {len(ints):3d}   traced to primary: "
          f"{len(ints) - len(miss_i)}   elsewhere: {len(miss_i)}")

    if miss_d or miss_i:
        print("\nNOT in reports/scale500/ — provenance of each:")
        for n in miss_d + miss_i:
            where = [w for w in locate(n) if not w.startswith("reports/FINAL_REPORT")]
            print(f"  {n:8s} -> {', '.join(where[:2]) if where else '*** NO ARTIFACT ***'}")

    # the CoT noise floor is a derived quantity: recompute it from the raw scale500 logs
    cot = [json.loads(l) for l in open(os.path.join(PRIMARY, "cot_compare.jsonl"),
                                       encoding="utf-8")]
    frozen = [json.loads(l) for l in open(os.path.join(PRIMARY, "perq_+rerank.jsonl"),
                                          encoding="utf-8")]
    cm = [r for r in cot if r["category"] == "multi_hop"]
    fm = [r for r in frozen if r["category"] == "multi_hop"]
    off = sum(r["off_correct"] for r in cm) / len(cm)
    fr = sum(r["correctness"] for r in fm) / len(fm)
    print(f"\nderived check — CoT noise floor (report states +0.029):")
    print(f"  perq_+rerank.jsonl multi_hop correctness  {fr:.3f}")
    print(f"  cot_compare.jsonl  CoT-off, same flags    {off:.3f}")
    print(f"  difference {off - fr:+.4f}  -> "
          f"{'MATCH' if abs((off - fr) - 0.029) < 6e-4 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
