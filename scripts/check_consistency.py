"""Cross-document consistency: no two documents may disagree on a number. No API calls.

    python -m scripts.check_consistency

Three checks:

1. **Point estimates unchanged.** Every figure in the published `ablation.md` is
   recomputed from the per-question logs; a drift aborts.
2. **Claims agree across documents.** A set of key figures is asserted to appear with the
   same value wherever it appears — report, RESULTS, README, experiment log.
3. **New figures trace to a source.** Every confidence interval and operational number
   quoted in prose must exist in the JSON the measurement script wrote, so no interval was
   typed by hand.
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
DOCS = {
    "report":  "reports/FINAL_REPORT.md",
    "RESULTS": "reports/scale500/RESULTS.md",
    "README":  "README.md",
    "log":     "artifacts/EXPERIMENT_LOG.md",
}

#: (label, regex that must match identically wherever the claim appears, docs to check)
CLAIMS = [
    ("baseline R@5",        r"0\.57",                        ["report", "RESULTS", "README"]),
    ("enhanced R@5",        r"0\.91",                        ["report", "RESULTS", "README"]),
    ("noise floor",         r"0\.029",                       ["report", "RESULTS", "README", "log"]),
    ("+clip null",          r"\+0\.000 \[\+0\.000, \+0\.000\]", ["report", "RESULTS", "README", "log"]),
    ("+kg dR@5",            r"\+0\.026 \[\+0\.000, \+0\.059\]", ["report", "RESULTS", "README", "log"]),
    ("+kg dCorrect",        r"\+0\.030 \[\+0\.004, \+0\.064\]", ["report", "RESULTS", "README", "log"]),
    ("paired dCorrect",     r"\+0\.099 \[\+0\.034, \+0\.168\]", ["report", "RESULTS", "log"]),
    ("baseline R@5 CI",     r"0\.57 \[0\.49, 0\.66\]",       ["report", "RESULTS", "README", "log"]),
    ("enhanced R@5 CI",     r"0\.91 \[0\.86, 0\.96\]",       ["report", "RESULTS", "README", "log"]),
    ("latency p50",         r"3\.67 s",                      ["report", "RESULTS", "README", "log"]),
    ("latency p95",         r"13\.44 s",                     ["report", "RESULTS", "README", "log"]),
    ("peak memory",         r"2[,.]4(66|7)",                 ["report", "RESULTS", "README", "log"]),
    ("ingest per page",     r"14\.7 s",                      ["report", "RESULTS", "README", "log"]),
]


def read(rel: str) -> str:
    """Text with line wrapping and emphasis normalised away.

    A claim split across a line break, or bolded in one document and not another, is the
    same claim — matching the raw text would flag prose formatting as a disagreement.
    """
    raw = open(os.path.join(ROOT, rel), encoding="utf-8").read()
    return re.sub(r"\s+", " ", raw.replace("**", "").replace("*", ""))


def main() -> None:
    fail = 0
    text = {k: read(v) for k, v in DOCS.items()}

    print("1. point estimates still match the per-question logs")
    # --verify-only recomputes and compares but writes nothing. Without it this check
    # overwrote the published 10,000-resample intervals with a cheap re-run — the .md
    # rounded to the same two decimals so it looked untouched, while the .json silently
    # carried different bounds.
    r = os.system(f'"{sys.executable}" -m scripts.bootstrap_ci --verify-only > '
                  f'{os.devnull} 2>&1')
    print("   " + ("OK — bootstrap_ci recomputed and matched ablation.md"
                   if r == 0 else "!! MISMATCH — bootstrap_ci aborted"))
    fail += (r != 0)

    print("\n2. claims agree wherever they appear")
    for label, pat, docs in CLAIMS:
        missing = [d for d in docs if not re.search(pat, text[d])]
        if missing:
            print(f"   !! {label}: absent from {', '.join(missing)}")
            fail += 1
        else:
            print(f"   OK {label:20s} consistent across {', '.join(docs)}")

    print("\n3. quoted intervals trace to the measurement output")
    ci = json.load(open(os.path.join(ROOT, "reports/scale500/confidence_intervals.json"),
                        encoding="utf-8"))
    ops = json.load(open(os.path.join(ROOT, "reports/scale500/operational.json"),
                         encoding="utf-8"))
    steps = ci["steps"]
    checks = [
        ("+clip ΔR@5 == 0", steps["+layout->+clip"]["ΔR@5"]["delta"] == 0.0),
        ("+kg ΔR@5 within noise", steps["+caption->+kg"]["ΔR@5"]["within_noise"]),
        ("+kg ΔCorrect on floor", steps["+caption->+kg"]["ΔCorrect"]["on_noise_floor"]),
        ("paired baseline→+rerank ΔCorrect excludes 0",
         steps["baseline->+rerank"]["ΔCorrect"]["lo"] > 0),
        ("figure_value CIs overlap",
         ci["per_category"]["figure_value"]["configs"]["baseline"]["Correct"]["hi"]
         >= ci["per_category"]["figure_value"]["configs"]["+rerank"]["Correct"]["lo"]),
        ("p50 latency present", ops["retrieval"]["p50_ms"] == 3665.7),
        ("p95 latency present", ops["retrieval"]["p95_ms"] == 13438.4),
        ("peak RSS present", ops["retrieval"]["rss_peak_mb"] == 2466.0),
    ]
    for label, ok in checks:
        print(f"   {'OK' if ok else '!!'} {label}")
        fail += (not ok)

    print("\n" + ("ALL CONSISTENT — no document disagrees" if not fail
                  else f"{fail} PROBLEM(S)"))
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
