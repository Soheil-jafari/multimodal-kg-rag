"""List exactly what goes in the submission, with sizes.

    python -m scripts.submission_manifest

Derived artifacts (chunk stores, FAISS indices, crops) are listed separately and
excluded by default: they are ~57 MB, they are regenerable from the scripts, and none of
the reported numbers depends on shipping them.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git"}
SKIP_EXT = (".log", ".pyc")

GROUPS = [
    ("A. DELIVERABLES — the submission proper", [
        "reports/FINAL_REPORT.pdf", "reports/FINAL_REPORT.md",
        "reports/EXPERIMENT_LOG.docx", "artifacts/EXPERIMENT_LOG.md", "reports/figures",
    ]),
    ("B. RESULTS — locked, every report number traces here", [
        "reports/scale500", "reports/ablation.md",
        "reports/ablation_v1_before_abstain_fix.md", "reports/vqa_gate_check.md",
        "reports/vqa_gate_check.json", "reports/vqa_figure_values.jsonl",
        "reports/phase9_retests.json",
    ]),
    ("C. GOLD SETS — evaluation ground truth", ["domain_packs/biomed/gold"]),
    ("D. SOURCE", [
        "platform_core", "evaluation", "domain_packs/biomed/predicates.yaml",
        "domain_packs/biomed/README.md", "domain_packs/__init__.py",
        "domain_packs/biomed/__init__.py", "configs", "scripts", "tests",
        "README.md", "requirements.txt", ".env.example",
    ]),
]
EXCLUDED = ["artifacts/dev", "artifacts/scale500"]


def size(n: int) -> str:
    return f"{n / 1048576:.1f} MB" if n >= 1048576 else f"{n / 1024:.0f} KB"


def walk(rel: str):
    p = os.path.join(ROOT, rel)
    if os.path.isfile(p):
        yield rel, os.path.getsize(p)
        return
    for dp, dn, fn in os.walk(p):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for f in sorted(fn):
            if f.endswith(SKIP_EXT):
                continue
            fp = os.path.join(dp, f)
            yield os.path.relpath(fp, ROOT).replace(os.sep, "/"), os.path.getsize(fp)


def main() -> None:
    grand = 0
    seen: set = set()
    for title, paths in GROUPS:
        print("=" * 72)
        print(title)
        print("=" * 72)
        sub = 0
        rows = []
        for rel in paths:
            for f, n in walk(rel):
                if f in seen:
                    continue
                seen.add(f)
                rows.append((f, n))
                sub += n
        for f, n in sorted(rows):
            print(f"  {size(n):>9}  {f}")
        print(f"  {'':>9}  -- {len(rows)} files, {size(sub)}\n")
        grand += sub

    print("=" * 72)
    print(f"SUBMISSION TOTAL: {len(seen)} files, {size(grand)}")
    print("=" * 72)
    print("\nEXCLUDED (regenerable, not needed to read or verify the results):")
    for rel in EXCLUDED:
        tot = sum(n for _, n in walk(rel))
        print(f"  {size(tot):>9}  {rel}/   (chunk store, FAISS indices, crops, KG)")
    print("\n  artifacts/EXPERIMENT_LOG.md is INCLUDED — only the binary data under")
    print("  artifacts/ is excluded. Rebuild with the commands in reports/scale500/README.md.")


if __name__ == "__main__":
    main()
