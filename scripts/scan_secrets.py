"""Scan every git-tracked file for a leaked credential. Exits non-zero on a hit.

    python -m scripts.scan_secrets

Reads the file list from git, not the filesystem, so it checks exactly what a push
would publish. A `.gitignore` entry is a claim; this verifies it.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: (name, pattern, is_fatal). Non-fatal patterns are reported for eyeballing — the word
#: "OPENAI" legitimately appears in variable names and prose all over this project.
RULES = [
    ("OpenAI secret key",  re.compile(r"sk-[A-Za-z0-9_\-]{20,}"), True),
    ("AWS access key",     re.compile(r"AKIA[0-9A-Z]{16}"), True),
    ("GitHub token",       re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"), True),
    ("Slack token",        re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), True),
    ("private key block",  re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), True),
    ("bearer token",       re.compile(r"[Bb]earer\s+[A-Za-z0-9._\-]{25,}"), True),
    # assignment of a long literal to a secret-looking name
    ("hardcoded secret",   re.compile(
        r"""(?i)(api[_-]?key|secret|passwd|password|token)\s*[:=]\s*["'][^"'\s]{16,}["']"""), True),
    ("mentions OPENAI",    re.compile(r"OPENAI"), False),
    ("mentions api_key",   re.compile(r"(?i)api[_-]?key"), False),
]
BINARY = (".pdf", ".docx", ".png", ".jpg", ".svg", ".sqlite", ".faiss", ".pkl", ".zip")


def tracked() -> list:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return [f for f in out.split("\0") if f]


def main() -> None:
    files = tracked()
    fatal, soft = [], {}
    for rel in files:
        path = os.path.join(ROOT, rel)
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except OSError:
            continue
        if b"\0" in raw[:4096] and rel.endswith(BINARY):
            text = ""  # binary deliverable: still scanned below via latin-1 fallback
            try:
                text = raw.decode("latin-1", errors="ignore")
            except Exception:
                pass
        else:
            text = raw.decode("utf-8", errors="ignore")
        for name, pat, is_fatal in RULES:
            for m in pat.finditer(text):
                if is_fatal:
                    ln = text[:m.start()].count("\n") + 1
                    fatal.append((rel, ln, name, m.group(0)[:14] + "..."))
                else:
                    soft.setdefault(name, set()).add(rel)

    print(f"tracked files scanned: {len(files)}\n")
    print("FATAL patterns (real credentials):")
    if fatal:
        for rel, ln, name, snip in fatal:
            print(f"  !! {rel}:{ln}  {name}  {snip}")
    else:
        print("  none — no key, token or private key in any tracked file")

    print("\nInformational — files merely MENTIONING these words:")
    for name, rels in sorted(soft.items()):
        shown = sorted(rels)[:6]
        print(f"  {name}: {len(rels)} file(s) — {', '.join(shown)}"
              + (" ..." if len(rels) > len(shown) else ""))

    print("\n.env tracked? " + ("YES — ABORT" if ".env" in files else "no"))
    print(".env.example tracked? " + ("yes (placeholders)" if ".env.example" in files else "no"))
    sys.exit(1 if fatal or ".env" in files else 0)


if __name__ == "__main__":
    main()
