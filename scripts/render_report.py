"""Render the final report Markdown to a print-ready PDF.

    python -m scripts.render_report

The Markdown is the single source: the PDF is generated FROM it, so the two deliverables
cannot drift. Handles only the constructs the report actually uses (headings, paragraphs,
tables, bullets, fenced blocks, inline emphasis and code) — a general Markdown engine is
not needed and would add a dependency.

PDF comes from headless Chrome, which is already on the machine and gives real CSS page
control. The brief caps the report at two pages, so the page count is asserted rather than
hoped for.
"""
from __future__ import annotations

import argparse
import html
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
CHROME = [
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
]

CSS = """
@page { size: A4; margin: 8mm 7.5mm 6mm 7.5mm; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin:0; font-family:"Charter","Bitstream Charter","Georgia",serif;
       font-size:7.55pt; line-height:1.265; color:#14181d; }
.head { border-bottom:1.4pt solid #14181d; padding-bottom:1.8mm; margin-bottom:2mm; }
h1 { font-family:"Segoe UI","Helvetica Neue",sans-serif; font-size:13pt; line-height:1.1;
     margin:0 0 1mm; letter-spacing:-.2pt; }
.lede { font-size:7.9pt; color:#3d4650; margin:0; max-width:none; }
.cols { column-count:2; column-gap:6.5mm; column-fill:auto; }
h2 { font-family:"Segoe UI","Helvetica Neue",sans-serif; font-size:8.2pt; margin:2.1mm 0 .85mm;
     padding-bottom:.6mm; border-bottom:.5pt solid #b9c2cc; break-after:avoid;
     letter-spacing:.1pt; }
h2:first-child { margin-top:0; }
p { margin:0 0 1.05mm; text-align:justify; hyphens:auto; }
ul { margin:0 0 1.05mm; padding-left:3.1mm; }
li { margin:0 0 .55mm; text-align:justify; }
strong { font-weight:700; }
code { font-family:"Cascadia Mono","Consolas",monospace; font-size:7.2pt;
       background:#eef1f4; padding:0 .5mm; border-radius:1pt; }
pre { font-family:"Cascadia Mono","Consolas",monospace; font-size:5.55pt; line-height:1.24;
      background:#f4f6f8; border:.5pt solid #d5dce3; border-radius:1.5pt;
      padding:1.4mm 1.6mm; margin:0 0 1.8mm; white-space:pre; overflow:hidden;
      break-inside:avoid; }
table { border-collapse:collapse; width:100%; margin:0 0 1.3mm; font-size:6.35pt;
        break-inside:avoid; font-variant-numeric:tabular-nums; }
th,td { border:.4pt solid #c3cbd4; padding:.34mm .7mm; text-align:right; }
th:first-child, td:first-child { text-align:left; }
thead th { background:#e7ecf1; font-weight:700; }
tbody tr:nth-child(even) td { background:#f6f8fa; }
em { font-style:italic; color:#3d4650; }
.cite { display:block; border-left:1.6pt solid #8b96a2; background:#f2f4f7;
        padding:1mm 1.6mm; margin:0 0 1.8mm; font-size:7.4pt; }
figure { margin:.4mm 0 1.4mm; break-inside:avoid; text-align:center; }
figure img { width:50mm; max-width:100%; height:auto; }
figcaption { font-size:6.4pt; color:#3d4650; text-align:justify; margin:1mm 0 0;
             font-style:italic; }
.refs { font-size:6.3pt; line-height:1.22; }
"""

_INLINE = [
    (re.compile(r"`([^`]+)`"), r"<code>\1</code>"),
    (re.compile(r"\*\*([^*]+)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)"), r"<em>\1</em>"),
]


def inline(text: str) -> str:
    out = html.escape(text)
    for pat, rep in _INLINE:
        out = pat.sub(rep, out)
    return out.replace("--&gt;", "&rarr;")


def to_html(md: str) -> tuple[str, str, str]:
    """(title, lede, body_html) — the lede is the paragraph under the H1."""
    lines = md.split("\n")
    title, lede, body = "", "", []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("# "):
            title = inline(line[2:].strip()); i += 1; continue
        if line.startswith("## "):
            body.append(f"<h2>{inline(line[3:].strip())}</h2>"); i += 1; continue
        if line.startswith("```"):
            block, i = [], i + 1
            while i < n and not lines[i].startswith("```"):
                block.append(lines[i]); i += 1
            i += 1
            body.append("<pre>" + html.escape("\n".join(block)) + "</pre>")
            continue
        if line.startswith("!["):
            m = re.match(r"!\[([^\]]*)\]\(([^)]*)\)", line.strip())
            i += 1
            if m:
                alt, src = m.group(1), m.group(2)
                body.append(f'<figure><img src="{html.escape(src)}" '
                            f'alt="{html.escape(alt)}">')
                # an italic paragraph immediately after the image is its caption
                while i < n and not lines[i].strip():
                    i += 1
                cap = []
                while i < n and lines[i].strip() and not lines[i].startswith(("#", "|", "- ", "```", "![")):
                    cap.append(lines[i].strip()); i += 1
                text = " ".join(cap)
                if text.startswith("*") and text.endswith("*"):
                    body.append(f"<figcaption>{inline(text)}</figcaption></figure>")
                else:
                    body.append("</figure>")
                    if text:
                        body.append(f"<p>{inline(text)}</p>")
            continue
        if line.startswith("|"):
            rows, i = [], i
            while i < n and lines[i].startswith("|"):
                rows.append(lines[i]); i += 1
            body.append(table_html(rows))
            continue
        if line.startswith("- "):
            items, i = [], i
            while i < n and lines[i].startswith("- "):
                item = lines[i][2:]
                i += 1
                while i < n and lines[i].startswith("  ") and lines[i].strip():
                    item += " " + lines[i].strip(); i += 1
                items.append(f"<li>{inline(item.strip())}</li>")
            body.append("<ul>" + "".join(items) + "</ul>")
            continue
        if not line.strip():
            i += 1; continue
        para, i = [], i
        while i < n and lines[i].strip() and not lines[i].startswith(("#", "|", "- ", "```")):
            para.append(lines[i].strip()); i += 1
        text = " ".join(para)
        cls = ' class="refs"' if text.startswith("**References.**") else ""
        rendered = f"<p{cls}>{inline(text)}</p>"
        if not title or lede or body:
            body.append(rendered)
        else:
            lede = inline(text)
    return title, lede, "\n".join(body)


def cells(row: str) -> list:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def table_html(rows: list) -> str:
    if len(rows) < 2:
        return ""
    head, out = cells(rows[0]), []
    body_rows = rows[2:]  # row 1 is the alignment separator
    out.append("<thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr></thead>")
    out.append("<tbody>")
    for r in body_rows:
        out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells(r)) + "</tr>")
    out.append("</tbody>")
    return "<table>" + "".join(out) + "</table>"


def page_count(pdf: str) -> int:
    with open(pdf, "rb") as f:
        data = f.read()
    return max(data.count(b"/Type /Page\n"), data.count(b"/Type/Page"),
               len(re.findall(rb"/Type\s*/Page[^s]", data)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=ROOT + "/reports/FINAL_REPORT.md")
    ap.add_argument("--pdf", default=ROOT + "/reports/FINAL_REPORT.pdf")
    ap.add_argument("--max-pages", type=int, default=2)
    args = ap.parse_args()

    md = open(args.md, encoding="utf-8").read()
    title, lede, body = to_html(md)
    doc = (f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>"
           f"<style>{CSS}</style></head><body>"
           f"<div class='head'><h1>{title}</h1><p class='lede'>{lede}</p></div>"
           f"<div class='cols'>{body}</div></body></html>")
    htm = args.pdf.replace(".pdf", ".html")
    with open(htm, "w", encoding="utf-8") as f:
        f.write(doc)

    exe = next((c for c in CHROME if os.path.exists(c)), None)
    if not exe:
        raise SystemExit("no Chrome/Edge found for PDF rendering")
    if os.path.exists(args.pdf):
        os.remove(args.pdf)
    subprocess.run([exe, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={args.pdf}", "file:///" + htm],
                   check=True, capture_output=True, timeout=180)
    pages = page_count(args.pdf)
    size = os.path.getsize(args.pdf) / 1024
    print(f"{args.pdf}  {pages} page(s), {size:.0f} KB")
    if pages > args.max_pages:
        print(f"OVER the {args.max_pages}-page limit — tighten before submitting")
        sys.exit(1)


if __name__ == "__main__":
    main()
