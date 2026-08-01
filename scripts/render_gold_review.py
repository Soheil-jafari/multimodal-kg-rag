"""Render the gold set as a visual spot-check worksheet with the crops embedded.

    python -m scripts.render_gold_review --config configs/scale500.yaml

A figure_value question cannot be verified from text: its answer is a number in a
grid, and both routes to that number in text form are known-unreliable (OCR has the
right digits in the wrong structure; the transcription has the right structure with
some wrong digits). So the only honest review surface puts the crop IMAGE next to the
claim and lets a person read it. Crops are inlined as data URIs at full resolution —
downscaling would destroy the digits that are the point of the exercise.

Writes a self-contained HTML file — crops inlined, no external assets — so it opens
from disk in a browser with nothing to serve.
"""
from __future__ import annotations

import argparse
import base64
import collections
import html
import json
import mimetypes
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from platform_core.config import AppConfig
from platform_core.stores.chunk_store import SQLiteChunkStore

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
DEFAULT_CONFIG = R + "/configs/scale500.yaml"
ORDER = ["single_fact", "multi_hop", "figure", "text_derived", "unanswerable"]


def e(s) -> str:
    return html.escape(str(s or ""))


def flat(s, n=400) -> str:
    one = " ".join((s or "").split())
    return one[:n] + ("…" if len(one) > n else "")


def data_uri(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode("ascii")


CSS = """
:root{
  --ground:#FBFCFD; --surface:#FFFFFF; --sunk:#F2F5F8;
  --ink:#16202B; --muted:#5C6B79; --rule:#DCE4EB;
  --accent:#1A6E68; --accent-soft:#E4F0EE;
  --warn:#A8560C; --warn-soft:#FBF0E2;
  --ok:#2C6E49; --ok-soft:#E6F2EA;
  --serif:ui-serif,Georgia,"Iowan Old Style","Times New Roman",serif;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:ui-monospace,"Cascadia Mono","SF Mono",Consolas,"Liberation Mono",monospace;
}
@media (prefers-color-scheme:dark){
  :root{
    --ground:#0E1419; --surface:#161E25; --sunk:#1B242C;
    --ink:#E6EDF3; --muted:#93A3B2; --rule:#26313A;
    --accent:#52B8B0; --accent-soft:#12312F;
    --warn:#E0973F; --warn-soft:#33230F;
    --ok:#6BBF8A; --ok-soft:#12291C;
  }
}
:root[data-theme="dark"]{
  --ground:#0E1419; --surface:#161E25; --sunk:#1B242C;
  --ink:#E6EDF3; --muted:#93A3B2; --rule:#26313A;
  --accent:#52B8B0; --accent-soft:#12312F;
  --warn:#E0973F; --warn-soft:#33230F;
  --ok:#6BBF8A; --ok-soft:#12291C;
}
:root[data-theme="light"]{
  --ground:#FBFCFD; --surface:#FFFFFF; --sunk:#F2F5F8;
  --ink:#16202B; --muted:#5C6B79; --rule:#DCE4EB;
  --accent:#1A6E68; --accent-soft:#E4F0EE;
  --warn:#A8560C; --warn-soft:#FBF0E2;
  --ok:#2C6E49; --ok-soft:#E6F2EA;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
  font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:48px 24px 96px;
  display:flex;flex-direction:column;gap:40px}
h1,h2,h3{font-family:var(--serif);font-weight:600;text-wrap:balance;margin:0;line-height:1.25}
h1{font-size:clamp(28px,4vw,40px);letter-spacing:-.015em}
h2{font-size:24px;letter-spacing:-.01em}
h3{font-size:17px}
p{margin:0;max-width:68ch}
a{color:var(--accent)}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--muted)}
.lede{color:var(--muted);font-size:15px}
header{display:flex;flex-direction:column;gap:12px;border-bottom:1px solid var(--rule);
  padding-bottom:28px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(132px,1fr));gap:1px;
  background:var(--rule);border:1px solid var(--rule);border-radius:8px;overflow:hidden}
.tile{background:var(--surface);padding:14px 16px;display:flex;flex-direction:column;gap:2px}
.tile .n{font-family:var(--mono);font-size:22px;font-variant-numeric:tabular-nums;
  letter-spacing:-.02em}
.tile .k{font-size:12px;color:var(--muted)}
.tile.short .n{color:var(--warn)}
.note{border-left:3px solid var(--warn);background:var(--warn-soft);padding:14px 18px;
  border-radius:0 6px 6px 0;display:flex;flex-direction:column;gap:6px}
.note.accent{border-left-color:var(--accent);background:var(--accent-soft)}
.note h3{font-family:var(--sans);font-size:14px;font-weight:650}
.note p{font-size:14px}
section{display:flex;flex-direction:column;gap:20px}
.sechead{display:flex;flex-direction:column;gap:6px;border-bottom:1px solid var(--rule);
  padding-bottom:12px}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:10px;
  overflow:hidden;display:flex;flex-direction:column}
.card-top{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;
  padding:14px 18px;border-bottom:1px solid var(--rule);background:var(--sunk)}
.slot{font-family:var(--mono);font-size:12px;font-weight:700;color:var(--accent);
  font-variant-numeric:tabular-nums}
.qid{font-family:var(--mono);font-size:12px;color:var(--muted)}
.chip{font-family:var(--mono);font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;
  padding:3px 8px;border-radius:99px;border:1px solid var(--rule);color:var(--muted);
  background:var(--surface);white-space:nowrap}
.chip.warn{color:var(--warn);border-color:var(--warn);background:var(--warn-soft)}
.chip.ok{color:var(--ok);border-color:var(--ok);background:var(--ok-soft)}
.chip.accent{color:var(--accent);border-color:var(--accent);background:var(--accent-soft)}
.split{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:0}
@media (max-width:860px){.split{grid-template-columns:1fr}}
.pane{padding:18px;display:flex;flex-direction:column;gap:12px;min-width:0}
.pane+.pane{border-left:1px solid var(--rule)}
@media (max-width:860px){.pane+.pane{border-left:0;border-top:1px solid var(--rule)}}
.cropbox{background:var(--sunk);border:1px solid var(--rule);border-radius:6px;
  padding:10px;overflow-x:auto}
.cropbox img{display:block;max-width:100%;height:auto;image-rendering:crisp-edges;
  background:#fff;border-radius:2px}
.q{font-size:16px;font-weight:600;line-height:1.45}
.kv{display:flex;flex-direction:column;gap:3px}
.kv .k{font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--muted)}
.val{font-family:var(--mono);font-size:19px;font-weight:700;color:var(--ink);
  font-variant-numeric:tabular-nums;word-break:break-word}
.val.unver{color:var(--warn)}
.ocr{font-family:var(--mono);font-size:11.5px;line-height:1.55;color:var(--muted);
  background:var(--sunk);border:1px solid var(--rule);border-radius:6px;padding:10px;
  max-height:150px;overflow:auto;white-space:pre-wrap;word-break:break-word}
.small{font-size:13px;color:var(--muted)}
ul.rows{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;
  border:1px solid var(--rule);border-radius:10px;overflow:hidden;background:var(--surface)}
ul.rows li{padding:14px 18px;display:flex;flex-direction:column;gap:6px}
ul.rows li+li{border-top:1px solid var(--rule)}
.rowtop{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}
.ans{font-family:var(--mono);font-size:13px;color:var(--accent);word-break:break-word}
.src{font-family:var(--mono);font-size:11.5px;color:var(--muted);word-break:break-word}
footer{border-top:1px solid var(--rule);padding-top:20px;color:var(--muted);font-size:13px}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--out", default=R + "/reports/scale500/gold_review.html")
    ap.add_argument("--per-category", type=int, default=6)
    args = ap.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    store = SQLiteChunkStore(cfg.paths.chunk_db)
    gold = [json.loads(l) for l in open(cfg.paths.gold_set, encoding="utf-8")]
    cand_path = os.path.join(cfg.paths.reports_dir, "figure_value_candidates.jsonl")
    cands = ([json.loads(l) for l in open(cand_path, encoding="utf-8")]
             if os.path.exists(cand_path) else [])
    counts = collections.Counter(g["category"] for g in gold)
    quota = {"single_fact": 55, "text_derived": 23, "multi_hop": 17,
             "unanswerable": 14, "figure": 12, "figure_value": 9}

    def chunk(cid):
        try:
            return store.get(cid)
        except KeyError:
            return None

    P: list[str] = []
    P.append('<div class="wrap">')
    P.append(f"""<header>
<div class="eyebrow">Phase 10 · spot-check before evaluation</div>
<h1>Gold set for the 500-page corpus</h1>
<p class="lede">{len(gold)} questions selected from the {cfg.name} knowledge graph, plus
{len(cands)} table-value candidates awaiting verification. Automated filters only —
nothing here is human-verified yet. The figure_value section comes first because those are
the items that cannot be checked from text.</p>
</header>""")

    tiles = "".join(
        f'<div class="tile{" short" if counts.get(c,0)<quota.get(c,0) else ""}">'
        f'<span class="n">{counts.get(c,0)}</span>'
        f'<span class="k">{c.replace("_"," ")} <span style="opacity:.6">/ {quota.get(c,0)}</span></span></div>'
        for c in ["single_fact", "text_derived", "multi_hop", "figure", "unanswerable", "figure_value"])
    P.append(f'<div class="tiles">{tiles}</div>')

    P.append("""<div class="note">
<h3>Two categories came up short, and both are findings rather than accidents</h3>
<p><strong>figure_value 3 / 9.</strong> The closed 8-predicate schema has no value-relation
&mdash; <code>measured_by</code> names the method, not the number &mdash; so a table's values
never become graph edges. This is the structural reason Phase 6A found zero clean table-value
edges. The 14 candidates below were proposed from table OCR to fill the gap, exactly as Phase 6A
hand-built its four.</p>
<p><strong>unanswerable 11 / 14.</strong> The absence check rejected 13 of 24 crafted candidates
because this corpus actually discusses them &mdash; warfarin/INR scored 0.729 against the index,
paracetamol 0.688, metformin 0.685. At 50 pages those topics were safely absent; at 500 they are
not. Without that check they would have been silently answerable and the 100% abstention result
would have measured nothing.</p></div>""")

    P.append("""<div class="note accent">
<h3>What to check on a table-value item</h3>
<p>Read the value off the <strong>image</strong>, not the OCR beside it. Confirm three things: the
number is right, it belongs to the row and column the question names, and the question does not
name a row or column the table lacks &mdash; that last one is the q061 defect, which no automated
filter caught before it reached the published set.</p></div>""")

    # ---------- figure_value: graph-derived first, then proposals ----------
    fv = [g for g in gold if g["category"] == "figure_value"]
    P.append('<section><div class="sechead"><div class="eyebrow">Priority</div>'
             '<h2>Table values &mdash; verify against the crop</h2>'
             f'<p class="small">{len(fv)} selected from graph edges &middot; '
             f'{len(cands)} proposed from table OCR. Every value below is unverified.</p></div>')

    slot = 0
    for g in fv:
        slot += 1
        c = chunk(g["supporting_chunk_ids"][0]) if g["supporting_chunk_ids"] else None
        uri = data_uri(g.get("crop_path") or (c.image_path if c else ""))
        cap = ""
        if c is not None and c.caption_id:
            cc = chunk(c.caption_id)
            cap = flat(cc.text, 220) if cc else ""
        P.append(f"""<article class="card">
<div class="card-top"><span class="slot">V{slot:02d}</span>
<span class="qid">{e(g['qid'])}</span>
<span class="chip accent">graph edge</span>
{"".join(f'<span class="chip">{e(p)}</span>' for p in g.get('source_predicates', []))}
<span class="chip warn">value unverified</span></div>
<div class="split">
<div class="pane">{'<div class="cropbox"><img src="' + uri + '" alt="table crop ' + e(g['supporting_chunk_ids'][0]) + '"></div>' if uri else '<p class="small">no crop on file</p>'}
<div class="src">{e(g['supporting_chunk_ids'][0] if g['supporting_chunk_ids'] else '')}</div></div>
<div class="pane">
<div class="q">{e(g['question'])}</div>
<div class="kv"><span class="k">answer from the edge</span>
<span class="val unver">{e(g['expected_answer'])}</span></div>
{f'<div class="kv"><span class="k">caption</span><span class="small">{e(cap)}</span></div>' if cap else ''}
<div class="kv"><span class="k">table OCR &mdash; digits right, grid wrong</span>
<div class="ocr">{e(flat(c.text, 900) if c else '')}</div></div>
</div></div></article>""")

    for r in cands:
        slot += 1
        ok = r.get("value_digits_present_in_ocr")
        uri = data_uri(r.get("crop_path"))
        P.append(f"""<article class="card">
<div class="card-top"><span class="slot">V{slot:02d}</span>
<span class="qid">candidate</span>
<span class="chip">table OCR proposal</span>
<span class="chip {'ok' if ok else 'warn'}">{'digits found in OCR' if ok else 'digits NOT in OCR'}</span>
<span class="chip warn">value unverified</span></div>
<div class="split">
<div class="pane">{'<div class="cropbox"><img src="' + uri + '" alt="table crop ' + e(r['supporting_chunk_ids'][0]) + '"></div>' if uri else '<p class="small">no crop on file</p>'}
<div class="src">{e(r['supporting_chunk_ids'][0])}</div></div>
<div class="pane">
<div class="q">{e(r['question'])}</div>
<div class="kv"><span class="k">proposed value</span>
<span class="val unver">{e(r['proposed_answer'])}</span></div>
{f'<div class="kv"><span class="k">row / column label used</span><span class="small">{e(r["row_label"])}</span></div>' if r.get('row_label') else ''}
<div class="kv"><span class="k">caption</span><span class="small">{e(flat(r.get('caption'), 220))}</span></div>
<div class="kv"><span class="k">table OCR &mdash; digits right, grid wrong</span>
<div class="ocr">{e(flat(r.get('ocr_text'), 900))}</div></div>
</div></div></article>""")
    P.append("</section>")

    # ---------- the other categories ----------
    by_cat: dict = collections.defaultdict(list)
    for g in gold:
        by_cat[g["category"]].append(g)

    for cat in ORDER:
        items = by_cat.get(cat, [])
        if not items:
            continue
        stride = max(1, len(items) // args.per_category)
        show = items[::stride][:args.per_category]
        P.append(f'<section><div class="sechead"><div class="eyebrow">{e(cat.replace("_"," "))}</div>'
                 f'<h2>{len(show)} of {len(items)}</h2></div><ul class="rows">')
        for g in show:
            chips = "".join(f'<span class="chip">{e(p)}</span>'
                            for p in g.get("source_predicates", []))
            if g.get("needs_mismatch_review"):
                chips += (f'<span class="chip warn">check qualifier &middot; '
                          f'{g.get("mismatch_miss_ratio")} missing: '
                          f'{e(", ".join(g.get("mismatch_missing_terms", [])[:4]))}</span>')
            if g.get("absence_check_top_cosine") is not None:
                chips += (f'<span class="chip ok">absent &middot; top cosine '
                          f'{g["absence_check_top_cosine"]}</span>')
            srcs = []
            for cid in g["supporting_chunk_ids"]:
                c = chunk(cid)
                srcs.append(f'{e(cid)} &mdash; {e(flat(c.text if c else "", 210))}')
            P.append(f"""<li><div class="rowtop"><span class="qid">{e(g['qid'])}</span>{chips}</div>
<div class="q">{e(g['question'])}</div>
<div class="ans">{e(flat(g['expected_answer'], 260))}</div>
{"".join(f'<div class="src">{s}</div>' for s in srcs)}</li>""")
        P.append("</ul></section>")

    P.append(f"""<footer><p>Generated from <code>{e(os.path.basename(cfg.paths.gold_set))}</code>
and the {e(cfg.name)} chunk store. Filters applied automatically: malformed questions,
answer leakage, OCR-garbled answers, degenerate multi-hops, question/table qualifier
mismatch, and an absence check on every unanswerable item. Approve, and the classic
six-config ablation runs on this set.</p></footer>""")
    P.append("</div>")

    doc = f"<title>Gold set spot-check &mdash; 500-page corpus</title>\n<style>{CSS}</style>\n" + "\n".join(P)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"wrote {args.out}  ({len(doc)/1024/1024:.2f} MB, {slot} crops embedded)")
    store.close()


if __name__ == "__main__":
    main()
