"""Schema-generalisation check: re-run free predicate discovery on a larger corpus.

    python -m scripts.discover_predicates --config configs/scale500.yaml

Phase 3 defined the closed 8-predicate schema by consolidating a FREE extraction over
30 dev chunks. That schema was derived from 50 pages; the question this script answers
is whether it still covers a 10x corpus, or whether new relation TYPES appear that the
8 cannot express.

Sampling deliberately mirrors Phase 3 exactly — text/list/table chunks of >=200 chars,
strided across pages, n=30 — so the two predicate distributions are comparable. The
only intended difference is which corpus they are drawn from.

Two-step, because "does this free predicate fit the schema?" is a judgement call and
should be recorded rather than hidden:
  1. FREE extraction (no schema shown to the model) -> raw predicate frequency.
  2. Each DISTINCT discovered predicate is classified into one of the closed 8 or
     NONE, by the LLM, one cheap call for the whole list. The residual NONE set,
     with frequencies, is the candidate-new-type evidence a human then judges.

Writes <reports_dir>/predicate_discovery.json and prints the summary.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from platform_core.config import AppConfig
from platform_core.graph.extraction import free_extract
from platform_core.graph.schema import load_predicates
from platform_core.llm.openai_client import OpenAIClient
from platform_core.stores.chunk_store import SQLiteChunkStore

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
DEFAULT_CONFIG = R + "/configs/scale500.yaml"
IN_PRICE, OUT_PRICE = 0.15 / 1e6, 0.60 / 1e6  # gpt-4o-mini

MIN_CHARS = 200          # phase 3 sampling criteria, held fixed for comparability
SAMPLE_TYPES = ("text", "list", "table")

_MAP_SYSTEM = (
    "You classify relation predicates against a fixed, closed schema. "
    "You never invent schema members. Return strict JSON."
)
_MAP_USER = """Closed schema:
{PREDICATES}

For each discovered predicate below, say which ONE closed predicate it is a variant of,
or "NONE" if it expresses a relation the closed schema genuinely cannot express.

Guidance (the schema's intended mappings):
- higher/elevated/greater level, amount or risk of X -> increases
- lower/reduced/decreased level, amount or risk of X -> decreases
- prevalence of / observed in / found in a population, sample or site -> occurs_in
- used to treat / therapy for a condition -> treats
- converted / oxidized / degraded / metabolized into Y -> transforms_to
- quantified / assessed / estimated by a method or assay -> measured_by
- produces / leads to / results in -> causes
- suppresses / blocks / reduces activity of -> inhibits
Answer NONE only when no closed predicate captures the relation's MEANING — not merely
because the wording differs.

Discovered predicates:
{DISCOVERED}

Return ONLY JSON: {"mapping": {"<discovered>": "<closed name or NONE>", ...}}"""


def sample_chunks(store, n: int) -> list:
    """n chunks strided across the corpus — same criteria as the phase-3 sample."""
    cands = [c for c in store.iter_chunks()
             if c.region_type.value in SAMPLE_TYPES and len(c.text.strip()) >= MIN_CHARS]
    cands.sort(key=lambda c: c.chunk_id)
    if len(cands) <= n:
        return cands
    stride = len(cands) / n  # float stride so the sample spans the WHOLE corpus
    return [cands[int(i * stride)] for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--n", type=int, default=30, help="chunks to sample (phase 3 used 30)")
    ap.add_argument("--dev-db", default=R + "/artifacts/dev/chunks.sqlite",
                    help="dev store, used only to measure how fresh the sample is")
    args = ap.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    print(cfg.describe())
    store = SQLiteChunkStore(cfg.paths.chunk_db)
    defs = load_predicates(R + "/domain_packs/biomed/predicates.yaml")
    closed = [d["name"] for d in defs]

    chunks = sample_chunks(store, args.n)
    # freshness is measured, not assumed: the dev schema must not simply be re-derived
    # from the pages it was already derived from
    dev_pages: set = set()
    if os.path.exists(args.dev_db):
        dev = SQLiteChunkStore(args.dev_db)
        dev_pages = set(dev.page_ids())
        dev.close()
    n_fresh = sum(1 for c in chunks if c.page_id not in dev_pages)
    print(f"\nsampled {len(chunks)} chunks ({SAMPLE_TYPES}, >={MIN_CHARS} chars, strided); "
          f"{n_fresh}/{len(chunks)} from pages NOT in the 50-page dev corpus")

    llm = OpenAIClient(model="gpt-4o-mini")
    triples: list[dict] = []
    for i, c in enumerate(chunks, 1):
        try:
            for t in free_extract(llm, c):
                t["_chunk_id"] = c.chunk_id
                triples.append(t)
        except Exception as exc:
            print(f"  ! {c.chunk_id}: {type(exc).__name__}")
        if i % 10 == 0 or i == len(chunks):
            print(f"  [{i}/{len(chunks)}] triples={len(triples)}")

    freq = collections.Counter(
        str(t.get("predicate", "")).strip().lower().replace(" ", "_")
        for t in triples if str(t.get("predicate", "")).strip()
    )
    distinct = sorted(freq)

    mapping: dict = {}
    if distinct:
        block = "\n".join(f"- {d['name']}: {d.get('description', '')}" for d in defs)
        out = llm.complete_json(
            _MAP_SYSTEM,
            _MAP_USER.replace("{PREDICATES}", block)
                     .replace("{DISCOVERED}", "\n".join(f"- {p} (x{freq[p]})" for p in distinct)),
        )
        mapping = out.get("mapping", {}) if isinstance(out, dict) else {}
    # anything the mapper skipped or named off-schema counts as unmapped, not as a pass
    norm = {p: (mapping.get(p) if mapping.get(p) in closed else "NONE") for p in distinct}

    mapped_types = [p for p in distinct if norm[p] != "NONE"]
    unmapped = [p for p in distinct if norm[p] == "NONE"]
    tok_mapped = sum(freq[p] for p in mapped_types)
    tok_unmapped = sum(freq[p] for p in unmapped)
    per_closed = collections.Counter()
    for p in mapped_types:
        per_closed[norm[p]] += freq[p]

    u = llm.total_usage
    cost = u["prompt_tokens"] * IN_PRICE + u["completion_tokens"] * OUT_PRICE
    report = {
        "config": cfg.name, "n_chunks": len(chunks), "n_fresh_pages": n_fresh,
        "n_triples": len(triples), "n_distinct_predicates": len(distinct),
        "frequency": dict(freq.most_common()),
        "mapping": norm,
        "coverage_by_triples": round(tok_mapped / len(triples), 3) if triples else None,
        "coverage_by_distinct": round(len(mapped_types) / len(distinct), 3) if distinct else None,
        "per_closed_predicate": dict(per_closed.most_common()),
        "unmapped": {p: freq[p] for p in sorted(unmapped, key=lambda x: -freq[x])},
        "usage": u, "cost_usd": round(cost, 4),
    }
    os.makedirs(cfg.paths.reports_dir, exist_ok=True)
    out_path = os.path.join(cfg.paths.reports_dir, "predicate_discovery.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n================ PREDICATE DISCOVERY (schema generalisation) ================")
    print(f"{len(triples)} triples, {len(distinct)} distinct predicates, "
          f"{len(chunks)} chunks, ${cost:.4f}")
    print("\nfrequency head (free extraction, no schema shown):")
    for p, k in freq.most_common(15):
        print(f"  {p:32s} x{k:<3d} -> {norm[p]}")
    print(f"\nCOVERAGE BY THE CLOSED 8:")
    print(f"  by triples : {tok_mapped}/{len(triples)} "
          f"({100*tok_mapped/len(triples):.0f}%)" if triples else "  no triples")
    print(f"  by distinct: {len(mapped_types)}/{len(distinct)} "
          f"({100*len(mapped_types)/len(distinct):.0f}%)" if distinct else "")
    print("\ntriples landing on each closed predicate:")
    for p in closed:
        print(f"  {p:14s} {per_closed.get(p, 0)}")
    print(f"\nUNMAPPED — candidate new relation TYPES ({len(unmapped)} distinct, "
          f"{tok_unmapped} triples):")
    for p in sorted(unmapped, key=lambda x: -freq[x])[:20]:
        ex = next((t for t in triples
                   if str(t.get("predicate", "")).strip().lower().replace(" ", "_") == p), {})
        print(f"  {p:32s} x{freq[p]:<3d} e.g. ({ex.get('subject', '?')} -> {ex.get('object', '?')})")
    print(f"\nwrote {out_path}")
    store.close()


if __name__ == "__main__":
    main()
