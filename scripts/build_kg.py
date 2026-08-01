"""Build the knowledge graph over the chunk store's text regions.

    python -m scripts.build_kg --config configs/scale500.yaml

Constrained extraction (closed schema) over every text/list/table chunk, into the
backend named by the config. Which corpus and which graph file both come from the
YAML, so a dev build and a scale build differ only by --config. A chunk that errors
is skipped and counted rather than aborting the run — this is the single most
expensive paid pass, so one bad chunk must not cost the whole spend.
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

from platform_core.graph.extraction import LLMTripleExtractor
from platform_core.graph.networkx_backend import NetworkXGraphBackend
from platform_core.graph.schema import PredicateSchema, load_predicates
from platform_core.llm.openai_client import OpenAIClient
from platform_core.stores.chunk_store import SQLiteChunkStore
from platform_core.types import RegionType

IN_PRICE, OUT_PRICE = 0.15 / 1e6, 0.60 / 1e6


R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
DEFAULT_CONFIG = R + "/configs/enhanced.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--pred", default=R + "/domain_packs/biomed/predicates.yaml")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from platform_core.config import AppConfig

    cfg = AppConfig.from_yaml(args.config)
    print(cfg.describe())
    out_path = cfg.paths.graph_store
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    store = SQLiteChunkStore(cfg.paths.chunk_db)
    defs = load_predicates(args.pred)
    schema = PredicateSchema.from_yaml(args.pred)
    chunks = [c for c in store.iter_chunks()
              if c.region_type.value in ("text", "list", "table") and c.text.strip()]
    if args.limit:
        chunks = chunks[:args.limit]

    llm = OpenAIClient(model=cfg.models.llm_model)
    ext = LLMTripleExtractor(llm, defs)
    graph = NetworkXGraphBackend()
    per_pred: collections.Counter = collections.Counter()
    failures = 0
    print(f"extracting over {len(chunks)} chunks; model={llm.model}")
    for i, c in enumerate(chunks, 1):
        try:
            edges = ext.extract(c, schema)
        except Exception as exc:
            failures += 1
            edges = []
            print(f"  ! {c.chunk_id}: {type(exc).__name__}")
        for e in edges:
            graph.add_edge(e)
            per_pred[e.predicate] += 1
        if i % 50 == 0 or i == len(chunks):
            print(f"  [{i}/{len(chunks)}] edges={len(graph)}")

    graph.save(out_path)
    with open(out_path + ".edges.jsonl", "w", encoding="utf-8") as f:
        for e in graph.all_edges():
            f.write(json.dumps({"subject": e.subject, "predicate": e.predicate,
                                "object": e.object, "chunk_id": e.chunk_id,
                                "evidence": e.evidence}, ensure_ascii=False) + "\n")

    # chunk_id -> entity nodes: the entry point for KG-neighbourhood retrieval
    chunk_nodes: dict = collections.defaultdict(set)
    for e in graph.all_edges():
        chunk_nodes[e.chunk_id].update((e.subject, e.object))
    with open(out_path + ".chunk_nodes.json", "w", encoding="utf-8") as f:
        json.dump({k: sorted(v) for k, v in chunk_nodes.items()}, f, ensure_ascii=False)

    # diverse sample: first edge of up to 5 distinct predicates, padded to 5
    sample, seen = [], set()
    for e in graph.all_edges():
        if e.predicate not in seen:
            sample.append(e); seen.add(e.predicate)
        if len(sample) == 5:
            break
    for e in graph.all_edges():
        if len(sample) >= 5:
            break
        if e not in sample:
            sample.append(e)

    u = llm.total_usage
    cost = u["prompt_tokens"] * IN_PRICE + u["completion_tokens"] * OUT_PRICE
    print(f"\n================ KNOWLEDGE GRAPH ({cfg.name}) ================")
    print(f"total edges: {len(graph)}   nodes: {graph.g.number_of_nodes()}   failures: {failures}")
    print("edges per predicate:")
    for p, n in per_pred.most_common():
        print(f"  {p:14s} {n}")
    print("5 sample edges:")
    for e in sample:
        print(f"  ({e.subject} -{e.predicate}-> {e.object})")
        print(f"     chunk_id={e.chunk_id}  evidence={e.evidence[:150]!r}")
    print(f"\nusage: {u}  cost: ${cost:.4f}")
    store.close()


if __name__ == "__main__":
    main()
