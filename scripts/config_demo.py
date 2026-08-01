"""Entry point: prove the pipeline is config-driven — no API calls, no code edits.

    python -m scripts.config_demo --config configs/enhanced.yaml

Loads a YAML through AppConfig.from_yaml, prints the RESOLVED settings, builds the
retriever from that config alone, and runs fixed probe queries. Run it, change one
value in the YAML, run it again: the output must differ. That diff is the evidence
that config drives behaviour.

Retrieval only — the generator needs an API key and costs money, so this stays free
to run repeatedly. `scripts/evaluate.py` is the paid end-to-end path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from platform_core.config import AppConfig
from platform_core.graph.networkx_backend import NetworkXGraphBackend
from platform_core.llm.embeddings import SentenceTransformerEmbedder, make_image_embedder
from platform_core.retrieval.rerank import CrossEncoderReranker
from platform_core.retrieval.retriever import FlagDrivenRetriever
from platform_core.stores.chunk_store import SQLiteChunkStore
from platform_core.stores.vector_store import FaissVectorIndex

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")

PROBES = [
    "How did SARS spread geographically over time?",
    "What was the mean age in years of the study cohort?",
    "What spatial clusters of SARS patients were identified in June 2003?",
]


def _units(base: str) -> dict:
    d = {}
    with open(base + ".units.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d[r["unit_id"]] = r["source_chunk_ids"]
    return d


def build_retriever(cfg: AppConfig) -> FlagDrivenRetriever:
    """Every argument below is derived from cfg — nothing hard-coded."""
    m = cfg.models
    store = SQLiteChunkStore(cfg.paths.chunk_db)
    text_index = FaissVectorIndex.load(cfg.text_index_path())
    unit_sources = _units(cfg.text_index_path())
    bge = SentenceTransformerEmbedder(m.text_embedding_model, m.device, m.embed_batch_size)

    image_index = img_enc = None
    if cfg.flags.use_clip_images:  # only pay to load the encoder if the flag needs it
        image_index = FaissVectorIndex.load(cfg.image_index_path())
        img_enc = make_image_embedder(m.image_embedding_model, m.device, m.embed_batch_size)

    graph = chunk_nodes = None
    if cfg.flags.use_kg:
        graph = NetworkXGraphBackend.load(cfg.paths.graph_store)
        with open(cfg.paths.graph_store + ".chunk_nodes.json", encoding="utf-8") as f:
            chunk_nodes = json.load(f)

    reranker = CrossEncoderReranker(m.reranker_model, m.device) if cfg.flags.use_rerank else None

    return FlagDrivenRetriever(cfg.flags, cfg.retrieval, store, text_index, bge,
                               image_index, img_enc, graph, chunk_nodes, reranker,
                               unit_sources), store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=R + "/configs/enhanced.yaml")
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    print("=" * 70)
    print(cfg.describe())
    print("=" * 70)

    retriever, store = build_retriever(cfg)
    for q in PROBES:
        results = retriever.retrieve(q)
        print(f"\nQ: {q}")
        print(f"   pool provenance: {retriever.last_trace.get('sources', {})}")
        for i, r in enumerate(results[:args.show], 1):
            print(f"   {i}. {r.score:7.4f}  {r.chunk.chunk_id:22s} "
                  f"[{r.chunk.region_type.value:6s}] via {r.source}")
    store.close()


if __name__ == "__main__":
    main()
