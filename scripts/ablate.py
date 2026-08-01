"""Entry point: run the ablation series over the gold set and write the tables.

    python -m scripts.ablate
    python -m scripts.ablate --config configs/enhanced_vqa.yaml

baseline -> +layout -> +clip -> +caption -> +kg -> +rerank. Writes reports/ablation.md.

Every setting comes from the YAML: paths, image encoder (and therefore which image
index is read), embedders, device, retrieval knobs, abstention threshold. The
ablation series itself overrides only the flag under test, so the config's own
`flags` block is not what varies here — everything else is.
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

from evaluation.ablation import SERIES, ablation_flags, render, run_ablation
from evaluation.gold_set import load_gold
from evaluation.harness import Harness
from platform_core.config import AppConfig
from platform_core.graph.networkx_backend import NetworkXGraphBackend
from platform_core.llm.embeddings import SentenceTransformerEmbedder, make_image_embedder
from platform_core.llm.openai_client import OpenAIClient
from platform_core.retrieval.rerank import CrossEncoderReranker
from platform_core.stores.chunk_store import SQLiteChunkStore
from platform_core.stores.vector_store import FaissVectorIndex

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
DEFAULT_CONFIG = R + "/configs/enhanced.yaml"


def _units(base: str) -> dict:
    d = {}
    with open(base + ".units.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d[r["unit_id"]] = r["source_chunk_ids"]
    return d


def build_harness(cfg: AppConfig) -> Harness:
    """Assemble the harness entirely from `cfg` — the only construction path."""
    store = SQLiteChunkStore(cfg.paths.chunk_db)
    ix = cfg.paths.index_dir
    indices = {
        # both text indices are loaded; the chunker flag picks one per config
        "text_layout": FaissVectorIndex.load(ix + "/text_layout"),
        "text_naive": FaissVectorIndex.load(ix + "/text_naive"),
        "image": FaissVectorIndex.load(cfg.image_index_path()),
    }
    unit_sources = {
        "text_layout": _units(ix + "/text_layout"),
        "text_naive": _units(ix + "/text_naive"),
    }
    # phase 9: present only after scripts/transcribe_tables.py has run
    if os.path.exists(ix + "/text_layout_tablevqa.faiss"):
        indices["text_layout_tablevqa"] = FaissVectorIndex.load(ix + "/text_layout_tablevqa")
        unit_sources["text_layout_tablevqa"] = _units(ix + "/text_layout_tablevqa")
    graph = NetworkXGraphBackend.load(cfg.paths.graph_store)
    with open(cfg.paths.graph_store + ".chunk_nodes.json", encoding="utf-8") as f:
        chunk_nodes = json.load(f)

    m, g = cfg.models, cfg.generation
    bge = SentenceTransformerEmbedder(m.text_embedding_model, m.device, m.embed_batch_size)
    img_enc = make_image_embedder(m.image_embedding_model, m.device, m.embed_batch_size)
    reranker = CrossEncoderReranker(m.reranker_model, m.device)
    return Harness(
        store, indices, unit_sources, bge, img_enc, reranker, graph, chunk_nodes,
        gen_llm=OpenAIClient(model=g.answer_model),
        verify_llm=OpenAIClient(model=g.verify_model),
        judge_llm=OpenAIClient(model=g.answer_model),
        vqa_llm=OpenAIClient(model=g.vqa_model),
        ret_params=cfg.retrieval, gen_params=g, reports_dir=cfg.paths.reports_dir,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--limit", type=int, default=None, help="first N gold questions (smoke)")
    ap.add_argument("--out", default=None, help="default: <reports_dir>/ablation.md")
    ap.add_argument("--resume", action="store_true",
                    help="reuse any perq_<config>.jsonl already complete in reports_dir, "
                         "so a run that died partway does not re-buy finished configs")
    ap.add_argument("--series", default="classic", choices=sorted(SERIES),
                    help="classic = the published 6 steps; phase9 adds "
                         "+tablevqa/+pagecrop/+vqa (see evaluation.ablation)")
    args = ap.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    print(cfg.describe())
    print("ablation series : " + " -> ".join(n for n, _ in ablation_flags(args.series)))
    harness = build_harness(cfg)
    gold = load_gold(cfg.paths.gold_set)
    reports = run_ablation(harness, gold, limit=args.limit, series=args.series,
                           resume=args.resume)
    md = render(reports)
    out = args.out or os.path.join(cfg.paths.reports_dir, "ablation.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)

    IN, OUT, MIN_IN, MIN_OUT = 2.5 / 1e6, 10 / 1e6, 0.15 / 1e6, 0.60 / 1e6
    gu, j, v = harness.gen_llm.total_usage, harness.judge_llm.total_usage, harness.verify_llm.total_usage
    cost = ((gu["prompt_tokens"] + j["prompt_tokens"]) * IN
            + (gu["completion_tokens"] + j["completion_tokens"]) * OUT
            + v["prompt_tokens"] * MIN_IN + v["completion_tokens"] * MIN_OUT)
    print(f"\nACTUAL API cost this invocation: ${cost:.2f}"
          + ("  (configs reloaded via --resume were not re-bought)" if args.resume else ""))


if __name__ == "__main__":
    main()
