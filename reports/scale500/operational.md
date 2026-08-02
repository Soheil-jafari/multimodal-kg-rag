# Operational characteristics

Measured on the development machine, not projected. Retrieval latency and memory were timed locally with no model API calls; ingest time and token spend come from the logs of the runs that produced the locked results. Anything neither locally measurable nor recorded is marked **not measured**.

Machine: Windows-10 · 6 physical / 12 logical cores · 15.7 GB RAM · Python 3.11.9 · **CPU only** (no CUDA build installed; the code is device-agnostic).

## Query latency — retrieval path, no LLM

Full configured retrieval per query: query embedding, text search, caption / page-crop / image / graph expansion, cross-encoder rerank. 127 gold questions, 3 discarded as warm-up.

| metric | ms |
|---|---:|
| p50 | 3665.7 |
| p95 | 13438.4 |
| p99 | 15300.6 |
| mean | 4794.5 |
| min / max | 759.5 / 15893.8 |

Cold start (load BGE + BiomedCLIP + cross-encoder + FAISS indices + graph): **14.6 s**, paid once per process.

## Memory

| stage | RSS |
|---|---:|
| interpreter baseline | 60 MB |
| after loading all models and indices | 2466 MB |
| **peak during retrieval** | **2466 MB** |

Three transformer models plus two FAISS indices and the graph are held concurrently. On a 4 GB GPU they would need sequencing; on CPU they coexist.

## End-to-end latency (including the answer model)

**not measured.** The evaluation harness records per-question answers and scores but never recorded per-question wall-clock, and re-running to obtain it would cost API budget. Retrieval latency above is a lower bound on the end-to-end figure; the answer and grounding-verification calls dominate it.

## Corpus build

| stage | measurement | source |
|---|---|---|
| OCR ingest, 490 pages | **119.7 min** (14.7 s/page, 5619 chunks) | run log |
| KG extraction, 4776 chunks | 3,876,391 tokens, **$0.6397** | run log |
| BGE text index (5,611 + 1,818 units) | not measured — wall-clock not recorded | — |
| BiomedCLIP image index (408 crops) | not measured — wall-clock not recorded | — |

Per-shard OCR: val00000 1670s · val00001 1424s · val00002 1495s · val00003 1359s · val00004 1233s.

OCR dominates corpus construction and is single-threaded CPU work; it is the one stage that would benefit most from parallelism or a GPU OCR backend.

## API cost

Per-config cost is **not measured**: the harness logs answers and scores per question but not token counts, and the ablation's cost line is emitted per invocation rather than per config. What was recorded:

| run | cost |
|---|---:|
| ablation (3 of 6 configs, resumed invocation) | $2.50 |
| chain-of-thought paired run | $0.55 |
| VQA-vs-text paired run | $0.13 |
| KG extraction over the corpus | $0.6397 |

The first ablation invocation crashed on a rate limit before printing its cost line, so the six-config total is known only to about **$5**; the three configs in the resumed invocation cost $2.50 together, i.e. roughly $0.83 per config at 127 questions. That per-config figure is a division of a measured total, not a separately measured quantity.

Throughput note: the answer model is capped at 30,000 tokens/minute on this account, and a ten-chunk answer sits against that cap for a full run — which is why the runner retries with backoff and can resume from a completed per-question log rather than re-buying it.

