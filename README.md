# Multimodal + Knowledge-Graph Document QA over Scientific Papers

A document-QA system over scientific paper pages that measures, rather than asserts,
whether multimodal and knowledge-graph retrieval beat a text-only RAG baseline.

The whole project is built around one constraint: **baseline and enhanced are the same
code path, separated only by a set of flags.** There is no `if baseline:` anywhere. That
is what makes the ablation trustworthy — a difference in the numbers can only come from a
flag, never from a divergent code path.

**Headline result** (500 pages, 127 human-verified questions — full tables in
[`reports/scale500/RESULTS.md`](reports/scale500/RESULTS.md)):

| | baseline | enhanced | |
|---|---|---|---|
| Recall@5 | 0.57 | **0.91** | +60% |
| nDCG | 0.47 | **0.89** | +89% |
| Correctness | 0.59 | 0.69 | +17% |

The enhancement gap **widens** with corpus size: on a 50-page corpus the same pipeline
scored 0.69 → 0.87 (+26%). Baseline retrieval degrades as the corpus grows while the
enhanced configuration improves, so the small-corpus numbers understated the enhancements
rather than flattering them.

---

## Architecture

**Single source of truth.** A canonical SQLite chunk store holds one row per labelled page
region. The text index, image index and knowledge graph are all *derived* from it and join
back by `chunk_id`. Nothing downstream keeps a second copy of the text.

**One flag-driven retriever.** `FlagDrivenRetriever.retrieve()` is the only retrieval path.
Dense BGE text search always runs — that block alone *is* the baseline. Each enhancement is
an independently gated block that only *adds* to a candidate pool keyed by `chunk_id`, and
every hit carries comma-joined provenance (`text`, `caption-link`, `page-crop`,
`clip-image`, `graph-hop`).

**Everything swappable behind an ABC.** `LLMClient`, `GraphBackend`, `VectorIndex`,
`ChunkStore`, `RegionOCR`, embedders. Selected in YAML, constructed in one place.

**Config-driven for real.** `AppConfig.from_yaml` resolves `extends:`, deep-merges so a
child overrides only the keys it names, makes paths absolute, and then *validates
strictly* — an unknown or misspelled key is an error, not a silently ignored line. Editing
a YAML value changes behaviour with no code change, and `describe()` prints the resolved
settings at the top of every run so a report cannot disagree with what executed.

```
platform_core/   ingestion, stores, graph, llm, retrieval, generation (+ config, types)
domain_packs/    biomed/ — closed predicate schema, gold sets
evaluation/      metrics, gold-set loader, harness, ablation runner
configs/         default -> baseline / enhanced / enhanced_vqa / scale500 / …
scripts/         entry points: ingest -> build -> evaluate -> ablate -> review
reports/         results tables, per-question logs, review artifacts
tests/           fast offline unit tests
```

## Dataset

PubLayNet, read directly from its parquet shards with `pyarrow`. It provides **bounding
boxes and region labels** (text / title / list / table / figure) but **no text**, so
ingestion OCRs every labelled region crop.

**Page = paper.** The parquet carries no document identifier — `image.path` is null on
every row and `id` is a per-page COCO image id — so multi-page papers cannot be
reconstructed and are not faked. `paper_id` and `page_id` are stored separately (equal
today) so a real grouper can be swapped in with no schema change, and `multi_hop`
questions are within-page by construction. This is a limitation of the source data, and it
is recorded as one.

OCR is **RapidOCR** (`rapidocr-onnxruntime`) behind the `RegionOCR` interface — pure pip,
CPU, no system binary required.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then set OPENAI_API_KEY
```

Everything in this project was produced on **CPU**; the code is device-agnostic and
`models.device: cuda` works once a CUDA build of torch is installed.

Two environment notes, both learned the hard way:

- `torchvision` must match `torch` (this project pins `torch 2.5.1` with
  `torchvision 0.20.1`). A mismatched pair fails with
  `operator torchvision::nms does not exist`.
- faiss (MKL) and timm/torchvision (LLVM) link different OpenMP runtimes, so
  `KMP_DUPLICATE_LIB_OK=TRUE` is required to run them in one process. That flag can mask
  genuine problems, so it was verified rather than trusted: crop scores computed inside a
  faiss-importing process match a faiss-free computation exactly.

## Running

Every entry point takes `--config`; which corpus is processed, which indices are read and
which models are used all come from the YAML.

```bash
python -m scripts.ingest              --config configs/scale500.yaml   # OCR -> chunk store
python -m scripts.build_indices       --config configs/scale500.yaml   # BGE + BiomedCLIP + captions
python -m scripts.build_kg            --config configs/scale500.yaml   # closed-schema extraction
python -m scripts.build_gold          --config configs/scale500.yaml --target 130
python -m scripts.ablate              --config configs/scale500.yaml --series classic --resume
```

`scripts.evaluate` runs a single config verbatim; `scripts.ablate` runs the series.
`--resume` reloads any per-question log that is already complete instead of re-paying for
it, which matters because the answer model sits against a 30,000 tokens/minute rate limit
for the duration of a full run.

Review the results interactively:

```bash
python -m scripts.build_review_queue --config configs/scale500.yaml --step +rerank
streamlit run scripts/review_app.py -- --config configs/scale500.yaml
```

## Configs

`baseline.yaml` and `enhanced.yaml` differ **only** in their `flags` block, and a test
enforces that. `scale500.yaml` differs from the dev configuration only in which pages the
corpus is and where its artifacts live, plus one documented flag — also enforced by a test.

| flag | effect |
|---|---|
| `use_layout_chunking` | layout-aware region units instead of naive fixed windows |
| `use_caption_anchor` | retrieve a crop through its caption — the primary figure route |
| `use_page_crop_expansion` | reach a crop through the page whose prose matched |
| `use_clip_images` | BiomedCLIP image similarity (weak secondary signal) |
| `use_kg` | expand retrieval by walking the knowledge graph |
| `use_rerank` | cross-encoder rerank of the merged pool |
| `use_vqa` | read values out of a retrieved crop with a vision model |
| `use_cot` | reason step-by-step before answering; captures the trace |
| `allow_abstain` | permit "insufficient evidence in the corpus" |

Abstention is a *measured capability*, not an ablated enhancement, so it is on for both
baseline and enhanced.

## Knowledge graph

A **closed** schema of exactly 8 predicates in `domain_packs/biomed/predicates.yaml`:
`treats`, `causes`, `inhibits`, `increases`, `decreases`, `transforms_to`, `occurs_in`,
`measured_by`. The extractor is prompted with the list and the schema is then enforced **in
code** — a triple whose predicate falls outside the set is dropped, not trusted. Every edge
records its source `chunk_id` and the verbatim evidence sentence, so any graph-derived
answer traces back to text.

The schema was consolidated from a free extraction over the corpus rather than assumed, and
its coverage was re-measured on a ten-times-larger corpus: it captures 22% of freely
extracted triples there. That result is reported rather than patched — see
[`reports/scale500/RESULTS.md`](reports/scale500/RESULTS.md) §5.1.

## Evaluation

**Gold set.** Questions are *selected* from the knowledge graph — the answer and the
supporting `chunk_id`s come from an edge, and the language model only phrases the selected
fact. It never supplies an answer or evidence. Coverage is driven across six categories
(`single_fact`, `multi_hop`, `figure`, `figure_value`, `text_derived`, `unanswerable`) and
all 8 predicates, plus deliberately unanswerable items to measure abstention and a
text-derived bias guard written from passages with the graph never consulted.

`figure_value` answers are **verified by hand against the table crop images**, because
every text route to a table value is unreliable: OCR has the right digits in the wrong
structure, a vision transcription has the right structure with some wrong digits, and using
the vision model to author its own answer key would make any result circular.

**Metrics.** Recall@k, Precision@k, MRR and nDCG against gold supporting chunks;
correctness by rubric judge; faithfulness as the grounded-sentence fraction; and decision
accuracy for the abstain/answer choice. Everything is broken out per category and per
predicate, with low-sample groups flagged rather than quietly averaged in.

## Reproducibility and limitations

`artifacts/EXPERIMENT_LOG.md` is the dated, append-only record of the whole build — what
was tried, what the numbers were, and what failed. Results are in `reports/`, with
per-question logs alongside every table.

The known limitations are catalogued in
[`reports/scale500/RESULTS.md`](reports/scale500/RESULTS.md) §5 with a diagnosis for each,
including: the closed schema does not generalise to a broader corpus; table-OCR-derived
values were wrong 53% of the time under human verification; abstention degrades as the
corpus grows; `occurs_in` over-applies and `treats` shows direction inversions; reranking
improves multi-hop retrieval while hurting multi-hop answering; and the VQA crop gate
weakens as more pages compete. The answer model is also not deterministic at temperature 0,
so single-question before/afters are evidence about a mechanism, never reproducible
fixtures — the run-to-run noise floor was measured directly at ±0.03 rather than assumed.
