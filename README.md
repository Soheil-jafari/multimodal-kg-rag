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
rather than flattering them. With 95% bootstrap intervals, R@5 goes 0.57 [0.49, 0.66] →
0.91 [0.86, 0.96] — non-overlapping.

**And what does *not* work is reported with the same weight.** The retrieval win is layout
chunking and caption anchoring, with reranking third. It is **not** image embeddings:
`+clip` contributes ΔR@5 of exactly +0.000 [+0.000, +0.000]. And on this corpus it is
**not** the knowledge graph — ΔR@5 +0.026 [+0.000, +0.059] and ΔCorrect +0.030 [+0.004,
+0.064], the latter's magnitude being precisely the measured run-to-run noise floor. The
diagnosis: the HuggingFace mirror this corpus came from drops the original filename, so
PMC article IDs are unavailable in this copy, page = paper, and every multi-hop question is
answerable within a single page — the regime where a graph helps least. A KG earns its cost when evidence is scattered across documents. That is a null
result about this corpus, not about GraphRAG, and the architecture is `paper_id`-agnostic
so a real multi-document corpus can be swapped in and the question asked properly.

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

**Page = paper.** The HuggingFace parquet mirror ingested here
(`jordanparker6/publaynet`) drops the original filename — `image.path` is
null on every row and `id` is a per-page COCO image id — so PMC article IDs
are unavailable in this copy and multi-page papers are not reconstructed.
This is a limitation of the mirror, not of PubLayNet: the official
distribution carries the article in the COCO `file_name` field
(`PMC<id>_<page>.jpg`), recoverable by joining the COCO `image_id` back to
`val.json`. `paper_id` and `page_id` are stored separately (equal today) so a
real multi-page grouper swaps in behind `PaperGrouper` with no schema change.
`multi_hop` questions are within-page by construction as a result.

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

**Python 3.11.** `torch==2.5.1` ships no wheels for 3.13.

### Verify the reported numbers (no API key, seconds)

Every per-question log behind every table is committed under
`reports/scale500/`, so the report can be checked without re-running anything:

```bash
python -m scripts.check_report_numbers
```

This walks every figure in `reports/FINAL_REPORT.md` back to the artifact it
came from, and names any figure it cannot trace to `reports/scale500/`.

### Re-run the pipeline (~2 h OCR, ~$5.8 API, `OPENAI_API_KEY` required)

```bash
python -m scripts.ingest              --config configs/scale500.yaml   # OCR -> chunk store
python -m scripts.build_indices       --config configs/scale500.yaml   # BGE + BiomedCLIP + captions
python -m scripts.build_kg            --config configs/scale500.yaml   # closed-schema extraction
python -m scripts.build_gold          --config configs/scale500.yaml --target 130   # optional — see note below
python -m scripts.ablate              --config configs/scale500.yaml --series classic --resume
```

`build_gold` is optional and LLM-driven, so it produces a *different*
evaluation set on every run. The gold set behind the reported results is
committed at `domain_packs/biomed/gold/gold_set_scale500.jsonl` — skip this
step unless you intend to build a new one.

`scripts.evaluate` runs a single config verbatim; `scripts.ablate` runs the series.
`--resume` reloads any per-question log that is already complete instead of re-paying for
it, which matters because the answer model sits against a 30,000 tokens/minute rate limit
for the duration of a full run.

To review the results interactively afterwards, see
[Human-in-the-loop review](#human-in-the-loop-review).

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

**Uncertainty is reported two ways, because they answer different questions.** Every
per-category and per-predicate row carries a 95% bootstrap interval over questions (10,000
resamples, `scripts/bootstrap_ci.py` → `reports/scale500/confidence_intervals.md`), which
is *sampling* uncertainty. Separately, runs are single-seed — a deliberate consequence of
the budget, since repeating a six-config ablation costs another ~$5 — so *run-to-run*
variance was measured directly instead of assumed: re-running an identical configuration
over the same 17 questions moved correctness by 0.029, and any effect at or below ±0.03 is
marked "within noise". Point estimates are recomputed from the per-question logs and
asserted equal to the published tables before any interval is emitted.

**Operational characteristics** are measured on the development machine rather than
projected (`scripts/measure_ops.py` → `reports/scale500/operational.md`): retrieval-only
query latency p50 3.67 s / p95 13.44 s on CPU, 14.6 s cold start, 2466 MB peak resident,
OCR ingest 14.7 s/page. End-to-end latency including the answer model and per-config API
cost are marked **not measured**, because the harness never recorded per-question
wall-clock or token counts and obtaining them would cost API budget.

## Human-in-the-loop review

A system that answers questions from scientific documents should not be trusted
unsupervised, so the oversight surface is part of the deliverable rather than an
afterthought. `scripts/review_app.py` is a Streamlit **review queue over the locked
results**: every answer the evaluation produced, sorted **worst-first**, so a reviewer
with limited time spends it where the system is least sure instead of reading from
`q001`.

```bash
python -m scripts.build_review_queue --config configs/scale500.yaml --step +rerank
streamlit run scripts/review_app.py -- --config configs/scale500.yaml
```

It runs locally and serves at **<http://localhost:8501>**; the queue it reads is committed
at `reports/scale500/review_queue.jsonl`, so the dashboard opens on the same results the
report was written from.

The queue is precomputed because scoring it needs a retrieval pass per question; the app
itself only reads. `--step` takes any column of the ablation series, so the same interface
reviews the baseline or any intermediate config under one definition of confidence.

**Confidence is `0.5·Rn + 0.5·G`** — normalised retrieval strength (the top query-context
cosine, the same value the abstention gate thresholds) and grounding (the fraction of
answer sentences supported by the chunk they cite). Both are **label-free**, and that is
the point. The obvious shortcut is to reuse the retrieval metrics already sitting in the
per-question logs, but recall@k and nDCG are computed against gold supporting chunks, so a
score built from them would rank items by how *right* they are — information a deployed
queue cannot have, and which would make the dashboard look far better than it is. The
weights are a flat 0.5/0.5 by choice; fitting them against the locked correctness column
would quietly turn the score back into the correctness proxy the definition exists to
avoid.

**Abstentions invert the priority.** There is no answer to grade, so they are ranked by
retrieval strength alone: abstaining on weak retrieval is usually correct and wastes a
reviewer's time, while abstaining on *strong* retrieval is the false-abstention failure
mode and is the most valuable thing a human can look at.

**Each card carries the evidence, not just the verdict** — the answer, the three signals
with the raw cosine, provenance counts showing how the context was reached, every cited
chunk with its source text pulled live from the store (flagged red if a cited id is not in
the store), the VQA gate's per-crop pass/block decisions, and the chain-of-thought trace
where one exists. **The gold answer and the automated judge's verdict are hidden behind a
toggle, off by default**, so the first judgement comes from the evidence rather than from
the expected string — revealing then audits both the reviewer and the judge.

**Verdicts close the loop.** Correct / incorrect / needs-review plus a free-text note
append to a log that records the confidence and abstention state *at review time*. That
last part is what makes the log useful later: it is the raw material for asking whether the
confidence score actually predicts human disagreement, which cannot be answered until
enough verdicts exist. The log is append-only, so a reviewer changing their mind adds a
line rather than overwriting one and the history survives.

The dashboard is **strictly read-only** over the locked results — it writes only the queue
sidecar and the feedback log, and no published metric can move because someone opened it.

As a check on whether the ranking means anything: ranking blind, with no access to gold,
the queue put three `figure` questions at the top of its worst-first order — and the
gold-backed metric independently reports figure decision-accuracy at 0.75, i.e. a quarter
of answerable figure questions wrongly abstained. A label-free score surfaced the same
defect the labelled metric measured.

*Honest limit:* confidence is a triage signal, not a correctness estimate. Grounding scores
each sentence against the whole cited chunk, so a short answer citing a long passage can
score zero while being correct — that happens in this queue, and it is why the score ranks
attention rather than claiming accuracy.

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
