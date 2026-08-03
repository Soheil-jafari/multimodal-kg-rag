# Multimodal + Knowledge-Graph Document QA over Scientific Papers

Does adding multimodal retrieval and a knowledge graph to a RAG pipeline actually beat a
text-only baseline? This measures it on 500 scientific paper pages and 127 human-verified
questions, and reports where it does not.

## 1. Problem & Approach

PubLayNet supplies region boxes and labels for scientific paper pages but no text, so every
region is OCR'd from its crop. The HuggingFace parquet mirror used here drops the original
filename, so PMC article IDs are unavailable in this copy and page = paper. The official
PubLayNet distribution encodes the article in the COCO `file_name` field
(`PMC<id>_<page>.jpg`), recoverable by joining the COCO `image_id` back to `val.json`.
`page_id` and `paper_id` are stored separately so a real multi-page grouper swaps in behind
`PaperGrouper` with no schema change — the KG null result below is therefore about this
corpus copy, not about GraphRAG.

The design constraint is the fairness guarantee: **baseline and enhanced are one code path
separated only by flags.** Dense BGE text retrieval always runs and that block alone *is*
the baseline; each enhancement is an independently gated block that only adds to a
candidate pool keyed by `chunk_id`. There is no `if baseline:` anywhere, so a difference in
the numbers can only come from a flag, never from a divergent code path. A test asserts
that the baseline and enhanced configs differ in nothing but their `flags` block, and that
the 500-page config differs from the 50-page one only in which pages it ingests and where
artifacts land.

## 2. System

![PubLayNet pages of region crops are OCR'd into a SQLite chunk store, the single source of truth keyed by chunk_id. Three indices are derived from it and joined back by chunk_id: a BGE text index in layout or naive form, a BiomedCLIP image index over crops and captions, and a knowledge graph of 8 predicates on NetworkX. All three feed the flag-driven retriever, a single retrieve() path combining text, caption-link, page-crop, clip-image and graph-hop routes plus a cross-encoder rerank, which passes to grounded generation with inline citations, abstention and a per-sentence grounding check.](figures/architecture.svg)

*Every arrow is config-selected: `AppConfig.from_yaml` resolves `extends:`, deep-merges so a
child overrides only the keys it names, and validates strictly — an unknown or misspelled
key is an error, not a silently ignored line; `describe()` prints the resolved settings at
the top of every run.*

So a report cannot disagree with what executed. Every external dependency sits behind an
interface (`LLMClient`, `GraphBackend`, `VectorIndex`, `ChunkStore`, `RegionOCR`), and
swapping the image encoder is a one-string change that also selects the index it reads.

Resolved for the reported run: BGE-base-en-v1.5 text, BiomedCLIP image, ms-marco-MiniLM
cross-encoder, gpt-4o answering with gpt-4o-mini verification, `top_k`/`rerank_k` 10/10,
`kg_hops` 2, `abstain_min_score` 0.45, CPU.

## 3. Evaluation & Results

**Gold set.** Every question is *selected* from a knowledge-graph edge — the answer and its
supporting `chunk_id`s come from the edge, and the language model only phrases the selected
fact. It never supplies an answer or evidence. Coverage is driven across six categories and
all eight predicates. Three integrity measures matter:

- **Text-derived bias guard** — 23 questions written from a passage with the graph never
  consulted. If the graph-derived categories scored higher, the set would flatter the
  pipeline that built it. They score highest at baseline (0.83), which is the right
  direction.
- **Unanswerable items are verified absent**, not assumed: a candidate whose top cosine
  against the corpus clears 0.62 is rejected. At 500 pages this rejected **13 of 24**
  candidates that had been safely absent at 50 pages — without the check, those would have
  been silently answerable and the abstention result would have meant nothing.
- **`figure_value` answers are human-verified against the crop images.** Using the vision
  model to author the answer key for the category that measures the vision model would make
  any result circular.

127 questions: single_fact 56, text_derived 23, multi_hop 17, figure 12, unanswerable 11,
figure_value 8. No category carries a low-sample flag — a first for this project.

**Headline: the enhancement gap widens with corpus size.**

| | 50 pages (n=55) | | 500 pages (n=127) | |
|---|---|---|---|---|
| | baseline | enhanced | baseline | enhanced |
| Recall@5 | 0.69 | 0.87 | **0.57** | **0.91** |
| MRR | 0.46 | 0.83 | 0.41 | 0.88 |
| nDCG | 0.52 | 0.84 | 0.47 | 0.89 |
| Correctness | 0.66 | 0.70 | 0.59 | 0.69 |
| Faithfulness | 0.73 | 0.78 | 0.82 | 0.82 |

Relative retrieval gain: **R@5 +26% → +60%; nDCG +62% → +89%.** The two ends move in
opposite directions, and that is the result. Baseline degrades as the corpus grows — naive
fixed-window chunking over ten times the candidates surfaces the right region less often.
The enhanced configuration does not degrade; it improves. The usual pattern is the reverse,
with small-corpus gains diluting as the corpus grows; here the 50-page numbers *understated*
the enhancements.

**Uncertainty, stated once.** 95% bootstrap intervals over questions (10,000 resamples).
Baseline R@5 **0.57 [0.49, 0.66]** → enhanced **0.91 [0.86, 0.96]**, non-overlapping.
Correctness 0.59 [0.51, 0.67] → 0.69 [0.61, 0.77] overlaps marginally, but the *paired*
difference is **+0.099 [+0.034, +0.168]**, excluding zero. Runs are single-seed by budget;
run-to-run variance was measured rather than assumed — an identical configuration re-run
over the same 17 questions moved correctness by **0.029**, so anything at or below ±0.03 is
indistinguishable from re-running the same system.

**Where the win comes from — before the table, not under it.** It is layout chunking and
caption anchoring, with reranking third. It is **not** image embeddings: `+clip`'s ΔR@5 is
**+0.000 [+0.000, +0.000]**, exactly nothing. And on this corpus it is **not** the
knowledge graph: ΔR@5 +0.026 [+0.000, +0.059] touches zero, and ΔCorrect +0.030 [+0.004,
+0.064] has a magnitude that *is* the measured noise floor. The diagnosis matters more than
the number — page = paper here, so every `multi_hop` question is answerable within one
page, the regime where a graph helps least because both hops already sit in the same
neighbourhood. A KG earns its cost when evidence is scattered across documents, which this
corpus copy cannot supply. A null result about this corpus, not about GraphRAG.

**Per-flag attribution (500 pages).** One flag added at a time from a pure-text baseline.

| config | R@5 | MRR | nDCG | Correct | Decis |
|---|---|---|---|---|---|
| baseline | 0.57 | 0.41 | 0.47 | 0.59 | 0.84 |
| +layout | 0.76 | 0.78 | 0.75 | 0.66 | 0.88 |
| +clip | 0.76 | 0.78 | 0.75 | 0.66 | 0.90 |
| +caption | 0.83 | 0.79 | 0.80 | 0.66 | 0.87 |
| +kg | 0.86 | 0.80 | 0.80 | 0.69 | 0.88 |
| +rerank | **0.91** | **0.88** | **0.89** | **0.69** | 0.89 |

Layout-aware chunking is the single biggest lever and alone recovers the entire baseline
degradation. `+caption` is the figure route (figure R@5 0.50→1.00). `+kg`'s apparent
correctness gain sits at the measured noise floor (above), and the feared dilution of
`single_fact` by graph noise has now failed to appear on two corpora. `+rerank` is the
largest retrieval step, a bigger contribution than at 50 pages — consistent with the rest,
since reranking matters more when the pool is deeper. **`+clip` contributes nothing measurable**: identical retrieval to
`+layout` on every metric.

**`figure_value` and the visual path.** This category was flat at 0.50 correctness across
*every* config at 50 pages. At 500 pages with human-verified values it responds: **0.38 →
0.75 correctness, MRR 0.28 → 0.83** — though at n=8 those are [0.12, 0.75] → [0.38, 1.00]
and overlap heavily, so the mechanism below is the evidence, not the jump. In a paired
comparison where retrieval runs once and
the answer is generated twice, the gold crop was retrieved **8/8** — but VQA did not beat
text (strict 4/8 both; point-estimate 6/8 text vs 5/8 VQA). Exactly two questions differ
and they cancel: on one, vision recovers a standard deviation that OCR linearisation lost;
on the other, vision misreads a cell in the *correct* crop and cites it properly.

**Operational, measured on the development machine** (6 cores, 15.7 GB, CPU only; no API
calls in the measurement). Retrieval-only query latency **p50 3.67 s, p95 13.44 s**, cold
start 14.6 s, peak RSS **2466 MB** with three transformer models and two FAISS indices
resident. OCR ingest 119.7 min for 490 pages (14.7 s/page) and dominates corpus build; KG
extraction 3876391 tokens for $0.64. End-to-end latency including the answer model, and
per-config API cost, are **not measured** — the harness never recorded per-question
wall-clock or token counts, and obtaining them would cost budget. Total API cost for the
full sequence: ~$5.8.

## 4. Findings & Limitations

**Related work.** These findings connect to established results. The compressed CLIP
similarity distribution is the modality gap (Liang et al., 2022), known to widen under
domain shift — which is why a domain encoder (BiomedCLIP) helps the score distribution;
consistent with cross-modal alignment work, it does not by itself close the retrieval gap.
Reaching a table crop through the prose of its page extends contextual retrieval
(Anthropic, 2024), which reduces retrieval failures by prepending document context before
embedding, to the cross-modal case: a crop, unretrievable by its own OCR, inherits
retrievability from the text that contextualises it. The knowledge-graph component follows
the GraphRAG paradigm (Edge et al., 2024); the coverage limitation observed at scale — a
closed schema capturing only a fraction of freely-extracted relations on a topically
broader corpus — is the documented schema-rigidity trade-off of predefined-schema GraphRAG.
The retrieval-generation divergence, where reranking lifts multi-hop retrieval while
depressing its correctness, is consistent with recent work arguing that multi-hop reasoning
needs hop-separated rather than collapsed representations (Liu et al., 2025).
Grounding and abstention follow the faithfulness objective of retrieval-augmented
generation (Lewis et al., 2020). To our knowledge the multimodal-safety observation — that
unconditional VQA converts a safe abstention into a confident, well-cited error unless
gated on provenance — is not directly addressed in the existing literature, and we offer it
as a contribution.

**The CLIP modality gap is real and only half-fixable.** General CLIP compressed 57
scientific crops into a 0.06-wide similarity band and separated rank 1 from rank 2 by
0.0003 — a coin toss (50-page corpus). BiomedCLIP widens the spread 6.7× and the top-1
margin 52×, and doubles image recall@1 (0.22→0.44). *Diagnosis:* that fixes the score *distribution*, not
the retrieval *win* — recall@5 was unchanged, and at 500 pages `+clip` adds nothing on top
of layout chunking. The caption anchor, not image similarity, is what makes figures
retrievable.

**A table cannot be retrieved by its own content.** Transcribing tables and indexing the
clean text made retrieval *worse* (gold table rank 31→40 and 19→36; cosine 0.536→0.459).
*Diagnosis:* a table states values, not the concepts that make those values relevant —
"lung cancer", "PAH" live in the page's prose, which outscores any possible table text. The
fix is to reach the crop through its page (`use_page_crop_expansion`), which took gold-crop
retrieval from 2/4 to 4/4 and holds at 8/8 on the larger corpus. Transcription was
therefore **tested and excluded** as contributing nothing to retrieval.

**Closed-schema GraphRAG trades coverage for tractability.** Re-running schema discovery on
the larger corpus: the closed 8 predicates cover only **22% of freely-extracted triples**.
Two missing types recur — a predictor/association relation, and a "has measured value"
relation. *Diagnosis:* the schema was consolidated from a toxicology/epidemiology slice and
the 500-page corpus spans robotics, dentistry and policy editorials. A closed schema buys
sparsity and precision (max entity degree 7, no hub explosion); an open one produced 46
distinct predicates over 30 chunks and never reused an edge type — a graph that cannot be
traversed. The schema was kept unchanged and the gap reported. **This is also the
structural reason `figure_value` finds no graph edges**: with no value-relation, `(mean
age) —has_value→ (26.0 (8.6))` cannot be represented at all, which is a schema limit, not
just an OCR one.

**Breadth helps multi-hop and dilutes single-fact — twice, independently.** The knowledge
graph appears to lift multi-hop correctness, though the effect is at the noise floor, while
adding context that a single-span question does not need. Chain-of-thought reproduces the
same shape from the generation side: multi_hop 0.38 →
0.41 but single_fact (the control) 0.88 → **0.80**, with abstention rising in both subsets
(0.24→0.29, 0.10→0.15). *Diagnosis:* reasoning surveys the retrieved set and folds in more
that is topically related; on a question with one correct span, a broader answer is a worse
one. **The multi-hop "gain" is not a result** — re-running the identical configuration
moved the same 17 questions by +0.029, so the measured noise floor equals the entire
effect. CoT is retained as an explainability feature, not an accuracy one.

**Retrieval and answering come apart.** Reranking lifts multi-hop R@5 0.62→0.82 while
*dropping* its correctness 0.44→0.35. *Diagnosis:* the cross-encoder scores each chunk
independently for query relevance, which is not the objective of assembling two chunks that
must be combined. More starkly: figure R@5 goes 0.38→1.00 across the series while figure
correctness goes 0.29→**0.25**. Solving retrieval did not solve answering, and reporting
only the retrieval metric would have hidden that.

**Attaching an image can convert a safe abstention into a confident error.** Handed an
off-topic crop, the vision model reads a real number off it and cites it. A score threshold
provably cannot gate this: the wrong crop out-scored the right one (0.5534 vs 0.4589), and
the inversion reproduces on raw OCR (0.5492 vs 0.5360) — both 50-page corpus. *Diagnosis:* a table's relevance is
established by the page that discusses it, not by its own tokens — so the gate keys on
**provenance**, with score demoted to a floor. *Limitation:* the guarantee weakens with
corpus size, since with `top_k`=10 up to ten different papers can clear "the crop's page had
a text hit", and one observed failure read the wrong paper's table.

**Table values read out of OCR are wrong more often than right.** Of 17 candidate values
put to human verification against the crop images, **9 were rejected (53%; 8 of 14 among
the OCR-derived proposals)** — misplacement, dropped exponents, label drift. *Diagnosis:*
OCR has the right digits in the wrong structure and a vision transcription the right
structure with some wrong digits, so neither may quote a value; the transcription is used
only to find and score a table.

**Abstention degrades at scale.** Unanswerable abstain-rate falls from 1.00 at every config
on 50 pages to 0.82 (0.91 at `+rerank`). *Diagnosis:* a larger corpus supplies more
near-miss context, so a question the corpus cannot answer still retrieves material that
looks answerable — and this is measured against a deliberately harder set, since the
absence check had already removed the easy cases.

**Extraction precision limits the graph.** `occurs_in` correctness collapses to 0.31 (n=16)
with retrieval fine at 0.81, and `treats` sits at 0.60 (n=15). *Diagnosis:* loose
subject/object typing and observed direction inversions (`patients —treats→ chemotherapy`).
Both are first candidates for an object-type constraint. Other standing limits: page =
paper; title OCR 20.4% empty at scale; 76 of 410 crops have no caption region at all; the
answer model is non-deterministic at temperature 0, so single-question comparisons are
evidence about a mechanism, never fixtures.

## 5. Human-in-the-Loop & Explainability

Oversight is designed in layers rather than bolted on, because each layer catches a
different failure. **Provenance** — every retrieved chunk carries how it was reached
(`text`, `caption-link`, `page-crop`, `clip-image`, `graph-hop`), and every graph edge
carries its source `chunk_id` and verbatim evidence sentence, so a graph-derived answer
traces back to text. **Citation** — answers cite inline, validated against the context
actually supplied. **Grounding** — each answer sentence is checked against its cited chunk,
with an LLM verifier deciding only the borderline band. **Abstention** — a cheap code gate
plus a prompt clause; the code gate is deliberately conservative and the semantic decision
lives in the prompt.

The **review dashboard** turns those signals into a worst-first queue. Confidence is
`0.5·Rn + 0.5·G` from retrieval strength and grounding — deliberately **label-free**, since
building it from the gold-based retrieval metrics already in the logs would rank items by
how right they are, which a deployed queue cannot know. Abstentions invert the priority: an
abstention on weak retrieval is probably correct, while one on *strong* retrieval is the
false-abstention failure mode and is what a reviewer should see first. Ranking blind, the
queue surfaced the same defect the gold-backed metric measured — its top three items are
figure questions, and figure decision-accuracy is 0.75.

Each item shows the answer, the three signals, provenance counts, every cited chunk with
its source text pulled live from the store, the VQA gate's per-crop decisions, and the
chain-of-thought trace where one exists. **The gold answer and the automated judge's
verdict are hidden behind a toggle**, so the first judgement comes from the evidence rather
than from the expected string — and revealing then audits both the reviewer and the judge.
Verdicts (correct / incorrect / needs-review, plus a note) append to a log that records the
confidence at review time, which is the raw material for asking whether confidence predicts
human disagreement. The dashboard is strictly read-only over the locked results.

*Honest limit:* confidence is not a correctness estimate. Grounding scores a sentence
against the whole cited chunk, so a short answer citing a long passage can score zero while
being correct — observed in the queue, and the reason confidence is presented as a
triage signal only.

**References.** Liang, W., Zhang, Y., Kwon, Y., Yeung, S., Zou, J. *Mind the Gap:
Understanding the Modality Gap in Multi-modal Contrastive Representation Learning.* NeurIPS
2022. · Anthropic. *Introducing Contextual Retrieval.* 19 September 2024. · Edge, D.,
Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., Metropolitansky, D.,
Ness, R. O., Larson, J. *From Local to Global: A Graph RAG Approach to Query-Focused
Summarization.* arXiv:2404.16130, 2024. · Lewis, P., Perez, E., Piktus, A., Petroni, F.,
Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S.,
Kiela, D. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 33
(2020), 9459–9474. · Liu, J., Bai, J., Zeng, S. *Think Parallax: Solving Multi-Hop Problems
via Multi-View Knowledge-Graph-Based Retrieval-Augmented Generation* (method: ParallaxRAG).
arXiv:2510.15552, 2025.
