# Scale-up results — 500 pages

**Frozen record.** These numbers are final and the report draws from this file. Produced
by `configs/scale500.yaml` over 500 PubLayNet validation pages; narrative and method in
`artifacts/EXPERIMENT_LOG.md` Phase 10 parts A and B.

- Corpus: 500 pages (5 validation shards × 100), 5,744 canonical chunks, 410 figure/table
  crops, 1,819 KG edges over 2,769 nodes.
- Gold set: **127 questions**, every one human-approved; the 8 `figure_value` items were
  verified value-by-value against their crop images.
- Series: classic six-config ablation, `baseline → +layout → +clip → +caption → +kg →
  +rerank`. Baseline is every enhancement off — pure BGE text over naive fixed windows.
- Total API cost of the full sequence: ~$5.8.

---

## 1. Headline — the enhancement gap widens with corpus size

| | dev 50pp (n=55) | | scale 500pp (n=127) | |
|---|---|---|---|---|
| | baseline | enhanced | baseline | enhanced |
| R@5 | 0.69 | 0.87 | **0.57** | **0.91** |
| P@5 | 0.16 | 0.22 | 0.13 | 0.23 |
| MRR | 0.46 | 0.83 | 0.41 | 0.88 |
| nDCG | 0.52 | 0.84 | 0.47 | 0.89 |
| Correctness | 0.66 | 0.70 | 0.59 | 0.69 |
| Faithfulness | 0.73 | 0.78 | 0.82 | 0.82 |

Relative retrieval gain: **R@5 +26% at 50 pages, +60% at 500. nDCG +62% → +89%.**

The two ends move in opposite directions and that is the finding. Baseline degrades with
corpus size — naive fixed-window chunking over ten times as many candidates surfaces the
right region less often, R@5 0.69 → 0.57. The enhanced configuration does not degrade; it
improves, 0.87 → 0.91.

This matters because the common result is the reverse: gains measured on a small corpus
usually shrink as the corpus grows and the easy wins dilute. These enhancements are not a
small-corpus artefact, and the 50-page numbers understated them. Layout-aware units, the
caption→crop route and cross-encoder reranking each have more to discriminate against at
500 pages, so each does more work.

Answer correctness does **not** track the retrieval gap (0.59 → 0.69, +17%). Retrieval is
no longer the binding constraint on answer quality; see limitations 1–4.

**Uncertainty.** 95% bootstrap intervals over questions (10,000 resamples; full tables in
`confidence_intervals.md`). Baseline R@5 **0.57 [0.49, 0.66]** → enhanced **0.91 [0.86,
0.96]** — non-overlapping, so the retrieval result is not a sampling artefact. Correctness
baseline **0.59 [0.51, 0.67]** → enhanced **0.69 [0.61, 0.77]**: the marginal intervals
overlap, but the *paired* difference — the same resampled questions scored under both
configs — is **+0.099 [+0.034, +0.168]**, which excludes zero. Both are reported because
the marginal overlap is the more conservative reading and the paired interval is the more
powerful one, and they disagree about how confident to be.

Runs are **single-seed**, a deliberate consequence of the budget: repeating a six-config
ablation costs another ~$5. Two separate uncertainties are therefore reported rather than
conflated — the bootstrap intervals above capture *sampling* uncertainty, and *run-to-run*
variance was measured directly rather than assumed: re-running an identical configuration
over the same 17 multi_hop questions moved correctness by **0.029** (§6). Any effect at or
below **±0.03** is indistinguishable from running the same system twice.

## 2. Full ablation

**Where the win actually comes from — stated before the table, not buried under it.**
The retrieval improvement is almost entirely **layout-aware chunking and caption
anchoring**, with reranking third. It is **not** image embeddings, and on this corpus it
is **not** the knowledge graph. Paired bootstrap on each step's difference:

| step | ΔR@5 | ΔCorrect | verdict |
|---|---|---|---|
| baseline → +layout | +0.190 [+0.107, +0.274] | +0.065 [+0.000, +0.132] | R@5 real |
| +layout → +clip | **+0.000 [+0.000, +0.000]** | +0.000 [−0.018, +0.023] | **within noise** |
| +clip → +caption | +0.069 [+0.035, +0.107] | +0.000 [−0.030, +0.030] | R@5 real |
| +caption → +kg | +0.026 [+0.000, +0.059] | +0.030 [+0.004, +0.064] | **on the ±0.03 floor** |
| +kg → +rerank | +0.056 [+0.022, +0.097] | +0.004 [−0.037, +0.048] | R@5 real |

**`+clip` adds exactly nothing** — the ΔR@5 interval is [0.000, 0.000], not merely small.
BiomedCLIP fixed the score *distribution* (Phase 7) but the caption anchor, not image
similarity, is what makes a figure findable.

**The knowledge graph's contribution here is within the noise floor.** Its retrieval
effect is +0.026 [+0.000, +0.059] — interval touching zero, magnitude below the floor. Its
correctness effect is +0.030 [+0.004, +0.064]: the sampling interval excludes zero, but the
magnitude *is* the measured run-to-run floor, so it cannot be separated from re-running the
same system. The honest claim is that the KG did not demonstrably help on this corpus.

**And the diagnosis matters more than the number.** The HuggingFace parquet mirror this
corpus was ingested from drops the original filename, so PMC article IDs are unavailable in
this copy and page = paper — which forces every `multi_hop` question to be answerable
*within a single page*. That is precisely the regime where a graph helps least: the two
hops are already in the same retrieval unit's neighbourhood, and dense text retrieval
reaches them without traversing anything. A knowledge graph earns its cost when evidence is
scattered **across documents** — that corpus is exactly what PubLayNet cannot provide here.
So this is a null result about *this corpus*, not about GraphRAG; the architecture is
`paper_id`-agnostic precisely so a real multi-document corpus can be swapped in and the
question asked properly. Reported as a diagnosed limitation rather than presented as a win.

| config | R@5 | P@5 | MRR | nDCG | Correct | Faith | Decis |
|---|---|---|---|---|---|---|---|
| baseline | 0.57 | 0.13 | 0.41 | 0.47 | 0.59 | 0.82 | 0.84 |
| +layout | 0.76 | 0.18 | 0.78 | 0.75 | 0.66 | 0.80 | 0.88 |
| +clip | 0.76 | 0.18 | 0.78 | 0.75 | 0.66 | 0.81 | 0.90 |
| +caption | 0.83 | 0.21 | 0.79 | 0.80 | 0.66 | 0.81 | 0.87 |
| +kg | 0.86 | 0.21 | 0.80 | 0.80 | 0.69 | 0.83 | 0.88 |
| +rerank | **0.91** | **0.23** | **0.88** | **0.89** | **0.69** | 0.82 | 0.89 |

What each step buys:

- **`+layout` — the single biggest lever, again.** R@5 0.57→0.76, MRR 0.41→0.78. It alone
  recovers the entire baseline degradation and more. Same verdict as Phase 6, on a corpus
  ten times larger.
- **`+clip` — nothing measurable.** Identical retrieval to `+layout` on every metric.
  BiomedCLIP fixed the ranking *distribution* (Phase 7: σ ×6.7, #1-vs-#2 margin ×52) but
  image similarity remains the weakest of the four routes; the caption anchor does the
  work.
- **`+caption` — the figure win.** figure R@5 0.50→1.00, nDCG 0.61→0.99. Exactly as
  designed, and exactly as at 50 pages.
- **`+kg` — apparent correctness gain sits at the measured noise floor (above).** Overall
  correctness 0.66→0.69, single_fact R@5 0.79→0.84, but the paired difference is +0.030
  against a 0.03 floor, so it cannot be separated from re-running the same system. The
  feared dilution of single_fact by graph noise has now failed to appear twice, on two
  corpora — which is a real negative result even where the positive one is not.
- **`+rerank` — the largest retrieval step here.** R@5 0.86→0.91, MRR 0.80→0.88, a bigger
  contribution than at 50 pages. Consistent with the rest: reranking matters more when the
  candidate pool is deeper.

## 3. Per category

**No category carries a low-sample flag.** First time in the project — Phase 6 had
`figure_value` at n=4 and `treats` at n=1, both footnoted as not meaningful. The bootstrap
intervals below say what "no flag" is and is not worth: n=8 still buys an interval half a
point wide.

Point estimates with 95% bootstrap intervals, baseline → enhanced:

| category | n | R@5 baseline | R@5 enhanced | Correct baseline | Correct enhanced |
|---|---:|---|---|---|---|
| single_fact | 56 | 0.61 [0.48, 0.73] | 0.89 [0.80, 0.96] | 0.66 [0.54, 0.78] | 0.80 [0.70, 0.90] |
| multi_hop | 17 | 0.41 [0.24, 0.59] | 0.82 [0.68, 0.94] | 0.35 [0.18, 0.53] | 0.35 [0.18, 0.53] |
| figure | 12 | 0.38 [0.25, 0.50] | 1.00 [1.00, 1.00] | 0.29 [0.08, 0.50] | 0.25 [0.08, 0.46] |
| figure_value | 8 | 0.62 [0.25, 0.88] | 0.88 [0.62, 1.00] | 0.38 [0.12, 0.75] | 0.75 [0.38, 1.00] |
| text_derived | 23 | 0.70 [0.52, 0.87] | 1.00 [1.00, 1.00] | 0.83 [0.67, 0.96] | 0.87 [0.74, 0.98] |
| unanswerable | 11 | — | — | decision 0.82 [0.55, 1.00] | 0.91 [0.73, 1.00] |

`single_fact` (n=56) is the only category whose retrieval intervals separate cleanly.
`figure` and `text_derived` reach a degenerate [1.00, 1.00] because every resample recalls
every gold chunk. Everything at n ≤ 17 should be read as direction, not measurement.

**`figure_value` moved for the first time — and the interval says how much to trust it.**
In Phase 6 this category was flat at 0.50 correctness across every single config, which is
what motivated Phases 8 and 9. Here it responds: correctness 0.38 → 0.75, MRR 0.28 → 0.83.
But at n=8 those are **0.38 [0.12, 0.75] → 0.75 [0.38, 1.00]** — the intervals overlap
heavily, so this is *not* a statistically separated improvement. The mechanism is
independently verified (page-crop expansion took gold-crop retrieval from 2/4 to 8/8, §6),
which is the stronger evidence here; the correctness jump is consistent with it rather than
proof of it. Eight questions cannot carry more than that.

**`text_derived` remains the bias guard and it still holds.** These questions are written
from a passage with the graph never consulted, and they score highest at baseline (0.83
correctness) — i.e. the questions that are *independent* of the KG are the easiest for the
system, and the graph-derived categories start lower and need the enhancements. The gold
set does not flatter the graph pipeline.

## 4. Per predicate (enhanced config)

Point estimates with 95% bootstrap intervals:

| predicate | n | Correct | R@5 |
|---|---:|---|---|
| inhibits | 8 | 0.88 [0.62, 1.00] | 1.00 [1.00, 1.00] |
| transforms_to | 7 | 0.86 [0.57, 1.00] | 0.86 [0.57, 1.00] |
| measured_by | 7 | 0.86 [0.57, 1.00] | 0.86 [0.57, 1.00] |
| increases | 11 | 0.82 [0.55, 1.00] | 0.86 [0.64, 1.00] |
| causes | 12 | 0.79 [0.58, 0.96] | 1.00 [1.00, 1.00] |
| decreases | 9 | 0.61 [0.33, 0.83] | 0.78 [0.50, 1.00] |
| treats | 15 | 0.60 [0.40, 0.80] | 0.90 [0.74, 1.00] |
| **occurs_in** | 16 | **0.31 [0.12, 0.50]** | 0.81 [0.60, 1.00] |

Every predicate now has n ≥ 7, but **every interval here is wide** — the narrowest spans
0.38 and most span more. Only `occurs_in` separates from the rest: its interval tops out at
0.50, below every other predicate's point estimate, so its weakness is the one
per-predicate result the sample can actually support. The others are ordered but not
distinguished, and no ranking among them should be read as real.

The predicate mix itself shifted with the corpus: `treats` went from 5% of dev edges to
17%, `transforms_to` from 6% to 2% — the 500-page corpus is more clinical and less chemical
than the dev slice.

---

## 5. Limitations — all measured, with diagnoses

### 5.1 The closed 8-predicate schema does not generalise

Re-running the Phase-3 discovery protocol on this corpus (30 fresh chunks, 26 of them from
pages outside the dev 50): **86 triples, 73 distinct predicates, and the closed 8 cover
only 22% of triples and 23% of distinct predicates.**

Two missing types recur and are substantive:

- `is_a_predictor_of` (×4) — prediction/association, precisely the space vacated when
  `associated_with` was dropped in Phase 3 as a catch-all that over-connects the graph.
- `has_values_of` / `has_levels_of` (×5) — "entity has this measured value". The schema has
  `measured_by` for the *method* and nothing for the *value*.

The unmapped tail explains itself on inspection: *IASP established Global Year Against
Pain*, *robot asks human information about the object*, *variations in esthetic norms
do_not_hinder perception of smile attractiveness*. The 500-page corpus spans robotics,
dentistry and pain-policy editorials; the 8 predicates were consolidated from a
toxicology/epidemiology slice. The schema captures the causal/quantitative biomedical
relations it was designed for and is **domain-slice-specific, not universal**.

This is the closed-schema tension stated plainly: a closed schema buys sparsity,
comparability and precision (dev max entity degree 7, no hub explosion) and pays in
coverage, and the price stays invisible until the corpus broadens. The open alternative
pays the other way — Phase 3's free extraction produced 46 distinct predicates over 30
chunks and never reused an edge type, which is a graph that cannot be traversed.

**Schema kept unchanged, deliberately.** Changing it forks comparability with the dev KG,
the gold set's per-predicate coverage and the whole ablation.

**Consequence, and the link to `figure_value`:** the missing value-relation is the
*structural* reason Phase 6A's measurement detector found zero clean table-value edges.
That was read at the time as an OCR problem. OCR is real, but even with perfect OCR
`(mean age) —has_value→ (26.0 (8.6))` has no predicate to be extracted into — the graph
cannot represent the fact. That is why `figure_value` had to be hand-built, why it stayed
flat across every Phase-6 config, and why it needs the visual path at all.

### 5.2 Table-OCR-derived values are wrong more often than right — 53%

To fill `figure_value` (the KG yields 3 items on 500 pages, for 5.1's reason), 14 candidate
questions were proposed from table OCR and rendered beside their crops for verification.
**9 of 17 rejected — 53% overall, 8 of 14 (57%) among the OCR-derived proposals, 1 of 3
among graph-derived.**

| slot | reason |
|---|---|
| V01 | circular — "CYP1A inhibits CYP1A protein expression" is self-referential |
| V05 | wrong value **and** a question/table mismatch: the table reports NO₂, the question says NO |
| V07 | wrong value |
| V09 | unverifiable — crop too poor to read |
| V11 | unreadable crop |
| V12 | value misplaced: 26 is not e432's maximum (81 is) |
| V13 | wrong value |
| V15 | wrong value — should be 8.03 ± 0.19 |
| V17 | wrong value — drops the ×10⁻⁴ exponent |

This is an **independent measurement** of the table-OCR claim, reached by a different route
than Phase 9's. Phase 9 inferred *right digits, wrong structure* from a cell-by-cell check
of a single transcription; this is a human checking 14 independently-proposed values
against their images, and the failure modes are exactly the predicted ones —
**misplacement** (V12 takes a number from the wrong row), **structural truncation** (V17
drops an exponent, V15 takes the wrong statistic), and **label drift** (V05 attaches NO₂'s
value to NO). A pipeline that quoted table values out of OCR would be wrong more often than
right on this corpus. That is the empirical case for the visual path, and for the standing
rule that a transcription is used only to *find and score* a table and never to quote a
value.

It also reproduced the question/table mismatch class unprompted (V05) — and that one
**survived the automated filter**, because `NO` and `NO2` are too short and too similar for
a token-overlap test to separate. The crop-backed categories need human verification by
construction, not by sampling.

### 5.3 Abstention degrades at scale

Unanswerable abstain-rate was **1.00 in every config at 50 pages**; here it is **0.82** for
five of six configs and 0.91 at `+rerank`. Two of eleven deliberately-unanswerable
questions get answered.

Diagnosis: the same scale effect as on the retrieval side, running the other way. A
500-page corpus contains far more near-miss material, so a question the corpus does not
answer still retrieves context that *looks* answerable, and the prompt gate — which is
where the real semantic decision lives (Phase 6D) — is deciding against a harder
background.

This is measured against a tougher set than the dev 6: the absence check rejected 13 of 24
crafted candidates for scoring ≥0.62 against the corpus index (warfarin/INR 0.729,
paracetamol 0.688, metformin 0.685, appendectomy 0.672). The 11 survivors are the hard
cases by construction. **Without that check, 13 "unanswerable" items would have been
silently answerable and the abstention result would have measured nothing** — the check
earned its place at this corpus size, and would not have been needed at 50 pages.

### 5.4 `occurs_in` over-application — correctness collapses to 0.31

Worst predicate by a wide margin (n=16), while its retrieval is fine at R@5 0.81 — so this
is an extraction/gold problem, not a retrieval one. Dev had 0.80 at n=5, which the larger
sample now contradicts.

Diagnosis: `occurs_in` edges are loose at scale. Representative:
`(patients undergoing AraSns for gynecologic malignancies -occurs_in-> postoperative
patients)` — a partly-garbled subject and an object that restates the subject's context
rather than naming a distinct population. A question phrased from such an edge inherits the
looseness and cannot be answered crisply. `occurs_in` is the predicate Phase 3 already had
to tighten once (object restricted to a population/sample/site, never a number); at scale it
remains the least well-behaved member of the schema and is the first candidate for a
further object-type constraint.

### 5.5 `treats` direction inversions — 0.60 at n=15

Now measurable for the first time (Phase 6 had n=1 and footnoted it as meaningless), and
mediocre. Diagnosis is visible in the graph: `(patients -treats-> chemotherapy)` has
subject and object **inverted** — chemotherapy treats patients, not the reverse. Direction
errors were invisible at one instance; at fifteen they show up as mid-range correctness.
`treats` also tripled in share (5% → 17% of edges) as the corpus turned more clinical, so
this is now a load-bearing predicate with a known precision defect. A directionality check
in the extraction prompt, or a post-hoc subject/object type constraint, is the fix.

### 5.6 Reranking hurts multi-hop answering while helping its retrieval

`multi_hop` R@5 jumps 0.62 → 0.82 at `+rerank` while correctness *drops* 0.44 → 0.35.

Diagnosis: the cross-encoder scores each chunk independently for query-passage relevance,
which is not the same objective as assembling the two chunks that must be *combined*. It
promotes the single most on-topic passage and can demote the second hop out of the answer
context even as recall improves. The same shape appeared on `figure` in Phase 6, where the
text reranker demoted text-less crops. Retrieval metrics and answer quality come apart
here, and the retrieval metric is the misleading one.

### 5.7 The VQA provenance gate weakens as the corpus grows

The gate's condition is that a crop may be read only if its page is one whose **prose** the
question matched. With `top_k = 10` and page == paper, up to ten different papers clear
that bar. At 50 pages the eligible set was effectively the answering page; at 500 it is not.

Observed on q124: the gold crop was retrieved but ranked 10th, and the gate passed two
crops from *other papers* (`val00003:411764:r7`, `val00004:391918:r9`), so the answer was
read off the wrong table. The gate still blocks the unrelated crops it was built for — the
Phase-9 result stands — but its guarantee has genuinely weakened with corpus size.
Tightening it to the page supplying the top-ranked prose, or to the page carrying the most
text hits, is the obvious next step. Not changed here: altering it after the ablation would
invalidate these numbers.

### 5.8 A gold-question defect class the mismatch filter cannot catch

q124 asks *"What is the Mean ± SD for Age (years)?"*. Every term appears in its source
table, so the question/table mismatch filter passes it — correctly, by its own definition.
But across 500 pages dozens of tables report a mean age: the question is well-formed
against its source and **under-specified against the corpus**. Same shape as q060 in Phase
8.

This is a distinct defect from the q061 class. q061: *the source lacks what the question
names*. q124: *the question does not identify its source*. Catching it needs a uniqueness
check against the whole corpus, not a presence check against the cited region.

### 5.9 Other corpus-level limitations

- **Title OCR degrades at scale**: 20.4% of title regions OCR empty, against 0% in dev
  (text 0.7%, table 0.8%, list 0%). `LayoutChunker` prepends the section title to a
  region's text, so a fifth of layout units lose that context signal — a plausible
  contributor to any softening in the `+layout` step.
- **76 crops have no caption region at all** (of 410). The caption was never emitted as a
  text region or OCR destroyed it. Not recoverable without inventing a caption; this is the
  case page-crop expansion exists to serve. Caption coverage after repair: 85% (349/410),
  up from 71% before — see EXPERIMENT_LOG Phase 10 part A.
- **gpt-4o is not deterministic at temperature 0.** Single-question before/afters are
  evidence about a mechanism, never reproducible fixtures.
- **page == paper — a property of the mirror, not of PubLayNet.** The HuggingFace parquet
  mirror drops the original filename, so PMC article IDs are unavailable in this copy and
  multi-hop questions are within-page. The official distribution encodes the article in the
  COCO `file_name` field (`PMC<id>_<page>.jpg`), recoverable by joining the COCO `image_id`
  back to `val.json`. Both `page_id` and `paper_id` are stored, and `PaperGrouper` is the
  swap point, so a real grouper drops in with no schema change.

---

## 6. VQA vs text on the verified `figure_value` set

Paired comparison, enhanced retrieval run **once** per question and answered twice, so the
delta is attributable to the crop image and nothing else. Answer key from human
verification — never from the vision model, which would let the system under test author
its own answers.

Gold crop retrieved **8/8** (2/4 in Phase 8, before page-crop expansion).

| | strict | point estimate |
|---|---|---|
| TEXT-only | 4/8 | **6/8** |
| VQA | 4/8 | 5/8 |

**Attaching the crop image did not improve value reading.** Only two questions differ
between the paths, and they cancel exactly:

- **q123 — VQA wins, for the predicted reason.** Question asks the mean age. OCR gives the
  text path `26.0` but has lost the paired `(8.6)` standard deviation, so it answers
  `26.0 years`. The vision path reads the grid and returns `26.0 with a standard deviation
  of 8.6`. This is the Phase-2 hypothesis — that VQA recovers structure OCR destroys —
  finally observed on a verified item.
- **q128 — VQA loses, for the Phase-9 reason.** Expected `19`. The text path reads `19`
  correctly. The vision path, handed **the correct crop**, reads **`2`** off it and cites it
  properly.

So VQA's one real capability and its one real failure mode are the same size at n=8. **No
VQA win is claimed from this data.** What is claimed is a characterisation: it recovers
row/column pairings that OCR linearisation destroys, and it misreads individual cells in
dense tables — and because it misreads confidently and with a correct citation, the error is
harder to detect downstream than an OCR failure. Combined with 5.2, the picture is
consistent: neither text nor vision can be trusted to quote a table value unsupervised.

---

## 7. Operational characteristics

Measured on the development machine, not projected. Latency and memory were timed locally
with **no model API calls**; corpus-build time and token spend come from the logs of the
runs that produced the results above. Anything neither locally measurable nor recorded is
marked **not measured** rather than estimated. Full detail in `operational.md`.

Machine: Windows, 6 physical / 12 logical cores, 15.7 GB RAM, Python 3.11, **CPU only** —
no CUDA build installed, though the code is device-agnostic.

**Query latency, retrieval path only** (query embedding → text search → caption / page-crop
/ image / graph expansion → cross-encoder rerank), over all 127 gold questions:

| p50 | p95 | p99 | mean | min / max | cold start |
|---:|---:|---:|---:|---:|---:|
| 3.67 s | 13.44 s | 15.30 s | 4.79 s | 0.76 s / 15.89 s | 14.6 s |

The spread is wide because the cross-encoder reranks a pool whose size depends on how many
routes fired: a question that triggers caption, page-crop and graph expansion pays for
reranking all of it. This is CPU inference — the same work on a GPU is a different number.
Cold start loads three transformer models plus two FAISS indices and the graph, and is paid
once per process.

**Memory:** 60 MB interpreter baseline → **2,466 MB peak** with everything resident. The
three models coexist on CPU; on the 4 GB GPU this project targets they would need
sequencing.

**End-to-end latency including the answer model: not measured.** The harness recorded
answers and scores per question but never wall-clock, and obtaining it now would cost API
budget. The retrieval figures above are a lower bound; the answer and grounding calls
dominate the total.

**Corpus build:** OCR ingest of 490 new pages took **119.7 min** (14.7 s/page, 5,619
chunks) — per-shard 1670/1424/1495/1359/1233 s. OCR is single-threaded CPU work and
dominates construction; it is the stage that would gain most from parallelism or a GPU OCR
backend. KG extraction over 4,776 chunks consumed **3,876,391 tokens for $0.6397**. BGE and
BiomedCLIP index build wall-clock: **not measured** — not recorded at the time.

**API cost.** Per-config cost is **not measured**: the harness logs no token counts per
question, and the ablation prints cost per invocation rather than per config. Recorded:
ablation resumed invocation (3 of 6 configs) **$2.50**, chain-of-thought paired run
**$0.55**, VQA-vs-text paired run **$0.13**, KG extraction **$0.64**. The first ablation
invocation crashed on a rate limit before printing its cost line, so the six-config total
is known only to about **$5** — roughly $0.83 per config at 127 questions, which is a
division of a measured total, not a separately measured quantity. The answer model is
capped at 30,000 tokens/minute on this account, which is why the runner backs off and can
resume from a completed per-question log instead of re-buying it.

## 8. Reproducing

```
python -m scripts.ingest              --config configs/scale500.yaml
python -m scripts.build_indices       --config configs/scale500.yaml
python -m scripts.build_kg            --config configs/scale500.yaml
python -m scripts.discover_predicates --config configs/scale500.yaml
python -m scripts.build_gold          --config configs/scale500.yaml --target 130
python -m scripts.propose_figure_values --config configs/scale500.yaml --n 14
python -m scripts.render_gold_review  --config configs/scale500.yaml   # human verifies
python -m scripts.finalize_gold       --config configs/scale500.yaml   # records the verdict
python -m scripts.ablate              --config configs/scale500.yaml --series classic --resume
python -m scripts.vqa_figure_values   --config configs/scale500.yaml
```

`--resume` reloads any complete `perq_<config>.jsonl` rather than re-buying it. Note the
30,000 tokens/minute gpt-4o limit: a ten-chunk answer means the ablation sits against that
cap for its whole duration, so client retries are set to 10 (the SDK default of 2 is not
enough and killed a six-config run three configs in).
