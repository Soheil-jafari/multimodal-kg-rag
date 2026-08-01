# Experiment Log — Multimodal + Knowledge-Graph Document-QA over PubLayNet

*Lab notebook. **Append-only**, dated sections. Records what we did, why, concrete
results, and honest caveats — including failures and the ugly parts. Written as
prose to be lifted into the final report. Converted to `.docx` only at the very
end of the project.*

---

## 2026-07-27 — Phase 0: Setup & architecture

Established the architecture contract: **one flag-driven pipeline** (baseline =
all flags off, enhanced = all on), a **single SQLite chunk store** as the source
of truth (vector/image/graph indices all *derived* and joined by `chunk_id`),
every external dependency behind an ABC (`LLMClient`, `GraphBackend`,
`VectorIndex`, …), and a `platform_core/` + `domain_packs/` layout.

**Dataset reconnaissance (jordanparker6/publaynet, parquet).** Read directly with
`pyarrow` — the `datasets` library is unnecessary and was not installed. Dev
shard `validation-00000-of-00008` = **1,406 pages** (8 val shards ≈ 11.2k pages).
Columns: `image` struct<bytes, path>, `id` int64, `annotations` COCO list. Boxes
are `bbox = [x, y, w, h]` in pixels; labels are a raw `category_id` int, mapped
(verified by the region distribution) as `1=text, 2=title, 3=list, 4=table,
5=figure`. **No text is provided — it must be OCR'd from each region crop.**

**Critical finding — no document identifier.** `image.path` is null for all
1,406 rows and `id` is only a per-page COCO image_id. We therefore **do not**
reconstruct multi-page papers. Decision (approved): **page = paper**, with
`paper_id = page_id = page_uid = "val00000:<id>"` and `chunk_id =
"<page_uid>:r<k>"`. `multi_hop` questions are **within-page** (≥2 regions on one
page). A consecutive-id proxy showed ~77% of pages isolated within a single
shard, so real multi-page structure is rare here anyway. This is logged as an
honest limitation; the pipeline is `paper_id`-agnostic, so a real PMC grouper can
be swapped in with no schema change.

**OCR engine.** Tesseract is not installed on this machine, so we chose
**RapidOCR** (`rapidocr-onnxruntime`, CPU, pure-pip, no admin) behind the
`RegionOCR` ABC — a one-line swap to EasyOCR/Tesseract later.

---

## 2026-07-27 — Phase 1: Ingestion (loader → OCR → canonical chunk store)

Ingested **50 dev pages → 682 canonical region rows** in
`artifacts/dev/chunks.sqlite`; figure+table crops (57) saved to
`artifacts/dev/crops/`. The schema keeps both `page_id` and `paper_id` (equal
now) so real grouping can be added later without migration.

Rows by region type: **text 547, title 57, table 45, list 21, figure 12.**

**OCR empty / garbled rate.** text 1.5% empty, title/list/table 0% empty;
**overall OCR'd 1.2% empty, 0.0% character-level garbled**; figures are crop-only
(not OCR'd), 12/12 crops saved.

> **Honest caveat (report limitation #2).** The 0.0% "garbled" number only flags
> *empty or character-level gibberish*. It does **not** capture **table
> structural de-layout**: RapidOCR linearises a table's rows/columns into one
> flat token stream. Values and headers survive (e.g. *"Age (years) 4320 26.0
> (8.6) … Protein C (unit/dl) 4315 102.4 (15.8)"*) but the grid structure is
> lost. So structural table questions cannot be answered from OCR text.

Sample quality by type: `text` reads cleanly with minor slips (`ewage`→sewage,
`Ihird`→Third, `nunber ef`→number of); `title` short (`C ase`→Case); `table`
messy-but-non-empty. **Useful structural finding:** figure/table **captions are
separate `text` regions** (e.g. "Figure 6. …", "Table 4. …"), which we exploit
for grounding. Verdict: OCR will not limit a text-centric gold set; it only
constrains how table/figure items are phrased.

---

## 2026-07-27 — Phase 2: Derived indices (BGE text, CLIP image, caption links, chunkers)

All indices built from the chunk store, keyed by `chunk_id`, FAISS
`IndexFlatIP` (cosine on L2-normalized vectors). Models loaded **one at a time**
(BGE, then CLIP). Ran on **CPU** (torch 2.5.1+cpu; no CUDA on this box) — the
code is device-agnostic and will use the GPU once a CUDA torch build is present.

**Index sizes (BGE-tokenizer counts):**

| index | model | dim | vectors |
|---|---|---:|---:|
| text_layout (enhanced) | BGE-base-en-v1.5 | 768 | 710 units |
| text_naive (baseline)  | BGE-base-en-v1.5 | 768 | 256 windows |
| image (figure+table crops) | CLIP ViT-B/32 | 512 | 57 |

**Chunker effect:** LayoutChunker (unit = region; over-long regions split with
overlap; section title prepended) → 710 units; NaiveChunker (~400-token windows,
structure-agnostic) → 256 windows, over the same corpus. (An earlier word-count
sanity check said 669/160; subword tokens inflate counts, so token-based
splitting yields more, smaller units — the token counts are authoritative.)

**Caption-link coverage** (each figure/table crop → nearest "Figure N"/"Table N"
text region, stored as `caption_id`): **figures 9/12 (75%), tables 45/45 (100%),
overall 54/57 (95%).** The 3 unlinked figures had no page text region beginning
"Figure/Fig" (caption OCR garbled or absent).

**Retrieval smoke test — query: *"How did SARS spread geographically over time?"***

- **TEXT (BGE) — strong.** Top-5 are *all* from the SARS spatial-spread page
  `val00000:415624`, cleanly separated: 0.713, 0.681, 0.665, 0.627, 0.622 — and
  rank 4 is the figure's caption region (`:r6`).
- **IMAGE (CLIP) — right answer #1, but essentially a coin toss.** The correct
  SARS cluster figure (`:r11`) ranked #1 at **0.2504**, but the entire 57-crop
  similarity distribution is **min 0.1896, mean 0.2263, max 0.2504, σ=0.0128** —
  the correct figure beats an unrelated dioxin table by **0.0003**.

> **HEADLINE FINDING.** CLIP ViT-B/32 **barely discriminates on out-of-domain
> scientific figures** (it is trained on natural images). Distribution
> 0.19–0.25, σ=0.013, correct figure wins by 0.0003. This confirms the predicted
> domain-shift limitation and is the single most valuable result so far.

**Design response.** The multimodal win comes **less from CLIP retrieval ranking,
more from caption-anchoring + VQA**: retrieve figures/tables via their **caption
text** (BGE is strong and surfaced the caption at rank 4) → follow `caption_id` →
pull the crop; treat CLIP similarity as a weak secondary signal; **answer**
figure/table (incl. structural-table) questions by feeding the crop to **VQA**,
not OCR. Tables thus become a second visual-path win alongside figures.

---

## Backlog (logged, not yet acted on)

- **Fuzzier caption matcher** — recover the 3 unlinked figures by tolerating OCR
  noise in the "Figure N" prefix.
- **Day 8: swap CLIP → BiomedCLIP / PubMedCLIP** and re-run *this exact* SARS
  smoke test as a before/after on the 0.19–0.25 distribution. (Hold — do not
  build yet.)
- **KG precision pass** — restrict `occurs_in` objects to a population/sample/site
  and skip negated / no-effect statements (see Phase 3 precision caveat).

---

## 2026-07-27 — Phase 3 (prototype): closed-schema KG extraction

Two-stage method to define the predicate schema from data rather than assume it.
Model: gpt-4o-mini. **Prototype only** — no full-corpus run pending schema and
edge-quality review.

### Stage 1 — free extraction (schema discovery)

30 sampled chunks (text/list/table, ≥200 chars, strided across pages) → **64
triples, 46 distinct predicates, 15,832 tokens, $0.0044**. The output was heavily
fragmented and non-reusable — frequency head: `have_lower_levels_of ×7`,
`have_higher_levels_of ×4`, `has_prevalence_of_drinkers ×3`,
`has_prevalence_of_hbv/hcv_carriers ×3`, `contain ×2`, `reduced ×2`, then a long
tail of one-offs (`was_carboxylated_via`, `had_rr_for_cll`, `is_less_effective_than`).
This fragmentation is the argument for a closed schema: left free, the graph never
reuses an edge type.

### Consolidated closed schema (8 predicates) → `domain_packs/biomed/predicates.yaml`

- `treats` — therapeutic use (from used_as / is_used_as).
- `causes` — production/causation (caused, is_formed_when).
- `inhibits` — suppression/blocking (canonical; thin sample support, retained for scale).
- `increases` — higher level/amount/risk (have_higher_levels_of, is_greater_in, overexpressed).
- `decreases` — lower level/amount/risk (have_lower_levels_of ×7, reduced, have_lower_wbc_count_than).
- `transforms_to` — chemical conversion (was_oxidized_to, underwent_degradation_to, was_carboxylated_via).
- `occurs_in` — occurrence/prevalence in a population or sample (has_prevalence_of_*, observed_in, identified_in, colonized_by).
- `measured_by` — quantified by a method/assay (had_concentrations, estimated_or, analyzed).

Versus the Phase-0 draft (treats, causes, inhibits, associated_with, increases,
decreases, measured_by, compared_with): **dropped** `associated_with` and
`compared_with` as vague catch-alls that over-connect the graph; **added**
`transforms_to` and `occurs_in`, both strongly evidenced (chemistry
transformations; epidemiological prevalence).

### Stage 2 — schema-constrained extraction (5 chunks incl. a table)

Enforcement is in code: normalize the predicate, drop it if not in the closed set
or if subject/object/evidence is missing; survivors become grounded edges
(`chunk_id` + verbatim evidence).

**Strict zero-shot prompt:** 1 kept edge across 5 chunks, 0 dropped. Zero drops
because gpt-4o-mini **self-censors** — it returns `[]` rather than emit an
out-of-schema predicate, so the code net has nothing to reject. Recall far too low
for a usable graph.

**Enforcement proven on real model output:** applying the closed schema to Stage
1's 64 free triples drops **64/64** as out-of-schema (exact name match after
normalization). The mechanism fires hard; the constrained prompt simply rarely
feeds it violations.

**Fix — mapping guidance + one worked example** in the constrained prompt
(comparative→increases/decreases, prevalence→occurs_in, conversion→transforms_to).
Re-run on the same 5 chunks: **28 grounded edges**, every one carrying `chunk_id`
+ evidence. Examples:
- `(HCC -occurs_in-> drinkers)` — ev: "The prevalence of drinkers was 92.3% (12 of 13) in HCC."
- `(nitrite -transforms_to-> methemoglobin)` — ev: "nitrite … oxidizes the ferrous iron in hemoglobin … to the ferric form"
- `(5 ng/L EE -causes-> complete reproductive failure in the F1 generation)`
- `(low-fructosamine group -decreases-> total cholesterol)`

**Precision caveat (honest):** ~2–3 of 28 are mis-mapped — `occurs_in` over-applied
to a non-population object ("methemoglobinemia occurs_in 2% of the total Hb"), and
one no-effect statement ("did not affect total reproductive success") wrongly
emitted. Recall is now good; precision needs a light tightening before the full
run: restrict `occurs_in` objects to a population/sample/site, and instruct the
extractor to skip negated / no-effect statements. The delayout'd gene table
yielded 0 edges — tables stay a visual-path (crop+VQA) problem, not a
text-extraction one.

### Cost

Stage 1 $0.0044 (30 chunks); Stage 2 $0.0021 (5 chunks, $0.00041/chunk). Full
50-page dev KG (605 text-bearing chunks) ≈ **$0.25**; full shard (~1,406 pages) ≈
**$7.0** — nearly the whole $8 credit, so the full-shard KG is a deliberate
one-time spend, not a dev loop.

### Status

Schema + extraction prompt not yet committed to a full-corpus run.

---

## 2026-07-28 — Phase 3: KG build over the dev corpus

Schema signed off (8 predicates as proposed). Precision tweak applied to the
constrained prompt before building: `occurs_in` object restricted to a
population/sample/site (never a number/percentage); negated / no-effect statements
skipped. Verified on the four prototype-rich chunks — the tweak removed both known
mis-maps (`occurs_in`→"2% of total Hb"; the "did not affect…" no-effect edge) while
preserving yield (17 edges on the fructosamine chunk).

Built over all **605** text/list/table chunks of the 50-page dev store
(gpt-4o-mini, NetworkX backend). Scope is deliberately the dev corpus only — the
full 1,406-page shard is never processed; the final run later targets ~150 pages.
Artifacts: `artifacts/dev/kg/graph.pkl` (+ `graph.pkl.edges.jsonl`).

**Result: 225 edges, 350 nodes, 0 failures, $0.083** (503,792 tokens — a third of
the $0.25 estimate; most chunks are sparse).

Edges per predicate: `increases 82 | causes 46 | decreases 36 | occurs_in 23 |
transforms_to 14 | treats 12 | inhibits 9 | measured_by 3`. Directional quantity
relations (increases+decreases = 118, 52%) and causation dominate — consistent with
an epidemiology/toxicology corpus; `measured_by` is rare as measurement statements
are sparse here.

Quality signals (objective):
- **Grounding: 225/225** edges carry `chunk_id` + verbatim evidence (0 missing).
- Redundancy: 222/225 unique (subject,predicate,object) — negligible duplication.
- No over-connection: max entity degree is 7 (`high-fructosamine group`); no runaway
  hub. Confirms that dropping `associated_with`/`compared_with` kept the graph sparse.
- Value-as-object looseness: 25/225 objects contain a number (11%); 6 (3%) are bare
  measurements (e.g. "C-reactive protein increases 6.1 mg/dl") — object should be the
  quantity, not its value.

Sample edges (hand-check): `(hyponatremia -causes-> generalized tonic-clonic
seizures)` and `(PCB153 exposure -decreases-> expression of BCL-2 and WEE1)` are
clean; `(cefuroxim -treats-> patient's temperature)` and `(C-reactive protein
-increases-> 6.1 mg/dl)` have loose object typing; `(tumor -transforms_to->
hematoma)` is a genuine mis-map ("identified as", not a transformation).

Honest read: recall and grounding are strong; precision is decent but imperfect —
recurring issues are loose object typing (a value/outcome in place of the canonical
quantity/condition) and occasional `transforms_to` over-application to diagnostic
"identified as" phrasing. **Backlog:** object-type constraints or a lightweight
verification pass to lift precision before the final ~150-page run.

---

## 2026-07-28 — Phase 4: flag-driven retriever

**One retrieve() path.** Dense BGE text retrieval always runs; `use_clip_images`,
`use_kg`, `use_rerank` each gate an independent block that *adds* to a pool keyed
by chunk_id. There is no baseline-vs-enhanced branch — behaviour is a pure function
of `RetrievalFlags`. `candidate_pool()` builds the merged, provenance-tagged set;
`retrieve()` reranks (if `use_rerank`) or sorts+truncates to `top_k`. Provenance is
comma-joined across sources: `text | caption-link | clip-image | graph-hop`.

**chunk→nodes entry** (`artifacts/dev/kg/graph.pkl.chunk_nodes.json`): `chunk_id ->
sorted entity list`, e.g. `"val00000:410520:r2": ["6.1 mg/dl", "C-reactive
protein"]`. Coverage: **103/605** text-bearing chunks map to ≥1 node; 502 map to
none (relation-poor methods/fragments — expected). `use_kg` seeds the walk from the
entities of the text-retrieved chunks, walks `kg_hops` (config, default 1), collects
the crossed edges, and pulls their evidence chunks by chunk_id; every graph hit
carries the edge's evidence sentence + chunk_id.

**Baseline purity** — query "How did SARS spread geographically over time?", all
flags off: 8 results, every `source == text`. Pure — no captions/graph/images leak
in.

**Enhanced merged pool**, same query (`use_clip_images + use_kg`): text 8;
**caption-link 2** — the SARS figure crops `r10`/`r11`, reached by retrieving their
"Figure …" caption then following `caption_id` back to the crop, scored ~**0.62** by
inheriting the caption's text score; **clip-image 4** weak hits at ~0.25 (unrelated
crops). The caption-anchored crops rank far above the CLIP hits — the primary figure
route dominates the weak secondary signal, as intended. `graph-hop` did not fire:
SARS spatial-epidemiology text yields no schema edges (spread/clustering aren't in
the 8 predicates), so those chunks have no nodes.

**graph-hop** demonstrated on a KG-rich query ("How does PCB exposure affect gene
expression?"): 3 of the top merged results are `graph-hop,text` — e.g.
`val00000:414928:r0` "PCB153 exposure decreased expression of the apoptotic genes
BCL-2 and WEE1." with evidence + chunk_id carried. No single query exercises all
four sources — SARS is image-rich but KG-poor; PCB is KG-rich but its captions
weren't text-retrieved. This is corpus content, not a wiring gap.

**Rerank** (cross-encoder ms-marco-MiniLM, final step over the merged set; text-less
crops scored on their caption): reranked SARS top-2 are the two most on-topic text
chunks (+3.07, +2.18); the figure crop and off-topic text fall to the bottom.

Note: BGE + CLIP + cross-encoder are held together during retrieval (CPU here). On
the 4GB GPU this needs sequencing — a wiring note for the GPU path, not a logic
change.

**graph-hop confirmation** (two-entity query "What method measured the compound that
transforms into methemoglobin?", kg_hops=2, CLIP off): top-8 text hits include
node-bearing chunks (`415717:r7 -> {Methemoglobin, nitrite}`; `415232:r9 -> {2-NA,
naphthalene, 2-methylnaphthalene, …}`); the walk yields `source breakdown {text:5,
graph-hop,text:3, graph-hop:3}`. The three pure `graph-hop` rows are NEW chunks not
in the text top-8 (e.g. `415232:r4` "…decalin-2-carboxylic acid … metabolized to
CO"; `415232:r6` "2-methylnaphthalene was also oxidized to 2-NA"), each carrying
evidence + chunk_id. Graph expansion fires correctly; the earlier SARS emptiness was
corpus content (no schema edges to seed from), not a bug.

---

## 2026-07-28 — Phase 5: grounded generation + abstention + grounding trust layer

One path (no baseline/enhanced fork): dense answering always; abstention gated by
`allow_abstain` — a code gate (top context-query cosine < `abstain_min_score`, no
LLM call) plus a prompt clause. Answering model **gpt-4o**; grounding verify
**gpt-4o-mini**. Grounding = per-sentence cosine (sentence vs cited chunk), LLM
yes/no only for borderline sentences (within 0.08 of the 0.45 threshold). Three
queries, enhanced flags (clip+kg+rerank+abstain), kg_hops=2:

**(a) single-fact** — "What did PCB153 exposure do to BCL-2 and WEE1 expression?"
relevance 0.822 → answered. *"PCB153 exposure decreased the expression of the
apoptotic genes BCL-2 and WEE1 [val00000:414928:r0]."* Cited `r0` (`graph-hop,text`).
Grounding 1/1 SUPPORTED (sim 0.85). Clean.

**(b) multi-hop via graph** — "Through what chemical transformations does
2-methylnaphthalene pass, and what is the final carbon product?" relevance 0.755 →
answered **using graph-hop chunks**: the whole naphthalene chain
(`415232:r9/r6/r4/r2`) entered via `graph-hop`, and all THREE cited chunks are
`graph-hop,text`. Answer traces 2-methylnaphthalene → 2-NA → decahydro-2-NA → CO2
(mineralized). Grounding: final claim SUPPORTED (sim 0.72), but the first two
sentences flagged UNGROUNDED/uncited — the model **clustered all citations at the
end** instead of inline per claim, and one sentence ("fumarate addition, analogous
to toluene") reads as mechanism detail beyond the cited chunks. The trust layer did
its job: it surfaced under-citation / possible embellishment rather than
rubber-stamping. Backlog: require a citation per sentence (or attribute trailing
citations).

**(c) not in corpus** — "What is the capital of France?" → **ABSTAINED**. relevance
**0.453, just ABOVE** the 0.45 code-gate threshold (BGE returns ~0.45 even for
unrelated text), so the code gate did NOT fire — the **prompt gate** caught it:
gpt-4o returned "insufficient evidence in the corpus". This is precisely why
belt-and-braces matters: the cheap code gate is deliberately conservative (avoid
false abstention on genuinely-answerable questions), and the LLM prompt instruction
is the semantic backstop.

Honest reads: (1) the code-gate cosine threshold is a blunt instrument — the real
semantic abstention lives in the prompt; keep both. (2) The grounding layer is
strict per-sentence, which correctly flags clustered/absent citations; that
strictness is the point of a trust layer, not a bug.

---

## 2026-07-28 — Phase 6A: gold set (build + cleanup)

Construction discipline: every question is SELECTED from our own KG edges (answer +
supporting chunk_ids known from the edge); gpt-4o-mini only PHRASES the selected
fact — it never supplies the answer or evidence. Coverage-driven across categories
and the 8 predicates, plus a text-derived bias guard (LLM reads a passage, no graph)
and crafted unanswerable items. First pass: 59 questions, $0.0019.

Audit found ~9 broken items + a structural figure_value failure. Cleanup dropped
BROKEN only (kept hard questions): 3 circular/degenerate multi-hops from
near-duplicate co-located edges (blood-lead), 2 OCR-garbled answers, a "height
increases age" reversal, an off-domain IARC-monograph figure, and the false-positive
figure_value. Re-tagged q002/q003 treats->causes (the KG had mislabelled
"promotes/causes" as treats).

**q013 — the KG direction is CORRECT; the defect was gold phrasing.** PM2.5 ->
"relative risk of cardiovascular mortality" carries four edges: three `increases`
("2.5%", "4.0%", "11.4% increase") and one `decreases` — and that decrease edge
faithfully captured a real sentence: "a 1.1% decrease ... 3 days later" (a lag-day
dip). The builder picked the lone lag-specific decrease and over-generalised it. So
there is NO KG direction error; the extractor was faithful to each sentence. Repaired
the question to the correct `increases` fact.

**figure_value — measured scarcity that motivates VQA.** The measurement detector
(number+unit) found **0** clean table-value edges in the KG: tables OCR so poorly that
no numeric measurement survived into an edge. Graph-sourced figure_value = 0. Rebuilt
4 by HAND-VERIFYING values directly against table crops (mean age 26.0 (8.6);
lung-cancer URR 1.20 (1.11-1.29); well-water 75.5% (71.0-80.0); brain-tumor SMR 1.4
(0.9-2.0)) — capped at 4 real values, not padded. The zero-from-graph result is itself
a finding: table values are unreachable via text/graph and need the visual (VQA) path.

**Extraction-coverage finding (bias guard worked).** The text-derived guard surfaced
q053 — "prevalence of drinkers among HCC cases = 92.3% (12 of 13)" — by reading the
source passage. The graph-based figure_value path missed this table value entirely (it
never became an edge). Measured evidence that KG extraction under-covers tabular facts,
and that an independent text-derived channel catches what the graph misses.

Final set: **55 questions, 0 validation problems (no dangling chunk_ids).**
Distribution: single_fact 23, text_derived 10, multi_hop 7, unanswerable 6, figure 5,
figure_value 4. Per-predicate coverage: treats 1, causes 8, inhibits 3, increases 8,
decreases 4, transforms_to 4, occurs_in 5, measured_by 3. Caveat: `treats` (1) and
`measured_by` (3) are naturally scarce — this is a toxicology/epidemiology corpus
(harmful exposures), so treatment/measurement relations are rare; the `treats`
per-category signal rests on a single question.

---

## 2026-07-29 — Phase 6B/C: full ablation (6 configs x 55 gold questions)

Actual API cost **$2.22** (gpt-4o gen+judge + mini verify), under the $2.40 estimate.
`allow_abstain` held ON across all configs. Full tables in `reports/ablation.md`;
per-question logs (predicates walked, provenance, abstention, correctness) in
`reports/perq_<config>.jsonl`. Metrics: R@5/P@5/MRR/nDCG vs gold supporting_chunk_ids;
Correct = gpt-4o rubric judge; Faith = grounded-sentence fraction; Decis = correct
abstain/answer decision.

### Overall (rows = configs)
| config | R@5 | P@5 | MRR | nDCG | Correct | Faith | Decis |
|---|---|---|---|---|---|---|---|
| baseline | 0.69 | 0.16 | 0.46 | 0.52 | 0.59 | 0.79 | 0.76 |
| +layout | 0.80 | 0.19 | 0.81 | 0.77 | 0.60 | 0.84 | 0.76 |
| +clip | 0.80 | 0.19 | 0.81 | 0.78 | 0.63 | 0.85 | 0.78 |
| +caption | 0.84 | 0.21 | 0.81 | 0.81 | 0.60 | 0.85 | 0.75 |
| +kg | 0.84 | 0.21 | 0.82 | 0.82 | 0.65 | 0.83 | 0.78 |
| +rerank | 0.87 | 0.22 | 0.83 | 0.84 | 0.61 | 0.82 | 0.78 |

Per-category (R@5 / Correct across configs; full tables in reports/ablation.md):
- **single_fact (n=23):** R@5 0.70→0.83 (all lift at +layout); Correct 0.63↔0.67 (noise).
- **multi_hop (n=7):** R@5 0.50→0.93; Correct 0.36→0.43. Retrieval lift from +layout & +rerank.
- **figure (n=5):** R@5 0.50→1.00 (jump at **+caption**); Correct 0.20→0.40 then 0.20 at +rerank.
- **figure_value (n=4 ⚠ LOW-N):** flat (R@5 0.50, Correct 0.50) across all configs — needs VQA.
- **text_derived (n=10):** R@5 1.00 and Correct 0.90 already at baseline → 0.95 enhanced.
- **unanswerable (n=6):** Decis 1.00 every config (all abstained).

Per-predicate (enhanced): treats 1 (n=1, footnoted, not meaningful); causes 0.56 (8);
inhibits 0.67 (3, low-n); increases 0.50 (8); decreases 0.50 (4, low-n); transforms_to
0.25 (4, low-n); occurs_in 0.80 (5); measured_by 0.67 (3, low-n).

### What each enhancement adds (per category — where it helps AND hurts)
- **+layout** — the dominant RETRIEVAL win everywhere: overall MRR 0.46→0.81,
  nDCG 0.52→0.77; single_fact R@5 +0.13, multi_hop R@5 +0.29, figure MRR 0.32→1.00.
  Layout-aware chunking is the single biggest lever.
- **+clip** — marginal (figure Correct +0.10, else flat). Consistent with the Phase-2
  CLIP-weak-on-scientific-figures finding.
- **+caption** — THE figure retrieval win (figure R@5 0.50→1.00, nDCG→1.00), as
  designed. BUT it HURTS multi_hop (R@5 0.79→0.71) — added figure crops dilute the
  text multi-hop pool. A real trade-off.
- **+kg** — lifts multi_hop Correct 0.29→0.43 and figure 0.30→0.40, and single_fact
  stays 0.63→0.67. The feared graph-noise dilution of single_fact **did not happen** —
  KG never hurt single_fact retrieval or correctness.
- **+rerank** — big ranking win on multi_hop (R@5 0.71→0.93) and single_fact (MRR→0.83).
  BUT it HURTS figure Correct (0.40→0.20): the text cross-encoder demotes text-less
  figure crops (reranked on their caption) out of the answer context.

Net: enhancements deliver a **large, clear RETRIEVAL improvement** (R@5 0.69→0.87,
+26%; nDCG 0.52→0.84, +62%; MRR 0.46→0.83). Answer **correctness is roughly flat/noisy**
(0.59→0.61 overall, peak 0.65 at +kg) — bottlenecked not by retrieval but by
over-abstention (below) and small per-category n. Measured, not hidden.

### Abstention
Unanswerable: **6/6 correctly abstained in every config (100%).** But attempt-rate on
answerable is only 0.71–0.76 — **~25% of answerable questions are WRONGLY abstained**
(→ correctness 0). Over-abstention is the primary answer-quality drag; the code-gate
cosine threshold (0.45) is too eager. Tuning it is the highest-leverage answer fix,
since retrieval is already strong. (Backlog.)

### text_derived vs graph-derived (bias guard)
text_derived (independent of the graph) scores **highest** — R@5 1.00, Correct 0.90 at
BASELINE → 0.95 enhanced. The graph-derived categories start lower (single_fact 0.63,
multi_hop 0.36 at baseline) and need the enhancements. So the gold set does **not**
flatter the graph pipeline: the independent text-derived questions are the easiest for
the system and the graph-built ones are harder. The enhancement gains are real, not an
artefact of graph-favouring questions.

(Pre-fix tables preserved at `reports/ablation_v1_before_abstain_fix.md`.)

---

## 2026-07-29 — Phase 6D: abstention fix (prompt, not threshold) + re-run

**Diagnosis** (enhanced config): the ~25% false abstention was NOT threshold-driven.
The 6 fixable false-abstentions (answerable, gold retrieved r@5=1.0) had relevance
**0.67–0.82 — all ABOVE the 0.45 code gate**, i.e. prompt-gated (gpt-4o refused despite
holding the answer). Unanswerable relevance (0.42–0.67) *overlaps* answerable
(0.61–0.87), so no threshold separates them — the semantic prompt must decide. Fix:
**kept the code gate at 0.45; softened the abstention PROMPT** to attempt when the facts
are present (even multi-hop/noisy) and abstain only when the context does not address
the question. Boundary check (12 questions): 4/6 fixable recovered, 6/6 unanswerable
preserved. Re-ran the full ablation, **$2.36**.

Overall Correct / Decis, before → after:
`baseline 0.59→0.66 / 0.76→0.91 · +layout 0.60→0.68 · +clip 0.63→0.67 · +caption
0.60→0.66 · +kg 0.65→0.65 · +rerank 0.61→0.70 / 0.78→0.91`. Retrieval metrics
unchanged (abstention is orthogonal to retrieval).

Attempt-rate on answerable rose ~0.73–0.76 → **~0.86–0.90** (false abstention roughly
halved); **unanswerable abstain-rate held at 1.00 in EVERY config** — no hallucination
introduced, the balance the tuning targeted.

Per-category correctness, enhanced (+rerank), before → after: **single_fact 0.63→0.80,
multi_hop 0.43→0.57**, figure 0.20→0.20 (rerank still demotes crops — separate issue),
figure_value 0.50→0.50 (n=4), text_derived 0.95→0.90 (noise).

**The pair:** enhanced answerable-correctness **0.61 → 0.70** with unanswerable
abstention **held at 100%**. Faithfulness dipped slightly (enhanced 0.82→0.78) — an
honest trade: attempting more answers surfaces a few less-grounded sentences. The
remaining false-abstentions are **retrieval misses** (gold not in top-k), a retrieval
limit not an abstention one.

Post-fix reading of the enhancement claim: the dominant win remains **retrieval**
(R@5 0.69→0.87, nDCG 0.52→0.84); overall correctness **0.66 (baseline) → 0.70
(enhanced)** — modest but real, concentrated in single_fact (0.74→0.80) and multi_hop
(→0.57) at +rerank. gpt-4o answers well from baseline context once the chunk is
retrieved, so the correctness gap is smaller than the retrieval gap.

---

## 2026-07-29 — Phase 7: BiomedCLIP swap (image encoder), 50-page dev corpus

Acting on the Phase-2 backlog item: replace general CLIP ViT-B/32 with
**BiomedCLIP** (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` —
ViT-B/16 image tower + PubMedBERT text tower, 512-d shared space) and re-run *the
exact* Phase-2 SARS smoke test as a before/after on the 0.19–0.25 distribution.

**Swap shape.** An interface-level change, exactly as the architecture intended. New
`BiomedClipEmbedder` in `platform_core/llm/embeddings.py` exposes the same
`embed_images` / `embed_text` / `dim` surface as `ClipEmbedder`; a one-entry
registry (`IMAGE_ENCODERS`) maps the model-id string to both the implementation
**and** its index basename, so each encoder writes its own FAISS file
(`index/image` vs `index/image_biomedclip`) and **both stay runnable with no
rebuild**. `make_image_embedder()` is the only construction site;
`scripts/build_indices.py` (new `--image-only`) and `scripts/ablate.py` (new
`--image-model`) both go through it. No retriever change: the image path already
took an injected embedder + index.

> **Honest correction to the "config-driven" claim (limitation #6).**
> ***→ RESOLVED in Phase 7.5 (below). Kept here as the record of what was true at
> the time of this phase.*** The selection
> point is `ModelConfig.image_embedding_model` + the `--image-model` CLI flag —
> **not** the YAML. `AppConfig.from_yaml` still raises `NotImplementedError` and is
> called from nowhere; no `extends:` resolution exists; `configs/*.yaml` are read by
> **no code at all** (the repo's only working YAML loader is
> `load_predicates` for the KG schema). `configs/enhanced_biomedclip.yaml` was added
> for parity with the existing files and documents the intended override, but it is
> declarative only until the config loader is written. The runtime truth is that
> every entry point instantiates the leaf dataclasses directly, and drift has already
> crept in: `scripts/ablate.py` uses `top_k=10, rerank_top_k=10` while both
> `config.py` and `default.yaml` say 8/5. So the swap *is* a one-string change behind
> a stable interface — the honest description — but "config-driven via YAML" is not
> yet true of this repo, for images or anything else.

**Environment friction (worth recording).** BiomedCLIP's HF repo ships **only**
`open_clip_pytorch_model.bin` + `open_clip_config.json` — there is no
transformers-format checkpoint, so `AutoModel` and `SentenceTransformer` cannot
read it and `open_clip` is mandatory. Installing it exposed a pre-existing break:
the project's venv had **torchvision 0.27.0 against torch 2.5.1** (`operator
torchvision::nms does not exist`), so torchvision was pinned back to the matching
**0.20.1+cpu**; then `open_clip_torch==2.24.0` + `timm==1.0.11`. faiss (MKL
`libiomp5md`) and timm/torchvision (LLVM `libomp140`) then collide on OpenMP, needing
`KMP_DUPLICATE_LIB_OK=TRUE`. That flag is documented as possibly producing silent
wrong results, so it was **verified rather than trusted**: the crop scores computed
inside the faiss-importing build script (0.378 / 0.360 / 0.293) match the
faiss-free direct computation (0.3779 / 0.3601 / 0.2930) exactly.

### Before/after — Phase-2 smoke query, all 57 figure+table crops

Query: *"How did SARS spread geographically over time?"* (`scripts/compare_image_encoders.py`)

| metric | general CLIP | BiomedCLIP | change |
|---|---|---|---|
| min | 0.1896 | −0.0342 | — |
| mean | 0.2263 | 0.1637 | |
| max | 0.2504 | 0.3779 | 1.5x |
| **sigma (spread)** | **0.0128** | **0.0859** | **6.7x** |
| **range (max−min)** | **0.0608** | **0.4121** | **6.8x** |
| **#1 vs #2 margin** | **0.0003** | **0.0178** | **51.6x** |
| top-1 z-score | 1.88σ | 2.49σ | 1.3x |

**The distribution does spread out, decisively — the Phase-2 headline limitation is
fixed.** General CLIP compressed all 57 crops into a 0.06-wide band and separated
#1 from #2 by 0.0003 (a coin toss). BiomedCLIP spans 0.41, and its floor goes
**negative** — it can now represent "this crop is unrelated", which general CLIP
structurally could not (its minimum was 0.19).

**But the Phase-2 gold crop drops from rank 1 to rank 10 — and that is a
mislabelling, not a regression.** Ranks on the SARS page:

| crop | what it actually is | CLIP | BiomedCLIP |
|---|---|---|---|
| `:r10` | six-panel **map of Hong Kong**, clusters + SD ellipses ("Figure 6. Extent and trend of spatial spread") | #8 (0.2423) | **#1 (0.3779)** |
| `:r9` | Table 4, index of spatial spread, nearest-neighbour | #14 (0.2365) | #8 (0.2714) |
| `:r11` | *(Phase-2 gold)* **scatter plot**, no. of SARS patients vs R-index | #1 (0.2504) | #10 (0.2611) |

Inspecting the crops settles it: `:r10` is the geographic-spread figure, `:r11` is a
statistical scatter plot. For *"spread **geographically** over time"*, `:r10` is the
right answer and BiomedCLIP picks it by a clear margin, while CLIP's rank-1 on
`:r11` was luck at 0.0003. Both crops are in fact gold for **different** gold-set
questions (q040→`:r10`, q041→`:r11`); the Phase-2 probe wording matches q040. Note
also that BiomedCLIP barely moved the gold crop's own score (0.2504→0.2611) — what
changed is that it **re-scored the field**.

### Image path alone, over the 9 image-dependent gold questions

One probe cannot settle an encoder swap, so the same comparison was run over every
gold question whose answer lives *in a crop* (image similarity only — no caption
anchor, no text index, no LLM):

| qid | category | gold rank CLIP | gold rank BiomedCLIP |
|---|---|---|---|
| q038 | figure | 1 | 1 |
| q039 | figure | 2 | **1** |
| q040 | figure | 2 | **1** |
| q041 | figure | 2 | 2 |
| q042 | figure | 1 | 1 |
| q060 | figure_value | 10 | **7** |
| q061 | figure_value | 9 | 15 |
| q062 | figure_value | 6 | 18 |
| q063 | figure_value | 26 | **7** |

| metric (image path only, n=9) | CLIP | BiomedCLIP |
|---|---|---|
| recall@1 | 0.22 | **0.44** |
| recall@5 | 0.56 | 0.56 |
| MRR | 0.44 | **0.55** |
| median rank | 2.00 | 2.00 |

**Split verdict, honestly.** On the 5 true `figure` questions BiomedCLIP is clearly
better — 4/5 now rank the gold crop **first** (was 3/5 at rank 1, rest at 2), and
recall@1 across all 9 doubles. On the 4 `figure_value` (table-value) questions it is
**erratic and no better**: two improve (q063 26→7, q060 10→7), two degrade
(q062 6→18, q061 9→15), and **none reach top-5 under either encoder**.

**Honest caveats.**
1. **BiomedCLIP pulls hard toward text-dense table crops.** 8 of its top-10 on the
   smoke query are tables (CLIP: 4), and its #2 is an unrelated cancer-cohort table
   at 0.3601 — nearly the correct figure's score. Higher spread means *confident*,
   not *always right*: it is now capable of being confidently wrong.
2. n is small (57 crops, 9 image-dependent questions, 4 of them `figure_value`).
   These are directional, not significant.
3. recall@5 is **unchanged** (0.56). The gain is concentrated at rank 1 — a ranking
   improvement, not a coverage one.
4. Still CPU (torch 2.5.1+cpu); ~2 min to embed 57 crops.

**What this does and does not buy the pipeline.** It fixes the Phase-2 finding that
image ranking was a coin toss, and it improves top-1 figure retrieval. It does
**not** address `figure_value`: those questions need the crop's *contents read*, and
no amount of retrieval ranking extracts "26.0 (8.6)" from a table image — which is
precisely the Phase-8 (VQA) hypothesis, now with evidence that retrieval is not the
bottleneck there. The `caption-anchor` path also remains the primary figure route
(Phase 6 ablation: figure R@5 0.50→1.00 from captions alone), so this swap upgrades
the *secondary* visual signal.

**Not yet done:** the full 6-config ablation has **not** been re-run under
BiomedCLIP (would cost ~$2.4 and change only the `+clip` step onward). Runnable via
`python -m scripts.ablate --image-model microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`.
The published `reports/ablation.md` remains the general-CLIP result.

---

## 2026-07-29 — Phase 7.5: making "config-driven" true (closes limitation #6)

Phase 7 exposed that the platform's central claim was aspirational: `configs/*.yaml`
were read by **no code at all**. This phase wires the loader in. Nothing was
rebuilt — the retriever, the `IMAGE_ENCODERS` registry, the existing indices and the
ablation runner are untouched; they are now *fed from* YAML instead of from literals.

**The loader.** `AppConfig.from_yaml` is implemented: it resolves `extends:`
recursively (relative to the child's own directory), deep-merges so a child overrides
only the keys it names, makes every path absolute against the repo root (so the
process's working directory is irrelevant), and then **validates**. Validation is
strict on purpose — a config that *looks* applied but isn't is worse than a crash:

| bad YAML | result |
|---|---|
| `flags: {use_reranking: true}` (typo) | `ValueError: unknown key(s) ['use_reranking'] in section 'flags'` |
| `image_embedding_model: clip-ViT-L-14` | `ValueError: … is not registered. Known: [...]` |
| `abstain_min_score: 1.7` | `ValueError: must be in [0,1]` |
| `a.yaml extends b.yaml extends a.yaml` | `ValueError: circular extends: …` |
| `extends: nonexistent.yaml` | `FileNotFoundError` naming both files |

**Entry points rerouted.** `scripts/ablate.py`, `scripts/evaluate.py` and
`scripts/build_indices.py` now take `--config` and read *everything* from it: paths,
flags, image encoder (and therefore which image index is loaded), text/rerank models,
device, batch size, retrieval knobs, abstention threshold, gold-set path, reports
directory. `build_harness(cfg)` is the single construction path; the per-model CLI
knobs are gone. `AppConfig.describe()` prints the resolved settings at the top of
every run, so a report can never disagree with what actually executed.

Two derived-path helpers keep layout knowledge in one place: `text_index_path()`
(chunker flag selects `text_layout` vs `text_naive` — it selects, it does not
rebuild) and `image_index_path()` (basename follows the encoder registry).

**Three drifts found and reconciled.** Because the YAML was dead, it had silently
diverged from what every run actually did:

1. `retrieval.top_k` / `rerank_top_k` — YAML and `config.py` said **8/5**; the real
   runs used **10/10** (hard-coded in `ablate.py`). Set to **10/10** so
   `reports/ablation.md` stays reproducible. *The published numbers were produced at
   10/10, not the 8/5 the config claimed.*
2. `baseline.yaml` said `allow_abstain: false`; every baseline run had it **true**
   (abstention is measured, not ablated). Corrected.
3. `enhanced.yaml` was missing `use_caption_anchor` entirely, though every enhanced
   run had it on — the omission was invisible while nothing read the file.

Also corrected the `config.py` docstring's claim that artifacts are namespaced by
config name. They are not, and should not be: baseline and enhanced deliberately
share one chunk DB and one index dir so both read **identical source chunks**, which
is what makes the comparison fair. The flags select among co-resident indices.

### Proof: one YAML value, no code edit

`scripts/config_demo.py` loads a config, prints resolved settings, builds the
retriever from that config alone and runs fixed probes. Retrieval only, so it is free
to re-run. Baseline run = `configs/enhanced.yaml` as committed.

**Edit 1 — `models.image_embedding_model: clip-ViT-B-32` → `microsoft/BiomedCLIP-…`
(one line in `default.yaml`).** Diff of the two runs:

```
- image encoder   : clip-ViT-B-32
- image index     : artifacts/dev/index/image
+ image encoder   : microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
+ image index     : artifacts/dev/index/image_biomedclip
  Q: How did SARS spread geographically over time?
-    5. -5.3188  val00000:415624:r10  [figure] via caption-link
+    5. -5.3188  val00000:415624:r10  [figure] via caption-link,clip-image
  Q: What was the mean age in years of the study cohort?
-    4. -7.8732  val00000:415197:r2   [text ] via text
+    4. -6.9323  val00000:415312:r14  [table] via clip-image
```

The encoder changed, **the index it reads changed with it** (derived, not restated),
and retrieval output changed: the SARS map crop now also arrives via `clip-image`
(BiomedCLIP finds it; general CLIP did not), and on the mean-age question a table
crop enters the top-5 via `clip-image`, displacing a text hit.

**Edit 2 — `flags.use_rerank: true` → `false` (one line in `enhanced.yaml`).** The
entire ranking changes character, scores included (cross-encoder logits → cosine):

```
- flags ON : use_layout_chunking, use_clip_images, use_caption_anchor, use_kg, use_rerank, allow_abstain
+ flags ON : use_layout_chunking, use_clip_images, use_caption_anchor, use_kg, allow_abstain
- 1.  3.0667  val00000:415624:r1   [text]     2.  2.1765  …:r2   3. -5.2390  …:r3
+ 1.  0.7126  val00000:415624:r2   [text]     2.  0.6805  …:r0   3.  0.6655  …:r1
```

Both YAML files were reverted after the demo.

**The paid path too.** `python -m scripts.ablate --config <smoke>.yaml --limit 3` ran
all six ablation steps end-to-end from YAML for **$0.14**. The smoke config overrode
only `paths.reports_dir`, and that alone redirected all six `perq_*.jsonl` into a
scratch directory — the published `reports/` files were untouched, which is itself
evidence that the path config is live.

**Tests.** `tests/test_config.py` is implemented (was a skipped skeleton): 10 tests
covering `extends:` inheritance, two-level chains, baseline-vs-enhanced differing
*only* in flags, index paths following encoder/chunker, absolute paths, and all three
rejection cases. `11 passed, 1 skipped` (the remaining skip is `test_metrics.py`, an
untouched skeleton from the eval phase — still outstanding).

**Limitation #6 is now closed:** editing a YAML changes behaviour, and the claim
"one config-driven pipeline, baseline vs enhanced differ only by flags" is
mechanically enforced by a test rather than asserted in prose. What is *not* claimed:
`backends.llm` / `backends.vector_store` are read and validated but only one
implementation of each is registered, so those particular swaps are unexercised —
`graph` (networkx/neo4j) and `image_embedding_model` are the two that genuinely
switch implementations today.

---

## 2026-07-29 — Phase 8: VQA on figure_value (image-grounded answering)

BiomedCLIP became the default image encoder — a one-line YAML change in
`default.yaml`, now that Phase 7.5 made that mean something. `configs/enhanced_clip.yaml`
pins general CLIP so the published `reports/ablation.md` stays regenerable;
`enhanced_biomedclip.yaml` was deleted as a no-op duplicate of the new default.

**The VQA path.** `LLMClient.answer_with_images(system, user, image_paths)` added to
the interface — deliberately **not** abstract, with a `supports_images` property, so a
text-only backend stays a valid `LLMClient` and callers degrade instead of crashing.
`OpenAIClient` implements it by inlining crops as base64 data URLs at
`detail: "high"` (these are dense tables; downsampling loses the digits). Gated by
`flags.use_vqa`, model by `generation.vqa_model`, budget by
`generation.vqa_max_images` — all from YAML. `configs/enhanced_vqa.yaml` turns it on.

**Trigger, and why it is not the gold label.** VQA fires when `use_vqa` is set and the
retrieved context contains a crop. There is deliberately **no** test for
"is this a figure_value question" — the pipeline does not know a question's category
at inference time, and gating on it would be label leakage dressed up as a feature.
"A crop was retrieved" is the only honest signal available.

**Method.** `scripts/vqa_figure_values.py` retrieves once per question through the
full configured path (layout text + caption-anchor + BiomedCLIP + KG + rerank), then
answers **twice off that same retrieval** — once with `use_vqa` off (OCR text only),
once on (crop image attached). The delta is therefore attributable to VQA and nothing
else. Crops are never hand-fed; whatever retrieval returned is what the model sees.

### Per-question results (n=4, gpt-4o vision, $0.06)

| qid | expected | VQA read | strict | point est. |
|---|---|---|---|---|
| q060 mean age | 26.0 (8.6) | *abstained* | ✗ | ✗ |
| q061 lung-cancer URR | 1.20 (1.11–1.29) | **3.95 (1.56–9.98)** — off the WRONG table | ✗ | ✗ |
| q062 well water tested | 75.5% (71.0–80.0) | 75.5% (no CI given) | ✗ | ✓ |
| q063 brain-tumor SMR | 1.4 (0.9–2.0) | 1.4 (95% CI: 0.9–2.0) | ✓ | ✓ |

Two graders are reported because several `expected_answer` strings carry a 95% CI the
question never asks for. **Strict** = every number reproduced in order; **point est.**
= the point estimate only. q062 asked "what percentage…" and got 75.5% — right about
what was asked, so point-estimate is the fair reading there; scoring it as a misread
would overstate VQA's error rate.

| | strict | point est. |
|---|---|---|
| gold crop retrieved | **2/4** | |
| TEXT-only correct | 1/4 | 2/4 |
| VQA correct | 1/4 | 2/4 |
| VQA **where the gold crop was retrieved** | 1/2 | **2/2** |

### Honest reading — three separate findings

**1. Retrieval, not reading, is the bottleneck.** The gold crop was retrieved for only
2 of 4. For q060/q061 it was **absent from the candidate pool entirely** (pool ranks
`[None, None, 1, 1]`) — not truncated below `top_k`, genuinely not found. Diagnosis
from the text index directly: the gold table crops rank **31st** (q060) and **19th**
(q061) of 50. Cause: **garbled table OCR produces a weak embedding**. q061's gold
table OCR reads *"All studies Cske, gis, aluminum Exclusions URR (95% C) pletja p …
1.20 (1.111.29)"* and scores **0.536** against the query, while clean prose paragraphs
on the same topic score **0.693** — the table loses to the surrounding text. Its
caption ("Table 4. Investigating the dependence of mean URR on high exposures") names
neither lung cancer nor PAH, so the caption-anchor route cannot rescue it either, and
BiomedCLIP had it at rank 15 (Phase 7). All three retrieval routes miss the same crop
for the same underlying reason. q060 adds a second cause: *"the mean age of the study
cohort"* is under-specified across a 50-page corpus where many papers describe cohorts.

**2. VQA did NOT beat the text path on these four.** 1/4 strict and 2/4 point-estimate
for both. The Phase-2 prediction — that table values need VQA because OCR destroys the
grid — is only **partly** borne out: for q062/q063 the values survived OCR adjacent to
their labels (*"75.5 (71.080.0)"*, *"1.4 (0.92.0)"*) and gpt-4o reconstructed them from
text alone, including inferring the missing CI hyphen. The case where VQA should win —
OCR mis-pairing a row with the wrong column — is not represented in these four
questions. **Claiming a VQA win here would not be supported by the data.**

**3. VQA's one behavioural change was a REGRESSION, and it is the important result.**
On q061 the text path **abstained** (correctly — it lacked the value); the VQA path
confidently answered **3.95 (1.56–9.98)**, read accurately off `val00000:415497:r17`,
a table from a *different paper* that retrieval supplied instead of the gold crop. So:
attaching an image converts a retrieval miss from a safe abstention into a confident,
fluent, well-cited wrong answer. The vision model will read *a* number off whatever
crop it is handed. This is the sharpest risk the multimodal path introduces, and it
argues that VQA must be **coupled to a crop-relevance gate**, not enabled
unconditionally.

**Framing.** n=4. This is a **capability demonstration — that the image path works
end-to-end from retrieval through to a value read off a rendered table — not a
statistical claim.** No aggregate figure_value score should be quoted from it. The
useful output of this phase is the three findings above, especially #1 and #3.

**Not re-run:** the full 6-config ablation. BiomedCLIP + VQA are now the default
config (`configs/enhanced_vqa.yaml`) for the single final run on the 500-page set.

**Backlog added.** (a) **Table-crop retrievability** is the top blocker for
figure_value — candidate fixes: index the caption *plus* its neighbouring prose as the
crop's retrieval text, or index a VQA-read transcription of each table so the crop
becomes findable by its actual contents. (b) A **crop-relevance gate** before VQA, per
finding #3. (c) The `figure_value` gold `expected_answer` strings should record the CI
separately from the point estimate so grading needs no two-grader workaround.

---

## 2026-07-30 — Phase 9: making figure_value work (part A) and safe (part B)

Scoped fix for the two Phase-8 findings. **Both fixes work, but neither worked as
originally specified** — the specified mechanisms failed for measurable reasons, and
what replaced them is a better answer than the plan. Both failures are recorded here
because the reasons are the interesting part.

### Part A — retrievability. The specified fix FAILED; provenance fixed it.

**Specified:** VQA-transcribe each table crop at ingest, embed the transcription as the
table's retrieval text. Done: `scripts/transcribe_tables.py`, 44/45 tables transcribed
for **$0.43** (one crop returned `NO_TABLE`), cached in a new `chunks.vqa_text` column,
built into a separate `index/text_layout_tablevqa` so `text_layout` stays the published
baseline.

**It made retrieval WORSE.** Gold table rank in the text index: q060 **31 to 40**,
q061 **19 to 36**. Cosine against the question actually *fell*, q061 **0.536 to 0.459**.

**Why — and this is the finding.** A table states values; it does not state the
concepts that make those values relevant. The q061 gold table contains **none** of
*"lung"*, *"PAH"*, *"polycyclic"*, *"relative risk"* — its rows are exclusion
thresholds and its columns are URRs. Those linking words live in the page's **prose**.
Cleaning the OCR made the text *more* purely tabular, therefore *less* matchable to a
natural-language question: the garbled OCR at least contained accidental word-like
tokens. Competing prose on the same topic scores **0.693** against the query while the
best possible table text scores **0.579** — so **no improvement to a table's own text
can rank it above the prose that discusses it.** The three retrieval routes all fail
for one shared reason, which is why none of them rescued q061 in Phase 8.

**What worked: reach the crop through its page, not through its own text.** New flag
`use_page_crop_expansion` — for any page whose prose was retrieved, add that page's
crops to the candidate pool, inheriting the page's best text score (so a crop never
outranks its own page). Sound here because page == paper in this corpus and gold
questions are within-page, so this never leaves the answering document.

Two supporting changes: the table's indexed text is now **caption + transcription**
(the caption carries the query's vocabulary — q061 0.579 vs OCR 0.536 vs transcription
alone 0.459), and the reranker now scores a crop on caption + transcription instead of
its linearised OCR, which had been demoting the very crops page-expansion recovered.

| gold table crop reaches… | phase 8 (OCR) | + caption+transcription | + page expansion |
|---|---|---|---|
| the candidate pool | **2/4** | 2/4 | **4/4** |
| the final top-k the answerer sees | **2/4** | 2/4 | **4/4** |

Per question, rank in final top-k: q060 (absent) to **1**, q061 (absent) to **6**,
q062 1 to 1, q063 1 to 1. Transcription contributed **nothing** to retrievability on
its own; page expansion is the entire gain. The transcription is still worth its $0.43
— it is what makes the safety gate's scoring meaningful — but it is not the retrieval
fix.

### Part B — safety. A score threshold provably cannot work; provenance can.

**Specified:** a config-set retrieval-score threshold; below it, abstain rather than
read the crop. Implemented and measured — **it fails, and it cannot be made to work by
tuning.** On q061:

| crop | what it is | crop-vs-question cosine |
|---|---|---|
| `val00000:415497:r17` | a similar table from **another paper** | **0.5534** |
| `val00000:415199:r10` | **the correct gold table** | **0.4589** |

**The ordering is inverted — the wrong crop out-scores the right one** — for exactly
the Part-A reason: the right table's text carries values, not vocabulary. Any
threshold admitting the correct crop also admits the wrong one. At 0.55 the gate
blocked the correct table and passed the impostor: strictly worse than no gate.

**The gate that works keys on provenance:** a crop may be read only if its page is one
whose **prose** the question matched. The score is demoted to picking among eligible
crops, with a low floor (`vqa_min_crop_score`, now 0.30) to skip junk; `<= 0` disables
gating for A/B runs. Rationale: a table's relevance is established by the page that
discusses it, not by its own tokens.

Re-test, q061, retrieval held identical within each block:

| | gate OFF | gate ON |
|---|---|---|
| **phase-8 retrieval** (gold crop absent) | reads 2 other-paper crops | both **BLOCKED — other paper**, so **abstains** |
| **phase-9 retrieval** (gold crop present) | reads the 2 higher-ranked **wrong** crops, answers **"3.95 (1.56–9.98)"** | impostors blocked, reads **only** the gold table |

The gate is doing exactly its job: it refuses to read another paper's figure, and when
the right crop is present it selects it even though the impostors score higher.

### figure_value end-to-end, phase 8 to phase 9 (n=4)

| | phase 8 | phase 9 |
|---|---|---|
| gold crop retrieved | 2/4 | **4/4** |
| TEXT-only correct (strict / point est.) | 1/4 / 2/4 | **2/4 / 3/4** |
| VQA correct (strict / point est.) | 1/4 / 2/4 | **2/4 / 3/4** |

q060 went from *abstain* to correct (`26.0 (SD 8.6)`) purely because page expansion
retrieved its table. q062 (point estimate) and q063 remain correct.

**q061 is still wrong, and the honest detail matters.** VQA now reads the *right*
table and returns `3.49 (2.12–5.90)` — but the table's real cells are
`No exclusions 1.20 (1.11–1.29)`, `>40 ug/m3 3.46 (2.03–5.90)`,
`>80 ug/m3 4.69 (1.99–21.12)`. So it took the `>40` row, labelled it `>80`, and
drifted the digits (3.46 to 3.49, 2.03 to 2.05/2.12 across runs). **The error changed
character — from reading the wrong paper to misreading a cell in the right table** —
which is the improvement the gate was for, but it is still an error. Dense
multi-column tables are where the vision model slips, precisely as predicted.

A gold-set problem also surfaced: q061 asks for *"the highest cumulative PAH exposure
category"*, but that table has **no** exposure-category rows — its rows are exclusion
thresholds. The question does not match its own source table, so the model cannot
answer it correctly by any route. Logged for the gold-set pass, not patched here.

### Two caveats worth their own lines

**The transcription itself hallucinates digits.** Checked cell-by-cell against
`val00000_409598_r5.png`: the transcription gets the row/column **structure** right and
the q060 target row right (`Age (years) | 4320 | 26.0 (8.6)`) but invents values
elsewhere — wrote `Life births | 1900 | 2.7 (1.1)` where the table says
`1910 | 1.7 (0.8)`, and `BMI | 4315 | 22.3 (3.1)` where it says `4309 | 23.3 (4.1)`.
So OCR has the right digits in the wrong structure, and the transcription the right
structure with some wrong digits. Consequence, enforced in code: the transcription is
used to **find and score** a table and is **never quoted as the source of a value** —
`_block` deliberately still shows OCR text, and the attached image is the only
authority for a number.

**gpt-4o is not deterministic at temperature 0.** The same q061 input produced an
abstention on one run and `3.95 (1.56–9.98)` on another, and the misread CI varied
between `2.05` and `2.12` across runs. Single-question before/afters in this log should
be read as illustrative of a mechanism, not as reproducible fixtures.

**Framing.** n=4. Capability + safety demonstration, **not** a statistical claim; no
aggregate figure_value score should be quoted from it. The transferable results are the
two mechanism findings: *a table cannot be retrieved by its own contents*, and
*crop-relevance cannot be gated by score because the score ordering is inverted* —
both fixed by using page provenance instead.

**Config.** `configs/enhanced_vqa.yaml` now carries `use_vqa`, `use_table_vqa_text` and
`use_page_crop_expansion`; the gate lives in `generation.vqa_min_crop_score`. This is
the default for the 500-page run. Full ablation still not re-run.

**Future work (not done):** split point estimate from CI in the gold `expected_answer`
(backlog (c), deferred); fix q061's question wording; consider reading a
crop's target row via a second, row-targeted vision call to reduce cell misreads.

---

## 2026-07-31 — Phase 10: the 500-page scale-up (part A — corpus, indices, schema check)

Ten times the dev corpus, to find out which of Phases 0–9's results were properties of
the pipeline and which were properties of fifty unusually convenient pages. Several were
the latter. The pipeline itself was not changed to get here: `configs/scale500.yaml`
differs from the phase-9 default in exactly two sections — which pages the corpus is, and
where its derived artifacts live — plus one deliberate flag, and a test enforces that.

### Two pre-checks

**`use_page_crop_expansion` is keyed on PAGE, not paper** — confirmed, no fix needed:
`_add_page_crop_hits` groups by `chunk.page_id` and calls `get_by_page`, which is
`WHERE page_id = ?`. Two things were added so it stays that way. `get_by_page` is now on
the `ChunkStore` ABC (the retriever depended on a method the interface did not declare),
and `tests/test_page_expansion.py` builds the corpus we do **not** have yet — one paper,
two pages, a crop on each — and asserts the expansion stays on the page whose prose
matched. Its stub store answers a lookup by page id *or* paper id, so a paper-keyed
regression passes silently without that test and fails with it. This matters only for a
future real multi-page corpus, where paper-keying would drag every crop of a twenty-page
document into the pool on one prose hit.

**q061 is on the cleanup list** (`domain_packs/biomed/gold/CLEANUP.md`), recorded as
defect class `question_table_mismatch` — its question names an exposure-category row that
its source table does not contain, so no route can answer it. The 55-question set is
**not** retroactively edited, since `reports/ablation.md` was measured on it as it stands;
the list is the exclusion spec for the next build. gpt-4o's non-determinism at
temperature 0 is recorded alongside it, because that is what makes every q061 result in
Phases 8–9 evidence about a *mechanism* rather than a reproducible fixture.

### The corpus

500 pages — 100 from each of validation shards 0–4, spread rather than taken deep from
one, because PubLayNet stores pages in COCO-id order and consecutive rows often come from
the same document. val00000's first 100 pages are a superset of the dev 50, so every dev
finding remains present and re-checkable. **5,744 chunks in 119.7 minutes** of CPU OCR.

| region | dev (50pp) | scale (500pp) | per page |
|---|---:|---:|---:|
| text | 547 | 4,415 | 8.8 (dev 10.9) |
| title | 57 | 764 | 1.5 |
| table | 45 | 240 | 0.48 (dev 0.90) |
| figure | 12 | 170 | 0.34 (dev 0.24) |
| list | 21 | 155 | 0.31 |

The corpus is declared in the YAML (a new validated `ingest` section) rather than passed
on a command line: the corpus is part of the experiment. Shard tags prefix every
`page_uid`, so duplicate tags are rejected at config-load time — two shards sharing a tag
would silently collide pages from different files into one id space. Ingest is resumable
(pages already stored are skipped but still counted against the per-shard limit, so a
resumed run covers exactly the same page set as a fresh one), which matters because OCR
is the one genuinely slow, non-idempotent-in-cost step.

> **Limitation — title OCR degrades at scale.** Title regions are **20.4% empty** here
> against **0%** in dev (text 0.7%, table 0.8%, list 0%; overall 3.4% empty, 0.3%
> low-quality). Titles are short, and RapidOCR misses short isolated strings more often
> than paragraphs. `LayoutChunker` prepends the section title to a region's text, so a
> fifth of layout units lose that context signal. Not fixed — recorded as an honest scale
> finding. It mildly weakens layout chunking, which was the single biggest retrieval lever
> in the Phase-6 ablation, so it is a plausible source of any softening in the +layout step
> at this scale.

### Derived indices, and a caption-coverage regression that the scale-up exposed

`text_layout` 5,611 units · `text_naive` 1,818 windows · `image_biomedclip` 408 crops.

**Caption-link coverage fell from 95% (dev) to 71%.** This is not cosmetic: the Phase-6
ablation showed the caption→crop path is *the* figure retrieval win (figure R@5
0.50→1.00 from captions alone), so 29% of crops arriving without a caption would have
weakened the figure numbers for a reason that has nothing to do with the pipeline being
tested. Diagnosis of all 119 unlinked crops:

| cause | n | recoverable |
|---|---:|---|
| no caption-like region on the page at all | 76 | no |
| OCR-mangled label prefix | 23 | yes |
| rejected by the zero-horizontal-overlap penalty | 20 | yes |

Both recoverable causes were fixed, because paying for an ablation on artificially weak
coverage risks paying for it twice.

*The overlap penalty could never link an adjacent-column caption.* A caption with zero
horizontal overlap took a flat 500 against a `_MAX_PENALTY` of 400, so in a two-column
layout a crop and its own caption could not link **even at zero vertical gap**. It is now
`250 + horizontal distance`, so an adjacent column links and a caption across the page
still loses.

*The label matcher missed one journal's small-caps style.* RapidOCR renders it
`FiGurz`, `FiGukz`, `FiGuke`, `FiGvre`, `Tablle` — common from val00001 on, invisible in
dev. A fuzzy first-token match (similarity >= 0.65 against figure/fig/table/tab) recovers
them; that threshold admits `figukz`→figure at 0.67 while figure and table stay 0.36
apart, so the fallback cannot confuse the two types. It runs **only where the strict
prefix found no candidate**, so pages that already linked are unaffected by construction.
No trailing-number requirement: the SARS caption OCRs as *"Figure Spatial clusters …"*
with the numeral destroyed, and it is a known-good link.

| | before | after |
|---|---|---|
| figures | 108/170 (64%) | **140/170 (82%)** |
| tables | 183/240 (76%) | **209/240 (87%)** |
| overall | 291/410 (71%) | **349/410 (85%)** |

**Strictly additive, and verified as such: 291 → 349 links, 0 changed, 0 lost.** Linking
was also made idempotent (`clear_captions` before re-linking) — `set_captions` only writes
the pairs it is given, so before this a re-run kept links an earlier matcher had made and
the reported coverage described no matcher that actually existed.

> The remaining **76 crops have no caption region on the page at all** — the caption was
> never emitted as a text region, or OCR destroyed it beyond recognition. Logged as a
> genuine limitation and deliberately not chased: recovering them would mean inventing a
> caption, and a crop with no caption is exactly the case the page-crop expansion route
> exists to serve.

**Continuity after the change** — the FAISS indices do not depend on `caption_id` (no
chunker reads it), so re-linking is a database operation and the indices stay valid. Both
smoke tests reproduce exactly: text top-5 **0.713 / 0.681 / 0.665 / 0.627 / 0.622**
(identical to Phase 2) and BiomedCLIP top-1 `val00000:415624:r10` at **0.378** (Phase 7:
0.3779). With caption-anchor and page-crop on, both SARS figure crops now arrive via
`caption-link` where Phase 4 surfaced them at a caption-inherited ~0.62.

### Table transcription excluded — on the evidence, not the budget

The transcription pass is **not** run at this scale and `use_table_vqa_text` is off. Phase
9 measured the transcription as contributing *nothing* to retrievability (gold table rank
31→40 and 19→36; cosine 0.536→0.459), because a table states values, not the concepts
that link them to a question. Page-crop expansion was the entire gain and remains on.
Paying $2.33 to rebuild a component measured as redundant would buy nothing, so this is an
evidence-based exclusion and a report point in its own right: *table transcription was
tested and excluded because it added nothing to retrieval.*

Its one remaining role was sharpening the VQA gate's crop scoring, so that was **measured
rather than assumed** before dropping it (`reports/vqa_gate_check.md`). With no
transcription the gate scores a table crop on its own OCR text and a figure crop on its
caption — retrieval held identical, the gate applied twice to the same result:

| | gold crop passes the gate | gold crop actually read |
|---|---|---|
| with transcription | 4/4 | 4/4 |
| without (this corpus) | 4/4 | 4/4 |

The gate is unaffected, for the reason it was built that way: its primary condition is
**provenance**, which does not depend on the transcription at all, and the 0.30 score floor
never binds (every page-eligible crop scored >= 0.53). One selection changed, benignly — on
q062 the second crop read flips to one on the *same page* as the gold crop.

> **The q061 score inversion reproduces on raw OCR.** With the transcription the wrong
> crop scored 0.5534 against the right one's 0.4589; without it, 0.5492 against 0.5360.
> The ordering is still inverted, the margin merely narrows. So Phase 9's "no threshold
> can separate them, because the score ordering is inverted" is **not** an artefact of the
> transcription — it holds on the raw OCR too. That strengthens the provenance-gate
> argument rather than weakening it.

### HEADLINE FINDING — the closed schema does not generalise to a broader corpus

Phase 3 consolidated the 8-predicate schema from a free extraction over 30 dev chunks. The
generalisation check re-runs *that exact protocol* — 30 chunks, text/list/table, >=200
chars, strided — on the 500-page corpus (26 of the 30 from pages outside the dev 50), then
asks an LLM to map each discovered predicate onto the closed 8 or to NONE. **86 triples, 73
distinct predicates, $0.0053.**

**The closed 8 cover 22% of triples and 23% of distinct predicates.** New relation types
appear; the 8 do not dominate. Two unmapped types are recurrent and substantive:

- **`is_a_predictor_of` (x4)** — prediction/association. This is precisely the space Phase
  3 *deliberately vacated* when it dropped `associated_with` as a vague catch-all that
  over-connects the graph. The corpus keeps asking for it back.
- **`has_values_of` / `has_levels_of` (x5)** — "this entity has this measured value". The
  schema has `measured_by` (the *method*) and nothing for the *value*.

The rest of the unmapped tail explains itself on inspection: *IASP established Global Year
Against Pain*, *robot asks human information about the object*, *variations in esthetic
norms do_not_hinder perception of smile attractiveness*. **The 500-page corpus is
topically far wider than the dev 50** — robotics, dentistry, pain-policy editorials — while
the 8 predicates were consolidated from a toxicology/epidemiology slice. The schema
captures the causal/quantitative biomedical relations it was designed for, and is
**domain-slice-specific, not universal.**

**This is the closed-schema GraphRAG tension, and naming it is the finding.** A closed
schema buys sparsity, comparability and precision — the dev graph's maximum entity degree
was 7, with no hub explosion, which is exactly what dropping `associated_with` and
`compared_with` achieved. It pays for that in coverage, and the price is invisible until
the corpus broadens. An open schema pays the opposite way: Phase 3's free extraction
produced 46 distinct predicates over 30 chunks and never reused an edge type, which is a
graph that cannot be traversed.

**This connects the schema finding to the `figure_value` finding.** The missing
value-relation is the **structural** reason Phase 6A's measurement detector found *zero*
clean table-value edges in the KG. That was read at the time as an OCR problem — tables
OCR so poorly that no numeric measurement survived into an edge — and OCR is real, but it
is not the whole cause. Even with perfect OCR, `(mean age) —has_value→ (26.0 (8.6))` has
**no predicate to be extracted into**. The graph could not represent the fact. That is why
`figure_value` had to be hand-built in Phase 6A, why it stayed flat across every ablation
config, and why it needed the visual path at all.

**The schema is NOT being changed** (decision taken explicitly). Changing it would fork
comparability with the dev KG, the gold set's per-predicate coverage and the whole
ablation, with no time to re-verify the cascade cleanly — and the finding is worth more
reported than fixed. This is exactly the "will the closed schema generalise?" risk the
design anticipated, and the generalisation check is the instrument that caught it
empirically rather than by assertion. Full output:
`reports/scale500/predicate_discovery.json`.

### Preflight code review — one confirmed defect in the new gold-set filters

Before letting new code produce the gold set that every downstream metric is computed
against, four read-only reviewers went over it under distinct lenses, each finding
adversarially verified by independent refuters. Seven of eight findings were refuted with
evidence. The one that survived was real and consequential.

The q061-mismatch filter — which rejects a question whose distinguishing terms are absent
from its source region — **also rejected roughly one in three well-formed crop-backed
questions** (q038, q039, q062 on the dev set). Two causes: a prefix rule cannot match
derivational paraphrase (a caption saying *"viable eggs"* does answer a question about
*"viability"*), and a question about a table legitimately uses framing nouns
(*"respondents"*, *"survey"*) that a table body never contains. The harm was not lost
volume; it was **selection bias**. The surviving crop-backed questions would have been
those that lexically echo their own caption — and since figure retrieval is
caption-primary, that would have inflated exactly the crop-backed retrieval numbers this
run exists to publish.

Fixed with a crude derivational stemmer and **two bands instead of one threshold**: drop
above 0.65, and between 0.40 and 0.65 *keep the item and flag it for human review* rather
than delete it. Calibrated against the real 55-question set rather than guessed — q061
scores 0.75 and the worst legitimate question 0.38 — and pinned by
`tests/test_gold_filters.py` so it cannot silently regress.

Four smaller defects found and fixed in the same pass: drop-counts were inflated ~3x
because the edge list was re-filtered per category while `clean_edge` incremented the
counters; the answer-leak guard missed short answers; the unanswerable pool was sized
exactly to its quota, so any rejection would permanently under-fill the one category whose
purpose is measuring abstention; and `last_vqa` was rebound rather than updated, destroying
the gate trace on exactly the calls where the gate *admitted* a crop.

### Cost model

Projections come from measured invoices divided by their units, not token estimates
(`scripts/estimate_cost.py`), and reproduce the dev run to the cent — 605 chunks → $0.08
against an actual $0.083; 45 tables → $0.44 against an actual $0.43. The approved plan
projects **$6.87**: KG $0.66, ablation (6 configs x 130 questions) $5.58, the separate
VQA-vs-text comparison $0.63, discovery and gold-set phrasing ~$0.01 each. Two corrections
were made to an earlier, higher figure — this corpus is less table-dense than dev (238
tables, not the ~450 a linear scaling predicted), and the classic ablation series has
`use_vqa` off in all six steps, so the ablation makes **zero** vision calls.

---

## 2026-08-01 — Phase 10 (part B): the 500-page ablation, and what changed at scale

127 human-approved gold questions, six configs, the classic series
(`baseline → +layout → +clip → +caption → +kg → +rerank`), plus a separate paired
VQA-vs-text comparison. Full tables in `reports/scale500/ablation.md`; per-question logs
in `reports/scale500/perq_<config>.jsonl`.

**For the first time in this project no category carries a low-sample flag.** The
Phase-6 set had `figure_value` at n=4 and `treats` at n=1, both footnoted as not
meaningful; here the smallest category is `figure_value` at n=8 and the thinnest
predicate is `measured_by` at n=7.

### HEADLINE FINDING — the enhancement gap WIDENS at scale

| | dev 50pp (n=55) | | scale 500pp (n=127) | |
|---|---|---|---|---|
| | baseline | enhanced | baseline | enhanced |
| R@5 | 0.69 | 0.87 | **0.57** | **0.91** |
| MRR | 0.46 | 0.83 | 0.41 | 0.88 |
| nDCG | 0.52 | 0.84 | 0.47 | 0.89 |
| Correct | 0.66 | 0.70 | 0.59 | 0.69 |

Relative retrieval gain: **R@5 +26% at 50 pages, +60% at 500. nDCG +62%, then +89%.**

The two ends move in opposite directions, and that is the result. **Baseline degrades
with corpus size** — R@5 0.69 → 0.57 — exactly as it should: naive fixed-window chunking
over ten times as many candidates surfaces the right region less often. **The enhanced
configuration does not degrade; it improves** — 0.87 → 0.91. Layout-aware units, the
caption route and cross-encoder reranking are each doing more work at 500 pages than at
50, because there is more for them to discriminate against.

This is worth stating plainly because the usual result is the opposite: gains measured on
a small corpus often shrink when the corpus grows, as the easy wins get diluted. Here the
enhancements are **not** a small-corpus artefact. If anything the 50-page numbers
understated them.

Per-step, the attribution is much the same as Phase 6 but sharper. **`+layout` remains the
single biggest lever** (R@5 0.57→0.76, MRR 0.41→0.78 — it alone recovers the entire
baseline degradation and more). **`+caption` is still the figure win** (figure R@5
0.50→1.00, nDCG→0.99). **`+kg` lifts overall correctness 0.66→0.69** and single_fact R@5
0.79→0.84 — and again does not hurt single_fact, the dilution that was feared in Phase 4
and has now failed to appear twice. **`+rerank` is the largest single retrieval step here**
(R@5 0.86→0.91, MRR 0.80→0.88), a bigger contribution than it made at 50 pages, which fits
the same story: reranking matters more when the candidate pool is deeper.

### `figure_value` moves for the first time

| config | R@5 | MRR | Correct |
|---|---|---|---|
| baseline | 0.62 | 0.28 | 0.38 |
| +caption | 0.88 | 0.60 | 0.50 |
| +rerank | 0.88 | **0.83** | **0.75** |

In the Phase-6 ablation this category was **flat at 0.50 correctness across every single
config** — nothing the pipeline did moved it, which is what motivated Phases 8 and 9. At
n=8 with human-verified answers it now responds: correctness 0.38 → 0.75, and MRR 0.28 →
0.83. The mechanism is the Phase-9 fix doing its job at scale — page-crop expansion puts
the table in the pool, and reranking on caption-plus-text lifts it to the top.

### Where it got worse, and why that is interesting

**Abstention degraded.** Unanswerable abstain-rate was **1.00 in every config** at 50
pages; here it is **0.82** for five of six configs and 0.91 at `+rerank`. Two of eleven
deliberately-unanswerable questions now get answered. This is a scale effect on the
*abstention* side that mirrors the one on the retrieval side: a 500-page corpus contains
far more near-miss material, so a question the corpus does not answer still retrieves
context that looks answerable. Note this is the abstention rate *after* the absence check
already rejected 13 of 24 candidate unanswerables for being too close to the corpus — the
surviving 11 are the hard cases by construction, so this number is measured against a
tougher set than the dev 6.

**`occurs_in` correctness collapsed: 0.80 (n=5) → 0.31 (n=16).** It is now the worst
predicate by a wide margin while its retrieval stays fine (R@5 0.81), so this is an
answering/gold problem, not a retrieval one. The KG sample makes the cause visible:
`(patients undergoing AraSns for gynecologic malignancies -occurs_in-> postoperative
patients)` is a loose, partly-garbled edge, and a question phrased from it inherits that
looseness. `occurs_in` was the predicate Phase 3 already had to tighten once (restricting
its object to a population/sample/site); at scale it remains the least well-behaved member
of the schema.

**`treats` is now measurable and mediocre: 0.60 at n=15.** Phase 6 could only report n=1
and footnoted it as meaningless. The KG makes the likely cause visible too —
`(patients -treats-> chemotherapy)` has subject and object **inverted**. Direction errors
on `treats` were invisible at 50 pages because there was one instance; with 15 they show
up as mid-range correctness. The predicate mix also shifted with the corpus: `treats` went
from 5% of dev edges to 17% here, and `transforms_to` from 6% to 2% — the broader corpus is
more clinical and less chemical, which is the same topical-breadth effect the schema
generalisation check measured.

**`multi_hop` correctness dips at `+rerank`** (0.44 → 0.35) while its retrieval jumps
(R@5 0.62 → 0.82). Retrieval and answering come apart: the reranker optimises for
query-passage relevance per chunk, which is not the same as assembling two chunks that
must be combined. The same shape appeared on `figure` in Phase 6.

### VQA vs text on the verified `figure_value` set — net zero, and the reason matters

Gold crop retrieved **8/8** — page-crop expansion delivers completely at scale, against
2/4 in Phase 8 before the fix.

| | strict | point estimate |
|---|---|---|
| TEXT-only | 4/8 | **6/8** |
| VQA | 4/8 | 5/8 |

**Attaching the crop image did not improve value reading.** Only two questions differ
between the two paths, and they cancel:

- **q123 — VQA wins, for the predicted reason.** The question asks the mean age; OCR gives
  the text path `26.0` but has lost the paired `(8.6)` standard deviation, so it answers
  `26.0 years`. The vision path reads the grid and returns `26.0 with a standard deviation
  of 8.6`. This is exactly the Phase-2 hypothesis — that VQA recovers structure OCR
  destroys — finally observed on a verified item.
- **q128 — VQA loses, for the Phase-9 reason.** Expected `19`. The text path reads `19`
  correctly. The vision path, handed **the correct crop**, reads **`2`** off it and cites
  it properly. A confident, well-cited cell misread on a dense table.

So the honest summary is that VQA's one real capability and its one real failure mode are
the same size at n=8. **No VQA win should be claimed from this data.** What can be claimed
is a precise characterisation: it recovers row-column pairings that OCR linearisation
destroys, and it misreads individual cells in dense tables — and because it misreads
*confidently and with a correct citation*, the misread is harder to detect downstream than
an OCR failure would be.

> **The provenance gate is weaker at 500 pages, and q124 shows it.** On that question the
> gold crop was retrieved but ranked 10th, and the gate passed two crops from *other
> papers* (`val00003:411764:r7`, `val00004:391918:r9`) — so the vision model answered off
> the wrong table. The gate's condition is that a crop's page must be one whose prose the
> question matched; with `top_k = 10` and page == paper, up to ten different papers clear
> that bar. At 50 pages the eligible set was effectively the answering page; at 500 it is
> not. The gate still blocks the *unrelated* crops it was built for, but its guarantee has
> genuinely weakened with corpus size, and tightening it — to the page that supplied the
> top-ranked prose, or to the page carrying the most text hits — is the obvious next step.
> Logged, not fixed, since changing it after the ablation would invalidate these numbers.

> **A gold-question defect class the mismatch filter cannot catch.** q124 asks *"What is
> the Mean ± SD for Age (years)?"* — every term appears in its source table, so the
> q061-class filter passes it, but across 500 pages dozens of tables report a mean age. The
> question is well-formed against its source and **under-specified against the corpus**.
> This is the same shape as q060 in Phase 8 (*"the mean age of the study cohort"*) and it
> is a distinct defect from q061's: not *the source lacks what the question names*, but
> *the question does not identify its source*. A filter for it would have to check
> uniqueness against the whole corpus, not just presence in the cited region.

### The table-OCR value proposals had a 53% defect rate — corroborating evidence

To fill `figure_value` (the KG yields almost nothing there — see part A), 14 candidate
questions were proposed from table OCR and rendered beside their crops for human
verification. **9 of 17 were rejected: 53% overall, 8 of 14 (57%) among the OCR-derived
proposals, 1 of 3 among the graph-derived ones.** Rejection reasons, verbatim:

| slot | reason |
|---|---|
| V01 | circular — "CYP1A inhibits CYP1A protein expression" is self-referential |
| V05 | wrong value **and** a q061-class defect: the table reports NO2, the question says NO |
| V07 | wrong value |
| V09 | unverifiable — crop too poor to read |
| V11 | unreadable crop |
| V12 | value misplaced: 26 is not e432's maximum (81 is) |
| V13 | wrong value |
| V15 | wrong value — should be 8.03 ± 0.19 |
| V17 | wrong value — drops the ×10⁻⁴ exponent |

This is not cleanup bookkeeping; it is an **independent measurement of the central claim
about table OCR**, arrived at by a different route than Phase 9's. Phase 9 concluded from
a cell-by-cell check of one transcription that OCR has *the right digits in the wrong
structure*. Here a human checked 14 independently-proposed values against their images and
found the majority wrong — and the failure modes are precisely the predicted ones:
**misplacement** (V12 takes a number from the wrong row), **truncation of structure**
(V17 drops an exponent, V15 takes the wrong statistic), and **label drift** (V05 attaches
NO2's value to NO). A pipeline that quoted table values from OCR would be wrong more
often than right on this corpus. That is the empirical case for the visual path existing
at all, and for the rule that the transcription is used only to find and score a table and
**never** to quote a value.

It also, unprompted, reproduced the q061 defect class: V05 named a species the table does
not report. That defect survived the automated mismatch filter and was caught only by
someone looking at the image — which is why the figure_value category is human-verified by
construction rather than by review of a sample.

### Two engineering findings from running at this scale

**A long paid run needs retry and resume, and had neither.** The first ablation attempt
died three configs in on an unretried HTTP 429: this account's gpt-4o limit is 30,000
tokens/minute, and an answer carrying ten retrieved chunks means the run sits against that
cap for its entire duration — a 429 is the steady state, not an anomaly. The OpenAI SDK's
default of 2 retries is not sized for that. Raised to 10 (it backs off exponentially and
honours `Retry-After`); zero rate-limit failures since.

Resume was added at the same time, because re-running would have re-bought roughly $2.50
of completed work: `--resume` reloads any `perq_<config>.jsonl` that is already complete
and re-aggregates it rather than re-calling the models. Because every metric is computed
from those per-question records, a reloaded config reproduces its report *exactly* rather
than approximating it. One bug worth recording: the first resume attempt crashed because
`recall`/`precision` are keyed by cutoff *k* as an integer, and JSON has no integer keys —
a round-trip returns `{"5": …}` and every lookup misses. Reloaded records are now coerced
back.

### Cost

| pass | $ |
|---|---|
| predicate discovery | 0.005 |
| KG build (4,776 chunks) | 0.640 |
| gold-set phrasing | 0.004 |
| figure_value proposals | 0.002 |
| ablation, 6 configs × 127 questions | ~5.0 |
| VQA-vs-text comparison | 0.126 |
| **total** | **~$5.8** |

Against a $6.87 projection, and the projection's own method — measured invoices divided by
their units — held up: the KG build came in at $0.6397 against $0.66 predicted, inside 3%.
The ablation figure is approximate for an honest reason: the first attempt crashed before
printing its cost line, so its three completed configs are estimated from the resumed
run's $2.50 for the other three.

---

## 2026-08-01 — Phase 11: chain-of-thought, as a paired side-experiment

Scoped deliberately small. The six-config scale-500 results are frozen
(`reports/scale500/RESULTS.md`), so CoT is **not** added as a seventh ablation step —
re-running the series would cost another ~$5 and put a locked result at risk for a change
that cannot affect retrieval at all. CoT is generation-side, so it is measured where it
could plausibly matter and nowhere else.

**The flag.** `use_cot` in `RetrievalFlags`, read from YAML like everything else
(`configs/enhanced_cot.yaml`). When on, `COT_CLAUSE` is appended **last** to the system
prompt — after the abstention clause, so the output-format instruction is the final thing
the model reads — asking for numbered reasoning over the retrieved passages under a
`REASONING:` marker, then the final answer alone under `ANSWER:`.

**The reasoning is stripped before anything scores the answer.** `split_cot` takes the
last `ANSWER:` marker; citation extraction, the abstention match and per-sentence grounding
all run on that section only, and the trace is kept on `last_reasoning` for explainability.
This is not tidiness — if reasoning text reached the citation regex it would count sources
the model *considered and rejected* as cited, and a response that merely discussed
insufficient evidence would be scored as an abstention. Then CoT would be changing what the
metrics mean, and the comparison would measure the parser instead of the prompt.
`tests/test_cot_split.py` pins all four of those cases, including the fallback that keeps a
malformed response scoreable rather than empty.

**Design of the comparison.** Flags come from `ablation_flags("classic")[-1]` — the
ablation's own `+rerank` step — not from a config's flags block, so the CoT-off arm is
bit-identical to a column already published. Retrieval runs **once** per question and both
arms answer from the same result, so the delta is attributable to the prompt alone. Both
arms go to the same judge. **17 multi_hop (the hypothesis) + 20 strided single_fact (the
control).** If CoT works by making the model combine passages, multi_hop should move and
the control should not.

### Result — no evidence that CoT helps

| subset | n | Correct OFF | Correct ON | delta | improved | regressed |
|---|---:|---:|---:|---:|---:|---:|
| multi_hop (hypothesis) | 17 | 0.38 | 0.41 | **+0.03** | 3 | 2 |
| single_fact (control) | 20 | 0.88 | 0.80 | **−0.07** | 1 | 3 |
| combined | 37 | 0.65 | 0.62 | −0.03 | 4 | 5 |

**The +0.03 on multi_hop is exactly the size of this pipeline's run-to-run noise, so it is
not a result.** The measurement that establishes that is free and sits in the same
artifacts: the CoT-off arm has flags *identical* to the frozen `+rerank` ablation and runs
the *same* 17 multi_hop questions, so the two are the same experiment twice.

| multi_hop, n=17, identical configuration | correctness |
|---|---|
| frozen ablation, `+rerank` | 0.353 |
| CoT-off arm, re-run | 0.382 |
| CoT-on arm | 0.412 |

Re-running an unchanged configuration moved correctness by **+0.029** — one question of 17
flipped (q062) purely from gpt-4o's non-determinism at temperature 0, the limitation logged
since Phase 9. The CoT "gain" is **+0.03**. A 17-question subset cannot separate a real
effect of that size from the noise floor, and this run measured the noise floor directly
rather than assuming it.

**The control is the more informative half, and it moved the wrong way.** single_fact fell
0.88 → 0.80. A control that moves *more* than the hypothesis — and downward — says the
prompt changed something other than reasoning quality.

### Diagnosis — CoT bought caution and breadth, and paid for it on single-span questions

Abstention rose in **both** subsets: multi_hop 0.24 → 0.29, single_fact 0.10 → 0.15. That
is the single clearest signal in the run, and it is the same direction in both, which is
why it reads as a property of the prompt rather than of the question type.

The regressions show the mechanism:

- **q037 (single_fact) — answered correctly with CoT off, abstained with CoT on.** Surveying
  the passages before committing made the model *less* willing to commit.
- **q023 (single_fact) — precision lost to breadth.** Expected "risk of PTE". Off:
  *"Blood transfusion increases the risk of postoperative pulmonary thromboembolism
  (PTE)"* — exact. On: *"…increases the risk of pulmonary thromboembolism (PTE) **and
  coronary artery disease (CAD) due to iron overload**"*, citing three chunks instead of
  one. The reasoning step surveys the retrieved set, finds more that is topically related,
  and folds it in. On a question with one correct span, a broader answer is a worse one.

The same mechanism is the upside on multi_hop, where breadth is what the question wants.
q059 asked which drugs treat breast cancer in women; the trace enumerates four passages and
the answer assembles a more complete drug list than the non-CoT arm:

```
1. [val00001:379772:r1] mentions that breast cancer treatment includes hormone therapy,
   chemotherapy, and targeted therapies, with doxorubicin being one of the drugs used.
2. [val00001:379772:r4] discusses the use of doxorubicin in combination with paclitaxel
   for metastatic breast cancer.
3. [val00002:356703:r7] and [val00002:356703:r5] mention the use of epirubicin, docetaxel,
   cyclophosphamide, capecitabine, and vinorelbine in chemotherapy for breast cancer.
4. [val00002:356703:r0] mentions the use of tamoxifen as part of endocrine therapy.
```

So CoT is not inert — it changes behaviour in a consistent and explicable direction. It is
just that the direction is *broader and more cautious*, which helps a category that needs
several passages combined and hurts one that needs a single span, and at these sample sizes
the two do not net out to anything measurable.

### What it does deliver

**Explainability, unconditionally.** Traces were captured for **37/37** answers, median 475
characters, each naming the chunk_ids it weighed and what it took from them. That is a real
product of the flag independent of the accuracy result: the trace shows which retrieved
passages the model actually used versus merely received, which the citation list alone does
not distinguish — q059's trace weighs five chunks and the answer cites five, but q062's
weighs passages it then drops.

### Honest limits of this experiment

- n=17 and n=20. Nothing here is significant, and the noise floor was measured at ±0.03 on
  the multi_hop half, which is the size of the entire observed effect.
- One prompt formulation, one model. A different CoT phrasing — particularly one that
  instructed the model to answer *only* what was asked after reasoning — might not lose the
  control. That was not tested.
- Correctness only. Faithfulness and grounding were not re-measured under CoT; the extra
  verifier calls were not worth it for a result this size.
- Cost **$0.55** against a ~$1 estimate. The frozen results were not touched.

**Verdict: do not enable `use_cot` for accuracy.** It is defensible as an explainability
feature, and the flag stays off by default (`configs/default.yaml`), opted into through
`configs/enhanced_cot.yaml`. Full output: `reports/scale500/cot_compare.md` and
`cot_compare.jsonl`.

---

## 2026-08-01 — Phase 12: human-in-the-loop review dashboard

A review and visualisation layer over the locked scale-500 run. Nothing it does can move a
published number: it reads `perq_+rerank.jsonl`, the gold set and the chunk store, and
writes exactly two new files — `review_queue.jsonl` (precomputed) and
`review_feedback.jsonl` (append-only reviewer verdicts). Every frozen artifact still
carries its original timestamp after the dashboard was built and exercised.

Two pieces: `scripts/build_review_queue.py` precomputes the queue offline, and
`scripts/review_app.py` is the Streamlit interface. Splitting them is deliberate — the
confidence score needs BGE and a retrieval pass per question, which is a two-minute model
load and ~20 minutes of CPU, and doing that at page-load would make the dashboard unusable
and would re-run retrieval every time someone opened it.

### The confidence score, and the trap in defining it

The obvious move is to reuse the retrieval metrics already sitting in the per-question log.
That would be wrong. `recall@k`, `MRR` and `nDCG` are all computed against the gold
`supporting_chunk_ids`, so a confidence built from them is a **correctness proxy in
disguise**: it would rank items by how right they are — information a deployed review queue
cannot have — and it would make the dashboard look far better than it is, because the
"low-confidence" items would be exactly the ones already known to be wrong.

So the score uses only signals the system has **at inference time**, both already produced
by the pipeline:

- **Retrieval strength `R`** — the top query-context cosine, recomputed exactly as
  `GroundedAnswerGenerator.relevance()` computes it. This is the same quantity the
  abstention code gate thresholds against `abstain_min_score`. Normalised against that
  floor, since below it the system declines to answer anyway:
  `Rn = clip((R − 0.45) / (0.85 − 0.45), 0, 1)`.
- **Grounding `G`** — the per-sentence grounded fraction from the locked run: the share of
  answer sentences whose cosine to their cited chunk cleared `grounding_min_sim`, with the
  LLM verifier deciding the borderline band.

**Answered:** `confidence = 0.5·Rn + 0.5·G`, `priority = 1 − confidence`.

**Abstained:** confidence is undefined — there is no answer to grade — so `priority = Rn`.
That inversion is the design, not an oversight: abstaining on weak retrieval is usually
correct and wastes a reviewer's time, while **abstaining on strong retrieval is the
false-abstention failure mode Phase 6D was built to catch**, and is the single most valuable
thing a human can look at.

Weights are a flat 0.5/0.5 **by choice**. Tuning them against the locked correctness column
would fit the score to this run's answers and quietly turn it back into the correctness
proxy the whole definition was designed to avoid. `tests/test_review_app.py` re-derives
both formulas from the raw fields and fails if either drifts from what the sidebar tells
the reviewer.

### What the queue found

127 items — 104 answered, 23 abstained. Sorted worst-first, the head of the queue is:

```
1.00  q075  figure        abstained     abstained despite strong retrieval
1.00  q082  figure        abstained     abstained despite strong retrieval
0.92  q078  figure        abstained     abstained despite strong retrieval
0.82  q115  unanswerable  conf 0.18     weak retrieval + ungrounded sentences
0.69  q005  single_fact   abstained     abstained despite strong retrieval
0.68  q054  single_fact   conf 0.32     ungrounded sentences
```

**10 items abstained despite strong retrieval; 8 answered with confidence below 0.50.**
The three at the top are `figure` questions, which is an independent confirmation rather
than a coincidence: the frozen ablation reports figure `Decis` at 0.75, i.e. a quarter of
answerable figure questions were wrongly abstained. A label-free score, ranking blind,
surfaced the same defect the gold-backed metric measured — which is the strongest available
evidence that the priority ordering is doing something real.

### What each item shows

Question, the answer (or an explicit abstention notice), the three bars (retrieval,
grounding, confidence) with the raw cosine, the **provenance counts** with what each route
means (`text`, `caption-link`, `graph-hop`, `clip-image`, `page-crop`), the graph predicates
walked, the VQA crop-gate decisions where the gate ran, every **cited chunk_id with its
source text pulled live from the chunk store** — flagged in red if a cited id is not in the
store — the full retrieved context behind a popover, and the **chain-of-thought reasoning
trace** where one exists (37 of 127, from Phase 11).

**The gold answer and the automated judge's verdict are hidden behind a toggle, off by
default.** A reviewer who sees the expected string first is comparing text; one who sees
only the question, the answer and the cited evidence is doing the judgement the loop exists
for — and can then reveal to check both themselves *and* the judge.

### The feedback loop

Three buttons — correct / incorrect / needs-review — plus a free-text note. Each verdict
appends a line to `review_feedback.jsonl` with the qid, reviewer, timestamp, the note, and
the confidence and abstention state at the time of review. **Append-only:** a reviewer who
changes their mind adds a line rather than overwriting one, and the reader takes the newest.
That keeps the history, which matters if the verdicts are ever used to audit the automated
judge. Recording the confidence alongside the verdict is what makes the log useful later —
it is the raw material for asking whether the confidence score actually predicts human
disagreement, which cannot be answered until enough verdicts exist.

### An honest limitation the dashboard exposed immediately

q054 sits sixth in the queue at confidence 0.32, flagged "ungrounded sentences" with
`G = 0.00` — and it is **correct**. The question asks what kit measures testosterone, the
answer says an EIA kit from Assay Designs citing `val00002:345509:r1`, and that chunk reads
*"…assayed in duplicate by EIA kit (Assay Designs, Inc.). The kit is used for the
quantitative measurement…"*. The judge scored it 1.0.

The grounding checker scores each answer sentence against the **whole** cited chunk, so a
one-line answer citing a long methods paragraph can fall below the 0.45 similarity
threshold despite being exactly supported by one sentence inside it. Grounding therefore
has a false-alarm mode on short answers citing long passages, and since it is half the
confidence score, those items are pushed up the queue. For a review tool that is a tolerable
failure — it costs a reviewer time, not correctness — but it means **confidence must not be
read as a correctness estimate**, and it identifies a concrete improvement: score a sentence
against its best-matching *sentence* in the cited chunk rather than against the chunk mean.
Not changed here; changing the grounding checker would alter the locked `faithfulness`
column.

### Verification

`tests/test_review_app.py` (6 tests): both confidence formulas re-derived from raw fields,
the queue's worst-first ordering, the label-free constraint on the scored signals, the
feedback log round-trip, its append-only semantics, and a headless render of the whole app
through Streamlit's `AppTest` asserting no exception and that the summary metrics are
present. Suite: **41 passed, 1 skipped** — the remaining skip is still `test_metrics.py`,
the untouched skeleton from the eval phase.

Run with:

```
python -m scripts.build_review_queue --config configs/scale500.yaml --step +rerank
streamlit run scripts/review_app.py -- --config configs/scale500.yaml
```

`--step` takes any column of the classic series, so the same interface reviews the baseline
run or any intermediate config against the identical confidence definition.
