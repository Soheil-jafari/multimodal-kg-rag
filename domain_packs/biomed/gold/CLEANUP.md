# Gold-set cleanup list

Known-broken items in `gold_set.jsonl`, kept as a list rather than silently patched
so the published numbers stay reproducible against the set that produced them.

**These are NOT dropped from the 55-question set.** `reports/ablation.md` was
measured on that set as it stands; deleting a question retroactively would change
published numbers. The list is the exclusion spec for the **next** gold set, and
each entry names the defect class so the builder can filter it automatically rather
than relying on a reviewer to spot it again.

## Entries

### q061 — question does not match its own source table (Phase 9)

```
q061  "What was the unadjusted relative risk (95% CI) of lung cancer in the
       highest cumulative PAH exposure category?"
      expected: 1.20 (1.11-1.29)   supporting: val00000:415199:r10
```

The cited table has **no exposure-category rows** — its rows are exclusion
thresholds (`No exclusions`, `>40 ug/m3`, `>80 ug/m3`) and its columns are URRs.
There is no "highest cumulative PAH exposure category" in it, so **no route can
answer this correctly** and the expected answer is not the answer to the question as
asked. Correctly classified as unanswerable-as-written.

Consequences to keep in mind when reading the log: every q061 result in Phases 8–9
is a measurement of a *mechanism* (which crop got retrieved, which crop the gate
allowed to be read), never of answer correctness. Phase 9's improvement on it is
real and narrower than "it now works" — the error changed from *reading another
paper's table* to *misreading a cell in the right table*.

**Defect class — `question_table_mismatch`.** The question's key qualifier names a
row/column the source table does not contain. Arises when the LLM phrases a question
from an edge whose evidence sentence discusses the table's *topic* while the value
was pulled from a different part of the grid.

*Filter for the next build:* for `figure_value` items, require the question's
distinguishing qualifier to appear in the source table's own text (OCR or
transcription). Drop the item when it does not — do not attempt to repair the
wording, since a repaired question is no longer a question about a verified value.

### Companion limitation (not a gold defect, but it shapes how to read q061)

**gpt-4o is not deterministic at temperature 0.** The same q061 input produced an
abstention on one run and `3.95 (1.56-9.98)` on another, and the misread CI varied
between `2.05` and `2.12`. Single-question before/afters are illustrative of a
mechanism, never reproducible fixtures — which is also why no aggregate
`figure_value` score is quoted from n=4.

### Defect class — `under_specified_question` (found at 500 pages, Phase 10)

```
q124  "What is the Mean ± SD for Age (years)?"        expected: 32 ± 10
q060  "What was the mean age in years of the study cohort?"   (Phase 8, same shape)
```

Every distinguishing term in the question **does** appear in its source table, so the
`question_table_mismatch` filter passes it — correctly, by its own definition. The defect
is the opposite direction: across 500 pages dozens of tables report a mean age, so the
question does not identify *which* table answers it. Well-formed against its source,
ambiguous against the corpus.

This is why q124 failed under both the text and vision paths in the Phase-10 VQA
comparison: the pipeline retrieved a mean-age table, just not the intended one.

*Filter for the next build:* presence-in-source is not enough — an item needs a
**uniqueness** check, e.g. embed the question and require its intended source region to be
the top-ranked match across the whole corpus, not merely a match. Not implemented; it
needs the text index at gold-build time, which the builder currently loads only for the
unanswerable absence check.

### Note on `question_table_mismatch` at scale

The class recurred in the Phase-10 candidate pool (slot V05: the table reports **NO2**, the
question asked about **NO**) and again **survived the automated filter** — the tokens
`NO`/`NO2` are too short and too similar for the token-overlap test to separate. It was
caught only by a person looking at the crop. Treat the automated filter as a first pass
that catches the coarse cases; the crop-backed categories still require human verification
by construction.

## Deferred (backlog, not defects)

- `figure_value` `expected_answer` strings mix the point estimate and the CI
  (`1.20 (1.11-1.29)`), forcing the two-grader workaround (strict vs point
  estimate). Splitting them into separate fields is agreed future work.
