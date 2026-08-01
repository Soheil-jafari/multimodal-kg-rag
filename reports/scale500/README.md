# 500-page scale-up — results index

Corpus: 500 PubLayNet validation pages (5 shards × 100), 5,744 chunks, 1,819 KG edges.
Config: `configs/scale500.yaml`. Narrative and caveats: `artifacts/EXPERIMENT_LOG.md`,
Phase 10 parts A and B.

Everything here was produced by `--config configs/scale500.yaml`; the dev artifacts that
back the published 50-page `reports/ablation.md` were not touched.

## Headline

**The enhancement gap widens with corpus size.** Baseline retrieval degrades as the corpus
grows (R@5 0.69 → 0.57 from 50 to 500 pages) while the enhanced configuration improves
(0.87 → 0.91). Relative gain goes from +26% to +60%. The enhancements are not a
small-corpus artefact — the 50-page numbers understated them.

| | dev 50pp (n=55) | | scale 500pp (n=127) | |
|---|---|---|---|---|
| | baseline | enhanced | baseline | enhanced |
| R@5 | 0.69 | 0.87 | 0.57 | **0.91** |
| MRR | 0.46 | 0.83 | 0.41 | **0.88** |
| nDCG | 0.52 | 0.84 | 0.47 | **0.89** |
| Correct | 0.66 | 0.70 | 0.59 | 0.69 |

No category carries a low-sample flag for the first time in the project.

## Files

| file | what it is |
|---|---|
| `ablation.md` | the six-config tables, overall / per-category / per-predicate |
| `perq_<config>.jsonl` | per-question log: provenance, predicates walked, abstention, correctness, VQA gate trace |
| `predicate_discovery.json` | schema-generalisation check — the closed 8 cover 22% of freely-extracted triples |
| `gold_review.html` | the spot-check worksheet, table crops embedded, used for human verification |
| `gold_review.md` | same, text form |
| `figure_value_candidates.jsonl` | the 14 table-OCR value proposals put to a human; 8 of 14 were rejected |
| `vqa_figure_values.jsonl` | paired VQA-vs-text comparison on the 8 verified figure_value questions |
| `vqa_gate_check.md` | *(in `reports/`)* the VQA provenance gate with and without table transcription |
| `*.log` | raw run output for each pass |

## Findings that are not in the tables

1. **The closed 8-predicate schema does not generalise.** It covers 22% of triples freely
   extracted from this broader corpus. Two recurrent missing types: a predictor/association
   relation, and a "has measured value" relation. The schema was kept unchanged — the
   finding is worth more reported than fixed. See Phase 10 part A.
2. **The missing value-relation is the structural reason `figure_value` finds no graph
   edges** — not just bad OCR. Even with perfect OCR there is no predicate for
   `(mean age) —has_value→ (26.0 (8.6))`.
3. **Table-OCR-derived values were wrong 53% of the time** (9 of 17 rejected on human
   verification against the crops), with exactly the predicted failure modes: misplacement,
   dropped exponents, label drift. Independent corroboration of the Phase-9 conclusion.
4. **VQA is net zero at n=8** — one genuine win (recovering a row's paired SD that OCR
   linearisation lost) and one confident, correctly-cited cell misread. No VQA win should be
   claimed from this data.
5. **Abstention degraded at scale**: unanswerable abstain-rate 1.00 → 0.82, because a
   larger corpus supplies more near-miss context. Measured against a harder set than dev's,
   since the absence check had already rejected 13 of 24 candidates.
6. **The VQA provenance gate weakens as the corpus grows.** With `top_k = 10` and
   page == paper, up to ten different papers clear "the crop's page had a text hit". Logged,
   not fixed — changing it after the ablation would invalidate these numbers.
7. **Caption-link coverage fell to 71% at scale and was repaired to 85%** (adjacent-column
   captions could never link; one journal's small-caps label defeated the prefix matcher).
   Strictly additive: 291 → 349 links, 0 changed, 0 lost.

## Reproducing

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

`--resume` on the ablation reloads any already-complete `perq_<config>.jsonl` instead of
re-buying it. Total API cost of the full sequence: **~$5.8**.
