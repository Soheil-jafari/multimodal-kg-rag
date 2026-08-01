# Gold set

Evaluation questions for the biomed corpus. Every question is **selected from the
knowledge graph** — the answer and its supporting `chunk_id`s come from an edge, and the
language model is used only to phrase the selected fact into natural language. It never
supplies an answer or evidence. That separation is what makes the set usable as ground
truth.

## Files

| file | corpus | n |
|---|---|---|
| `gold_set.jsonl` | 50-page dev corpus | 55 |
| `gold_set_scale500.jsonl` | 500-page corpus | 127 |
| `gold_set_scale500_prereview.jsonl` | as generated, before human verification | 122 |
| `CLEANUP.md` | known-broken items and their defect classes | — |

The pre-review file is kept so the effect of human verification is inspectable: the
difference between it and the final set is the `figure_value` verdict, where 9 of 17
candidate values were rejected on inspection against the crop images.

## Record schema

```json
{
  "qid": "q123",
  "question": "What is the mean age in years of the cohort of young women at baseline?",
  "category": "figure_value",
  "expected_answer": "26.0 (8.6)",
  "supporting_chunk_ids": ["val00000:409598:r5"],
  "source_predicates": [],
  "value_verified": true,
  "crop_path": "artifacts/scale500/crops/val00000_409598_r5.png"
}
```

`supporting_chunk_ids` are canonical region ids from the chunk store, so scoring stays fair
across chunkers: a retrieved unit counts as a hit when it covers a gold region, which means
the naive-window baseline and the layout-aware enhanced path are graded against the same
anchors. Optional fields appear only where they apply — `value_verified` and `crop_path` on
hand-checked table values, `absence_check_top_cosine` on unanswerable items,
`needs_mismatch_review` where an automated filter wants a second opinion.

## Categories

| category | what it tests |
|---|---|
| `single_fact` | one span answers it |
| `multi_hop` | needs ≥2 regions combined |
| `figure` | needs a figure region or its caption |
| `figure_value` | needs a value read out of a table image |
| `text_derived` | bias guard — written from a passage, graph never consulted |
| `unanswerable` | the corpus does not answer it; the system should abstain |

`text_derived` exists to keep the set honest. Those questions are independent of the
knowledge graph, so if the graph-derived categories scored higher, the set would be
flattering the pipeline that built it. They score highest at baseline, which is the
expected direction.

## Verification

Automated filters reject malformed questions, answers leaked into their own question,
OCR-garbled answers, degenerate multi-hops built from co-located near-duplicate edges, and
questions whose distinguishing terms are absent from the cited region. Drop counts are
reported by reason, since the drop rate is itself a measurement.

Automation is a first pass only. `figure_value` items are verified by a person against the
crop images before use — see `CLEANUP.md` for the defect classes that survived the
automated filters and had to be caught by eye.

Unanswerable items are checked against the corpus index before inclusion: a candidate whose
top cosine against the corpus clears 0.62 is rejected as possibly answerable. At 500 pages
this rejected 13 of 24 candidates that had been safely absent from the 50-page corpus, so
without the check those items would have been silently answerable and the abstention
measurement would have meant nothing.
