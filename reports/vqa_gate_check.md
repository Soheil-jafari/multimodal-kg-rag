# VQA provenance gate — with vs without table transcription

Retrieval is run ONCE per question and the gate is applied twice to that same result, so the only variable is what the gate scores a crop on: its transcription (dev) or its OCR text (the 500-page corpus, where no transcription exists).

Gate floor `vqa_min_crop_score` = 0.3; `vqa_max_images` = 2.

## q060 — What was the mean age in years of the study cohort?

gold crop(s): `val00000:409598:r5`

**with transcription** — 5 crop(s) in context, 3 passed, read: `val00000:409598:r5, val00000:415100:r6`

| crop | scored on | score | on supported page | passed | gold |
|---|---|---:|:---:|:---:|:---:|
| `val00000:409598:r5` | transcription | 0.5521 | yes | PASS | **GOLD** |
| `val00000:416029:r11` | transcription | 0.5647 | NO | block |  |
| `val00000:415100:r6` | transcription | 0.5515 | yes | PASS |  |
| `val00000:416125:r13` | transcription | 0.5251 | yes | PASS |  |
| `val00000:415312:r14` | transcription | 0.4959 | NO | block |  |

**without (500-page)** — 5 crop(s) in context, 3 passed, read: `val00000:409598:r5, val00000:415100:r6`

| crop | scored on | score | on supported page | passed | gold |
|---|---|---:|:---:|:---:|:---:|
| `val00000:409598:r5` | OCR text | 0.5568 | yes | PASS | **GOLD** |
| `val00000:416029:r11` | OCR text | 0.5744 | NO | block |  |
| `val00000:415100:r6` | OCR text | 0.5313 | yes | PASS |  |
| `val00000:416125:r13` | OCR text | 0.5262 | yes | PASS |  |
| `val00000:415312:r14` | OCR text | 0.5408 | NO | block |  |

## q061 — What was the unadjusted relative risk (95% CI) of lung cancer in the highest cumulative PAH exposure category?

gold crop(s): `val00000:415199:r10`

**with transcription** — 3 crop(s) in context, 1 passed, read: `val00000:415199:r10`

| crop | scored on | score | on supported page | passed | gold |
|---|---|---:|:---:|:---:|:---:|
| `val00000:415497:r17` | transcription | 0.5534 | NO | block |  |
| `val00000:415600:r9` | transcription | 0.5311 | NO | block |  |
| `val00000:415199:r10` | transcription | 0.4589 | yes | PASS | **GOLD** |

**without (500-page)** — 3 crop(s) in context, 1 passed, read: `val00000:415199:r10`

| crop | scored on | score | on supported page | passed | gold |
|---|---|---:|:---:|:---:|:---:|
| `val00000:415497:r17` | OCR text | 0.5492 | NO | block |  |
| `val00000:415600:r9` | OCR text | 0.503 | NO | block |  |
| `val00000:415199:r10` | OCR text | 0.536 | yes | PASS | **GOLD** |

## q062 — In the 1996 survey, what percentage of respondents said their well water had ever been tested?

gold crop(s): `val00000:415448:r15`

**with transcription** — 3 crop(s) in context, 3 passed, read: `val00000:415448:r15, val00000:415312:r15`

| crop | scored on | score | on supported page | passed | gold |
|---|---|---:|:---:|:---:|:---:|
| `val00000:415448:r15` | transcription | 0.7032 | yes | PASS | **GOLD** |
| `val00000:415312:r15` | transcription | 0.4783 | yes | PASS |  |
| `val00000:415448:r16` | transcription | 0.4703 | yes | PASS |  |

**without (500-page)** — 3 crop(s) in context, 3 passed, read: `val00000:415448:r15, val00000:415448:r16`

| crop | scored on | score | on supported page | passed | gold |
|---|---|---:|:---:|:---:|:---:|
| `val00000:415448:r15` | OCR text | 0.7195 | yes | PASS | **GOLD** |
| `val00000:415312:r15` | OCR text | 0.4967 | yes | PASS |  |
| `val00000:415448:r16` | OCR text | 0.5241 | yes | PASS |  |

## q063 — In Milham (1988), what was the SMR (95% CI) for brain tumor?

gold crop(s): `val00000:415100:r6`

**with transcription** — 5 crop(s) in context, 1 passed, read: `val00000:415100:r6`

| crop | scored on | score | on supported page | passed | gold |
|---|---|---:|:---:|:---:|:---:|
| `val00000:415100:r6` | transcription | 0.6121 | yes | PASS | **GOLD** |
| `val00000:415199:r10` | transcription | 0.4363 | NO | block |  |
| `val00000:415497:r17` | transcription | 0.4773 | NO | block |  |
| `val00000:415448:r16` | transcription | 0.3262 | NO | block |  |
| `val00000:415821:r14` | transcription | 0.4685 | NO | block |  |

**without (500-page)** — 5 crop(s) in context, 1 passed, read: `val00000:415100:r6`

| crop | scored on | score | on supported page | passed | gold |
|---|---|---:|:---:|:---:|:---:|
| `val00000:415100:r6` | OCR text | 0.6132 | yes | PASS | **GOLD** |
| `val00000:415199:r10` | OCR text | 0.4901 | NO | block |  |
| `val00000:415497:r17` | OCR text | 0.4879 | NO | block |  |
| `val00000:415448:r16` | OCR text | 0.3529 | NO | block |  |
| `val00000:415821:r14` | OCR text | 0.4649 | NO | block |  |

## Verdict

- gold crop passes the gate: **4/4** with transcription, **4/4** without.
- gold crop is among the crops actually read: **4/4** vs **4/4**.
- questions where the selected crop set CHANGED: q062.

The gate's primary condition is provenance — a crop is eligible only if its page is one whose prose the question matched — and provenance does not depend on the transcription at all. The score survives only as a floor and a tie-break among already-eligible crops, which is why the decision is stable under a change of scoring text.
