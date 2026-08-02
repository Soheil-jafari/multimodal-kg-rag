# Bootstrap confidence intervals — 500-page ablation

95% percentile bootstrap over questions, 10,000 resamples, seed 0. Computed from the committed per-question logs; no model was called. Point estimates are recomputed from those logs and verified identical to `ablation.md` before any interval below is reported.

**Two uncertainties, kept separate.** The intervals are *sampling* uncertainty — how much a score depends on which questions were drawn. Separately, run-to-run variance was measured directly: re-running an identical configuration over the same 17 multi_hop questions moved correctness by **0.029**, so any effect at or below **±0.03** is indistinguishable from re-running the same system twice. An effect is marked *within noise* when its paired interval contains zero, or its magnitude is at or below that floor.


## single_fact  (n=56)

| config | R@5 | Correct | Faith | Decis |
|---|---|---|---|---|
| baseline | 0.61 [0.48, 0.73] | 0.66 [0.54, 0.78] | 0.89 [0.82, 0.95] | 0.84 [0.73, 0.93] |
| +layout | 0.79 [0.68, 0.89] | 0.75 [0.64, 0.85] | 0.84 [0.77, 0.91] | 0.91 [0.84, 0.98] |
| +clip | 0.79 [0.68, 0.89] | 0.74 [0.63, 0.84] | 0.87 [0.81, 0.93] | 0.93 [0.86, 0.98] |
| +caption | 0.79 [0.68, 0.89] | 0.73 [0.62, 0.84] | 0.87 [0.80, 0.93] | 0.88 [0.79, 0.95] |
| +kg | 0.84 [0.73, 0.93] | 0.76 [0.65, 0.86] | 0.89 [0.83, 0.95] | 0.89 [0.80, 0.96] |
| +rerank | 0.89 [0.80, 0.96] | 0.80 [0.70, 0.90] | 0.88 [0.81, 0.95] | 0.89 [0.80, 0.96] |

## multi_hop  (n=17)

| config | R@5 | Correct | Faith | Decis |
|---|---|---|---|---|
| baseline | 0.41 [0.24, 0.59] | 0.35 [0.18, 0.53] | 0.65 [0.50, 0.81] | 0.71 [0.47, 0.88] |
| +layout | 0.62 [0.47, 0.76] | 0.38 [0.21, 0.59] | 0.61 [0.51, 0.71] | 0.76 [0.53, 0.94] |
| +clip | 0.62 [0.47, 0.76] | 0.38 [0.21, 0.59] | 0.63 [0.53, 0.73] | 0.76 [0.53, 0.94] |
| +caption | 0.62 [0.47, 0.76] | 0.44 [0.26, 0.65] | 0.54 [0.45, 0.63] | 0.76 [0.53, 0.94] |
| +kg | 0.62 [0.47, 0.76] | 0.44 [0.26, 0.65] | 0.54 [0.41, 0.65] | 0.76 [0.53, 0.94] |
| +rerank | 0.82 [0.68, 0.94] | 0.35 [0.18, 0.53] | 0.61 [0.43, 0.78] | 0.76 [0.53, 0.94] |

## figure  (n=12)

| config | R@5 | Correct | Faith | Decis |
|---|---|---|---|---|
| baseline | 0.38 [0.25, 0.50] | 0.29 [0.08, 0.50] | 0.62 [0.49, 0.77] | 0.83 [0.58, 1.00] |
| +layout | 0.50 [0.50, 0.50] | 0.25 [0.08, 0.46] | 0.61 [0.48, 0.77] | 0.75 [0.50, 1.00] |
| +clip | 0.50 [0.50, 0.50] | 0.33 [0.12, 0.54] | 0.50 [0.38, 0.64] | 0.83 [0.58, 1.00] |
| +caption | 1.00 [1.00, 1.00] | 0.21 [0.04, 0.42] | 0.68 [0.49, 0.88] | 0.67 [0.42, 0.92] |
| +kg | 1.00 [1.00, 1.00] | 0.38 [0.17, 0.62] | 0.68 [0.51, 0.85] | 0.75 [0.50, 1.00] |
| +rerank | 1.00 [1.00, 1.00] | 0.25 [0.08, 0.46] | 0.62 [0.47, 0.78] | 0.75 [0.50, 1.00] |

_n=12: intervals are wide by construction. Read these rows as direction, not as a measurement._

## figure_value  (n=8)

| config | R@5 | Correct | Faith | Decis |
|---|---|---|---|---|
| baseline | 0.62 [0.25, 0.88] | 0.38 [0.12, 0.75] | 0.85 [0.65, 1.00] | 1.00 [1.00, 1.00] |
| +layout | 0.62 [0.25, 0.88] | 0.50 [0.12, 0.88] | 0.83 [0.62, 1.00] | 0.88 [0.62, 1.00] |
| +clip | 0.62 [0.25, 0.88] | 0.50 [0.12, 0.88] | 0.83 [0.62, 1.00] | 0.88 [0.62, 1.00] |
| +caption | 0.88 [0.62, 1.00] | 0.50 [0.12, 0.88] | 0.85 [0.65, 1.00] | 1.00 [1.00, 1.00] |
| +kg | 0.88 [0.62, 1.00] | 0.50 [0.12, 0.88] | 0.85 [0.65, 1.00] | 1.00 [1.00, 1.00] |
| +rerank | 0.88 [0.62, 1.00] | 0.75 [0.38, 1.00] | 0.85 [0.65, 1.00] | 1.00 [1.00, 1.00] |

_n=8: intervals are wide by construction. Read these rows as direction, not as a measurement._

## text_derived  (n=23)

| config | R@5 | Correct | Faith | Decis |
|---|---|---|---|---|
| baseline | 0.70 [0.52, 0.87] | 0.83 [0.67, 0.96] | 0.84 [0.72, 0.94] | 0.91 [0.78, 1.00] |
| +layout | 1.00 [1.00, 1.00] | 0.89 [0.76, 1.00] | 0.88 [0.78, 0.97] | 1.00 [1.00, 1.00] |
| +clip | 1.00 [1.00, 1.00] | 0.87 [0.74, 0.98] | 0.88 [0.78, 0.97] | 1.00 [1.00, 1.00] |
| +caption | 1.00 [1.00, 1.00] | 0.91 [0.80, 1.00] | 0.88 [0.78, 0.97] | 1.00 [1.00, 1.00] |
| +kg | 1.00 [1.00, 1.00] | 0.91 [0.80, 1.00] | 0.88 [0.78, 0.97] | 1.00 [1.00, 1.00] |
| +rerank | 1.00 [1.00, 1.00] | 0.87 [0.74, 0.98] | 0.87 [0.76, 0.97] | 1.00 [1.00, 1.00] |

## unanswerable  (n=11)

| config | R@5 | Correct | Faith | Decis |
|---|---|---|---|---|
| baseline | — | — | — | 0.82 [0.55, 1.00] |
| +layout | — | — | — | 0.82 [0.55, 1.00] |
| +clip | — | — | — | 0.82 [0.55, 1.00] |
| +caption | — | — | — | 0.82 [0.55, 1.00] |
| +kg | — | — | — | 0.82 [0.55, 1.00] |
| +rerank | — | — | — | 0.91 [0.73, 1.00] |

_n=11: intervals are wide by construction. Read these rows as direction, not as a measurement._

## OVERALL  (n=127)

| config | R@5 | Correct | Faith | Decis |
|---|---|---|---|---|
| baseline | 0.57 [0.49, 0.66] | 0.59 [0.51, 0.67] | 0.82 [0.77, 0.87] | 0.84 [0.78, 0.91] |
| +layout | 0.76 [0.69, 0.83] | 0.66 [0.57, 0.73] | 0.80 [0.75, 0.85] | 0.88 [0.83, 0.94] |
| +clip | 0.76 [0.69, 0.83] | 0.66 [0.58, 0.73] | 0.81 [0.76, 0.85] | 0.90 [0.84, 0.94] |
| +caption | 0.83 [0.76, 0.89] | 0.66 [0.57, 0.73] | 0.81 [0.76, 0.86] | 0.87 [0.80, 0.92] |
| +kg | 0.86 [0.80, 0.91] | 0.69 [0.61, 0.76] | 0.83 [0.77, 0.87] | 0.88 [0.83, 0.94] |
| +rerank | 0.91 [0.86, 0.96] | 0.69 [0.61, 0.77] | 0.82 [0.77, 0.87] | 0.89 [0.83, 0.94] |

## Per predicate — enhanced (`+rerank`)

| predicate | n | Correct | R@5 |
|---|---:|---|---|
| treats | 15 | 0.60 [0.40, 0.80] | 0.90 [0.80, 1.00] |
| causes | 12 | 0.79 [0.58, 0.96] | 1.00 [1.00, 1.00] |
| inhibits | 8 | 0.88 [0.62, 1.00] | 1.00 [1.00, 1.00] |
| increases | 11 | 0.82 [0.55, 1.00] | 0.86 [0.64, 1.00] |
| decreases | 9 | 0.61 [0.33, 0.83] | 0.78 [0.44, 1.00] |
| transforms_to | 7 | 0.86 [0.57, 1.00] | 0.86 [0.57, 1.00] |
| occurs_in | 16 | 0.31 [0.12, 0.50] | 0.81 [0.62, 0.97] |
| measured_by | 7 | 0.86 [0.57, 1.00] | 0.86 [0.57, 1.00] |

_Every predicate has n between 7 and 16, so all of these intervals are wide; they are reported for transparency, not as reliable per-predicate scores._

## What each flag adds — paired bootstrap on the difference

The same resampled questions are scored under both configs, since the two arms are the same questions. An interval containing zero means the step is not distinguishable from no change on this corpus.

| step | ΔR@5 | ΔCorrect | verdict |
|---|---|---|---|
| baseline → +layout | +0.190 [+0.107, +0.274] | +0.065 [+0.000, +0.132] | R@5 real; Correct within noise |
| +layout → +clip | +0.000 [+0.000, +0.000] | +0.000 [-0.018, +0.023] | **within noise** |
| +clip → +caption | +0.069 [+0.035, +0.107] | +0.000 [-0.030, +0.030] | R@5 real; Correct within noise |
| +caption → +kg | +0.026 [+0.000, +0.059] | +0.030 [+0.004, +0.064] | Correct sits ON the ±0.03 floor |
| +kg → +rerank | +0.056 [+0.022, +0.097] | +0.004 [-0.037, +0.048] | R@5 real; Correct within noise |
| baseline → +rerank | +0.341 [+0.258, +0.424] | +0.099 [+0.034, +0.168] | both real |
