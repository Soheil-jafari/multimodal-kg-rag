### Overall (rows = configs)
| config | R@5 | P@5 | MRR | nDCG | Correct | Faith | Decis |
|---|---|---|---|---|---|---|---|
| baseline | 0.57 | 0.13 | 0.41 | 0.47 | 0.59 | 0.82 | 0.84 |
| +layout | 0.76 | 0.18 | 0.78 | 0.75 | 0.66 | 0.80 | 0.88 |
| +clip | 0.76 | 0.18 | 0.78 | 0.75 | 0.66 | 0.81 | 0.90 |
| +caption | 0.83 | 0.21 | 0.79 | 0.80 | 0.66 | 0.81 | 0.87 |
| +kg | 0.86 | 0.21 | 0.80 | 0.80 | 0.69 | 0.83 | 0.88 |
| +rerank | 0.91 | 0.23 | 0.88 | 0.89 | 0.69 | 0.82 | 0.89 |

abstention — attempt-rate on answerable / abstain-rate on unanswerable:
  baseline   attempt=0.84 abstain=0.82
  +layout    attempt=0.89 abstain=0.82
  +clip      attempt=0.91 abstain=0.82
  +caption   attempt=0.87 abstain=0.82
  +kg        attempt=0.89 abstain=0.82
  +rerank    attempt=0.89 abstain=0.91

### single_fact  (n=56)
| config | R@5 | P@5 | MRR | nDCG | Correct | Faith | Decis |
|---|---|---|---|---|---|---|---|
| baseline | 0.61 | 0.12 | 0.39 | 0.47 | 0.66 | 0.89 | 0.84 |
| +layout | 0.79 | 0.16 | 0.74 | 0.77 | 0.75 | 0.84 | 0.91 |
| +clip | 0.79 | 0.16 | 0.74 | 0.77 | 0.74 | 0.87 | 0.93 |
| +caption | 0.79 | 0.16 | 0.73 | 0.76 | 0.73 | 0.87 | 0.88 |
| +kg | 0.84 | 0.17 | 0.75 | 0.78 | 0.76 | 0.89 | 0.89 |
| +rerank | 0.89 | 0.18 | 0.88 | 0.88 | 0.80 | 0.88 | 0.89 |

### multi_hop  (n=17)
| config | R@5 | P@5 | MRR | nDCG | Correct | Faith | Decis |
|---|---|---|---|---|---|---|---|
| baseline | 0.41 | 0.16 | 0.38 | 0.37 | 0.35 | 0.65 | 0.71 |
| +layout | 0.62 | 0.25 | 0.79 | 0.68 | 0.38 | 0.61 | 0.76 |
| +clip | 0.62 | 0.25 | 0.79 | 0.68 | 0.38 | 0.63 | 0.76 |
| +caption | 0.62 | 0.25 | 0.79 | 0.68 | 0.44 | 0.54 | 0.76 |
| +kg | 0.62 | 0.25 | 0.79 | 0.68 | 0.44 | 0.54 | 0.76 |
| +rerank | 0.82 | 0.33 | 0.81 | 0.78 | 0.35 | 0.61 | 0.76 |

### figure  (n=12)
| config | R@5 | P@5 | MRR | nDCG | Correct | Faith | Decis |
|---|---|---|---|---|---|---|---|
| baseline | 0.38 | 0.15 | 0.44 | 0.35 | 0.29 | 0.62 | 0.83 |
| +layout | 0.50 | 0.20 | 1.00 | 0.61 | 0.25 | 0.61 | 0.75 |
| +clip | 0.50 | 0.20 | 1.00 | 0.61 | 0.33 | 0.50 | 0.83 |
| +caption | 1.00 | 0.40 | 1.00 | 0.99 | 0.21 | 0.68 | 0.67 |
| +kg | 1.00 | 0.40 | 1.00 | 0.99 | 0.38 | 0.68 | 0.75 |
| +rerank | 1.00 | 0.40 | 1.00 | 0.99 | 0.25 | 0.62 | 0.75 |

### figure_value  (n=8)
| config | R@5 | P@5 | MRR | nDCG | Correct | Faith | Decis |
|---|---|---|---|---|---|---|---|
| baseline | 0.62 | 0.12 | 0.28 | 0.42 | 0.38 | 0.85 | 1.00 |
| +layout | 0.62 | 0.12 | 0.45 | 0.51 | 0.50 | 0.83 | 0.88 |
| +clip | 0.62 | 0.12 | 0.45 | 0.51 | 0.50 | 0.83 | 0.88 |
| +caption | 0.88 | 0.17 | 0.60 | 0.70 | 0.50 | 0.85 | 1.00 |
| +kg | 0.88 | 0.17 | 0.60 | 0.70 | 0.50 | 0.85 | 1.00 |
| +rerank | 0.88 | 0.17 | 0.83 | 0.87 | 0.75 | 0.85 | 1.00 |

### text_derived  (n=23)
| config | R@5 | P@5 | MRR | nDCG | Correct | Faith | Decis |
|---|---|---|---|---|---|---|---|
| baseline | 0.70 | 0.14 | 0.52 | 0.61 | 0.83 | 0.84 | 0.91 |
| +layout | 1.00 | 0.20 | 0.86 | 0.89 | 0.89 | 0.88 | 1.00 |
| +clip | 1.00 | 0.20 | 0.86 | 0.89 | 0.87 | 0.88 | 1.00 |
| +caption | 1.00 | 0.20 | 0.86 | 0.89 | 0.91 | 0.88 | 1.00 |
| +kg | 1.00 | 0.20 | 0.86 | 0.89 | 0.91 | 0.88 | 1.00 |
| +rerank | 1.00 | 0.20 | 0.92 | 0.94 | 0.87 | 0.87 | 1.00 |

### unanswerable  (n=11)
| config | R@5 | P@5 | MRR | nDCG | Correct | Faith | Decis |
|---|---|---|---|---|---|---|---|
| baseline |   - |   - |   - |   - |   - |   - | 0.82 |
| +layout |   - |   - |   - |   - |   - |   - | 0.82 |
| +clip |   - |   - |   - |   - |   - |   - | 0.82 |
| +caption |   - |   - |   - |   - |   - |   - | 0.82 |
| +kg |   - |   - |   - |   - |   - |   - | 0.82 |
| +rerank |   - |   - |   - |   - |   - |   - | 0.91 |

### per-predicate correctness — enhanced config (single_fact + figure_value)
| predicate | n | Correct | R@5 | note |
|---|---|---|---|---|
| treats | 15 | 0.60 | 0.90 |  |
| causes | 12 | 0.79 | 1.00 |  |
| inhibits | 8 | 0.88 | 1.00 |  |
| increases | 11 | 0.82 | 0.86 |  |
| decreases | 9 | 0.61 | 0.78 |  |
| transforms_to | 7 | 0.86 | 0.86 |  |
| occurs_in | 16 | 0.31 | 0.81 |  |
| measured_by | 7 | 0.86 | 0.86 |  |

_Low-sample (n<5) categories/predicates are flagged; n=1 predicates are reported for transparency only, not as reliable scores._