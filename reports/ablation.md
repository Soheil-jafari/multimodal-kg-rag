### Overall (rows = configs)
| config | R@5 | P@5 | MRR | nDCG | Correct | Faith | Decis |
|---|---|---|---|---|---|---|---|
| baseline | 0.69 | 0.16 | 0.46 | 0.52 | 0.66 | 0.73 | 0.91 |
| +layout | 0.80 | 0.19 | 0.81 | 0.77 | 0.68 | 0.80 | 0.87 |
| +clip | 0.80 | 0.19 | 0.81 | 0.78 | 0.67 | 0.80 | 0.91 |
| +caption | 0.84 | 0.21 | 0.81 | 0.81 | 0.66 | 0.78 | 0.89 |
| +kg | 0.84 | 0.21 | 0.82 | 0.82 | 0.65 | 0.79 | 0.87 |
| +rerank | 0.87 | 0.22 | 0.83 | 0.84 | 0.70 | 0.78 | 0.91 |

abstention — attempt-rate on answerable / abstain-rate on unanswerable:
  baseline   attempt=0.90 abstain=1.00
  +layout    attempt=0.86 abstain=1.00
  +clip      attempt=0.90 abstain=1.00
  +caption   attempt=0.88 abstain=1.00
  +kg        attempt=0.86 abstain=1.00
  +rerank    attempt=0.90 abstain=1.00

### single_fact  (n=23)
| config | R@5 | P@5 | MRR | nDCG | Correct | Faith | Decis |
|---|---|---|---|---|---|---|---|
| baseline | 0.70 | 0.14 | 0.46 | 0.54 | 0.74 | 0.74 | 0.87 |
| +layout | 0.83 | 0.17 | 0.75 | 0.77 | 0.72 | 0.86 | 0.83 |
| +clip | 0.83 | 0.17 | 0.75 | 0.77 | 0.74 | 0.87 | 0.91 |
| +caption | 0.83 | 0.17 | 0.75 | 0.77 | 0.72 | 0.84 | 0.87 |
| +kg | 0.83 | 0.17 | 0.78 | 0.79 | 0.76 | 0.84 | 0.87 |
| +rerank | 0.83 | 0.17 | 0.83 | 0.83 | 0.80 | 0.81 | 0.91 |

### multi_hop  (n=7)
| config | R@5 | P@5 | MRR | nDCG | Correct | Faith | Decis |
|---|---|---|---|---|---|---|---|
| baseline | 0.50 | 0.20 | 0.52 | 0.50 | 0.43 | 0.61 | 1.00 |
| +layout | 0.79 | 0.31 | 0.76 | 0.72 | 0.50 | 0.68 | 0.86 |
| +clip | 0.79 | 0.31 | 0.76 | 0.72 | 0.43 | 0.71 | 0.86 |
| +caption | 0.71 | 0.29 | 0.74 | 0.70 | 0.43 | 0.68 | 0.86 |
| +kg | 0.71 | 0.29 | 0.74 | 0.72 | 0.36 | 0.68 | 0.86 |
| +rerank | 0.93 | 0.37 | 0.82 | 0.84 | 0.57 | 0.69 | 1.00 |

### figure  (n=5)
| config | R@5 | P@5 | MRR | nDCG | Correct | Faith | Decis |
|---|---|---|---|---|---|---|---|
| baseline | 0.50 | 0.20 | 0.32 | 0.30 | 0.40 | 0.52 | 1.00 |
| +layout | 0.50 | 0.20 | 1.00 | 0.61 | 0.40 | 0.52 | 1.00 |
| +clip | 0.50 | 0.20 | 1.00 | 0.65 | 0.40 | 0.47 | 1.00 |
| +caption | 1.00 | 0.40 | 1.00 | 1.00 | 0.40 | 0.47 | 1.00 |
| +kg | 1.00 | 0.40 | 1.00 | 1.00 | 0.20 | 0.46 | 0.80 |
| +rerank | 1.00 | 0.40 | 0.90 | 0.94 | 0.20 | 0.47 | 0.80 |

### figure_value  (n=4)  ⚠ LOW-N (<5)
| config | R@5 | P@5 | MRR | nDCG | Correct | Faith | Decis |
|---|---|---|---|---|---|---|---|
| baseline | 0.50 | 0.10 | 0.11 | 0.20 | 0.50 | 1.00 | 0.50 |
| +layout | 0.50 | 0.10 | 0.50 | 0.50 | 0.50 | 1.00 | 0.50 |
| +clip | 0.50 | 0.10 | 0.50 | 0.50 | 0.50 | 1.00 | 0.50 |
| +caption | 0.50 | 0.10 | 0.50 | 0.50 | 0.50 | 1.00 | 0.50 |
| +kg | 0.50 | 0.10 | 0.50 | 0.50 | 0.50 | 1.00 | 0.50 |
| +rerank | 0.50 | 0.10 | 0.50 | 0.50 | 0.50 | 1.00 | 0.50 |

### text_derived  (n=10)
| config | R@5 | P@5 | MRR | nDCG | Correct | Faith | Decis |
|---|---|---|---|---|---|---|---|
| baseline | 1.00 | 0.20 | 0.62 | 0.72 | 0.85 | 0.85 | 1.00 |
| +layout | 1.00 | 0.20 | 1.00 | 1.00 | 0.95 | 0.87 | 1.00 |
| +clip | 1.00 | 0.20 | 1.00 | 1.00 | 0.90 | 0.85 | 1.00 |
| +caption | 1.00 | 0.20 | 1.00 | 1.00 | 0.90 | 0.85 | 1.00 |
| +kg | 1.00 | 0.20 | 1.00 | 1.00 | 0.90 | 0.85 | 1.00 |
| +rerank | 1.00 | 0.20 | 0.95 | 0.96 | 0.90 | 0.85 | 1.00 |

### unanswerable  (n=6)
| config | R@5 | P@5 | MRR | nDCG | Correct | Faith | Decis |
|---|---|---|---|---|---|---|---|
| baseline |   - |   - |   - |   - |   - |   - | 1.00 |
| +layout |   - |   - |   - |   - |   - |   - | 1.00 |
| +clip |   - |   - |   - |   - |   - |   - | 1.00 |
| +caption |   - |   - |   - |   - |   - |   - | 1.00 |
| +kg |   - |   - |   - |   - |   - |   - | 1.00 |
| +rerank |   - |   - |   - |   - |   - |   - | 1.00 |

### per-predicate correctness — enhanced config (single_fact + figure_value)
| predicate | n | Correct | R@5 | note |
|---|---|---|---|---|
| treats | 1 | 1.00 | 1.00 | n=1 — not statistically meaningful |
| causes | 8 | 0.81 | 0.94 |  |
| inhibits | 3 | 1.00 | 0.67 | low-n (<5) |
| increases | 8 | 0.69 | 0.88 |  |
| decreases | 4 | 0.50 | 0.75 | low-n (<5) |
| transforms_to | 4 | 0.50 | 1.00 | low-n (<5) |
| occurs_in | 5 | 0.90 | 0.90 |  |
| measured_by | 3 | 0.67 | 0.67 | low-n (<5) |

_Low-sample (n<5) categories/predicates are flagged; n=1 predicates are reported for transparency only, not as reliable scores._