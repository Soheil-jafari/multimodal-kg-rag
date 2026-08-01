### Overall (rows = configs)
| config | R@5 | P@5 | MRR | nDCG | Correct | Faith | Decis |
|---|---|---|---|---|---|---|---|
| baseline | 0.69 | 0.16 | 0.46 | 0.52 | 0.59 | 0.79 | 0.76 |
| +layout | 0.80 | 0.19 | 0.81 | 0.77 | 0.60 | 0.84 | 0.76 |
| +clip | 0.80 | 0.19 | 0.81 | 0.78 | 0.63 | 0.85 | 0.78 |
| +caption | 0.84 | 0.21 | 0.81 | 0.81 | 0.60 | 0.85 | 0.75 |
| +kg | 0.84 | 0.21 | 0.82 | 0.82 | 0.65 | 0.83 | 0.78 |
| +rerank | 0.87 | 0.22 | 0.83 | 0.84 | 0.61 | 0.82 | 0.78 |

abstention — attempt-rate on answerable / abstain-rate on unanswerable:
  baseline   attempt=0.73 abstain=1.00
  +layout    attempt=0.73 abstain=1.00
  +clip      attempt=0.76 abstain=1.00
  +caption   attempt=0.71 abstain=1.00
  +kg        attempt=0.76 abstain=1.00
  +rerank    attempt=0.76 abstain=1.00

### single_fact  (n=23)
| config | R@5 | P@5 | MRR | nDCG | Correct | Faith | Decis |
|---|---|---|---|---|---|---|---|
| baseline | 0.70 | 0.14 | 0.46 | 0.54 | 0.63 | 0.78 | 0.70 |
| +layout | 0.83 | 0.17 | 0.75 | 0.77 | 0.67 | 0.75 | 0.74 |
| +clip | 0.83 | 0.17 | 0.75 | 0.77 | 0.67 | 0.75 | 0.74 |
| +caption | 0.83 | 0.17 | 0.75 | 0.77 | 0.63 | 0.80 | 0.70 |
| +kg | 0.83 | 0.17 | 0.78 | 0.79 | 0.67 | 0.78 | 0.74 |
| +rerank | 0.83 | 0.17 | 0.83 | 0.83 | 0.63 | 0.80 | 0.70 |

### multi_hop  (n=7)
| config | R@5 | P@5 | MRR | nDCG | Correct | Faith | Decis |
|---|---|---|---|---|---|---|---|
| baseline | 0.50 | 0.20 | 0.52 | 0.50 | 0.36 | 0.60 | 0.57 |
| +layout | 0.79 | 0.31 | 0.76 | 0.72 | 0.29 | 0.89 | 0.43 |
| +clip | 0.79 | 0.31 | 0.76 | 0.72 | 0.43 | 0.79 | 0.57 |
| +caption | 0.71 | 0.29 | 0.74 | 0.70 | 0.29 | 0.83 | 0.43 |
| +kg | 0.71 | 0.29 | 0.74 | 0.72 | 0.43 | 0.75 | 0.57 |
| +rerank | 0.93 | 0.37 | 0.82 | 0.84 | 0.43 | 0.67 | 0.71 |

### figure  (n=5)
| config | R@5 | P@5 | MRR | nDCG | Correct | Faith | Decis |
|---|---|---|---|---|---|---|---|
| baseline | 0.50 | 0.20 | 0.32 | 0.30 | 0.20 | 0.59 | 0.80 |
| +layout | 0.50 | 0.20 | 1.00 | 0.61 | 0.20 | 0.88 | 0.80 |
| +clip | 0.50 | 0.20 | 1.00 | 0.65 | 0.30 | 1.00 | 0.80 |
| +caption | 1.00 | 0.40 | 1.00 | 1.00 | 0.30 | 0.72 | 0.80 |
| +kg | 1.00 | 0.40 | 1.00 | 1.00 | 0.40 | 0.72 | 0.80 |
| +rerank | 1.00 | 0.40 | 0.90 | 0.94 | 0.20 | 0.76 | 0.80 |

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
| baseline | 1.00 | 0.20 | 0.62 | 0.72 | 0.90 | 0.90 | 1.00 |
| +layout | 1.00 | 0.20 | 1.00 | 1.00 | 0.90 | 0.92 | 1.00 |
| +clip | 1.00 | 0.20 | 1.00 | 1.00 | 0.90 | 0.95 | 1.00 |
| +caption | 1.00 | 0.20 | 1.00 | 1.00 | 0.95 | 0.95 | 1.00 |
| +kg | 1.00 | 0.20 | 1.00 | 1.00 | 0.95 | 0.95 | 1.00 |
| +rerank | 1.00 | 0.20 | 0.95 | 0.96 | 0.95 | 0.92 | 1.00 |

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
| causes | 8 | 0.56 | 0.94 |  |
| inhibits | 3 | 0.67 | 0.67 | low-n (<5) |
| increases | 8 | 0.50 | 0.88 |  |
| decreases | 4 | 0.50 | 0.75 | low-n (<5) |
| transforms_to | 4 | 0.25 | 1.00 | low-n (<5) |
| occurs_in | 5 | 0.80 | 0.90 |  |
| measured_by | 3 | 0.67 | 0.67 | low-n (<5) |

_Low-sample (n<5) categories/predicates are flagged; n=1 predicates are reported for transparency only, not as reliable scores._