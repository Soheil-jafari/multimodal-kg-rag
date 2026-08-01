"""Retrieval metrics: Recall@k, Precision@k, MRR, nDCG.

Retrieved chunk_ids are compared against a question's gold supporting_chunk_ids
(binary relevance). Ranked list order matters for MRR/nDCG.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

KS = (1, 3, 5, 10)


def _dcg(rels: Sequence[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def retrieval_scores(retrieved_ids: Sequence[str], gold_ids: Sequence[str],
                     ks: Sequence[int] = KS) -> dict:
    """All retrieval metrics for one question. Undefined (returns None) if no gold."""
    gold = set(gold_ids)
    if not gold:
        return {}
    recall, precision = {}, {}
    for k in ks:
        hit = len(set(retrieved_ids[:k]) & gold)
        recall[k] = hit / len(gold)
        precision[k] = hit / k
    mrr = 0.0
    for i, cid in enumerate(retrieved_ids, 1):
        if cid in gold:
            mrr = 1.0 / i
            break
    k = max(ks)
    rels = [1 if cid in gold else 0 for cid in retrieved_ids[:k]]
    idcg = _dcg([1] * min(len(gold), k))
    ndcg = (_dcg(rels) / idcg) if idcg else 0.0
    return {"recall": recall, "precision": precision, "mrr": mrr, "ndcg": ndcg}
