"""FAISS-backed vector index (derived from the chunk store).

Implements :class:`~platform_core.stores.base.VectorIndex`. Inner-product index
over L2-normalized embeddings (== cosine). Stores a parallel list of ids
(``unit_id`` for text, ``chunk_id`` for images) so ``search`` returns ids, never
row offsets. The same class backs both the text and image indices.
"""
from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any

import faiss
import numpy as np

from platform_core.stores.base import VectorIndex


class FaissVectorIndex(VectorIndex):
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.ids: list[str] = []

    def add(self, chunk_ids: Sequence[str], vectors: Any) -> None:
        vecs = np.asarray(vectors, dtype="float32")
        if vecs.ndim == 1:
            vecs = vecs[None, :]
        if len(vecs) != len(chunk_ids):
            raise ValueError("vectors and ids length mismatch")
        self.index.add(vecs)
        self.ids.extend(list(chunk_ids))

    def search(self, query_vector: Any, k: int) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        q = np.asarray(query_vector, dtype="float32")
        if q.ndim == 1:
            q = q[None, :]
        scores, idxs = self.index.search(q, min(k, len(self.ids)))
        return [(self.ids[i], float(s)) for s, i in zip(scores[0], idxs[0]) if i >= 0]

    def __len__(self) -> int:
        return len(self.ids)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        faiss.write_index(self.index, path + ".faiss")
        with open(path + ".ids.json", "w", encoding="utf-8") as f:
            json.dump({"dim": self.dim, "ids": self.ids}, f)

    @classmethod
    def load(cls, path: str) -> "FaissVectorIndex":
        with open(path + ".ids.json", encoding="utf-8") as f:
            meta = json.load(f)
        obj = cls(meta["dim"])
        obj.ids = meta["ids"]
        obj.index = faiss.read_index(path + ".faiss")
        return obj
