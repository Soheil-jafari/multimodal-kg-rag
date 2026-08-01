"""CLIP image index for FIGURE/TABLE region crops (derived from the chunk store).

A separate FAISS index (own file) over CLIP image-crop embeddings, keyed by the
crop's ``chunk_id``. Queries embed the question with the CLIP *text* tower, so
image hits still resolve to canonical chunks. Only active when ``use_clip_images``
is set. Structurally identical to the text index, so it reuses
:class:`~platform_core.stores.vector_store.FaissVectorIndex`.
"""
from __future__ import annotations

from platform_core.stores.vector_store import FaissVectorIndex


class ClipImageIndex(FaissVectorIndex):
    """FAISS index over CLIP image-crop embeddings (figures + tables)."""
