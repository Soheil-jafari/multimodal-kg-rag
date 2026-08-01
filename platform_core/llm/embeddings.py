"""Local embedding backends (text via BGE, images via CLIP).

* :class:`SentenceTransformerEmbedder` — implements
  :class:`~platform_core.llm.base.TextEmbeddingModel` using ``BAAI/bge-base-en-v1.5``.
  ``embed`` L2-normalizes (cosine via inner product). ``embed_query`` prepends
  BGE's retrieval instruction. ``token_len`` exposes the tokenizer for chunkers.
* :class:`ClipEmbedder` — implements
  :class:`~platform_core.llm.base.ImageEmbeddingModel` using a CLIP dual encoder
  (image tower for crops, text tower for queries).
* :class:`BiomedClipEmbedder` — same interface, biomedical-domain weights
  (ViT-B/16 + PubMedBERT) loaded via ``open_clip``.
* :func:`make_image_embedder` — the single selection point; config's
  ``models.image_embedding_model`` string picks the implementation.

4GB-GPU discipline: models are constructed one at a time by the caller (build
script) and freed before the next; encoding runs in config-sized batches. Device
is auto-selected (falls back to CPU when CUDA is unavailable).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# BGE v1.5 retrieval instruction (applied to the QUERY side only).
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

CLIP_MODEL = "clip-ViT-B-32"
BIOMEDCLIP_MODEL = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def _resolve_device(pref: str) -> str:
    import torch

    if pref == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return pref or ("cuda" if torch.cuda.is_available() else "cpu")


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5",
                 device: str = "cuda", batch_size: int = 16) -> None:
        from sentence_transformers import SentenceTransformer

        self.device = _resolve_device(device)
        self.model = SentenceTransformer(model_name, device=self.device)
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> Any:
        return self.model.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )

    def embed_query(self, text: str) -> Any:
        return self.embed([BGE_QUERY_INSTRUCTION + text])[0]

    @property
    def dim(self) -> int:
        d = self.model.get_sentence_embedding_dimension()
        if d is None:  # some CLIP wrappers don't expose this; probe instead
            d = int(self.model.encode(["_"], convert_to_numpy=True).shape[1])
        return d

    def token_len(self, text: str) -> int:
        return len(self.model.tokenizer.tokenize(text))


class ClipEmbedder:
    def __init__(self, model_name: str = "clip-ViT-B-32",
                 device: str = "cuda", batch_size: int = 16) -> None:
        from sentence_transformers import SentenceTransformer

        self.device = _resolve_device(device)
        self.model = SentenceTransformer(model_name, device=self.device)
        self.batch_size = batch_size

    def embed_images(self, image_paths: Sequence[str]) -> Any:
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in image_paths]
        return self.model.encode(
            images, batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )

    def embed_text(self, texts: Sequence[str]) -> Any:
        return self.model.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )

    @property
    def dim(self) -> int:
        d = self.model.get_sentence_embedding_dimension()
        if d is None:  # some CLIP wrappers don't expose this; probe instead
            d = int(self.model.encode(["_"], convert_to_numpy=True).shape[1])
        return d


class BiomedClipEmbedder:
    """BiomedCLIP (ViT-B/16 image tower + PubMedBERT text tower, 512-d shared space).

    Same surface as :class:`ClipEmbedder`. Loaded through ``open_clip`` because the
    HF repo ships only ``open_clip_pytorch_model.bin`` — there is no
    transformers-format checkpoint, so ``AutoModel``/``SentenceTransformer`` cannot
    read it. The tokenizer carries the model's own 256-token context length.
    """

    def __init__(self, model_name: str = BIOMEDCLIP_MODEL,
                 device: str = "cuda", batch_size: int = 16) -> None:
        import open_clip
        import torch

        self.torch = torch
        self.device = _resolve_device(device)
        ref = model_name if model_name.startswith("hf-hub:") else f"hf-hub:{model_name}"
        self.model, self.preprocess = open_clip.create_model_from_pretrained(ref)
        self.tokenizer = open_clip.get_tokenizer(ref)
        self.model = self.model.to(self.device).eval()
        self.batch_size = batch_size
        self._dim: int | None = None

    def _encode(self, items: Sequence[Any], tower) -> Any:
        import numpy as np

        out = []
        for i in range(0, len(items), self.batch_size):
            batch = self.torch.cat(list(items[i:i + self.batch_size])).to(self.device)
            with self.torch.no_grad():
                v = tower(batch)
            # L2-normalize so FAISS inner product == cosine (store-wide convention)
            v = v / v.norm(dim=-1, keepdim=True)
            out.append(v.cpu().numpy())
        return np.concatenate(out).astype("float32") if out else np.zeros((0, 0), "float32")

    def embed_images(self, image_paths: Sequence[str]) -> Any:
        from PIL import Image

        tensors = [self.preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
                   for p in image_paths]
        return self._encode(tensors, self.model.encode_image)

    def embed_text(self, texts: Sequence[str]) -> Any:
        tokens = [self.tokenizer([t]) for t in texts]
        return self._encode(tokens, self.model.encode_text)

    @property
    def dim(self) -> int:
        if self._dim is None:  # open_clip exposes no dim attribute; probe once
            self._dim = int(self.embed_text(["_"]).shape[1])
        return self._dim


# --- config-level swap point -------------------------------------------------
# Adding an image encoder = one registry entry. `index` keeps each encoder's
# vectors in its own FAISS file so the indices never clobber each other.
IMAGE_ENCODERS: dict[str, dict] = {
    CLIP_MODEL:      {"cls": ClipEmbedder,        "index": "image"},
    BIOMEDCLIP_MODEL: {"cls": BiomedClipEmbedder, "index": "image_biomedclip"},
}


def make_image_embedder(model_name: str = CLIP_MODEL, device: str = "cuda",
                        batch_size: int = 16) -> Any:
    """Construct the ImageEmbeddingModel named by config."""
    try:
        cls = IMAGE_ENCODERS[model_name]["cls"]
    except KeyError:
        raise ValueError(
            f"unknown image_embedding_model {model_name!r}; "
            f"known: {sorted(IMAGE_ENCODERS)}"
        ) from None
    return cls(model_name, device, batch_size)


def image_index_name(model_name: str = CLIP_MODEL) -> str:
    """FAISS basename for this encoder's image index."""
    try:
        return IMAGE_ENCODERS[model_name]["index"]
    except KeyError:
        raise ValueError(f"unknown image_embedding_model {model_name!r}") from None
