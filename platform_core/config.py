"""Typed configuration schema and loader (mirrors the YAML in ``configs/``).

The entire pipeline is driven by a single :class:`AppConfig`. "Baseline" and
"enhanced" are NOT two codebases — they are two YAML files that set the
:class:`RetrievalFlags` differently. Backend choices (LLM, graph, vector store)
and the image encoder are config-driven too, so nothing is hard-coded.

:meth:`AppConfig.from_yaml` resolves ``extends:`` (a child overrides only the keys
it names, recursively, relative to its own directory), then builds the nested
dataclasses with **strict key checking** — an unknown or misspelled key is an error,
not a silently ignored line. That strictness is the point: a config that looks
applied but isn't is worse than a crash.

Derived artifacts are **corpus-scoped, not config-scoped** (``artifacts/dev/…``).
Baseline and enhanced deliberately share one chunk DB and one index directory:
``use_layout_chunking`` selects between two co-resident text indices
(``text_layout`` / ``text_naive``) rather than rebuilding, and the image encoder
selects its own index basename. So the two pipelines cannot clobber each other
while still reading identical source chunks — which is what makes the comparison
fair.

Secrets are NEVER stored here — API keys are read from environment variables.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalFlags:
    """The switches that separate baseline from enhanced.

    Baseline = every flag ``False``. Enhanced = every flag ``True``. The ablation
    runner toggles exactly one flag at a time, starting from the baseline.
    """

    use_layout_chunking: bool = False  # layout-aware regions vs naive fixed windows
    use_clip_images: bool = False      # CLIP image hits for figures/tables
    use_caption_anchor: bool = False   # caption->crop path (primary figure route)
    use_kg: bool = False               # expand/retrieve via the knowledge graph
    use_rerank: bool = False           # cross-encoder reranking of candidates
    use_vqa: bool = False              # read values OUT of a retrieved crop (vision)
    use_table_vqa_text: bool = False   # index tables by their vision-read transcription
    # pull a retrieved page's crops into the pool. A table states values but not the
    # concepts that link them to a question ("lung cancer", "PAH") — those live in the
    # page's prose — so no table-text improvement can rank it above that prose
    # (measured, phase 9). Reaching the crop THROUGH its page is what works.
    use_page_crop_expansion: bool = False
    # reason step-by-step over the retrieved passages before answering, and keep the
    # trace. Generation-side only: it changes how the answer is produced, never what
    # is retrieved, so it is orthogonal to the ablation series above.
    use_cot: bool = False
    allow_abstain: bool = False        # permit "insufficient evidence" answers


@dataclass
class BackendConfig:
    """Which swappable implementation to construct for each interface."""

    llm: str = "openai"          # -> platform_core.llm.base.LLMClient
    graph: str = "networkx"      # "networkx" | "neo4j"
    vector_store: str = "faiss"  # -> platform_core.stores.base.VectorIndex


@dataclass
class ModelConfig:
    """Model identifiers. Loaded one at a time to fit a 4GB GPU."""

    llm_model: str = "gpt-4o-mini"
    text_embedding_model: str = "BAAI/bge-base-en-v1.5"
    # one of platform_core.llm.embeddings.IMAGE_ENCODERS; picks the encoder AND
    # which image index it reads, so encoders stay runnable side by side
    image_embedding_model: str = "clip-ViT-B-32"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cuda"          # "cuda" | "cpu"
    embed_batch_size: int = 16    # keep small for a 4GB GPU


@dataclass
class IngestConfig:
    """Which PubLayNet shards make up the corpus, and how many pages from each.

    The corpus is part of the experiment, so it belongs in the YAML rather than in
    an entry point's argv: ``configs/scale500.yaml`` differs from the dev corpus only
    by this section plus its paths. ``tag`` becomes the ``page_uid`` prefix
    (``"<tag>:<coco_id>"``), so tags must be unique across shards — two shards sharing
    a tag would collide page ids from different files into one id space.
    """

    repo: str = "jordanparker6/publaynet"
    shards: list = field(default_factory=list)  # [{file: <path in repo>, tag: <short id>}, ...]
    pages_per_shard: int = 50


@dataclass
class PathsConfig:
    """Filesystem locations, corpus-scoped (see module docstring)."""

    publaynet_root: str = "data/publaynet"
    chunk_db: str = "artifacts/dev/chunks.sqlite"
    index_dir: str = "artifacts/dev/index"
    graph_store: str = "artifacts/dev/kg/graph.pkl"
    crops_dir: str = "artifacts/dev/crops"
    reports_dir: str = "reports"
    gold_set: str = "domain_packs/biomed/gold/gold_set.jsonl"


@dataclass
class RetrievalParams:
    """Retrieval knobs shared by both pipelines."""

    top_k: int = 10
    rerank_top_k: int = 10
    image_k: int = 5   # CLIP image hits when use_clip_images
    kg_hops: int = 2   # bounded graph-walk depth when use_kg


@dataclass
class GenerationParams:
    """Answer-generation, abstention, and grounding-check knobs."""

    answer_model: str = "gpt-4o"        # quality matters for answering
    verify_model: str = "gpt-4o-mini"   # cheap LLM for borderline grounding checks
    vqa_model: str = "gpt-4o"           # vision model that reads figure/table crops
    vqa_max_images: int = 2             # crops attached per answer; each costs tokens
    # Crop-relevance gate (phase 9), guarding phase-8 finding #3: without it a
    # retrieval miss becomes a confident wrong answer, because the model will read *a*
    # number off whatever crop it is handed. The gate's PRIMARY condition is
    # provenance — the crop's page must be one whose prose the question matched — since
    # scores are inverted here (the wrong crop out-scored the right one). This value is
    # only a low floor to skip obvious junk among page-eligible crops, and to select
    # the best ones; <= 0 disables gating entirely.
    vqa_min_crop_score: float = 0.30
    abstain_min_score: float = 0.45     # min top context-query cosine to attempt an answer
    grounding_min_sim: float = 0.45     # min sentence-vs-cited-chunk cosine to call supported
    grounding_border: float = 0.08      # band around the threshold sent to the LLM verifier
    grounding_llm_verify: bool = True   # LLM yes/no verify for borderline sentences


_SECTIONS = {
    "flags": RetrievalFlags, "backends": BackendConfig, "models": ModelConfig,
    "paths": PathsConfig, "retrieval": RetrievalParams, "generation": GenerationParams,
    "ingest": IngestConfig,
}


def text_index_name(flags: RetrievalFlags) -> str:
    """Which text index a flag set reads. One definition, shared by every caller.

    Flags SELECT among co-resident indices; they never rebuild. ``use_table_vqa_text``
    changes what a table region's indexed text is, so it gets its own index rather
    than mutating the shared one — same discipline as layout-vs-naive, and it keeps
    the published baseline reproducible.
    """
    if not flags.use_layout_chunking:
        return "text_naive"
    return "text_layout_tablevqa" if flags.use_table_vqa_text else "text_layout"


def _deep_merge(base: dict, over: dict) -> dict:
    """Child wins per key; nested dicts merge rather than replace wholesale."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_raw(path: str, seen: tuple = ()) -> dict:
    """Read one YAML and splice it onto whatever it ``extends:`` (recursively)."""
    import yaml

    path = os.path.abspath(path)
    if path in seen:
        raise ValueError(f"circular extends: {' -> '.join(seen + (path,))}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    parent_path = parent if os.path.isabs(parent) else os.path.join(os.path.dirname(path), parent)
    if not os.path.exists(parent_path):
        raise FileNotFoundError(f"{path}: extends {parent!r} -> missing {parent_path}")
    return _deep_merge(_load_raw(parent_path, seen + (path,)), raw)


def _build(cls, data: Any, section: str, path: str):
    """Instantiate a section dataclass, rejecting unknown keys."""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ValueError(f"{path}: section {section!r} must be a mapping, got {type(data).__name__}")
    allowed = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(
            f"{path}: unknown key(s) {unknown} in section {section!r}. "
            f"Allowed: {sorted(allowed)}"
        )
    return cls(**data)


@dataclass
class AppConfig:
    """Root config object that flows through the entire pipeline."""

    name: str = "baseline"
    flags: RetrievalFlags = field(default_factory=RetrievalFlags)
    backends: BackendConfig = field(default_factory=BackendConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    retrieval: RetrievalParams = field(default_factory=RetrievalParams)
    generation: GenerationParams = field(default_factory=GenerationParams)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    domain_pack: str = "biomed"
    source_path: str = ""  # which YAML this came from (provenance for reports/logs)

    @classmethod
    def from_yaml(cls, path: str, root: str | None = None) -> "AppConfig":
        """Load + resolve ``extends:`` + validate.

        ``root`` (default: the repo root) makes relative paths in the YAML absolute,
        so a config works regardless of the process's working directory.
        """
        raw = _load_raw(path)
        top = {k: v for k, v in raw.items() if k not in _SECTIONS}
        allowed_top = {f.name for f in dataclasses.fields(cls)} - set(_SECTIONS)
        unknown = sorted(set(top) - allowed_top)
        if unknown:
            raise ValueError(f"{path}: unknown top-level key(s) {unknown}. "
                             f"Allowed: {sorted(allowed_top | set(_SECTIONS))}")
        cfg = cls(
            **top,
            **{name: _build(kls, raw.get(name), name, path) for name, kls in _SECTIONS.items()},
        )
        cfg.source_path = os.path.abspath(path)
        cfg._absolutize(root or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cfg.validate()
        return cfg

    def _absolutize(self, root: str) -> None:
        for f in dataclasses.fields(PathsConfig):
            v = getattr(self.paths, f.name)
            if v and not os.path.isabs(v):
                setattr(self.paths, f.name, os.path.join(root, v).replace("\\", "/"))

    def validate(self) -> None:
        """Fail loudly on values the pipeline cannot honour."""
        from platform_core.llm.embeddings import IMAGE_ENCODERS

        if self.models.image_embedding_model not in IMAGE_ENCODERS:
            raise ValueError(
                f"{self.source_path}: models.image_embedding_model "
                f"{self.models.image_embedding_model!r} is not registered. "
                f"Known: {sorted(IMAGE_ENCODERS)}"
            )
        if self.backends.graph not in ("networkx", "neo4j"):
            raise ValueError(f"{self.source_path}: backends.graph must be networkx|neo4j")
        if self.models.device not in ("cuda", "cpu"):
            raise ValueError(f"{self.source_path}: models.device must be cuda|cpu")
        if not 0.0 <= self.generation.abstain_min_score <= 1.0:
            raise ValueError(f"{self.source_path}: generation.abstain_min_score must be in [0,1]")
        if self.retrieval.top_k < 1:
            raise ValueError(f"{self.source_path}: retrieval.top_k must be >= 1")
        self._validate_ingest()

    def _validate_ingest(self) -> None:
        """Shard entries must be well-formed and uniquely tagged.

        A duplicate/missing tag is silent corruption rather than a crash — pages from
        two different parquet files would be minted with colliding ``page_uid``s and
        overwrite each other in the chunk store — so it is rejected at load time.
        """
        if not isinstance(self.ingest.shards, list):
            raise ValueError(f"{self.source_path}: ingest.shards must be a list")
        tags: set = set()
        for i, s in enumerate(self.ingest.shards):
            if not isinstance(s, dict) or not s.get("file") or not s.get("tag"):
                raise ValueError(
                    f"{self.source_path}: ingest.shards[{i}] must be a mapping with "
                    f"non-empty 'file' and 'tag' keys, got {s!r}"
                )
            if set(s) - {"file", "tag"}:
                raise ValueError(f"{self.source_path}: ingest.shards[{i}] has unknown "
                                 f"key(s) {sorted(set(s) - {'file', 'tag'})}")
            if s["tag"] in tags:
                raise ValueError(f"{self.source_path}: duplicate ingest shard tag "
                                 f"{s['tag']!r} — tags prefix page ids and must be unique")
            tags.add(s["tag"])
        if self.ingest.pages_per_shard < 1:
            raise ValueError(f"{self.source_path}: ingest.pages_per_shard must be >= 1")

    # --- derived artifact locations (single place that knows the layout) ---
    def text_index_path(self) -> str:
        """Which text index this config reads (see :func:`text_index_name`)."""
        return os.path.join(self.paths.index_dir,
                            text_index_name(self.flags)).replace("\\", "/")

    def image_index_path(self) -> str:
        from platform_core.llm.embeddings import image_index_name

        return os.path.join(self.paths.index_dir,
                            image_index_name(self.models.image_embedding_model)).replace("\\", "/")

    def describe(self) -> str:
        """Resolved settings, for logs and reports — shows what actually took effect."""
        f = self.flags
        on = [k for k, v in dataclasses.asdict(f).items() if v]
        shards = ", ".join(s["tag"] for s in self.ingest.shards) or "(none)"
        return "\n".join([
            f"config          : {self.name}  ({self.source_path})",
            f"corpus          : {shards} x {self.ingest.pages_per_shard}pp -> {self.paths.chunk_db}",
            f"flags ON        : {', '.join(on) if on else '(none — baseline)'}",
            f"image encoder   : {self.models.image_embedding_model}",
            f"image index     : {self.image_index_path()}",
            f"text index      : {self.text_index_path()}",
            f"text embedder   : {self.models.text_embedding_model}",
            f"reranker        : {self.models.reranker_model}",
            f"answer / verify : {self.generation.answer_model} / {self.generation.verify_model}",
            f"vqa model       : {self.generation.vqa_model}",
            f"abstain_min_score: {self.generation.abstain_min_score}",
            f"top_k / rerank_k / image_k / kg_hops: {self.retrieval.top_k} / "
            f"{self.retrieval.rerank_top_k} / {self.retrieval.image_k} / {self.retrieval.kg_hops}",
            f"device / batch  : {self.models.device} / {self.models.embed_batch_size}",
        ])
