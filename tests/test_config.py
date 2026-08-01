"""Config loading: `extends:` resolves, baseline/enhanced differ only in flags.

These are the guard-rails for the claim that the pipeline is config-driven: if a
YAML key stops taking effect, or a typo starts being silently ignored, these fail.
"""
from __future__ import annotations

import dataclasses
import os

import pytest

from platform_core.config import AppConfig

CONFIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def _cfg(name: str) -> AppConfig:
    return AppConfig.from_yaml(os.path.join(CONFIGS, name))


def test_extends_resolves_and_child_overrides_only_named_keys():
    default, enhanced = _cfg("default.yaml"), _cfg("enhanced.yaml")
    # enhanced.yaml names only `name` and `flags`; everything else is inherited
    assert enhanced.models == default.models
    assert enhanced.paths == default.paths
    assert enhanced.retrieval == default.retrieval
    assert enhanced.generation == default.generation
    assert enhanced.name == "enhanced" and default.name == "default"


#: Flags outside the ablation series. allow_abstain is a measured capability, not an
#: enhancement; the phase 8/9 visual flags each depend on a paid pass (vision calls at
#: answer time, an ingest transcription) or on that transcription's index, so they are
#: opted into via configs/enhanced_vqa.yaml rather than being part of plain "enhanced".
NOT_ABLATED = ("allow_abstain", "use_vqa", "use_table_vqa_text",
               "use_page_crop_expansion", "use_cot")


def test_baseline_all_retrieval_flags_off_enhanced_all_on():
    base, enh = _cfg("baseline.yaml"), _cfg("enhanced.yaml")
    ablated = [f.name for f in dataclasses.fields(base.flags) if f.name not in NOT_ABLATED]
    assert not any(getattr(base.flags, f) for f in ablated), "baseline must have every enhancement off"
    assert all(getattr(enh.flags, f) for f in ablated), "enhanced must have every enhancement on"
    # abstention is a measured capability, not an ablated enhancement — on for both
    assert base.flags.allow_abstain and enh.flags.allow_abstain


def test_baseline_and_enhanced_differ_only_in_flags():
    base, enh = _cfg("baseline.yaml"), _cfg("enhanced.yaml")
    for section in ("backends", "models", "paths", "retrieval", "generation", "ingest"):
        assert getattr(base, section) == getattr(enh, section), \
            f"{section} differs — baseline/enhanced must differ ONLY in flags"


def test_vqa_config_turns_on_both_phase9_fixes():
    """enhanced_vqa must carry the reading fix AND both retrievability fixes."""
    vqa = _cfg("enhanced_vqa.yaml")
    assert vqa.flags.use_vqa and vqa.flags.use_table_vqa_text
    # page expansion is what actually made tables retrievable (phase 9 part A)
    assert vqa.flags.use_page_crop_expansion
    assert vqa.text_index_path().endswith("text_layout_tablevqa"), \
        "use_table_vqa_text must select its own index, not mutate text_layout"
    assert 0.0 < vqa.generation.vqa_min_crop_score <= 1.0, "safety gate must be armed"
    # the flags it inherits are unchanged from enhanced
    enh = _cfg("enhanced.yaml")
    for f in ("use_layout_chunking", "use_clip_images", "use_caption_anchor",
              "use_kg", "use_rerank", "allow_abstain"):
        assert getattr(vqa.flags, f) == getattr(enh.flags, f)


def test_two_level_extends_chain_overrides_one_key():
    # enhanced_clip.yaml -> enhanced.yaml -> default.yaml, overriding one key
    enh, clip = _cfg("enhanced.yaml"), _cfg("enhanced_clip.yaml")
    assert clip.flags == enh.flags                       # inherited through 2 levels
    assert clip.models.image_embedding_model != enh.models.image_embedding_model
    assert clip.models.text_embedding_model == enh.models.text_embedding_model


def test_image_index_path_follows_the_encoder():
    enh, clip = _cfg("enhanced.yaml"), _cfg("enhanced_clip.yaml")
    assert enh.image_index_path() != clip.image_index_path(), \
        "encoders must not share an index file"
    assert "biomedclip" in os.path.basename(enh.image_index_path()), \
        "BiomedCLIP is the default encoder from Phase 8 on"
    assert os.path.basename(clip.image_index_path()) == "image"


def test_text_index_path_follows_the_chunker_flag():
    assert _cfg("baseline.yaml").text_index_path().endswith("text_naive")
    assert _cfg("enhanced.yaml").text_index_path().endswith("text_layout")


def test_scale_corpus_differs_from_dev_only_in_corpus_and_paths():
    """The 500-page config must change WHAT is ingested and WHERE it lands — nothing
    else. If a flag or model drifted here, a 50-page and a 500-page result would not
    be comparable and the scale-up would measure two things at once."""
    vqa, scale = _cfg("enhanced_vqa.yaml"), _cfg("scale500.yaml")
    # ONE deliberate flag difference: the transcription pass is not run at this scale,
    # because phase 9 measured it as contributing nothing to retrievability. Every other
    # flag must match, or the two corpora are not measuring the same pipeline.
    import dataclasses as _dc
    for f in _dc.fields(scale.flags):
        if f.name == "use_table_vqa_text":
            assert scale.flags.use_table_vqa_text is False and vqa.flags.use_table_vqa_text is True
            continue
        assert getattr(scale.flags, f.name) == getattr(vqa.flags, f.name), \
            f"flags.{f.name} drifted between the dev and scale-up configs"
    # with no transcription there is no tablevqa index, so it must select the plain one
    assert scale.text_index_path().endswith("text_layout")
    assert scale.models == vqa.models
    assert scale.retrieval == vqa.retrieval
    assert scale.generation == vqa.generation
    assert scale.ingest != vqa.ingest and scale.paths != vqa.paths
    # derived artifacts must not collide with the dev corpus backing reports/ablation.md
    assert scale.paths.chunk_db != vqa.paths.chunk_db
    assert scale.paths.index_dir != vqa.paths.index_dir
    assert scale.paths.reports_dir != vqa.paths.reports_dir
    assert sum(1 for _ in scale.ingest.shards) * scale.ingest.pages_per_shard == 500


def test_duplicate_shard_tag_is_rejected(tmp_path):
    """Tags prefix page ids; two shards sharing one would collide pages silently."""
    p = tmp_path / "dup.yaml"
    p.write_text(f"extends: {os.path.join(CONFIGS, 'enhanced.yaml')}\nname: dup\n"
                 f"ingest:\n  shards:\n    - {{file: a.parquet, tag: v0}}\n"
                 f"    - {{file: b.parquet, tag: v0}}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate ingest shard tag"):
        AppConfig.from_yaml(str(p))


def test_malformed_shard_entry_is_rejected(tmp_path):
    p = tmp_path / "bad_shard.yaml"
    p.write_text(f"extends: {os.path.join(CONFIGS, 'enhanced.yaml')}\nname: bad\n"
                 f"ingest:\n  shards:\n    - {{file: a.parquet}}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty 'file' and 'tag'"):
        AppConfig.from_yaml(str(p))


def test_unknown_key_is_rejected_not_ignored(tmp_path):
    p = tmp_path / "typo.yaml"
    p.write_text(f"extends: {os.path.join(CONFIGS, 'enhanced.yaml')}\n"
                 f"name: typo\nflags:\n  use_reranking: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown key"):
        AppConfig.from_yaml(str(p))


def test_unregistered_image_encoder_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(f"extends: {os.path.join(CONFIGS, 'enhanced.yaml')}\n"
                 f"name: bad\nmodels:\n  image_embedding_model: clip-ViT-L-14\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not registered"):
        AppConfig.from_yaml(str(p))


def test_circular_extends_is_detected(tmp_path):
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("extends: b.yaml\nname: a\n", encoding="utf-8")
    b.write_text("extends: a.yaml\nname: b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="circular extends"):
        AppConfig.from_yaml(str(a))


def test_paths_are_absolute_so_cwd_does_not_matter():
    cfg = _cfg("enhanced.yaml")
    for f in dataclasses.fields(cfg.paths):
        assert os.path.isabs(getattr(cfg.paths, f.name)), f"paths.{f.name} must be absolute"
