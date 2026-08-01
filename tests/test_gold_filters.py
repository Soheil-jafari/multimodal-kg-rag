"""Gold-set filter calibration, replayed against the real 55-question dev set.

The q061-mismatch filter has two jobs that pull against each other: reject a question
naming rows its source table does not contain (the defect that made q061 unanswerable
by any route), while keeping questions that merely *paraphrase* their source. A single
tight threshold did the first job and failed the second — replay showed it also rejected
q038, q039 and q062, all of which are correct against their crops. Deleting those would
have biased the surviving crop-backed set toward questions that lexically echo their own
caption, and figure retrieval is caption-primary, so that bias would have inflated the
very numbers the gold set exists to measure.

These tests pin the calibration to the data rather than to a guess. They skip when the
dev artifacts are absent (a fresh clone), rather than failing.
"""
from __future__ import annotations

import json
import os

import pytest

from scripts.build_gold import (
    MISMATCH_DROP, MISMATCH_FLAG, content_tokens, covered, mismatch, region_haystack, stem,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.path.join(ROOT, "domain_packs/biomed/gold/gold_set.jsonl")
DB = os.path.join(ROOT, "artifacts/dev/chunks.sqlite")
CROP_CATS = ("figure", "figure_value")

#: The q061 defect, verbatim from domain_packs/biomed/gold/CLEANUP.md. Its source table's
#: rows are exclusion thresholds; it has no exposure-category rows at all.
Q061 = "q061"


def _dev():
    if not (os.path.exists(GOLD) and os.path.exists(DB)):
        pytest.skip("dev corpus artifacts not present")
    from platform_core.stores.chunk_store import SQLiteChunkStore

    store = SQLiteChunkStore(DB)
    cache: dict = {}

    def get(cid):
        if cid not in cache:
            try:
                cache[cid] = store.get(cid)
            except KeyError:
                cache[cid] = None
        return cache[cid]

    gold = [json.loads(l) for l in open(GOLD, encoding="utf-8")]
    return gold, get


def _ratios():
    gold, get = _dev()
    out = {}
    for r in gold:
        if r["category"] in CROP_CATS:
            ratio, missing = mismatch(r["question"],
                                      region_haystack(get, r["supporting_chunk_ids"]))
            out[r["qid"]] = (ratio, missing)
    return out


def test_q061_is_dropped():
    ratios = _ratios()
    assert Q061 in ratios, "q061 must still be present in the dev gold set"
    ratio, missing = ratios[Q061]
    assert ratio >= MISMATCH_DROP, (
        f"q061 scored {ratio:.2f} < drop threshold {MISMATCH_DROP} — the one defect this "
        f"filter exists to catch would now pass. Missing terms: {missing}"
    )


def test_no_other_crop_backed_question_is_dropped():
    """Every other crop-backed dev question is correct against its crop, so dropping any
    of them is a false positive — and a biased one, since it selects for lexical echo."""
    ratios = _ratios()
    wrongly = {q: (round(r, 2), m) for q, (r, m) in ratios.items()
               if q != Q061 and r >= MISMATCH_DROP}
    assert not wrongly, f"false positives — these are good questions: {wrongly}"


def test_drop_threshold_keeps_a_real_margin_over_the_worst_good_question():
    """q061 must not merely sit above the line; it must sit clear of the good items, or
    the calibration is fitting noise."""
    ratios = _ratios()
    good = [r for q, (r, _) in ratios.items() if q != Q061]
    if not good:
        pytest.skip("no crop-backed questions besides q061")
    assert ratios[Q061][0] - max(good) >= 0.15, (
        f"q061 {ratios[Q061][0]:.2f} vs worst good {max(good):.2f} — too close to separate"
    )


def test_flag_band_is_below_the_drop_band():
    assert 0.0 < MISMATCH_FLAG < MISMATCH_DROP <= 1.0


@pytest.mark.parametrize("question_word,source_word", [
    ("viability", "viable eggs were counted"),      # derivation, not inflection
    ("degradation", "the compound was degraded"),
    ("exposures", "exposure to benzene"),           # plain inflection
    ("concentrations", "at a concentration of"),
    ("mortality", "mortal"),
])
def test_stemming_matches_derivational_variants(question_word, source_word):
    """These exact pairs were false-positive causes on the dev set under a prefix rule."""
    assert covered(question_word, source_word.lower()), (
        f"{question_word!r} should be covered by {source_word!r} "
        f"(stem={stem(question_word)!r})"
    )


def test_stemming_does_not_collapse_unrelated_words():
    hay = "exclusion thresholds and urr columns"
    for tok in ("category", "cumulative", "polycyclic"):
        assert not covered(tok, hay), f"{tok!r} wrongly matched {hay!r}"


def test_content_tokens_drops_framing_words_but_keeps_specifics():
    toks = content_tokens("What was the mean age in years of the study cohort?")
    assert "cohort" in toks and "years" in toks
    for stopword in ("mean", "study", "what", "the"):
        assert stopword not in toks
