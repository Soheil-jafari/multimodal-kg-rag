"""Ablation runner: the same gold set across configs, one flag added at a time.

baseline (all off) -> +layout -> +clip -> +caption -> +kg -> +rerank (= enhanced).
allow_abstain is held ON across the series (abstention is a capability, not one of
the ablated retrieval flags). Renders one config x metric table per category.
"""
from __future__ import annotations

from platform_core.config import RetrievalFlags

_STEPS = [
    ("baseline", {}),
    ("+layout", {"use_layout_chunking": True}),
    ("+clip", {"use_clip_images": True}),
    ("+caption", {"use_caption_anchor": True}),
    ("+kg", {"use_kg": True}),
    ("+rerank", {"use_rerank": True}),
]

#: The phase 8/9 flags, as further ablation steps. They are NOT in the classic series
#: because they did not exist when reports/ablation.md was measured, and because
#: `ablation_flags` builds its flags from scratch — so passing configs/enhanced_vqa.yaml
#: to the classic series runs them OFF while describe() prints them ON. Ordered so the
#: phase-9 finding is reproduced rather than asserted: +tablevqa indexes tables by their
#: transcription (measured on 4 dev questions to make retrieval WORSE, because a table
#: states values, not the concepts linking them to a question), then +pagecrop reaches
#: the crop through its page's prose (the fix that actually worked), then +vqa reads it.
_PHASE9_STEPS = [
    ("+tablevqa", {"use_table_vqa_text": True}),
    ("+pagecrop", {"use_page_crop_expansion": True}),
    ("+vqa", {"use_vqa": True}),
]

SERIES = {"classic": _STEPS, "phase9": _STEPS + _PHASE9_STEPS}
_METRICS = [("recall@5", "R@5"), ("prec@5", "P@5"), ("mrr", "MRR"),
            ("ndcg", "nDCG"), ("correctness", "Correct"), ("faithfulness", "Faith"),
            ("decision_acc", "Decis")]
_CATS = ["single_fact", "multi_hop", "figure", "figure_value", "text_derived", "unanswerable"]


def ablation_flags(series: str = "classic"):
    """The series, as (name, flags). Baseline is every enhancement OFF — pure BGE text.

    Flags are built from scratch rather than from the config's own `flags` block, which
    is what keeps `baseline` genuinely pure no matter which config supplies the paths and
    models. The consequence is that a flag absent from the chosen series is OFF for every
    step, so a config declaring it (e.g. enhanced_vqa) will not have it measured — pick
    the "phase9" series to ablate those.
    """
    if series not in SERIES:
        raise ValueError(f"unknown ablation series {series!r}; known: {sorted(SERIES)}")
    cur = {"allow_abstain": True}
    out = []
    for name, delta in SERIES[series]:
        cur = dict(cur, **delta)
        out.append((name, RetrievalFlags(**cur)))
    return out


def _fmt(v):
    return f"{v:.2f}" if isinstance(v, (int, float)) else "  -"


def _table(names, groups):
    head = "| config | " + " | ".join(lbl for _, lbl in _METRICS) + " |"
    sep = "|" + "---|" * (len(_METRICS) + 1)
    rows = [f"| {n} | " + " | ".join(_fmt(g.get(k)) for k, _ in _METRICS) + " |"
            for n, g in zip(names, groups)]
    return "\n".join([head, sep, *rows])


def render(reports: dict) -> str:
    """reports: ordered {config_name: harness_report}."""
    names = list(reports)
    out = ["### Overall (rows = configs)",
           _table(names, [reports[n]["overall"] for n in names])]
    # abstention split
    out.append("\nabstention — attempt-rate on answerable / abstain-rate on unanswerable:")
    for n in names:
        r = reports[n]
        out.append(f"  {n:10s} attempt={_fmt(r['attempt_rate_answerable'])} "
                   f"abstain={_fmt(r['abstain_rate_unanswerable'])}")
    # per category
    for cat in _CATS:
        groups = [reports[n]["per_category"].get(cat, {}) for n in names]
        n_cat = next((g.get("n") for g in groups if g), 0)
        flag = "  ⚠ LOW-N (<5)" if n_cat and n_cat < 5 else ""
        out.append(f"\n### {cat}  (n={n_cat}){flag}")
        out.append(_table(names, groups))
    # per-predicate (enhanced config only), low-n + n=1 footnoted
    enh = reports[names[-1]]["per_predicate"]
    out.append("\n### per-predicate correctness — enhanced config (single_fact + figure_value)")
    out.append("| predicate | n | Correct | R@5 | note |")
    out.append("|---|---|---|---|---|")
    for p in ["treats", "causes", "inhibits", "increases", "decreases", "transforms_to", "occurs_in", "measured_by"]:
        g = enh.get(p)
        if not g:
            continue
        note = "n=1 — not statistically meaningful" if g["n1"] else ("low-n (<5)" if g["low_n"] else "")
        out.append(f"| {p} | {g['n']} | {_fmt(g['correctness'])} | {_fmt(g['recall@5'])} | {note} |")
    out.append("\n_Low-sample (n<5) categories/predicates are flagged; n=1 predicates are "
               "reported for transparency only, not as reliable scores._")
    return "\n".join(out)


def run_ablation(harness, gold, limit=None, series="classic", resume=False) -> dict:
    reports = {}
    for name, flags in ablation_flags(series):
        reports[name], _ = harness.run_config(name, flags, gold, limit=limit, resume=resume)
    return reports
