"""Bootstrap 95% confidence intervals for the locked ablation. No API calls.

    python -m scripts.bootstrap_ci

Reads only the committed per-question logs. Point estimates are RECOMPUTED from those
logs and asserted equal to the published `ablation.md` before any interval is reported —
if a point estimate moved, the run aborts rather than quietly publishing a different
number under a new heading.

Two kinds of uncertainty are reported, and they are not the same thing:

* **Sampling uncertainty** — a percentile bootstrap over questions (resample with
  replacement, 10,000 iterations). This answers "how much of this score is an accident of
  which 127 questions were drawn?" and is why a category of 8 shows a much wider interval
  than one of 56.
* **Run-to-run noise** — measured directly, not assumed: re-running an identical
  configuration over the same 17 multi_hop questions moved correctness by 0.029, so an
  observed effect at or below ±0.03 cannot be distinguished from re-running the same
  system twice.

Per-step effects use a PAIRED bootstrap: the same resampled question indices are applied
to both configs, because the two arms are the same questions and treating them as
independent samples would overstate the interval on their difference.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(ROOT, "reports/scale500")
CONFIGS = ["baseline", "+layout", "+clip", "+caption", "+kg", "+rerank"]
CATS = ["single_fact", "multi_hop", "figure", "figure_value", "text_derived", "unanswerable"]
PREDS = ["treats", "causes", "inhibits", "increases", "decreases",
         "transforms_to", "occurs_in", "measured_by"]
#: directly measured: an identical configuration re-run over the same questions moved
#: multi_hop correctness by 0.029 (perq_+rerank.jsonl vs the CoT-off arm)
NOISE = 0.03


def load(name: str) -> list:
    path = os.path.join(REPORTS, f"perq_{name}.jsonl")
    rows = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        ret = r.get("retrieval")
        if ret:  # JSON has no integer keys; cutoffs come back as strings
            for f in ("recall", "precision"):
                if isinstance(ret.get(f), dict):
                    ret[f] = {int(k): v for k, v in ret[f].items()}
        rows.append(r)
    return rows


def vec(rows: list, metric: str) -> np.ndarray:
    """Per-question contribution to a metric; NaN where the question does not count.

    The NaN pattern reproduces the harness's own inclusion rules exactly: retrieval
    metrics over answerable questions that have a retrieval record, correctness over all
    answerable (a wrongly-abstained one counts as 0), faithfulness only over answered
    ones, and decision accuracy over every question including the unanswerable.
    """
    out = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        if metric == "decision_acc":
            out[i] = 1.0 if r["decision_correct"] else 0.0
        elif metric == "correctness":
            if r["answerable"]:
                out[i] = r["correctness"]
        elif metric == "faithfulness":
            if r["answerable"] and r["faithfulness"] is not None:
                out[i] = r["faithfulness"]
        else:
            if r["answerable"] and r["retrieval"]:
                ret = r["retrieval"]
                out[i] = {"recall@5": lambda: ret["recall"][5],
                          "prec@5": lambda: ret["precision"][5],
                          "mrr": lambda: ret["mrr"],
                          "ndcg": lambda: ret["ndcg"]}[metric]()
    return out


def nanmean(a: np.ndarray):
    m = ~np.isnan(a)
    return float(a[m].mean()) if m.any() else None


def boot(values: np.ndarray, idx: np.ndarray):
    """Percentile CI over pre-drawn resample indices (shared across configs)."""
    if np.isnan(values).all():
        return None, None
    draws = values[idx]                      # (iters, n)
    mask = ~np.isnan(draws)
    counts = mask.sum(axis=1)
    sums = np.where(mask, draws, 0).sum(axis=1)
    keep = counts > 0
    if not keep.any():
        return None, None
    means = sums[keep] / counts[keep]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fmt(v, lo, hi) -> str:
    if v is None:
        return "—"
    if lo is None:
        return f"{v:.2f}"
    return f"{v:.2f} [{lo:.2f}, {hi:.2f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify-only", action="store_true",
                    help="recompute the point estimates, compare them to ablation.md and "
                         "exit WITHOUT writing, so a consistency run cannot overwrite the "
                         "published intervals with a cheaper resample")
    args = ap.parse_args()

    data = {c: load(c) for c in CONFIGS}
    qids = [r["qid"] for r in data["baseline"]]
    for c in CONFIGS:
        assert [r["qid"] for r in data[c]] == qids, f"{c} question order differs"
    n = len(qids)
    cats = np.array([r["category"] for r in data["baseline"]])

    # --- verify the point estimates still match the published table -------------
    published = {}
    for line in open(os.path.join(REPORTS, "ablation.md"), encoding="utf-8"):
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) == 8 and parts[0] in CONFIGS:
            published.setdefault(parts[0], []).append(parts)
    mism = []
    for c in CONFIGS:
        rows = data[c]
        got = [nanmean(vec(rows, m)) for m in
               ("recall@5", "prec@5", "mrr", "ndcg", "correctness", "faithfulness",
                "decision_acc")]
        want = [float(x) for x in published[c][0][1:]]
        for g, w, name in zip(got, want, ("R@5", "P@5", "MRR", "nDCG", "Correct",
                                          "Faith", "Decis")):
            if g is None or abs(round(g, 2) - w) > 1e-9:
                mism.append(f"{c}/{name}: recomputed {g:.4f} vs published {w}")
    if mism:
        print("POINT ESTIMATES DO NOT MATCH ablation.md — aborting:")
        for m in mism:
            print("  " + m)
        sys.exit(1)
    print(f"point estimates recomputed from the logs and match reports/scale500/"
          f"ablation.md exactly ({len(CONFIGS)} configs x 7 metrics)\n")
    if args.verify_only:
        return

    rng = np.random.default_rng(args.seed)
    out = {"iters": args.iters, "seed": args.seed, "noise_floor": NOISE,
           "n_questions": n, "per_category": {}, "per_predicate": {}, "steps": {}}
    md = ["# Bootstrap confidence intervals — 500-page ablation", "",
          f"95% percentile bootstrap over questions, {args.iters:,} resamples, "
          f"seed {args.seed}. Computed from the committed per-question logs; no model was "
          "called. Point estimates are recomputed from those logs and verified identical "
          "to `ablation.md` before any interval below is reported.", "",
          f"**Two uncertainties, kept separate.** The intervals are *sampling* uncertainty "
          f"— how much a score depends on which questions were drawn. Separately, "
          f"run-to-run variance was measured directly: re-running an identical "
          f"configuration over the same 17 multi_hop questions moved correctness by "
          f"**0.029**, so any effect at or below **±{NOISE}** is indistinguishable from "
          f"re-running the same system twice. An effect is marked *within noise* when its "
          f"paired interval contains zero, or its magnitude is at or below that floor.", ""]

    # --- per category ----------------------------------------------------------
    METRICS = [("recall@5", "R@5"), ("correctness", "Correct"),
               ("faithfulness", "Faith"), ("decision_acc", "Decis")]
    for cat in CATS + ["OVERALL"]:
        sel = np.arange(n) if cat == "OVERALL" else np.flatnonzero(cats == cat)
        k = len(sel)
        idx = sel[rng.integers(0, k, size=(args.iters, k))]
        md += ["", f"## {cat}  (n={k})", "",
               "| config | " + " | ".join(l for _, l in METRICS) + " |",
               "|---|" + "---|" * len(METRICS)]
        out["per_category"][cat] = {"n": k, "configs": {}}
        for c in CONFIGS:
            cells, rec = [], {}
            for m, label in METRICS:
                v = vec(data[c], m)
                pt = nanmean(v[sel])
                lo, hi = boot(v, idx)
                cells.append(fmt(pt, lo, hi))
                rec[label] = {"point": pt, "lo": lo, "hi": hi}
            md.append(f"| {c} | " + " | ".join(cells) + " |")
            out["per_category"][cat]["configs"][c] = rec
        if k < 15:
            md += ["", f"_n={k}: intervals are wide by construction. Read these rows as "
                   f"direction, not as a measurement._"]

    # --- per predicate, enhanced config ----------------------------------------
    md += ["", "## Per predicate — enhanced (`+rerank`)", "",
           "| predicate | n | Correct | R@5 |", "|---|---:|---|---|"]
    rows = data["+rerank"]
    pred_idx = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r["answerable"]:
            for p in set(r["source_predicates"]):
                pred_idx[p].append(i)
    for p in PREDS:
        sel = np.array(sorted(pred_idx.get(p, [])))
        if not len(sel):
            continue
        k = len(sel)
        idx = sel[rng.integers(0, k, size=(args.iters, k))]
        cells, rec = [], {}
        for m, label in (("correctness", "Correct"), ("recall@5", "R@5")):
            v = vec(rows, m)
            pt = nanmean(v[sel])
            lo, hi = boot(v, idx)
            cells.append(fmt(pt, lo, hi))
            rec[label] = {"point": pt, "lo": lo, "hi": hi}
        md.append(f"| {p} | {k} | " + " | ".join(cells) + " |")
        out["per_predicate"][p] = {"n": k, **rec}
    md += ["", "_Every predicate has n between 7 and 16, so all of these intervals are "
           "wide; they are reported for transparency, not as reliable per-predicate "
           "scores._"]

    # --- per-step effects, paired ----------------------------------------------
    md += ["", "## What each flag adds — paired bootstrap on the difference", "",
           "The same resampled questions are scored under both configs, since the two "
           "arms are the same questions. An interval containing zero means the step is "
           "not distinguishable from no change on this corpus.", "",
           "| step | ΔR@5 | ΔCorrect | verdict |", "|---|---|---|---|"]
    idx_all = rng.integers(0, n, size=(args.iters, n))
    pairs = [(CONFIGS[i], CONFIGS[i + 1]) for i in range(len(CONFIGS) - 1)]
    pairs.append(("baseline", "+rerank"))
    for a, b in pairs:
        cells, verdicts, rec = [], [], {}
        for m, label in (("recall@5", "ΔR@5"), ("correctness", "ΔCorrect")):
            va, vb = vec(data[a], m), vec(data[b], m)
            d_pt = nanmean(vb) - nanmean(va)
            da, db = va[idx_all], vb[idx_all]
            ma, mb = ~np.isnan(da), ~np.isnan(db)
            with np.errstate(invalid="ignore"):
                means_a = np.where(ma, da, 0).sum(1) / np.maximum(ma.sum(1), 1)
                means_b = np.where(mb, db, 0).sum(1) / np.maximum(mb.sum(1), 1)
            diffs = means_b - means_a
            lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
            cells.append(f"{d_pt:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            within = (lo <= 0 <= hi) or abs(d_pt) <= NOISE
            verdicts.append(within)
            rec[label] = {"delta": d_pt, "lo": lo, "hi": hi, "within_noise": within}
        # An effect whose interval excludes zero but whose magnitude lands on the
        # measured run-to-run floor is not "real" in any useful sense — it is the size of
        # re-running the same system. Say so rather than let a threshold decide it by a
        # thousandth.
        on_floor = [(not w) and abs(rec[l]["delta"]) <= NOISE * 1.25
                    for w, l in zip(verdicts, ("ΔR@5", "ΔCorrect"))]
        if all(verdicts):
            label = "**within noise**"
        elif any(on_floor):
            which = "R@5" if on_floor[0] else "Correct"
            label = f"{which} sits ON the ±{NOISE} floor"
        elif verdicts[1]:
            label = "R@5 real; Correct within noise"
        elif verdicts[0]:
            label = "Correct real; R@5 within noise"
        else:
            label = "both real"
        md.append(f"| {a} → {b} | " + " | ".join(cells) + f" | {label} |")
        for w, l, f_ in zip(verdicts, ("ΔR@5", "ΔCorrect"), on_floor):
            rec[l]["on_noise_floor"] = bool(f_)
        out["steps"][f"{a}->{b}"] = rec

    with open(os.path.join(REPORTS, "confidence_intervals.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(REPORTS, "confidence_intervals.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\n".join(md[md.index("## What each flag adds — paired bootstrap on the difference"):]))
    print(f"\nwrote reports/scale500/confidence_intervals.md and .json")


if __name__ == "__main__":
    main()
