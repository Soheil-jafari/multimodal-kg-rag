"""Project the API cost of a corpus's paid passes, from MEASURED per-unit rates.

    python -m scripts.estimate_cost --config configs/scale500.yaml --gold 130

Every rate below is an actual invoice from a completed run recorded in
EXPERIMENT_LOG.md, divided by the units that run covered — not a token guess. The
counts come from the config's own chunk store, so the projection describes the
corpus that will really be processed.

Rates are per-unit and the corpus is ~10x the dev corpus, so anything superlinear
(retries, longer tables) makes these FLOORS, not ceilings. Treated as such below.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from platform_core.config import AppConfig
from platform_core.stores.chunk_store import SQLiteChunkStore
from platform_core.types import RegionType

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
DEFAULT_CONFIG = R + "/configs/scale500.yaml"

#: (measured cost, units, what one unit is, where it was measured)
RATES = {
    "kg":          (0.083, 605, "text/list/table chunk", "Phase 3 dev KG build"),
    "transcribe":  (0.430, 44,  "table crop",            "Phase 9 part A"),
    "gold":        (0.0019, 59, "gold question phrased", "Phase 6A gold build"),
    "ablation":    (2.360, 55 * 6, "question x config",  "Phase 6D full ablation"),
    "vqa":         (0.060, 4,   "vision answer",         "Phase 8 VQA run"),
    "discovery":   (0.0044, 30, "chunk free-extracted",  "Phase 3 stage 1"),
}


def rate(key: str) -> float:
    cost, units, _, _ = RATES[key]
    return cost / units


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--gold", type=int, default=130, help="planned gold-set size")
    ap.add_argument("--series", default="classic", help="ablation series (classic|phase9)")
    ap.add_argument("--transcribe", action="store_true",
                    help="include the table-transcription pass (off by default: phase 9 "
                         "measured it as contributing nothing to retrievability)")
    ap.add_argument("--vqa-compare", action="store_true", default=True,
                    help="include a separate paired VQA-vs-text comparison on crop-backed Qs")
    ap.add_argument("--spent", type=float, default=5.35,
                    help="API spend already booked in EXPERIMENT_LOG (default: sum of phases 3-9)")
    ap.add_argument("--budget", type=float, default=8.0, help="total credit")
    args = ap.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    store = SQLiteChunkStore(cfg.paths.chunk_db)
    chunks = list(store.iter_chunks())
    n_pages = len(store.page_ids())
    n_text = sum(1 for c in chunks
                 if c.region_type.value in ("text", "list", "table") and c.text.strip())
    n_tables = sum(1 for c in chunks
                   if c.region_type == RegionType.TABLE and c.image_path)
    store.close()

    from evaluation.ablation import ablation_flags

    steps = ablation_flags(args.series)
    n_configs = len(steps)
    # A vision call happens only in a config that actually has use_vqa on. In the classic
    # series that is none of them, so the ablation costs zero vision tokens — counting
    # them per-config overstated this by ~$1.9.
    vqa_configs = sum(1 for _, f in steps if f.use_vqa)
    crop_qs = round(args.gold * 0.16)  # figure + figure_value share of the phase-6A mix

    rows = [
        ("predicate discovery (30-chunk sample)", 30, rate("discovery"), "fixed sample, not scaled"),
        ("KG build", n_text, rate("kg"), "every text/list/table chunk"),
        ("gold-set phrasing", args.gold, rate("gold"), "gpt-4o-mini, phrasing only"),
        (f"ablation ({n_configs} configs x {args.gold} Qs)", args.gold * n_configs,
         rate("ablation"), "gpt-4o answer + judge, mini verify"),
        (f"VQA calls inside the ablation ({vqa_configs} config(s) with use_vqa)",
         crop_qs * vqa_configs, rate("vqa"),
         "zero for the classic series" if not vqa_configs else "crop-backed Qs"),
    ]
    if args.transcribe:
        rows.insert(3, ("table transcription", n_tables, rate("transcribe"),
                        "gpt-4o vision, once per table"))
    if args.vqa_compare:
        # one paired run: each crop-backed question answered twice off ONE retrieval
        rows.append(("separate VQA-vs-text comparison", crop_qs * 2, rate("vqa"),
                     "paired, off a single retrieval — the phase-8 method"))

    print(f"corpus: {cfg.name}  ({n_pages} pages, {len(chunks)} chunks, "
          f"{n_text} text-bearing, {n_tables} table crops)")
    print(f"planned gold set: {args.gold} questions x {n_configs} ablation configs "
          f"(series={args.series}); transcription "
          f"{'INCLUDED' if args.transcribe else 'SKIPPED (phase-9 evidence)'}\n")
    print(f"{'pass':42s} {'units':>7s} {'$/unit':>10s} {'$ est':>8s}  note")
    print("-" * 104)
    total = 0.0
    for name, units, r, note in rows:
        cost = units * r
        total += cost
        print(f"{name:42s} {units:7d} {r:10.6f} {cost:8.2f}  {note}")
    print("-" * 104)
    print(f"{'TOTAL (floor)':42s} {'':7s} {'':10s} {total:8.2f}")

    left = args.budget - args.spent
    print(f"\nbudget: ${args.budget:.2f} credit - ${args.spent:.2f} already spent "
          f"= ${left:.2f} remaining")
    verdict = "FITS" if total <= left else f"OVER BY ${total - left:.2f}"
    print(f"projected ${total:.2f} vs ${left:.2f} available  ->  {verdict}")

    if total > left:
        print("\nlevers, largest saving first:")
        t_cost = n_tables * rate("transcribe")
        abl = args.gold * n_configs * rate("ablation")
        print(f"  drop table transcription            -${t_cost:6.2f}  "
              f"phase 9 measured it contributes NOTHING to retrievability; page-crop "
              f"expansion is the whole gain. It only sharpens the VQA gate's scoring.")
        print(f"  gold set {args.gold}->80 questions           "
              f"-${(args.gold - 80) * n_configs * rate('ablation'):6.2f}  "
              f"still 45% larger than the published 55")
        print(f"  ablation on 3 configs not {n_configs}        "
              f"-${abl / n_configs * (n_configs - 3):6.2f}  "
              f"baseline / +layout / full — loses the per-flag attribution")
        print(f"  judge with gpt-4o-mini              -~40% of ${abl:.2f}  "
              f"cheaper judge; would need a re-validation against the gpt-4o judge")


if __name__ == "__main__":
    main()
