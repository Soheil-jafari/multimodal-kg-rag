"""Entry point: evaluate ONE config over the gold set.

    python -m scripts.evaluate --config configs/baseline.yaml
    python -m scripts.evaluate --config configs/enhanced.yaml
    python -m scripts.evaluate --config configs/scale500.yaml

Unlike the ablation runner, this honours the config's own `flags` block verbatim,
so baseline vs enhanced is entirely a matter of which YAML is passed.
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

from evaluation.gold_set import load_gold
from platform_core.config import AppConfig
from scripts.ablate import R, build_harness


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=R + "/configs/enhanced.yaml")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    print(cfg.describe())
    harness = build_harness(cfg)
    report, _ = harness.run_config(cfg.name, cfg.flags, load_gold(cfg.paths.gold_set),
                                   limit=args.limit)
    o = report["overall"]
    print(f"\n== {cfg.name} (overall) ==")
    for k in ("recall@5", "prec@5", "mrr", "ndcg", "correctness", "faithfulness", "decision_acc"):
        v = o.get(k)
        print(f"  {k:14s} {v:.3f}" if isinstance(v, float) else f"  {k:14s} -")
    print(f"  attempt(answerable)={report['attempt_rate_answerable']:.2f} "
          f"abstain(unanswerable)={report['abstain_rate_unanswerable']:.2f}")


if __name__ == "__main__":
    main()
