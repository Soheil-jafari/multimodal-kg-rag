"""Operational characteristics: latency, memory, build times, cost. No API calls.

    python -m scripts.measure_ops

Retrieval latency and peak memory are measured on this machine now. Ingest time, KG
token spend and the paid-run costs are read from the run logs that produced the locked
results. Anything neither measurable locally nor present in a log is reported as
**not measured** — an estimate dressed as a measurement is worse than a gap.

Retrieval is the whole local pipeline up to the answer: embed the query, search the text
index, expand by caption / page-crop / image / graph, then cross-encoder rerank. The
answer model is never called, so this costs nothing and is the honest number for "how
long before the LLM step starts".
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(ROOT, "reports/scale500")
NOT_MEASURED = "not measured"


def read(p: str) -> str:
    try:
        return open(p, encoding="utf-8", errors="ignore").read()
    except OSError:
        return ""


def from_logs() -> dict:
    """Timings and token spend recorded by the runs that produced the locked results."""
    out = {}
    ing = read(os.path.join(ROOT, "artifacts/scale500/ingest.log"))
    shards = re.findall(r"(val\d+): pages=(\d+) chunks=(\d+) skipped=\d+ \((\d+)s\)", ing)
    tot = re.search(r"TOTAL new: pages=(\d+) chunks=(\d+).*?\[([\d.]+) min\]", ing)
    if shards:
        out["ingest_shards"] = [{"shard": s, "pages": int(p), "chunks": int(c),
                                 "seconds": int(sec)} for s, p, c, sec in shards]
    if tot:
        pages, chunks, mins = int(tot.group(1)), int(tot.group(2)), float(tot.group(3))
        out["ingest_total"] = {"pages": pages, "chunks": chunks, "minutes": mins,
                               "sec_per_page": round(mins * 60 / pages, 2)}
    kg = read(os.path.join(ROOT, "artifacts/scale500/kg_build.log"))
    m = re.search(r"usage: \{'prompt_tokens': (\d+), 'completion_tokens': (\d+), "
                  r"'total_tokens': (\d+)\}\s+cost: \$([\d.]+)", kg)
    if m:
        chunks = re.search(r"extracting over (\d+) chunks", kg)
        out["kg_build"] = {"prompt_tokens": int(m.group(1)),
                           "completion_tokens": int(m.group(2)),
                           "total_tokens": int(m.group(3)), "usd": float(m.group(4)),
                           "chunks": int(chunks.group(1)) if chunks else None}
    costs = {}
    for name, path, pat in [
        ("ablation (3 of 6 configs, resumed invocation)",
         "reports/scale500/ablate_resume.log", r"ACTUAL API cost this invocation: \$([\d.]+)"),
        ("chain-of-thought paired run", "reports/scale500/cot_run.log",
         r"ACTUAL API cost: \$([\d.]+)"),
        ("VQA-vs-text paired run", "reports/scale500/vqa_compare.log",
         r"ACTUAL API cost: \$([\d.]+)"),
    ]:
        m = re.search(pat, read(os.path.join(ROOT, path)))
        if m:
            costs[name] = float(m.group(1))
    out["logged_costs"] = costs
    return out


def measure_retrieval(cfg_path: str, limit: int, warmup: int) -> dict:
    """Per-query wall-clock for the full local retrieval path, plus peak RSS."""
    import psutil

    from evaluation.ablation import ablation_flags
    from evaluation.gold_set import load_gold
    from platform_core.config import AppConfig
    from scripts.ablate import build_harness

    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    t0 = time.perf_counter()
    cfg = AppConfig.from_yaml(cfg_path)
    harness = build_harness(cfg)
    flags = dict(ablation_flags("classic"))["+rerank"]
    retriever = harness._retriever(flags)
    load_s = time.perf_counter() - t0
    rss_loaded = proc.memory_info().rss

    gold = load_gold(cfg.paths.gold_set)[:limit]
    for g in gold[:warmup]:
        retriever.retrieve(g.question)

    lat = []
    peak = rss_loaded
    for g in gold:
        t = time.perf_counter()
        retriever.retrieve(g.question)
        lat.append((time.perf_counter() - t) * 1000.0)
        peak = max(peak, proc.memory_info().rss)
    harness.store.close()
    lat.sort()

    def pct(p):
        return lat[min(len(lat) - 1, int(round(p / 100 * (len(lat) - 1))))]

    return {
        "n_queries": len(lat), "warmup": warmup,
        "p50_ms": round(statistics.median(lat), 1), "p95_ms": round(pct(95), 1),
        "p99_ms": round(pct(99), 1), "mean_ms": round(statistics.fmean(lat), 1),
        "min_ms": round(lat[0], 1), "max_ms": round(lat[-1], 1),
        "cold_start_s": round(load_s, 1),
        "rss_baseline_mb": round(rss0 / 1048576, 0),
        "rss_after_load_mb": round(rss_loaded / 1048576, 0),
        "rss_peak_mb": round(peak / 1048576, 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=ROOT + "/configs/scale500.yaml")
    ap.add_argument("--limit", type=int, default=127)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--skip-retrieval", action="store_true")
    args = ap.parse_args()

    ops = {"machine": {}, "retrieval": None, "from_logs": from_logs()}
    try:
        import platform

        import psutil
        ops["machine"] = {
            "platform": platform.platform(terse=True),
            "python": platform.python_version(),
            "cpu_logical": psutil.cpu_count(logical=True),
            "cpu_physical": psutil.cpu_count(logical=False),
            "ram_total_gb": round(psutil.virtual_memory().total / 1073741824, 1),
            "device": "cpu (no CUDA build installed)",
        }
    except Exception:
        pass

    if not args.skip_retrieval:
        print("measuring retrieval latency locally (no model API calls) ...")
        ops["retrieval"] = measure_retrieval(args.config, args.limit, args.warmup)

    with open(os.path.join(REPORTS, "operational.json"), "w", encoding="utf-8") as f:
        json.dump(ops, f, indent=2)

    m, r, L = ops["machine"], ops["retrieval"], ops["from_logs"]
    md = ["# Operational characteristics", "",
          "Measured on the development machine, not projected. Retrieval latency and "
          "memory were timed locally with no model API calls; ingest time and token spend "
          "come from the logs of the runs that produced the locked results. Anything "
          f"neither locally measurable nor recorded is marked **{NOT_MEASURED}**.", "",
          f"Machine: {m.get('platform', '?')} · {m.get('cpu_physical', '?')} physical / "
          f"{m.get('cpu_logical', '?')} logical cores · {m.get('ram_total_gb', '?')} GB RAM "
          f"· Python {m.get('python', '?')} · **CPU only** (no CUDA build installed; the "
          "code is device-agnostic).", ""]

    if r:
        md += ["## Query latency — retrieval path, no LLM", "",
               f"Full configured retrieval per query: query embedding, text search, "
               f"caption / page-crop / image / graph expansion, cross-encoder rerank. "
               f"{r['n_queries']} gold questions, {r['warmup']} discarded as warm-up.", "",
               "| metric | ms |", "|---|---:|",
               f"| p50 | {r['p50_ms']} |", f"| p95 | {r['p95_ms']} |",
               f"| p99 | {r['p99_ms']} |", f"| mean | {r['mean_ms']} |",
               f"| min / max | {r['min_ms']} / {r['max_ms']} |", "",
               f"Cold start (load BGE + BiomedCLIP + cross-encoder + FAISS indices + "
               f"graph): **{r['cold_start_s']} s**, paid once per process.", "",
               "## Memory", "",
               "| stage | RSS |", "|---|---:|",
               f"| interpreter baseline | {r['rss_baseline_mb']:.0f} MB |",
               f"| after loading all models and indices | {r['rss_after_load_mb']:.0f} MB |",
               f"| **peak during retrieval** | **{r['rss_peak_mb']:.0f} MB** |", "",
               "Three transformer models plus two FAISS indices and the graph are held "
               "concurrently. On a 4 GB GPU they would need sequencing; on CPU they "
               "coexist.", ""]
    else:
        md += ["## Query latency", "", f"{NOT_MEASURED} (skipped).", ""]

    md += ["## End-to-end latency (including the answer model)", "",
           f"**{NOT_MEASURED}.** The evaluation harness records per-question answers and "
           "scores but never recorded per-question wall-clock, and re-running to obtain it "
           "would cost API budget. Retrieval latency above is a lower bound on the "
           "end-to-end figure; the answer and grounding-verification calls dominate it.", ""]

    if "ingest_total" in L:
        t = L["ingest_total"]
        md += ["## Corpus build", "",
               "| stage | measurement | source |", "|---|---|---|",
               f"| OCR ingest, {t['pages']} pages | **{t['minutes']:.1f} min** "
               f"({t['sec_per_page']:.1f} s/page, {t['chunks']} chunks) | run log |"]
        if "kg_build" in L:
            k = L["kg_build"]
            md.append(f"| KG extraction, {k['chunks']} chunks | "
                      f"{k['total_tokens']:,} tokens, **${k['usd']:.4f}** | run log |")
        md += [f"| BGE text index (5,611 + 1,818 units) | {NOT_MEASURED} — wall-clock not "
               f"recorded | — |",
               f"| BiomedCLIP image index (408 crops) | {NOT_MEASURED} — wall-clock not "
               f"recorded | — |", ""]
        md += ["Per-shard OCR: " + " · ".join(
            f"{s['shard']} {s['seconds']}s" for s in L.get("ingest_shards", [])) + ".", "",
            "OCR dominates corpus construction and is single-threaded CPU work; it is the "
            "one stage that would benefit most from parallelism or a GPU OCR backend.", ""]

    md += ["## API cost", "",
           "Per-config cost is **" + NOT_MEASURED + "**: the harness logs answers and "
           "scores per question but not token counts, and the ablation's cost line is "
           "emitted per invocation rather than per config. What was recorded:", "",
           "| run | cost |", "|---|---:|"]
    for k, v in L.get("logged_costs", {}).items():
        md.append(f"| {k} | ${v:.2f} |")
    if "kg_build" in L:
        md.append(f"| KG extraction over the corpus | ${L['kg_build']['usd']:.4f} |")
    md += ["", "The first ablation invocation crashed on a rate limit before printing its "
           "cost line, so the six-config total is known only to about **$5**; the three "
           "configs in the resumed invocation cost $2.50 together, i.e. roughly $0.83 per "
           "config at 127 questions. That per-config figure is a division of a measured "
           "total, not a separately measured quantity.", "",
           "Throughput note: the answer model is capped at 30,000 tokens/minute on this "
           "account, and a ten-chunk answer sits against that cap for a full run — which "
           "is why the runner retries with backoff and can resume from a completed "
           "per-question log rather than re-buying it.", ""]

    with open(os.path.join(REPORTS, "operational.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print("wrote reports/scale500/operational.md and .json")


if __name__ == "__main__":
    main()
