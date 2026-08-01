"""Human-in-the-loop review dashboard over a LOCKED evaluation run.

    streamlit run scripts/review_app.py -- --config configs/scale500.yaml

A review and visualisation layer, nothing more. It reads `review_queue.jsonl` (built by
`scripts.build_review_queue`) and appends reviewer verdicts to `review_feedback.jsonl`.
It never writes to `ablation.md`, any `perq_*.jsonl`, the gold set or the chunk store, so
no frozen number can move because someone opened the dashboard.

The queue is sorted worst-first by `priority`, so a reviewer with limited time spends it
where the system is least sure. Two things are deliberately hidden behind a toggle: the
gold answer and the automated judge's verdict. A reviewer who sees the expected answer
first is comparing strings; one who sees only the question, the answer and the cited
evidence is doing the job the loop exists for — and can then check both themselves and the
judge by revealing.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
VERDICTS = ("correct", "incorrect", "needs-review")
PROV_HELP = {
    "text": "dense BGE hit on the passage itself",
    "caption-link": "reached through its caption (primary figure route)",
    "graph-hop": "reached by walking the knowledge graph from a retrieved chunk",
    "clip-image": "BiomedCLIP image similarity (weak secondary signal)",
    "page-crop": "pulled in because its page's prose matched",
}


def parse_args():
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=ROOT + "/configs/scale500.yaml")
    known, _ = ap.parse_known_args(argv)
    return known


@st.cache_data(show_spinner=False)
def load_queue(path: str, mtime: float):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_feedback(path: str) -> dict:
    """Last verdict per qid. Append-only log, so the newest entry wins."""
    out: dict = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[r["qid"]] = r
    return out


def record(path: str, entry: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def conf_bar(value, label: str) -> str:
    if value is None:
        return f"{label}: n/a"
    blocks = int(round(value * 20))
    return f"{label}: {'█' * blocks}{'░' * (20 - blocks)}  {value:.2f}"


def main() -> None:
    args = parse_args()
    from platform_core.config import AppConfig

    cfg = AppConfig.from_yaml(args.config)
    qpath = os.path.join(cfg.paths.reports_dir, "review_queue.jsonl")
    fpath = os.path.join(cfg.paths.reports_dir, "review_feedback.jsonl")

    st.set_page_config(page_title="Answer review queue", layout="wide")
    st.title("Answer review queue")

    if not os.path.exists(qpath):
        st.error(f"No review queue at `{qpath}`.\n\n"
                 f"Build it first:\n\n"
                 f"`python -m scripts.build_review_queue --config {args.config} --step +rerank`")
        return

    rows = load_queue(qpath, os.path.getmtime(qpath))
    feedback = load_feedback(fpath)
    step = rows[0].get("source_step", "?") if rows else "?"
    st.caption(
        f"`{cfg.name}` · ablation column `{step}` · {len(rows)} answers, sorted "
        f"least-confident first. Read-only over the locked run — verdicts append to "
        f"`{os.path.basename(fpath)}` and change no published metric."
    )

    # ---------------- sidebar ----------------
    sb = st.sidebar
    sb.header("Queue")
    cats = sorted({r["category"] for r in rows})
    pick_cat = sb.multiselect("Category", cats, default=cats)
    show = sb.radio("Show", ["Needs attention", "Unreviewed only", "Everything"], index=0)
    reveal = sb.toggle("Reveal gold answer and judge verdict", value=False,
                       help="Off by default so the first judgement comes from the cited "
                            "evidence rather than from the expected string.")
    per_page = sb.slider("Items per page", 5, 50, 10)

    sb.divider()
    sb.subheader("How confidence is computed")
    sb.markdown(
        f"""Two signals the system has **at inference time** — no gold labels, or this
would rank items by how right they are, which a live queue cannot know.

- **Retrieval strength** `R` — top query-context cosine, the same value the abstention
  gate thresholds at `{cfg.generation.abstain_min_score}`. Normalised
  `Rn = clip((R − {cfg.generation.abstain_min_score}) / (0.85 − {cfg.generation.abstain_min_score}), 0, 1)`.
- **Grounding** `G` — fraction of answer sentences supported by their cited chunk
  (threshold `{cfg.generation.grounding_min_sim}`, LLM verifier on the borderline band).

**Answered:** `confidence = 0.5·Rn + 0.5·G`, priority `= 1 − confidence`.

**Abstained:** confidence is undefined — there is no answer to grade — so priority `= Rn`.
That inversion is the point: abstaining on weak retrieval is usually right, while
abstaining on *strong* retrieval is the false-abstention failure mode worth a human's time.

Weights are a flat 0.5/0.5 by choice. Fitting them against the locked correctness column
would turn the score back into a correctness proxy."""
    )

    # ---------------- filter ----------------
    sel = [r for r in rows if r["category"] in pick_cat]
    if show == "Needs attention":
        sel = [r for r in sel
               if r["abstained"] or (r["confidence"] is not None and r["confidence"] < 0.6)]
    elif show == "Unreviewed only":
        sel = [r for r in sel if r["qid"] not in feedback]

    # ---------------- summary ----------------
    done = sum(1 for r in rows if r["qid"] in feedback)
    abst = sum(1 for r in rows if r["abstained"])
    lowc = sum(1 for r in rows
               if not r["abstained"] and r["confidence"] is not None and r["confidence"] < 0.5)
    false_abst = sum(1 for r in rows if r["abstained"] and r["retrieval_norm"] >= 0.5)
    c = st.columns(5)
    c[0].metric("In queue", len(rows))
    c[1].metric("Reviewed", done, f"{done - len(rows)}" if done < len(rows) else "complete")
    c[2].metric("Abstained", abst)
    c[3].metric("Abstained, strong retrieval", false_abst,
                help="Evidence was there and the system declined — the highest-value cases")
    c[4].metric("Confidence < 0.50", lowc)
    if done:
        tally = {v: sum(1 for r in feedback.values() if r["verdict"] == v) for v in VERDICTS}
        st.caption("Reviewer verdicts so far — "
                   + " · ".join(f"**{v}** {tally[v]}" for v in VERDICTS))
    st.divider()

    if not sel:
        st.success("Nothing matches the current filter.")
        return

    page = st.number_input(f"Page (of {max(1, (len(sel) - 1) // per_page + 1)})",
                           min_value=1, max_value=max(1, (len(sel) - 1) // per_page + 1),
                           value=1, step=1)
    window = sel[(page - 1) * per_page: page * per_page]

    # ---------------- items ----------------
    for r in window:
        prior = feedback.get(r["qid"])
        mark = {"correct": "✅", "incorrect": "❌", "needs-review": "🔶"}.get(
            prior["verdict"] if prior else "", "")
        head = (f"{mark} {r['qid']} · {r['category']} · "
                + ("ABSTAINED" if r["abstained"] else f"confidence {r['confidence']:.2f}")
                + f" — {r['flag_reason']}")
        with st.expander(head, expanded=(len(window) <= 3)):
            st.markdown(f"### {r['question']}")

            left, right = st.columns([3, 2])
            with left:
                if r["abstained"]:
                    st.warning("**The system abstained** — no answer was produced.")
                else:
                    st.markdown("**Answer**")
                    st.info(r["answer"])
            with right:
                st.text(conf_bar(r["retrieval_norm"], "retrieval "))
                st.text(conf_bar(r["grounding"], "grounding "))
                st.text(conf_bar(r["confidence"], "CONFIDENCE"))
                st.caption(f"raw top cosine {r['retrieval_strength']:.4f} · "
                           f"priority {r['priority']:.2f}")
                prov = r.get("provenance") or {}
                if prov:
                    st.markdown("**Provenance** — how the context was reached")
                    for k, v in sorted(prov.items(), key=lambda kv: -kv[1]):
                        st.caption(f"`{k}` ×{v} — {PROV_HELP.get(k, '')}")
                if r.get("graph_predicates"):
                    st.caption("graph predicates walked: "
                               + ", ".join(f"`{p}`" for p in r["graph_predicates"]))
                gate = (r.get("vqa") or {}).get("gate")
                if gate:
                    st.markdown("**VQA crop gate**")
                    for gitem in gate:
                        st.caption(
                            f"{'PASS' if gitem['passed'] else 'blocked'} "
                            f"`{gitem['chunk_id']}` score={gitem['score']} "
                            f"on-supported-page={gitem['on_supported_page']}")

            st.markdown("**Cited evidence**")
            if r["citations"]:
                for cit in r["citations"]:
                    if cit["in_store"]:
                        st.markdown(f"`{cit['chunk_id']}`")
                        st.caption(cit["text"][:900])
                    else:
                        st.error(f"`{cit['chunk_id']}` — cited but not in the chunk store")
            else:
                st.caption("_no inline citations in this answer_"
                           if not r["abstained"] else "_none — the system abstained_")

            if r.get("reasoning"):
                with st.popover("Reasoning trace (chain-of-thought)"):
                    st.caption("Which chunks were weighed and what was taken from each. "
                               "Captured with `use_cot`; not part of the locked run.")
                    st.code(r["reasoning"], language=None)

            with st.popover("Full retrieved context"):
                for h in r.get("retrieved_top", []):
                    st.caption(f"`{h['chunk_id']}` [{h['region']}] via {h['source']}")

            if reveal:
                st.markdown("**Gold** (revealed)")
                st.success(f"expected: {r['expected_answer']}")
                jc = r.get("judge_correctness")
                st.caption(f"gold chunks: {', '.join(r['gold_chunk_ids']) or '—'} · "
                           f"automated judge scored this "
                           f"{'—' if jc is None else f'{jc:.2f}'}")

            st.divider()
            if prior:
                st.caption(f"last verdict **{prior['verdict']}** by {prior['reviewer']} "
                           f"at {prior['at']}" + (f" — {prior['note']}" if prior.get("note") else ""))
            note = st.text_input("Note (optional)", key=f"note_{r['qid']}",
                                 placeholder="what was wrong, or why it needs another look")
            bcols = st.columns(len(VERDICTS))
            for i, v in enumerate(VERDICTS):
                if bcols[i].button(v, key=f"{v}_{r['qid']}", use_container_width=True):
                    record(fpath, {
                        "qid": r["qid"], "verdict": v, "note": note,
                        "reviewer": os.environ.get("REVIEWER", "reviewer"),
                        "at": dt.datetime.now().isoformat(timespec="seconds"),
                        "config": r.get("source_config"), "step": r.get("source_step"),
                        "confidence": r["confidence"], "abstained": r["abstained"],
                    })
                    st.rerun()


if __name__ == "__main__":
    main()
