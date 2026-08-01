"""Build the gold set by SELECTING facts from our own KG.

    python -m scripts.build_gold --config configs/scale500.yaml --target 130

Construction discipline (unchanged from Phase 6A, and the reason the set is
trustworthy): every question is SELECTED from a KG edge, so the answer and the
supporting chunk_ids are already known from the edge. The LLM only PHRASES the
selected fact into natural language — it never supplies the answer or the evidence.
Coverage-driven across categories and the 8 predicates, plus a text-derived bias
guard (a passage read directly, no graph) and crafted unanswerable items.

Quotas scale with `--target`; the category mix is held at the Phase-6A proportions
so a 55-question result and a ~130-question result are comparable.

AUTOMATED FIRST PASS ONLY. Everything here is a filter that can be applied
mechanically; the output still requires human spot-checking before it is used to
measure anything. Filters are counted and reported by reason rather than silently
applied, so the drop rate is itself a measurement. See
domain_packs/biomed/gold/CLEANUP.md for the defect classes these encode.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from platform_core.config import AppConfig
from platform_core.graph.schema import load_predicates
from platform_core.llm.openai_client import OpenAIClient
from platform_core.stores.chunk_store import SQLiteChunkStore
from platform_core.types import RegionType

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
DEFAULT_CONFIG = R + "/configs/scale500.yaml"
PRED = R + "/domain_packs/biomed/predicates.yaml"
IN_PRICE, OUT_PRICE = 0.15 / 1e6, 0.60 / 1e6  # gpt-4o-mini

#: Phase-6A category proportions (of the final 55), held fixed so the scaled set is
#: comparable. single_fact is expressed per-predicate for coverage, not as a lump.
MIX = {"single_fact": 0.42, "text_derived": 0.18, "multi_hop": 0.13,
       "unanswerable": 0.11, "figure": 0.09, "figure_value": 0.07}

#: Crafted to be clearly outside a toxicology/epidemiology page corpus. Each is
#: VERIFIED against the actual corpus below — at 500 pages an assumed-absent topic
#: may well be present, and one answerable item here would corrupt the abstention
#: measurement that the unanswerable category exists to make.
UNANSWERABLE_POOL = [
    "What is the recommended daily dose of ibuprofen for adults?",
    "How does the mRNA COVID-19 vaccine trigger an immune response?",
    "Which gene mutation causes cystic fibrosis?",
    "What is the elimination half-life of caffeine in humans?",
    "Who discovered penicillin and in what year?",
    "What is the normal resting heart rate range for healthy adults?",
    "What is the mechanism of action of metformin in type 2 diabetes?",
    "How many chambers does the human heart have?",
    "What is the standard surgical procedure for an appendectomy?",
    "Which vitamin deficiency causes scurvy?",
    "What is the incubation period of the measles virus?",
    "How does dialysis replace kidney function?",
    "What is the typical shelf life of a stored blood transfusion unit?",
    "Which neurotransmitter is primarily depleted in Parkinson's disease?",
    "What is the recommended storage temperature for insulin vials?",
    "How is the Apgar score calculated for a newborn?",
    "What is the usual duration of antibiotic therapy for strep throat?",
    "Which blood type is the universal donor for red cell transfusion?",
    "What is the mode of inheritance of haemophilia A?",
    "How does a pulse oximeter measure oxygen saturation?",
    "What is the maximum safe dose of paracetamol in 24 hours?",
    "Which cranial nerve controls the muscles of facial expression?",
    "What is the target INR range for a patient on warfarin?",
    "How long does a standard course of rabies post-exposure prophylaxis last?",
]
#: The pool is deliberately LONGER than any quota it feeds. The absence check below
#: rejects candidates the corpus turns out to discuss, and with a pool sized exactly to
#: quota every rejection would permanently under-fill the abstention category — the one
#: category whose whole purpose is measuring that the system declines to answer.

STOP = set("""a an the of in on for to and or with by from at as is are was were be been being that this
these those it its their his her they he she we you i what which who whom whose when where why how did
does do done has have had not no than then there here about into over under between among during after
before above below can could may might must shall should will would per each any all both more most
much many such same other another new old high low higher lower increase increased decrease decreased
study studies group groups patient patients level levels value values reported observed result results
mean median total number percentage percent rate ratio risk associated compared using used use""".split())

_WORD = re.compile(r"[a-z0-9]+")

#: Longest-first, so "ization" is tried before "ion".
_SUFFIXES = ("ationally", "ational", "ization", "isation", "ability", "ibility",
             "fulness", "ations", "ities", "ement", "ation", "ility", "ously",
             "ness", "ment", "ions", "ance", "ence", "ical", "ised", "ized",
             "ing", "ion", "ity", "ies", "ous", "ive", "ed", "es", "ly", "al", "s")

#: A question's wording is a paraphrase of its source, not a quotation of it, so the
#: match has to survive derivation as well as inflection: a caption saying "viable
#: eggs" does answer a question about "viability". Measured on the 55-question dev set,
#: a plain prefix rule scored those legitimate paraphrases as mismatches.
ROOT_LEN = 4

#: Two bands, not one threshold. Above DROP is the q061 class — the question names rows
#: or columns the source region simply does not contain, and no route can answer it.
#: Between FLAG and DROP is genuinely ambiguous, so the item is KEPT and marked for the
#: human spot-check instead of being deleted. Dropping that band silently biased the set
#: toward questions that lexically echo their own caption, which would have inflated the
#: crop-backed retrieval numbers this gold set exists to measure. Values are calibrated
#: against the dev set in tests/test_gold_filters.py, not guessed.
MISMATCH_DROP = 0.65
MISMATCH_FLAG = 0.40


def stem(w: str) -> str:
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[:len(w) - len(suf)]
    return w


def content_tokens(s: str) -> set:
    """Distinguishing words of a question — what a source must actually contain."""
    return {w for w in _WORD.findall(s.lower()) if len(w) >= 4 and w not in STOP}


def covered(tok: str, haystack: str) -> bool:
    return stem(tok)[:ROOT_LEN] in haystack


def mismatch(question: str, haystack: str) -> tuple:
    """(ratio of the question's distinguishing words absent from the source, those words).

    Ratio 0.0 means every distinguishing word is present in the source region.
    """
    toks = content_tokens(question)
    if not toks:
        return 0.0, []
    hay = haystack.lower()
    missing = sorted(t for t in toks if not covered(t, hay))
    return len(missing) / len(toks), missing


def region_haystack(get_chunk, chunk_ids) -> str:
    """Everything the source region legitimately says — its OCR, any transcription, and
    its caption. Deliberately NOT the rest of the page (see the mismatch check)."""
    parts: list = []
    for cid in chunk_ids:
        c = get_chunk(cid)
        if c is None:
            continue
        parts += [c.text or "", c.vqa_text or ""]
        if c.caption_id:
            cap = get_chunk(c.caption_id)
            if cap is not None:
                parts.append(cap.text or "")
    return " ".join(parts).lower()


def looks_garbled(text: str) -> bool:
    dense = text.strip().replace(" ", "").replace("\n", "")
    if not dense:
        return True
    return (sum(ch.isalpha() for ch in dense) < 3
            or (sum(ch.isalnum() for ch in dense) / len(dense)) < 0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--target", type=int, default=130, help="approx total questions")
    ap.add_argument("--no-verify-unanswerable", action="store_true",
                    help="skip the BGE check that unanswerable items really are absent")
    args = ap.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    print(cfg.describe())
    store = SQLiteChunkStore(cfg.paths.chunk_db)
    edges_path = cfg.paths.graph_store + ".edges.jsonl"
    edges = [json.loads(l) for l in open(edges_path, encoding="utf-8")]
    predicates = [p["name"] for p in load_predicates(PRED)]
    llm = OpenAIClient(model="gpt-4o-mini")

    quota = {c: max(1, round(args.target * f)) for c, f in MIX.items()}
    per_pred = max(1, round(quota["single_fact"] / len(predicates)))
    print(f"\ntarget ~{args.target}; quotas {quota}  (single_fact = {per_pred}/predicate)")

    drops: collections.Counter = collections.Counter()
    _cache: dict = {}

    def chunk_of(cid: str):
        if cid not in _cache:
            try:
                _cache[cid] = store.get(cid)
            except KeyError:
                _cache[cid] = None
        return _cache[cid]

    def is_value(o: str) -> bool:
        return any(ch.isdigit() for ch in o)

    def clean_edge(e: dict) -> bool:
        s, o, ev = e["subject"].strip(), e["object"].strip(), e["evidence"].strip()
        if not (2 <= len(s) <= 60 and 1 <= len(o) <= 70 and len(ev) >= 20):
            drops["edge_malformed"] += 1
            return False
        if e["predicate"] == "transforms_to" and "identif" in ev.lower():
            drops["transforms_to_diagnostic_mismap"] += 1  # "identified as" is not a transformation
            return False
        if looks_garbled(o):
            drops["ocr_garbled_answer"] += 1
            return False
        if chunk_of(e["chunk_id"]) is None:
            drops["dangling_chunk_id"] += 1
            return False
        return True

    flagged: list[dict] = []

    def accept(question: str, answer: str, chunk_ids: list, category: str):
        """Filters that apply to a PHRASED question. Returns extra record fields, or None
        to reject. An empty dict means accepted with nothing to flag."""
        q = question.strip()
        if len(q) < 15 or not q.endswith("?"):
            drops["malformed_question"] += 1
            return None
        # a question containing its own answer verbatim measures nothing. Word-boundary
        # matched so a short answer ("CO", "ELISA") is caught without a 5-letter answer
        # being flagged for appearing inside an unrelated longer word.
        a = str(answer).strip().lower()
        if a and len(a) <= 60 and re.search(rf"(?<!\w){re.escape(a)}(?!\w)", q.lower()):
            drops["answer_leaked_into_question"] += 1
            return None
        # q061 class: the question's distinguishing qualifiers must actually exist in the
        # source region. q061 asked for the "highest cumulative PAH exposure category" of
        # a table whose rows are exclusion thresholds — unanswerable by any route, and no
        # reviewer caught it. Checked against the region's OWN text (plus its caption),
        # deliberately NOT the page's prose: the whole point is that the page discusses
        # the concepts while the table lacks the rows, so widening the haystack to the
        # page would defeat the check.
        if category in ("figure_value", "figure"):
            ratio, missing = mismatch(q, region_haystack(chunk_of, chunk_ids))
            if ratio >= MISMATCH_DROP:
                drops["question_table_mismatch"] += 1
                return None
            if ratio >= MISMATCH_FLAG:
                flagged.append({"question": q, "category": category,
                                "miss_ratio": round(ratio, 2), "missing": missing})
                return {"needs_mismatch_review": True,
                        "mismatch_miss_ratio": round(ratio, 2),
                        "mismatch_missing_terms": missing}
        return {}

    # ONE pass: clean_edge() increments `drops`, so filtering the edge list per category
    # counted every rejection once per category and inflated the reported drop rates ~3x.
    clean = [e for e in edges if clean_edge(e)]
    print(f"edges: {len(edges)} total -> {len(clean)} usable after edge-level filters")

    records: list[dict] = []

    def add(question, category, expected_answer, chunk_ids, preds, extra=None):
        marks = accept(question, expected_answer, chunk_ids, category)
        if marks is None:
            return False
        rec = {"qid": f"q{len(records) + 1:03d}", "question": question.strip(),
               "category": category, "expected_answer": str(expected_answer).strip(),
               "supporting_chunk_ids": chunk_ids, "source_predicates": preds}
        rec.update(marks)
        if extra:
            rec.update(extra)
        records.append(rec)
        return True

    def n_of(cat: str) -> int:
        return sum(1 for r in records if r["category"] == cat)

    def phrase(system: str, user: str, key: str = "question"):
        for _ in range(2):
            out = llm.complete_json(system, user)
            if isinstance(out, dict) and out.get(key):
                return out
        drops["phrasing_failed"] += 1
        return None

    # ---- 1. single_fact — per-predicate coverage, non-value objects, diverse subjects
    Q_SINGLE = ('You turn a known fact into a natural exam question. '
                'Output only the question, never the answer.')
    by_pred: dict = collections.defaultdict(list)
    for e in clean:
        if not is_value(e["object"]):
            by_pred[e["predicate"]].append(e)
    for p in predicates:
        seen_subj: set = set()
        picked = 0
        for e in by_pred.get(p, []):
            if picked >= per_pred:
                break
            if e["subject"] in seen_subj:
                continue
            pr = e["predicate"].replace("_", " ")
            out = phrase(Q_SINGLE, f'Fact: {e["subject"]} — {pr} — {e["object"]}\n'
                         f'Evidence: "{e["evidence"]}"\nWrite ONE clear natural question whose correct '
                         f'answer is exactly "{e["object"]}". Return JSON {{"question": "..."}}.')
            if out and add(out["question"], "single_fact", e["object"],
                           [e["chunk_id"]], [e["predicate"]]):
                seen_subj.add(e["subject"]); picked += 1

    # ---- 2. multi_hop — two edges sharing a node, from DIFFERENT chunks
    Q_MULTI = ('You turn two known facts into ONE natural question that requires BOTH. '
               'Output only the question, never the answer.')
    node2edges: dict = collections.defaultdict(list)
    for e in clean:
        node2edges[e["subject"]].append(e)
        node2edges[e["object"]].append(e)
    used_pairs: set = set()
    for node, es in sorted(node2edges.items(), key=lambda kv: -len(kv[1])):
        if n_of("multi_hop") >= quota["multi_hop"]:
            break
        diff_chunk, seen_chunks = [], set()
        for e in es:
            if e["chunk_id"] not in seen_chunks:
                diff_chunk.append(e); seen_chunks.add(e["chunk_id"])
        if len(diff_chunk) < 2 or len(node) < 3:
            continue
        e1, e2 = diff_chunk[0], diff_chunk[1]
        # degenerate hop: the two "facts" restate one relation, so nothing is combined.
        # Phase 6A dropped 3 of these by hand (near-duplicate co-located blood-lead edges).
        if (e1["subject"], e1["object"]) == (e2["subject"], e2["object"]) or (
                e1["subject"] == e2["subject"] and e1["predicate"] == e2["predicate"]):
            drops["degenerate_multi_hop"] += 1
            continue
        key = tuple(sorted((e1["chunk_id"], e2["chunk_id"])))
        if key in used_pairs:
            continue
        used_pairs.add(key)
        p1, p2 = e1["predicate"].replace("_", " "), e2["predicate"].replace("_", " ")
        out = phrase(Q_MULTI, f'Fact 1: {e1["subject"]} — {p1} — {e1["object"]}\n'
                     f'Fact 2: {e2["subject"]} — {p2} — {e2["object"]}\n'
                     f'Both involve "{node}". Write ONE natural question answerable only by combining '
                     f'BOTH facts. Return JSON {{"question": "..."}}.')
        if out:
            ans = f'{e1["subject"]} {p1} {e1["object"]}; {e2["subject"]} {p2} {e2["object"]}'
            add(out["question"], "multi_hop", ans, [e1["chunk_id"], e2["chunk_id"]],
                [e1["predicate"], e2["predicate"]])

    # ---- 3. figure — figure crop + its caption
    Q_FIG = ('You turn a figure caption into a natural question the figure answers. '
             'Output only the question.')
    figs = [c for c in store.iter_chunks([RegionType.FIGURE]) if c.caption_id]
    figs.sort(key=lambda c: c.chunk_id)
    stride = max(1, len(figs) // max(1, quota["figure"] * 2))
    for c in figs[::stride]:
        if n_of("figure") >= quota["figure"]:
            break
        cap_chunk = chunk_of(c.caption_id)
        if cap_chunk is None:
            continue
        cap = " ".join(cap_chunk.text.split())
        if len(cap) < 25 or looks_garbled(cap):
            drops["caption_unusable"] += 1
            continue
        out = phrase(Q_FIG, f'Figure caption: "{cap}"\nWrite ONE natural question the figure '
                     f'answers. Return JSON {{"question": "..."}}.')
        if out:
            add(out["question"], "figure", cap[:220], [c.chunk_id, c.caption_id], [])

    # ---- 4. figure_value — a numeric value in a table (the VQA-path category)
    # NOTE: the expected_answer here comes from a KG edge over the table's OCR, which
    # phase 9 showed is structurally unreliable (right digits, wrong grid) — and the
    # transcription is worse for digits. So every item is emitted with
    # needs_value_verification and MUST be checked against the crop by a human before
    # use. The builder proposes; it does not certify.
    Q_VAL = 'You turn a reported measurement into a natural question. Output only the question.'
    val_edges = [e for e in clean if is_value(e["object"])
                 and chunk_of(e["chunk_id"]) is not None
                 and chunk_of(e["chunk_id"]).region_type.value == "table"]
    seen_val: set = set()
    for e in val_edges:
        if n_of("figure_value") >= quota["figure_value"]:
            break
        if e["chunk_id"] in seen_val:
            continue
        seen_val.add(e["chunk_id"])
        c = chunk_of(e["chunk_id"])
        pr = e["predicate"].replace("_", " ")
        out = phrase(Q_VAL, f'A table reports: {e["subject"]} — {pr} — {e["object"]}\n'
                     f'Evidence: "{e["evidence"]}"\nWrite ONE natural question whose answer is the '
                     f'value "{e["object"]}" read from the table. Return JSON {{"question": "..."}}.')
        if out:
            add(out["question"], "figure_value", e["object"], [e["chunk_id"]], [e["predicate"]],
                extra={"needs_value_verification": True, "crop_path": c.image_path})

    # ---- 5. text_derived — bias guard: Q+A read from a passage, graph not consulted
    Q_TEXT = ('Read the passage. Produce ONE question the passage clearly answers and its concise '
              'answer, both grounded ONLY in the passage.')
    text_chunks = [c for c in store.iter_chunks([RegionType.TEXT])
                   if len(c.text.strip()) >= 320 and not looks_garbled(c.text)]
    text_chunks.sort(key=lambda c: c.chunk_id)
    stride = max(1, len(text_chunks) // max(1, quota["text_derived"] * 2))
    for c in text_chunks[::stride]:
        if n_of("text_derived") >= quota["text_derived"]:
            break
        out = phrase(Q_TEXT, f'Passage: "{" ".join(c.text.split())}"\n'
                     f'Return JSON {{"question": "...", "answer": "..."}}.', key="answer")
        if out and out.get("question"):
            add(out["question"], "text_derived", out.get("answer", ""), [c.chunk_id], [])

    # ---- 6. unanswerable — crafted, then VERIFIED absent from this corpus
    picked_unans = UNANSWERABLE_POOL[:quota["unanswerable"]]
    unans_scores: dict = {}
    if not args.no_verify_unanswerable:
        try:
            from platform_core.llm.embeddings import SentenceTransformerEmbedder
            from platform_core.stores.vector_store import FaissVectorIndex

            bge = SentenceTransformerEmbedder(cfg.models.text_embedding_model,
                                              cfg.models.device, cfg.models.embed_batch_size)
            ix = FaissVectorIndex.load(cfg.paths.index_dir + "/text_layout")
            kept = []
            for q in UNANSWERABLE_POOL:
                if len(kept) >= quota["unanswerable"]:
                    break
                top = ix.search(bge.embed_query(q), 1)
                score = top[0][1] if top else 0.0
                unans_scores[q] = round(float(score), 4)
                # BGE floors around 0.45 even for unrelated text (phase 5), so this
                # threshold flags "the corpus may actually discuss this", not "similar".
                if score >= 0.62:
                    drops["unanswerable_may_be_answerable"] += 1
                    continue
                kept.append(q)
            picked_unans = kept
            del bge, ix
            if len(picked_unans) < quota["unanswerable"]:
                print(f"  ! unanswerable under-filled: {len(picked_unans)}/"
                      f"{quota['unanswerable']} — the pool ran out of items this corpus "
                      f"does not discuss; add more candidates to UNANSWERABLE_POOL")
        except Exception as exc:
            # A half-finished verification is worse than none: `unans_scores` would hold
            # scores for some items and not others, and the report would print a verdict
            # it did not actually reach. Discard the partial state and fall back to the
            # unverified slice, loudly.
            unans_scores = {}
            picked_unans = UNANSWERABLE_POOL[:quota["unanswerable"]]
            print(f"  ! unanswerable verification FAILED ({type(exc).__name__}: {exc}) — "
                  f"falling back to UNVERIFIED candidates; these may be answerable and "
                  f"must be checked by hand before the abstention numbers are trusted")
    for q in picked_unans:
        add(q, "unanswerable", "insufficient evidence in the corpus", [], [],
            extra={"absence_check_top_cosine": unans_scores.get(q)})

    # ---- write
    out_path = cfg.paths.gold_set
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- report
    print(f"\nwrote {len(records)} gold questions -> {out_path}")
    print("\ncategory distribution (quota in brackets):")
    counts = collections.Counter(r["category"] for r in records)
    for cat in MIX:
        print(f"  {cat:14s} {counts.get(cat, 0):3d}  [{quota[cat]}]")
    print("\nper-predicate coverage (records citing each predicate):")
    pred_count: collections.Counter = collections.Counter()
    for r in records:
        for p in set(r["source_predicates"]):
            pred_count[p] += 1
    for p in predicates:
        low = "  ⚠ low-n (<5)" if pred_count.get(p, 0) < 5 else ""
        print(f"  {p:14s} {pred_count.get(p, 0):3d}{low}")
    print("\nAUTOMATED FILTER DROPS (by reason):")
    if drops:
        for reason, k in drops.most_common():
            print(f"  {reason:36s} {k}")
    else:
        print("  (none)")
    if unans_scores:
        print("\nunanswerable absence check (top cosine vs corpus; >=0.62 rejected):")
        for q, s in sorted(unans_scores.items(), key=lambda kv: -kv[1])[:8]:
            mark = "REJECTED" if s >= 0.62 else "ok"
            print(f"  {s:.4f} {mark:9s} {q[:66]}")

    bad = [r for r in records for cid in r["supporting_chunk_ids"] if chunk_of(cid) is None]
    print(f"\nvalidation: dangling supporting_chunk_ids = {len(bad)}")
    u = llm.total_usage
    print(f"phrasing cost (gpt-4o-mini): "
          f"${u['prompt_tokens']*IN_PRICE + u['completion_tokens']*OUT_PRICE:.4f}  {u}")
    print("\nNEXT: automated pass only — spot-check before evaluating "
          "(python -m scripts.review_gold --config ...)")
    store.close()


if __name__ == "__main__":
    main()
