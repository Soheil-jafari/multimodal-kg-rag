"""Load and validate the gold set."""
from __future__ import annotations

import json

from evaluation.types import GoldQuestion, QuestionCategory


def load_gold(path: str) -> list[GoldQuestion]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(GoldQuestion(
                qid=d["qid"], question=d["question"], category=d["category"],
                expected_answer=d.get("expected_answer", ""),
                supporting_chunk_ids=d.get("supporting_chunk_ids", []),
                source_predicates=d.get("source_predicates", []),
            ))
    return out


def validate_gold(gold: list[GoldQuestion], store) -> list[str]:
    """Return a list of problems (dangling chunk_ids, empty answers, bad categories)."""
    cats = {c.value for c in QuestionCategory}
    known = set(store.page_ids()) if False else None  # chunk existence checked per-id below
    problems = []
    for g in gold:
        if g.category not in cats:
            problems.append(f"{g.qid}: unknown category {g.category}")
        if g.is_answerable and not g.expected_answer.strip():
            problems.append(f"{g.qid}: empty expected_answer")
        if g.is_answerable and not g.supporting_chunk_ids:
            problems.append(f"{g.qid}: no supporting_chunk_ids")
        for cid in g.supporting_chunk_ids:
            try:
                store.get(cid)
            except KeyError:
                problems.append(f"{g.qid}: dangling chunk_id {cid}")
    return problems
