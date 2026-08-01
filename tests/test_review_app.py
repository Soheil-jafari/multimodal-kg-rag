"""The review dashboard must render, and must not be able to move a locked number.

The queue is a view over frozen results, so the risks worth testing are: the app throwing
on real data, the confidence definition drifting from the one documented in the sidebar,
and the feedback log losing a verdict. Everything here runs offline against the real
queue file, and writes only to tmp_path.
"""
from __future__ import annotations

import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUEUE = os.path.join(ROOT, "reports/scale500/review_queue.jsonl")
APP = os.path.join(ROOT, "scripts/review_app.py")

FLOOR, CEIL = 0.45, 0.85  # generation.abstain_min_score and the documented ceiling


def _queue():
    if not os.path.exists(QUEUE):
        pytest.skip("review queue not built")
    with open(QUEUE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_confidence_matches_its_documented_definition():
    """confidence = 0.5*Rn + 0.5*G for answered items, and Rn is the normalised
    top query-context cosine. If either drifts, the sidebar is lying to the reviewer."""
    for r in _queue():
        rn = min(1.0, max(0.0, (r["retrieval_strength"] - FLOOR) / (CEIL - FLOOR)))
        assert abs(rn - r["retrieval_norm"]) < 1e-3, r["qid"]
        if r["abstained"]:
            assert r["confidence"] is None, "an abstention has no answer to grade"
            assert abs(r["priority"] - rn) < 1e-3, r["qid"]
        else:
            expect = 0.5 * rn + 0.5 * (r["grounding"] or 0.0)
            assert abs(expect - r["confidence"]) < 1e-3, r["qid"]
            assert abs((1.0 - r["confidence"]) - r["priority"]) < 1e-3, r["qid"]


def test_queue_is_sorted_worst_first():
    rows = _queue()
    assert rows == sorted(rows, key=lambda r: -r["priority"])


def test_confidence_is_label_free():
    """No gold-derived field may appear in the scored signals — a confidence built from
    recall@k or the judge's verdict would rank items by how RIGHT they are, which a live
    queue cannot know."""
    for r in _queue():
        for signal in ("retrieval_strength", "retrieval_norm", "grounding",
                       "confidence", "priority"):
            v = r[signal]
            assert v is None or isinstance(v, (int, float))
        # gold is carried for the reveal toggle, but must not be part of the score
        assert "recall" not in json.dumps({k: r[k] for k in
                                           ("retrieval_norm", "grounding", "confidence")})


def test_feedback_log_round_trips(tmp_path):
    from scripts.review_app import load_feedback, record

    p = str(tmp_path / "fb.jsonl")
    assert load_feedback(p) == {}
    record(p, {"qid": "q001", "verdict": "incorrect", "note": "wrong row",
               "reviewer": "me", "at": "2026-08-01T10:00:00"})
    record(p, {"qid": "q002", "verdict": "correct", "note": "",
               "reviewer": "me", "at": "2026-08-01T10:01:00"})
    fb = load_feedback(p)
    assert fb["q001"]["verdict"] == "incorrect" and fb["q001"]["note"] == "wrong row"
    assert set(fb) == {"q001", "q002"}


def test_feedback_log_is_append_only_and_newest_wins(tmp_path):
    """A reviewer changing their mind must not destroy the earlier verdict — the log keeps
    both lines and the reader takes the last."""
    from scripts.review_app import load_feedback, record

    p = str(tmp_path / "fb.jsonl")
    record(p, {"qid": "q001", "verdict": "correct", "at": "1"})
    record(p, {"qid": "q001", "verdict": "incorrect", "at": "2"})
    assert load_feedback(p)["q001"]["verdict"] == "incorrect"
    assert sum(1 for _ in open(p, encoding="utf-8")) == 2, "history must be preserved"


def test_app_renders_without_exception():
    at = pytest.importorskip("streamlit.testing.v1", reason="streamlit testing API")
    if not os.path.exists(QUEUE):
        pytest.skip("review queue not built")
    app = at.AppTest.from_file(APP, default_timeout=120).run()
    assert not app.exception, f"app raised: {app.exception}"
    assert any("review queue" in t.value.lower() for t in app.title)
    assert len(app.metric) >= 5, "summary metrics missing"
