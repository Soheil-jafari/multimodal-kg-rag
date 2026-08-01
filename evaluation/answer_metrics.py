"""Answer metrics: correctness (LLM judge), faithfulness (grounding), abstention.

The correctness rubric is a module constant so it is auditable and reported
verbatim. Judge model is gpt-4o.
"""
from __future__ import annotations

from collections.abc import Sequence

CORRECTNESS_SYSTEM = (
    "You are a strict factual grader for a scientific document-QA system. "
    "You compare a CANDIDATE answer to a REFERENCE answer for a QUESTION and grade "
    "ONLY factual correctness against the reference — never reward fluency, length, "
    "or plausibility.\n"
    "Rubric:\n"
    "- CORRECT (1.0): the candidate states the reference fact accurately and does not "
    "contradict it. Extra correct context or different wording is fine.\n"
    "- PARTIAL (0.5): the candidate captures part of the reference fact but is "
    "incomplete, vague, or hedged, OR is correct but buries it among unsupported claims.\n"
    "- INCORRECT (0.0): the candidate is wrong, contradicts the reference, omits the key "
    "fact, answers a different question, or abstains when an answer was expected.\n"
    'Return JSON: {"grade": "CORRECT|PARTIAL|INCORRECT", "reason": "<one sentence>"}.'
)
_GRADE = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}


def judge_correctness(llm, question: str, reference: str, candidate: str) -> float:
    out = llm.complete_json(
        CORRECTNESS_SYSTEM,
        f"QUESTION: {question}\nREFERENCE: {reference}\nCANDIDATE: {candidate}",
    )
    grade = str(out.get("grade", "INCORRECT")).strip().upper() if isinstance(out, dict) else "INCORRECT"
    return _GRADE.get(grade, 0.0)


def faithfulness(grounding_report: Sequence) -> float:
    """Fraction of answer sentences supported by their cited chunk (uncited = unsupported)."""
    if not grounding_report:
        return 0.0
    return sum(1 for g in grounding_report if g.supported) / len(grounding_report)
