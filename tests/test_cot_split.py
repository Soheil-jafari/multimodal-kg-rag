"""Chain-of-thought responses must be split before anything scores them.

The paired CoT comparison is only meaningful if turning the flag on changes how an
answer is *produced* and not how it is *parsed*. If reasoning text reached the citation
regex it would invent citations; if it reached the abstention match, a response that
merely discussed insufficient evidence would be scored as an abstention. These pin the
split so the comparison measures the prompt rather than the parser.
"""
from __future__ import annotations

from platform_core.generation.generator import split_cot


def test_splits_reasoning_from_answer():
    raw = ("REASONING:\n1. Passage r0 reports the exposure.\n"
           "2. Passage r4 gives the outcome.\n"
           "ANSWER:\nPCB153 decreased BCL-2 expression [val00000:414928:r0].")
    answer, reasoning = split_cot(raw)
    assert answer == "PCB153 decreased BCL-2 expression [val00000:414928:r0]."
    assert "1. Passage r0" in reasoning and "ANSWER" not in reasoning


def test_citations_in_reasoning_do_not_reach_the_answer():
    raw = ("REASONING:\nThe candidates are [val00000:1:r1] and [val00000:2:r2].\n"
           "ANSWER:\nIt increased [val00000:2:r2].")
    answer, _ = split_cot(raw)
    assert "val00000:1:r1" not in answer, "a citation weighed and rejected in the reasoning "\
        "must not be counted as cited"


def test_abstention_phrase_in_reasoning_does_not_abstain_the_answer():
    raw = ("REASONING:\nAt first this looks like insufficient evidence in the corpus, "
           "but passage r3 states it directly.\nANSWER:\nThe rate was 12% [x:1:r3].")
    answer, _ = split_cot(raw)
    assert "insufficient evidence" not in answer.lower()


def test_missing_marker_falls_back_to_the_whole_response():
    """A malformed response must stay scoreable rather than becoming an empty answer."""
    raw = "The compound was oxidized to 2-NA [val00000:415232:r6]."
    answer, reasoning = split_cot(raw)
    assert answer == raw and reasoning == ""


def test_last_answer_marker_wins():
    """The reasoning may quote the word; only the final section is the answer."""
    raw = ("REASONING:\nStep 1 — the ANSWER: hint appears in passage r1.\n"
           "ANSWER:\nFinal value is 4.9 [x:1:r1].")
    answer, _ = split_cot(raw)
    assert answer == "Final value is 4.9 [x:1:r1]."


def test_empty_answer_section_falls_back_rather_than_returning_nothing():
    raw = "REASONING:\nSome reasoning with no conclusion.\nANSWER:\n"
    answer, _ = split_cot(raw)
    assert answer.strip(), "an empty ANSWER section must not produce an empty answer"
