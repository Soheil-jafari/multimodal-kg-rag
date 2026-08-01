"""Prompt templates for grounded QA and abstention.

ABSTENTION_TEXT is the single canonical string both generation and evaluation
match against. The abstain clause is appended only when allow_abstain is set.
"""
from __future__ import annotations

ABSTENTION_TEXT = "insufficient evidence in the corpus"

ANSWER_SYSTEM = (
    "Answer the question strictly and only from the provided context passages. "
    "Do not use any outside knowledge. After each claim, cite the supporting "
    "passage inline in square brackets using its chunk_id exactly as shown, e.g. "
    "[val00000:409598:r5]. Cite only chunk_ids that appear in the context."
)

ABSTAIN_CLAUSE = (
    " The context passages are retrieved and may be noisy or split across several "
    "passages. Attempt an answer whenever the needed facts are present in the context, "
    "even across multiple passages or with imperfect wording. Reply with exactly "
    '"insufficient evidence in the corpus" (and nothing else) ONLY when the context '
    "does not address the question at all."
)

ANSWER_USER = "Question: {Q}\n\nContext passages:\n{BLOCK}\n\nAnswer:"

# --- chain-of-thought (use_cot) ---
# Appended LAST so the output-format instruction is the final thing the model reads.
# The two sections are delimited because everything downstream — citation extraction,
# the abstention match, per-sentence grounding — must run on the ANSWER text alone. If
# the reasoning were scored too, CoT would change what the metrics mean and the paired
# comparison would measure the parser rather than the prompt.
COT_CLAUSE = (
    " Think step by step before answering. First write a line containing exactly "
    "'REASONING:' followed by brief numbered steps over the context passages: which "
    "passages bear on the question, what each states, and how they combine. Then write a "
    "line containing exactly 'ANSWER:' followed by the final answer alone, with its "
    "inline citations. Put the citations only in the ANSWER section, and make that "
    "section stand on its own."
)
COT_REASONING_MARK = "REASONING:"
COT_ANSWER_MARK = "ANSWER:"

# --- VQA path (use_vqa): retrieved figure/table crops attached as images ---
# The OCR text of a table is linearised and loses its grid, so row/column values can
# be mis-paired in the text passages. The image is authoritative for anything read
# out of a cell; the text stays available for surrounding prose.
VQA_SYSTEM = (
    "Answer the question using the attached figure/table image(s) together with the "
    "context passages. The image(s) are the retrieved figure or table regions named "
    "in the context. "
    "IMPORTANT: the context passages for figures and tables come from OCR that "
    "flattens a table's rows and columns into one line, so values there may be "
    "mis-paired or run together (e.g. a 95% CI '1.11-1.29' may appear as '1.111.29'). "
    "When the question asks for a value that appears in an image, READ IT FROM THE "
    "IMAGE and trust the image over the OCR text. Locate the correct row and column "
    "before reading a number, and reproduce it exactly as printed, including "
    "parentheses, ranges and units. "
    "Cite the supporting passage inline in square brackets using its chunk_id exactly "
    "as shown, e.g. [val00000:409598:r5]. Cite only chunk_ids that appear in the "
    "context; cite the crop's own chunk_id when the value came from its image."
)

VQA_USER = (
    "Question: {Q}\n\nContext passages:\n{BLOCK}\n\n"
    "Attached image(s), in order: {IMGS}\n\nAnswer:"
)

# --- table transcription (ingest-time, once per table crop) ---
# Purpose is RETRIEVAL, not display: the output is embedded as the table's retrieval
# text, so it must name the row and column labels a question would use. OCR of the
# same crop is linearised and loses the grid, which left tables unfindable (Phase 8
# finding #1).
TRANSCRIBE_SYSTEM = (
    "You transcribe scientific tables from an image into clean text for a search "
    "index. Reproduce the table faithfully as pipe-separated rows, one row per line, "
    "starting with the header row. Preserve every column and row label verbatim, and "
    "every value exactly as printed including parentheses, ranges, units, "
    "confidence intervals and footnote markers. Keep a value in the same row as its "
    "own label — never shift values between rows or columns. If the table has a "
    "title or caption inside the image, put it on the first line. If a cell is "
    "empty, write an empty cell. If the image is unreadable or is not a table, reply "
    "with exactly: NO_TABLE. Output only the transcription, no commentary."
)

TRANSCRIBE_USER = "Transcribe this table for a search index."
