"""LLM triple extraction: free (stage 1) and schema-constrained (stage 2).

* :func:`free_extract` — no schema; lets the LLM propose predicates naturally, so
  we can consolidate a closed schema from data (the two-stage method).
* :class:`LLMTripleExtractor` — constrained: the prompt gives the CLOSED predicate
  list, and :meth:`extract_report` ENFORCES it in code, dropping any triple whose
  predicate is not in the schema (case/space-normalized) or that lacks a
  subject/object/evidence. Each kept triple becomes a grounded
  :class:`~platform_core.types.GraphEdge` carrying the source ``chunk_id`` and the
  verbatim evidence sentence.

Enforcement lives in code, not in trust: we prove the closed list fires by
reporting the dropped triples, not by assuming the LLM obeyed.
"""
from __future__ import annotations

from platform_core.graph.base import TripleExtractor
from platform_core.graph.schema import PredicateSchema
from platform_core.types import Chunk, GraphEdge

_FREE_SYSTEM = (
    "You are an information-extraction system for biomedical scientific text. "
    "You extract factual relational triples and return strict JSON."
)
_FREE_USER = """From the passage, extract factual relational triples as JSON.
Return ONLY JSON of the form:
{"triples": [{"subject": "...", "predicate": "...", "object": "...", "evidence": "..."}]}
- predicate: a short lowercase verb phrase (1-3 words) naming the relation.
- evidence: the exact sentence (verbatim) from the passage that supports it.
Extract only relations clearly stated in the passage. If none, return {"triples": []}.

PASSAGE:
\"\"\"{TEXT}\"\"\""""

_CONSTRAINED_SYSTEM = (
    "You are an information-extraction system. You extract triples ONLY using a "
    "fixed, closed list of predicates. You never invent predicates. Return strict JSON."
)
_CONSTRAINED_USER = """Extract factual triples from the passage using ONLY these predicates (the CLOSED schema):
{PREDICATES}

Map relations onto the closest predicate; never invent one. Guidance:
- higher / elevated / increased level, amount, or risk of X -> increases (object = X)
- lower / reduced / decreased level, amount, or risk of X   -> decreases (object = X)
- prevalence of / observed in / found in / colonizes a population, patient group, sample, or site -> occurs_in (object MUST be that population/sample/site, never a number or percentage)
- used to treat / therapy for / treatment of a condition    -> treats
- converted / oxidized / degraded / metabolized into Y       -> transforms_to (object = Y)
- quantified / assessed / measured by a method or assay      -> measured_by

Example passage: "Serum ferritin was significantly lower in cases than in controls, and HBV infection was more prevalent among heavy drinkers."
Example output: {"triples": [
  {"subject": "cases", "predicate": "decreases", "object": "serum ferritin", "evidence": "Serum ferritin was significantly lower in cases than in controls"},
  {"subject": "HBV infection", "predicate": "occurs_in", "object": "heavy drinkers", "evidence": "HBV infection was more prevalent among heavy drinkers"}]}

Return ONLY JSON: {"triples": [{"subject": "...", "predicate": "<one of the closed predicate names>", "object": "...", "evidence": "<verbatim sentence>"}]}
Rules:
- The predicate MUST be exactly one of the closed predicate names above. If a relation fits none of them, omit it.
- Skip negated or no-effect statements ("did not affect", "no significant difference", "not associated", "no effect") — output nothing for them.
- evidence MUST be a sentence copied verbatim from the passage.
If no triples fit, return {"triples": []}.

PASSAGE:
\"\"\"{TEXT}\"\"\""""


def _normalize(predicate: str) -> str:
    return predicate.strip().lower().replace(" ", "_")


def free_extract(llm, chunk: Chunk) -> list[dict]:
    out = llm.complete_json(_FREE_SYSTEM, _FREE_USER.replace("{TEXT}", chunk.text))
    return out.get("triples", []) if isinstance(out, dict) else []


class LLMTripleExtractor(TripleExtractor):
    def __init__(self, llm, predicate_defs: list[dict]) -> None:
        self.llm = llm
        self.block = "\n".join(f"- {d['name']}: {d.get('description', '')}" for d in predicate_defs)

    def extract_report(self, chunk: Chunk, schema: PredicateSchema) -> dict:
        """Return {raw, kept, dropped, usage} — kept are grounded GraphEdges."""
        user = _CONSTRAINED_USER.replace("{PREDICATES}", self.block).replace("{TEXT}", chunk.text)
        out = self.llm.complete_json(_CONSTRAINED_SYSTEM, user)
        raw = out.get("triples", []) if isinstance(out, dict) else []
        kept: list[GraphEdge] = []
        dropped: list[dict] = []
        for t in raw:
            pred = _normalize(str(t.get("predicate", "")))
            subj = str(t.get("subject", "")).strip()
            obj = str(t.get("object", "")).strip()
            ev = str(t.get("evidence", "")).strip()
            if not schema.is_allowed(pred):
                dropped.append({**t, "_reason": f"predicate '{pred}' not in closed schema"})
            elif not (subj and obj and ev):
                dropped.append({**t, "_reason": "missing subject/object/evidence"})
            else:
                kept.append(GraphEdge(subject=subj, predicate=pred, object=obj,
                                      chunk_id=chunk.chunk_id, evidence=ev))
        return {"raw": raw, "kept": kept, "dropped": dropped, "usage": dict(self.llm.last_usage)}

    def extract(self, chunk: Chunk, schema: PredicateSchema) -> list[GraphEdge]:
        return self.extract_report(chunk, schema)["kept"]
