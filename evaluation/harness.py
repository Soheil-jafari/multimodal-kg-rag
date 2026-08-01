"""Evaluation harness: run one flag config over the gold set -> per-category report.

Logs, per question: predicates walked (graph), retrieval provenance, abstention,
and correctness — so failures and coverage are readable afterwards.
"""
from __future__ import annotations

import collections
import json
import os
from typing import Optional

from evaluation.answer_metrics import faithfulness, judge_correctness
from evaluation.retrieval_metrics import retrieval_scores
from platform_core.config import text_index_name
from platform_core.generation.generator import GroundedAnswerGenerator
from platform_core.generation.grounding import GroundingChecker
from platform_core.retrieval.retriever import FlagDrivenRetriever

LOW_N = 5  # categories/predicates below this are flagged as low-sample


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _restore(rec: dict) -> dict:
    """Undo JSON's key stringification on a reloaded per-question record.

    ``recall``/``precision`` are keyed by cutoff k as an int in memory; JSON has no
    integer keys, so a round-trip returns {"5": ...} and every metric lookup misses.
    """
    r = rec.get("retrieval")
    if r:
        for field in ("recall", "precision"):
            if isinstance(r.get(field), dict):
                r[field] = {int(k): v for k, v in r[field].items()}
    return rec


class Harness:
    def __init__(self, store, indices, unit_sources, bge, clip, reranker, graph, chunk_nodes,
                 gen_llm, verify_llm, judge_llm, ret_params, gen_params, reports_dir,
                 vqa_llm=None):
        self.store = store
        self.indices = indices  # {"text_layout","text_naive","image"}
        self.unit_sources = unit_sources  # {"text_layout":{...}, "text_naive":{...}}
        self.bge, self.clip, self.reranker = bge, clip, reranker
        self.graph, self.chunk_nodes = graph, chunk_nodes
        self.gen_llm, self.verify_llm, self.judge_llm = gen_llm, verify_llm, judge_llm
        self.vqa_llm = vqa_llm
        self.ret_params, self.gen_params = ret_params, gen_params
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)

    def _retriever(self, flags) -> FlagDrivenRetriever:
        # same resolver the config uses, so a flag set always reads the index it names
        key = text_index_name(flags)
        if key not in self.indices:  # transcription pass not run for this corpus
            key = "text_layout" if flags.use_layout_chunking else "text_naive"
        return FlagDrivenRetriever(flags, self.ret_params, self.store, self.indices[key], self.bge,
                                   self.indices.get("image"), self.clip, self.graph,
                                   self.chunk_nodes, self.reranker, self.unit_sources[key])

    def run_config(self, name: str, flags, gold, limit: Optional[int] = None,
                   resume: bool = False):
        """Evaluate one flag config over the gold set.

        ``resume`` re-reads a completed config's per-question log instead of re-running
        it. Every question costs a gpt-4o answer plus a gpt-4o judge, so a run that dies
        in config four must not re-buy configs one to three — and because the metrics are
        computed from the per-question records, reloading them reproduces the config's
        report exactly rather than approximating it.
        """
        path = os.path.join(self.reports_dir, f"perq_{name}.jsonl")
        want = len(gold[:limit] if limit else gold)
        if resume and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                perq = [_restore(json.loads(line)) for line in f if line.strip()]
            if len(perq) == want:
                print(f"  [resume] {name}: reusing {len(perq)} logged questions")
                return self._aggregate(perq), perq
            print(f"  [resume] {name}: log has {len(perq)}/{want} — re-running")

        retriever = self._retriever(flags)
        generator = GroundedAnswerGenerator(self.gen_llm, flags, self.gen_params, self.bge,
                                            self.store, vqa_llm=self.vqa_llm)
        checker = GroundingChecker(self.bge, self.gen_params.grounding_min_sim,
                                   self.gen_params.grounding_border,
                                   self.verify_llm if self.gen_params.grounding_llm_verify else None)
        perq = []
        for g in (gold[:limit] if limit else gold):
            results = retriever.retrieve(g.question)
            trace = retriever.last_trace
            gen = generator.generate(g.question, results)
            rec = {"qid": g.qid, "category": g.category, "source_predicates": g.source_predicates,
                   "answerable": g.is_answerable, "abstained": gen.abstained,
                   "provenance": trace.get("sources", {}), "graph_predicates": trace.get("graph_predicates", []),
                   "vqa": generator.last_vqa, "answer": gen.answer,
                   "retrieval": None, "correctness": None, "faithfulness": None, "attempted": None}
            if g.is_answerable:
                retrieved_ids = [r.chunk.chunk_id for r in results]
                rec["retrieval"] = retrieval_scores(retrieved_ids, g.supporting_chunk_ids)
                if gen.abstained:
                    rec["correctness"], rec["attempted"] = 0.0, False  # wrongly abstained
                else:
                    rec["correctness"] = judge_correctness(self.judge_llm, g.question, g.expected_answer, gen.answer)
                    rec["faithfulness"] = faithfulness(checker.check(gen.answer, results, self.store))
                    rec["attempted"] = True
                rec["decision_correct"] = not gen.abstained
            else:
                rec["decision_correct"] = gen.abstained
            perq.append(rec)

        with open(path, "w", encoding="utf-8") as f:
            for r in perq:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return self._aggregate(perq), perq

    @staticmethod
    def _agg_group(items) -> dict:
        ans = [r for r in items if r["answerable"]]
        wr = [r for r in ans if r["retrieval"]]
        return {
            "n": len(items), "n_answerable": len(ans),
            "recall@5": _mean([r["retrieval"]["recall"][5] for r in wr]),
            "prec@5": _mean([r["retrieval"]["precision"][5] for r in wr]),
            "mrr": _mean([r["retrieval"]["mrr"] for r in wr]),
            "ndcg": _mean([r["retrieval"]["ndcg"] for r in wr]),
            "correctness": _mean([r["correctness"] for r in ans]),  # abstained-answerable = 0
            "faithfulness": _mean([r["faithfulness"] for r in ans]),
            "decision_acc": _mean([1.0 if r["decision_correct"] else 0.0 for r in items]),
            "low_n": len(items) < LOW_N,
        }

    def _aggregate(self, perq) -> dict:
        cats = collections.defaultdict(list)
        for r in perq:
            cats[r["category"]].append(r)
        per_category = {c: self._agg_group(v) for c, v in cats.items()}

        preds = collections.defaultdict(list)
        for r in perq:
            if r["answerable"]:
                for p in set(r["source_predicates"]):
                    preds[p].append(r)
        per_predicate = {}
        for p, v in preds.items():
            per_predicate[p] = {
                "n": len(v),
                "correctness": _mean([r["correctness"] for r in v]),
                "recall@5": _mean([r["retrieval"]["recall"][5] for r in v if r["retrieval"]]),
                "low_n": len(v) < LOW_N, "n1": len(v) == 1,
            }

        ans = [r for r in perq if r["answerable"]]
        una = [r for r in perq if not r["answerable"]]
        return {
            "per_category": per_category,
            "overall": self._agg_group(perq),
            "per_predicate": per_predicate,
            "attempt_rate_answerable": _mean([1.0 if r["attempted"] else 0.0 for r in ans]),
            "abstain_rate_unanswerable": _mean([1.0 if r["decision_correct"] else 0.0 for r in una]),
        }
