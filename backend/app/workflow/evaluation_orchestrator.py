"""LangGraph orchestration for evaluation flows."""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from app.evaluation.rag_evaluator import RAGEvaluator
from app.services.llm_service import LLMService


class EvaluationState(TypedDict, total=False):
    question: str
    answer: str
    retrieved_contexts: List[str]
    ground_truth_answer: str
    evaluation: Dict[str, float]


class EvaluationOrchestrator:
    def __init__(self, evaluator: RAGEvaluator, llm_service: LLMService) -> None:
        self.evaluator = evaluator
        self.llm_service = llm_service
        self.graph = self._build_graph()

    async def _ensure_answer(self, state: EvaluationState) -> EvaluationState:
        if not (state.get("answer") or "").strip():
            answer, _ = await self.llm_service.generate(
                question=state["question"],
                context_chunks=state["retrieved_contexts"],
            )
            state["answer"] = answer
        return state

    async def _evaluate(self, state: EvaluationState) -> EvaluationState:
        result = await self.evaluator.evaluate(
            question=state["question"],
            answer=state["answer"],
            retrieved_contexts=state["retrieved_contexts"],
            ground_truth=state["ground_truth_answer"],
        )
        state["evaluation"] = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
            "overall_score": result.overall_score,
        }
        return state

    def _build_graph(self):
        graph = StateGraph(EvaluationState)
        graph.add_node("ensure_answer", self._ensure_answer)
        graph.add_node("evaluate", self._evaluate)
        graph.set_entry_point("ensure_answer")
        graph.add_edge("ensure_answer", "evaluate")
        graph.add_edge("evaluate", END)
        return graph.compile()

    async def run(self, payload: Dict[str, Any]) -> Dict[str, float]:
        state = await self.graph.ainvoke(payload)
        return state["evaluation"]
