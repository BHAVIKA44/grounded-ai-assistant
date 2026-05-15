"""LangGraph-based orchestration for question answering flow."""

from __future__ import annotations

import time
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from app.core.config import get_settings
from app.core.exceptions import GenerationError, RetrievalError
from app.core.logging import get_logger
from app.retrieval.hybrid import HybridRetrieval
from app.retrieval.reranker import Reranker
from app.schemas.document import Citation
from app.services.llm_service import LLMService

logger = get_logger(__name__)
settings = get_settings()

OUT_OF_CONTEXT_MESSAGE = "I can't answer this from the uploaded documents."


class QAState(TypedDict, total=False):
    question: str
    top_k: int
    use_reranking: bool
    retrieval_time_ms: float
    generation_time_ms: float
    answer: str
    sources: List[str]
    citations: List[Citation]
    model_used: str
    trace_steps: List[Dict[str, Any]]
    error: Dict[str, str]
    context_chunks: List[str]
    retrieved_chunks: List[Any]


class LangGraphQAOrchestrator:
    """Minimal LangGraph workflow for planner/research/synthesizer/validator."""

    def __init__(self, hybrid: HybridRetrieval, reranker: Reranker, llm: LLMService) -> None:
        self.hybrid = hybrid
        self.reranker = reranker
        self.llm = llm
        self.graph = self._build_graph()

    def _trace(self, state: QAState, node: str, status: str, details: Dict[str, Any] | None = None) -> None:
        trace = state.setdefault("trace_steps", [])
        trace.append(
            {
                "node": node,
                "status": status,
                "ts": time.time(),
                "details": details or {},
            }
        )

    def _planner(self, state: QAState) -> QAState:
        self._trace(state, "planner", "completed", {"question_len": len(state["question"])})
        return state

    async def _research(self, state: QAState) -> QAState:
        start = time.perf_counter()
        self._trace(state, "research", "started")
        query = state["question"]
        top_k = state["top_k"]
        for attempt in range(2):
            try:
                chunks = await self.hybrid.retrieve(
                    query=query,
                    top_k=top_k,
                    use_bm25=True,
                    use_vector=True,
                    use_reranking=state["use_reranking"],
                )
                if state["use_reranking"] and chunks:
                    chunks = await self.reranker.rerank(query=query, chunks=chunks, top_k=settings.rerank_top_k)
                state["retrieved_chunks"] = chunks
                state["retrieval_time_ms"] = (time.perf_counter() - start) * 1000
                self._trace(state, "research", "completed", {"chunks": len(chunks), "attempt": attempt + 1})
                return state
            except Exception as exc:
                self._trace(state, "research", "retrying", {"attempt": attempt + 1, "error": str(exc)})
                if attempt == 1:
                    raise RetrievalError(str(exc))
        return state

    async def _synthesizer(self, state: QAState) -> QAState:
        self._trace(state, "synthesizer", "started")
        chunks = state.get("retrieved_chunks") or []
        if not chunks:
            state["answer"] = OUT_OF_CONTEXT_MESSAGE
            state["sources"] = []
            state["citations"] = []
            state["generation_time_ms"] = 0.0
            state["model_used"] = "none"
            self._trace(state, "synthesizer", "completed", {"reason": "no_context"})
            return state

        top_score = chunks[0].score if chunks else 0.0
        if top_score < 0.08:
            state["answer"] = OUT_OF_CONTEXT_MESSAGE
            state["sources"] = []
            state["citations"] = []
            state["generation_time_ms"] = 0.0
            state["model_used"] = "none"
            self._trace(state, "synthesizer", "completed", {"reason": "low_confidence", "score": top_score})
            return state

        start = time.perf_counter()
        selected = chunks[: min(4, len(chunks))]
        context_chunks = [f"[Source: {c.source}]\n{c.content}" for c in selected]
        sources = list(dict.fromkeys(c.source for c in selected))
        try:
            answer, _, _ = await self.llm.generate_with_citations(
                question=state["question"],
                context_chunks=context_chunks,
                sources=sources,
            )
        except Exception as exc:
            raise GenerationError(str(exc))

        citations: List[Citation] = []
        seen = set()
        for chunk in selected:
            if chunk.source in seen:
                continue
            seen.add(chunk.source)
            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    source=chunk.source,
                    score=chunk.score,
                    page=chunk.page,
                )
            )
            if len(citations) >= 5:
                break

        state["answer"] = answer
        state["sources"] = sources
        state["citations"] = citations
        state["generation_time_ms"] = (time.perf_counter() - start) * 1000
        state["model_used"] = self.llm.model
        self._trace(state, "synthesizer", "completed", {"sources": len(sources)})
        return state

    def _validator(self, state: QAState) -> QAState:
        ok = bool(state.get("answer")) and (state.get("answer") == OUT_OF_CONTEXT_MESSAGE or bool(state.get("citations")))
        self._trace(state, "validator", "completed", {"citation_grounded": ok})
        if not ok:
            state["error"] = {"code": "validation_error", "message": "Answer is not citation-grounded"}
        return state

    def _build_graph(self):
        graph = StateGraph(QAState)
        graph.add_node("planner", self._planner)
        graph.add_node("research", self._research)
        graph.add_node("synthesizer", self._synthesizer)
        graph.add_node("validator", self._validator)
        graph.set_entry_point("planner")
        graph.add_edge("planner", "research")
        graph.add_edge("research", "synthesizer")
        graph.add_edge("synthesizer", "validator")
        graph.add_edge("validator", END)
        return graph.compile()

    async def run(self, question: str, top_k: int, use_reranking: bool) -> QAState:
        initial: QAState = {
            "question": question,
            "top_k": top_k,
            "use_reranking": use_reranking,
            "trace_steps": [],
        }
        return await self.graph.ainvoke(initial)
