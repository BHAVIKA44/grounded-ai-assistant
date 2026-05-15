"""
Chat API routes for question answering with RAG.
"""

import time

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import get_settings
from app.core.exceptions import GenerationError, RetrievalError
from app.core.logging import get_logger
from app.db.session import get_session_dependency
from app.retrieval.hybrid import HybridRetrieval, get_hybrid_retrieval
from app.retrieval.reranker import Reranker, get_reranker
from app.schemas.document import AnswerRequest, AnswerResponse, Citation
from app.services.cache_service import cache_key_question, get_cache_service
from app.services.llm_service import get_llm_service
from app.workflow.langgraph_orchestrator import LangGraphQAOrchestrator

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/v1", tags=["chat"])
OUT_OF_CONTEXT_MESSAGE = "I can't answer this from the uploaded documents."


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(
    request: AnswerRequest,
    _session=Depends(get_session_dependency),
    hybrid_retrieval: HybridRetrieval = Depends(get_hybrid_retrieval),
    reranker: Reranker = Depends(get_reranker),
) -> AnswerResponse:
    """
    Ask a question and get an answer grounded in the document context.
    
    This endpoint:
    1. Checks cache for previous answers
    2. Performs hybrid retrieval (BM25 + vector)
    3. Applies cross-encoder reranking if enabled
    4. Generates answer using LLM with citations
    5. Caches the response
    
    Args:
        request: AnswerRequest with question and options
        session: Database session
        hybrid_retrieval: Hybrid retrieval service
        reranker: Reranking service
        
    Returns:
        AnswerResponse with answer, citations, and timing info
    """
    total_start = time.perf_counter()

    logger.info("question_received", question=request.question[:100])

    # Check cache
    cache_service = await get_cache_service()
    cache_key = cache_key_question(request.question)

    if request.use_caching:
        cached = await cache_service.get(cache_key)
        if cached:
            logger.info("cache_hit", question=request.question[:50])
            cached["cached"] = True
            return AnswerResponse(**cached)

    try:
        llm_service = await get_llm_service()
        orchestrator = LangGraphQAOrchestrator(hybrid_retrieval, reranker, llm_service)
        state = await orchestrator.run(
            question=request.question,
            top_k=request.top_k or 10,
            use_reranking=request.use_reranking,
        )
    except RetrievalError as e:
        logger.error("chat_retrieval_failed", error=str(e), question=request.question[:100])
        raise HTTPException(
            status_code=500,
            detail={"code": e.code, "message": f"Retrieval failed: {str(e)}"},
        )
    except GenerationError as e:
        logger.error("chat_generation_failed", error=str(e), question=request.question[:100])
        raise HTTPException(
            status_code=500,
            detail={"code": e.code, "message": f"Generation failed: {str(e)}"},
        )
    except Exception as e:
        logger.error("chat_generation_failed", error=str(e), question=request.question[:100])
        raise HTTPException(
            status_code=500,
            detail={"code": "generation_error", "message": f"Generation failed: {str(e)}"},
        )

    total_time = (time.perf_counter() - total_start) * 1000

    response = AnswerResponse(
        answer=state.get("answer", OUT_OF_CONTEXT_MESSAGE),
        question=request.question,
        citations=state.get("citations", []),
        sources=state.get("sources", []),
        retrieval_time_ms=state.get("retrieval_time_ms", 0.0),
        generation_time_ms=state.get("generation_time_ms", 0.0),
        total_time_ms=total_time,
        model_used=state.get("model_used", "none"),
        cached=False,
    )

    # Cache the response
    if request.use_caching:
        await cache_service.set(
            cache_key,
            response.model_dump(),
            ttl=3600,  # 1 hour
        )

    logger.info(
        "answer_generated",
        question=request.question[:50],
        total_time_ms=total_time,
        cached=False,
    )

    return response


@router.get("/retrieval")
async def test_retrieval(
    query: str,
    top_k: int = 10,
    _session=Depends(get_session_dependency),
    hybrid_retrieval: HybridRetrieval = Depends(get_hybrid_retrieval),
):
    """
    Test retrieval without generating an answer.
    
    Useful for debugging retrieval quality.
    """
    start_time = time.perf_counter()

    retrieved_chunks = await hybrid_retrieval.retrieve(
        query=query,
        top_k=top_k,
        use_bm25=True,
        use_vector=True,
    )

    retrieval_time = (time.perf_counter() - start_time) * 1000

    return {
        "query": query,
        "results": [
            {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                "source": chunk.source,
                "score": chunk.score,
                "page": chunk.page,
                "method": chunk.retrieval_method,
            }
            for chunk in retrieved_chunks
        ],
        "retrieval_time_ms": retrieval_time,
    }
