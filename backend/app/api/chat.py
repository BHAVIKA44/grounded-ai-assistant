"""
Chat API routes for question answering with RAG.
"""

import time
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.db_connection import get_session_dependency
from app.retrieval.hybrid import HybridRetrieval, get_hybrid_retrieval
from app.retrieval.reranker import Reranker, get_reranker
from app.schemas.document import AnswerRequest, AnswerResponse, Citation
from app.services.cache_service import cache_key_question, get_cache_service
from app.services.llm_service import LLMProvider, get_llm_service

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(
    request: AnswerRequest,
    session: AsyncSession = Depends(get_session_dependency),
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
            return AnswerResponse(**cached, cached=True)

    # Retrieval phase
    retrieval_start = time.perf_counter()

    try:
        # Perform hybrid retrieval
        retrieved_chunks = await hybrid_retrieval.retrieve(
            query=request.question,
            top_k=request.top_k,
            use_bm25=True,
            use_vector=True,
            use_reranking=request.use_reranking,
        )

        # Apply reranking if enabled
        if request.use_reranking and retrieved_chunks:
            retrieved_chunks = await reranker.rerank(
                query=request.question,
                chunks=retrieved_chunks,
                top_k=settings.rerank_top_k,
            )

        retrieval_time = (time.perf_counter() - retrieval_start) * 1000

        logger.info(
            "retrieval_completed",
            chunks_retrieved=len(retrieved_chunks),
            retrieval_time_ms=retrieval_time,
        )

    except Exception as e:
        logger.error("retrieval_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    # Check if we have any context
    if not retrieved_chunks:
        logger.warning("no_context_found", question=request.question)
        return AnswerResponse(
            answer="I don't have enough context to answer this question. Please upload relevant documents first.",
            question=request.question,
            citations=[],
            sources=[],
            retrieval_time_ms=retrieval_time,
            generation_time_ms=0,
            total_time_ms=(time.perf_counter() - total_start) * 1000,
            model_used="none",
            cached=False,
        )

    # Generation phase
    generation_start = time.perf_counter()

    try:
        # Prepare context
        context_chunks = [chunk.content for chunk in retrieved_chunks]
        sources = list(set(chunk.source for chunk in retrieved_chunks))

        # Generate answer
        llm_service = await get_llm_service()
        answer, cited_sources, gen_time = await llm_service.generate_with_citations(
            question=request.question,
            context_chunks=context_chunks,
            sources=sources,
        )

        generation_time = (time.perf_counter() - generation_start) * 1000

    except Exception as e:
        logger.error("generation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    # Build citations
    citations = [
        Citation(
            chunk_id=chunk.chunk_id,
            content=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
            source=chunk.source,
            score=chunk.score,
            page=chunk.page,
        )
        for chunk in retrieved_chunks[:5]  # Top 5 citations
    ]

    # Calculate total time
    total_time = (time.perf_counter() - total_start) * 1000

    # Build response
    response = AnswerResponse(
        answer=answer,
        question=request.question,
        citations=citations,
        sources=sources,
        retrieval_time_ms=retrieval_time,
        generation_time_ms=generation_time,
        total_time_ms=total_time,
        model_used=llm_service.model,
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
    session: AsyncSession = Depends(get_session_dependency),
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