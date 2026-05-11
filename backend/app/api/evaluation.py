"""
Evaluation API routes.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import GenerationError
from app.core.logging import get_logger
from app.db.session import get_session_dependency
from app.evaluation.rag_evaluator import RAGEvaluator, get_rag_evaluator
from app.schemas.document import EvaluationRequest, EvaluationResponse
from app.services.llm_service import get_llm_service

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/evaluation", tags=["evaluation"])


@router.post("", response_model=EvaluationResponse)
async def evaluate_rag(
    request: EvaluationRequest,
    evaluator: RAGEvaluator = Depends(get_rag_evaluator),
) -> EvaluationResponse:
    """
    Evaluate RAG system performance on a question.
    
    Calculates:
    - Faithfulness: Does answer match context?
    - Answer Relevancy: Does answer address question?
    - Context Precision: Are contexts relevant to question?
    - Context Recall: Does context contain ground truth?
    
    Args:
        request: EvaluationRequest with question, answer, and contexts
        evaluator: RAG evaluator service
        
    Returns:
        EvaluationResponse with metrics
    """
    try:
        answer = (request.answer or "").strip()
        if not answer:
            llm_service = await get_llm_service()
            answer, _ = await llm_service.generate(
                question=request.question,
                context_chunks=request.retrieved_contexts,
            )

        result = await evaluator.evaluate(
            question=request.question,
            answer=answer,
            retrieved_contexts=request.retrieved_contexts,
            ground_truth=request.ground_truth_answer,
        )

        return EvaluationResponse(
            faithfulness=result.faithfulness,
            answer_relevancy=result.answer_relevancy,
            context_precision=result.context_precision,
            context_recall=result.context_recall,
            overall_score=result.overall_score,
        )

    except GenerationError as e:
        logger.error("evaluation_generation_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={"code": e.code, "message": str(e)},
        )
    except Exception as e:
        logger.error("evaluation_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={"code": "evaluation_failed", "message": str(e)},
        )


@router.post("/batch")
async def evaluate_batch(
    evaluations: list[EvaluationRequest],
    evaluator: RAGEvaluator = Depends(get_rag_evaluator),
):
    """
    Evaluate multiple questions at once.
    
    Returns aggregated metrics.
    """
    results = []
    try:
        for eval_request in evaluations:
            answer = (eval_request.answer or "").strip()
            if not answer:
                llm_service = await get_llm_service()
                answer, _ = await llm_service.generate(
                    question=eval_request.question,
                    context_chunks=eval_request.retrieved_contexts,
                )

            result = await evaluator.evaluate(
                question=eval_request.question,
                answer=answer,
                retrieved_contexts=eval_request.retrieved_contexts,
                ground_truth=eval_request.ground_truth_answer,
            )
            results.append(result)
    except GenerationError as e:
        logger.error("batch_evaluation_generation_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={"code": e.code, "message": str(e)},
        )
    except Exception as e:
        logger.error("batch_evaluation_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={"code": "batch_evaluation_failed", "message": str(e)},
        )

    # Calculate averages
    avg_faithfulness = sum(r.faithfulness for r in results) / len(results)
    avg_relevancy = sum(r.answer_relevancy for r in results) / len(results)
    avg_precision = sum(r.context_precision for r in results) / len(results)
    avg_recall = sum(r.context_recall for r in results) / len(results)
    avg_overall = sum(r.overall_score for r in results) / len(results)

    return {
        "count": len(results),
        "averages": {
            "faithfulness": avg_faithfulness,
            "answer_relevancy": avg_relevancy,
            "context_precision": avg_precision,
            "context_recall": avg_recall,
            "overall_score": avg_overall,
        },
    }
