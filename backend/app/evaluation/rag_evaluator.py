"""
RAG evaluation service with RAGAs-style metrics.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Evaluation result with metrics."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    overall_score: float


class RAGEvaluator:
    """Evaluator for RAG system performance."""

    def __init__(self):
        pass

    async def evaluate(
        self,
        question: str,
        answer: str,
        retrieved_contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate RAG system performance.
        
        Args:
            question: User question
            answer: Generated answer
            retrieved_contexts: List of retrieved context chunks
            ground_truth: Optional ground truth answer
            
        Returns:
            EvaluationResult with metrics
        """
        logger.info("evaluating_rag", question=question[:50])

        # Calculate individual metrics
        faithfulness = self._evaluate_faithfulness(answer, retrieved_contexts)
        answer_relevancy = self._evaluate_answer_relevancy(question, answer)
        context_precision = self._evaluate_context_precision(
            question, retrieved_contexts
        )

        # Calculate context recall if ground truth is provided
        if ground_truth:
            context_recall = self._evaluate_context_recall(
                ground_truth, retrieved_contexts
            )
        else:
            context_recall = 0.5  # Default neutral score

        # Calculate overall score (weighted average)
        overall = (
            faithfulness * 0.3
            + answer_relevancy * 0.3
            + context_precision * 0.25
            + context_recall * 0.15
        )

        result = EvaluationResult(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_recall=context_recall,
            overall_score=overall,
        )

        logger.info(
            "evaluation_completed",
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            overall=overall,
        )

        return result

    def _evaluate_faithfulness(
        self, answer: str, contexts: List[str]
    ) -> float:
        """
        Evaluate faithfulness - does the answer match the context?
        
        Uses simple heuristic: check for claims in answer that can be
        verified against context.
        """
        if not answer or not contexts:
            return 0.0

        # Combine contexts
        combined_context = " ".join(contexts).lower()

        # Simple claim extraction (sentences)
        answer_sentences = re.split(r"[.!?]+", answer.lower())
        answer_sentences = [s.strip() for s in answer_sentences if s.strip()]

        if not answer_sentences:
            return 0.5

        # Check each sentence against context
        supported = 0
        for sentence in answer_sentences:
            # Skip short sentences
            if len(sentence.split()) < 3:
                continue

            # Check if key words from sentence appear in context
            words = sentence.split()
            key_words = [w for w in words if len(w) > 4]
            if key_words:
                matches = sum(1 for w in key_words if w in combined_context)
                if matches / len(key_words) > 0.5:
                    supported += 1

        return supported / len(answer_sentences) if answer_sentences else 0.5

    def _evaluate_answer_relevancy(
        self, question: str, answer: str
    ) -> float:
        """
        Evaluate answer relevancy - does the answer address the question?
        
        Uses keyword overlap between question and answer.
        """
        if not question or not answer:
            return 0.0

        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())

        # Remove common stop words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all",
            "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "just", "what", "which", "who", "whom",
            "this", "that", "these", "those", "am", "and", "but", "if",
            "or", "because", "until", "while", "about", "against",
        }

        question_keywords = question_words - stop_words
        answer_keywords = answer_words - stop_words

        if not question_keywords:
            return 0.5

        overlap = len(question_keywords & answer_keywords)
        return min(overlap / len(question_keywords), 1.0)

    def _evaluate_context_precision(
        self, question: str, contexts: List[str]
    ) -> float:
        """
        Evaluate context precision - how relevant are the contexts?
        
        Uses keyword overlap between question and each context.
        """
        if not question or not contexts:
            return 0.0

        question_words = set(question.lower().split())
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "to", "of", "in", "for", "on", "with", "at",
            "by", "from", "as", "what", "which", "who", "how", "why",
        }

        question_keywords = question_words - stop_words

        if not question_keywords:
            return 0.5

        # Score each context
        scores = []
        for context in contexts:
            context_words = set(context.lower().split())
            context_keywords = context_words - stop_words

            if context_keywords:
                overlap = len(question_keywords & context_keywords)
                score = overlap / len(question_keywords)
                scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0

    def _evaluate_context_recall(
        self, ground_truth: str, contexts: List[str]
    ) -> float:
        """
        Evaluate context recall - does context contain ground truth?
        
        Uses keyword overlap between ground truth and contexts.
        """
        if not ground_truth or not contexts:
            return 0.0

        truth_words = set(ground_truth.lower().split())
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "to", "of", "in", "for", "on", "with", "at",
            "by", "from", "as", "and", "or", "but", "if", "then", "so",
        }

        truth_keywords = truth_words - stop_words

        if not truth_keywords:
            return 0.5

        # Check if keywords appear in any context
        combined_context = " ".join(contexts).lower()
        matched = sum(1 for w in truth_keywords if w in combined_context)

        return matched / len(truth_keywords)


# Singleton instance
rag_evaluator = RAGEvaluator()


async def get_rag_evaluator() -> RAGEvaluator:
    """Get the global RAG evaluator instance."""
    return rag_evaluator