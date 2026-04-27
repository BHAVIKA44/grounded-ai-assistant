"""
Cross-encoder reranking service for improved retrieval results.
"""

from typing import List, Optional

import numpy as np
from sentence_transformers import CrossEncoder

from app.core.config import get_settings
from app.core.logging import get_logger
from app.retrieval.hybrid import RetrievedChunk

logger = get_logger(__name__)
settings = get_settings()

# Global model instance
_model = None


class Reranker:
    """Cross-encoder reranker for improving retrieval results."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.rerank_model
        self._model = None

    @property
    def model(self) -> CrossEncoder:
        """Lazy load the cross-encoder model."""
        global _model
        if _model is None:
            logger.info("loading_reranker_model", model=self.model_name)
            _model = CrossEncoder(self.model_name)
            logger.info("reranker_model_loaded", model=self.model_name)
        return _model

    async def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """
        Rerank chunks based on relevance to query.
        
        Args:
            query: Query string
            chunks: List of retrieved chunks
            top_k: Number of top results to return (default: all)
            
        Returns:
            Reranked list of RetrievedChunk objects
        """
        if not chunks:
            return []

        top_k = top_k or len(chunks)

        # Prepare document-query pairs
        pairs = [(query, chunk.content) for chunk in chunks]

        # Get relevance scores
        scores = self.model.predict(pairs)

        # Add scores to chunks and sort
        for i, chunk in enumerate(chunks):
            chunk.score = float(scores[i])

        # Sort by score descending
        reranked = sorted(chunks, key=lambda x: x.score, reverse=True)

        logger.info(
            "reranking_completed",
            original_count=len(chunks),
            returned_count=min(top_k, len(chunks)),
        )

        return reranked[:top_k]

    async def score(
        self,
        query: str,
        chunks: List[RetrievedChunk],
    ) -> List[float]:
        """
        Score chunks for relevance without reranking.
        
        Args:
            query: Query string
            chunks: List of retrieved chunks
            
        Returns:
            List of relevance scores
        """
        if not chunks:
            return []

        pairs = [(query, chunk.content) for chunk in chunks]
        scores = self.model.predict(pairs)

        return [float(s) for s in scores]


# Singleton instance
reranker = Reranker()


async def get_reranker() -> Reranker:
    """Get the global reranker instance."""
    return reranker