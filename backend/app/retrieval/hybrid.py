"""
Hybrid retrieval service combining BM25 and vector search.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.core.config import get_settings
from app.core.logging import get_logger
from app.retrieval.bm25 import BM25Index, bm25_index as default_bm25_index
from app.services.vector_store import VectorStore, vector_store as default_vector_store

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class RetrievedChunk:
    """A retrieved chunk with metadata."""

    chunk_id: str
    content: str
    source: str
    score: float
    page: Optional[int] = None
    retrieval_method: str = "hybrid"


class HybridRetrieval:
    """Hybrid retrieval combining BM25 and vector search."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
    ):
        self.vector_store = vector_store or default_vector_store
        self.bm25_index = bm25_index or default_bm25_index

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_bm25: bool = True,
        use_vector: bool = True,
        use_reranking: bool = False,
        bm25_weight: float = 0.5,
    ) -> List[RetrievedChunk]:
        """
        Perform hybrid retrieval combining BM25 and vector search.
        
        Args:
            query: Query string
            top_k: Number of results to return
            use_bm25: Whether to use BM25
            use_vector: Whether to use vector search
            use_reranking: Whether to apply reranking
            bm25_weight: Weight for BM25 results (0-1)
            
        Returns:
            List of RetrievedChunk objects
        """
        results: List[RetrievedChunk] = []

        # BM25 retrieval
        if use_bm25:
            bm25_ids, bm25_docs, bm25_metadatas, bm25_scores = self.bm25_index.search(
                query, top_k=top_k * 2
            )

            for i, doc_id in enumerate(bm25_ids):
                results.append(
                    RetrievedChunk(
                        chunk_id=doc_id,
                        content=bm25_docs[i],
                        source=bm25_metadatas[i].get("source", "unknown"),
                        score=bm25_scores[i],
                        page=bm25_metadatas[i].get("page"),
                        retrieval_method="bm25",
                    )
                )

            logger.info("bm25_retrieval_completed", results=len(results))

        # Vector retrieval
        if use_vector:
            vector_ids, vector_docs, vector_metadatas, vector_distances = self.vector_store.search(
                query, n_results=top_k * 2
            )

            # Convert distances to similarity scores (1 - distance for cosine)
            vector_scores = [1 - d for d in vector_distances]

            for i, doc_id in enumerate(vector_ids):
                results.append(
                    RetrievedChunk(
                        chunk_id=doc_id,
                        content=vector_docs[i],
                        source=vector_metadatas[i].get("source", "unknown"),
                        score=vector_scores[i],
                        page=vector_metadatas[i].get("page"),
                        retrieval_method="vector",
                    )
                )

            logger.info("vector_retrieval_completed", results=len(results))

        # Combine and deduplicate results
        combined = self._combine_results(
            results, top_k=top_k, bm25_weight=bm25_weight
        )

        # Apply reranking if requested
        if use_reranking and combined:
            combined = await self._rerank(query, combined, top_k)

        return combined[:top_k]

    def _combine_results(
        self,
        results: List[RetrievedChunk],
        top_k: int,
        bm25_weight: float = 0.5,
    ) -> List[RetrievedChunk]:
        """Combine and rank results from multiple retrieval methods."""
        if not results:
            return []

        # Deduplicate by chunk_id, keeping highest score
        seen: dict[str, RetrievedChunk] = {}
        for chunk in results:
            if chunk.chunk_id not in seen:
                seen[chunk.chunk_id] = chunk
            else:
                # Keep the one with higher score
                if chunk.score > seen[chunk.chunk_id].score:
                    seen[chunk.chunk_id] = chunk

        # Normalize scores within each method
        normalized = []
        bm25_chunks = [c for c in seen.values() if c.retrieval_method == "bm25"]
        vector_chunks = [c for c in seen.values() if c.retrieval_method == "vector"]

        # Normalize BM25 scores
        if bm25_chunks:
            max_bm25 = max(c.score for c in bm25_chunks) or 1
            for chunk in bm25_chunks:
                chunk.score = (chunk.score / max_bm25) * bm25_weight

        # Normalize vector scores
        if vector_chunks:
            max_vector = max(c.score for c in vector_chunks) or 1
            for chunk in vector_chunks:
                chunk.score = (chunk.score / max_vector) * (1 - bm25_weight)

        # Combine all normalized scores
        all_chunks = list(seen.values())
        all_chunks.sort(key=lambda x: x.score, reverse=True)

        return all_chunks[:top_k]

    async def _rerank(
        self, query: str, chunks: List[RetrievedChunk], top_k: int
    ) -> List[RetrievedChunk]:
        """Rerank chunks using cross-encoder (placeholder)."""
        # This will be implemented in the reranking module
        # For now, return the top-k from combined results
        logger.info("reranking_requested", chunks=len(chunks))
        return chunks[:top_k]

    async def add_documents(
        self,
        texts: List[str],
        metadatas: List[dict],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents to both BM25 and vector stores.
        
        Args:
            texts: List of text contents
            metadatas: List of metadata dicts
            ids: Optional list of IDs
            
        Returns:
            List of document IDs
        """
        # Add to BM25
        bm25_ids = self.bm25_index.add_documents(texts, metadatas, ids)

        # Add to vector store
        vector_ids = self.vector_store.add_documents(texts, metadatas, ids)

        logger.info(
            "documents_added_to_hybrid",
            bm25=len(bm25_ids),
            vector=len(vector_ids),
        )

        return bm25_ids


# Singleton instance
hybrid_retrieval = HybridRetrieval()


async def get_hybrid_retrieval() -> HybridRetrieval:
    """Get the global hybrid retrieval instance."""
    return hybrid_retrieval
