"""
BM25 retrieval service for keyword-based search.
"""

import uuid
from typing import List, Optional, Tuple

from rank_bm25 import BM25Okapi

from app.core.logging import get_logger

logger = get_logger(__name__)


class BM25Index:
    """BM25 index for keyword-based document retrieval."""

    def __init__(self, corpus_name: str = "documents"):
        self.corpus_name = corpus_name
        self._documents: List[str] = []
        self._metadatas: List[dict] = []
        self._ids: List[str] = []
        self._index: Optional[BM25Okapi] = None

    def add_documents(
        self,
        texts: List[str],
        metadatas: List[dict],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents to the BM25 index.
        
        Args:
            texts: List of text contents
            metadatas: List of metadata dicts
            ids: Optional list of IDs
            
        Returns:
            List of document IDs
        """
        if not texts:
            return []

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        # Tokenize documents
        tokenized_docs = [doc.lower().split() for doc in texts]

        # Add to existing or create new index
        if self._index is not None:
            # Rebuild index with new documents
            self._documents.extend(texts)
            self._metadatas.extend(metadatas)
            self._ids.extend(ids)
            all_tokenized = [doc.lower().split() for doc in self._documents]
            self._index = BM25Okapi(all_tokenized)
        else:
            self._documents = texts
            self._metadatas = metadatas
            self._ids = ids
            self._index = BM25Okapi(tokenized_docs)

        logger.info(
            "bm25_documents_added",
            count=len(texts),
            total=len(self._documents),
        )

        return ids

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> Tuple[List[str], List[str], List[dict], List[float]]:
        """
        Search for documents using BM25.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            Tuple of (ids, documents, metadatas, scores)
        """
        if self._index is None or not self._documents:
            logger.warning("bm25_index_empty")
            return [], [], [], []

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get scores
        scores = self._index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        # Extract results
        ids = [self._ids[i] for i in top_indices]
        documents = [self._documents[i] for i in top_indices]
        metadatas = [self._metadatas[i] for i in top_indices]
        result_scores = [scores[i] for i in top_indices]

        logger.info(
            "bm25_search_completed",
            query=query[:50],
            n_results=len(ids),
        )

        return ids, documents, metadatas, result_scores

    def get_documents(self, ids: List[str]) -> Tuple[List[str], List[dict]]:
        """
        Get documents by IDs.
        
        Args:
            ids: List of document IDs
            
        Returns:
            Tuple of (documents, metadatas)
        """
        documents = []
        metadatas = []

        for doc_id in ids:
            if doc_id in self._ids:
                idx = self._ids.index(doc_id)
                documents.append(self._documents[idx])
                metadatas.append(self._metadatas[idx])

        return documents, metadatas

    def delete(self, ids: List[str]) -> None:
        """
        Delete documents by IDs.
        
        Args:
            ids: List of document IDs to delete
        """
        if self._index is None:
            return

        # Find indices to remove
        indices_to_remove = []
        for doc_id in ids:
            if doc_id in self._ids:
                indices_to_remove.append(self._ids.index(doc_id))

        if not indices_to_remove:
            return

        # Remove documents
        self._documents = [
            doc for i, doc in enumerate(self._documents) if i not in indices_to_remove
        ]
        self._metadatas = [
            meta for i, meta in enumerate(self._metadatas) if i not in indices_to_remove
        ]
        self._ids = [
            doc_id for i, doc_id in enumerate(self._ids) if i not in indices_to_remove
        ]

        # Rebuild index
        if self._documents:
            tokenized_docs = [doc.lower().split() for doc in self._documents]
            self._index = BM25Okapi(tokenized_docs)
        else:
            self._index = None

        logger.info("bm25_documents_deleted", count=len(ids))

    def clear(self) -> None:
        """Clear all documents from the index."""
        self._documents = []
        self._metadatas = []
        self._ids = []
        self._index = None
        logger.info("bm25_index_cleared")

    def count(self) -> int:
        """Get the number of documents in the index."""
        return len(self._documents)


# Singleton instance
bm25_index = BM25Index()


async def get_bm25_index() -> BM25Index:
    """Get the global BM25 index instance."""
    return bm25_index