"""
Vector store service using ChromaDB.
"""

import uuid
from typing import List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.embedding_service import embedding_service

logger = get_logger(__name__)
settings = get_settings()


class VectorStore:
    """Vector store using ChromaDB for similarity search."""

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory or settings.chroma_persist_dir
        self._client = None
        self._collection = None

    @property
    def client(self) -> chromadb.Client:
        """Lazy initialize ChromaDB client."""
        if self._client is None:
            logger.info("initializing_chromadb", persist_dir=self.persist_directory)
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        """Get or create the collection."""
        if self._collection is None:
            # Get embedding dimension
            dim = embedding_service.get_embedding_dimension()

            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
                # Note: Chroma doesn't support specifying dimension directly
                # It will be inferred from first insert
            )
            logger.info("collection_ready", name=self.collection_name)
        return self._collection

    def add_documents(
        self,
        texts: List[str],
        metadatas: List[dict],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            texts: List of text contents
            metadatas: List of metadata dicts
            ids: Optional list of IDs (generated if not provided)
            
        Returns:
            List of document IDs
        """
        if not texts:
            return []

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        # Generate embeddings
        embeddings = embedding_service.embed_texts(texts)

        # Add to collection
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        logger.info(
            "documents_added",
            count=len(texts),
            collection=self.collection_name,
        )

        return ids

    def search(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
    ) -> Tuple[List[str], List[str], List[dict], List[List[float]]]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            n_results: Number of results to return
            where: Metadata filter
            where_document: Document content filter
            
        Returns:
            Tuple of (ids, documents, metadatas, distances)
        """
        # Generate query embedding
        query_embedding = embedding_service.embed_query(query)

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"],
        )

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        logger.info(
            "vector_search_completed",
            query=query[:50],
            n_results=len(ids),
        )

        return ids, documents, metadatas, distances

    def get_by_id(self, ids: List[str]) -> Tuple[List[str], List[str], List[dict]]:
        """
        Get documents by IDs.
        
        Args:
            ids: List of document IDs
            
        Returns:
            Tuple of (ids, documents, metadatas)
        """
        results = self.collection.get(
            ids=ids,
            include=["documents", "metadatas"],
        )

        return (
            results.get("ids", []),
            results.get("documents", []),
            results.get("metadatas", []),
        )

    def delete(self, ids: List[str]) -> None:
        """
        Delete documents by IDs.
        
        Args:
            ids: List of document IDs to delete
        """
        self.collection.delete(ids=ids)
        logger.info("documents_deleted", count=len(ids))

    def reset(self) -> None:
        """Reset the collection (delete all documents)."""
        self.client.delete_collection(self.collection_name)
        self._collection = None
        logger.info("collection_reset", name=self.collection_name)

    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()


# Singleton instance
vector_store = VectorStore()


async def get_vector_store() -> VectorStore:
    """Get the global vector store instance."""
    return vector_store