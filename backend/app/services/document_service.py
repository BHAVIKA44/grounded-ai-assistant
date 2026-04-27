"""
Document service for managing document lifecycle.
Handles upload, storage, chunking, and retrieval.
"""

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import Document, DocumentChunk
from app.services.chunker import ChunkingConfig, ChunkingStrategy, DocumentChunker
from app.services.document_parser import parse_document

logger = get_logger(__name__)
settings = get_settings()


class DocumentService:
    """Service for managing documents."""

    def __init__(self):
        self.chunker = DocumentChunker(
            ChunkingConfig(
                chunk_size=500,
                chunk_overlap=50,
                strategy=ChunkingStrategy.RECURSIVE,
            )
        )

    async def create_document(
        self,
        session: AsyncSession,
        title: str,
        content: str,
        document_type: str,
        filename: str,
    ) -> Document:
        """
        Create a new document with chunks.
        
        Args:
            session: Database session
            title: Document title
            content: Document content
            document_type: Type (pdf, txt, docx)
            filename: Original filename
            
        Returns:
            Created Document object
        """
        document_id = str(uuid.uuid4())

        # Create document record
        document = Document(
            id=document_id,
            title=title,
            document_type=document_type,
            content=content,
            chunk_count=0,
            metadata={"filename": filename},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        session.add(document)
        await session.flush()

        # Parse and chunk the document
        try:
            # Determine file extension
            ext = filename.split(".")[-1] if "." in filename else document_type

            # Parse document (for text content, use directly)
            if document_type == "text":
                pages = [(content, None)]
            else:
                # For file-based parsing, we'd need the actual file bytes
                # This is a simplified version
                pages = [(content, None)]

            # Create chunks
            chunks = []
            for page_text, page_num in pages:
                page_chunks = self.chunker.chunk_document(
                    text=page_text,
                    document_id=document_id,
                    source=filename,
                    page=page_num,
                )
                chunks.extend(page_chunks)

            # Save chunks to database
            for chunk in chunks:
                db_chunk = DocumentChunk(
                    id=chunk.id,
                    document_id=document_id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    source=chunk.source,
                    page=chunk.page,
                    metadata={},
                    created_at=datetime.utcnow(),
                )
                session.add(db_chunk)

            # Update chunk count
            document.chunk_count = len(chunks)

            logger.info(
                "document_created",
                document_id=document_id,
                title=title,
                chunk_count=len(chunks),
            )

            return document

        except Exception as e:
            logger.error("document_creation_failed", error=str(e), title=title)
            await session.rollback()
            raise

    async def get_document(
        self, session: AsyncSession, document_id: str
    ) -> Optional[Document]:
        """Get a document by ID."""
        result = await session.execute(
            select(Document).where(Document.id == document_id)
        )
        return result.scalar_one_or_none()

    async def list_documents(
        self, session: AsyncSession, page: int = 1, page_size: int = 10
    ) -> tuple[List[Document], int]:
        """List documents with pagination."""
        # Get total count
        count_result = await session.execute(select(Document))
        total = len(count_result.scalars().all())

        # Get paginated results
        offset = (page - 1) * page_size
        result = await session.execute(
            select(Document)
            .order_by(Document.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        documents = list(result.scalars().all())

        return documents, total

    async def delete_document(
        self, session: AsyncSession, document_id: str
    ) -> bool:
        """Delete a document and its chunks."""
        document = await self.get_document(session, document_id)
        if not document:
            return False

        await session.delete(document)
        await session.flush()

        logger.info("document_deleted", document_id=document_id)
        return True

    async def get_document_chunks(
        self, session: AsyncSession, document_id: str
    ) -> List[DocumentChunk]:
        """Get all chunks for a document."""
        result = await session.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        )
        return list(result.scalars().all())


# Singleton instance
document_service = DocumentService()
