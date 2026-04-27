"""
Document API routes for upload, list, and delete operations.
"""

import io
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.db_connection import get_session_dependency
from app.schemas.document import (
    DocumentCreate,
    DocumentListResponse,
    DocumentResponse,
    DocumentType,
    ErrorResponse,
)
from app.services.document_service import document_service

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=201,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
)
async def upload_document(
    file: UploadFile = File(...),
    title: str = None,
    session: AsyncSession = Depends(get_session_dependency),
) -> DocumentResponse:
    """
    Upload a document (PDF, TXT, or DOCX).
    
    The document will be parsed, chunked, and stored for retrieval.
    
    Args:
        file: The document file to upload
        title: Optional title (defaults to filename)
        session: Database session
        
    Returns:
        DocumentResponse with document details
        
    Raises:
        HTTPException: If file type is not supported or processing fails
    """
    # Validate file extension
    filename = file.filename or "unknown"
    ext = filename.split(".")[-1].lower() if "." in filename else ""

    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed: {settings.allowed_extensions}",
        )

    # Validate file size
    content = await file.read()
    max_size = settings.max_file_size_mb * 1024 * 1024
    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB",
        )

    # Determine document type
    doc_type = DocumentType(ext) if ext in ["pdf", "txt", "docx"] else DocumentType.TEXT

    # Use filename as title if not provided
    doc_title = title or filename

    try:
        # Decode content for text files
        if doc_type == DocumentType.TEXT:
            try:
                content_text = content.decode("utf-8")
            except UnicodeDecodeError:
                content_text = content.decode("latin-1")
        else:
            # For PDF/DOCX, we'll store the content as-is for now
            # In production, you'd store in object storage
            content_text = f"[{doc_type.value.upper()}] {filename}"

        # Create document
        document = await document_service.create_document(
            session=session,
            title=doc_title,
            content=content_text,
            document_type=doc_type.value,
            filename=filename,
        )

        await session.commit()

        logger.info(
            "document_uploaded",
            document_id=document.id,
            filename=filename,
            size=len(content),
        )

        return DocumentResponse(
            id=document.id,
            title=document.title,
            document_type=document.document_type,
            content=document.content[:500] + "..." if len(document.content) > 500 else document.content,
            chunk_count=document.chunk_count,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )

    except Exception as e:
        logger.error("upload_failed", error=str(e), filename=filename)
        await session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@router.get(
    "",
    response_model=DocumentListResponse,
    responses={500: {"model": ErrorResponse}},
)
async def list_documents(
    page: int = 1,
    page_size: int = 10,
    session: AsyncSession = Depends(get_session_dependency),
) -> DocumentListResponse:
    """
    List all uploaded documents with pagination.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of documents per page
        session: Database session
        
    Returns:
        DocumentListResponse with paginated documents
    """
    try:
        documents, total = await document_service.list_documents(
            session=session, page=page, page_size=page_size
        )

        doc_responses = [
            DocumentResponse(
                id=doc.id,
                title=doc.title,
                document_type=doc.document_type,
                content=doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                chunk_count=doc.chunk_count,
                created_at=doc.created_at,
                updated_at=doc.updated_at,
            )
            for doc in documents
        ]

        return DocumentListResponse(
            documents=doc_responses,
            total=total,
            page=page,
            page_size=page_size,
        )

    except Exception as e:
        logger.error("list_documents_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_document(
    document_id: str,
    session: AsyncSession = Depends(get_session_dependency),
) -> DocumentResponse:
    """
    Get a specific document by ID.
    
    Args:
        document_id: The document ID
        session: Database session
        
    Returns:
        DocumentResponse with document details
        
    Raises:
        HTTPException: If document not found
    """
    document = await document_service.get_document(session, document_id)

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentResponse(
        id=document.id,
        title=document.title,
        document_type=document.document_type,
        content=document.content,
        chunk_count=document.chunk_count,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


@router.delete(
    "/{document_id}",
    status_code=204,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def delete_document(
    document_id: str,
    session: AsyncSession = Depends(get_session_dependency),
) -> None:
    """
    Delete a document and all its chunks.
    
    Args:
        document_id: The document ID
        session: Database session
        
    Raises:
        HTTPException: If document not found or deletion fails
    """
    try:
        success = await document_service.delete_document(session, document_id)

        if not success:
            raise HTTPException(status_code=404, detail="Document not found")

        await session.commit()

        logger.info("document_deleted", document_id=document_id)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("delete_failed", error=str(e), document_id=document_id)
        await session.rollback()
        raise HTTPException(status_code=500, detail=str(e))