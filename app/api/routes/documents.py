"""Document upload and management endpoints."""

import json
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import func, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.middleware.clerk_auth import ClerkUser, require_org_membership
from app.config import settings
from app.dependencies import get_db, get_org_uuid
from app.models.schemas import (
    Document,
    DocumentList,
    DocumentStatus,
    DocumentUploadResponse,
    ErrorResponse,
)
from app.services.document_processor import document_processor
from app.services.embedding import embedding_service
from app.services.storage import storage_service
from app.services.vector_store import vector_store_service

router = APIRouter(prefix="/documents", tags=["documents"])


async def process_document_background(
    document_id: UUID,
    org_id: UUID,
    filename: str,
    file_content: bytes,
) -> None:
    """Background task to process uploaded document."""
    from app.dependencies import get_db_context

    async with get_db_context() as db:
        try:
            # Update status to processing
            await db.execute(
                text("""
                    UPDATE documents
                    SET status = 'processing', updated_at = NOW()
                    WHERE id = :doc_id::uuid
                """),
                {"doc_id": str(document_id)},
            )
            await db.commit()

            # Process document and generate embeddings
            processed, embeddings = await document_processor.process_and_embed(
                file_content, filename
            )

            # Prepare chunks with embeddings
            chunks_data = []
            for chunk, embedding in zip(processed.chunks, embeddings):
                chunks_data.append({
                    "content": chunk.content,
                    "embedding": embedding,
                    "page_number": chunk.page_number,
                    "section_header": chunk.section_header,
                    "token_count": chunk.token_count,
                    "metadata": chunk.metadata,
                })

            # Insert chunks into vector store
            chunk_count = await vector_store_service.insert_chunks(
                db, document_id, chunks_data
            )

            # Update document status to completed
            await db.execute(
                text("""
                    UPDATE documents
                    SET status = 'completed',
                        chunk_count = :chunk_count,
                        updated_at = NOW()
                    WHERE id = :doc_id::uuid
                """),
                {"doc_id": str(document_id), "chunk_count": chunk_count},
            )

            # Update organization storage usage
            file_size_mb = len(file_content) / (1024 * 1024)
            await db.execute(
                text("""
                    UPDATE organizations
                    SET storage_used_mb = storage_used_mb + :size_mb,
                        updated_at = NOW()
                    WHERE id = :org_id::uuid
                """),
                {"org_id": str(org_id), "size_mb": file_size_mb},
            )
            await db.commit()

        except Exception as e:
            # Update status to failed
            await db.execute(
                text("""
                    UPDATE documents
                    SET status = 'failed',
                        error_message = :error,
                        updated_at = NOW()
                    WHERE id = :doc_id::uuid
                """),
                {"doc_id": str(document_id), "error": str(e)},
            )
            await db.commit()
            raise


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = None,
    description: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    org_id: UUID = Depends(get_org_uuid),
    user: ClerkUser = Depends(require_org_membership),
):
    """Upload a document for processing.

    Accepts PDF files up to the configured max size.
    Processing happens in the background.
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported",
        )

    # Read file content
    content = await file.read()
    file_size = len(content)

    # Validate file size
    max_size = settings.max_file_size_mb * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum of {settings.max_file_size_mb}MB",
        )

    # Create document record
    result = await db.execute(
        text("""
            INSERT INTO documents (
                organization_id, filename, title, description,
                file_path, file_size, mime_type, status
            ) VALUES (
                :org_id::uuid, :filename, :title, :description,
                '', :file_size, :mime_type, 'pending'
            )
            RETURNING id
        """),
        {
            "org_id": str(org_id),
            "filename": file.filename,
            "title": title or file.filename,
            "description": description,
            "file_size": file_size,
            "mime_type": file.content_type or "application/pdf",
        },
    )
    document_id = result.fetchone()[0]
    await db.commit()

    # Upload to storage
    file_path = await storage_service.upload_file(
        org_id, document_id, file.filename, content
    )

    # Update file path
    await db.execute(
        text("UPDATE documents SET file_path = :path WHERE id = :id::uuid"),
        {"path": file_path, "id": str(document_id)},
    )
    await db.commit()

    # Process document in background
    background_tasks.add_task(
        process_document_background,
        document_id,
        org_id,
        file.filename,
        content,
    )

    return DocumentUploadResponse(
        id=document_id,
        filename=file.filename,
        status=DocumentStatus.PENDING,
        message="Document uploaded and queued for processing",
    )


@router.get("", response_model=DocumentList)
async def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status_filter: Optional[DocumentStatus] = Query(None, alias="status"),
    db: AsyncSession = Depends(get_db),
    org_id: UUID = Depends(get_org_uuid),
    user: ClerkUser = Depends(require_org_membership),
):
    """List documents for the organization."""
    offset = (page - 1) * page_size

    # Build query
    where_clause = "WHERE d.organization_id = :org_id::uuid"
    params = {"org_id": str(org_id), "limit": page_size, "offset": offset}

    if status_filter:
        where_clause += " AND d.status = :status"
        params["status"] = status_filter.value

    # Get documents
    result = await db.execute(
        text(f"""
            SELECT
                d.id, d.organization_id, d.filename, d.title, d.description,
                d.file_path, d.file_size, d.mime_type, d.status,
                d.chunk_count, d.error_message, d.created_at, d.updated_at
            FROM documents d
            {where_clause}
            ORDER BY d.created_at DESC
            LIMIT :limit OFFSET :offset
        """),
        params,
    )
    rows = result.fetchall()

    # Get total count
    count_result = await db.execute(
        text(f"""
            SELECT COUNT(*) FROM documents d {where_clause}
        """),
        {"org_id": str(org_id), "status": status_filter.value if status_filter else None},
    )
    total = count_result.scalar()

    documents = [
        Document(
            id=row.id,
            organization_id=row.organization_id,
            filename=row.filename,
            title=row.title,
            description=row.description,
            file_path=row.file_path,
            file_size=row.file_size,
            mime_type=row.mime_type,
            status=row.status,
            chunk_count=row.chunk_count,
            error_message=row.error_message,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
        for row in rows
    ]

    return DocumentList(
        documents=documents,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{document_id}", response_model=Document)
async def get_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
    org_id: UUID = Depends(get_org_uuid),
    user: ClerkUser = Depends(require_org_membership),
):
    """Get a specific document by ID."""
    result = await db.execute(
        text("""
            SELECT
                id, organization_id, filename, title, description,
                file_path, file_size, mime_type, status,
                chunk_count, error_message, created_at, updated_at
            FROM documents
            WHERE id = :doc_id::uuid AND organization_id = :org_id::uuid
        """),
        {"doc_id": str(document_id), "org_id": str(org_id)},
    )
    row = result.fetchone()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return Document(
        id=row.id,
        organization_id=row.organization_id,
        filename=row.filename,
        title=row.title,
        description=row.description,
        file_path=row.file_path,
        file_size=row.file_size,
        mime_type=row.mime_type,
        status=row.status,
        chunk_count=row.chunk_count,
        error_message=row.error_message,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
    org_id: UUID = Depends(get_org_uuid),
    user: ClerkUser = Depends(require_org_membership),
):
    """Delete a document and its chunks."""
    # Get document info first
    result = await db.execute(
        text("""
            SELECT filename, file_size
            FROM documents
            WHERE id = :doc_id::uuid AND organization_id = :org_id::uuid
        """),
        {"doc_id": str(document_id), "org_id": str(org_id)},
    )
    row = result.fetchone()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    # Delete from storage
    await storage_service.delete_document_files(org_id, document_id)

    # Delete document (cascades to chunks)
    await db.execute(
        text("DELETE FROM documents WHERE id = :doc_id::uuid"),
        {"doc_id": str(document_id)},
    )

    # Update organization storage usage
    file_size_mb = row.file_size / (1024 * 1024)
    await db.execute(
        text("""
            UPDATE organizations
            SET storage_used_mb = GREATEST(0, storage_used_mb - :size_mb),
                updated_at = NOW()
            WHERE id = :org_id::uuid
        """),
        {"org_id": str(org_id), "size_mb": file_size_mb},
    )

    await db.commit()


@router.post("/{document_id}/reprocess", response_model=DocumentUploadResponse)
async def reprocess_document(
    document_id: UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    org_id: UUID = Depends(get_org_uuid),
    user: ClerkUser = Depends(require_org_membership),
):
    """Reprocess a failed document."""
    # Get document
    result = await db.execute(
        text("""
            SELECT id, filename, file_path, status
            FROM documents
            WHERE id = :doc_id::uuid AND organization_id = :org_id::uuid
        """),
        {"doc_id": str(document_id), "org_id": str(org_id)},
    )
    row = result.fetchone()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    if row.status not in ["failed", "completed"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document is currently processing",
        )

    # Delete existing chunks
    await vector_store_service.delete_document_chunks(db, document_id)

    # Reset status
    await db.execute(
        text("""
            UPDATE documents
            SET status = 'pending', error_message = NULL, chunk_count = 0
            WHERE id = :doc_id::uuid
        """),
        {"doc_id": str(document_id)},
    )
    await db.commit()

    # Download file from storage
    file_content = await storage_service.download_file(
        org_id, document_id, row.filename
    )

    # Reprocess in background
    background_tasks.add_task(
        process_document_background,
        document_id,
        org_id,
        row.filename,
        file_content,
    )

    return DocumentUploadResponse(
        id=document_id,
        filename=row.filename,
        status=DocumentStatus.PENDING,
        message="Document queued for reprocessing",
    )
