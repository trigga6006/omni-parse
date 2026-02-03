"""Query endpoints for RAG pipeline."""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.middleware.clerk_auth import ClerkUser, require_org_membership
from app.dependencies import get_db, get_org_uuid
from app.models.schemas import QueryRequest, QueryResponse, QueryHistory, ErrorResponse
from app.services.rag import rag_service

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
    org_id: UUID = Depends(get_org_uuid),
    user: ClerkUser = Depends(require_org_membership),
):
    """Query documents using RAG pipeline.

    Performs hybrid search, reranking, and LLM response generation.
    """
    # Check if organization has any documents
    result = await db.execute(
        text("""
            SELECT COUNT(*) FROM documents
            WHERE organization_id = :org_id::uuid AND status = 'completed'
        """),
        {"org_id": str(org_id)},
    )
    doc_count = result.scalar()

    if doc_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No processed documents available. Please upload and wait for processing to complete.",
        )

    # Validate document IDs if provided
    if request.document_ids:
        result = await db.execute(
            text("""
                SELECT COUNT(*) FROM documents
                WHERE id = ANY(:doc_ids::uuid[])
                    AND organization_id = :org_id::uuid
                    AND status = 'completed'
            """),
            {
                "doc_ids": [str(d) for d in request.document_ids],
                "org_id": str(org_id),
            },
        )
        valid_count = result.scalar()

        if valid_count != len(request.document_ids):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="One or more document IDs are invalid or not processed",
            )

    # Execute RAG pipeline
    response = await rag_service.query(
        db=db,
        query=request.query,
        org_id=org_id,
        session_id=request.session_id,
        document_ids=request.document_ids,
        top_k=request.top_k,
        include_sources=request.include_sources,
    )

    # Log query (for analytics)
    await db.execute(
        text("""
            INSERT INTO query_logs (
                organization_id, session_id, query, answer,
                source_chunks, processing_time_ms, cached
            ) VALUES (
                :org_id::uuid, :session_id, :query, :answer,
                :sources::jsonb, :processing_time, :cached
            )
        """),
        {
            "org_id": str(org_id),
            "session_id": response.session_id,
            "query": request.query,
            "answer": response.answer,
            "sources": [{"chunk_id": str(s.chunk_id), "score": s.relevance_score} for s in response.sources],
            "processing_time": response.processing_time_ms,
            "cached": response.cached,
        },
    )
    await db.commit()

    return response


@router.post("/stream")
async def query_documents_streaming(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
    org_id: UUID = Depends(get_org_uuid),
    user: ClerkUser = Depends(require_org_membership),
):
    """Query documents with streaming response.

    Returns Server-Sent Events (SSE) with response chunks.
    """
    # Validate similar to non-streaming endpoint
    result = await db.execute(
        text("""
            SELECT COUNT(*) FROM documents
            WHERE organization_id = :org_id::uuid AND status = 'completed'
        """),
        {"org_id": str(org_id)},
    )
    doc_count = result.scalar()

    if doc_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No processed documents available",
        )

    async def generate():
        async for chunk in rag_service.query_streaming(
            db=db,
            query=request.query,
            org_id=org_id,
            session_id=request.session_id,
            document_ids=request.document_ids,
            top_k=request.top_k,
        ):
            yield chunk

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history", response_model=list[QueryHistory])
async def get_query_history(
    limit: int = 20,
    offset: int = 0,
    session_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    org_id: UUID = Depends(get_org_uuid),
    user: ClerkUser = Depends(require_org_membership),
):
    """Get query history for the organization."""
    params = {
        "org_id": str(org_id),
        "limit": limit,
        "offset": offset,
    }

    where_clause = "WHERE organization_id = :org_id::uuid"
    if session_id:
        where_clause += " AND session_id = :session_id"
        params["session_id"] = session_id

    result = await db.execute(
        text(f"""
            SELECT id, query, answer, created_at
            FROM query_logs
            {where_clause}
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """),
        params,
    )
    rows = result.fetchall()

    return [
        QueryHistory(
            id=row.id,
            query=row.query,
            answer=row.answer,
            created_at=row.created_at,
        )
        for row in rows
    ]
