"""Vector store service for hybrid search using Supabase/PostgreSQL."""

import json
from typing import Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.schemas import ChunkWithScore
from app.services.embedding import embedding_service


class VectorStoreService:
    """Hybrid search service using pgvector and pg_trgm."""

    async def hybrid_search(
        self,
        db: AsyncSession,
        query: str,
        org_id: UUID,
        document_ids: Optional[list[UUID]] = None,
        top_k: int = 20,
        semantic_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None,
        similarity_threshold: Optional[float] = None,
    ) -> list[ChunkWithScore]:
        """Perform hybrid semantic + keyword search.

        Args:
            db: Database session
            query: Search query
            org_id: Organization ID
            document_ids: Optional filter by document IDs
            top_k: Number of results to return
            semantic_weight: Weight for semantic similarity (0-1)
            keyword_weight: Weight for keyword matching (0-1)
            similarity_threshold: Minimum similarity score

        Returns:
            List of chunks with scores
        """
        # Get query embedding
        query_embedding = await embedding_service.get_query_embedding(query)

        # Use configured weights or defaults
        sem_weight = semantic_weight or settings.semantic_weight
        kw_weight = keyword_weight or settings.keyword_weight
        threshold = similarity_threshold or settings.similarity_threshold

        # Format embedding for PostgreSQL
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        # Format document IDs if provided
        doc_ids_param = None
        if document_ids:
            doc_ids_param = [str(d) for d in document_ids]

        # Execute hybrid search function
        result = await db.execute(
            text("""
                SELECT
                    chunk_id,
                    document_id,
                    content,
                    page_number,
                    section_header,
                    semantic_score,
                    keyword_score,
                    combined_score
                FROM hybrid_search(
                    CAST(:embedding AS vector),
                    :query_text,
                    CAST(:org_id AS uuid),
                    CAST(:doc_ids AS uuid[]),
                    :match_count,
                    :semantic_weight,
                    :keyword_weight,
                    :similarity_threshold
                )
            """),
            {
                "embedding": embedding_str,
                "query_text": query,
                "org_id": str(org_id),
                "doc_ids": doc_ids_param,
                "match_count": top_k,
                "semantic_weight": sem_weight,
                "keyword_weight": kw_weight,
                "similarity_threshold": threshold,
            },
        )

        rows = result.fetchall()

        # Convert to ChunkWithScore objects
        chunks = []
        for row in rows:
            chunk = ChunkWithScore(
                id=row.chunk_id,
                document_id=row.document_id,
                content=row.content,
                page_number=row.page_number,
                section_header=row.section_header,
                chunk_index=0,  # Not returned by function
                token_count=0,  # Not returned by function
                score=row.combined_score,
                semantic_score=row.semantic_score,
                keyword_score=row.keyword_score,
            )
            chunks.append(chunk)

        return chunks

    async def semantic_search(
        self,
        db: AsyncSession,
        query: str,
        org_id: UUID,
        document_ids: Optional[list[UUID]] = None,
        top_k: int = 20,
        similarity_threshold: Optional[float] = None,
    ) -> list[ChunkWithScore]:
        """Perform pure semantic/vector search.

        Args:
            db: Database session
            query: Search query
            org_id: Organization ID
            document_ids: Optional filter by document IDs
            top_k: Number of results
            similarity_threshold: Minimum similarity score

        Returns:
            List of chunks with scores
        """
        query_embedding = await embedding_service.get_query_embedding(query)
        threshold = similarity_threshold or settings.similarity_threshold

        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        sql = """
            SELECT
                dc.id as chunk_id,
                dc.document_id,
                dc.content,
                dc.page_number,
                dc.section_header,
                dc.chunk_index,
                dc.token_count,
                1 - (dc.embedding <=> CAST(:embedding AS vector)) as score
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE d.organization_id = CAST(:org_id AS uuid)
                AND d.status = 'completed'
                AND 1 - (dc.embedding <=> CAST(:embedding AS vector)) > :threshold
        """

        if document_ids:
            sql += " AND dc.document_id = ANY(CAST(:doc_ids AS uuid[]))"

        sql += " ORDER BY dc.embedding <=> CAST(:embedding AS vector) LIMIT :limit"

        params = {
            "embedding": embedding_str,
            "org_id": str(org_id),
            "threshold": threshold,
            "limit": top_k,
        }

        if document_ids:
            params["doc_ids"] = [str(d) for d in document_ids]

        result = await db.execute(text(sql), params)
        rows = result.fetchall()

        return [
            ChunkWithScore(
                id=row.chunk_id,
                document_id=row.document_id,
                content=row.content,
                page_number=row.page_number,
                section_header=row.section_header,
                chunk_index=row.chunk_index,
                token_count=row.token_count,
                score=row.score,
                semantic_score=row.score,
                keyword_score=0.0,
            )
            for row in rows
        ]

    async def keyword_search(
        self,
        db: AsyncSession,
        query: str,
        org_id: UUID,
        document_ids: Optional[list[UUID]] = None,
        top_k: int = 20,
    ) -> list[ChunkWithScore]:
        """Perform pure keyword/full-text search.

        Args:
            db: Database session
            query: Search query
            org_id: Organization ID
            document_ids: Optional filter by document IDs
            top_k: Number of results

        Returns:
            List of chunks with scores
        """
        sql = """
            SELECT
                dc.id as chunk_id,
                dc.document_id,
                dc.content,
                dc.page_number,
                dc.section_header,
                dc.chunk_index,
                dc.token_count,
                ts_rank_cd(dc.content_tsvector, plainto_tsquery('english', :query)) +
                similarity(dc.content, :query) * 0.5 as score
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE d.organization_id = CAST(:org_id AS uuid)
                AND d.status = 'completed'
                AND (
                    dc.content_tsvector @@ plainto_tsquery('english', :query)
                    OR dc.content % :query
                )
        """

        if document_ids:
            sql += " AND dc.document_id = ANY(CAST(:doc_ids AS uuid[]))"

        sql += " ORDER BY score DESC LIMIT :limit"

        params = {
            "query": query,
            "org_id": str(org_id),
            "limit": top_k,
        }

        if document_ids:
            params["doc_ids"] = [str(d) for d in document_ids]

        result = await db.execute(text(sql), params)
        rows = result.fetchall()

        return [
            ChunkWithScore(
                id=row.chunk_id,
                document_id=row.document_id,
                content=row.content,
                page_number=row.page_number,
                section_header=row.section_header,
                chunk_index=row.chunk_index,
                token_count=row.token_count,
                score=row.score,
                semantic_score=0.0,
                keyword_score=row.score,
            )
            for row in rows
        ]

    async def insert_chunks(
        self,
        db: AsyncSession,
        document_id: UUID,
        chunks: list[dict],
    ) -> int:
        """Insert document chunks with embeddings.

        Args:
            db: Database session
            document_id: Document ID
            chunks: List of chunk dicts with content, embedding, metadata

        Returns:
            Number of chunks inserted
        """
        if not chunks:
            return 0

        for i, chunk in enumerate(chunks):
            embedding_str = "[" + ",".join(str(x) for x in chunk["embedding"]) + "]"

            await db.execute(
                text("""
                    INSERT INTO document_chunks (
                        document_id, chunk_index, content, embedding,
                        page_number, section_header, token_count, metadata
                    ) VALUES (
                        CAST(:document_id AS uuid), :chunk_index, :content,
                        CAST(:embedding AS vector), :page_number, :section_header,
                        :token_count, CAST(:metadata AS jsonb)
                    )
                """),
                {
                    "document_id": str(document_id),
                    "chunk_index": i,
                    "content": chunk["content"],
                    "embedding": embedding_str,
                    "page_number": chunk.get("page_number"),
                    "section_header": chunk.get("section_header"),
                    "token_count": chunk.get("token_count", len(chunk["content"].split())),
                    "metadata": json.dumps(chunk.get("metadata", {})),
                },
            )

        await db.commit()
        return len(chunks)

    async def delete_document_chunks(
        self, db: AsyncSession, document_id: UUID
    ) -> int:
        """Delete all chunks for a document.

        Args:
            db: Database session
            document_id: Document ID

        Returns:
            Number of chunks deleted
        """
        result = await db.execute(
            text("DELETE FROM document_chunks WHERE document_id = CAST(:doc_id AS uuid)"),
            {"doc_id": str(document_id)},
        )
        await db.commit()
        return result.rowcount


# Global vector store instance
vector_store_service = VectorStoreService()
