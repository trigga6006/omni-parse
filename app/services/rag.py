"""Full RAG pipeline orchestration."""

import time
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.schemas import QueryResponse, QuerySource, ChunkWithScore
from app.services.cache import cache_service
from app.services.llm import llm_service
from app.services.memory import memory_service
from app.services.reranker import reranker_service
from app.services.vector_store import vector_store_service


class RAGService:
    """Full RAG pipeline orchestration service."""

    async def query(
        self,
        db: AsyncSession,
        query: str,
        org_id: UUID,
        session_id: Optional[str] = None,
        document_ids: Optional[list[UUID]] = None,
        top_k: int = 5,
        include_sources: bool = True,
        use_cache: bool = True,
    ) -> QueryResponse:
        """Execute full RAG pipeline.

        Args:
            db: Database session
            query: User query
            org_id: Organization ID
            session_id: Optional conversation session ID
            document_ids: Optional filter by document IDs
            top_k: Number of sources to include
            include_sources: Whether to include source documents
            use_cache: Whether to use query cache

        Returns:
            QueryResponse with answer and sources
        """
        start_time = time.time()
        query_id = uuid4()

        # Check cache first
        if use_cache:
            cached = await cache_service.get_query_response(
                str(org_id),
                query,
                [str(d) for d in document_ids] if document_ids else None,
            )
            if cached:
                processing_time = int((time.time() - start_time) * 1000)
                return QueryResponse(
                    answer=cached["answer"],
                    sources=[QuerySource(**s) for s in cached["sources"]],
                    session_id=session_id or str(uuid4()),
                    query_id=query_id,
                    cached=True,
                    processing_time_ms=processing_time,
                )

        # Get conversation context if session exists
        conversation_history = None
        if session_id:
            exists = await memory_service.session_exists(str(org_id), session_id)
            if exists:
                conversation_history = await memory_service.get_context_messages(
                    str(org_id), session_id, limit=6
                )
            else:
                # Create new session
                session_id = await memory_service.create_session(str(org_id))

        if not session_id:
            session_id = await memory_service.create_session(str(org_id))

        # Step 1: Hybrid search - retrieve initial candidates
        initial_chunks = await vector_store_service.hybrid_search(
            db=db,
            query=query,
            org_id=org_id,
            document_ids=document_ids,
            top_k=settings.initial_results,
        )

        if not initial_chunks:
            # No relevant documents found
            answer = "I couldn't find any relevant information in the documentation to answer your question. Please try rephrasing your question or ensure the relevant documents have been uploaded."

            await self._save_to_memory(
                org_id, session_id, query, answer, []
            )

            processing_time = int((time.time() - start_time) * 1000)
            return QueryResponse(
                answer=answer,
                sources=[],
                session_id=session_id,
                query_id=query_id,
                cached=False,
                processing_time_ms=processing_time,
            )

        # Step 2: Rerank to get top-k most relevant
        reranked_chunks = await reranker_service.rerank(
            query=query,
            chunks=initial_chunks,
            top_n=top_k,
        )

        # Step 3: Generate response with LLM
        answer = await llm_service.generate_response(
            query=query,
            chunks=reranked_chunks,
            conversation_history=conversation_history,
        )

        # Build sources
        sources = []
        if include_sources:
            sources = await self._build_sources(db, reranked_chunks)

        # Save to conversation memory
        await self._save_to_memory(
            org_id, session_id, query, answer, sources
        )

        # Cache the response
        if use_cache:
            await cache_service.set_query_response(
                str(org_id),
                query,
                {
                    "answer": answer,
                    "sources": [s.model_dump() for s in sources],
                },
                [str(d) for d in document_ids] if document_ids else None,
            )

        processing_time = int((time.time() - start_time) * 1000)

        return QueryResponse(
            answer=answer,
            sources=sources,
            session_id=session_id,
            query_id=query_id,
            cached=False,
            processing_time_ms=processing_time,
        )

    async def query_streaming(
        self,
        db: AsyncSession,
        query: str,
        org_id: UUID,
        session_id: Optional[str] = None,
        document_ids: Optional[list[UUID]] = None,
        top_k: int = 5,
    ):
        """Execute RAG pipeline with streaming response.

        Args:
            db: Database session
            query: User query
            org_id: Organization ID
            session_id: Optional conversation session ID
            document_ids: Optional filter by document IDs
            top_k: Number of sources to include

        Yields:
            Response chunks as SSE events
        """
        import json

        # Get or create session
        if not session_id:
            session_id = await memory_service.create_session(str(org_id))
        else:
            exists = await memory_service.session_exists(str(org_id), session_id)
            if not exists:
                session_id = await memory_service.create_session(str(org_id))

        # Get conversation context
        conversation_history = await memory_service.get_context_messages(
            str(org_id), session_id, limit=6
        )

        # Hybrid search
        initial_chunks = await vector_store_service.hybrid_search(
            db=db,
            query=query,
            org_id=org_id,
            document_ids=document_ids,
            top_k=settings.initial_results,
        )

        if not initial_chunks:
            yield f"data: {json.dumps({'type': 'error', 'content': 'No relevant documents found'})}\n\n"
            return

        # Rerank
        reranked_chunks = await reranker_service.rerank(
            query=query,
            chunks=initial_chunks,
            top_n=top_k,
        )

        # Send sources first
        sources = await self._build_sources(db, reranked_chunks)
        yield f"data: {json.dumps({'type': 'sources', 'content': [s.model_dump() for s in sources]})}\n\n"

        # Stream response
        full_response = []
        async for chunk in llm_service.generate_response_streaming(
            query=query,
            chunks=reranked_chunks,
            conversation_history=conversation_history,
        ):
            full_response.append(chunk)
            yield f"data: {json.dumps({'type': 'text', 'content': chunk})}\n\n"

        # Save to memory
        answer = "".join(full_response)
        await self._save_to_memory(org_id, session_id, query, answer, sources)

        # Send completion
        yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"

    async def _build_sources(
        self, db: AsyncSession, chunks: list[ChunkWithScore]
    ) -> list[QuerySource]:
        """Build source objects from chunks."""
        from sqlalchemy import text

        sources = []
        for chunk in chunks:
            # Get document title
            result = await db.execute(
                text("SELECT title, filename FROM documents WHERE id = CAST(:doc_id AS uuid)"),
                {"doc_id": str(chunk.document_id)},
            )
            row = result.fetchone()
            doc_title = row.title if row and row.title else row.filename if row else "Unknown"

            sources.append(QuerySource(
                document_id=chunk.document_id,
                document_title=doc_title,
                chunk_id=chunk.id,
                content=chunk.content[:500],  # Truncate for response
                page_number=chunk.page_number,
                section_header=chunk.section_header,
                relevance_score=chunk.score,
            ))

        return sources

    async def _save_to_memory(
        self,
        org_id: UUID,
        session_id: str,
        query: str,
        answer: str,
        sources: list[QuerySource],
    ) -> None:
        """Save query and response to conversation memory."""
        # Add user message
        await memory_service.add_message(
            str(org_id), session_id, "user", query
        )

        # Add assistant message with sources
        await memory_service.add_message(
            str(org_id),
            session_id,
            "assistant",
            answer,
            [s.model_dump() for s in sources] if sources else None,
        )


# Global RAG service instance
rag_service = RAGService()
