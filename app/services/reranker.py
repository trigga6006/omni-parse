"""Cohere reranking service for improved retrieval precision."""

from typing import Optional

import cohere

from app.config import settings
from app.models.schemas import ChunkWithScore


class RerankerService:
    """Cohere reranking service."""

    def __init__(self):
        self._client: Optional[cohere.AsyncClient] = None

    def _get_client(self) -> cohere.AsyncClient:
        """Get or create Cohere client."""
        if not self._client:
            self._client = cohere.AsyncClient(api_key=settings.cohere_api_key)
        return self._client

    async def rerank(
        self,
        query: str,
        chunks: list[ChunkWithScore],
        top_n: Optional[int] = None,
    ) -> list[ChunkWithScore]:
        """Rerank chunks using Cohere reranker.

        Args:
            query: Search query
            chunks: List of chunks to rerank
            top_n: Number of top results to return

        Returns:
            Reranked list of chunks
        """
        if not chunks:
            return []

        top_n = top_n or settings.rerank_top_n

        # Don't rerank if we have fewer chunks than top_n
        if len(chunks) <= top_n:
            return chunks

        client = self._get_client()

        # Prepare documents for reranking
        documents = [chunk.content for chunk in chunks]

        # Call Cohere rerank API
        response = await client.rerank(
            model=settings.rerank_model,
            query=query,
            documents=documents,
            top_n=top_n,
            return_documents=False,
        )

        # Map results back to chunks with updated scores
        reranked_chunks = []
        for result in response.results:
            chunk = chunks[result.index]
            # Create new chunk with reranked score
            reranked_chunk = ChunkWithScore(
                id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                page_number=chunk.page_number,
                section_header=chunk.section_header,
                chunk_index=chunk.chunk_index,
                token_count=chunk.token_count,
                metadata=chunk.metadata,
                score=result.relevance_score,
                semantic_score=chunk.semantic_score,
                keyword_score=chunk.keyword_score,
            )
            reranked_chunks.append(reranked_chunk)

        return reranked_chunks

    async def rerank_with_context(
        self,
        query: str,
        chunks: list[ChunkWithScore],
        conversation_context: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> list[ChunkWithScore]:
        """Rerank chunks with conversation context.

        Args:
            query: Current search query
            chunks: List of chunks to rerank
            conversation_context: Previous conversation for context
            top_n: Number of top results to return

        Returns:
            Reranked list of chunks
        """
        if not chunks:
            return []

        # Enhance query with conversation context
        enhanced_query = query
        if conversation_context:
            enhanced_query = f"Context: {conversation_context}\n\nQuery: {query}"

        return await self.rerank(enhanced_query, chunks, top_n)


# Global reranker instance
reranker_service = RerankerService()
