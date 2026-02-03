"""OpenAI embedding service with caching."""

from typing import Optional

import openai

from app.config import settings
from app.services.cache import cache_service


class EmbeddingService:
    """OpenAI embedding service with Redis caching."""

    def __init__(self):
        self._client: Optional[openai.AsyncOpenAI] = None

    def _get_client(self) -> openai.AsyncOpenAI:
        """Get or create OpenAI client."""
        if not self._client:
            self._client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        return self._client

    async def get_embedding(self, text: str, use_cache: bool = True) -> list[float]:
        """Get embedding for a single text.

        Args:
            text: Text to embed
            use_cache: Whether to use cache

        Returns:
            Embedding vector
        """
        # Normalize text
        text = text.strip().replace("\n", " ")

        # Check cache
        if use_cache:
            cached = await cache_service.get_embedding(text)
            if cached:
                return cached

        # Get from OpenAI
        client = self._get_client()
        response = await client.embeddings.create(
            model=settings.embedding_model,
            input=text,
            dimensions=settings.embedding_dimensions,
        )

        embedding = response.data[0].embedding

        # Cache result
        if use_cache:
            await cache_service.set_embedding(text, embedding)

        return embedding

    async def get_embeddings_batch(
        self, texts: list[str], use_cache: bool = True
    ) -> list[list[float]]:
        """Get embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            use_cache: Whether to use cache

        Returns:
            List of embedding vectors in same order as input
        """
        if not texts:
            return []

        # Normalize texts
        normalized = [t.strip().replace("\n", " ") for t in texts]

        # Check cache for all texts
        cached_embeddings = {}
        uncached_texts = normalized

        if use_cache:
            cached_embeddings, uncached_texts = await cache_service.get_embeddings_batch(
                normalized
            )

        # Get embeddings for uncached texts
        new_embeddings = {}
        if uncached_texts:
            client = self._get_client()

            # Keep API batches small to limit memory from holding many embeddings
            batch_size = 50
            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i : i + batch_size]

                response = await client.embeddings.create(
                    model=settings.embedding_model,
                    input=batch,
                    dimensions=settings.embedding_dimensions,
                )

                for j, item in enumerate(response.data):
                    new_embeddings[batch[j]] = item.embedding

            # Cache new embeddings
            if use_cache and new_embeddings:
                await cache_service.set_embeddings_batch(new_embeddings)

        # Combine results in original order
        all_embeddings = {**cached_embeddings, **new_embeddings}
        return [all_embeddings[text] for text in normalized]

    async def get_query_embedding(self, query: str) -> list[float]:
        """Get embedding for a search query.

        Uses caching and same model as document embeddings.
        """
        return await self.get_embedding(query, use_cache=True)


# Global embedding service instance
embedding_service = EmbeddingService()
