"""Redis caching service for queries and embeddings."""

import hashlib
import json
from typing import Any, Optional

import redis.asyncio as redis

from app.config import settings


class CacheService:
    """Redis-backed caching service."""

    def __init__(self):
        self._client: Optional[redis.Redis] = None

    async def connect(self) -> None:
        """Initialize Redis connection."""
        self._client = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()

    @property
    def client(self) -> redis.Redis:
        """Get Redis client."""
        if not self._client:
            raise RuntimeError("Cache not connected. Call connect() first.")
        return self._client

    def _generate_key(self, prefix: str, data: str) -> str:
        """Generate cache key from data hash."""
        hash_value = hashlib.sha256(data.encode()).hexdigest()[:16]
        return f"{prefix}:{hash_value}"

    # Query caching
    async def get_query_response(
        self, org_id: str, query: str, document_ids: Optional[list[str]] = None
    ) -> Optional[dict[str, Any]]:
        """Get cached query response."""
        cache_data = f"{org_id}:{query}:{sorted(document_ids) if document_ids else ''}"
        key = self._generate_key("query", cache_data)

        cached = await self.client.get(key)
        if cached:
            return json.loads(cached)
        return None

    async def set_query_response(
        self,
        org_id: str,
        query: str,
        response: dict[str, Any],
        document_ids: Optional[list[str]] = None,
        ttl: Optional[int] = None,
    ) -> None:
        """Cache query response."""
        cache_data = f"{org_id}:{query}:{sorted(document_ids) if document_ids else ''}"
        key = self._generate_key("query", cache_data)

        await self.client.setex(
            key,
            ttl or settings.query_cache_ttl,
            json.dumps(response),
        )

    # Embedding caching
    async def get_embedding(self, text: str) -> Optional[list[float]]:
        """Get cached embedding for text."""
        key = self._generate_key("emb", text)
        cached = await self.client.get(key)
        if cached:
            return json.loads(cached)
        return None

    async def set_embedding(
        self, text: str, embedding: list[float], ttl: Optional[int] = None
    ) -> None:
        """Cache embedding for text."""
        key = self._generate_key("emb", text)
        await self.client.setex(
            key,
            ttl or settings.embedding_cache_ttl,
            json.dumps(embedding),
        )

    # Batch embedding caching
    async def get_embeddings_batch(
        self, texts: list[str]
    ) -> tuple[dict[str, list[float]], list[str]]:
        """Get cached embeddings for multiple texts.

        Returns:
            Tuple of (cached_embeddings dict, uncached_texts list)
        """
        cached = {}
        uncached = []

        pipe = self.client.pipeline()
        keys = [self._generate_key("emb", text) for text in texts]

        for key in keys:
            pipe.get(key)

        results = await pipe.execute()

        for text, result in zip(texts, results):
            if result:
                cached[text] = json.loads(result)
            else:
                uncached.append(text)

        return cached, uncached

    async def set_embeddings_batch(
        self,
        embeddings: dict[str, list[float]],
        ttl: Optional[int] = None,
    ) -> None:
        """Cache multiple embeddings."""
        pipe = self.client.pipeline()
        ttl = ttl or settings.embedding_cache_ttl

        for text, embedding in embeddings.items():
            key = self._generate_key("emb", text)
            pipe.setex(key, ttl, json.dumps(embedding))

        await pipe.execute()

    # Generic cache operations
    async def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        return await self.client.get(key)

    async def set(
        self, key: str, value: str, ttl: Optional[int] = None
    ) -> None:
        """Set value with optional TTL."""
        if ttl:
            await self.client.setex(key, ttl, value)
        else:
            await self.client.set(key, value)

    async def delete(self, key: str) -> None:
        """Delete key."""
        await self.client.delete(key)

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return await self.client.exists(key) > 0

    async def invalidate_org_cache(self, org_id: str) -> None:
        """Invalidate all cache entries for an organization."""
        pattern = f"query:{org_id}:*"
        async for key in self.client.scan_iter(match=pattern):
            await self.client.delete(key)

    async def health_check(self) -> bool:
        """Check Redis connection health."""
        try:
            await self.client.ping()
            return True
        except Exception:
            return False


# Global cache instance
cache_service = CacheService()
