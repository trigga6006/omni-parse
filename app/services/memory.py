"""Conversation memory service using Redis."""

import json
from datetime import datetime
from typing import Optional
from uuid import uuid4

import redis.asyncio as redis

from app.config import settings
from app.models.schemas import SessionMessage


class ConversationMemory:
    """Redis-backed conversation memory for sessions."""

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
            raise RuntimeError("Memory not connected. Call connect() first.")
        return self._client

    def _session_key(self, org_id: str, session_id: str) -> str:
        """Generate session key."""
        return f"session:{org_id}:{session_id}"

    def _session_meta_key(self, org_id: str, session_id: str) -> str:
        """Generate session metadata key."""
        return f"session_meta:{org_id}:{session_id}"

    def _org_sessions_key(self, org_id: str) -> str:
        """Generate org sessions index key."""
        return f"org_sessions:{org_id}"

    async def create_session(
        self, org_id: str, title: Optional[str] = None
    ) -> str:
        """Create a new conversation session."""
        session_id = str(uuid4())
        now = datetime.utcnow().isoformat()

        metadata = {
            "id": session_id,
            "organization_id": org_id,
            "title": title or f"Session {session_id[:8]}",
            "created_at": now,
            "last_activity": now,
            "message_count": 0,
        }

        # Store session metadata
        await self.client.setex(
            self._session_meta_key(org_id, session_id),
            settings.session_ttl_seconds,
            json.dumps(metadata),
        )

        # Add to org sessions index
        await self.client.zadd(
            self._org_sessions_key(org_id),
            {session_id: datetime.utcnow().timestamp()},
        )

        return session_id

    async def get_session(self, org_id: str, session_id: str) -> Optional[dict]:
        """Get session metadata."""
        meta = await self.client.get(self._session_meta_key(org_id, session_id))
        if meta:
            return json.loads(meta)
        return None

    async def add_message(
        self,
        org_id: str,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[list[dict]] = None,
    ) -> None:
        """Add a message to the conversation."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "sources": sources,
        }

        session_key = self._session_key(org_id, session_id)
        meta_key = self._session_meta_key(org_id, session_id)

        # Add message to list
        await self.client.rpush(session_key, json.dumps(message))

        # Trim to max turns (each turn = 2 messages)
        max_messages = settings.max_conversation_turns * 2
        await self.client.ltrim(session_key, -max_messages, -1)

        # Update session TTL
        await self.client.expire(session_key, settings.session_ttl_seconds)

        # Update metadata
        meta = await self.client.get(meta_key)
        if meta:
            metadata = json.loads(meta)
            metadata["last_activity"] = datetime.utcnow().isoformat()
            metadata["message_count"] = await self.client.llen(session_key)
            await self.client.setex(
                meta_key,
                settings.session_ttl_seconds,
                json.dumps(metadata),
            )

        # Update org sessions index score
        await self.client.zadd(
            self._org_sessions_key(org_id),
            {session_id: datetime.utcnow().timestamp()},
        )

    async def get_messages(
        self, org_id: str, session_id: str, limit: Optional[int] = None
    ) -> list[SessionMessage]:
        """Get conversation messages."""
        session_key = self._session_key(org_id, session_id)

        if limit:
            messages = await self.client.lrange(session_key, -limit, -1)
        else:
            messages = await self.client.lrange(session_key, 0, -1)

        return [
            SessionMessage(**json.loads(msg))
            for msg in messages
        ]

    async def get_context_messages(
        self, org_id: str, session_id: str, limit: int = 10
    ) -> list[dict]:
        """Get recent messages formatted for LLM context."""
        messages = await self.get_messages(org_id, session_id, limit)
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

    async def list_sessions(
        self, org_id: str, limit: int = 20, offset: int = 0
    ) -> tuple[list[dict], int]:
        """List sessions for an organization."""
        org_sessions_key = self._org_sessions_key(org_id)

        # Get total count
        total = await self.client.zcard(org_sessions_key)

        # Get session IDs sorted by recent activity (descending)
        session_ids = await self.client.zrevrange(
            org_sessions_key, offset, offset + limit - 1
        )

        sessions = []
        for session_id in session_ids:
            meta = await self.get_session(org_id, session_id)
            if meta:
                sessions.append(meta)

        return sessions, total

    async def delete_session(self, org_id: str, session_id: str) -> bool:
        """Delete a session."""
        session_key = self._session_key(org_id, session_id)
        meta_key = self._session_meta_key(org_id, session_id)
        org_sessions_key = self._org_sessions_key(org_id)

        pipe = self.client.pipeline()
        pipe.delete(session_key)
        pipe.delete(meta_key)
        pipe.zrem(org_sessions_key, session_id)
        results = await pipe.execute()

        return any(results)

    async def update_session_title(
        self, org_id: str, session_id: str, title: str
    ) -> bool:
        """Update session title."""
        meta_key = self._session_meta_key(org_id, session_id)
        meta = await self.client.get(meta_key)

        if not meta:
            return False

        metadata = json.loads(meta)
        metadata["title"] = title
        await self.client.setex(
            meta_key,
            settings.session_ttl_seconds,
            json.dumps(metadata),
        )
        return True

    async def session_exists(self, org_id: str, session_id: str) -> bool:
        """Check if session exists."""
        meta_key = self._session_meta_key(org_id, session_id)
        return await self.client.exists(meta_key) > 0

    async def cleanup_expired_sessions(self, org_id: str) -> int:
        """Remove expired sessions from org index."""
        org_sessions_key = self._org_sessions_key(org_id)
        session_ids = await self.client.zrange(org_sessions_key, 0, -1)

        removed = 0
        for session_id in session_ids:
            if not await self.session_exists(org_id, session_id):
                await self.client.zrem(org_sessions_key, session_id)
                removed += 1

        return removed


# Global memory instance
memory_service = ConversationMemory()
