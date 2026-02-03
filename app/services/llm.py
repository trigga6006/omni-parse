"""Anthropic Claude LLM service for response generation."""

from typing import Optional

import anthropic

from app.config import settings
from app.models.schemas import ChunkWithScore


class LLMService:
    """Anthropic Claude LLM service."""

    def __init__(self):
        self._client: Optional[anthropic.AsyncAnthropic] = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        """Get or create Anthropic client."""
        if not self._client:
            self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        return self._client

    def _format_context(self, chunks: list[ChunkWithScore]) -> str:
        """Format chunks into context for the LLM."""
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            header = f"[Source {i}]"
            if chunk.section_header:
                header += f" - {chunk.section_header}"
            if chunk.page_number:
                header += f" (Page {chunk.page_number})"

            context_parts.append(f"{header}\n{chunk.content}")

        return "\n\n---\n\n".join(context_parts)

    def _build_system_prompt(self) -> str:
        """Build the system prompt for RAG responses."""
        return """You are a helpful technical documentation assistant. Your role is to answer questions based on the provided documentation context.

Guidelines:
1. Base your answers ONLY on the provided context. If the context doesn't contain enough information to answer the question, say so clearly.
2. Be precise and technical when appropriate, but explain complex concepts clearly.
3. When referencing specific information, mention which source it comes from (e.g., "According to Source 1...").
4. If multiple sources provide relevant information, synthesize them into a coherent answer.
5. For procedural questions, provide step-by-step instructions when the context supports it.
6. If asked about something not in the documentation, explicitly state that you cannot find that information in the provided documents.
7. Be concise but thorough - don't add unnecessary filler text.
8. If the question is ambiguous, ask for clarification rather than guessing."""

    async def generate_response(
        self,
        query: str,
        chunks: list[ChunkWithScore],
        conversation_history: Optional[list[dict]] = None,
    ) -> str:
        """Generate a response using Claude.

        Args:
            query: User's question
            chunks: Relevant document chunks
            conversation_history: Previous messages for context

        Returns:
            Generated response string
        """
        client = self._get_client()

        # Format context from chunks
        context = self._format_context(chunks)

        # Build messages
        messages = []

        # Add conversation history if present
        if conversation_history:
            for msg in conversation_history[-10:]:  # Last 10 messages
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        # Add current query with context
        user_message = f"""Based on the following documentation context, please answer the user's question.

DOCUMENTATION CONTEXT:
{context}

USER QUESTION:
{query}

Please provide a helpful, accurate answer based on the documentation above."""

        messages.append({"role": "user", "content": user_message})

        # Generate response
        response = await client.messages.create(
            model=settings.anthropic_model,
            max_tokens=settings.max_tokens,
            system=self._build_system_prompt(),
            messages=messages,
        )

        return response.content[0].text

    async def generate_response_streaming(
        self,
        query: str,
        chunks: list[ChunkWithScore],
        conversation_history: Optional[list[dict]] = None,
    ):
        """Generate a streaming response using Claude.

        Args:
            query: User's question
            chunks: Relevant document chunks
            conversation_history: Previous messages for context

        Yields:
            Response text chunks
        """
        client = self._get_client()

        # Format context from chunks
        context = self._format_context(chunks)

        # Build messages
        messages = []

        if conversation_history:
            for msg in conversation_history[-10:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        user_message = f"""Based on the following documentation context, please answer the user's question.

DOCUMENTATION CONTEXT:
{context}

USER QUESTION:
{query}

Please provide a helpful, accurate answer based on the documentation above."""

        messages.append({"role": "user", "content": user_message})

        # Stream response
        async with client.messages.stream(
            model=settings.anthropic_model,
            max_tokens=settings.max_tokens,
            system=self._build_system_prompt(),
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def summarize_for_title(self, query: str, response: str) -> str:
        """Generate a short title for a conversation session.

        Args:
            query: First user query
            response: First assistant response

        Returns:
            Short title string (max 50 chars)
        """
        client = self._get_client()

        messages = [
            {
                "role": "user",
                "content": f"""Generate a very short title (max 5 words) for a conversation that started with this exchange:

User: {query[:200]}
Assistant: {response[:300]}

Reply with ONLY the title, nothing else.""",
            }
        ]

        result = await client.messages.create(
            model=settings.anthropic_model,
            max_tokens=50,
            messages=messages,
        )

        title = result.content[0].text.strip()
        # Ensure it's not too long
        if len(title) > 50:
            title = title[:47] + "..."

        return title


# Global LLM service instance
llm_service = LLMService()
