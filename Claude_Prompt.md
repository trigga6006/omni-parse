# TechDocs AI - Enhanced Production RAG Backend

You are building a **production-grade** RAG backend for a B2B SaaS called **TechDocs AI**. This service allows small shops (mechanics, repair technicians, etc.) to upload technical/service manuals and query them via an AI assistant.

## Business Context

- **Pricing**: $20/month for 3-5 seats per organization
- **Users**: Small businesses with multiple technicians
- **Use case**: Fast, accurate answers from uploaded technical documentation
- **Critical**: Must handle part numbers, model codes, torque specs, and follow-up questions

-----

## ENHANCED TECH STACK

|Component |Technology |Purpose |
|--------------|------------------------------------------|----------------------------|
|Framework |FastAPI (async) |Speed |
|Database |Supabase (PostgreSQL + pgvector + pg_trgm)|Hybrid search |
|Cache/Memory |Redis (Upstash free tier) |Session memory + query cache|
|Auth |Clerk (with organizations) |Multi-tenant |
|LLM |Anthropic Claude Sonnet 4 |RAG responses |
|Embeddings |OpenAI text-embedding-3-small |Vector search |
|Reranking |Cohere Rerank |Accuracy boost |
|File Storage |Supabase Storage |PDFs |
|Payments |Stripe |Subscriptions |
|PDF Processing|PyMuPDF + pdfplumber |Structure-aware extraction |
|Deployment |Docker → Render |Hosting |

-----

## KEY ENHANCEMENTS OVER BASIC RAG

### 1. Hybrid Search (Semantic + Keyword)

- pgvector for semantic similarity
- pg_trgm for fuzzy keyword matching (part numbers!)
- Combined scoring: `0.7 * semantic + 0.3 * keyword`

### 2. Conversation Memory

- Redis-backed session storage
- Last 5 Q&A pairs per session
- Enables follow-up questions: “What about the torque specs?”

### 3. Cohere Reranking

- Retrieve top-20 chunks
- Rerank to top-5 with Cohere
- Dramatically improves precision

### 4. Smart Chunking

- Respects document structure (headers, sections)
- Keeps tables intact
- Parent-child relationships for context

### 5. Query Caching

- Cache embeddings for repeated queries
- Cache full responses for identical questions
- TTL-based expiration

-----

## PROJECT STRUCTURE

```
techdocs-backend/
├── app/
│ ├── __init__.py
│ ├── main.py
│ ├── config.py
│ ├── dependencies.py
│ │
│ ├── api/
│ │ ├── __init__.py
│ │ ├── routes/
│ │ │ ├── __init__.py
│ │ │ ├── health.py
│ │ │ ├── auth.py
│ │ │ ├── documents.py
│ │ │ ├── query.py
│ │ │ ├── sessions.py # NEW: Session management
│ │ │ ├── organizations.py
│ │ │ └── billing.py
│ │ └── middleware/
│ │ ├── __init__.py
│ │ └── clerk_auth.py
│ │
│ ├── services/
│ │ ├── __init__.py
│ │ ├── embedding.py
│ │ ├── llm.py
│ │ ├── rag.py # ENHANCED: Full pipeline
│ │ ├── reranker.py # NEW: Cohere reranking
│ │ ├── document_processor.py # ENHANCED: Smart chunking
│ │ ├── vector_store.py # ENHANCED: Hybrid search
│ │ ├── memory.py # NEW: Conversation memory
│ │ ├── cache.py # NEW: Query caching
│ │ └── storage.py
│ │
│ ├── models/
│ │ ├── __init__.py
│ │ ├── schemas.py
│ │ └── database.py
│ │
│ └── utils/
│ ├── __init__.py
│ └── helpers.py
│
├── migrations/
│ └── 001_initial_schema.sql # ENHANCED: Hybrid search indexes
│
├── tests/
│ └── test_api.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

-----

## COMPLETE FILE IMPLEMENTATIONS

### 1. `app/config.py`

```python
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
# App
app_name: str = "TechDocs AI"
debug: bool = False

# Supabase
supabase_url: str
supabase_anon_key: str
supabase_service_role_key: str
database_url: str

# Redis (Upstash)
redis_url: str # redis://default:xxx@xxx.upstash.io:6379

# Clerk
clerk_secret_key: str
clerk_webhook_secret: str
clerk_publishable_key: str

# OpenAI (embeddings)
openai_api_key: str
embedding_model: str = "text-embedding-3-small"
embedding_dimensions: int = 1536

# Anthropic (LLM)
anthropic_api_key: str
claude_model: str = "claude-sonnet-4-20250514"

# Cohere (reranking)
cohere_api_key: str
rerank_model: str = "rerank-english-v3.0"

# Stripe
stripe_secret_key: str
stripe_webhook_secret: str
stripe_price_id: str

# RAG Settings
chunk_size: int = 400 # Smaller for precision
chunk_overlap: int = 100 # More overlap for context
retrieval_top_k: int = 20 # Retrieve more, rerank down
rerank_top_k: int = 5 # Final chunks after reranking
max_file_size_mb: int = 50

# Memory Settings
memory_ttl_seconds: int = 3600 # 1 hour session memory
memory_max_turns: int = 5 # Last 5 Q&A pairs
cache_ttl_seconds: int = 86400 # 24 hour query cache

# Hybrid Search Weights
semantic_weight: float = 0.7
keyword_weight: float = 0.3

class Config:
env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
return Settings()
```

### 2. `app/main.py`

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import redis.asyncio as redis

from app.config import get_settings
from app.api.routes import health, auth, documents, query, sessions, organizations, billing
from app.services.vector_store import VectorStore
from app.services.cache import CacheService
from app.services.memory import MemoryService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
# Startup
logger.info("Starting TechDocs AI Backend (Enhanced)...")

# Initialize vector store
vector_store = VectorStore()
await vector_store.initialize()
app.state.vector_store = vector_store

# Initialize Redis connection pool
redis_client = redis.from_url(
settings.redis_url,
encoding="utf-8",
decode_responses=True
)
app.state.redis = redis_client

# Initialize services
app.state.cache = CacheService(redis_client)
app.state.memory = MemoryService(redis_client)

logger.info("All services initialized")
yield

# Shutdown
await redis_client.close()
logger.info("Shutdown complete")

app = FastAPI(
title=settings.app_name,
version="2.0.0",
description="Production-grade RAG for technical documentation",
lifespan=lifespan
)

# CORS
app.add_middleware(
CORSMiddleware,
allow_origins=[
"http://localhost:3000",
"http://localhost:5173",
"https://*.netlify.app",
],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)

# Routes
app.include_router(health.router, tags=["Health"])
app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(query.router, prefix="/api/query", tags=["Query"])
app.include_router(sessions.router, prefix="/api/sessions", tags=["Sessions"])
app.include_router(organizations.router, prefix="/api/organizations", tags=["Organizations"])
app.include_router(billing.router, prefix="/api/billing", tags=["Billing"])
```

### 3. `app/services/cache.py` (NEW)

```python
import redis.asyncio as redis
import json
import hashlib
from typing import Optional, List, Any
import logging

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

class CacheService:
def __init__(self, redis_client: redis.Redis):
self.redis = redis_client
self.embedding_prefix = "emb:"
self.response_prefix = "resp:"

def _hash_key(self, text: str) -> str:
"""Create consistent hash for cache key"""
return hashlib.sha256(text.encode()).hexdigest()[:16]

async def get_embedding(self, text: str) -> Optional[List[float]]:
"""Get cached embedding"""
key = f"{self.embedding_prefix}{self._hash_key(text)}"
try:
cached = await self.redis.get(key)
if cached:
logger.debug(f"Embedding cache hit: {key}")
return json.loads(cached)
except Exception as e:
logger.warning(f"Cache get error: {e}")
return None

async def set_embedding(self, text: str, embedding: List[float]):
"""Cache embedding"""
key = f"{self.embedding_prefix}{self._hash_key(text)}"
try:
await self.redis.setex(
key,
settings.cache_ttl_seconds,
json.dumps(embedding)
)
except Exception as e:
logger.warning(f"Cache set error: {e}")

async def get_response(self, org_id: str, question: str, doc_ids: Optional[List[str]] = None) -> Optional[dict]:
"""Get cached RAG response"""
cache_input = f"{org_id}:{question}:{sorted(doc_ids) if doc_ids else ''}"
key = f"{self.response_prefix}{self._hash_key(cache_input)}"
try:
cached = await self.redis.get(key)
if cached:
logger.info(f"Response cache hit for: {question[:50]}...")
return json.loads(cached)
except Exception as e:
logger.warning(f"Cache get error: {e}")
return None

async def set_response(
self,
org_id: str,
question: str,
response: dict,
doc_ids: Optional[List[str]] = None,
ttl: int = None
):
"""Cache RAG response"""
cache_input = f"{org_id}:{question}:{sorted(doc_ids) if doc_ids else ''}"
key = f"{self.response_prefix}{self._hash_key(cache_input)}"
try:
await self.redis.setex(
key,
ttl or settings.cache_ttl_seconds,
json.dumps(response)
)
except Exception as e:
logger.warning(f"Cache set error: {e}")

async def invalidate_org_cache(self, org_id: str):
"""Invalidate all cached responses for an org (after document changes)"""
try:
pattern = f"{self.response_prefix}*"
# Note: In production, use SCAN instead of KEYS
keys = await self.redis.keys(pattern)
if keys:
await self.redis.delete(*keys)
logger.info(f"Invalidated {len(keys)} cache entries for org {org_id}")
except Exception as e:
logger.warning(f"Cache invalidation error: {e}")
```

### 4. `app/services/memory.py` (NEW)

```python
import redis.asyncio as redis
import json
from typing import Optional, List
from datetime import datetime
import logging
from pydantic import BaseModel

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

class ConversationTurn(BaseModel):
question: str
answer: str
timestamp: str
sources_summary: Optional[str] = None

class ConversationMemory(BaseModel):
session_id: str
org_id: str
user_id: str
turns: List[ConversationTurn]
created_at: str
updated_at: str

class MemoryService:
def __init__(self, redis_client: redis.Redis):
self.redis = redis_client
self.prefix = "memory:"

def _session_key(self, session_id: str) -> str:
return f"{self.prefix}{session_id}"

async def get_memory(self, session_id: str) -> Optional[ConversationMemory]:
"""Get conversation memory for a session"""
try:
data = await self.redis.get(self._session_key(session_id))
if data:
return ConversationMemory(**json.loads(data))
except Exception as e:
logger.warning(f"Memory get error: {e}")
return None

async def create_session(self, session_id: str, org_id: str, user_id: str) -> ConversationMemory:
"""Create new conversation session"""
now = datetime.utcnow().isoformat()
memory = ConversationMemory(
session_id=session_id,
org_id=org_id,
user_id=user_id,
turns=[],
created_at=now,
updated_at=now
)
await self._save_memory(memory)
return memory

async def add_turn(
self,
session_id: str,
question: str,
answer: str,
sources_summary: Optional[str] = None
) -> Optional[ConversationMemory]:
"""Add a Q&A turn to conversation memory"""
memory = await self.get_memory(session_id)
if not memory:
logger.warning(f"Session not found: {session_id}")
return None

turn = ConversationTurn(
question=question,
answer=answer[:500], # Truncate for storage efficiency
timestamp=datetime.utcnow().isoformat(),
sources_summary=sources_summary
)

memory.turns.append(turn)

# Keep only last N turns
if len(memory.turns) > settings.memory_max_turns:
memory.turns = memory.turns[-settings.memory_max_turns:]

memory.updated_at = datetime.utcnow().isoformat()
await self._save_memory(memory)

return memory

async def _save_memory(self, memory: ConversationMemory):
"""Save memory to Redis with TTL"""
try:
await self.redis.setex(
self._session_key(memory.session_id),
settings.memory_ttl_seconds,
memory.model_dump_json()
)
except Exception as e:
logger.warning(f"Memory save error: {e}")

async def delete_session(self, session_id: str):
"""Delete a conversation session"""
try:
await self.redis.delete(self._session_key(session_id))
except Exception as e:
logger.warning(f"Memory delete error: {e}")

def format_memory_for_prompt(self, memory: ConversationMemory) -> str:
"""Format conversation history for LLM context"""
if not memory or not memory.turns:
return ""

formatted = "Previous conversation:\n"
for turn in memory.turns[-3:]: # Last 3 for prompt
formatted += f"User: {turn.question}\n"
formatted += f"Assistant: {turn.answer}\n\n"

return formatted.strip()
```

### 5. `app/services/reranker.py` (NEW)

```python
import cohere
from typing import List
import logging

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

class RerankerService:
def __init__(self):
self.client = cohere.Client(settings.cohere_api_key)
self.model = settings.rerank_model

async def rerank(
self,
query: str,
chunks: List[dict],
top_k: int = 5
) -> List[dict]:
"""
Rerank chunks using Cohere Rerank API

Args:
query: The user's question
chunks: List of chunk dicts with 'content' field
top_k: Number of top results to return

Returns:
Reranked and filtered chunks
"""
if not chunks:
return []

if len(chunks) <= top_k:
return chunks

try:
# Extract content for reranking
documents = [c["content"] for c in chunks]

# Call Cohere Rerank
response = self.client.rerank(
model=self.model,
query=query,
documents=documents,
top_n=top_k,
return_documents=False
)

# Reorder chunks based on rerank results
reranked = []
for result in response.results:
chunk = chunks[result.index].copy()
chunk["rerank_score"] = result.relevance_score
reranked.append(chunk)

logger.info(f"Reranked {len(chunks)} chunks to top {len(reranked)}")
return reranked

except Exception as e:
logger.error(f"Reranking failed: {e}, falling back to original order")
return chunks[:top_k]
```

### 6. `app/services/document_processor.py` (ENHANCED)

```python
import fitz # PyMuPDF
import pdfplumber
from typing import List, Tuple, Optional
import logging
import re

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

class DocumentProcessor:
def __init__(self):
self.chunk_size = settings.chunk_size
self.chunk_overlap = settings.chunk_overlap

def extract_with_structure(self, pdf_bytes: bytes) -> List[dict]:
"""
Extract text preserving document structure.
Returns list of sections with metadata.
"""
sections = []
current_section = {"title": "Introduction", "content": "", "page": 1}

with pdfplumber.open(pdf_bytes) as pdf:
for page_num, page in enumerate(pdf.pages, start=1):
# Extract text
text = page.extract_text() or ""

# Extract tables separately
tables = page.extract_tables()

# Detect headers (simple heuristic: short lines in caps or with numbers)
lines = text.split('\n')
for line in lines:
stripped = line.strip()

# Check if line looks like a header
if self._is_header(stripped):
# Save current section if has content
if current_section["content"].strip():
sections.append(current_section.copy())
# Start new section
current_section = {
"title": stripped,
"content": "",
"page": page_num
}
else:
current_section["content"] += line + "\n"

# Add tables as separate chunks
for table in tables:
if table:
table_text = self._format_table(table)
sections.append({
"title": f"Table (Page {page_num})",
"content": table_text,
"page": page_num,
"is_table": True
})

# Don't forget last section
if current_section["content"].strip():
sections.append(current_section)

return sections

def _is_header(self, line: str) -> bool:
"""Detect if a line is likely a section header"""
if not line or len(line) > 100:
return False

# Numbered sections: "1.2 Installation" or "Chapter 3"
if re.match(r'^(\d+\.?)+\s+\w+', line):
return True
if re.match(r'^(Chapter|Section|Part)\s+\d+', line, re.IGNORECASE):
return True

# ALL CAPS headers
if line.isupper() and len(line.split()) <= 6:
return True

return False

def _format_table(self, table: List[List]) -> str:
"""Format extracted table as readable text"""
if not table:
return ""

lines = []
for row in table:
# Filter None values and join
cells = [str(cell) if cell else "" for cell in row]
lines.append(" | ".join(cells))

return "\n".join(lines)

def chunk_sections(self, sections: List[dict]) -> List[dict]:
"""
Chunk sections intelligently:
- Keep small sections intact
- Split large sections with overlap
- Preserve tables as single chunks
"""
chunks = []

for section in sections:
content = section["content"].strip()

if not content:
continue

# Tables stay intact
if section.get("is_table"):
chunks.append({
"content": f"[Table: {section['title']}]\n{content}",
"page_number": section["page"],
"chunk_index": len(chunks),
"section_title": section["title"],
"is_table": True
})
continue

# Small sections stay intact
words = content.split()
if len(words) <= self.chunk_size:
chunks.append({
"content": f"[{section['title']}]\n{content}",
"page_number": section["page"],
"chunk_index": len(chunks),
"section_title": section["title"]
})
continue

# Split large sections
section_chunks = self._split_with_overlap(
content,
section["title"],
section["page"]
)
for i, sc in enumerate(section_chunks):
sc["chunk_index"] = len(chunks)
chunks.append(sc)

return chunks

def _split_with_overlap(
self,
text: str,
section_title: str,
page_number: int
) -> List[dict]:
"""Split text into overlapping chunks"""
words = text.split()
chunks = []
start = 0

while start < len(words):
end = start + self.chunk_size
chunk_words = words[start:end]
chunk_text = " ".join(chunk_words)

chunks.append({
"content": f"[{section_title}]\n{chunk_text}",
"page_number": page_number,
"section_title": section_title
})

# Move forward with overlap
start = end - self.chunk_overlap
if start >= len(words):
break

return chunks

def process_document(self, pdf_bytes: bytes) -> Tuple[List[dict], int]:
"""
Full document processing pipeline.
Returns: (chunks, page_count)
"""
try:
# Get page count
with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
page_count = len(doc)

# Extract with structure
sections = self.extract_with_structure(pdf_bytes)

# Chunk intelligently
chunks = self.chunk_sections(sections)

logger.info(
f"Processed document: {page_count} pages, "
f"{len(sections)} sections, {len(chunks)} chunks"
)

return chunks, page_count

except Exception as e:
logger.error(f"Document processing error: {e}")
# Fallback to basic extraction
return self._fallback_process(pdf_bytes)

def _fallback_process(self, pdf_bytes: bytes) -> Tuple[List[dict], int]:
"""Basic fallback processing if structure extraction fails"""
chunks = []

with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
page_count = len(doc)

for page_num, page in enumerate(doc, start=1):
text = page.get_text("text").strip()
if not text:
continue

words = text.split()
start = 0
chunk_idx = 0

while start < len(words):
end = start + self.chunk_size
chunk_text = " ".join(words[start:end])

chunks.append({
"content": chunk_text,
"page_number": page_num,
"chunk_index": len(chunks),
"section_title": f"Page {page_num}"
})

start = end - self.chunk_overlap
chunk_idx += 1

return chunks, page_count
```

### 7. `app/services/embedding.py` (ENHANCED with caching)

```python
from openai import AsyncOpenAI
from typing import List, Optional
import logging

from app.config import get_settings
from app.services.cache import CacheService

settings = get_settings()
logger = logging.getLogger(__name__)

class EmbeddingService:
def __init__(self, cache: Optional[CacheService] = None):
self.client = AsyncOpenAI(api_key=settings.openai_api_key)
self.model = settings.embedding_model
self.cache = cache

async def embed_text(self, text: str, use_cache: bool = True) -> List[float]:
"""Generate embedding for a single text with caching"""
# Check cache first
if use_cache and self.cache:
cached = await self.cache.get_embedding(text)
if cached:
return cached

response = await self.client.embeddings.create(
model=self.model,
input=text
)
embedding = response.data[0].embedding

# Cache the result
if use_cache and self.cache:
await self.cache.set_embedding(text, embedding)

return embedding

async def embed_batch(
self,
texts: List[str],
batch_size: int = 100,
use_cache: bool = False # Disable cache for batch processing
) -> List[List[float]]:
"""Generate embeddings for multiple texts with batching"""
all_embeddings = []

for i in range(0, len(texts), batch_size):
batch = texts[i:i + batch_size]
response = await self.client.embeddings.create(
model=self.model,
input=batch
)
all_embeddings.extend([d.embedding for d in response.data])
logger.info(f"Embedded batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

return all_embeddings
```

### 8. `app/services/vector_store.py` (ENHANCED with hybrid search)

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from typing import List, Optional
import logging
import uuid

from app.config import get_settings
from app.models.database import Base

settings = get_settings()
logger = logging.getLogger(__name__)

class VectorStore:
def __init__(self):
db_url = settings.database_url
if db_url.startswith("postgres://"):
db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
elif db_url.startswith("postgresql://"):
db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

self.engine = create_async_engine(db_url, echo=settings.debug)
self.async_session = sessionmaker(
self.engine, class_=AsyncSession, expire_on_commit=False
)

async def initialize(self):
"""Create tables and enable extensions"""
async with self.engine.begin() as conn:
await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
await conn.run_sync(Base.metadata.create_all)
logger.info("Database initialized with pgvector and pg_trgm")

async def get_session(self) -> AsyncSession:
return self.async_session()

async def store_chunks(
self,
document_id: str,
org_id: str,
chunks: List[dict],
embeddings: List[List[float]]
):
"""Store document chunks with embeddings"""
async with self.async_session() as session:
for chunk, embedding in zip(chunks, embeddings):
# Also store searchable text for keyword matching
searchable_text = chunk["content"].lower()

await session.execute(
text("""
INSERT INTO document_chunks
(id, document_id, org_id, content, searchable_text,
page_number, chunk_index, section_title, embedding)
VALUES (:id, :doc_id, :org_id, :content, :searchable,
:page, :idx, :section, :embedding::vector)
"""),
{
"id": str(uuid.uuid4()),
"doc_id": document_id,
"org_id": org_id,
"content": chunk["content"],
"searchable": searchable_text,
"page": chunk["page_number"],
"idx": chunk["chunk_index"],
"section": chunk.get("section_title", ""),
"embedding": f"[{','.join(map(str, embedding))}]"
}
)

await session.commit()

logger.info(f"Stored {len(chunks)} chunks for document {document_id}")

async def hybrid_search(
self,
org_id: str,
query_embedding: List[float],
query_text: str,
top_k: int = 20,
document_ids: Optional[List[str]] = None,
semantic_weight: float = None,
keyword_weight: float = None
) -> List[dict]:
"""
Hybrid search combining semantic similarity and keyword matching.

Uses:
- pgvector cosine similarity for semantic search
- pg_trgm similarity for fuzzy keyword matching
"""
semantic_w = semantic_weight or settings.semantic_weight
keyword_w = keyword_weight or settings.keyword_weight

async with self.async_session() as session:
embedding_str = f"[{','.join(map(str, query_embedding))}]"
search_text = query_text.lower()

# Build document filter
doc_filter = ""
params = {
"org_id": org_id,
"embedding": embedding_str,
"search_text": search_text,
"top_k": top_k,
"sem_weight": semantic_w,
"kw_weight": keyword_w
}

if document_ids:
doc_filter = "AND dc.document_id = ANY(:doc_ids)"
params["doc_ids"] = document_ids

query = text(f"""
WITH semantic_scores AS (
SELECT
dc.id,
1 - (dc.embedding <=> :embedding::vector) as semantic_score
FROM document_chunks dc
WHERE dc.org_id = :org_id {doc_filter}
),
keyword_scores AS (
SELECT
dc.id,
similarity(dc.searchable_text, :search_text) as keyword_score
FROM document_chunks dc
WHERE dc.org_id = :org_id {doc_filter}
)
SELECT
dc.id,
dc.document_id,
dc.content,
dc.page_number,
dc.section_title,
d.filename as document_name,
ss.semantic_score,
ks.keyword_score,
(ss.semantic_score * :sem_weight + COALESCE(ks.keyword_score, 0) * :kw_weight) as combined_score
FROM document_chunks dc
JOIN documents d ON dc.document_id = d.id
JOIN semantic_scores ss ON dc.id = ss.id
LEFT JOIN keyword_scores ks ON dc.id = ks.id
WHERE dc.org_id = :org_id {doc_filter}
ORDER BY combined_score DESC
LIMIT :top_k
""")

result = await session.execute(query, params)
rows = result.fetchall()

return [
{
"id": str(row.id),
"document_id": str(row.document_id),
"content": row.content,
"page_number": row.page_number,
"section_title": row.section_title,
"document_name": row.document_name,
"semantic_score": float(row.semantic_score),
"keyword_score": float(row.keyword_score) if row.keyword_score else 0,
"relevance_score": float(row.combined_score)
}
for row in rows
]

async def delete_document_chunks(self, document_id: str):
"""Delete all chunks for a document"""
async with self.async_session() as session:
await session.execute(
text("DELETE FROM document_chunks WHERE document_id = :doc_id"),
{"doc_id": document_id}
)
await session.commit()
```

### 9. `app/services/llm.py` (ENHANCED with memory)

```python
import anthropic
from typing import AsyncGenerator, List, Optional
import logging

from app.config import get_settings
from app.services.memory import ConversationMemory, MemoryService

settings = get_settings()
logger = logging.getLogger(__name__)

class LLMService:
def __init__(self):
self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
self.model = settings.claude_model

def _build_rag_prompt(
self,
question: str,
context_chunks: List[dict],
conversation_history: Optional[str] = None
) -> str:
"""Build RAG prompt with optional conversation history"""

context_text = "\n\n---\n\n".join([
f"[Source: {c['document_name']}, Page {c['page_number']}, Section: {c.get('section_title', 'N/A')}]\n{c['content']}"
for c in context_chunks
])

history_section = ""
if conversation_history:
history_section = f"""
<conversation_history>
{conversation_history}
</conversation_history>

Use the conversation history to understand context for follow-up questions.
"""

return f"""You are a helpful technical assistant for service technicians. Answer the question based ONLY on the provided documentation context.

{history_section}
<documentation>
{context_text}
</documentation>

<guidelines>
- Be concise, accurate, and practical
- Include specific steps, measurements, part numbers, and torque specs when available
- If the documentation shows a table, format your response clearly
- If information is not in the documentation, say so clearly
- For follow-up questions, refer to the conversation history for context
</guidelines>

<question>
{question}
</question>

Provide a clear, actionable answer based on the documentation."""

async def generate_answer(
self,
question: str,
context_chunks: List[dict],
conversation_history: Optional[str] = None
) -> str:
"""Generate a complete answer"""
prompt = self._build_rag_prompt(question, context_chunks, conversation_history)

response = await self.client.messages.create(
model=self.model,
max_tokens=1024,
messages=[{"role": "user", "content": prompt}]
)

return response.content[0].text

async def generate_answer_stream(
self,
question: str,
context_chunks: List[dict],
conversation_history: Optional[str] = None
) -> AsyncGenerator[str, None]:
"""Generate answer with streaming"""
prompt = self._build_rag_prompt(question, context_chunks, conversation_history)

async with self.client.messages.stream(
model=self.model,
max_tokens=1024,
messages=[{"role": "user", "content": prompt}]
) as stream:
async for text in stream.text_stream:
yield text
```

### 10. `app/services/rag.py` (ENHANCED - Full Pipeline)

```python
from typing import List, Optional, AsyncGenerator
import time
import logging

from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.services.vector_store import VectorStore
from app.services.reranker import RerankerService
from app.services.cache import CacheService
from app.services.memory import MemoryService, ConversationMemory
from app.models.schemas import SourceChunk, QueryResponse
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

class RAGService:
def __init__(
self,
vector_store: VectorStore,
cache: CacheService,
memory: MemoryService
):
self.vector_store = vector_store
self.cache = cache
self.memory = memory
self.embedding_service = EmbeddingService(cache)
self.llm_service = LLMService()
self.reranker = RerankerService()

async def query(
self,
org_id: str,
question: str,
session_id: Optional[str] = None,
top_k: int = None,
document_ids: Optional[List[str]] = None,
use_cache: bool = True
) -> QueryResponse:
"""
Full RAG pipeline:
1. Check cache
2. Get conversation history
3. Embed question
4. Hybrid search (semantic + keyword)
5. Rerank results
6. Generate answer with context
7. Save to memory
8. Cache response
"""
start_time = time.time()
retrieval_k = settings.retrieval_top_k
rerank_k = top_k or settings.rerank_top_k

# 1. Check cache
if use_cache:
cached = await self.cache.get_response(org_id, question, document_ids)
if cached:
cached["query_time_ms"] = int((time.time() - start_time) * 1000)
cached["from_cache"] = True
return QueryResponse(**cached)

# 2. Get conversation history
conversation_history = None
if session_id:
memory = await self.memory.get_memory(session_id)
if memory:
conversation_history = self.memory.format_memory_for_prompt(memory)

# 3. Embed question (with cache)
query_embedding = await self.embedding_service.embed_text(question)

# 4. Hybrid search
chunks = await self.vector_store.hybrid_search(
org_id=org_id,
query_embedding=query_embedding,
query_text=question,
top_k=retrieval_k,
document_ids=document_ids
)

if not chunks:
return QueryResponse(
answer="I couldn't find any relevant information in your documents to answer this question.",
sources=[],
query_time_ms=int((time.time() - start_time) * 1000)
)

# 5. Rerank
reranked_chunks = await self.reranker.rerank(
query=question,
chunks=chunks,
top_k=rerank_k
)

# 6. Generate answer
answer = await self.llm_service.generate_answer(
question=question,
context_chunks=reranked_chunks,
conversation_history=conversation_history
)

# 7. Format sources
sources = [
SourceChunk(
document_id=c["document_id"],
document_name=c["document_name"],
page_number=c["page_number"],
content=c["content"][:500],
relevance_score=c.get("rerank_score", c["relevance_score"])
)
for c in reranked_chunks
]

query_time_ms = int((time.time() - start_time) * 1000)

response = QueryResponse(
answer=answer,
sources=sources,
query_time_ms=query_time_ms
)

# 8. Save to memory
if session_id:
sources_summary = ", ".join([f"{s.document_name} p.{s.page_number}" for s in sources[:3]])
await self.memory.add_turn(
session_id=session_id,
question=question,
answer=answer,
sources_summary=sources_summary
)

# 9. Cache response
if use_cache:
await self.cache.set_response(
org_id=org_id,
question=question,
response=response.model_dump(),
doc_ids=document_ids
)

logger.info(f"RAG query completed in {query_time_ms}ms (reranked {len(chunks)} → {len(reranked_chunks)})")

return response

async def query_stream(
self,
org_id: str,
question: str,
session_id: Optional[str] = None,
top_k: int = None,
document_ids: Optional[List[str]] = None
) -> AsyncGenerator[dict, None]:
"""RAG query with streaming response"""
retrieval_k = settings.retrieval_top_k
rerank_k = top_k or settings.rerank_top_k

# Get conversation history
conversation_history = None
if session_id:
memory = await self.memory.get_memory(session_id)
if memory:
conversation_history = self.memory.format_memory_for_prompt(memory)

# Embed question
query_embedding = await self.embedding_service.embed_text(question)

# Hybrid search
chunks = await self.vector_store.hybrid_search(
org_id=org_id,
query_embedding=query_embedding,
query_text=question,
top_k=retrieval_k,
document_ids=document_ids
)

if not chunks:
yield {"type": "chunk", "content": "I couldn't find any relevant information in your documents."}
yield {"type": "done"}
return

# Rerank
reranked_chunks = await self.reranker.rerank(
query=question,
chunks=chunks,
top_k=rerank_k
)

# Send sources first
sources = [
{
"document_id": c["document_id"],
"document_name": c["document_name"],
"page_number": c["page_number"],
"relevance_score": c.get("rerank_score", c["relevance_score"])
}
for c in reranked_chunks
]
yield {"type": "sources", "sources": sources}

# Stream answer
full_answer = ""
async for text_chunk in self.llm_service.generate_answer_stream(
question=question,
context_chunks=reranked_chunks,
conversation_history=conversation_history
):
full_answer += text_chunk
yield {"type": "chunk", "content": text_chunk}

# Save to memory after streaming completes
if session_id:
sources_summary = ", ".join([f"{s['document_name']} p.{s['page_number']}" for s in sources[:3]])
await self.memory.add_turn(
session_id=session_id,
question=question,
answer=full_answer,
sources_summary=sources_summary
)

yield {"type": "done"}
```

### 11. `app/models/schemas.py` (ENHANCED)

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

# === Auth ===
class ClerkWebhookEvent(BaseModel):
type: str
data: dict

# === Sessions ===
class CreateSessionRequest(BaseModel):
"""Create a new conversation session"""
pass # org_id and user_id come from auth

class SessionResponse(BaseModel):
session_id: str
created_at: str
message: str

class SessionHistoryResponse(BaseModel):
session_id: str
turns: List[dict]
created_at: str
updated_at: str

# === Documents ===
class DocumentStatus(str, Enum):
PENDING = "pending"
PROCESSING = "processing"
READY = "ready"
FAILED = "failed"

class DocumentUploadResponse(BaseModel):
document_id: str
filename: str
status: DocumentStatus
message: str

class DocumentMetadata(BaseModel):
id: str
org_id: str
filename: str
file_size: int
page_count: int
chunk_count: int
status: DocumentStatus
created_at: datetime
updated_at: datetime

class DocumentListResponse(BaseModel):
documents: List[DocumentMetadata]
total: int

# === Query ===
class QueryRequest(BaseModel):
question: str = Field(..., min_length=3, max_length=1000)
session_id: Optional[str] = None # For conversation memory
document_ids: Optional[List[str]] = None
top_k: int = Field(default=5, ge=1, le=10)
use_cache: bool = True

class SourceChunk(BaseModel):
document_id: str
document_name: str
page_number: int
content: str
relevance_score: float

class QueryResponse(BaseModel):
answer: str
sources: List[SourceChunk]
query_time_ms: int
from_cache: bool = False

class StreamingQueryResponse(BaseModel):
type: str # "chunk", "sources", "done", "error"
content: Optional[str] = None
sources: Optional[List[dict]] = None

# === Organizations ===
class OrgStats(BaseModel):
document_count: int
total_chunks: int
total_queries_this_month: int
seat_count: int
seat_limit: int

class UsageRecord(BaseModel):
date: str
query_count: int

# === Billing ===
class SubscriptionStatus(BaseModel):
status: str
current_period_end: Optional[datetime]
seat_limit: int
price_per_month: int

class CreateCheckoutRequest(BaseModel):
success_url: str
cancel_url: str

class CheckoutResponse(BaseModel):
checkout_url: str
```

### 12. `app/models/database.py` (ENHANCED)

```python
from sqlalchemy import Column, String, Integer, DateTime, Text, Float, ForeignKey, Boolean, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
import uuid

from app.config import get_settings

settings = get_settings()
Base = declarative_base()

class Organization(Base):
__tablename__ = "organizations"

id = Column(String, primary_key=True)
name = Column(String(255), nullable=False)
stripe_customer_id = Column(String(255), unique=True)
stripe_subscription_id = Column(String(255))
subscription_status = Column(String(50), default="inactive")
seat_limit = Column(Integer, default=5)
created_at = Column(DateTime(timezone=True), server_default=func.now())
updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class User(Base):
__tablename__ = "users"

id = Column(String, primary_key=True)
org_id = Column(String, ForeignKey("organizations.id"))
email = Column(String(255), nullable=False)
role = Column(String(50), default="member")
created_at = Column(DateTime(timezone=True), server_default=func.now())

class Document(Base):
__tablename__ = "documents"

id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
org_id = Column(String, ForeignKey("organizations.id"), nullable=False)
filename = Column(String(500), nullable=False)
storage_path = Column(String(1000), nullable=False)
file_size = Column(Integer)
page_count = Column(Integer, default=0)
chunk_count = Column(Integer, default=0)
status = Column(String(50), default="pending")
error_message = Column(Text)
created_at = Column(DateTime(timezone=True), server_default=func.now())
updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

__table_args__ = (
Index("idx_documents_org_id", "org_id"),
)

class DocumentChunk(Base):
__tablename__ = "document_chunks"

id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
org_id = Column(String, ForeignKey("organizations.id"), nullable=False)
content = Column(Text, nullable=False)
searchable_text = Column(Text) # Lowercase for keyword search
page_number = Column(Integer)
chunk_index = Column(Integer)
section_title = Column(String(500))
is_table = Column(Boolean, default=False)
embedding = Column(Vector(settings.embedding_dimensions))
created_at = Column(DateTime(timezone=True), server_default=func.now())

__table_args__ = (
Index("idx_chunks_document_id", "document_id"),
Index("idx_chunks_org_id", "org_id"),
)

class QueryLog(Base):
__tablename__ = "query_logs"

id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
org_id = Column(String, ForeignKey("organizations.id"), nullable=False)
user_id = Column(String, ForeignKey("users.id"))
session_id = Column(String(255))
question = Column(Text, nullable=False)
answer = Column(Text)
sources_used = Column(Integer)
latency_ms = Column(Integer)
from_cache = Column(Boolean, default=False)
created_at = Column(DateTime(timezone=True), server_default=func.now())

__table_args__ = (
Index("idx_query_logs_org_id_created", "org_id", "created_at"),
)
```

### 13. `app/api/routes/sessions.py` (NEW)

```python
from fastapi import APIRouter, Depends, Request, HTTPException
import uuid
import logging

from app.api.middleware.clerk_auth import require_org, ClerkUser
from app.services.memory import MemoryService
from app.models.schemas import CreateSessionRequest, SessionResponse, SessionHistoryResponse

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/", response_model=SessionResponse)
async def create_session(
request: Request,
user: ClerkUser = Depends(require_org)
):
"""Create a new conversation session"""
memory: MemoryService = request.app.state.memory

session_id = str(uuid.uuid4())
session = await memory.create_session(
session_id=session_id,
org_id=user.org_id,
user_id=user.user_id
)

return SessionResponse(
session_id=session_id,
created_at=session.created_at,
message="Session created. Use this session_id in queries for conversation memory."
)

@router.get("/{session_id}", response_model=SessionHistoryResponse)
async def get_session_history(
request: Request,
session_id: str,
user: ClerkUser = Depends(require_org)
):
"""Get conversation history for a session"""
memory: MemoryService = request.app.state.memory

session = await memory.get_memory(session_id)

if not session:
raise HTTPException(status_code=404, detail="Session not found or expired")

if session.org_id != user.org_id:
raise HTTPException(status_code=403, detail="Access denied")

return SessionHistoryResponse(
session_id=session.session_id,
turns=[t.model_dump() for t in session.turns],
created_at=session.created_at,
updated_at=session.updated_at
)

@router.delete("/{session_id}")
async def delete_session(
request: Request,
session_id: str,
user: ClerkUser = Depends(require_org)
):
"""Delete a conversation session"""
memory: MemoryService = request.app.state.memory

session = await memory.get_memory(session_id)

if session and session.org_id != user.org_id:
raise HTTPException(status_code=403, detail="Access denied")

await memory.delete_session(session_id)

return {"message": "Session deleted"}
```

### 14. `app/api/routes/query.py` (ENHANCED)

```python
from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse
import json
import logging

from app.api.middleware.clerk_auth import require_org, ClerkUser
from app.services.rag import RAGService
from app.services.vector_store import VectorStore
from app.services.cache import CacheService
from app.services.memory import MemoryService
from app.models.schemas import QueryRequest, QueryResponse
from app.models.database import QueryLog
from sqlalchemy import text

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/", response_model=QueryResponse)
async def query_documents(
request: Request,
query: QueryRequest,
user: ClerkUser = Depends(require_org)
):
"""Query documents with enhanced RAG pipeline"""
vector_store: VectorStore = request.app.state.vector_store
cache: CacheService = request.app.state.cache
memory: MemoryService = request.app.state.memory

rag_service = RAGService(vector_store, cache, memory)

response = await rag_service.query(
org_id=user.org_id,
question=query.question,
session_id=query.session_id,
top_k=query.top_k,
document_ids=query.document_ids,
use_cache=query.use_cache
)

# Log query
async with await vector_store.get_session() as session:
log = QueryLog(
org_id=user.org_id,
user_id=user.user_id,
session_id=query.session_id,
question=query.question,
answer=response.answer[:1000],
sources_used=len(response.sources),
latency_ms=response.query_time_ms,
from_cache=response.from_cache
)
session.add(log)
await session.commit()

return response

@router.post("/stream")
async def query_documents_stream(
request: Request,
query: QueryRequest,
user: ClerkUser = Depends(require_org)
):
"""Query documents with streaming response"""
vector_store: VectorStore = request.app.state.vector_store
cache: CacheService = request.app.state.cache
memory: MemoryService = request.app.state.memory

rag_service = RAGService(vector_store, cache, memory)

async def generate():
async for chunk in rag_service.query_stream(
org_id=user.org_id,
question=query.question,
session_id=query.session_id,
top_k=query.top_k,
document_ids=query.document_ids
):
yield f"data: {json.dumps(chunk)}\n\n"

return StreamingResponse(
generate(),
media_type="text/event-stream",
headers={
"Cache-Control": "no-cache",
"Connection": "keep-alive"
}
)
```

### 15. `app/api/routes/documents.py` (ENHANCED - with cache invalidation)

```python
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request, BackgroundTasks
from typing import List
import logging
import uuid

from app.api.middleware.clerk_auth import require_org, ClerkUser
from app.services.document_processor import DocumentProcessor
from app.services.embedding import EmbeddingService
from app.services.storage import StorageService
from app.services.vector_store import VectorStore
from app.services.cache import CacheService
from app.models.schemas import DocumentUploadResponse, DocumentMetadata, DocumentListResponse, DocumentStatus
from app.models.database import Document
from app.config import get_settings
from sqlalchemy import text

settings = get_settings()
logger = logging.getLogger(__name__)
router = APIRouter()

storage_service = StorageService()
document_processor = DocumentProcessor()

async def process_document_background(
document_id: str,
org_id: str,
storage_path: str,
vector_store: VectorStore,
cache: CacheService
):
"""Background task to process uploaded document"""
embedding_service = EmbeddingService() # No cache for batch

async with await vector_store.get_session() as session:
try:
# Update status
await session.execute(
text("UPDATE documents SET status = 'processing' WHERE id = :doc_id"),
{"doc_id": document_id}
)
await session.commit()

# Download and process
pdf_bytes = storage_service.download_file(storage_path)
chunks, page_count = document_processor.process_document(pdf_bytes)

if not chunks:
raise ValueError("No text content found in document")

# Generate embeddings
chunk_texts = [c["content"] for c in chunks]
embeddings = await embedding_service.embed_batch(chunk_texts)

# Store
await vector_store.store_chunks(
document_id=document_id,
org_id=org_id,
chunks=chunks,
embeddings=embeddings
)

# Update status
await session.execute(
text("""
UPDATE documents
SET status = 'ready', page_count = :page_count, chunk_count = :chunk_count
WHERE id = :doc_id
"""),
{"doc_id": document_id, "page_count": page_count, "chunk_count": len(chunks)}
)
await session.commit()

# Invalidate cache for this org
await cache.invalidate_org_cache(org_id)

logger.info(f"Document {document_id} processed: {len(chunks)} chunks")

except Exception as e:
logger.error(f"Error processing document {document_id}: {str(e)}")
await session.execute(
text("UPDATE documents SET status = 'failed', error_message = :error WHERE id = :doc_id"),
{"doc_id": document_id, "error": str(e)}
)
await session.commit()

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
request: Request,
background_tasks: BackgroundTasks,
file: UploadFile = File(...),
user: ClerkUser = Depends(require_org)
):
"""Upload a document for processing"""
if not file.filename.lower().endswith(".pdf"):
raise HTTPException(status_code=400, detail="Only PDF files are supported")

contents = await file.read()
file_size = len(contents)

if file_size > settings.max_file_size_mb * 1024 * 1024:
raise HTTPException(
status_code=400,
detail=f"File too large. Maximum size is {settings.max_file_size_mb}MB"
)

storage_path = storage_service.upload_file(
org_id=user.org_id,
file_bytes=contents,
filename=file.filename
)

document_id = str(uuid.uuid4())
vector_store: VectorStore = request.app.state.vector_store
cache: CacheService = request.app.state.cache

async with await vector_store.get_session() as session:
doc = Document(
id=document_id,
org_id=user.org_id,
filename=file.filename,
storage_path=storage_path,
file_size=file_size,
status="pending"
)
session.add(doc)
await session.commit()

background_tasks.add_task(
process_document_background,
document_id,
user.org_id,
storage_path,
vector_store,
cache
)

return DocumentUploadResponse(
document_id=document_id,
filename=file.filename,
status=DocumentStatus.PENDING,
message="Document uploaded. Processing with smart chunking started."
)

@router.get("/", response_model=DocumentListResponse)
async def list_documents(
request: Request,
user: ClerkUser = Depends(require_org)
):
"""List all documents"""
vector_store: VectorStore = request.app.state.vector_store

async with await vector_store.get_session() as session:
result = await session.execute(
text("""
SELECT id, org_id, filename, file_size, page_count, chunk_count,
status, created_at, updated_at
FROM documents
WHERE org_id = :org_id
ORDER BY created_at DESC
"""),
{"org_id": user.org_id}
)
rows = result.fetchall()

documents = [
DocumentMetadata(
id=str(row.id),
org_id=row.org_id,
filename=row.filename,
file_size=row.file_size,
page_count=row.page_count or 0,
chunk_count=row.chunk_count or 0,
status=DocumentStatus(row.status),
created_at=row.created_at,
updated_at=row.updated_at
)
for row in rows
]

return DocumentListResponse(documents=documents, total=len(documents))

@router.delete("/{document_id}")
async def delete_document(
request: Request,
document_id: str,
user: ClerkUser = Depends(require_org)
):
"""Delete a document"""
vector_store: VectorStore = request.app.state.vector_store
cache: CacheService = request.app.state.cache

async with await vector_store.get_session() as session:
result = await session.execute(
text("SELECT storage_path FROM documents WHERE id = :doc_id AND org_id = :org_id"),
{"doc_id": document_id, "org_id": user.org_id}
)
row = result.fetchone()

if not row:
raise HTTPException(status_code=404, detail="Document not found")

try:
storage_service.delete_file(row.storage_path)
except Exception as e:
logger.warning(f"Failed to delete file: {e}")

await vector_store.delete_document_chunks(document_id)

await session.execute(
text("DELETE FROM documents WHERE id = :doc_id"),
{"doc_id": document_id}
)
await session.commit()

# Invalidate cache
await cache.invalidate_org_cache(user.org_id)

return {"message": "Document deleted"}
```

### 16. `migrations/001_initial_schema.sql` (ENHANCED)

```sql
-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Organizations
CREATE TABLE IF NOT EXISTS organizations (
id VARCHAR(255) PRIMARY KEY,
name VARCHAR(255) NOT NULL,
stripe_customer_id VARCHAR(255) UNIQUE,
stripe_subscription_id VARCHAR(255),
subscription_status VARCHAR(50) DEFAULT 'inactive',
seat_limit INTEGER DEFAULT 5,
created_at TIMESTAMPTZ DEFAULT NOW(),
updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Users
CREATE TABLE IF NOT EXISTS users (
id VARCHAR(255) PRIMARY KEY,
org_id VARCHAR(255) REFERENCES organizations(id) ON DELETE SET NULL,
email VARCHAR(255) NOT NULL,
role VARCHAR(50) DEFAULT 'member',
created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Documents
CREATE TABLE IF NOT EXISTS documents (
id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
org_id VARCHAR(255) REFERENCES organizations(id) ON DELETE CASCADE NOT NULL,
filename VARCHAR(500) NOT NULL,
storage_path VARCHAR(1000) NOT NULL,
file_size INTEGER,
page_count INTEGER DEFAULT 0,
chunk_count INTEGER DEFAULT 0,
status VARCHAR(50) DEFAULT 'pending',
error_message TEXT,
created_at TIMESTAMPTZ DEFAULT NOW(),
updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_documents_org_id ON documents(org_id);

-- Document chunks with hybrid search support
CREATE TABLE IF NOT EXISTS document_chunks (
id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
document_id UUID REFERENCES documents(id) ON DELETE CASCADE NOT NULL,
org_id VARCHAR(255) REFERENCES organizations(id) ON DELETE CASCADE NOT NULL,
content TEXT NOT NULL,
searchable_text TEXT, -- Lowercase for trigram search
page_number INTEGER,
chunk_index INTEGER,
section_title VARCHAR(500),
is_table BOOLEAN DEFAULT FALSE,
embedding vector(1536),
created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for hybrid search
CREATE INDEX idx_chunks_document_id ON document_chunks(document_id);
CREATE INDEX idx_chunks_org_id ON document_chunks(org_id);

-- HNSW index for fast vector similarity
CREATE INDEX idx_chunks_embedding ON document_chunks
USING hnsw (embedding vector_cosine_ops);

-- GIN index for trigram text search (part numbers, codes)
CREATE INDEX idx_chunks_searchable_trgm ON document_chunks
USING gin (searchable_text gin_trgm_ops);

-- Query logs
CREATE TABLE IF NOT EXISTS query_logs (
id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
org_id VARCHAR(255) REFERENCES organizations(id) ON DELETE CASCADE NOT NULL,
user_id VARCHAR(255) REFERENCES users(id) ON DELETE SET NULL,
session_id VARCHAR(255),
question TEXT NOT NULL,
answer TEXT,
sources_used INTEGER,
latency_ms INTEGER,
from_cache BOOLEAN DEFAULT FALSE,
created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_query_logs_org_created ON query_logs(org_id, created_at);
CREATE INDEX idx_query_logs_session ON query_logs(session_id);

-- Auto-update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
NEW.updated_at = NOW();
RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_organizations_updated_at
BEFORE UPDATE ON organizations
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_documents_updated_at
BEFORE UPDATE ON documents
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Enable RLS
ALTER TABLE organizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE query_logs ENABLE ROW LEVEL SECURITY;
```

### 17. `requirements.txt` (ENHANCED)

```
# Core
fastapi==0.109.2
uvicorn[standard]==0.27.1
pydantic==2.6.1
pydantic-settings==2.1.0

# Database
sqlalchemy[asyncio]==2.0.25
asyncpg==0.29.0
pgvector==0.2.4

# Supabase
supabase==2.3.4

# Redis
redis[hiredis]==5.0.1

# Auth
python-jose[cryptography]==3.3.0
svix==1.17.0
httpx==0.26.0

# AI/ML
openai==1.12.0
anthropic==0.18.1
cohere==4.47

# Document processing
PyMuPDF==1.23.21
pdfplumber==0.10.3

# Payments
stripe==8.2.0

# Utils
python-multipart==0.0.9
python-dotenv==1.0.1
```

### 18. `.env.example` (ENHANCED)

```env
# App
DEBUG=false

# Supabase
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
DATABASE_URL=postgresql://postgres:password@db.xxxxx.supabase.co:5432/postgres

# Redis (Upstash)
REDIS_URL=redis://default:xxxxx@xxxxx.upstash.io:6379

# Clerk
CLERK_SECRET_KEY=sk_test_xxxxx
CLERK_WEBHOOK_SECRET=whsec_xxxxx
CLERK_PUBLISHABLE_KEY=pk_test_xxxxx

# OpenAI (embeddings)
OPENAI_API_KEY=sk-xxxxx

# Anthropic (LLM)
ANTHROPIC_API_KEY=sk-ant-xxxxx

# Cohere (reranking)
COHERE_API_KEY=xxxxx

# Stripe
STRIPE_SECRET_KEY=sk_test_xxxxx
STRIPE_WEBHOOK_SECRET=whsec_xxxxx
STRIPE_PRICE_ID=price_xxxxx
```

### 19. `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
build-essential \
&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 20. `docker-compose.yml`

```yaml
version: '3.8'

services:
api:
build: .
ports:
- "8000:8000"
env_file:
- .env
volumes:
- ./app:/app/app
command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 21. Remaining route files (health.py, auth.py, organizations.py, billing.py, clerk_auth.py)

These remain the same as the basic version - copy from the original prompt.

-----

## FINAL CHECKLIST

After generating all files:

1. ✅ All `__init__.py` files created
1. ✅ Redis service initialized in main.py
1. ✅ Hybrid search using both pgvector and pg_trgm
1. ✅ Cohere reranking integrated
1. ✅ Conversation memory with Redis
1. ✅ Query caching with Redis
1. ✅ Smart document chunking preserving structure
1. ✅ Cache invalidation on document changes
1. ✅ Session management endpoints
1. ✅ All env vars documented

Generate the complete codebase. Include all route files (health.py, auth.py, organizations.py, billing.py) and middleware (clerk_auth.py) from the basic version unchanged. Do not use placeholders.
