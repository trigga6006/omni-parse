# TechDocs AI - API Reference

## Base Configuration

| Setting | Value |
|---------|-------|
| Base URL | `http://localhost:8000` (dev) |
| API Prefix | `/api/v1` |
| Content-Type | `application/json` |

---

## Authentication

All protected endpoints require a **Clerk JWT token** in the Authorization header:

```
Authorization: Bearer <clerk_jwt_token>
```

Alternatively, the token can be passed via the `__session` cookie (for browser requests).

### Required Claims in JWT
- `sub` - User ID
- `org_id` - Organization ID (required for most endpoints)
- `org_role` - Organization role (`admin`, `org:admin`, `member`)

### Authentication Errors
| Status | Description |
|--------|-------------|
| `401 Unauthorized` | Missing or invalid token |
| `403 Forbidden` | Organization membership required / Admin access required |

---

## Endpoints Overview

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/` | No | App info |
| GET | `/health` | No | Full health check |
| GET | `/health/live` | No | Liveness probe |
| GET | `/health/ready` | No | Readiness probe |
| POST | `/api/v1/auth/webhook/clerk` | Webhook | Clerk webhook handler |
| POST | `/api/v1/documents/upload` | Yes | Upload document |
| GET | `/api/v1/documents` | Yes | List documents |
| GET | `/api/v1/documents/{id}` | Yes | Get document |
| DELETE | `/api/v1/documents/{id}` | Yes | Delete document |
| POST | `/api/v1/documents/{id}/reprocess` | Yes | Reprocess failed document |
| POST | `/api/v1/query` | Yes | Query documents (RAG) |
| POST | `/api/v1/query/stream` | Yes | Query with streaming response |
| GET | `/api/v1/query/history` | Yes | Get query history |
| POST | `/api/v1/sessions` | Yes | Create session |
| GET | `/api/v1/sessions` | Yes | List sessions |
| GET | `/api/v1/sessions/{id}` | Yes | Get session with messages |
| PATCH | `/api/v1/sessions/{id}` | Yes | Update session title |
| DELETE | `/api/v1/sessions/{id}` | Yes | Delete session |
| GET | `/api/v1/organizations/current` | Yes | Get current organization |
| GET | `/api/v1/organizations/current/stats` | Yes | Get organization stats |
| GET | `/api/v1/organizations/current/usage` | Yes | Get usage against limits |
| PATCH | `/api/v1/organizations/current` | Admin | Update organization |
| POST | `/api/v1/billing/checkout` | Admin | Create Stripe checkout |
| GET | `/api/v1/billing/subscription` | Yes | Get subscription status |
| POST | `/api/v1/billing/portal` | Admin | Create billing portal session |
| POST | `/api/v1/billing/webhook` | Webhook | Stripe webhook handler |

---

## Health Endpoints

### GET `/health`
Full health check with dependency status.

**Response** `200 OK`
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "development",
  "database": "healthy",
  "redis": "healthy"
}
```

Status values: `healthy`, `degraded`

### GET `/health/live`
Kubernetes liveness probe.

**Response** `200 OK`
```json
{
  "status": "alive"
}
```

### GET `/health/ready`
Kubernetes readiness probe.

**Response** `200 OK`
```json
{
  "status": "ready"
}
```

---

## Document Endpoints

### POST `/api/v1/documents/upload`
Upload a PDF document for processing.

**Headers**
```
Authorization: Bearer <token>
Content-Type: multipart/form-data
```

**Request Body** (multipart/form-data)
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | Yes | PDF file (max 50MB) |
| `title` | string | No | Document title |
| `description` | string | No | Document description |

**Response** `200 OK`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "manual.pdf",
  "status": "pending",
  "message": "Document uploaded and queued for processing"
}
```

**Errors**
| Status | Description |
|--------|-------------|
| `400` | Only PDF files supported |
| `413` | File size exceeds 50MB limit |
| `403` | Document limit reached (upgrade required) |

---

### GET `/api/v1/documents`
List documents for the organization.

**Query Parameters**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `page` | int | 1 | Page number (min: 1) |
| `page_size` | int | 20 | Items per page (1-100) |
| `status` | string | - | Filter by status: `pending`, `processing`, `completed`, `failed` |

**Response** `200 OK`
```json
{
  "documents": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "organization_id": "...",
      "filename": "manual.pdf",
      "title": "User Manual",
      "description": "Product user manual",
      "file_path": "org_id/doc_id/manual.pdf",
      "file_size": 1048576,
      "mime_type": "application/pdf",
      "status": "completed",
      "chunk_count": 45,
      "error_message": null,
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-15T10:32:00Z"
    }
  ],
  "total": 15,
  "page": 1,
  "page_size": 20
}
```

**Document Status Values**
| Status | Description |
|--------|-------------|
| `pending` | Uploaded, waiting to process |
| `processing` | Currently being processed |
| `completed` | Ready for querying |
| `failed` | Processing failed (check `error_message`) |

---

### GET `/api/v1/documents/{document_id}`
Get a specific document by ID.

**Path Parameters**
| Param | Type | Description |
|-------|------|-------------|
| `document_id` | UUID | Document ID |

**Response** `200 OK`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "organization_id": "...",
  "filename": "manual.pdf",
  "title": "User Manual",
  "description": "Product user manual",
  "file_path": "org_id/doc_id/manual.pdf",
  "file_size": 1048576,
  "mime_type": "application/pdf",
  "status": "completed",
  "chunk_count": 45,
  "error_message": null,
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:32:00Z"
}
```

**Errors**
| Status | Description |
|--------|-------------|
| `404` | Document not found |

---

### DELETE `/api/v1/documents/{document_id}`
Delete a document and all its chunks.

**Response** `204 No Content`

**Errors**
| Status | Description |
|--------|-------------|
| `404` | Document not found |

---

### POST `/api/v1/documents/{document_id}/reprocess`
Reprocess a failed or completed document.

**Response** `200 OK`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "manual.pdf",
  "status": "pending",
  "message": "Document queued for reprocessing"
}
```

**Errors**
| Status | Description |
|--------|-------------|
| `400` | Document is currently processing |
| `404` | Document not found |

---

## Query Endpoints

### POST `/api/v1/query`
Query documents using RAG pipeline.

**Request Body**
```json
{
  "query": "What is the warranty period for the product?",
  "session_id": "optional-session-uuid",
  "document_ids": ["uuid1", "uuid2"],
  "top_k": 5,
  "include_sources": true
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | - | Question (1-2000 chars) |
| `session_id` | string | No | - | Session ID for conversation continuity |
| `document_ids` | UUID[] | No | - | Filter to specific documents |
| `top_k` | int | No | 5 | Number of sources (1-20) |
| `include_sources` | bool | No | true | Include source documents |

**Response** `200 OK`
```json
{
  "answer": "The warranty period is 2 years from the date of purchase, as stated in the product documentation...",
  "sources": [
    {
      "document_id": "550e8400-e29b-41d4-a716-446655440000",
      "document_title": "User Manual",
      "chunk_id": "660e8400-e29b-41d4-a716-446655440001",
      "content": "Warranty Information: This product comes with a 2-year limited warranty...",
      "page_number": 15,
      "section_header": "Warranty Terms",
      "relevance_score": 0.92
    }
  ],
  "session_id": "770e8400-e29b-41d4-a716-446655440002",
  "query_id": "880e8400-e29b-41d4-a716-446655440003",
  "cached": false,
  "processing_time_ms": 1250
}
```

**Errors**
| Status | Description |
|--------|-------------|
| `400` | No processed documents available |
| `400` | Invalid document IDs |
| `429` | Monthly query limit reached |

---

### POST `/api/v1/query/stream`
Query with Server-Sent Events (SSE) streaming response.

**Request Body** - Same as `/api/v1/query`

**Response** `200 OK` (text/event-stream)
```
data: {"type": "sources", "content": [{"document_id": "...", ...}]}

data: {"type": "text", "content": "The "}

data: {"type": "text", "content": "warranty "}

data: {"type": "text", "content": "period is..."}

data: {"type": "done", "session_id": "..."}

```

**Event Types**
| Type | Description |
|------|-------------|
| `sources` | Array of source documents (sent first) |
| `text` | Text chunk of the response |
| `done` | Stream complete with session_id |
| `error` | Error occurred |

**Frontend Implementation Example**
```javascript
const eventSource = new EventSource('/api/v1/query/stream', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({ query: "What is..." })
});

// Or using fetch with ReadableStream
const response = await fetch('/api/v1/query/stream', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({ query: "What is..." })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      if (data.type === 'text') {
        // Append to response
      }
    }
  }
}
```

---

### GET `/api/v1/query/history`
Get query history for the organization.

**Query Parameters**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `limit` | int | 20 | Number of results |
| `offset` | int | 0 | Offset for pagination |
| `session_id` | string | - | Filter by session |

**Response** `200 OK`
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "query": "What is the warranty period?",
    "answer": "The warranty period is 2 years...",
    "created_at": "2024-01-15T10:30:00Z"
  }
]
```

---

## Session Endpoints

### POST `/api/v1/sessions`
Create a new conversation session.

**Request Body**
```json
{
  "title": "Product Questions"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `title` | string | No | Session title |

**Response** `200 OK`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "organization_id": "...",
  "title": "Product Questions",
  "created_at": "2024-01-15T10:30:00Z",
  "last_activity": "2024-01-15T10:30:00Z",
  "message_count": 0
}
```

---

### GET `/api/v1/sessions`
List conversation sessions.

**Query Parameters**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `limit` | int | 20 | Number of results (1-100) |
| `offset` | int | 0 | Offset for pagination |

**Response** `200 OK`
```json
{
  "sessions": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "organization_id": "...",
      "title": "Product Questions",
      "created_at": "2024-01-15T10:30:00Z",
      "last_activity": "2024-01-15T10:35:00Z",
      "message_count": 4
    }
  ],
  "total": 12
}
```

---

### GET `/api/v1/sessions/{session_id}`
Get session with all messages.

**Response** `200 OK`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "organization_id": "...",
  "title": "Product Questions",
  "created_at": "2024-01-15T10:30:00Z",
  "last_activity": "2024-01-15T10:35:00Z",
  "message_count": 4,
  "messages": [
    {
      "role": "user",
      "content": "What is the warranty period?",
      "timestamp": "2024-01-15T10:30:00Z",
      "sources": null
    },
    {
      "role": "assistant",
      "content": "The warranty period is 2 years...",
      "timestamp": "2024-01-15T10:30:05Z",
      "sources": [
        {
          "document_id": "...",
          "document_title": "User Manual",
          "chunk_id": "...",
          "content": "...",
          "page_number": 15,
          "section_header": "Warranty",
          "relevance_score": 0.92
        }
      ]
    }
  ]
}
```

---

### PATCH `/api/v1/sessions/{session_id}`
Update session title.

**Query Parameters**
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `title` | string | Yes | New title |

**Response** `200 OK` - Returns updated session object

---

### DELETE `/api/v1/sessions/{session_id}`
Delete a conversation session.

**Response** `204 No Content`

---

## Organization Endpoints

### GET `/api/v1/organizations/current`
Get current organization details.

**Response** `200 OK`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "clerk_org_id": "org_abc123",
  "name": "Acme Corp",
  "subscription_tier": "basic",
  "document_count": 15,
  "query_count": 250,
  "storage_used_mb": 45.5,
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-15T10:00:00Z"
}
```

**Subscription Tiers**
| Tier | Description |
|------|-------------|
| `free` | Free tier |
| `basic` | Basic paid plan |
| `pro` | Professional plan |
| `enterprise` | Enterprise plan |

---

### GET `/api/v1/organizations/current/stats`
Get detailed organization statistics.

**Response** `200 OK`
```json
{
  "total_documents": 15,
  "total_chunks": 450,
  "total_queries": 250,
  "storage_used_mb": 45.5,
  "queries_this_month": 85,
  "subscription_tier": "basic"
}
```

---

### GET `/api/v1/organizations/current/usage`
Get current usage against tier limits.

**Response** `200 OK`
```json
{
  "queries_used": 85,
  "queries_limit": 1000,
  "documents_used": 15,
  "documents_limit": 100,
  "storage_used_mb": 45.5,
  "storage_limit_mb": 1024
}
```

**Note**: `-1` means unlimited (enterprise tier)

---

### PATCH `/api/v1/organizations/current`
Update organization name (admin only).

**Query Parameters**
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | New organization name |

**Response** `200 OK` - Returns updated organization object

---

## Billing Endpoints

### POST `/api/v1/billing/checkout`
Create a Stripe checkout session (admin only).

**Request Body**
```json
{
  "price_id": "price_abc123",
  "success_url": "https://app.example.com/billing/success",
  "cancel_url": "https://app.example.com/billing/cancel"
}
```

**Response** `200 OK`
```json
{
  "checkout_url": "https://checkout.stripe.com/c/pay/cs_abc123",
  "session_id": "cs_abc123"
}
```

---

### GET `/api/v1/billing/subscription`
Get current subscription status.

**Response** `200 OK`
```json
{
  "tier": "basic",
  "status": "active",
  "current_period_end": "2024-02-15T00:00:00Z",
  "cancel_at_period_end": false
}
```

**Status Values**
| Status | Description |
|--------|-------------|
| `active` | Subscription is active |
| `past_due` | Payment failed |
| `canceled` | Subscription canceled |
| `unpaid` | Subscription unpaid |

---

### POST `/api/v1/billing/portal`
Create a Stripe customer portal session (admin only).

**Query Parameters**
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `return_url` | string | Yes | URL to return after portal |

**Response** `200 OK`
```json
{
  "url": "https://billing.stripe.com/p/session/..."
}
```

---

## Webhook Endpoints

### POST `/api/v1/auth/webhook/clerk`
Clerk webhook handler for user/organization sync.

**Headers**
```
svix-id: msg_abc123
svix-timestamp: 1234567890
svix-signature: v1=abc123...
```

**Handled Events**
- `organization.created`
- `organization.updated`
- `organization.deleted`
- `user.created`
- `user.deleted`
- `organizationMembership.created`
- `organizationMembership.deleted`

---

### POST `/api/v1/billing/webhook`
Stripe webhook handler for subscription events.

**Headers**
```
stripe-signature: t=1234567890,v1=abc123...
```

**Handled Events**
- `checkout.session.completed`
- `customer.subscription.updated`
- `customer.subscription.deleted`

---

## Error Response Format

All errors return a consistent JSON format:

```json
{
  "error": "Error message here",
  "detail": "Additional details (optional)",
  "code": "ERROR_CODE (optional)"
}
```

### Common HTTP Status Codes
| Code | Description |
|------|-------------|
| `400` | Bad Request - Invalid input |
| `401` | Unauthorized - Missing/invalid auth |
| `403` | Forbidden - Insufficient permissions |
| `404` | Not Found - Resource doesn't exist |
| `413` | Payload Too Large - File too big |
| `429` | Too Many Requests - Rate/usage limit |
| `500` | Internal Server Error |

---

## Rate Limits & Usage Limits

### Tier Limits
| Tier | Queries/Month | Documents | Storage |
|------|---------------|-----------|---------|
| Free | 100 | 10 | 100 MB |
| Basic | 1,000 | 100 | 1 GB |
| Pro | 10,000 | 500 | 10 GB |
| Enterprise | Unlimited | Unlimited | Unlimited |

---

## CORS Configuration

Allowed origins:
- `http://localhost:3000`
- `http://localhost:5173`
- `https://*.vercel.app`

All methods and headers are allowed. Credentials are supported.

---

## TypeScript Types

```typescript
// Document
interface Document {
  id: string;
  organization_id: string;
  filename: string;
  title: string | null;
  description: string | null;
  file_path: string;
  file_size: number;
  mime_type: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  chunk_count: number;
  error_message: string | null;
  created_at: string;
  updated_at: string | null;
}

// Query Request
interface QueryRequest {
  query: string;
  session_id?: string;
  document_ids?: string[];
  top_k?: number;
  include_sources?: boolean;
}

// Query Response
interface QueryResponse {
  answer: string;
  sources: QuerySource[];
  session_id: string;
  query_id: string;
  cached: boolean;
  processing_time_ms: number;
}

// Query Source
interface QuerySource {
  document_id: string;
  document_title: string;
  chunk_id: string;
  content: string;
  page_number: number | null;
  section_header: string | null;
  relevance_score: number;
}

// Session
interface Session {
  id: string;
  organization_id: string;
  title: string | null;
  created_at: string;
  last_activity: string;
  message_count: number;
}

// Session Message
interface SessionMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  sources: QuerySource[] | null;
}

// Organization
interface Organization {
  id: string;
  clerk_org_id: string;
  name: string;
  subscription_tier: 'free' | 'basic' | 'pro' | 'enterprise';
  document_count: number;
  query_count: number;
  storage_used_mb: number;
  created_at: string;
  updated_at: string | null;
}

// Usage Stats
interface UsageStats {
  queries_used: number;
  queries_limit: number;
  documents_used: number;
  documents_limit: number;
  storage_used_mb: number;
  storage_limit_mb: number;
}

// Subscription Status
interface SubscriptionStatus {
  tier: 'free' | 'basic' | 'pro' | 'enterprise';
  status: string;
  current_period_end: string | null;
  cancel_at_period_end: boolean;
}
```

---

## Frontend Integration Checklist

- [ ] Configure Clerk for authentication
- [ ] Set up API base URL from environment
- [ ] Implement auth token injection in requests
- [ ] Handle 401 errors with token refresh
- [ ] Implement file upload with progress
- [ ] Poll document status after upload
- [ ] Implement SSE streaming for queries
- [ ] Store session_id for conversation continuity
- [ ] Display usage limits from `/organizations/current/usage`
- [ ] Implement Stripe checkout flow
- [ ] Handle webhook-driven subscription updates
