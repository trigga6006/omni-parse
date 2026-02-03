-- TechDocs AI Initial Schema
-- Requires: pgvector extension, pg_trgm extension

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Create enum types
CREATE TYPE document_status AS ENUM ('pending', 'processing', 'completed', 'failed');
CREATE TYPE subscription_tier AS ENUM ('free', 'basic', 'pro', 'enterprise');

-- Organizations table (multi-tenancy)
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    clerk_org_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    subscription_tier subscription_tier DEFAULT 'free',
    stripe_customer_id VARCHAR(255),
    stripe_subscription_id VARCHAR(255),
    document_count INTEGER DEFAULT 0,
    query_count INTEGER DEFAULT 0,
    storage_used_mb FLOAT DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ
);

CREATE INDEX idx_organizations_clerk ON organizations(clerk_org_id);

-- Documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    filename VARCHAR(500) NOT NULL,
    title VARCHAR(500),
    description TEXT,
    file_path VARCHAR(1000) NOT NULL,
    file_size INTEGER NOT NULL,
    mime_type VARCHAR(100) NOT NULL,
    status document_status DEFAULT 'pending',
    chunk_count INTEGER DEFAULT 0,
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ
);

CREATE INDEX idx_documents_org ON documents(organization_id);
CREATE INDEX idx_documents_org_status ON documents(organization_id, status);
CREATE INDEX idx_documents_created ON documents(created_at DESC);

-- Document chunks table with vector embeddings
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_tsvector TSVECTOR,
    embedding vector(1536),
    page_number INTEGER,
    section_header VARCHAR(500),
    token_count INTEGER NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_chunks_document ON document_chunks(document_id, chunk_index);

-- Vector similarity index (HNSW - no rebuild needed after inserts)
CREATE INDEX idx_chunks_embedding ON document_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Full-text search index
CREATE INDEX idx_chunks_tsvector ON document_chunks USING GIN(content_tsvector);

-- Trigram index for fuzzy matching
CREATE INDEX idx_chunks_content_trgm ON document_chunks USING GIN(content gin_trgm_ops);

-- Query logs for analytics
CREATE TABLE query_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    session_id VARCHAR(100),
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    source_chunks JSONB DEFAULT '[]',
    processing_time_ms INTEGER NOT NULL,
    cached BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_queries_org ON query_logs(organization_id);
CREATE INDEX idx_queries_org_session ON query_logs(organization_id, session_id);
CREATE INDEX idx_queries_created ON query_logs(created_at DESC);

-- Function to update tsvector on insert/update
CREATE OR REPLACE FUNCTION update_chunk_tsvector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.content_tsvector := to_tsvector('english', NEW.content);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for tsvector update
CREATE TRIGGER chunk_tsvector_update
    BEFORE INSERT OR UPDATE OF content ON document_chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_chunk_tsvector();

-- Function for hybrid search (semantic + keyword)
CREATE OR REPLACE FUNCTION hybrid_search(
    query_embedding vector(1536),
    query_text TEXT,
    org_id UUID,
    doc_ids UUID[] DEFAULT NULL,
    match_count INTEGER DEFAULT 20,
    semantic_weight FLOAT DEFAULT 0.7,
    keyword_weight FLOAT DEFAULT 0.3,
    similarity_threshold FLOAT DEFAULT 0.5
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    page_number INTEGER,
    section_header VARCHAR(500),
    semantic_score FLOAT,
    keyword_score FLOAT,
    combined_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH semantic_results AS (
        SELECT
            dc.id,
            dc.document_id,
            dc.content,
            dc.page_number,
            dc.section_header,
            1 - (dc.embedding <=> query_embedding) AS sem_score
        FROM document_chunks dc
        JOIN documents d ON dc.document_id = d.id
        WHERE d.organization_id = org_id
            AND d.status = 'completed'
            AND (doc_ids IS NULL OR dc.document_id = ANY(doc_ids))
            AND 1 - (dc.embedding <=> query_embedding) > similarity_threshold
        ORDER BY dc.embedding <=> query_embedding
        LIMIT match_count * 2
    ),
    keyword_results AS (
        SELECT
            dc.id,
            ts_rank_cd(dc.content_tsvector, plainto_tsquery('english', query_text)) +
            similarity(dc.content, query_text) * 0.5 AS kw_score
        FROM document_chunks dc
        JOIN documents d ON dc.document_id = d.id
        WHERE d.organization_id = org_id
            AND d.status = 'completed'
            AND (doc_ids IS NULL OR dc.document_id = ANY(doc_ids))
            AND (
                dc.content_tsvector @@ plainto_tsquery('english', query_text)
                OR dc.content % query_text
            )
    ),
    combined AS (
        SELECT
            sr.id AS chunk_id,
            sr.document_id,
            sr.content,
            sr.page_number,
            sr.section_header,
            sr.sem_score AS semantic_score,
            COALESCE(kr.kw_score, 0.0) AS keyword_score,
            (sr.sem_score * semantic_weight + COALESCE(kr.kw_score, 0.0) * keyword_weight) AS combined_score
        FROM semantic_results sr
        LEFT JOIN keyword_results kr ON sr.id = kr.id
    )
    SELECT
        c.chunk_id,
        c.document_id,
        c.content,
        c.page_number,
        c.section_header,
        c.semantic_score::FLOAT,
        c.keyword_score::FLOAT,
        c.combined_score::FLOAT
    FROM combined c
    ORDER BY c.combined_score DESC
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Function to update organization stats
CREATE OR REPLACE FUNCTION update_org_document_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE organizations
        SET document_count = document_count + 1,
            updated_at = NOW()
        WHERE id = NEW.organization_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE organizations
        SET document_count = document_count - 1,
            storage_used_mb = storage_used_mb - (OLD.file_size::FLOAT / 1048576),
            updated_at = NOW()
        WHERE id = OLD.organization_id;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_org_doc_count
    AFTER INSERT OR DELETE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_org_document_count();

-- Function to increment query count
CREATE OR REPLACE FUNCTION increment_query_count()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE organizations
    SET query_count = query_count + 1,
        updated_at = NOW()
    WHERE id = NEW.organization_id;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_org_query_count
    AFTER INSERT ON query_logs
    FOR EACH ROW
    EXECUTE FUNCTION increment_query_count();

-- Row Level Security (RLS) policies
ALTER TABLE organizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE query_logs ENABLE ROW LEVEL SECURITY;

-- Note: RLS policies should be configured based on your auth setup
-- Example policy (customize based on your needs):
-- CREATE POLICY "Organizations are viewable by members" ON organizations
--     FOR SELECT USING (auth.uid() IN (
--         SELECT user_id FROM organization_members WHERE organization_id = id
--     ));
