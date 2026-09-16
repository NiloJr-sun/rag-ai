-- T1.4 -- documents, chunks, and their embeddings.
--
-- Run once against your Supabase project: paste into the SQL Editor, or
--     psql "$DATABASE_URL" -f backend/migrations/0001_init.sql
--
-- Every statement is idempotent, so re-running is safe.

-- pgvector supplies the `vector` column type and the distance operators.
create extension if not exists vector;

-- One row per source file. Kept separate from chunks so a document can be
-- re-ingested or deleted as a unit (T5.7, T5.8) and so a retrieved chunk can
-- name its origin in a citation (T2.11).
create table if not exists documents (
    id           text primary key,
    source       text not null unique,
    content_hash text not null,
    ingested_at  timestamptz not null default now()
);

comment on column documents.content_hash is
    'Hash of the file contents. Lets a re-sync skip unchanged documents (T5.4), '
    'which matters because embedding is deterministic -- see docs/rag/embeddings.md.';

-- One row per chunk. 768 dimensions is fixed by nomic-embed-text; a different
-- embedding model means a different width and a full re-index (T3.7).
create table if not exists document_chunks (
    id          text primary key,
    document_id text not null references documents (id) on delete cascade,
    chunk_index integer not null,
    content     text not null,
    embedding   vector(768),
    created_at  timestamptz not null default now(),
    unique (document_id, chunk_index)
);

-- Fetching or deleting every chunk of one document.
create index if not exists document_chunks_document_id_idx
    on document_chunks (document_id);

-- Approximate nearest-neighbour search. vector_cosine_ops matches the cosine
-- similarity used in app/rag/retrieval.py -- an index built for a different
-- distance would simply be ignored by a cosine query.
--
-- HNSW rather than IVFFlat: it can be built on an empty table, whereas
-- IVFFlat needs representative data present before its lists are meaningful.
create index if not exists document_chunks_embedding_idx
    on document_chunks using hnsw (embedding vector_cosine_ops);

-- Supabase exposes every table through PostgREST. Enabling RLS with no
-- policies denies anon and authenticated clients outright, while the service
-- role key the backend uses bypasses RLS. Without this, the tables would be
-- readable by anyone holding the public anon key.
alter table documents enable row level security;
alter table document_chunks enable row level security;
