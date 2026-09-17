# RAG-AI Development Roadmap

> Learn vector databases and RAG to build a chatbot that answers questions from files synced from Google Drive.

---

## Target 1 — Foundations: Vector DB & Basic RAG Loop

**Goal:** Get a working script-level RAG loop end-to-end.

### T1.1 — Learn Embeddings

- [x] Understand what embeddings are
- [x] Generate embeddings using a local Ollama embedding model
- [x] Experiment with embeddings for similar/different sentences
- [x] Understand embedding dimensions
- [x] Document observations

**Output:** Small script demonstrating text → embedding.

---

### T1.2 — Learn Cosine Similarity

- [x] Understand cosine similarity
- [x] Implement cosine similarity manually
- [x] Compare vectors generated from similar/different text
- [x] Understand similarity scores and their limitations

**Output:** Script that ranks text by similarity to a query.

---

### T1.3 — Learn Chunking Strategies

- [x] Understand why documents need to be chunked
- [x] Implement fixed-size chunking
- [x] Experiment with different chunk sizes
- [x] Compare sentence-based vs fixed-size chunking
- [x] Document trade-offs

**Output:** Chunking module with configurable chunk size.

---

### T1.4 — Set Up Supabase + pgvector

- [x] Create Supabase project
- [x] Enable pgvector
- [x] Create documents table
- [x] Create document chunks table
- [x] Add embedding vector column
- [x] Add basic indexes
- [x] Test inserting and retrieving vectors

**Output:** Working Supabase pgvector database.

---

### T1.5 — Set Up Ollama

- [x] Install Ollama
- [x] Select embedding model
- [x] Select chat model
- [x] Verify models locally
- [x] Test embedding generation
- [x] Test chat completion

**Output:** Local Ollama setup documented.

---

### T1.6 — Implement Document Chunking

- [x] Read local text files
- [x] Split documents into chunks
- [x] Generate stable document/chunk IDs
- [x] Preserve original document information
- [x] Return structured chunks

**Output:** `file → chunks`

---

### T1.7 — Implement Embedding Pipeline

- [x] Generate embedding for each chunk
- [x] Attach embeddings to chunks
- [x] Handle embedding failures
- [x] Measure embedding generation time

**Output:** `chunks → embeddings`

---

### T1.8 — Store Chunks in Supabase

- [x] Insert documents
- [x] Insert chunks
- [x] Store embeddings
- [x] Verify stored vectors
- [x] Implement basic cleanup/reset functionality

**Output:** `chunks + embeddings → Supabase`

---

### T1.9 — Implement Vector Retrieval

- [x] Generate embedding for a user query
- [x] Perform vector similarity search
- [x] Return top-k chunks
- [x] Experiment with different k values
- [x] Inspect retrieved chunks manually

**Output:** `query → relevant chunks`

---

### T1.10 — Implement Basic RAG Answering

- [x] Build prompt containing retrieved chunks
- [x] Send prompt to Ollama chat model
- [x] Generate answer
- [x] Return answer + retrieved sources

**Output:** `question → retrieve → answer`

---

### T1.11 — Complete Basic RAG CLI

- [x] Connect all components
- [x] Add CLI interface
- [x] Support multiple local files
- [x] Ask questions interactively
- [x] Display retrieved sources
- [x] Document how to run it

**Target 1 Complete When:**

```text
file
 ↓
chunk
 ↓
embed
 ↓
store
 ↓
retrieve
 ↓
answer
```

---

# Target 2 — Production-Ready RAG Pipeline

**Goal:** Turn the script into a real backend service.

### T2.1 — Create FastAPI Application

- [ ] Set up FastAPI project
- [ ] Define project structure
- [ ] Add configuration management
- [ ] Add health endpoint
- [ ] Add API documentation

**Output:** Running FastAPI service.

---

### T2.2 — Create Document Ingestion API

- [ ] Add document upload endpoint
- [ ] Validate supported file types
- [ ] Parse uploaded files
- [ ] Chunk documents
- [ ] Generate embeddings
- [ ] Store documents/chunks

**Output:** API-based document ingestion.

---

### T2.3 — Add Document Metadata

Add metadata fields:

```text
filename
page
source
document_id
chunk_id
```

- [ ] Update database schema
- [ ] Store metadata during ingestion
- [ ] Return metadata during retrieval
- [ ] Verify metadata integrity

---

### T2.4 — Improve Chunking

- [ ] Add chunk overlap
- [ ] Make chunk size configurable
- [ ] Experiment with overlap size
- [ ] Handle page boundaries
- [ ] Preserve document structure where possible

---

### T2.5 — Add PDF Support

- [ ] Select PDF parser
- [ ] Extract text
- [ ] Preserve page numbers
- [ ] Handle malformed PDFs
- [ ] Test with real PDFs

---

### T2.6 — Add DOCX Support

- [ ] Select DOCX parser
- [ ] Extract paragraphs
- [ ] Preserve basic document structure
- [ ] Handle malformed documents
- [ ] Test with real DOCX files

---

### T2.7 — Add TXT Support

- [ ] Add TXT parser
- [ ] Handle encoding issues
- [ ] Test different text files

---

### T2.8 — Implement Top-K Retrieval

- [ ] Make top-k configurable
- [ ] Test multiple values
- [ ] Compare retrieved context
- [ ] Record observations

---

### T2.9 — Implement Hybrid Search

Combine:

```text
Keyword Search
      +
Vector Search
      ↓
Combined Results
```

- [ ] Implement keyword search
- [ ] Implement vector search
- [ ] Combine results
- [ ] Experiment with weighting
- [ ] Compare against vector-only retrieval

---

### T2.10 — Implement Re-ranking

- [ ] Research re-ranking approaches
- [ ] Select a local/appropriate re-ranker
- [ ] Re-rank retrieved chunks
- [ ] Compare before/after results
- [ ] Measure latency impact

---

### T2.11 — Add Answer + Source Citations

- [ ] Return answer
- [ ] Return source filename
- [ ] Return page number where available
- [ ] Return chunk/source metadata
- [ ] Ensure citations map to retrieved context

**Output:**

```json
{
  "answer": "...",
  "sources": [
    {
      "filename": "example.pdf",
      "page": 4,
      "chunk_id": "..."
    }
  ]
}
```

---

### T2.12 — Evaluate Retrieval Strategies Manually

Test:

- [ ] Different chunk sizes
- [ ] Different overlap sizes
- [ ] Different top-k values
- [ ] Vector-only retrieval
- [ ] Hybrid retrieval
- [ ] Re-ranking

**Output:** Comparison notes explaining what works better and why.

---

# Target 3 — RAG Evaluation

**Goal:** Measure RAG quality objectively instead of relying on manual inspection.

### T3.1 — Learn RAG Evaluation Concepts

- [ ] Precision
- [ ] Recall
- [ ] Context relevance
- [ ] Answer relevance
- [ ] Faithfulness
- [ ] Groundedness

**Output:** Evaluation notes/documentation.

---

### T3.2 — Build Evaluation Dataset

Create:

```text
question
expected_answer
expected_sources
```

- [ ] Collect representative documents
- [ ] Create questions
- [ ] Define expected answers
- [ ] Identify expected source chunks
- [ ] Include easy questions
- [ ] Include difficult questions
- [ ] Include questions requiring multiple chunks

---

### T3.3 — Set Up Evaluation Framework

- [ ] Add pytest
- [ ] Select RAG evaluation tooling
- [ ] Create evaluation runner
- [ ] Define evaluation configuration
- [ ] Store evaluation results

---

### T3.4 — Evaluate Retrieval Quality

Measure:

- [ ] Precision
- [ ] Recall
- [ ] Context relevance
- [ ] Source retrieval accuracy

---

### T3.5 — Evaluate Answer Quality

Measure:

- [ ] Answer relevance
- [ ] Faithfulness
- [ ] Groundedness
- [ ] Hallucination behavior

---

### T3.6 — Compare Chunking Strategies

Evaluate:

```text
chunk size A
chunk size B
chunk size C
```

- [ ] Run evaluation suite
- [ ] Compare metrics
- [ ] Record latency
- [ ] Identify best configuration

---

### T3.7 — Compare Embedding Models

- [ ] Select multiple embedding models
- [ ] Re-index evaluation dataset
- [ ] Run evaluation
- [ ] Compare quality
- [ ] Compare performance/cost

---

### T3.8 — Compare Retrieval Strategies

Compare:

- [ ] Vector search
- [ ] Hybrid search
- [ ] Different top-k values
- [ ] Re-ranking

---

### T3.9 — Generate Baseline Report

Create a report containing:

```text
Configuration
    ↓
Metrics
    ↓
Latency
    ↓
Observations
```

**Target 3 Complete When:**

> I can say "this change improved the RAG pipeline" using measurable results rather than subjective judgment.

---

# Target 4 — Chatbot UI

**Goal:** Give the RAG pipeline a usable interface.

### T4.1 — Create Next.js Application

- [ ] Set up Next.js
- [ ] Set up TypeScript
- [ ] Create basic application layout
- [ ] Configure environment variables

---

### T4.2 — Build Chat Interface

- [ ] Message list
- [ ] User messages
- [ ] Assistant messages
- [ ] Input field
- [ ] Send button
- [ ] Loading state
- [ ] Error state

---

### T4.3 — Connect UI to FastAPI

- [ ] Create API client
- [ ] Send user questions
- [ ] Display answers
- [ ] Handle API errors
- [ ] Handle loading states

---

### T4.4 — Add Streaming Responses

- [ ] Implement streaming endpoint
- [ ] Stream tokens to frontend
- [ ] Render partial responses
- [ ] Handle interrupted streams

---

### T4.5 — Display Source Citations

- [ ] Display source filenames
- [ ] Display page numbers
- [ ] Display relevant chunks
- [ ] Make citations easy to understand

---

### T4.6 — Add Multi-Turn Conversation

- [ ] Store conversation messages
- [ ] Send conversation context to backend
- [ ] Handle conversation IDs
- [ ] Prevent unbounded context growth
- [ ] Test follow-up questions

**Output:**

```text
Browser
   ↓
Next.js
   ↓
FastAPI
   ↓
RAG Pipeline
   ↓
Answer + Sources
```

---

# Target 5 — Google Drive Integration

**Goal:** Replace manual uploads with a live Google Drive source.

### T5.1 — Set Up Google Drive API

- [ ] Create Google Cloud project
- [ ] Enable Google Drive API
- [ ] Configure OAuth
- [ ] Create credentials
- [ ] Test authentication
- [ ] Store credentials securely

---

### T5.2 — Implement Google Drive File Listing

- [ ] Authenticate
- [ ] List files
- [ ] Retrieve file metadata
- [ ] Filter supported file types
- [ ] Identify file IDs

---

### T5.3 — Implement File Download

- [ ] Download supported files
- [ ] Handle Google-native documents where applicable
- [ ] Handle download failures
- [ ] Store temporary files safely

---

### T5.4 — Implement Change Detection

Track:

```text
file_id
modified_time
checksum/hash
last_synced_at
```

- [ ] Detect new files
- [ ] Detect updated files
- [ ] Detect unchanged files
- [ ] Detect deleted files

---

### T5.5 — Build Google Drive Ingestion Pipeline

Implement:

```text
list
 ↓
detect changes
 ↓
download
 ↓
parse
 ↓
chunk
 ↓
embed
 ↓
upsert
```

---

### T5.6 — Handle New Files

- [ ] Detect new files
- [ ] Download
- [ ] Parse
- [ ] Chunk
- [ ] Embed
- [ ] Store

---

### T5.7 — Handle Updated Files

- [ ] Detect modified files
- [ ] Remove/replace stale chunks
- [ ] Re-process document
- [ ] Re-embed
- [ ] Update metadata

---

### T5.8 — Handle Deleted Files

- [ ] Detect deleted Drive files
- [ ] Mark documents deleted or remove them
- [ ] Remove/disable associated chunks
- [ ] Ensure deleted documents aren't retrieved

---

### T5.9 — Make Sync Idempotent

Ensure repeated syncs don't create duplicates.

- [ ] Define unique document identity
- [ ] Use upsert
- [ ] Test repeated sync
- [ ] Test interrupted sync
- [ ] Test partial failure

---

### T5.10 — Add Sync Status

Track:

```text
pending
syncing
completed
failed
```

Store:

```text
last_synced_at
last_error
sync_status
```

---

### T5.11 — Add Manual Sync Trigger

- [ ] Add API endpoint
- [ ] Trigger Google Drive sync
- [ ] Return sync status
- [ ] Prevent conflicting syncs

---

### T5.12 — Add Scheduled Sync

- [ ] Choose scheduling approach
- [ ] Configure cron/scheduler
- [ ] Trigger sync automatically
- [ ] Record sync results

---

### T5.13 — End-to-End Google Drive Test

Verify:

```text
Google Drive
     ↓
Sync
     ↓
Parse
     ↓
Chunk
     ↓
Embed
     ↓
Supabase
     ↓
Retrieve
     ↓
Chatbot
```

**Target 5 Complete When:**

> Adding or updating a document in Google Drive eventually makes its content available to the chatbot without manual upload.

---

# Target 6 — Productionization & Observability

**Goal:** Make the application reliable and operable, not just functional.

### T6.1 — Dockerize Application

- [ ] Dockerize FastAPI
- [ ] Dockerize Next.js
- [ ] Configure Docker Compose
- [ ] Configure environment variables
- [ ] Add local development setup
- [ ] Document startup process

---

### T6.2 — Add Redis

- [ ] Add Redis
- [ ] Configure connection
- [ ] Test queue communication
- [ ] Document Redis role

---

### T6.3 — Move Ingestion to Background Jobs

- [ ] Select job queue implementation
- [ ] Create ingestion job
- [ ] Move embedding/processing out of request lifecycle
- [ ] Return job ID
- [ ] Track job status

---

### T6.4 — Add Job Status Tracking

Track:

```text
queued
processing
completed
failed
```

Include:

```text
job_id
document_id
started_at
completed_at
error
```

---

### T6.5 — Add Retry Handling

- [ ] Identify retryable failures
- [ ] Configure retry limits
- [ ] Add exponential backoff
- [ ] Record failed jobs
- [ ] Prevent infinite retries

---

### T6.6 — Add Structured Logging

Log:

```text
request_id
job_id
document_id
operation
status
duration
error
```

- [ ] Standardize log format
- [ ] Add log levels
- [ ] Ensure errors contain useful context

---

### T6.7 — Add Distributed Tracing

Trace:

```text
API request
 ↓
retrieval
 ↓
embedding
 ↓
LLM request
```

For ingestion:

```text
sync
 ↓
download
 ↓
parse
 ↓
chunk
 ↓
embed
 ↓
database
```

---

### T6.8 — Track LLM Usage

Track:

- [ ] Input tokens
- [ ] Output tokens
- [ ] Total tokens
- [ ] Requests
- [ ] Model
- [ ] Estimated cost where applicable

---

### T6.9 — Track Latency

Measure:

- [ ] API latency
- [ ] Embedding latency
- [ ] Retrieval latency
- [ ] Re-ranking latency
- [ ] LLM latency
- [ ] End-to-end response latency
- [ ] Ingestion latency

---

### T6.10 — Create RAG Observability Dashboard/Report

Expose:

```text
Requests
Latency
Token usage
Retrieval performance
Failed jobs
Sync status
```

---

### T6.11 — Integrate RAG Evaluation into CI

- [ ] Run evaluation suite automatically
- [ ] Define baseline metrics
- [ ] Define acceptable thresholds
- [ ] Detect metric regressions
- [ ] Fail CI when important quality metrics regress

---

### T6.12 — Create Production Readiness Checklist

Verify:

- [ ] Error handling
- [ ] Retries
- [ ] Logging
- [ ] Tracing
- [ ] Monitoring
- [ ] Evaluation
- [ ] Database reliability
- [ ] Background jobs
- [ ] Configuration management
- [ ] Secrets management
- [ ] Documentation

---

# Final Architecture

The final system should evolve toward:

```text
                         ┌─────────────────┐
                         │   Google Drive  │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │   Sync Worker   │
                         └────────┬────────┘
                                  │
                     ┌────────────┴────────────┐
                     │                         │
                     ▼                         ▼
                Parse Files                Metadata
                     │
                     ▼
                  Chunking
                     │
                     ▼
                 Embeddings
                     │
                     ▼
              ┌───────────────┐
              │   Supabase    │
              │   + pgvector  │
              └───────┬───────┘
                      │
                      │ Retrieval
                      ▼
┌─────────────┐   ┌───────────────┐
│   Next.js   │──▶│    FastAPI    │
│  Chat UI    │   │   RAG API     │
└─────────────┘   └───────┬───────┘
                          │
                    ┌─────┴─────┐
                    │           │
                    ▼           ▼
               Retrieval      Ollama
                    │           │
                    └─────┬─────┘
                          ▼
                   Answer + Sources


        ┌──────────────────────────────┐
        │      Evaluation System       │
        │                              │
        │ Retrieval Quality            │
        │ Answer Quality               │
        │ Faithfulness                 │
        │ Regression Detection         │
        └──────────────────────────────┘
```

# Suggested Ticket Progression

If this is a **learning project**, I would implement the tickets in this order:

```text
T1.1  Embeddings
  ↓
T1.2  Cosine Similarity
  ↓
T1.3  Chunking
  ↓
T1.4  Supabase + pgvector
  ↓
T1.5  Ollama
  ↓
T1.6–T1.10  Basic RAG
  ↓
T2.1  FastAPI
  ↓
T2.3–T2.11  Production RAG
  ↓
T3.1–T3.9  Evaluation
  ↓
T4.1–T4.6  Chat UI
  ↓
T5.1–T5.13  Google Drive
  ↓
T6.1–T6.12  Productionization
```

## Milestones

| Milestone | What you should have |
|---|---|
| **M1 — RAG Hello World** | Local files → chunks → embeddings → pgvector → answer |
| **M2 — RAG API** | FastAPI + PDF/DOCX/TXT + citations |
| **M3 — Measurable RAG** | Evaluation dataset + metrics + benchmark |
| **M4 — Chatbot** | Next.js UI + streaming + conversation |
| **M5 — Google Drive RAG** | Automatic Drive → RAG synchronization |
| **M6 — Production RAG** | Workers + retries + observability + regression evaluation |

## Definition of Done

A feature should generally be considered **done** when:

- [ ] Implementation works
- [ ] Tests exist where appropriate
- [ ] The behavior is documented
- [ ] Errors are handled
- [ ] The change doesn't introduce a known RAG quality regression
- [ ] Evaluation results are checked for RAG-related changes

The ultimate goal isn't just:

> **"I built a chatbot using RAG."**

It is:

> **"I understand how each part of a RAG system works, I can measure its quality, and I can make informed engineering decisions about improving it."**