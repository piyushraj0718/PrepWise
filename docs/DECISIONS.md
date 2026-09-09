# Architecture Decision Record

## How to use this file

Record a decision here before or alongside a change when it has lasting impact on structure, dependencies, data, testing, AI behavior, security, or deployment. Add new entries; do not rewrite past decisions without noting why they were superseded.

## ADR-001: Establish documentation before implementation

- Status: Accepted
- Date: 2026-09-09

### Context

PrepWise begins as a nearly empty repository. The project is intended to be production-minded while also supporting a student learning process.

### Decision

Create the engineering and architecture documentation before application code or dependencies.

### Rationale

This makes scope, boundaries, testability, and deferred choices visible before implementation creates accidental commitments.

### Consequences

The project has no runnable application yet. Each future milestone must translate these principles into the smallest useful implementation slice.

## ADR-002: Use modular, layered Python boundaries as the target design

- Status: Accepted
- Date: 2026-09-09

### Context

PrepWise will use AI and likely external services, while business behavior must remain understandable and independently testable.

### Decision

Separate domain rules, application use cases, AI adapters, infrastructure adapters, and delivery interfaces when the relevant code is introduced.

### Rationale

The separation permits deterministic rules and use cases to be tested with fakes, contains provider-specific complexity, and supports debugging without prematurely committing to a framework.

### Consequences

Concrete integrations will be introduced behind small interfaces only when an approved milestone requires them. No package skeleton is created yet.

## ADR-003: Defer technology and provider selection

- Status: Accepted
- Date: 2026-09-09

### Context

No first user workflow or non-functional constraints have been specified.

### Decision

Do not select frameworks, databases, AI providers, deployment targets, or dependencies during setup.

### Rationale

Selecting them now would be speculative and could add unnecessary complexity.

### Consequences

The first feature proposal must document the options and rationale for any choice that materially affects the architecture.

## ADR-004: Use FastAPI, SQLAlchemy, PostgreSQL, and local storage for Milestone 0

- Status: Accepted
- Date: 2026-09-09

### Context

The first approved workflow accepts a PDF, stores it locally, creates document metadata, and returns an identifier and status.

### Decision

Use FastAPI for the HTTP boundary, Pydantic for configuration/response schemas, SQLAlchemy with PostgreSQL for document metadata, and the local `data/uploads` directory for file bytes. Configure values through `.env` environment variables.

### Rationale

These are explicit requirements of the milestone and provide a small, conventional Python implementation with testable boundaries.

### Consequences

Production requires a reachable PostgreSQL database. Local file storage is temporary and must be replaced or abstracted further only when a later milestone requires it.

## ADR-005: Persist a failed document record after storage failure

- Status: Accepted
- Date: 2026-09-09

### Context

The workflow requires a document status including `failed`, and storage can fail after metadata creation.

### Decision

Create the document record first with `uploaded` status. If local storage fails, update the record to `failed` with a bounded failure reason and return HTTP 500 without exposing storage internals.

### Rationale

This provides an auditable status for failures while keeping operational details out of client responses.

### Consequences

For the brief storage window, the record is initially `uploaded`; a future workflow with asynchronous processing should introduce a `pending` state if it needs one.

## ADR-006: Persist extracted text separately and process synchronously

- Status: Accepted
- Date: 2026-09-09

### Context

Milestone 1 adds embedded-text extraction for uploaded PDFs. The application needs an inspectable record of the extracted content without introducing queues or background workers.

### Decision

Use `pypdf` in the document service to extract text during an explicit processing request. Store the result in a one-to-one `document_texts` table keyed by document ID. Persist `processing` before extraction, then `processed` only after text storage succeeds; mark the document `failed` for unreadable or textless PDFs.

### Rationale

This keeps parsing out of API routes, preserves metadata/text separation, and makes the status progression observable and testable while retaining a small synchronous implementation.

### Consequences

Processing may occupy an HTTP request for the duration of extraction. OCR and asynchronous processing remain out of scope until a later approved milestone.

## ADR-007: Add deterministic structure-aware chunking after text extraction

- Status: Accepted
- Date: 2026-09-09

### Context

The first transformation of extracted PDF text needs to preserve document structure and remain reproducible without introducing embeddings, retrieval systems, or external AI behavior.

### Decision

Add a synchronous chunk-generation step for processed documents that stores a `document_chunks` table keyed by a unique chunk ID. Chunk content is produced deterministically from paragraph/sentence structure, with page metadata when available and overlap to preserve local context.

### Rationale

This keeps the chunking layer explicit, inspectable, and easy to test while staying within the approved boundaries of the current milestone. It avoids arbitrary character splitting and preserves enough context for later retrieval-oriented work without starting a retrieval stack.

### Consequences

Chunk generation remains a local, deterministic transformation of extracted document text. Future retrieval or AI improvements can build on this data model without changing the current chunking contract.

## ADR-008: Persist chunk embeddings behind a local provider port

- Status: Accepted
- Date: 2026-09-09

### Context

Milestone 2B needs to turn persisted chunks into vectors without coupling application behavior to one model implementation or requiring a hosted API key.

### Decision

Persist embeddings in `document_embeddings` with a foreign key to `document_chunks`, model name, dimension, JSON vector data, and creation timestamp. Enforce uniqueness for each chunk and model name. Expose generation through an `EmbeddingProvider` protocol and use `sentence-transformers` with `all-MiniLM-L6-v2` as the default local adapter. Generation validates all provider output before replacing rows in one transaction; repeated requests reuse complete rows unless `rebuild=true` is requested.

### Rationale

The provider boundary keeps the service testable with deterministic fakes, while the model key permits future model/version records without changing the chunk contract. JSON vector storage is portable across the existing PostgreSQL and SQLite test databases and is sufficient until a separately approved retrieval milestone selects an index or vector database.

### Consequences

Installing `sentence-transformers` and the first-use model download are local development prerequisites for the default provider. Vector similarity search, indexing, retrieval, and RAG remain out of scope.

## ADR-009: Implement vector retrieval over existing JSON embeddings

- Status: Accepted
- Date: 2026-09-09

### Context

Milestone 2C needs query-time retrieval while the project already stores validated embedding vectors as JSON and has no configured vector extension or vector database.

### Decision

Add a retrieval service that uses the existing embedding provider for query vectors, loads processed-document candidates through the document repository, calculates cosine similarity in Python, and returns deterministic top-k chunk results. Support an optional document ID filter and expose the service through `POST /search`.

### Rationale

This establishes a small, testable retrieval boundary without adding pgvector, a new database, or a paid model API. Dimension mismatches, malformed vectors, and zero-magnitude vectors are ignored for stored candidates, while provider failures and invalid queries have explicit errors.

### Consequences

Candidate retrieval is a linear scan and is appropriate for this milestone's scope, but it will need an indexed strategy when corpus size requires it. Hybrid BM25 retrieval, reranking, and RAG remain separate future milestones.

## ADR-010: Combine local BM25 and dense rankings with reciprocal-rank fusion

- Status: Accepted
- Date: 2026-09-09

### Context

Milestone 2D needs lexical retrieval in addition to M2C dense retrieval, while keeping the implementation local, deterministic, and free of an external search service.

### Decision

Reuse `RetrievalService` and the existing query embedding provider. Scan processed chunks for BM25 scores using case-insensitive Unicode word tokens, retain only positive lexical matches, and combine dense and lexical ranks with reciprocal-rank fusion using `1 / (60 + rank)`. Return the existing dense similarity alongside BM25 and hybrid scores, with stable metadata tie-breakers.

### Rationale

Reciprocal-rank fusion combines rankings without pretending cosine and BM25 raw scores are directly comparable. A local scan keeps the milestone small and testable while supporting lexical-only matches and skipping malformed stored vectors for dense scoring.

### Consequences

Search remains linear in the number of processed chunks and does not maintain a persistent lexical index. Larger corpora may require a dedicated indexed strategy in a separately approved milestone. Reranking, RAG, and LLM calls remain out of scope.

## ADR-011: Add a lazy local cross-encoder reranking stage

- Status: Accepted
- Date: 2026-09-09

### Context

M2E needs a semantic reranking stage after M2D hybrid retrieval without changing the existing dense/BM25 pipeline, calling an LLM API, or requiring a remote reranking service.

### Decision

Add a `RerankerProvider` protocol and a `RerankingService` that accepts the original query and hybrid candidate chunks. The default adapter uses Sentence Transformers `cross-encoder/ms-marco-MiniLM-L-6-v2`, configured centrally and loaded only when scoring is first requested. `candidate_k` is validated and applied before reranking; only final `top_k` results are returned.

### Rationale

A cross-encoder scores each query/chunk pair directly and is a better semantic relevance signal than comparing independent embeddings. The provider boundary keeps tests deterministic with fakes and permits changing models later. Lazy loading avoids model work during import and keeps tests independent of model availability.

### Consequences

Reranking adds local model latency and remains bounded by `candidate_k`; the default model may need a first-use download and local resources. No RAG generation, LLM call, or external reranking service is introduced.

## ADR-012: Add grounded RAG with a replaceable Gemini provider

- Status: Accepted
- Date: 2026-09-09

### Context

M2F needs the first question-answering vertical slice while keeping retrieval, reranking, and generation separate. The project had no previously selected hosted LLM provider, so one provider must be chosen without coupling the application to it.

### Decision

Add a `LLMProvider` port and a `RAGService` that consumes existing reranked chunks. Use Gemini's REST `generateContent` API as the first real adapter, configured by `GEMINI_API_KEY`, `GEMINI_MODEL_NAME`, and `GEMINI_TIMEOUT_SECONDS`. Centralize grounding instructions and request structured JSON containing an answer and supplied chunk IDs. Validate IDs against retrieved candidates and build final citations from application-owned metadata.

### Rationale

Gemini provides a straightforward structured-generation API and can be replaced behind the provider port. Environment-only credentials avoid secrets in source control. Application-side citation mapping prevents fabricated sources, while fake providers keep normal tests deterministic and offline.

### Consequences

Real `/ask` requests require a configured Gemini API key, network access, and model availability. The optional smoke script may incur provider usage and latency. No question generation, quiz generation, adaptive learning, agents, or evaluation framework is included.
