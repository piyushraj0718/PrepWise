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
