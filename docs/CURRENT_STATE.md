# Current Project State

Last reviewed: 2026-09-09

## Repository inventory

- A FastAPI application now implements the Milestone 0 document-upload vertical slice.
- `requirements.txt` defines the runtime and test dependencies; `.env.example` documents configuration.
- PostgreSQL is the application database, configured with `DATABASE_URL`; tests use an injected SQLite database.
- Local uploads are stored under `data/uploads` by default and are ignored by Git.

## Completed in Milestone 0: Foundation

- Added `GET /health`.
- Added `POST /documents`, which accepts a PDF multipart upload, validates filename/content type/PDF signature, stores the file locally, and creates a `documents` metadata record.
- Added document statuses: `uploaded` and `failed`. A storage failure marks its metadata record as `failed` and returns a clear server error.
- Added focused API, repository integration, and failure-path tests.
- Added structured application logging configuration.

## Completed in Milestone 1: PDF Text Processing

- Added `processing` and `processed` document statuses.
- Added `POST /documents/{document_id}/process` for explicit synchronous PDF text extraction.
- Added `GET /documents/{document_id}` to retrieve a document's current status.
- Added `document_texts`, a one-to-one persistent table for extracted document text.
- Uses `pypdf` to extract embedded PDF text. Corrupt PDFs and PDFs with no extractable text are marked `failed` with a bounded reason.
- Added processing, extraction, persistence, failure-path, and Milestone 0 regression tests.

## Completed in Milestone 2A: Structure-Aware Document Chunking

- Added a persistent `document_chunks` table with chunk IDs, document association, ordering, content, optional page metadata, and timestamps.
- Added `POST /documents/{document_id}/chunks` to generate or rebuild deterministic chunks for a processed document.
- Added `GET /documents/{document_id}/chunks` to inspect chunk order, content, and metadata.
- Chunking splits extracted text by document structure (paragraphs/sentences) instead of arbitrary character boundaries, while preserving overlap for context.
- Added focused chunk-persistence, ordering, metadata, deterministic-output, and invalid-state regression tests.

## Completed in Milestone 2B: Document Embeddings

- Added a persistent `document_embeddings` table associated with chunks by foreign key and uniquely keyed by chunk and model name, allowing future model versions to coexist.
- Added an application-facing `EmbeddingProvider` protocol and a local `sentence-transformers` adapter using `all-MiniLM-L6-v2` by default.
- Added `POST /documents/{document_id}/embeddings`, with `?rebuild=true` for explicit replacement, and `GET /documents/{document_id}/embeddings` for vector-free status summaries.
- Embedding vectors and dimensions are validated before a single persistence transaction; provider and persistence failures do not create partial embedding rows.
- Added deterministic fake-provider tests for generation, metadata, idempotency, rebuild, validation, and failure paths.

## Explicitly not implemented

- PrepWise product features or user workflows.
- AI prompts, hosted model calls, or API keys.
- OCR, scanned-document/image processing, RAG, LLM calls, question generation, adaptive learning, authentication, queues, frontend, Docker, deployment, migrations, or CI.

## Known constraints and next inputs

To run locally: create a PostgreSQL database, copy `.env.example` to `.env` and set `DATABASE_URL`, install `requirements.txt`, then run `uvicorn app.main:app --reload`. The first use of the default embedding provider downloads `all-MiniLM-L6-v2` through `sentence-transformers`; subsequent uses use the local cache. Tables are created at application startup for this foundation milestone.

The next milestone should define a separate approved retrieval scope. Similarity search, reranking, RAG, and other retrieval behavior are not implemented here.
