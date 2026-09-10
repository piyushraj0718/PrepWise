# Current Project State

Last reviewed: 2026-09-10

## Repository inventory

- A FastAPI application now implements the Milestone 0 document-upload vertical slice.
- `requirements.txt` defines the runtime and test dependencies; `.env.example` documents configuration.
- PostgreSQL is the application database, configured with `DATABASE_URL`; tests use an injected SQLite database.
- Local uploads are stored under `data/uploads` by default and are ignored by Git.
- Alembic is configured with a baseline migration for the current schema. Startup table creation remains for compatibility while migration adoption is phased in.

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

## Completed in Milestone 2C: Vector Similarity Retrieval

- Added `POST /search` for embedding a query and returning ranked chunk results.
- Retrieval uses the existing embedding provider and JSON-stored vectors with deterministic cosine similarity.
- Search supports all processed documents or an optional `document_id` filter, includes similarity scores, and preserves chunk text, order, and page metadata.
- Added validation for empty queries, bounded `top_k`, invalid document IDs, malformed or missing embeddings, zero-magnitude vectors, and provider failures.
- Added deterministic fake-provider tests for ranking, filtering, empty results, response structure, validation, and failure handling.

## Completed in Milestone 2D: Hybrid BM25 + Dense Retrieval

- Extended `POST /search` to combine the existing dense cosine ranking with local BM25 ranking over persisted chunk text.
- Uses reciprocal-rank fusion with a constant of 60, positive lexical matches only, and deterministic score/metadata tie-breakers.
- Search preserves chunk IDs, document IDs, text, page metadata, dense similarity, BM25 score, and hybrid score.
- Supports query, bounded `top_k`, optional document filtering, lexical-only matches, and explicit handling for missing documents, malformed embeddings, empty queries, and provider failures.
- Added deterministic tests for BM25 ranking, dense ranking, hybrid promotion, filtering, top-k, empty results, malformed vectors, validation, and provider failure.

## Completed in Milestone 2E: Retrieval Reranking

- Added a replaceable reranker provider protocol and local Sentence Transformers cross-encoder adapter.
- Extended `POST /search` with optional `candidate_k`; hybrid retrieval supplies the candidate set and reranking returns the final `top_k`.
- The default `cross-encoder/ms-marco-MiniLM-L-6-v2` model loads lazily and is never required by the deterministic test suite.
- Search results preserve dense, BM25, and hybrid scores plus chunk metadata and add `reranker_score` and `final_rank`.
- Added deterministic fake-reranker tests for reorder behavior, score preservation, candidate limits, filtering, empty candidates, malformed data, validation, and provider failures.

## Completed in Milestone 2F: Grounded RAG

- Added a replaceable `LLMProvider` abstraction and a Gemini REST provider using environment-backed API configuration.
- Added centralized grounding instructions and `POST /ask`, which composes hybrid retrieval, reranking, context selection, and answer generation.
- LLM output is validated and citations are mapped only from retrieved chunk IDs, preserving source text, document/chunk metadata, and retrieval/reranking scores.
- Added deterministic no-context fallback and explicit handling for empty queries, missing documents, missing API configuration, provider failures, timeouts, and malformed responses.
- Added deterministic fake-LLM tests for grounded answers, citation mapping, document filtering, no context, failures, malformed output, and full retrieval-to-generation integration.
- Added optional manual smoke path at `scripts/smoke_rag.py`; it is not collected by pytest.

## Explicitly not implemented

- PrepWise product features or user workflows.
- AI prompts, hosted model calls, or API keys.
- OCR, scanned-document/image processing, question generation, adaptive learning, authentication, queues, frontend, Docker, deployment, and CI.

## Completed in Milestone 3A: Assessment Foundation

- Added Python/Pydantic assessment enums for MCQ, difficulty, and Bloom's taxonomy.
- Added SQLAlchemy models for generation runs, questions, MCQ options, and question citations.
- Added repository methods for retrieval and atomic generation-batch persistence, including source snapshots and retrieval metadata.
- Added separate Pydantic request/response schemas with strict enum and count validation.
- Added Alembic configuration and an initial migration representing the existing document schema plus assessment tables.
- Question generation, structured LLM output, citation validation during generation, quiz sessions, scoring, and adaptive selection remain out of scope.

## Completed in Milestone 3B: Question Generation Engine

- Added structured generation support to the existing replaceable LLM provider abstraction.
- Added grounded MCQ generation over the existing hybrid retrieval and reranking pipeline.
- Added strict Pydantic output parsing and deterministic validation for MCQ structure, duplicate stems, citation IDs, citation markup, and forbidden option patterns.
- Added atomic persistence and API endpoints for generation, question retrieval/listing, and generation-run retrieval.
- Added an optional Gemini smoke script at `scripts/smoke_questions.py`; ordinary tests use fake providers and remain offline.
- Added bounded Gemini retries for transient rate-limit, server, and connection failures with environment-backed settings; permanent errors are not retried.

## Known constraints and next inputs

To run locally: create a PostgreSQL database, copy `.env.example` to `.env` and set `DATABASE_URL`, install `requirements.txt`, then run `uvicorn app.main:app --reload`. The first use of the default embedding provider downloads `all-MiniLM-L6-v2` through `sentence-transformers`; subsequent uses use the local cache. Tables are created at application startup for this foundation milestone.

The next milestone should define learner-facing assessment workflows. Correct answers are currently included in the administrative generation response; answer submission, scoring, quiz sessions, adaptive selection, and authentication are not implemented.
