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

## ADR-013: Add Alembic and an assessment persistence foundation

- Status: Accepted
- Date: 2026-09-10

### Context

M3A needs durable assessment records while preserving the existing M0-M2 document and RAG behavior. The application previously relied on startup `create_all()` and had no migration history.

### Decision

Add Alembic with an initial baseline migration representing the current document/text/chunk/embedding schema and the M3A assessment tables. Keep startup `create_all()` temporarily for compatibility. Existing databases are adopted with `alembic stamp head`; new databases use `alembic upgrade head`.

Persist generation runs, versioned MCQ questions, normalized options, and chunk-linked citation snapshots. Use strict application enums for question type, difficulty, and Bloom level. Repository batch persistence commits only after all child records are flushed successfully.

### Rationale

Alembic provides an explicit, reviewable schema history without rewriting existing data. Separate models and schemas preserve the domain/API boundary. Citation snapshots retain the exact evidence used, and atomic batches prevent incomplete assessment sets.

### Consequences

M3A adds migration tooling and assessment persistence but no question-generation provider or learner workflow. A future migration should define chunk replacement/staleness behavior for questions that cite rebuilt chunks. The initial migration must not be run as an upgrade against an already-created M0-M2 database; stamp it after schema review.

## ADR-014: Generate grounded MCQs through the existing LLM provider port

- Status: Accepted
- Date: 2026-09-10

### Context

M3B needs structured MCQ generation grounded in M2 retrieval results without coupling the application service to Gemini or duplicating retrieval logic.

### Decision

Extend the existing `LLMProvider` with a structured JSON generation capability. `AssessmentGenerationService` owns request validation, document state checks, retrieval/reranking orchestration, prompt construction, Pydantic parsing, deterministic validation, trusted citation mapping, and atomic M3A persistence. Gemini remains the only configured implementation. The model receives only application chunk IDs and source text, and returned IDs must belong to the reranked candidate set.

### Rationale

This keeps future Groq/GLM adapters behind the same provider boundary, preserves one retrieval pipeline, and ensures model output cannot manufacture source metadata or bypass deterministic policy checks.

### Consequences

M3B provides administrative generation/read APIs and includes correct answers in those responses. Semantic distractor quality and Bloom classification remain imperfect model-dependent concerns. Learner-facing schemas, submissions, scoring, adaptive sequencing, and additional providers remain future work.

## ADR-015: Retry transient Gemini transport failures in the provider adapter

- Status: Accepted
- Date: 2026-09-10

### Context

Real Gemini smoke tests can receive temporary 429 or 5xx responses and connection failures. Retrying in the assessment service would couple application behavior to Gemini and would also leave `/ask` without the same resilience.

### Decision

Implement bounded exponential retry handling inside `GeminiLLMProvider`. Retry HTTP 429, 500, 502, 503, and 504, plus narrowly recognized connection reset/refusal/abort/timeout failures. Do not retry authentication, malformed-request, unsupported-model, other permanent 4xx, or malformed JSON failures. Configure retry count and base delay through `LLM_MAX_RETRIES` and `LLM_RETRY_BASE_DELAY_SECONDS`, with provider-side upper bounds.

### Consequences

Both `/ask` and structured M3B generation inherit the same provider resilience without changing their abstractions. A request can take longer than one timeout because each bounded attempt has its own timeout plus backoff delay; the default is two retries with 0.5 and 1.0 second delays.

## ADR-016: Add deterministic assessment quality as the M3C-1 authority boundary

- Status: Accepted
- Date: 2026-09-10

### Context

M3B validates structured MCQ shape and trusted citation membership, but it does not provide batch duplicate detection, transparent quality findings, or a deterministic score before persistence.

### Decision

Add a standalone deterministic assessment quality validator after structured parsing and citation validation and before repository persistence. It validates structural completeness, normalized option uniqueness, citation trust, forbidden patterns, and exact/near-duplicate stems. It returns structured hard failures, warnings, deterministic scores, and a decision. Add a bounded `RegenerationDecisionPolicy` abstraction, but do not invoke regeneration or another LLM call in M3C-1.

### Rationale

Deterministic rules must remain authoritative for malformed or untrusted assessment content. Keeping this layer independent of Gemini, retrieval, routes, and SQLAlchemy makes it testable and prevents a future semantic evaluator from silently becoming the source of truth.

### Consequences

No schema changes are needed for M3C-1 because quality results are internal and accepted questions continue using the existing atomic persistence path. The score measures only deterministic structural/provenance completeness, not semantic correctness. LLM evaluation, attempt tracking, regeneration loops, quality persistence, and advanced coverage/diversity are deferred to later M3C stages.

## ADR-017: Persist bounded per-slot regeneration attempts

- Status: Accepted
- Date: 2026-09-10

### Context

M3C-2 must recover from individual deterministic quality failures without discarding valid questions or allowing unbounded LLM calls.

### Decision

Create an `assessment_question_attempts` audit record for every generated candidate, uniquely keyed by generation run, question slot, and attempt number. Reuse `RegenerationDecisionPolicy` to bound regeneration per slot. Persist accepted questions only after every slot passes M3C-1, in the existing final batch transaction; link accepted attempt records to those questions within that transaction.

### Rationale

Per-slot records preserve rejected candidates and deterministic failure codes while partial regeneration avoids replacing already accepted questions. Keeping the final assessment batch atomic prevents an exhausted run from appearing as a successful partial assessment.

### Consequences

Attempt records and failed/exhausted runs intentionally persist for auditability, even though their final questions do not. Generation remains synchronous and uses the existing provider retry behavior; quality dashboards remain out of scope.

## ADR-018: Use optional secondary semantic evaluation after deterministic validation

- Status: Accepted
- Date: 2026-09-10

### Context

M3C-1 can prove structural and provenance properties but cannot reliably judge whether source evidence supports a selected answer, distractors are plausible, or a question matches intended cognitive difficulty.

### Decision

Add a replaceable evaluator protocol with a Gemini structured-output adapter, disabled by default. Evaluate only candidates that have passed M3C-1, using bounded text from application-owned cited chunks. Require a strict evaluator schema, a `pass` recommendation, and a configurable overall-score threshold before the existing M3C-2 decision path accepts a candidate.

### Rationale

This adds a useful secondary quality signal without making an LLM the source of truth for deterministic rules or source metadata. Reusing the existing provider transport retains one bounded retry implementation.

### Consequences

Semantic rejection uses the existing bounded per-slot regeneration policy. Evaluator errors fail safely and retain an audit attempt, while durable evaluator-result history and quality APIs are deliberately deferred to M3C-4.

## ADR-019: Persist assessment quality evaluations and expose read APIs

- Status: Accepted
- Date: 2026-09-10

### Context

M3C-3 introduced optional semantic evaluation but kept evaluation results in memory only. Operators and tests need durable history tied to generation attempts without changing M3C-1 authority or M3C-2 regeneration behavior.

### Decision

Add immutable `assessment_quality_evaluations` rows keyed to generation runs and question attempts, with at most one deterministic and one semantic record per attempt. Persist evaluations during generation using the same independent-commit audit pattern as attempt records. Link accepted evaluation rows to final questions inside the existing final batch transaction. Expose administrative read APIs for question-scoped and run-scoped history.

When semantic evaluation fails after deterministic validation, persist the deterministic evaluation with status `evaluation_failed` before failing the run.

### Rationale

This completes the M3C audit trail without making stored scores authoritative for acceptance. Separate read APIs keep existing generation and question responses unchanged while making evaluation history inspectable.

### Consequences

Evaluation records may exist for runs that never produce final questions. Deterministic records store hard-failure codes rather than full warning lists, and semantic records store dimension scores rather than per-dimension reasons. Quality dashboards, coverage/diversity analysis, and learner-facing workflows remain out of scope.
