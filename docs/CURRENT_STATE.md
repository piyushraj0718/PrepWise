# Current Project State

Last reviewed: 2026-09-10 (updated for M4B)

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

## Completed in Milestone 3C-1: Deterministic Assessment Quality

- Added authoritative deterministic MCQ quality results for question and batch validation before persistence.
- Added normalized option/stem comparison, configurable near-duplicate detection, structured failure findings, warnings, and transparent deterministic scoring.
- Added a tested accept/regenerate/reject decision policy without regeneration loops or additional LLM calls.
- Preserved application-owned citation validation and atomic M3B persistence; no database schema changes were required.
- Semantic evaluation, quality persistence, topic coverage, advanced diversity, and regeneration attempts remain out of scope.

## Completed in Milestone 3C-2: Bounded Question Regeneration

- Added persistent, per-slot attempt records containing the attempt number, decision status, structured candidate payload, deterministic failure codes, and an optional link to the final accepted question.
- Generation creates a run in `generating` status before calls to the LLM. M3C-1 remains the authoritative quality gate for every candidate.
- Only slots rejected by M3C-1 are regenerated. Accepted slots are retained and included in duplicate checks for later candidates.
- Regeneration is controlled solely by `RegenerationDecisionPolicy` and its configured maximum attempts. Exhaustion leaves no final assessment questions, marks the run `exhausted`, and retains the complete audit trail.
- Existing M3B API routes and successful response schemas remain unchanged.

## Completed in Milestone 3C-3: Secondary Semantic Evaluation

- Added an optional `AssessmentQuestionEvaluator` protocol and Gemini implementation with strict structured Pydantic output for groundedness, correctness, distractors, explanation, difficulty, Bloom alignment, and an overall recommendation.
- M3C-1 remains authoritative: deterministic failures skip semantic evaluation entirely. Only deterministic candidates receive bounded, application-owned cited-chunk evidence for semantic review.
- Semantic recommendation plus a configurable threshold determines whether a deterministic candidate is accepted or returns to the existing M3C-2 per-slot regeneration policy.
- Malformed, unavailable, or failed evaluator responses never accept a question. They create an `evaluation_failed` attempt and fail the generation run without persisting final questions.
- Semantic evaluation is disabled by default and configured through environment-backed evaluator enablement, model, timeout, and threshold settings.
- Added optional manual `scripts/smoke_semantic_evaluation.py`; it requires a configured Gemini key and explicitly enabled evaluator, and is not part of pytest.

## Completed in Milestone 3C-4: Quality Evaluation History and Read APIs

- Added persistent `assessment_quality_evaluations` records for deterministic and optional semantic evaluation outcomes tied to generation runs and question attempts.
- Generation writes one deterministic evaluation per attempt and, when semantic evaluation is enabled, one semantic evaluation per evaluated attempt. Accepted attempts link evaluation records to final questions during the existing batch transaction.
- Evaluator failures after deterministic validation persist the deterministic evaluation with status `evaluation_failed` before failing the run.
- Added administrative read APIs at `GET /questions/{question_id}/quality` and `GET /question-generation-runs/{run_id}/quality`.
- Added Alembic migration `0003_assessment_quality_evaluations.py` and focused persistence/API tests.

## Known constraints and next inputs

To run locally: create a PostgreSQL database, copy `.env.example` to `.env` and set `DATABASE_URL`, install `requirements.txt`, then run `uvicorn app.main:app --reload`. The first use of the default embedding provider downloads `all-MiniLM-L6-v2` through `sentence-transformers`; subsequent uses use the local cache. Tables are created at application startup for this foundation milestone.

The next milestone should define the next approved assessment-quality capability. Correct answers are currently included in the administrative generation response; answer submission, scoring, quiz sessions, adaptive selection, and authentication are not implemented.

## Completed in Milestone 4A: Quiz Session Foundation

- Added `LearnerQuizSession` SQLAlchemy model and `learner_quiz_sessions` table with Alembic migration `0004_learner_quiz_sessions`.
- Added `LearnerRepository` with `create_session` and `get_session`.
- Added learner-facing Pydantic schemas: `QuizSessionCreateRequest`, `QuizSessionResponse`, `LearnerQuestionResponse` (omits `correct_option_key` and citations), `QuizSessionQuestionsResponse`.
- Added `POST /quiz-sessions`, `GET /quiz-sessions/{session_id}`, and `GET /quiz-sessions/{session_id}/questions`.
- `GET /quiz-sessions/{session_id}/questions` returns questions in creation order and deliberately withholds `correct_option_key` and citations.
- Added focused schema, repository, and API tests.

## Completed in Milestone 4B: Answer Submission and Deterministic MCQ Scoring

- Added `LearnerAnswerSubmission` SQLAlchemy model (already present in M4A model file) and Alembic migration `0005_learner_answer_submissions` creating the table with a unique constraint on `(session_id, question_id)`.
- Added `score_mcq_answer(submitted_key, correct_key) -> bool` as a pure deterministic domain function in `app/domain/assessment.py`; it has no I/O dependencies and is the sole authority for MCQ correctness.
- Extended `LearnerRepository` with `submit_answer`, `get_submission`, `get_submissions_for_session`, and `complete_session`. `submit_answer` raises `DuplicateSubmissionError` on a unique-constraint violation.
- Added learner-facing schemas `AnswerSubmissionRequest`, `AnswerSubmissionResponse`, and `SessionResultResponse`.
- Added three API routes:
  - `POST /quiz-sessions/{session_id}/answers` — validates the session is active and the question belongs to the session, scores with `score_mcq_answer`, persists the submission; returns 201. Returns 404 for unknown session/question, 422 for inactive session, 409 for duplicate submission.
  - `GET /quiz-sessions/{session_id}/answers/{question_id}` — retrieves a single submission or 404.
  - `POST /quiz-sessions/{session_id}/complete` — marks the session `completed`, computes `correct_count` and `score_percent` from all submissions, returns `SessionResultResponse`; returns 409 if already completed.
- Added focused domain, schema, repository, and API tests in `tests/test_answer_submission.py`.
- Adaptive selection, authentication, and learner history remain out of scope.

## Completed in Milestone 4C: Performance Accumulation and Weak-Topic Detection

- Added deterministic domain functions in `app/domain/assessment.py`:
  - `compute_accuracy(attempts, correct) -> float` — pure, no I/O.
  - `detect_weak_areas(performances, min_attempts, accuracy_threshold) -> list[WeakArea]` — pure, no LLM calls.
  - `AreaPerformance` and `WeakArea` frozen dataclasses.
  - Module-level constants `WEAK_TOPIC_MIN_ATTEMPTS = 3` and `WEAK_TOPIC_ACCURACY_THRESHOLD = 0.60`.
- Extended `LearnerRepository` with three read methods that aggregate directly from persisted submissions:
  - `get_all_submissions_for_learner` — all submissions across all sessions for a learner.
  - `get_topic_performance_for_learner` — per-topic attempt/correct counts via JOIN to `assessment_questions`.
  - `get_skill_performance_for_learner` — per-skill equivalent.
- Added learner performance schemas: `AreaPerformanceResponse`, `LearnerPerformanceResponse`, `WeakAreaResponse`, `LearnerWeakTopicsResponse`.
- Added two read-only API endpoints:
  - `GET /learners/{learner_id}/performance` — overall accuracy, per-topic, and per-skill breakdowns; returns empty aggregates for unknown learners.
  - `GET /learners/{learner_id}/weak-topics` — topics and skills identified as weak, with the thresholds used in the response.
- No new tables, migrations, or external services.
- Added focused domain, repository, and API tests in `tests/test_learner_performance.py`.
- Adaptive selection, authentication, learner history, dashboards, and LLM-assisted diagnosis remain out of scope.
