# Architecture

## Status

This is a target architecture, not implemented code. It defines boundaries for future milestones without prematurely choosing a web framework, database, AI provider, or deployment platform.

## Architectural approach

PrepWise will use a modular, layered Python architecture. A feature should be built as a thin vertical slice, with deterministic business behavior separated from AI and delivery concerns.

```text
User interface / API / CLI
          |
          v
Application use cases
     |              |
     v              v
Domain rules     Ports (interfaces)
                     |
          ------------------------
          |                      |
          v                      v
   AI provider adapter     Persistence/external adapters
```

## Responsibilities

| Area           | Responsibility                                                                     | Must not contain                                               |
| -------------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Domain         | Entities, value objects, policies, validation, deterministic rules                 | Provider calls, database queries, HTTP/UI concerns             |
| Application    | Use-case orchestration and transaction boundaries                                  | Framework-specific request handling or prompt-provider details |
| AI             | Prompt assets, provider calls, structured-output parsing, safety/format validation | Core policy decisions hidden in prompts                        |
| Infrastructure | Database, file, queue, and third-party implementations                             | Business-rule ownership                                        |
| Interfaces     | Request parsing, presentation, authentication integration, response mapping        | Direct implementation of business rules                        |

## AI design rules

- AI interactions must be exposed through an application-facing port.
- Prompts and expected output shapes must be versioned and tested where practical.
- Validate structured outputs before use; malformed or unsupported output must have an explicit failure path.
- Preserve enough metadata to debug an AI result without logging secrets or sensitive user content unnecessarily.
- Do not let an AI response be the sole source of truth for deterministic policy or authorization decisions.

## Test strategy

- Unit-test domain rules without I/O or model calls.
- Unit-test application use cases using fake ports.
- Contract-test adapters at their boundaries when external integrations are introduced.
- Add end-to-end tests only for completed user-facing slices.

## Growth rule

Create a module or interface when an active use case needs it. Do not create empty layers, repositories, services, or packages solely to anticipate future features.

## Retrieval implementation

The M2C retrieval service uses the existing `EmbeddingProvider` to embed each query and scans persisted `document_embeddings` rows joined to their chunks. It calculates cosine similarity in Python with finite-value and zero-norm checks, sorts deterministically by score and chunk metadata, and applies `top_k` after optional `document_id` filtering. JSON vector storage remains compatible with PostgreSQL and SQLite tests; pgvector and a separate vector database are deferred.

M2D extends the same search service with local BM25 scoring over processed `document_chunks`. BM25 tokenizes case-insensitively using Unicode word characters and uses `k1=1.5` and `b=0.75`. Only positive lexical matches receive a lexical rank. Dense and lexical rankings are combined with reciprocal-rank fusion:

```text
hybrid_score = sum(1 / (60 + rank))
```

The final order is descending `hybrid_score`, then descending BM25 score, document ID, and chunk index. Results retain dense `similarity`, `bm25_score`, `hybrid_score`, chunk text, IDs, order, and page metadata. This remains a linear local scan; no external search service or separate index is introduced.

M2E adds a replaceable reranking port after hybrid retrieval. The service requests `candidate_k` hybrid candidates, sends the original query and candidate chunk text to the reranker, and returns the requested final `top_k` after sorting by reranker score with deterministic hybrid and metadata tie-breakers. Results additionally expose `reranker_score` and `final_rank` while preserving all M2D fields.

The default local adapter uses the Sentence Transformers cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`, selected as a compact, general semantic relevance model for query-passage pairs. It loads lazily on the first non-empty reranking request, so application import and ordinary tests do not download or initialize the model. The model name is configured through `RERANKER_MODEL_NAME` / `reranker_model_name`.

M2F adds `POST /ask` as a separate grounded-answer use case. It runs the existing retrieval and reranking services, formats the reranked chunks into context, and passes that context plus the original query to a replaceable `LLMProvider`. The centralized grounding prompt requires context-only answers, explicit uncertainty when context is insufficient, and chunk-ID citations. The application validates every returned citation against the supplied candidate IDs and maps valid IDs back to authoritative chunk metadata and retrieval scores; the model cannot create source metadata.

The first real provider is Gemini's REST `generateContent` API, configured with `GEMINI_API_KEY`, `GEMINI_MODEL_NAME`, `GEMINI_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES`, and `LLM_RETRY_BASE_DELAY_SECONDS`. The provider requests JSON containing an answer and chunk-ID list, uses standard-library HTTP, and never loads or calls during application import. Bounded retries cover HTTP 429 and 5xx responses plus narrowly recognized connection failures; permanent 4xx responses and malformed JSON fail immediately. No-context requests return a deterministic fallback without calling the LLM. An optional manual smoke path is `python scripts/smoke_rag.py "your question"` after configuring the environment and preparing documents; it prints answers and chunk IDs only, never credentials.

## M3A assessment foundation

M3A introduces assessment persistence without implementing question generation. A generation run stores request and model metadata, questions store versioned MCQ content and learning metadata, options store normalized choices, and citations link questions to existing document chunks.

The controlled vocabulary is represented by Python/Pydantic enums: `mcq`; `easy`, `medium`, and `hard`; and the six Bloom levels from `remember` through `create`. The schema supports only MCQ until another question type is approved.

Question citations have a foreign key to `document_chunks` and store retrieval rank, reranker/hybrid scores, and source text/page snapshots. A future generation service must validate cited IDs against its retrieved candidate set before persistence; the model must never invent source metadata.

M3A adds Alembic with a baseline migration containing the current M0-M2 schema and assessment tables. Startup `create_all()` remains temporarily compatible with existing tests and deployments. New databases should use `alembic upgrade head`; existing M0-M2 databases should be reviewed and then marked with `alembic stamp head`, which does not modify data.

Batch persistence is designed as one transaction. Repository methods flush child records, while the batch method commits only after the run, questions, options, and citations are prepared successfully. Prompt/content versions and request/model metadata support later reproducibility; adaptive learning, sessions, scoring, authentication, and LLM question generation remain outside M3A.

## M3B question generation engine

M3B adds `AssessmentGenerationService` as the application use case. It accepts the existing M3A request schema, verifies an optional document is processed, calls the existing hybrid retrieval and reranking service, formats only application-owned chunk IDs and source text into a bounded prompt, and invokes structured generation through the existing `LLMProvider` capability. Gemini is the only configured adapter; future providers implement the same structured-generation method without changing the service.

The structured output is parsed through strict Pydantic models. Deterministic validation requires exactly four unique options, one existing correct option key, non-empty content, citations, unique question stems, retrieved citation IDs, no citation markup, and no all/none-of-the-above options. Semantic distractor quality remains a model-quality concern and is not falsely treated as provable by deterministic checks. Valid citations are mapped to reranked results and persisted with source snapshots and retrieval metadata.

M3B exposes `POST /questions/generate`, `GET /questions/{id}`, `GET /questions`, and `GET /question-generation-runs/{run_id}`. Generation responses are administrative/development responses and include the correct answer; a future learner-facing schema must omit it. The optional `scripts/smoke_questions.py` path uses configured Gemini settings and never prints credentials.

## M3C-1 deterministic assessment quality

M3C-1 adds a provider-independent deterministic quality boundary after M3B structured parsing and citation membership validation, before question values are passed to the atomic assessment repository. The quality layer receives generated MCQs and the application-owned retrieved chunk ID set; it does not call Gemini, retrieval, reranking, FastAPI, or SQLAlchemy.

Hard failures are authoritative and prevent persistence: invalid option count, missing/duplicate content, invalid correct key, invalid difficulty/Bloom values, invalid or duplicate citations, citation markup, forbidden all/none-of-the-above options, and duplicate or near-duplicate batch stems. Warnings are advisory only, such as unusually short but non-empty text. Semantic correctness, distractor quality, explanation entailment, and actual difficulty/Bloom fit are deliberately not evaluated in this stage.

Comparison normalization case-folds text, replaces punctuation with spaces, and collapses whitespace without changing stored question text. Near-duplicates use `difflib.SequenceMatcher` with an explicit configurable threshold, default `0.92`. The deterministic score is a structural completeness score only:

```text
score = structure * 0.25
     + citations * 0.20
     + option_uniqueness * 0.20
     + explanation_presence * 0.15
     + batch_uniqueness * 0.20
```

The score must not be interpreted as semantic or educational correctness. M3C-1 introduced `RegenerationDecisionPolicy`; M3C-2 now uses that policy for bounded partial regeneration.

## M3C-2 attempt tracking and bounded partial regeneration

M3C-2 creates a generation run before LLM calls and records every candidate in `assessment_question_attempts`. An attempt is uniquely identified by generation run, question slot, and attempt number. It retains the candidate payload, deterministic M3C-1 failure codes, a decision status (`accepted`, `rejected`, or `exhausted`), and, for accepted candidates, a link to the final persisted question.

After each LLM response, M3C-1 validates the new candidates together with questions already accepted for the batch. Only rejected slots are requested again; accepted slots remain unchanged. Duplicate and near-duplicate checks therefore cover accepted candidates, siblings in the current response, and previously rejected candidate stems. `RegenerationDecisionPolicy.max_attempts` bounds calls per slot and makes exhaustion deterministic.

Attempt records and failed/exhausted runs commit independently for auditability. Final questions, options, citations, completed-run status, and accepted-attempt links are committed in one existing repository transaction only after every slot is accepted. An exhausted run has no final assessment questions and cannot appear as a successful partial assessment. Provider transport retries remain inside the existing LLM adapter; M3C-2 adds no second retry mechanism.

## M3C-3 secondary semantic evaluation

M3C-3 adds an optional `AssessmentQuestionEvaluator` port. Its initial Gemini adapter reuses the existing structured-provider transport and retry behavior but has a dedicated, versioned evaluation prompt and strict Pydantic result schema. It receives only generated MCQ content, requested constraints, and bounded application-owned text from that question's validated cited chunks; it does not retrieve independently or control citation identity/metadata.

Semantic scores are advisory rather than objective. `SemanticQualityPolicy` requires the evaluator's `pass` recommendation plus the explicit configurable threshold (default `0.75`) for the overall score and every semantic dimension. M3C-1 deterministic hard failures are checked first and always skip evaluator calls. A semantic failure becomes a normal M3C-2 regeneration candidate, so the same per-slot maximum applies. An evaluator failure, malformed result, or unavailable provider records an `evaluation_failed` audit attempt and fails the run; it can never silently accept a question.

## M3C-4 durable quality evaluation history and read APIs

M3C-4 persists immutable quality evaluation records in `assessment_quality_evaluations`, keyed to generation runs and question attempts. Each attempt may have at most one deterministic record and one semantic record. Records commit independently during generation, matching the M3C-2 audit pattern.

Deterministic records store the M3C-1 score, hard-failure codes, recommendation, and attempt status. Semantic records store evaluator provider/model metadata, prompt version, overall and dimension scores, recommendation, rationale, and attempt status. Accepted attempts link their evaluation records to the final persisted question during the existing final batch transaction; rejected, exhausted, and evaluation-failed attempts keep `question_id` null.

When semantic evaluation is enabled but the evaluator fails after deterministic validation, the service still persists the deterministic evaluation with status `evaluation_failed` before failing the run. Semantic evaluation remains disabled by default.

Administrative read APIs expose the stored history without changing generation behavior:

- `GET /questions/{question_id}/quality`
- `GET /question-generation-runs/{run_id}/quality`

Topic coverage, advanced diversity, learner workflows, and quality dashboards remain out of scope.
