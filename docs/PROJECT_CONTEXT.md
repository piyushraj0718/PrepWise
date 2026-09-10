# PrepWise Project Context

## Purpose

PrepWise is a GenAI-assisted application for learning and preparation. The repository now contains document ingestion, retrieval/RAG, the M3A assessment persistence foundation, and the M3B grounded MCQ generation engine; learner workflows remain a separate future slice.

## Product principles

- Build iteratively so each milestone is understandable to a student developer.
- Use AI where it provides clear value, while keeping important product rules explicit and testable.
- Prefer reliable, inspectable behavior over impressive but opaque automation.
- Protect user data and avoid placing secrets or sensitive data in source control.
- Make it possible to swap or fake external services during testing.

## Scope for the current project state

The current implementation uses FastAPI, SQLAlchemy, PostgreSQL configuration, local document storage, replaceable retrieval/reranking/LLM ports, and Alembic migrations. M3A defines persisted assessment runs, MCQ questions/options, citations with source snapshots, controlled learning metadata, and atomic batch boundaries. M3B adds grounded structured MCQ generation and administrative question APIs. M3C adds an authoritative deterministic quality gate, audited per-slot attempts, bounded partial regeneration, and an optional secondary semantic evaluator before final persistence. Durable evaluation history, learner-facing answers, scoring, sessions, and adaptive selection remain out of scope.

## Open product questions

- Who are the first target users and what preparation outcome matters most to them?
- Which initial workflow should be the first vertical slice?
- Which user data is necessary, and what retention/privacy expectations apply?
- Which AI provider, model, latency, quality, and cost constraints apply?
- Will the first interface be a CLI, web application, API, or another surface?

Answers should be captured before they become implementation assumptions.
