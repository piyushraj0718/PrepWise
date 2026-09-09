# PrepWise Contribution Guide

PrepWise is being built incrementally as a production-style, learning-oriented GenAI application. Preserve that balance: code should be explicit, small, testable, and easy for a student maintainer to trace.

## Working rules

- Read the relevant documents in `docs/` before changing architecture or beginning a milestone.
- Work on one approved milestone at a time. Do not build future features early.
- Do not add a dependency unless the active milestone needs it and the reason is recorded in `docs/DECISIONS.md`.
- Keep business rules deterministic and independent of AI providers, HTTP handlers, databases, and UI code.
- Treat model output as untrusted input. Validate it at a boundary before it affects application state or user-visible decisions.
- Prefer standard-library, type-annotated Python and plain functions/classes over frameworks or abstraction layers that do not yet earn their cost.
- Keep configuration outside source code. Never commit secrets, tokens, or personal data.
- Modify only files needed for the active task. Do not reformat or reorganize unrelated work.

## Intended module boundaries

When implementation begins, organize code so these areas can be tested in isolation:

- `domain`: entities, invariants, and deterministic business rules.
- `application`: use cases that orchestrate domain behavior through interfaces.
- `ai`: provider adapters, prompt assets, output schemas, and validation.
- `infrastructure`: persistence and external-service implementations.
- `interfaces`: CLI, web/API, or other delivery adapters.

Dependencies should point inward: interfaces and infrastructure may depend on application/domain contracts; domain must not depend on them.

## Verification expectations

- Add or update focused tests with each implemented subsystem.
- Keep AI-dependent behavior behind a replaceable interface and use fakes in ordinary tests.
- Run the smallest relevant test/check command before handoff and report what was run.
- Update `docs/CURRENT_STATE.md` after a milestone materially changes the repository.
- Record decisions with lasting architectural impact in `docs/DECISIONS.md` before or alongside the change.

## Change hygiene

- Use clear, narrow commits when commits are requested.
- Explain assumptions and unresolved choices in documentation rather than silently guessing.
- If a requirement conflicts with the current architecture, pause and document options before making a breaking change.
