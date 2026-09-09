# PrepWise Project Context

## Purpose

PrepWise is planned as a GenAI-assisted application for learning and preparation. The repository currently contains only the engineering foundation; product workflows, user experience, data model, and AI capabilities have not been implemented or finalized.

## Product principles

- Build iteratively so each milestone is understandable to a student developer.
- Use AI where it provides clear value, while keeping important product rules explicit and testable.
- Prefer reliable, inspectable behavior over impressive but opaque automation.
- Protect user data and avoid placing secrets or sensitive data in source control.
- Make it possible to swap or fake external services during testing.

## Scope for the current setup milestone

This milestone establishes project guidance and architecture documentation only. It does not select a runtime framework, install dependencies, create application code, define production prompts, or implement PrepWise features.

## Open product questions

- Who are the first target users and what preparation outcome matters most to them?
- Which initial workflow should be the first vertical slice?
- Which user data is necessary, and what retention/privacy expectations apply?
- Which AI provider, model, latency, quality, and cost constraints apply?
- Will the first interface be a CLI, web application, API, or another surface?

Answers should be captured before they become implementation assumptions.
