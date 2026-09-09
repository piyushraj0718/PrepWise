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

| Area | Responsibility | Must not contain |
| --- | --- | --- |
| Domain | Entities, value objects, policies, validation, deterministic rules | Provider calls, database queries, HTTP/UI concerns |
| Application | Use-case orchestration and transaction boundaries | Framework-specific request handling or prompt-provider details |
| AI | Prompt assets, provider calls, structured-output parsing, safety/format validation | Core policy decisions hidden in prompts |
| Infrastructure | Database, file, queue, and third-party implementations | Business-rule ownership |
| Interfaces | Request parsing, presentation, authentication integration, response mapping | Direct implementation of business rules |

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
