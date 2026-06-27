---
name: python-backend
description: Use when writing, reviewing, or refactoring Python backend code. Use when adding new layers, services, clients, domain entities, or utilities. Triggers on phrases like "backend", "service layer", "domain", "infrastructure", "API layer", "dependency injection", or "client".
---

# Python Backend

## Overview

Four-layer architecture (api → services → infrastructure → domain) plus a cross-cutting utils layer. Each layer has strict import rules and responsibilities. Fail fast, type strictly, no loose `Any`/`dict`/`tuple` types.

## Linting

```bash
uv run poe lint
```

Run after every change.

## Typing Rules

- **Never** type as `Any`, `dict`, or `tuple` — use concrete types or named models
- `| None` (Optional) only when genuinely necessary
- Fail fast and close to the root cause
- **Never** add `from __future__ import annotations` — only include it when a forward reference cannot be resolved with a quoted string literal or by reordering definitions

## Layer Architecture

```
api  →  services  →  infrastructure  →  domain
                                      ↑
                             utils (cross-cutting)
```

### Domain Layer

- Imports: itself + utils only
- Contains: domain entities, validation rules/invariants, business logic, **interfaces** for external communication
- Interfaces defined here are injected with Infrastructure concrete implementations throughout the codebase

### Infrastructure Layer

- Imports: itself + domain + utils
- Contains: concrete client implementations for Domain interfaces; fake/stub clients for testing
- Clients are never manually instantiated — only wired via DI config
- Contains: Infrastructure entities (data retrieved by clients), DI configuration
- Every real client has a matching fake client (unless a fake is genuinely not needed)

### Services Layer

- Imports: itself + infrastructure + domain + utils
- Contains: Service classes that orchestrate Infrastructure and Domain code
- May contain Pydantic models for method signatures and validation
- **No interfaces/abstract classes**
- Output to API layer must use Service or Domain models — never expose Infrastructure entities directly

### API Layer

- Imports: itself + services + utils
- Contains: API/server code and payload validation only
- **No business logic** beyond API validation
- **No try/except**

### Utils Layer

- Contains: custom exceptions, logging, profiling, environment variable settings, other cross-cutting concerns

## Quick Reference

| Rule | Layer |
|------|-------|
| No `try/except` | API |
| No interfaces/abstract classes | Services |
| Clients never manually instantiated | Infrastructure |
| Fake client per real client | Infrastructure |
| Never return Infrastructure models to API | Services |
| All interfaces live here | Domain |
| Env vars, logging, exceptions | Utils |

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Returning Infrastructure entity from Service | Map to Service/Domain model first |
| Adding business logic in API route handler | Move to Service |
| Manually instantiating a client | Wire through DI config |
| Typing a parameter as `dict` | Define a Pydantic model or TypedDict |
| Services importing from API | Invert the dependency |
| Missing fake client for a real client | Add `FakeXxxClient` in Infrastructure |
