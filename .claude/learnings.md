# Classiflow — Design Learnings

Decisions and patterns settled during development that are not obvious from the code alone.
Each entry has a **context**, the **decision**, and the **why**.

---

## Exception style — dataclass subclasses per service

**Context:** `AuditService` needed to surface two distinct failure modes to callers:
missing required fields and a persistence failure from the repository layer.

**Decision:** Each service gets its own `exceptions.py` with:
- A plain base class (e.g. `AuditError(Exception)`)
- `@dataclass` subclasses for each distinct error case, each with typed fields,
  `__post_init__` calling `super().__init__(str(self))`, and `__str__` building the message

```python
from dataclasses import dataclass

class AuditError(Exception): ...

@dataclass
class MissingFieldError(AuditError):
    field: str
    def __post_init__(self) -> None: super().__init__(str(self))
    def __str__(self) -> str: return f"{self.field} is required"

@dataclass
class PersistenceError(AuditError):
    job_id: str
    agent: str
    event: str
    def __post_init__(self) -> None: super().__init__(str(self))
    def __str__(self) -> str:
        return f"Failed to persist for job={self.job_id} agent={self.agent} event={self.event}"
```

**Why:**
- Callers can catch the base class for broad handling or the specific subclass to inspect fields
- `__post_init__` wires `super().__init__` so `str(exc)`, `repr(exc)`, and loguru all work correctly
- Messages live inside the exception class, satisfying ruff rules TRY003 / EM101
- No classmethod factories — those hide the structured data from callers and type checkers

**Do NOT use:**
- Bare `except Exception` — always catch a specific type
- Classmethod factories on a single exception class (e.g. `AuditError.required(...)`)
- `@dataclass` without `__post_init__` calling `super().__init__(str(self))` — `str(exc)` breaks

---

## `__init__` vs `BaseModel` — which to use where

**Context:** Reviewing all `__init__` usages in `src/classiflow/` to check whether any
should be replaced with Pydantic `BaseModel`.

**Decision:** The split is by class role, not by personal preference:

| Role | Pattern | Examples |
|---|---|---|
| Domain / value object | `BaseModel` | `AgentEvent`, `FileReceptionResult`, `User`, `AuthToken` |
| Service / repository | plain `__init__` | `AuditService`, `EventBroadcaster`, `SqlHashRepository`, `InMemoryHashRepository` |

**Why:**
- Services and repositories hold mutable runtime state (`AsyncSession`, `asyncio.Queue`,
  `dict` store) that Pydantic cannot and should not manage.
- Domain objects are pure typed data — Pydantic gives validation, JSON serialization,
  and `model_dump` for free.
- Using `BaseModel` for a service that takes a DB session as a constructor argument
  would break Pydantic's field validation model entirely.

**Rule:** If the class holds a dependency injected at construction (session, repo, queue),
use plain `__init__`. If it is a value that moves between layers, use `BaseModel`.

---

## `__init__.py` content rules (RUF067)

**Context:** `src/classiflow/__init__.py` was calling `configure_container()` at module
level, which caused `ModuleNotFoundError` whenever any `classiflow.*` submodule was imported
during tests.

**Decision:** `__init__.py` files must only contain:
- A `__version__` string (package root only)
- Re-exports (`from .module import Name`)
- `__all__` declarations

No executable statements, no function definitions, no side-effectful calls.

**Why:**
- Python executes `__init__.py` on every `import classiflow.*` — any side effect
  (DB connection, container wiring, network call) runs at import time, including in tests.
- ruff rule RUF067 enforces this and will fail `poe check` if violated.
- `configure_container()` belongs in `create_app()` (T16), not at import time.

**Do NOT put in `__init__.py`:**
```python
# WRONG — runs at import time
configure_container()

# WRONG — function definitions belong in a proper module
def configure_container() -> Container: ...
```

---

## Import style — re-exports via `__init__.py`

**Context:** Deciding between importing from specific submodules (full path) or from the
package `__init__.py` (short path).

**Decision:** Option B — each package exposes its public surface via re-exports in
`__init__.py` with an explicit sorted `__all__`. Callers always import from the package,
never from the internal submodule.

```python
# shared/auth/__init__.py
from classiflow.shared.auth.exceptions import AuthError, ExpiredTokenError, InvalidTokenError
from classiflow.shared.auth.jwt import DecodedPayload, decode_token, encode_token

__all__ = [
    "AuthError",
    "DecodedPayload",
    "ExpiredTokenError",
    "InvalidTokenError",
    "decode_token",
    "encode_token",
]

# caller
from classiflow.shared.auth import AuthError, decode_token  ✓
from classiflow.shared.auth.jwt import decode_token          ✗
```

**Why:**
- Stable import paths — if an internal file is renamed or split, only `__init__.py` changes;
  all callers remain untouched.
- `__all__` makes the public surface explicit and enforced by linters.
- ruff checks that `__all__` is sorted (RUF022) — keep it in isort order (uppercase before lowercase).

**Rules:**
- Direct submodule imports are only acceptable *inside* the same package.
- `__init__.py` content is still restricted to re-exports and `__all__` — no logic, no side effects (RUF067).

---

## Avoid `from __future__ import annotations`

**Context:** `shared/database/models.py` used `from __future__ import annotations` solely to
allow forward references to `DocumentStep` and `HumanDecision` inside the `Job` class, which
is defined earlier in the same file.

**Decision:** Do not use `from __future__ import annotations` unless there is a true circular
cross-file import that cannot be resolved any other way. Use explicit string quotes on the
specific annotations that need them instead.

```python
# WRONG — silently makes every annotation lazy
from __future__ import annotations

steps: Mapped[list[DocumentStep]] = relationship(...)

# RIGHT — only the forward references are quoted
steps: Mapped["list[DocumentStep]"] = relationship(...)
decisions: Mapped["list[HumanDecision]"] = relationship(...)
```

**Why:**
- `from __future__ import annotations` affects every annotation in the file and can interact
  unexpectedly with runtime consumers (SQLAlchemy column resolution, Pydantic `model_rebuild`,
  `get_type_hints`).
- Explicit quotes are surgical — only the problematic annotation pays the cost.
- When removing the import, check for now-unused `# noqa: TC003` comments on imports.

**For `TYPE_CHECKING` guards:** prefer a real runtime import over a guarded import + string
quote. If the import is genuinely heavy or circular, keep the guard and quote the annotation;
do not use `from __future__ import annotations` as a shortcut.

---

## Composed types → named Pydantic models

**Context:** `make_audit_record` had `detail: dict[str, object] | None` as a parameter type.
`AuditService.record` had `detail: dict[str, Any] | None`.

**Decision:** Replace any inline composed type used as a function parameter or return type
with a named `BaseModel` subclass. For open-ended / caller-defined payloads use
`model_config = ConfigDict(extra="allow")`.

```python
# WRONG
def make_audit_record(..., detail: dict[str, object] | None) -> AuditRecord: ...

# RIGHT
class AuditDetail(BaseModel):
    model_config = ConfigDict(extra="allow")

def make_audit_record(..., detail: AuditDetail | None) -> AuditRecord:
    return AuditRecord(
        ...
        detail=detail.model_dump() if detail else None,
    )
```

**Why:**
- Named models are self-documenting and appear in IDE completions and error messages.
- `extra="allow"` gives a typed container without locking down the shape prematurely.
- Serialize back to `dict` only at the infrastructure boundary (SQLAlchemy JSON column,
  HTTP response serializer) — never carry raw dicts between service layers.

**Propagation rule:** when `make_audit_record` changes its signature, update every caller
(`AuditService.record`) and every test that constructs the detail inline (`{"key": value}`
→ `AuditDetail(key=value)`). Test assertions against the stored record compare against
`.model_dump()` since the column stores the plain dict.

---

## Domain models inherit from `BaseEntity`, never plain `BaseModel`

**Context:** `JobState` in `ingesta/domain/state.py` was written with `BaseModel` directly
instead of `BaseEntity`.

**Decision:** Every domain model inside `classiflow/ingesta/domain/` (and any future agent
domain package) must inherit from `BaseEntity` defined in that package's `domain/base.py`.
Never use `pydantic.BaseModel` directly for domain objects.

```python
# WRONG
from pydantic import BaseModel

class JobState(BaseModel): ...

# RIGHT
from classiflow.ingesta.domain.base import BaseEntity

class JobState(BaseEntity): ...
```

**Why:**
- `BaseEntity` centralises shared config (`alias_generator`, `populate_by_name`, and any
  future shared validators or serialisers) so all domain models stay consistent.
- Changing a config setting in `base.py` propagates to every model automatically.
- Plain `BaseModel` bypasses that shared config and creates silent inconsistencies.

**Rule:** `base.py` is the single place to touch when domain-wide Pydantic config changes.
All domain result models, state models, and value objects import from it.

---

## DI container wiring — correct package name and startup timing

**Context:** `injections/__init__.py` contained `container.wire(packages=["app"])`,
a copy-paste artifact from the T01 skeleton template. `"app"` does not exist in this project.

**Decision:**
- Wire target must be `packages=["classiflow"]` (the actual package name).
- `configure_container()` is called once inside `create_app()` (FastAPI app factory, T16),
  never at module import time.
- Until T16 is implemented, `Container` and `TestContainer` remain empty stubs — do not
  add providers to them prematurely.

**Why:** Calling `container.wire()` with a wrong package name raises `ModuleNotFoundError`
at import time and breaks every test that touches any `classiflow.*` module.

**T09 follow-up:** `configure_container()` was accidentally left in `classiflow/__init__.py`
from a prior PR. This caused `container.wire(packages=["classiflow"])` to run at import time,
importing every module in the package — including system-library-backed modules like
`ingesta/mime.py` (which does `import magic` → needs `libmagic` installed). Removing the
call from `__init__.py` fixed the issue. The call belongs exclusively in `create_app()` (T16).

---
