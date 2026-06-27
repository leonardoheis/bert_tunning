---
name: pydantic
description: Use when defining data models, request/response schemas, config settings, or adding validation to Python classes. Triggers on "pydantic", "BaseModel", "schema", "validation", "Field", "validator", "model_dump", "model_validate", "BaseSettings", "ConfigDict", "alias", "request body", "response schema", "data class with validation", "parse JSON".
---

# Pydantic (v2)

## Overview

Pydantic validates data at the boundary — API inputs, config, inter-service contracts.
Inside the system, trust your own types. Define a model once; get validation, serialization, and OpenAPI docs for free.

**Project convention (from CFO_Copilot):** two separate base classes — `BaseEntity` for domain models and `BaseSchema` for API schemas — each with different `ConfigDict`.

---

## BaseModel — the foundation

```python
from pydantic import BaseModel, Field

class Document(BaseModel):
    title: str
    page_count: int = Field(gt=0, description="Number of pages")
    category: str | None = None       # optional field, defaults to None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
```

Fields **without** a default are required. Fields **with** a default are optional.
Pydantic coerces compatible types (`"42"` → `int`) by default; use strict mode to prevent it.

---

## Field() — constraints, aliases, defaults

```python
from pydantic import BaseModel, Field

class ClassificationResult(BaseModel):
    category: str = Field(min_length=1, max_length=100, description="Document type")
    confidence: float = Field(ge=0.0, le=1.0)
    page_count: int = Field(gt=0)

    # alias: external name ≠ Python name (e.g. "type" is a reserved word)
    doc_type: str = Field(alias="type")

    # default_factory for mutable defaults
    tags: list[str] = Field(default_factory=list)
```

| Constraint | Meaning |
|-----------|---------|
| `gt` / `ge` | greater than / greater than or equal |
| `lt` / `le` | less than / less than or equal |
| `min_length` / `max_length` | string or list length |
| `pattern` | regex pattern for strings |
| `description` | shows in OpenAPI docs |
| `alias` | external field name for parsing and serialization |

---

## ConfigDict — model-level settings

```python
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

class BaseEntity(BaseModel):
    """Domain model base — accepts snake_case or camelCase, serializes as camelCase."""
    model_config = ConfigDict(
        alias_generator=to_camel,    # auto snake_case → camelCase aliases
        populate_by_name=True,       # allow both "page_count" and "pageCount"
    )

class BaseSchema(BaseModel):
    """API schema base — allows non-serializable types (Path, numpy arrays)."""
    model_config = ConfigDict(
        alias_generator=to_camel,
        arbitrary_types_allowed=True,  # needed for Path, sklearn objects, etc.
    )
```

| `ConfigDict` option | When to use |
|--------------------|-------------|
| `alias_generator=to_camel` | API responses that must be camelCase |
| `populate_by_name=True` | Accept both original name and alias on input |
| `arbitrary_types_allowed=True` | Fields typed as `Path`, ML models, etc. |
| `strict=True` | No coercion — `"42"` does NOT become `42` |
| `frozen=True` | Immutable model — enables hashing |

---

## Validators

### `@field_validator` — single field

```python
from pydantic import BaseModel, field_validator

class Document(BaseModel):
    sha256: str
    language: str

    @field_validator("sha256")
    @classmethod
    def must_be_hex(cls, v: str) -> str:
        if len(v) != 64 or not all(c in "0123456789abcdef" for c in v):
            raise ValueError("sha256 must be a 64-char hex string")
        return v

    @field_validator("language", mode="before")  # runs before type coercion
    @classmethod
    def normalize_language(cls, v: object) -> str:
        return str(v).lower().strip()
```

### `@model_validator` — cross-field validation

```python
from typing import Self
from pydantic import BaseModel, model_validator
from app.exceptions import DimensionalityMismatchError

class TrainRequest(BaseModel):
    features: list[list[float]]
    labels: list[float]

    @model_validator(mode="after")       # runs after all fields are parsed
    def same_length(self) -> Self:
        if len(self.features) != len(self.labels):
            raise DimensionalityMismatchError(
                x_dim=len(self.features),
                y_dim=len(self.labels),
            )
        return self
```

`mode="after"` → access validated field values via `self`.
`mode="before"` → receives raw input dict, useful for pre-processing.

---

## Serialization & Parsing

```python
doc = Document(title="Decreto 123", page_count=5, category="decreto")

# → dict
doc.model_dump()                      # {"title": "Decreto 123", "page_count": 5, ...}
doc.model_dump(by_alias=True)         # uses alias names (camelCase if alias_generator set)
doc.model_dump(exclude_none=True)     # omit None fields
doc.model_dump(exclude={"confidence"})

# → JSON string
doc.model_dump_json()
doc.model_dump_json(by_alias=True)

# Parse from dict / JSON
Document.model_validate({"title": "...", "page_count": 3})
Document.model_validate_json('{"title": "...", "pageCount": 3}')
```

---

## Project Pattern — Domain vs API separation

```
src/<package>/
├── domain/
│   ├── base.py          # BaseEntity (alias_generator, populate_by_name)
│   ├── document.py      # domain models — business invariants
│   └── __init__.py
└── api/
    ├── schema.py        # BaseSchema (arbitrary_types_allowed)
    └── routes/
        └── classify/
            └── schemas.py   # request/response — API contract
```

```python
# domain/base.py
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

class BaseEntity(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

# domain/document.py
from pydantic import Field
from .base import BaseEntity

class ClassificationInput(BaseEntity):
    text: str = Field(min_length=1, description="Raw document text")
    sha256: str = Field(description="SHA-256 hash of the source file")

class ClassificationOutput(BaseEntity):
    category: str = Field(description="Predicted document type")
    confidence: float = Field(ge=0.0, le=1.0)

# api/schema.py
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

class BaseSchema(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, arbitrary_types_allowed=True)

# api/routes/classify/schemas.py
from pydantic import Field
from app.api.schema import BaseSchema

class ClassifyRequest(BaseSchema):
    input_: str = Field(alias="input", min_length=1)   # "input" is reserved in some contexts

class ClassifyResponse(BaseSchema):
    category: str
    confidence: float
```

---

## Services as BaseModel (project pattern)

When a service has configuration injected by the DI container, model it as a `BaseModel` rather than a plain class — you get free validation on construction:

```python
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from app.settings import Settings

class ClassificationService(BaseModel):
    model_path: Path = Field(default=Settings.MODEL_PATH)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def classify(self, text: str) -> ClassificationOutput: ...
```

---

## BaseSettings — config from environment

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class _Settings(BaseSettings):
    LLM_MODEL: str = "phi4-mini"
    DB_HOST: str = "localhost"
    API_PORT: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",     # silently drop unknown env vars
    )

    @property
    def MODELS_DIR(self) -> Path:
        return Path(__file__).parent / "models"

Settings = _Settings()  # singleton — import this, never re-instantiate
```

Reading order: keyword args → env vars → `.env` file → field defaults.
`extra="ignore"` prevents errors when `.env` has keys not defined in the model.

---

## Test Data — Polyfactory + ExamplerMixIn

```python
# utils/exampler.py  (project pattern from CFO_Copilot)
import secrets
from typing import Any, Self
from polyfactory.factories.pydantic_factory import ModelFactory

class ExamplerMixIn:
    @classmethod
    def create_example(cls, *, seed: int | None = None, **kwargs: Any) -> Self:
        class Factory(ModelFactory[cls]):  # type: ignore[valid-type]
            __random_seed__ = seed or secrets.randbits(32)
        return Factory.build(**kwargs)

# usage in tests and OpenAPI examples
doc = ClassificationInput.create_example()
doc = ClassificationInput.create_example(confidence=0.95)   # override specific fields
```

For OpenAPI examples:
```python
EXAMPLES = {
    "valid": {"value": ClassifyRequest.create_example().model_dump(by_alias=True)},
}
```

---

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Using mutable default (`default=[]`) | Use `default_factory=list` |
| `@validator` (v1 API) | Replace with `@field_validator` (v2) |
| `model.dict()` or `model.json()` (v1) | Use `model.model_dump()` / `model.model_dump_json()` |
| Forgetting `populate_by_name=True` with `alias_generator` | Without it, only the alias works on input |
| Raising `ValueError` in model_validator without returning `self` | Always `return self` at the end of `mode="after"` validators |
| Services instantiating infrastructure in `__init__` | Pass as Field with default from Settings; inject via DI |
| Missing `arbitrary_types_allowed=True` for `Path` / ML objects | Add to `ConfigDict` on the base class |
