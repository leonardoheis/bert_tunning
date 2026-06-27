---
name: python-oop
description: Use this skill when building new Python software, creating new classes or modules, refactoring procedural/functional code to OOP, or when the user asks to design a class hierarchy, use OOP patterns, or apply object-oriented principles. Triggers on phrases like "add class", "OOP", "object oriented", "refactor to classes", "design classes", "class structure", "separate concerns", "encapsulation".
version: 1.0.0
---

# Python OOP Development Guidelines

Reference: https://realpython.com/python3-object-oriented-programming/

## The Four Pillars

**Encapsulation** — bundle related data and behavior in one class; hide internals behind a clean interface.
**Abstraction** — callers interact with *what* a class does, not *how*; name methods after intent.
**Inheritance** — use when a subclass truly *is-a* variant of the parent. Prefer composition for code reuse.
**Polymorphism** — design classes to be interchangeable via shared method signatures (duck typing).

---

## Class Design Rules

1. **CapWords naming**: `DataCleaner`, not `data_cleaner`.
2. **All instance state in `__init__`** — never set new attributes outside of it.
3. **Fitted state uses trailing underscore**: `self.model_`, `self.mean_` — signals "set by fit()".
4. **One class = one responsibility** — if you need a conjunction ("clean *and* encode"), split it.
5. **`@staticmethod`** for operations that don't use `self` or `cls`.
6. **`@classmethod`** for alternative constructors: `DataCleaner.from_config(path)`.
7. **`@property`** for computed read-only attributes that look like data.
8. **`super().__init__()`** always in child `__init__` when inheriting.
9. **`isinstance()`** to check type relationships; never compare `type(x) == Foo`.

---

## Sklearn-Compatible Fit/Transform Pattern

Use this pattern for any class that learns from data (encoders, scalers, imputers):

```python
class SomeTransformer:
    def __init__(self, hyperparams):
        self.hyperparams = hyperparams
        self.artifact_: SomeType | None = None  # fitted state — None until fit()

    def fit(self, data, target=None) -> "SomeTransformer":
        self.artifact_ = ...  # learn from data
        return self  # always return self for chaining

    def transform(self, data):
        if self.artifact_ is None:
            raise RuntimeError("Call fit() before transform().")
        return ...  # apply without re-fitting

    def fit_transform(self, data, target=None):
        # Override when fit+transform has a different (e.g. CV-aware) path:
        trained_result = self._fit_core(data, target)
        return trained_result
        # Otherwise: return self.fit(data, target).transform(data)
```

**Key rule**: `transform()` must never fit. If calling `transform` on train requires CV-adjusted values (e.g., `TargetEncoder`), override `fit_transform` to store and return the CV output separately.

---

## Separation of Concerns — When to Split

| Responsibility | Class |
|---|---|
| Loading raw data | `@staticmethod load()` on first transformer, or a `DataLoader` |
| Cleaning / imputation | `DataCleaner` |
| Feature engineering | Separate class, or phase methods on `DataCleaner` |
| Encoding (fit on train only) | `TabularEncoder`, `TextEncoder` |
| Scaling | Use sklearn's `StandardScaler` directly |
| Export / persistence | `DatasetExporter` |
| Orchestration | `Pipeline` class composed of the above |

---

## Anti-Patterns to Avoid

| Anti-pattern | Fix |
|---|---|
| God class — one class does everything | Split by responsibility |
| Anemic model — data-only class with no methods | Add behavior |
| Attributes set outside `__init__` | Move to `__init__`, initialize to `None` |
| Mutable class attributes shared across instances | Use instance attributes |
| Calling `fit` inside `transform` silently | Make fit explicit; raise `RuntimeError` if not fitted |
| Single-letter variable names in class scope | Use descriptive names |

---

## Docstring Convention (for this project)

Use the `Usage::` block to show callers the expected sequence:

```python
class DataCleaner:
    """Cleans raw Wine Reviews data and engineers base features.

    Usage::

        cleaner = DataCleaner()
        cleaner.fit(df)
        cleaned  = cleaner.clean(df)
        featured = cleaner.add_features(cleaned)
    """
```

Single-line comments only for non-obvious WHY; no block comments explaining what the code does.

---

## Imports

- **Never** add `from __future__ import annotations` — only include it when a forward reference cannot be resolved with a quoted string literal or by reordering definitions.
