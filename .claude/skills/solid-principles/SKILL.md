---
name: solid-principles
description: Use when designing classes, spotting why code is hard to test or extend, refactoring a class with too many responsibilities, or adding a feature that requires modifying existing code. Triggers on "SOLID", "SRP", "OCP", "LSP", "ISP", "DIP", "class design", "too many responsibilities", "hard to extend", "dependency on concrete", "fat interface", "open closed", "liskov", "interface segregation", "dependency inversion".
---

# SOLID Principles — Python

## Overview

Five principles for class design that make code easier to extend, test, and reason about.
In Python: prefer `Protocol` (structural typing) over ABCs for abstractions — it enables
duck typing with type-checker support and doesn't force inheritance hierarchies.

| Principle | One-line rule |
|-----------|--------------|
| **SRP** — Single Responsibility | A class has one reason to change |
| **OCP** — Open / Closed | Open for extension, closed for modification |
| **LSP** — Liskov Substitution | Subtypes must be usable wherever the parent is used |
| **ISP** — Interface Segregation | No class should depend on methods it doesn't use |
| **DIP** — Dependency Inversion | Depend on abstractions, not concrete classes |

---

## S — Single Responsibility

One class, one job. Split when you write "and" describing what a class does.

```python
# ❌ Violation: Document knows how to save itself AND render to PDF
class Document:
    def __init__(self, content: str) -> None:
        self.content = content
    def save(self, path: str) -> None: ...       # I/O — not Document's job
    def to_pdf(self) -> bytes: ...               # rendering — not Document's job

# ✅ Fix: one class per responsibility
class Document:
    def __init__(self, content: str) -> None:
        self.content = content

class DocumentRepository:
    def save(self, doc: Document, path: str) -> None: ...

class PDFRenderer:
    def render(self, doc: Document) -> bytes: ...
```

**Signal:** a class is hard to unit-test because setting it up requires unrelated infrastructure.

---

## O — Open / Closed

Add behavior by writing new code, not by editing existing code.
Achieve this with polymorphism: a shared Protocol/ABC + injection.

```python
# ❌ Violation: every new export format requires editing ReportExporter
class ReportExporter:
    def export(self, report: Report, fmt: str) -> bytes:
        if fmt == "pdf": ...
        elif fmt == "csv": ...
        # must touch this for every new format

# ✅ Fix: inject the formatter — ReportExporter never changes again
from typing import Protocol

class ReportFormatter(Protocol):
    def format(self, report: Report) -> bytes: ...

class PDFFormatter:
    def format(self, report: Report) -> bytes: ...

class CSVFormatter:
    def format(self, report: Report) -> bytes: ...

class ReportExporter:
    def export(self, report: Report, formatter: ReportFormatter) -> bytes:
        return formatter.format(report)   # closed for modification
```

**Signal:** adding a new variant requires an `elif` in an existing method.

---

## L — Liskov Substitution

Any subclass must honour the parent's behavioural contract — not just its method signatures.
Violations appear when subclass overrides raise unexpected exceptions or change invariants.

```python
# ❌ Violation: Square breaks Rectangle's invariant (w and h are independent)
class Rectangle:
    def set_width(self, w: float) -> None: self.width = w
    def set_height(self, h: float) -> None: self.height = h

class Square(Rectangle):
    def set_width(self, w: float) -> None:
        self.width = self.height = w    # side-effect breaks caller expectations

# ✅ Fix: don't force an IS-A relationship; use a shared Protocol instead
from typing import Protocol

class Shape(Protocol):
    def area(self) -> float: ...

class Rectangle:
    def __init__(self, w: float, h: float) -> None:
        self.width, self.height = w, h
    def area(self) -> float:
        return self.width * self.height

class Square:
    def __init__(self, side: float) -> None:
        self.side = side
    def area(self) -> float:
        return self.side ** 2
```

**Signal:** a subclass raises `NotImplementedError` or silently changes behaviour callers depend on.

---

## I — Interface Segregation

Keep interfaces small and focused. Clients should not be forced to implement methods they don't use.
In Python: use multiple small `Protocol` classes instead of one fat ABC.

```python
# ❌ Violation: PDFReader is forced to implement send_by_email
from abc import ABC, abstractmethod

class DocumentProcessor(ABC):
    @abstractmethod
    def read(self) -> str: ...
    @abstractmethod
    def write(self, content: str) -> None: ...
    @abstractmethod
    def convert_to_pdf(self) -> bytes: ...
    @abstractmethod
    def send_by_email(self, address: str) -> None: ...

# ✅ Fix: composable Protocols — implement only what you need
from typing import Protocol

class Readable(Protocol):
    def read(self) -> str: ...

class Writable(Protocol):
    def write(self, content: str) -> None: ...

class PDFExportable(Protocol):
    def convert_to_pdf(self) -> bytes: ...

# A class can satisfy multiple protocols without inheritance
class PDFReader:
    def read(self) -> str: ...
    def convert_to_pdf(self) -> bytes: ...
    # no write, no email — and that's fine
```

**Signal:** a class has stub methods with `pass` or `raise NotImplementedError` for things it doesn't support.

---

## D — Dependency Inversion

High-level modules depend on abstractions, not concrete implementations.
Pass dependencies in — don't instantiate them inside the class.

```python
# ❌ Violation: DocumentClassifier is welded to OllamaClient
class DocumentClassifier:
    def __init__(self) -> None:
        self.llm = OllamaClient()          # can't test without a running Ollama

    def classify(self, text: str) -> str:
        return self.llm.complete(f"Classify: {text}")

# ✅ Fix: depend on a Protocol, inject the implementation
from typing import Protocol

class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...

class DocumentClassifier:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm                     # caller decides the implementation

    def classify(self, text: str) -> str:
        return self.llm.complete(f"Classify: {text}")

# Production
classifier = DocumentClassifier(llm=OllamaClient())

# Tests — no server needed
class FakeLLM:
    def complete(self, prompt: str) -> str:
        return '{"category": "decreto", "confidence": 0.95}'

classifier = DocumentClassifier(llm=FakeLLM())
```

**Signal:** a class is hard to test because its `__init__` creates infrastructure objects.

---

## Python-Specific Notes

| Situation | Recommendation |
|-----------|---------------|
| Defining an abstraction | Prefer `typing.Protocol` over `ABC` — no inheritance required |
| Enforcing at runtime | Use `ABC` only when you need `isinstance()` checks or runtime enforcement |
| Multiple small behaviours | Compose `Protocol` classes; use mixins sparingly |
| Duck typing | Satisfies LSP naturally — no base class needed if the interface is consistent |
| Dependency injection | Pass deps as `__init__` params; use `@lru_cache` or a factory for singletons |

---

## Violation Signals — Quick Checklist

- [ ] Class name contains "And", "Manager", "Handler", "Util" → likely SRP violation
- [ ] Adding a feature requires editing an existing `if/elif` chain → OCP violation
- [ ] Subclass raises `NotImplementedError` on inherited methods → LSP violation
- [ ] Class has stub methods returning `None` or `...` it doesn't use → ISP violation
- [ ] `__init__` instantiates a DB client, HTTP client, or LLM → DIP violation

---

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Using `ABC` for every abstraction | Use `Protocol` — it's lighter and doesn't require inheritance |
| Splitting classes so far they share no state | SRP is about *reasons to change*, not raw line count |
| DIP via module-level singletons | Inject via `__init__`; singletons hide dependencies |
| Protocols without `runtime_checkable` used in `isinstance` | Add `@runtime_checkable` only when you truly need `isinstance` checks |
| LSP ignored because Python won't enforce it | mypy + `Protocol` will catch substitutability violations statically |
