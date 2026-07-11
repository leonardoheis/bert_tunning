---
name: design-patterns
description: Use when a design pattern might improve the code — before suggesting or applying any pattern, present the candidate pattern(s) to the user and wait for explicit approval. Triggers on "refactor", "improve structure", "decouple", "extract", "simplify dependencies", "make extensible", "avoid duplication", or when recognizing a known pattern mismatch in existing code.
---

# Design Patterns (Python)

Reference: https://refactoring.guru/design-patterns/python

## MANDATORY RULE: Always Validate Before Applying

**Never apply a design pattern without user confirmation.**

When a pattern seems applicable:
1. Identify the problem in the current code (one sentence)
2. Name the candidate pattern and explain why it fits
3. Show the trade-off (what complexity it adds vs. what it solves)
4. Ask: _"Should I apply the [Pattern Name] pattern here?"_
5. Wait for explicit approval before writing any code

Do NOT ask about multiple patterns at once — propose one, get approval, then proceed.

---

## Creational Patterns — *how objects are created*

| Pattern | Solves | Python note |
|---|---|---|
| **Abstract Factory** | Produce families of related objects without coupling to concrete classes | Use `ABC` + multiple `create_*` methods; swap entire factory at runtime |
| **Builder** | Construct complex objects step-by-step, separating construction from representation | Return `self` from each setter for fluent chaining; add `build()` as final step |
| **Factory Method** | Let subclasses decide which class to instantiate | Define `create_product()` as `@abstractmethod`; override in subclasses |
| **Prototype** | Clone existing objects without depending on their class | Use `copy.copy()` / `copy.deepcopy()` or implement `__copy__`/`__deepcopy__` |
| **Singleton** | Ensure only one instance exists globally | Use `__new__` guard or a module-level instance; prefer DI container over global state |

### When to use each Creational pattern

```
Need one instance globally?         → Singleton (but prefer DI injection)
Need to clone objects?              → Prototype
Need subclasses to control creation? → Factory Method
Need families of related objects?   → Abstract Factory
Complex multi-step construction?    → Builder
```

---

## Structural Patterns — *how objects are composed*

| Pattern | Solves | Python note |
|---|---|---|
| **Adapter** | Make incompatible interfaces work together | Wrap the third-party object; expose the expected interface |
| **Bridge** | Decouple abstraction from implementation so both can vary independently | Inject the implementation as a constructor dependency |
| **Composite** | Treat individual objects and compositions uniformly | Share a common `Component` ABC with `operation()` and optional `add()`/`remove()` |
| **Decorator** | Add behaviour to objects at runtime without subclassing | Wrap the component; delegate to it; Python `@functools.wraps` preserves metadata |
| **Facade** | Simplify a complex subsystem behind a single entry point | One class, thin methods, no business logic inside the facade itself |
| **Flyweight** | Share common state across many fine-grained objects to save memory | Use `__slots__` and a factory that caches instances by intrinsic state |
| **Proxy** | Control access to another object (lazy load, logging, auth, cache) | Implement the same interface as the real subject; delegate calls |

### When to use each Structural pattern

```
Wrapping a third-party library?              → Adapter
Adding optional behaviour at runtime?        → Decorator
Hiding subsystem complexity?                 → Facade
Controlling/intercepting access?             → Proxy
Tree of uniform leaf+composite nodes?        → Composite
Abstraction × implementation axes vary?      → Bridge
Millions of similar objects, memory matters? → Flyweight
```

---

## Behavioral Patterns — *how objects communicate*

| Pattern | Solves | Python note |
|---|---|---|
| **Chain of Responsibility** | Pass a request along a handler chain; each decides to handle or forward | Link handlers via `set_next()`; return `None` or a result to break the chain |
| **Command** | Encapsulate a request as an object (queue, undo, log) | Dataclass with `execute()` + optional `undo()`; store in a list for history |
| **Iterator** | Traverse a collection without exposing its internals | Implement `__iter__` + `__next__`; use `yield` for generator-based iterators |
| **Mediator** | Reduce direct coupling between many objects via a central hub | Components know only the mediator; mediator coordinates them |
| **Memento** | Save and restore object state without breaking encapsulation | Inner `Memento` dataclass holds a snapshot; `Caretaker` holds the history stack |
| **Observer** | Notify multiple subscribers when an object's state changes | Maintain a `list[Observer]`; call `update()` on each; Python `weakref` avoids leaks |
| **State** | Let an object change behaviour when its internal state changes | Extract each state into its own class; context delegates to the current state object |
| **Strategy** | Swap algorithms at runtime without changing the client | Inject the strategy as a callable or an ABC; keep the context free of conditionals |
| **Template Method** | Define an algorithm skeleton in a base class; let subclasses fill in steps | Mark steps as `@abstractmethod`; call them from the non-overridable template method |
| **Visitor** | Add new operations to objects without modifying their classes | `accept(visitor)` on elements; `visit_concrete_x()` methods on the visitor |

### When to use each Behavioral pattern

```
One event → many listeners?                  → Observer
Swappable algorithms?                        → Strategy
Object behaviour changes with internal state? → State
Decouple many components from each other?    → Mediator
Undoable operations / command queue?         → Command
New operations on a stable object hierarchy? → Visitor
Save/restore snapshots?                      → Memento
Traverse without exposing internals?         → Iterator
Fixed algorithm, variable steps?             → Template Method
Dynamic processing pipeline?                 → Chain of Responsibility
```

---

## Python-specific guidance

- **Prefer `Protocol` over `ABC`** for structural typing when callers shouldn't import the base class.
- **Dataclasses** work well as Command, Memento, and Value Object implementations.
- **`functools.singledispatch`** can replace Visitor when adding operations to existing types.
- **Module-level singletons** (a plain instance at the bottom of a module) are idiomatic Python and simpler than `__new__` tricks.
- **Generators** (`yield`) are Python's native Iterator — only build a class-based iterator when state management is complex.
- **`contextlib.contextmanager`** is often a better Decorator for resource-wrapping than a wrapper class.

---

## Common Mistakes

| Mistake | Fix |
|---|---|
| Applying a pattern because it "feels right" | Name the concrete problem first; if you can't, skip the pattern |
| Singleton for everything shared | Use dependency injection; singletons hide coupling |
| Decorator class when a function wrapper suffices | Use `functools.wraps` + a plain closure for simple cases |
| Observer without weak references | Use `weakref.WeakSet` to avoid memory leaks on long-lived subjects |
| Strategy with a single implementation | Wait until there are two strategies before extracting the abstraction |
| Abstract Factory when Factory Method is enough | Start simpler; add a factory only when families are needed |
