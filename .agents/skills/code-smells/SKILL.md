---
name: code-smells
description: Use when reviewing or refactoring code and something feels off but isn't a bug — long methods, bloated classes, excessive parameters, duplicated logic, deep coupling, dead code, magic numbers, boolean flags, misleading names, or over-engineered abstractions. Triggers on "code smell", "this feels messy", "hard to maintain", "too many parameters", "duplicate code", "feature envy", "god class", "flag argument", "magic number", "refactor this", or when a diff review needs to name what's wrong beyond "it works."
---

# Code Smells

References: https://refactoring.guru/refactoring/smells · https://luzkan.github.io/smells/ (extended catalog)

## Overview

A code smell is a surface indicator of a deeper design problem — not a bug, the code still runs, but it's harder to understand, test, or change than it should be. Naming the smell turns a vague "this feels off" into a concrete, actionable finding with a known refactoring target.

## MANDATORY RULE: Diagnose Before Refactoring

**Name the smell and the concrete cost before touching code.** State: which smell, where (file:line), and what it actually costs (an "and" in a description, a change that ripples across files, a method nobody can hold in their head). Then ask before applying a fix — smells are judgment calls, not lint errors; what looks like Duplicate Code at 2 call sites may be intentional divergence waiting to happen.

## Quick Reference — Five Categories

| Category | What it means | Smells |
|---|---|---|
| **Bloaters** | Something has grown too large to reason about | Long Method, Large Class, Primitive Obsession, Long Parameter List, Data Clumps |
| **Object-Orientation Abusers** | OOP principles applied incompletely or wrong | Switch Statements, Temporary Field, Refused Bequest, Alternative Classes with Different Interfaces |
| **Change Preventers** | One change forces edits in many places | Divergent Change, Shotgun Surgery, Parallel Inheritance Hierarchies |
| **Dispensables** | Unnecessary code — removing it improves clarity | Comments (as a crutch), Duplicate Code, Data Class, Dead Code, Lazy Class, Speculative Generality |
| **Couplers** | Classes too entangled with each other | Feature Envy, Inappropriate Intimacy, Message Chains, Middle Man, Incomplete Library Class |

## Bloaters

| Smell | Signal | Fix direction |
|---|---|---|
| **Long Method** | A method you can't summarize in one sentence without "and" | Extract Method — pull cohesive chunks into named functions |
| **Large Class** | A class doing several unrelated jobs (see SRP) | Extract Class — split by responsibility |
| **Primitive Obsession** | Strings/ints standing in for a concept with its own rules (e.g. a raw `str` for an email) | Introduce a small value object or enum |
| **Long Parameter List** | 4+ params, especially several of the same type | Introduce Parameter Object, or pass a config/context object |
| **Data Clumps** | The same 3-4 variables always travel together as separate args | Bundle into a class/dataclass/NamedTuple |

## Object-Orientation Abusers

| Smell | Signal | Fix direction |
|---|---|---|
| **Switch Statements** | The same `if/elif`-on-type chain repeated in multiple methods | Replace with polymorphism (Strategy/Factory) |
| **Temporary Field** | An instance attribute only set/valid in specific call sequences | Extract a class scoped to that lifecycle, or pass as a parameter instead of storing |
| **Refused Bequest** | A subclass overrides most parent methods to no-op or raise | Favor composition, or split the hierarchy (see LSP) |
| **Alternative Classes with Different Interfaces** | Two classes do the same job with differently-named methods | Rename to a shared interface/Protocol |

## Change Preventers

| Smell | Signal | Fix direction |
|---|---|---|
| **Divergent Change** | One class keeps changing for unrelated reasons (billing logic AND email formatting) | Extract Class per reason-to-change (SRP) |
| **Shotgun Surgery** | One conceptual change requires edits across many files | Consolidate the scattered logic behind one seam |
| **Parallel Inheritance Hierarchies** | Every new subclass in hierarchy A forces a matching subclass in hierarchy B | Merge the hierarchies, or have B reference A directly |

## Dispensables

| Smell | Signal | Fix direction |
|---|---|---|
| **Comments (as a crutch)** | A comment explaining *what* code does because the code itself doesn't say it | Rename/restructure so the comment is unnecessary; keep only *why* comments |
| **Duplicate Code** | Same logic copy-pasted at 2+ sites | Extract shared function — but confirm it's truly the same rule, not coincidental similarity |
| **Data Class** | A class with only fields/getters/setters, no behavior | Fine if it's genuinely a DTO/schema; smell only if behavior *should* live there and got extracted elsewhere as procedural code operating on it |
| **Dead Code** | Unreachable branches, unused functions/params, `# noqa` masking unused imports | Delete it — version control remembers |
| **Lazy Class** | A class/wrapper with no meaningful behavior of its own | Inline it into its one caller |
| **Speculative Generality** | An abstraction, hook, or config flag for a "someday" case with no current second implementation | Delete/simplify — YAGNI (see `ponytail`, `design-patterns` skills) |

## Couplers

| Smell | Signal | Fix direction |
|---|---|---|
| **Feature Envy** | Method `A.foo()` reads mostly from `B`'s internals, barely touches `A` | Move the method to `B` |
| **Inappropriate Intimacy** | Two classes reach into each other's private fields/internals routinely | Tighten encapsulation; define an explicit interface between them |
| **Message Chains** | `a.get_b().get_c().get_d().value` | Hide the chain behind a method on `a` (Law of Demeter) |
| **Middle Man** | A class whose methods just forward to another object, adding nothing | Remove the middle man; call the real object directly |
| **Incomplete Library Class** | A third-party class is missing a method you need repeatedly | Wrap it (Adapter/Decorator) rather than patching call sites everywhere |

## Extended Catalog (luzkan.github.io)

Smells beyond the classic refactoring.guru list — same five categories plus four more the extended catalog adds. Aliases to the list above are noted.

### More Bloaters

| Smell | Signal | Fix direction |
|---|---|---|
| **Combinatorial Explosion** | Near-identical code multiplies for every combination of a few variants | Extract the shared shape; parameterize the variation, don't copy-paste it |
| **Null Check** | `if x is not None` guards scattered everywhere before use | Null Object pattern, or narrow the type so `None` isn't a valid state to begin with |
| **Oddball Solution** | The same problem is solved a different way each time it recurs | Consolidate to one implementation, reuse it |
| **Required Setup or Teardown Code** | Callers must remember to call an init/cleanup pair around every use | Wrap in a context manager (`with`) or constructor/destructor so it can't be forgotten |

### More Change Preventers

| Smell | Signal | Fix direction |
|---|---|---|
| **Callback Hell** | Nested callbacks several levels deep | Flatten with async/await, promises, or early returns |
| **Flag Argument** | A boolean parameter selects between two different behaviors inside the function | Split into two functions, or replace with a named enum (see this session's `ConfidenceTier`/`OodEvidence` pattern) |
| **Special Case** | A pile of `if value == SPECIAL` checks before the real logic runs | Null Object / Special Case pattern — give the special case the same interface as the normal one |

### More Couplers / Data Dealers

| Smell | Signal | Fix direction |
|---|---|---|
| **Global Data** | Mutable state reachable from anywhere, not passed explicitly | Pass it as a parameter or inject it (see `dependency-injection-python` skill) |
| **Hidden Dependencies** | A function silently reaches out to a global/singleton instead of receiving it as a param | Make the dependency an explicit argument |
| **Tramp Data** | A parameter is passed through several layers of calls just to reach the one place that uses it | Restructure the call chain so the data reaches its consumer directly |
| **Insider Trading** | Alias for **Inappropriate Intimacy** above — classes reaching into each other's internals | Tighten the interface between them |
| **Indecent Exposure** | A class exposes internal details (fields, helper methods) it shouldn't | Reduce to a minimal public interface |
| **Afraid To Fail** | Excessive try/except or defensive checks around things that can't actually go wrong | Trust the boundary; validate only at real trust boundaries |

### Functional Abusers (relevant even outside FP-first languages)

| Smell | Signal | Fix direction |
|---|---|---|
| **Mutable Data** | Shared state changes from far-away code, causing action-at-a-distance bugs | Prefer immutable values (`frozen=True`, tuples, `@dataclass(frozen=True)`) |
| **Side Effects** | A function named/expected to compute a value also mutates state or does I/O | Separate the computation from the effect; make effects explicit in the name |
| **Imperative Loops** | A hand-written loop does what a comprehension/map/filter/reduce would say more directly | Prefer the declarative form when it's not less readable |

### Lexical Abusers (naming & literals)

| Smell | Signal | Fix direction |
|---|---|---|
| **Magic Number** | A bare literal (`0.7`, `26.125`) with no name explaining what it means | Name it as a constant — this project's `Settings.OOD_COSINE_THRESHOLD` is the pattern to follow |
| **Boolean Blindness** | `configure(true, false, true)` — call site gives no clue what each bool means | Named/keyword args, or replace booleans with an enum |
| **Uncommunicative Name** | `x`, `tmp`, `data2` — the name doesn't say what the thing is | Rename to state intent |
| **Fallacious Method/Comment** | The name or comment says one thing, the code does another | Fix the mismatch — whichever is wrong, name or code |
| **Inconsistent Names / Style** | The same concept is called different things in different files | Pick one term, use it everywhere |
| **Complicated Regex/Boolean Expression** | A one-liner regex or boolean condition nobody can parse at a glance | Break into named intermediate variables, or a small helper function |

### Obfuscators

| Smell | Signal | Fix direction |
|---|---|---|
| **Clever Code** | Implementation optimizes for "look how smart this is" over readability | Prefer the boring, obvious version |
| **Obscured Intent** | Structure (deep nesting, indirection) hides what the code is actually trying to do | Flatten, extract, and name the intent explicitly |
| **Status Variable** | A mutable flag tracks "what happened so far" for a later `if` | Restructure control flow so the state doesn't need to be remembered |
| **Vertical Separation** | A variable is declared far from where it's used | Move the declaration close to its first use |

### More Object-Orientation Abusers

| Smell | Signal | Fix direction |
|---|---|---|
| **Conditional Complexity** | Alias for **Switch Statements** above — long if/elif or switch chains | Polymorphism, or a lookup table for pure data-driven cases |
| **Base Class depends on Subclass** | A parent class reaches down to call something only a specific child defines | Invert the dependency — parent shouldn't know about children at all |

## Common Mistakes

| Mistake | Fix |
|---|---|
| Flagging every long function regardless of cohesion | A 40-line function doing one clear linear thing isn't Long Method; one doing 4 unrelated things at 15 lines is |
| Treating Duplicate Code and coincidental similarity as the same thing | Two call sites that happen to look alike today may diverge tomorrow — confirm it's the same *rule*, not just the same *text*, before extracting |
| "Refactoring" Speculative Generality by adding docs instead of deleting | The fix is deletion; documenting an unused abstraction doesn't un-smell it |
| Applying a smell's textbook fix without checking blast radius | Grep callers first — Extract Method/Class changes call sites; verify nothing else relies on the current shape |
