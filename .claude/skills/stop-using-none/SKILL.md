---
name: stop-using-none
description: Use when a function returns X | None (or Optional[X]) to mean more than one distinct kind of "missing" — no record found vs. invalid input vs. lookup failed — or when code is accumulating "if x is not None" chains before every attribute access. Symptoms include AttributeError:'NoneType' object has no attribute surfacing far from the actual cause, and callers unable to tell why a value is absent.
---

# Stop Using None for Missing Values

## Overview

`None` is one value pressed into service for many different meanings: "not found," "not loaded yet," "invalid input," "the lookup failed," "explicitly empty." Callers can't tell which one they got, so every `None` check is a guess, and the real failure surfaces later as `AttributeError: 'NoneType' object has no attribute 'x'` — far from where the ambiguity was introduced.

**Core principle:** name the specific kind of absence instead of collapsing it into `None`. If you can't describe *why* the value is missing in one phrase, you haven't modeled it yet.

## When to Use

- A function's return type is `X | None` and callers do different things depending on *why* it was `None` (not found vs. invalid ID vs. backend error) — but the signature can't express that difference.
- You're writing `if x is not None and x.y is not None and x.y.z is not None` before you can do anything.
- A bug report says "it crashed with `NoneType has no attribute`" three call frames away from where the value actually went missing.
- You're tempted to return `None` for "nothing to iterate" — an empty list already means that, and needs no check at all.
- You're about to write `Optional[X]` on an internal function parameter or return type "just in case," not because the value is genuinely optional at that specific boundary.

**Don't use when:** the value is genuinely, unambiguously optional in only one way, and `None`'s single meaning is exactly what's needed — e.g. `dict.get(key)` returning `None` for "key not present" with no other reason to be absent. Reach for the patterns below only once a second meaning creeps in.

**The test, concretely:** before adding a result type or a `reason` field, name a caller that does something *different* for each reason. If every caller's handling is "do the same fallback regardless of why," a plain `X | None` already says everything the caller needs — a `reason` field nobody branches on is the same overload problem in a fancier box. Two sibling functions in the same module can land on opposite sides of this: one returning `X | None` because callers only ever ask "did I get it or not," another returning a result-with-reason type because a caller genuinely does different things per reason. Don't let the second infect the first just because they sit next to each other.

## Quick Reference

| Situation | Instead of `None`, use |
|---|---|
| Distinguishing "not called yet" from "called, found nothing" | A sentinel object (`_MISSING = object()`) as the default, never `None` |
| Caller needs to know *why* something failed | A result/outcome type carrying `value` or `error` |
| Value is only ever absent at a system boundary (API input, config, DB row) | `Optional[X]` / `X | None` — this is the *one* place it still fits |
| "Nothing here" | An empty collection (`[]`, `{}`) — no None-check needed, iterate/loop over it directly |
| A quantity that has a sensible zero state | The domain default (`Decimal("0.00")`, not `None`) |
| Caller has no reasonable way to proceed without the value | Raise a specific exception instead of returning `None` |

## Core Pattern

**Before** — one `None` doing four jobs, discovered by an `AttributeError` downstream:

```python
def find_active_session(user_id: str) -> Session | None:
    return ACTIVE_SESSIONS.get(user_id)

# Caller can't tell: never logged in? expired? cancelled? Just "falsy".
session = find_active_session(user_id)
if session is not None:
    charge(session.cart_total)  # silently skipped for every kind of "missing"
```

**After** — the reason is part of the type, so the compiler/reviewer catches a forgotten case, not a runtime crash three frames later:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class SessionFound:
    session: Session

@dataclass(frozen=True)
class NoActiveSession:
    reason: str  # "never_started" | "expired" | "cancelled"

def find_active_session(user_id: str) -> SessionFound | NoActiveSession:
    session = ACTIVE_SESSIONS.get(user_id)
    if session is None:
        return NoActiveSession(reason="never_started")
    if session.expired:
        return NoActiveSession(reason="expired")
    return SessionFound(session=session)

result = find_active_session(user_id)
match result:
    case SessionFound(session):
        charge(session.cart_total)
    case NoActiveSession(reason="expired"):
        prompt_relogin()
    case NoActiveSession():
        prompt_login()
```

Only reach for this when a caller genuinely branches on the reason. If every caller just needs "do I have a value or not," a plain `X | None` is not the problem — see "When to Use."

### Sentinel for "not provided" vs. "provided as None"

The other common collision: a function parameter where the caller might legitimately pass `None` on purpose, so `None` can't also mean "argument omitted."

```python
_MISSING = object()

def update_user(user_id: str, email: str | None | object = _MISSING) -> None:
    if email is _MISSING:
        return  # caller didn't touch email — leave it alone
    users[user_id].email = email  # caller explicitly wants to set it, even to None/clear it
```

## Common Mistakes

| Mistake | Fix |
|---|---|
| `Optional[X]` on every internal function "to be safe" | Reserve `Optional` for actual system boundaries (API/config/DB); internal functions should have a value or raise |
| Returning `None` for "empty list of results" | Return `[]` — no caller needs an `is None` check to iterate zero items |
| Treating a `0`/`""`/`False` return the same as `None` | Use a sentinel or result type; don't let a falsy check swallow "found nothing meaningful" and "found the value zero" as the same case |
| Adding a 4th `if x is not None` to a chain instead of stepping back | That's the signal to introduce a result type or raise early — the chain is the symptom, not the fix |
| Using `None` as a function-parameter default when the caller might legitimately pass `None` | Use a private sentinel object (`_MISSING = object()`) instead, so "omitted" and "explicitly None" are distinguishable |

## References

- Article: ["Never Use None for Missing Values Again — Do This Instead"](https://medium.com/the-pythonworld/never-use-none-for-missing-values-again-do-this-instead-ad1ce9117a13) — source for the six patterns this skill distills.
- Video: ["Stop Checking for None Everywhere"](https://www.youtube.com/watch?v=h8ZwhU3PpVw) — companion viewing on the same theme.
