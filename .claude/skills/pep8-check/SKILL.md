---
name: pep8-check
description: Use when writing, reviewing, or auditing Python code for PEP8 style compliance — naming conventions, whitespace, imports, line length, docstrings. Triggers on "check PEP8", "is this PEP8 compliant", "lint this for style", "review Python style", or before committing new Python files.
---

# PEP8 Compliance Check

Sources: [PEP 8 — Style Guide for Python Code](https://peps.python.org/pep-0008/), [El Libro de Python — PEP8](https://ellibrodepython.com/python-pep8)

## How to Use

1. Check whether the project has an automated linter configured for style (ruff, flake8, pycodestyle — look in `pyproject.toml`, `setup.cfg`, `tox.ini`, `.flake8`). If so, run it first (e.g. `ruff check .`, `ruff format --check .`) — a linter catches the large majority of mechanical PEP8 violations far more reliably than manual reading. Reserve manual review for what a linter can't judge: naming semantics, docstring completeness, comment quality/staleness, consistency of `return` usage.
2. Read the linter config for project-specific overrides (line length, ignored rules) and apply those instead of PEP8 defaults. Don't flag a 100-char line as a violation if the project's own config sets `line-length = 100`.
3. Walk the changed/reviewed files against the checklist below. Report violations with `file:line` and the specific rule broken.

## Checklist

### Layout
- 4 spaces per indentation level, never tabs; never mix the two
- Line length ≤ 79 chars for code, ≤ 72 for docstrings/comments — unless the project's linter config overrides this
- Two blank lines around top-level functions/classes; one blank line between methods
- Break before binary operators when splitting a long expression across lines
- No trailing whitespace on any line

### Naming
| Element | Convention | Example |
|---|---|---|
| Module/package | lowercase_with_underscores | `my_module.py` |
| Class | CapWords | `MyClass` |
| Function/method/variable | lowercase_with_underscores | `my_function` |
| Constant | UPPER_CASE_WITH_UNDERSCORES | `MAX_SIZE` |
| Internal/non-public | leading underscore | `_internal` |
| Name-mangled class attribute | leading double underscore | `__private` |
| Exception class | suffix `Error` | `ValidationError` |

- Avoid single-character names `l`, `O`, `I` — easily confused with digits or each other
- Avoid shadowing builtins/keywords; append a trailing underscore instead (`class_`, `list_`)

### Whitespace
- No spaces immediately inside parentheses, brackets, or braces
- Single space around binary operators (`=`, `==`, `<`, `and`, `or`, `is`, etc.)
- No space around `=` for keyword arguments or default parameter values
- No space before a comma, colon, or semicolon; one space after (except slice colons, where spacing is symmetric or omitted)

### Imports
- One import per line, except `from module import a, b`
- Grouped and blank-line-separated: standard library → third-party → local application
- Absolute imports preferred over implicit relative imports
- No wildcard imports (`from module import *`)
- Imports live at the top of the file, after the module docstring, before module-level constants

### Comments & Docstrings
- Block comments: same indentation as the code they describe, start with `# `, written as complete sentences
- Inline comments: at least two spaces before the `#`, used sparingly, never stating the obvious
- Every public module, function, class, and method has a docstring
- A multi-line docstring's closing `"""` sits on its own line; one-liners keep it on the same line
- Keep comments synchronized with the code — a stale comment is worse than no comment

### Strings
- Pick single or double quotes consistently within a file; switch only to avoid escaping an embedded quote of the same kind
- Triple-quoted docstrings use `"""`

### Programming Recommendations
- Compare to `None` with `is`/`is not`, never `==`
- Use `isinstance()` for type checks, not `type(x) == Y`
- Never use a bare `except:` — name the exception type(s) being caught
- Use `with` statements for resource management (files, locks, connections)
- Prefer `.startswith()`/`.endswith()` over string slicing for prefix/suffix checks
- Be consistent within a function: either every `return` carries a value, or none do

## When Not to Flag

PEP8 itself says consistency with surrounding code, backward compatibility, and a project's own configured style overrides its defaults. Before filing a finding, confirm it isn't already permitted by the project's linter config — a violation only under the vanilla 79-char/no-override reading of PEP8 is not a real finding if the project has explicitly configured something else.

## Output

For each real violation: `file:line — rule broken — one-line fix suggestion`, grouped by file. If nothing violates, say so plainly rather than inventing nitpicks.
