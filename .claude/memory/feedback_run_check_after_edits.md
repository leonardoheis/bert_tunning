---
name: feedback-run-check-after-edits
description: After every code modification in this project, run uv run poe check before reporting done
metadata:
  type: feedback
---

After every code modification (edit, write, or set of related edits), run `uv run poe check` before reporting the task as complete.

**Why:** User wants lint + typecheck + tests to be verified as part of each change, not left as a manual step.

**How to apply:** Any time I edit `.py` files or project config, run `uv run poe check` at the end and report the result. If it fails, fix the issues before declaring done.
