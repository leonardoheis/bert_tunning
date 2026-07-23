# Trim Docker Build Context / Image — Design Spec

## Motivation

`.dockerignore` already excludes the big offenders — `models/` (6.2 GB, confirmed via `du`),
`data/` (20 MB), `reports/`, `logs/`, `samples/`, `htmlcov/`, `docs/`, `.venv/`, `.git/`,
`.claude/`. Runtime models still get mounted at container start with `-v` (unchanged by this
spec, per your confirmation) — `models/` staying excluded is already correct, not a gap.

What's actually missing: `.agents/`, `.codegraph/`, `.codex/`, and `.github/` aren't in
`.dockerignore` at all, so `COPY . /app` (`Dockerfile`'s builder stage) currently pulls them
into the build context and bakes them into the image. Measured sizes:

| Directory | Size | What it is |
|---|---|---|
| `.codegraph/` | 4.5 MB | Local CodeGraph SQLite index — a dev-tool artifact, never read by any runtime code path |
| `.agents/` | 128 KB | Agent session state |
| `.codex/` | 2 KB | Codex session state |
| `.github/` | 1 KB | CI workflow config |

None of these are imported or read by `src/` at runtime — they're all editor/agent/CI tooling
state that has no reason to exist inside a running container.

## Scope

- `.dockerignore` only. No `Dockerfile` changes — the multi-stage build itself is unaffected,
  this just shrinks what gets sent into the build context and copied by `COPY . /app`.
- Also flagging `src/playground/` (724 KB, exploratory notebooks like
  `pca_train_vs_predict.ipynb`) as a same-rationale addition: nothing under `src/api`,
  `src/inference`, or `src/training` imports from it, so it has no runtime purpose either —
  including it unless you'd rather keep it out of this pass.

## Design

Add to `.dockerignore`:

```
.agents/
.codegraph/
.codex/
.github/
src/playground/
```

Placed alongside the existing `.claude/` line (same category: local tooling state, not
runtime code).

## Backward compatibility

None of these directories are referenced by anything under `COPY --from=builder` /
`COPY --from=frontend-builder` in the final image stage, and none are imported by `src/`
(verified: `src/playground/` isn't imported anywhere outside itself). Purely a build-context
reduction — no behavior change to the running container.

## Touch list

| Path | Change |
|---|---|
| `.dockerignore` | add `.agents/`, `.codegraph/`, `.codex/`, `.github/`, `src/playground/` |

## Open questions

Whether to include `src/playground/` in this pass, or leave it for a separate decision — your
call during review.
