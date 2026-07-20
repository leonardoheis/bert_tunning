# React Frontend for PDF Classification — Design Spec

## Motivation

The project has a working FastAPI backend (`POST /predict`, multipart PDF upload,
`src/api/`) and a CLI, but no way to try it visually — testing today means Swagger UI,
`curl`, or the CLI's `predict`/`predict-folder` commands. This spec adds a React frontend:
upload one file or several, see the classification result(s).

## Scope decisions made during brainstorming

- **No batch API endpoint.** The backend only has single-file `POST /predict`
  (confirmed — `src/api/routes/predict/endpoints.py` is the only `/predict` route, no
  batch route exists anywhere in `src/api/routes/`). "Batch" in the UI means the frontend
  calling `/predict` once per selected file and assembling a results table — no backend
  changes.
- **Result detail: essentials + expandable.** Each result shows label/confidence/
  certain/review route at a glance; click to expand the full breakdown (OOD metrics, SVM
  scores, extracted text) for that one document. `PredictResponse` carries a lot
  (`ood_metrics`, `svm_scores`, `extracted_text`, etc.) — showing all of it for every row
  in a batch of 20 would be unreadable.
- **Deployment: one Docker image, via `StaticFiles`, not a second live process.** Unlike a
  Streamlit-style frontend (a live Python process needing its own launcher), a React build
  is static files. `create_app()` mounts the built `frontend/dist/` directly — `python -m
  src` (already the Docker `CMD`) serves both the API and the frontend from the same
  uvicorn process, no changes to `src/__main__.py`'s multiprocessing structure. Dev mode is
  separate: Vite's dev server (hot reload) proxies API calls to a locally-running backend —
  this is a dev-convenience detail, not part of the production path.
- **Placement: top-level `frontend/`, sibling to `src/`**, not nested inside it. `src/` in
  this repo is exclusively the Python package root (confirmed: `[tool.uv] package = false`,
  no non-Python directories exist under it today). A Node/React project has its own
  toolchain (`package.json`, `node_modules`, its own build step) that doesn't belong inside
  a Python package directory. Docker doesn't require nesting either way — `COPY . /app` in
  the existing `Dockerfile` already copies the whole repo regardless of layout.

## Touch list

| Path | What it is |
|---|---|
| `frontend/` (new) | The whole React project — `package.json`, `vite.config.ts`, `eslint.config.js`, `.prettierrc`, `.husky/pre-commit`, `src/`, `index.html` |
| `src/api/app.py` | `create_app()` gains a `StaticFiles` mount serving `frontend/dist/` (production only — see below) |
| `Dockerfile` | New Node build stage (`npm ci && npm run build` inside `frontend/`), copies `frontend/dist/` into the final image |
| `docs/superpowers/specs/2026-07-19-react-frontend-design.md` | This spec |

**Not touched:** `src/api/routes/predict/`, `src/inference/`, any Python business logic — this is additive only, no existing behavior changes. `src/__main__.py` needs zero edits (see Motivation).

## Design

### Stack

- **Vite + React + TypeScript** (`npm create vite@latest frontend -- --template react-ts`).
  Vite over Create React App (deprecated, no longer recommended by the React team) or
  Next.js (a full framework — SSR/routing/server-components are unneeded complexity for a
  single-page tool hitting one API). TypeScript, not plain JS, to keep the API response
  shape honest at compile time — mirrors this codebase's own mypy-strict discipline on the
  Python side, and the `PredictResponse` contract (15 fields, several nested/optional) is
  exactly the kind of shape that's easy to typo in JS and have it silently show `undefined`.
- **Tailwind CSS** (v4, the current Vite-plugin-based setup — no `tailwind.config.js`/
  PostCSS config needed):
  ```bash
  npm install tailwindcss @tailwindcss/vite
  ```
  ```typescript
  // vite.config.ts
  import tailwindcss from "@tailwindcss/vite";

  export default defineConfig({
    plugins: [react(), tailwindcss()],
  });
  ```
  ```css
  /* frontend/src/index.css */
  @import "tailwindcss";
  ```
  No component library (MUI/Chakra/etc.) — Tailwind is utility classes, not pre-built
  components, so it doesn't pull in a component API surface this small a UI doesn't need.
- **No React Router.** One screen (upload + results). A router is unneeded ceremony until
  there's a second page.

### Dev tooling: ESLint + Prettier + Husky + lint-staged

Modeled on [denivladislav/react-vite-ts-setup](https://github.com/denivladislav/react-vite-ts-setup)
(verified against its actual `package.json`/`eslint.config.js`/`.husky/pre-commit`, not just
the README) — same spirit as this repo's own `ruff`+`mypy` pre-commit discipline on the
Python side, applied to the frontend.

```bash
npm install -D eslint @eslint/js typescript-eslint eslint-plugin-react-hooks \
  eslint-plugin-react-refresh eslint-config-prettier prettier globals \
  husky lint-staged
```

`eslint.config.js` (flat config — ESLint 9+, this is what `npm create vite@latest ...
--template react-ts` already scaffolds a starting point for):

```javascript
import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import tseslint from "typescript-eslint";
import eslintConfigPrettier from "eslint-config-prettier";

export default tseslint.config(
  { ignores: ["dist"] },
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommended, eslintConfigPrettier],
    files: ["**/*.{ts,tsx}"],
    languageOptions: { ecmaVersion: 2020, globals: globals.browser },
    plugins: { "react-hooks": reactHooks, "react-refresh": reactRefresh },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "react-refresh/only-export-components": ["warn", { allowConstantExport: true }],
    },
  },
);
```

`eslint-config-prettier` disables every ESLint rule that'd conflict with Prettier's own
formatting opinions — ESLint handles code-quality rules, Prettier handles formatting, no
fighting between the two.

`.husky/pre-commit`:
```
npx lint-staged
```

`package.json`:
```json
{
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "lint": "eslint .",
    "preview": "vite preview",
    "prepare": "husky"
  },
  "lint-staged": {
    "src/**/*.{ts,tsx}": ["eslint --fix", "prettier --write"],
    "src/**/*.{json,css,md}": ["prettier --write"]
  }
}
```

`npm install` runs `prepare` automatically (npm's lifecycle script), which runs `husky` to
install the git hooks — no manual setup step beyond `npm install`.

### `create_app()` change (`src/api/app.py`)

```python
from pathlib import Path
from fastapi.staticfiles import StaticFiles

_FRONTEND_DIST = Path(__file__).parent.parent.parent / "frontend" / "dist"

def create_app(model_path: str, threshold: float = 0.70) -> FastAPI:
    ...
    for router in ROUTERS:
        app.include_router(router)

    if _FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")

    return app
```

The `is_dir()` guard means local `poe`/pytest runs (no `frontend/dist/` built) are
unaffected — the mount simply doesn't happen, existing API-only behavior is unchanged. In
Docker, the build stage always produces `frontend/dist/` before the Python stage runs, so
the mount is always active there.

### Dockerfile addition

A new stage before the existing builder stage:

```dockerfile
FROM node:22-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build
```

The existing final stage gains one more `COPY --from=frontend-builder`:

```dockerfile
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist
```

Everything else in the existing `Dockerfile` (the uv builder stage, the Python runtime
stage, `CMD ["python", "-m", "src"]`) is unchanged.

### Types (`frontend/src/types/api.ts`)

```typescript
export interface OodMetrics {
  mahalanobisPValue: number;
  mahalanobisPValueTheoretical: number;
  cosineZ: number;
  knnDistance: number;
  tfidfCosineZ: number | null;
  inDistribution: boolean;
  mahalanobisCalibrationStatus: "calibrated" | "not_calibrated" | "refused_degenerate";
  cosineCalibrationStatus: "calibrated" | "not_calibrated";
  knnDistanceCalibrationStatus: "calibrated" | "not_calibrated";
  tfidfCalibrationStatus: "calibrated" | "not_calibrated" | null;
}

export interface PredictResponse {
  filename: string;
  label: string | null;
  confidence: number;
  certain: boolean;
  allScores: Record<string, number>;
  error: string | null;
  oodMetrics: OodMetrics | null;
  extractedText: string;
  extractorUsed: string;
  reviewRoute: "" | "accept" | "llm_judge" | "human_review";
  foreignMunicipality: string | null;
  foreignMunicipalityContext: string | null;
  svmScores: Record<string, number>;
  svmPredictedLabel: string;
  svmAgreesWithPrediction: boolean;
}
```

Field names are camelCase, matching the actual wire format — `BaseSchema`
(`src/api/schema.py`) uses `alias_generator=to_camel`, and FastAPI serializes
`response_model`s by alias by default, so the JSON the backend actually sends is camelCase
already (verified against `src/api/routes/predict/schemas.py`'s field list).

### API client (`frontend/src/api.ts`)

A small typed wrapper, one function per backend call — no fetch library dependency needed,
`fetch` is sufficient for two endpoints.

```typescript
import type { PredictResponse } from "./types/api";

export async function predict(file: File): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch("/predict", { method: "POST", body: formData });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `Request failed: ${res.status}`);
  }
  return res.json();
}
```

### Components

Two small, justified additions from the boilerplate article's naming convention (`types/`,
`utils/`) — not the rest of its architecture (see above). Both hold real content, not empty
placeholder folders:

```
frontend/src/
├── main.tsx              # ReactDOM.createRoot, renders <App />
├── App.tsx               # top-level layout, owns the results list state
├── api.ts                # predict() -- fetch wrapper only, imports types from types/
├── types/
│   └── api.ts             # OodMetrics, PredictResponse interfaces (moved out of api.ts --
│                           # "shape of the data" is a different concern from "how we fetch
│                           # it," and any future file needing the type shouldn't have to
│                           # import it from a fetch module)
├── utils/
│   └── format.ts          # formatConfidence(n) -> "90.0%", reviewRouteLabel(route) -> display text --
│                           # pure functions ResultsTable/ResultDetail both need
├── components/
│   ├── FileUploadForm.tsx    # <input type="file" multiple> + "Classify" button
│   ├── ResultsTable.tsx      # one row per result: filename, label, confidence, certain, reviewRoute
│   └── ResultDetail.tsx      # expanded row content: OOD metrics, SVM scores, extracted text
└── index.css
```

- **`FileUploadForm`**: a single `<input type="file" multiple accept="application/pdf">` —
  selecting one file or several both flow through the same input (no separate "single" vs
  "batch" UI mode; the distinction is purely how many files were selected).
- **`App`**: on submit, calls `predict()` once per selected file (`Promise.allSettled`, not
  `Promise.all` — one failed file must not blank out the others' results), updates a
  `results: (PredictResponse | { filename: string; error: string })[]` state as each
  request resolves, so results appear incrementally rather than waiting for the slowest file.
- **`ResultsTable`**: renders the essentials row per result (via `utils/format.ts`'s
  `formatConfidence`/`reviewRouteLabel`); a failed request renders an error row (filename +
  error message) in the same table rather than a separate error UI.
- **`ResultDetail`**: only rendered for the currently-expanded row (click-to-toggle),
  reading straight from the already-fetched `PredictResponse` — no extra network call.

### Error handling

- A single failed upload (e.g. a corrupt PDF, a 413 for an oversized file) shows an error
  row for that file only — the rest of the batch keeps working, per `Promise.allSettled`
  above.
- Network failure (backend unreachable) surfaces the same way — `predict()`'s `catch`
  produces a `{ filename, error }` entry, same rendering path as a backend-returned error.

## Testing

- No backend tests needed — `src/api/app.py`'s change is additive and guarded by
  `is_dir()`; existing `tests/api/test_predict.py` (which never builds `frontend/dist/`)
  is unaffected.
- Frontend: out of scope for this first version. Vite's `react-ts` template ships without a
  test runner by default; adding Vitest/Testing Library is a reasonable follow-up once the
  UI itself exists, not a blocker for the first working version.

## Backward compatibility

- Zero behavior change for the existing API-only usage (CLI, `curl`, Swagger UI,
  `tests/api/test_predict.py`) — the static mount only activates when `frontend/dist/`
  exists, which it won't in any environment that doesn't run the frontend's build step.
