# Frontend: Full W&B-Style Results Table — Design Spec

## Motivation

The current frontend (`frontend/`, built on `feature/react-frontend`, PR #63 closed
unmerged) shows only the essentials per result (filename/label/confidence/certain/review
route) with a click-to-expand panel for the rest. The user wants one flat, wide table
instead — matching W&B's own predictions table (screenshots reviewed during
brainstorming): every field as its own column, horizontally scrollable, with a
show/hide-columns control, mirroring W&B's "Columns" button.

**Scope, confirmed during brainstorming:** one table, not two. This replaces
`ResultsTable`'s current essentials-row-plus-expand pattern and removes `ResultDetail`
entirely — every field becomes a togglable column in the same table instead of hidden
behind a click.

## Column set — matches `_PREDICTION_COLUMNS` exactly, not all of `PredictResponse`

`src/wandb.py`'s `_PREDICTION_COLUMNS` (21 fields) is already this project's own answer to
"which fields matter enough to show in a table view" — it's the literal source of the W&B
table in the user's screenshots. Matching it exactly, rather than dumping every
`PredictResponse` field, means the new frontend table shows precisely what the user is
already used to seeing on the W&B site, and deliberately excludes `extractedText`/
`allScores` — both real fields, both already deliberately left out of `_PREDICTION_COLUMNS`
(confirmed by reading the actual list, not assumed).

| Column | Source field | Notes |
|---|---|---|
| filename | `filename` | |
| label | `label` | |
| confidence | `confidence` | |
| certain | `certain` | |
| mahalanobis_p_value | `oodMetrics.mahalanobisPValue` | flattened out of the nested `oodMetrics` |
| mahalanobis_p_value_theoretical | `oodMetrics.mahalanobisPValueTheoretical` | |
| cosine_z | `oodMetrics.cosineZ` | |
| knn_distance | `oodMetrics.knnDistance` | |
| tfidf_cosine_z | `oodMetrics.tfidfCosineZ` | |
| in_distribution | `oodMetrics.inDistribution` | |
| mahalanobis_calibration_status | `oodMetrics.mahalanobisCalibrationStatus` | |
| cosine_calibration_status | `oodMetrics.cosineCalibrationStatus` | |
| knn_distance_calibration_status | `oodMetrics.knnDistanceCalibrationStatus` | |
| tfidf_calibration_status | `oodMetrics.tfidfCalibrationStatus` | |
| foreign_municipality | `foreignMunicipality` | |
| foreign_municipality_context | `foreignMunicipalityContext` | |
| review_route | `reviewRoute` | |
| extractor_used | `extractorUsed` | |
| error | `error` | (or the client-side upload/network error, for a `PredictFailure`) |
| svm_scores.\<class\> | `svmScores[class]` | **one column per class name seen across the current result set**, not a single dict column — mirrors what the screenshots actually show (W&B auto-flattens a dict-valued column into sub-columns for display; the backend still logs/returns it as one dict, only the table view expands it) |
| svm_predicted_label | `svmPredictedLabel` | |
| svm_agrees_with_prediction | `svmAgreesWithPrediction` | |

All `oodMetrics.*` columns show a dash (`—`) when `oodMetrics` is `null` (no `ood_stats.npz`
loaded for the served model) — same convention `flatten_predict_result()` already uses on
the backend for the identical reason.

A `PredictFailure` row (a file that errored before reaching a real `PredictResponse`) shows
its `filename` and `error`, dashes everywhere else — same as today's failure-row handling,
just inside the wider table instead of a separate short row.

**Deliberately out of scope:** filtering (the screenshots also show a "Filter" button, but
the user's explicit ask was about columns, not filtering — TanStack Table's filtering
plugs into the same setup this spec adds, so it's a cheap follow-up, not a reason to expand
this spec now). Sorting is a similarly cheap, deferred follow-up for the same reason.

## Library choice: TanStack Table (`@tanstack/react-table`) v8

A new dependency — worth justifying given this project's stated bias toward minimal
dependencies (no component library, no CSS framework beyond the already-approved Tailwind).
The bar here: **column show/hide state management for a table whose column set grows
dynamically** (one column per class name, which varies by model) isn't a few lines to hand-roll
well — TanStack Table is the standard, headless (no imposed styling, works directly with
Tailwind classes) library for exactly this, with built-in `columnVisibility` state and a
`getIsVisible()`/`toggleVisibility()` API purpose-built for a "Columns" picker. Hand-rolling
this would mean re-implementing a well-established wheel for no real benefit — this is the
one case in the frontend so far where reaching for a library is the lazier, more correct
choice, not the reverse.

## Design

### `frontend/src/utils/flatten.ts` (new)

Mirrors `flatten_predict_result()` (`src/schema.py`) on the frontend — turns a
`PredictOutcome` into one flat row object keyed by the column names above.

```typescript
import type { PredictOutcome } from "../types/api";
import { isPredictFailure } from "../types/api";

export interface FlatResultRow {
  filename: string;
  label: string | null;
  confidence: number | null;
  certain: boolean | null;
  mahalanobisPValue: number | null;
  mahalanobisPValueTheoretical: number | null;
  cosineZ: number | null;
  knnDistance: number | null;
  tfidfCosineZ: number | null;
  inDistribution: boolean | null;
  mahalanobisCalibrationStatus: string | null;
  cosineCalibrationStatus: string | null;
  knnDistanceCalibrationStatus: string | null;
  tfidfCalibrationStatus: string | null;
  foreignMunicipality: string | null;
  foreignMunicipalityContext: string | null;
  reviewRoute: string | null;
  extractorUsed: string | null;
  error: string | null;
  svmScores: Record<string, number>;
  svmPredictedLabel: string | null;
  svmAgreesWithPrediction: boolean | null;
}

const NULL_ROW: Omit<FlatResultRow, "filename" | "error"> = {
  label: null,
  confidence: null,
  certain: null,
  mahalanobisPValue: null,
  mahalanobisPValueTheoretical: null,
  cosineZ: null,
  knnDistance: null,
  tfidfCosineZ: null,
  inDistribution: null,
  mahalanobisCalibrationStatus: null,
  cosineCalibrationStatus: null,
  knnDistanceCalibrationStatus: null,
  tfidfCalibrationStatus: null,
  foreignMunicipality: null,
  foreignMunicipalityContext: null,
  reviewRoute: null,
  extractorUsed: null,
  svmScores: {},
  svmPredictedLabel: null,
  svmAgreesWithPrediction: null,
};

export function flattenResult(outcome: PredictOutcome): FlatResultRow {
  if (isPredictFailure(outcome)) {
    return {
      filename: outcome.filename,
      error: outcome.error,
      // every other field null -- rendered as "—" by the cell formatter
      ...NULL_ROW,
    };
  }
  const ood = outcome.oodMetrics;
  return {
    filename: outcome.filename,
    label: outcome.label,
    confidence: outcome.confidence,
    certain: outcome.certain,
    mahalanobisPValue: ood?.mahalanobisPValue ?? null,
    mahalanobisPValueTheoretical: ood?.mahalanobisPValueTheoretical ?? null,
    cosineZ: ood?.cosineZ ?? null,
    knnDistance: ood?.knnDistance ?? null,
    tfidfCosineZ: ood?.tfidfCosineZ ?? null,
    inDistribution: ood?.inDistribution ?? null,
    mahalanobisCalibrationStatus: ood?.mahalanobisCalibrationStatus ?? null,
    cosineCalibrationStatus: ood?.cosineCalibrationStatus ?? null,
    knnDistanceCalibrationStatus: ood?.knnDistanceCalibrationStatus ?? null,
    tfidfCalibrationStatus: ood?.tfidfCalibrationStatus ?? null,
    foreignMunicipality: outcome.foreignMunicipality,
    foreignMunicipalityContext: outcome.foreignMunicipalityContext,
    reviewRoute: outcome.reviewRoute,
    extractorUsed: outcome.extractorUsed,
    error: outcome.error,
    svmScores: outcome.svmScores,
    svmPredictedLabel: outcome.svmPredictedLabel,
    svmAgreesWithPrediction: outcome.svmAgreesWithPrediction,
  };
}
```

### `frontend/src/components/PredictionsTable.tsx` (new, replaces `ResultsTable.tsx`)

- Builds `FlatResultRow[]` via `flattenResult()` for the current `results` array.
- Computes the dynamic `svmScores.<class>` column set as the union of keys seen across all
  rows (so a batch mixing models, if that ever happens, still shows every class column
  rather than only the first row's).
- Column defs: the 20 fixed columns above, plus one generated per class name.
- `useReactTable` with `getCoreRowModel()` + column-visibility state (`columnVisibility` +
  `onColumnVisibilityChange`).
- A "Columns" button (top-right, matching the screenshots' own placement) opens a checklist
  of every column, each toggled via TanStack Table's own `column.getToggleVisibilityHandler()`.
- Table wrapped in `overflow-x-auto` for horizontal scroll on the wide column set (matching
  this project's own artifact/responsive-design convention already used elsewhere).
- Cell formatting: `null`/`undefined` → `—`; `boolean` → `yes`/`no`; numbers rounded to a
  fixed precision matching what the backend itself already rounds to
  (`OodMetrics.mahalanobisPValue` etc. are already rounded server-side, e.g. `round(x, 6)` /
  `round(x, 4)` in `OodScorer.score()` — the frontend doesn't need to re-round, just display
  as received).

### `frontend/src/App.tsx`

`<ResultsTable results={results} />` → `<PredictionsTable results={results} />`. No other
change — `App.tsx` still owns `results` state and the upload flow exactly as before.

## Touch list

| Path | Change |
|---|---|
| `frontend/package.json` | add `@tanstack/react-table` |
| `frontend/src/utils/flatten.ts` (new) | `flattenResult()` |
| `frontend/src/components/PredictionsTable.tsx` (new) | the table itself |
| `frontend/src/components/ResultsTable.tsx` | deleted |
| `frontend/src/components/ResultDetail.tsx` | deleted |
| `frontend/src/App.tsx` | swap component |

**Not touched:** any backend file — every field this table needs is already returned by the
existing `/predict` response; this is a frontend-only change.

## Backward compatibility / continuation

Continues on `feature/react-frontend` (not a new branch) — that branch/PR #63 is closed but
unmerged, and this is a direct extension of the same not-yet-shipped frontend rather than an
independent unit of work.
