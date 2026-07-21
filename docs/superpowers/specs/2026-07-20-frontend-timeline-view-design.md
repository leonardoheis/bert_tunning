# Frontend: Timeline View — Design Spec

## Motivation

Second of the four frontend increments (table → **timeline** → config page → visual
redesign). The user's reference is W&B's charts view — a metric plotted across training
steps. That doesn't translate literally: a batch of independent PDF classifications has no
training-step axis. This spec's working assumption, **not yet confirmed by the user, flagged
explicitly for the spec-review step**: the honest equivalent is *each uploaded file, in
upload order, plotted by a selectable metric* — confidence, or one of the OOD signals
(Mahalanobis p, cosine z, k-NN distance, TF-IDF cosine z). If that's not what "timeline"
meant, this is the section to correct before implementation starts.

## Scope

- Frontend-only. Every field this view needs (`confidence`, `oodMetrics.*`, `label`,
  `reviewRoute`) is already in the `results` state `App.tsx` already holds after a batch
  upload — no backend change.
- One chart, one metric visible at a time, selected via a dropdown — not a dashboard of
  simultaneous charts (matching this project's YAGNI bias; a multi-metric dashboard is a
  natural later extension of the same component, not something to build speculatively now).
- X-axis: upload order within the current batch (index 1..N, labeled by filename on hover).
  Not wall-clock time — there's no meaningful time delta between files uploaded together in
  one batch submit (`Promise.allSettled`, all requests fired at once).

## Library: Recharts

A new dependency, same justification bar as `@tanstack/react-table` in the previous
increment: a line/scatter chart with axis labels, tooltips, and per-point coloring is not a
few lines to hand-roll well, and Recharts is the standard, most-used React charting
library — composable via JSX (`<LineChart><Line .../></LineChart>`), works directly with
Tailwind for surrounding layout, no imposed design system to fight.

## Design

### `frontend/src/utils/flatten.ts`

Unchanged — `FlatResultRow` already has every field this view needs.

### `frontend/src/components/TimelineView.tsx` (new)

```typescript
import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { PredictOutcome } from "../types/api";
import { flattenResult } from "../utils/flatten";

const METRICS = [
  { key: "confidence", label: "Confidence" },
  { key: "mahalanobisPValue", label: "Mahalanobis p" },
  { key: "cosineZ", label: "Cosine z" },
  { key: "knnDistance", label: "k-NN distance" },
  { key: "tfidfCosineZ", label: "TF-IDF cosine z" },
] as const;

export function TimelineView({ results }: { results: PredictOutcome[] }) {
  const [metric, setMetric] = useState<(typeof METRICS)[number]["key"]>("confidence");

  const points = useMemo(
    () =>
      results.map((outcome, index) => {
        const row = flattenResult(outcome);
        return { index: index + 1, filename: row.filename, value: row[metric] };
      }),
    [results, metric],
  );

  if (results.length === 0) {
    return null;
  }

  return (
    <div>
      <div className="mb-2 flex items-center gap-2">
        <label htmlFor="metric-select" className="text-sm text-gray-600">
          Metric
        </label>
        <select
          id="metric-select"
          value={metric}
          onChange={(e) => setMetric(e.target.value as (typeof METRICS)[number]["key"])}
          className="rounded border border-gray-300 px-2 py-1 text-sm"
        >
          {METRICS.map((m) => (
            <option key={m.key} value={m.key}>
              {m.label}
            </option>
          ))}
        </select>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={points}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="index" label={{ value: "Upload order", position: "bottom" }} />
          <YAxis />
          <Tooltip
            formatter={(value: number) => value}
            labelFormatter={(index: number) => points[index - 1]?.filename ?? ""}
          />
          <Line type="monotone" dataKey="value" stroke="#2563eb" connectNulls />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
```

`points[index].value` is `null` for a `PredictFailure` row or when `oodMetrics` is absent
for the selected metric — Recharts' `connectNulls` on `<Line>` skips gaps rather than
breaking the line into disconnected fragments, which reads better for a mostly-complete
batch with the occasional failed file.

### `frontend/src/App.tsx`

Needs a place to put a second view without a full navigation rebuild (that's the visual
redesign increment's job, not this one). Simplest addition that doesn't front-load a
routing library before it's earned: a plain tab-state switch.

```typescript
const [view, setView] = useState<"table" | "timeline">("table");
// ...
<div className="mb-4 flex gap-2 border-b border-gray-200">
  <button onClick={() => setView("table")} className={/* active/inactive styles */}>Table</button>
  <button onClick={() => setView("timeline")} className={/* ... */}>Timeline</button>
</div>
{view === "table" ? <PredictionsTable results={results} /> : <TimelineView results={results} />}
```

**Explicitly not a router.** No URL-addressability requirement was stated, and introducing
`react-router` for two same-page view-modes would be adding a dependency ahead of a real
need — the same reasoning that kept routing out of the original frontend spec ("no React
Router... until there's a second page" — this is that second page, but a tab switch covers
it without a new dependency). Revisit if deep-linking to a specific view ever matters.

## Touch list

| Path | Change |
|---|---|
| `frontend/package.json` | add `recharts` |
| `frontend/src/components/TimelineView.tsx` (new) | the chart |
| `frontend/src/App.tsx` | tab state, renders `PredictionsTable` or `TimelineView` |

**Not touched:** any backend file.

## Open question for spec review

The metric-over-upload-order interpretation is a best guess at "timeline," not a confirmed
requirement — please confirm or redirect before this moves to implementation.
