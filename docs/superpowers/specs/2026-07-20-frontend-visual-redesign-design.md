# Frontend: W&B-Style Visual Redesign — Design Spec

## Motivation

Fourth and last of the four frontend increments (table → timeline → config page → **visual
redesign**), applied last on purpose: by this point the app has three real sections (Table,
Timeline, Config) sharing one ad-hoc tab-state switch introduced in the timeline spec. This
increment replaces that placeholder with the actual navigation shell the user's W&B
screenshots show — a dark sidebar, a workspace-style layout — and is a styling/structure
pass over the three sections that already exist, not a feature in itself. Doing this first
would mean re-styling a one-section app, then re-doing the nav shell twice more as each
section got added; doing it last means one navigation shell, built once, for the final
three-section shape.

## Scope

- Frontend-only, no backend changes.
- A persistent sidebar (matching the screenshots: dark background, nav items for
  Table/Timeline/Config, a project/workspace name at the top).
- A dark color scheme for the whole app, not just the sidebar — matching the W&B
  screenshots' overall dark workspace look, not a light page with a dark sidebar bolted on.
- Replaces the tab-state switch (`view: "table" | "timeline" | "config"` in `App.tsx`) with
  the same state, just rendered as sidebar nav items instead of top tabs — **still no
  router** (see the timeline spec's reasoning; nothing about this redesign introduces a
  URL-addressability requirement either).

## Design

### Layout shell (`frontend/src/components/Layout.tsx`, new)

```typescript
interface LayoutProps {
  active: "table" | "timeline" | "config";
  onNavigate: (view: "table" | "timeline" | "config") => void;
  children: React.ReactNode;
}

const NAV_ITEMS = [
  { key: "table" as const, label: "Predictions" },
  { key: "timeline" as const, label: "Timeline" },
  { key: "config" as const, label: "Config" },
];

export function Layout({ active, onNavigate, children }: LayoutProps) {
  return (
    <div className="flex min-h-screen bg-gray-950 text-gray-100">
      <aside className="w-56 shrink-0 border-r border-gray-800 bg-gray-900 p-4">
        <h1 className="mb-6 text-lg font-semibold">Bert Tunning</h1>
        <nav className="space-y-1">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.key}
              onClick={() => onNavigate(item.key)}
              className={`block w-full rounded px-3 py-2 text-left text-sm ${
                active === item.key
                  ? "bg-gray-800 text-white"
                  : "text-gray-400 hover:bg-gray-800/50 hover:text-gray-200"
              }`}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </aside>
      <main className="flex-1 overflow-x-auto p-6">{children}</main>
    </div>
  );
}
```

### `frontend/src/App.tsx`

`view` state stays exactly as the timeline/config specs left it — only the tab-button JSX
gets replaced by `<Layout active={view} onNavigate={setView}>...</Layout>` wrapping the
existing per-view render logic.

### Existing components need dark-mode-compatible classes

`PredictionsTable`, `FileUploadForm`, `TimelineView`, `ConfigView` were all built with
light-mode Tailwind classes (`bg-white`, `text-gray-900`, `border-gray-200`, etc.) — those
need updating to dark equivalents (`bg-gray-900`, `text-gray-100`, `border-gray-800`) to
actually look like one consistent dark workspace rather than dark sidebar + light content
panels. This is the bulk of the real work in this increment — a pass through every existing
component's `className` strings, not just the new `Layout` component.

## Touch list

| Path | Change |
|---|---|
| `frontend/src/components/Layout.tsx` (new) | sidebar + main content shell |
| `frontend/src/App.tsx` | wrap render in `<Layout>` |
| `frontend/src/components/PredictionsTable.tsx` | dark-mode class pass |
| `frontend/src/components/FileUploadForm.tsx` | dark-mode class pass |
| `frontend/src/components/TimelineView.tsx` | dark-mode class pass (chart colors need adjusting too — Recharts' default axis/grid colors assume a light background) |
| `frontend/src/components/ConfigView.tsx` | dark-mode class pass |

## Backward compatibility

N/A — purely additive/visual, no data or API contract changes. Depends on the timeline and
config-page increments already existing (this spec assumes `App.tsx`'s three-way `view`
state from those two specs is already in place).
