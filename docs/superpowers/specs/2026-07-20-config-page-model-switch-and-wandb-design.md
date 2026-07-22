# Config Page: Live W&B Logging — Design Spec

## Motivation

Third of the four frontend increments (table → timeline → **config page** → visual
redesign). Originally scoped to also let you switch which trained model serves `/predict`
(e.g. BETO v1 vs. BETO v2) — dropped: you've confirmed only BETO v2 is actually used, so a
model-switch UI would be a dropdown with nothing meaningful to switch to. This spec now covers
just the one real remaining capability:

**Log live `/predict` calls to W&B.** Nothing today does this. `WandbLogger` (`src/wandb.py`)
is training-shaped — `init()` requires a `Hyperparams` object that doesn't exist for a live
inference session, so it can't be reused as-is. `log_predict_folder_results()` is a one-shot
"log this whole finished batch" function, not an incremental per-request logger. Both are
staying exactly as they are; this spec adds a third, purpose-built piece rather than bending
either to a shape it wasn't built for.

## Scope

- Backend: one new endpoint (`POST /config/wandb`) to toggle live logging, plus wiring
  `/predict` to log when the toggle is on.
- Frontend: a `Config` view (same tab-switch mechanism the timeline spec introduces) with a
  single W&B logging checkbox — no model dropdown.
- **Deliberately out of scope:** authentication/access control on the new endpoint (this is
  an internal dev tool served on a private network today, matching every other endpoint's
  current lack of auth — not a new gap this spec introduces). Model switching entirely —
  not deferred, dropped: revisit as a separate spec if a second model ever comes back into
  active use.

## Design

### Live W&B logging

A new small module (`src/api/wandb_session.py`) — deliberately not a `WandbLogger` method,
since `WandbLogger` is scoped to a single training run's lifecycle (its own `init`/`finish`
pairing tied to `Hyperparams`), a different shape than "an API server's logging toggle that
outlives any individual request."

```python
class ApiWandbSession:
    """One long-lived W&B run per toggle-on period -- turned on, logs every /predict
    result as its own step in this run's history until turned off. Not a wandb.Table
    (which is meant to be logged as one finished snapshot) -- each prediction becomes one
    wandb.log() call, which W&B's own UI already renders as both a table (History) and,
    for free, exactly the per-metric line charts the timeline view's brainstorming
    referenced -- a live W&B session gets you that chart with zero extra code on our side."""

    def __init__(self) -> None:
        self._run: wandb.sdk.wandb_run.Run | None = None

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def start(self) -> None:
        if self._run is None:
            self._run = wandb.init(
                entity=Settings.WANDB_ENTITY, project=Settings.WANDB_PROJECT, job_type="api"
            )

    def stop(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None

    def log(self, result: PredictResult) -> None:
        if self._run is not None:
            self._run.log(flatten_predict_result(result))
```

`app.state.wandb_session = ApiWandbSession()` constructed once in `create_app()`'s
`lifespan`, alongside `app.state.clf`. `POST /config/wandb {"enabled": true|false}` calls
`start()`/`stop()`. The existing `/predict` endpoint
(`src/api/routes/predict/endpoints.py`) gets one added line after building its
`PredictResult`: `request.app.state.wandb_session.log(result)`.

### Frontend `Config` view (`frontend/src/components/ConfigView.tsx`, new)

```typescript
export function ConfigView() {
  const [wandbEnabled, setWandbEnabled] = useState(false);

  async function handleWandbToggle(enabled: boolean) {
    await fetch("/config/wandb", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    });
    setWandbEnabled(enabled);
  }

  // renders a single checkbox bound to handleWandbToggle
}
```

## Touch list

| Path | Change |
|---|---|
| `src/api/routes/config/` (new package) | `endpoints.py`, `schemas.py` — `POST /config/wandb` |
| `src/api/wandb_session.py` (new) | `ApiWandbSession` |
| `src/api/app.py` | `create_app()`'s `lifespan` also constructs `app.state.wandb_session`; register the new router |
| `src/api/routes/predict/endpoints.py` | one added line logging to `wandb_session` if enabled |
| `frontend/src/components/ConfigView.tsx` (new) | the page |
| `frontend/src/App.tsx` | third tab |

## Backward compatibility

- Existing `/predict` behavior is unchanged when W&B logging is off (the default) — the
  added line is a no-op (`ApiWandbSession.log()` checks `self._run is not None` first).
- No existing test constructs `app.state` manually in a way this would break — `create_app()`
  remains the only place `app.state.clf`/`app.state.wandb_session` are set.
