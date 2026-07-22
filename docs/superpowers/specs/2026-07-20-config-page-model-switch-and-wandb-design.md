# Config Page: Live W&B Logging — Design Spec

## Motivation

Third of the four frontend increments (table → timeline → **config page** → visual
redesign). Originally scoped to also let you switch which trained model serves `/predict`
(e.g. BETO v1 vs. BETO v2) — dropped: you've confirmed only BETO v2 is actually used, so a
model-switch UI would be a dropdown with nothing meaningful to switch to. This spec now covers
the two remaining capabilities:

1. **Log live `/predict` calls to W&B.** Nothing today does this. `WandbLogger`
   (`src/wandb.py`) is training-shaped — `init()` requires a `Hyperparams` object that
   doesn't exist for a live inference session, so it can't be reused as-is.
   `log_predict_folder_results()` is a one-shot "log this whole finished batch" function, not
   an incremental per-request logger. Both are staying exactly as they are; this spec adds a
   third, purpose-built piece rather than bending either to a shape it wasn't built for.
2. **Provide W&B credentials.** Nothing today lets you authenticate W&B from the running app —
   today that means `wandb login` in a shell, or a `WANDB_API_KEY` env var set before the
   container starts. The Config page needs a way to enter an API key at runtime too, defaulting
   to whatever credential is already available in the environment.

## Scope

- Backend: `POST /config/wandb` to toggle live logging, `POST /config/wandb/credentials` to
  set/override the W&B API key, `GET /config/wandb` to report current state — plus wiring
  `/predict` to log when the toggle is on.
- Frontend: a `Config` view (same tab-switch mechanism the timeline spec introduces) with a
  connection status indicator, a field to enter/override the API key, and a W&B logging
  checkbox — no model dropdown.
- **No database.** Confirmed this project has none (no `sqlite3`/`sqlalchemy` anywhere in
  `pyproject.toml`) — adding one solely to store a single credential for a single-user
  internal tool would be real overkill. Credential handling reuses the wandb SDK's own auth
  mechanism (`wandb.login()`, which itself writes to `~/.netrc`) instead of inventing storage.
  See "W&B credentials" below.
- **Deliberately out of scope:** authentication/access control on the new endpoints (this is
  an internal dev tool served on a private network today, matching every other endpoint's
  current lack of auth — not a new gap this spec introduces). Model switching entirely —
  not deferred, dropped: revisit as a separate spec if a second model ever comes back into
  active use. Persisting the entered key across container restarts — in-memory only, same
  restart-tolerant precedent already accepted for `_JOBS` (`src/api/routes/predict/endpoints.py`).

## Design

### W&B credentials

`ApiWandbSession` tracks its own `_authenticated` flag rather than probing wandb's internal
state — simpler and doesn't depend on undocumented SDK internals. At construction, it tries
`wandb.login()` with no key argument: the SDK itself checks `WANDB_API_KEY` (an env var, set
the same way `-e OUTPUT_DIR=...` already gets passed to `docker run` in this project — no new
`Settings` field needed, wandb reads it directly) and any existing `~/.netrc` entry, with no
interactive prompt in a non-TTY server process. If that succeeds, the credential already
present in the environment becomes the default — nothing to enter. `POST
/config/wandb/credentials {"apiKey": "..."}` lets you override it with a different key at
runtime, calling `wandb.login(key=api_key, relogin=True)`.

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
        # Picks up WANDB_API_KEY or an existing ~/.netrc automatically; wandb.login() with
        # no key argument doesn't prompt when there's no TTY (a server process never has
        # one), so this can't hang startup -- it just returns False if nothing's configured.
        self._authenticated = wandb.login(timeout=0)

    @property
    def authenticated(self) -> bool:
        return self._authenticated

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def set_credentials(self, api_key: str) -> bool:
        self._authenticated = wandb.login(key=api_key, relogin=True)
        return self._authenticated

    def start(self) -> None:
        if not self._authenticated:
            raise BertTunningError("W&B is not authenticated -- set an API key first")
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
`start()`/`stop()` (a `start()` call raising `BertTunningError` because `_authenticated` is
`False` propagates as a 400 — a clear, immediate rejection rather than `wandb.init()` failing
deep inside a later `/predict` call). `GET /config/wandb` returns
`{"authenticated": bool, "enabled": bool}` — **never the key itself**, only whether one is
configured. The existing `/predict` endpoint (`src/api/routes/predict/endpoints.py`) gets one
added line after building its `PredictResult`: `request.app.state.wandb_session.log(result)`.

### Frontend `Config` view (`frontend/src/components/ConfigView.tsx`, new)

```typescript
export function ConfigView() {
  const [authenticated, setAuthenticated] = useState(false);
  const [wandbEnabled, setWandbEnabled] = useState(false);
  const [apiKey, setApiKey] = useState("");

  useEffect(() => {
    fetch("/config/wandb")
      .then((r) => r.json())
      .then((data) => {
        setAuthenticated(data.authenticated);
        setWandbEnabled(data.enabled);
      });
  }, []);

  async function handleSetCredentials() {
    const res = await fetch("/config/wandb/credentials", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ apiKey }),
    });
    if (res.ok) {
      setAuthenticated(true);
      setApiKey(""); // never keep the key in state longer than the one request needs it
    }
  }

  async function handleWandbToggle(enabled: boolean) {
    await fetch("/config/wandb", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    });
    setWandbEnabled(enabled);
  }

  // renders: a "Connected"/"Not connected" indicator from `authenticated`, a password-type
  // input + "Connect" button bound to handleSetCredentials (visible whether or not already
  // authenticated, so an existing credential can be overridden), and the logging checkbox
  // bound to handleWandbToggle (disabled while !authenticated)
}
```

## Touch list

| Path | Change |
|---|---|
| `src/api/routes/config/` (new package) | `endpoints.py`, `schemas.py` — `GET /config/wandb`, `POST /config/wandb`, `POST /config/wandb/credentials` |
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
- A deployment with `WANDB_API_KEY` already set (or no W&B use at all) sees no behavior
  change — `ApiWandbSession.__init__`'s `wandb.login(timeout=0)` either picks up the existing
  credential silently (`authenticated=True`, nothing new to do) or returns `False`
  (`authenticated=False`, same as W&B being unused today) — either way nothing blocks startup.
