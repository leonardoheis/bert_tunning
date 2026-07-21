# Config Page: Model Switch + Live W&B Logging — Design Spec

## Motivation

Third of the four frontend increments (table → timeline → **config page** → visual
redesign). Unlike the first two, this one needs real backend work — two genuinely new
capabilities, not just new frontend surface over existing data:

1. **Switch which trained model serves `/predict`.** Today, `create_app()`
   (`src/api/app.py`) constructs exactly one `BertTunningClassifier` at FastAPI startup
   (`lifespan`), from `Settings.default_model_path`, and never touches it again. There is no
   code path that swaps it — this spec adds one.
2. **Log live `/predict` calls to W&B.** Nothing today does this. `WandbLogger`
   (`src/wandb.py`) is training-shaped — `init()` requires a `Hyperparams` object that
   doesn't exist for a live inference session, so it can't be reused as-is.
   `log_predict_folder_results()` is a one-shot "log this whole finished batch" function,
   not an incremental per-request logger. Both are staying exactly as they are; this spec
   adds a third, purpose-built piece rather than bending either to a shape it wasn't built
   for.

## Scope

- Backend: two new endpoints (list available models + switch model; toggle W&B logging),
  plus wiring `/predict` to log when the toggle is on.
- Frontend: a `Config` view (same tab-switch mechanism the timeline spec introduces) with a
  model dropdown and a W&B logging checkbox.
- **Deliberately out of scope:** authentication/access control on the new endpoints (this is
  an internal dev tool served on a private network today, matching every other endpoint's
  current lack of auth — not a new gap this spec introduces). Concurrent-request safety
  during a model swap is handled with the simplest option (see below), not a full
  read-write lock — disclosed as a deliberate simplification, not an oversight.

## Design

### Discovering available models (`src/api/routes/config/`, new)

Scans `models/*/final/` for directories containing `config.json` — the same signal
`AutoModelForSequenceClassification.from_pretrained` itself needs to succeed, so "listed as
available" and "actually loadable" can't drift apart. Verified live against the current
`models/` directory: this correctly finds `bert_tunning_model_beto`,
`bert_tunning_model_beto_v2`, `bert_tunning_model_minilm`,
`bert_tunning_model_xlmroberta_v1`, `bert_tunning_model_xmlroberta_v2`, and correctly
excludes a stray `bert_tunning_model_beto_v2;C` directory that lacks `final/config.json`.

```python
def list_available_models() -> list[str]:
    models_root = Path("models")
    return sorted(
        d.name
        for d in models_root.iterdir()
        if d.is_dir() and (d / "final" / "config.json").is_file()
    )
```

Each entry is the directory name itself (`bert_tunning_model_beto_v2`) — not a
`MODEL_REGISTRY` key (`xlm-roberta`/`beto`/`minilm`), which is a training-time architecture
choice, unrelated to which trained checkpoint is being served. This also matches how you've
been referring to these models throughout this project ("BETO v2", "BETO v1") — the
directory name is already the meaningful identifier.

### Swapping the served model

`app.state.clf` is reassigned in place — `BertTunningClassifier.__init__` already does
everything needed (load tokenizer/model, `OodScorer.load`+`validate`, SVM classifiers
load+validate) given just a path, so reloading is exactly "construct a new one and replace
the reference."

```python
# src/api/routes/config/endpoints.py
@router.post("/config/model")
def switch_model(request: Request, body: SwitchModelRequest) -> SwitchModelResponse:
    model_dir = body.model_name
    if model_dir not in list_available_models():
        raise HTTPException(404, f"Unknown model: {model_dir}")
    model_path = str(Path("models") / model_dir / "final")
    request.app.state.clf = BertTunningClassifier(
        model_path, confidence_threshold=Settings.model_threshold
    )
    return SwitchModelResponse(active_model=model_dir)
```

**Concurrency, disclosed simplification:** no lock guards the reassignment against an
in-flight `/predict` request. `app.state.clf = BertTunningClassifier(...)` is a single
Python reference assignment (atomic under the GIL) — a request already mid-flight keeps
using the classifier instance it grabbed at the start of its own handler, so no request
crashes mid-execution; the only real effect is a request that starts *during* the ~few
seconds a reload takes might use either the old or new classifier, non-deterministically.
For an internal tool where a human explicitly clicks "switch model" and then presumably
waits before uploading again, this is an acceptable, disclosed simplification, not
something worth a locking mechanism for. Revisit if this ever serves concurrent production
traffic.

`BertTunningClassifier.__init__` raising (e.g. a class-mismatch `BertTunningError` if the
`ood_stats.npz`/`svm_classifiers.joblib` next to the new model don't match its own
`id2label`) propagates as a 500 — `app.state.clf` is left holding whatever it held before
the failed assignment even started (the old classifier was never touched), so a failed
switch doesn't take the API down, it just doesn't switch.

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
  const [models, setModels] = useState<string[]>([]);
  const [activeModel, setActiveModel] = useState<string | null>(null);
  const [wandbEnabled, setWandbEnabled] = useState(false);
  const [switching, setSwitching] = useState(false);

  useEffect(() => {
    fetch("/config/models").then((r) => r.json()).then((data) => {
      setModels(data.models);
      setActiveModel(data.activeModel);
    });
  }, []);

  async function handleSwitch(modelName: string) {
    setSwitching(true);
    const res = await fetch("/config/model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ modelName }),
    });
    if (res.ok) {
      setActiveModel((await res.json()).activeModel);
    }
    setSwitching(false);
  }

  async function handleWandbToggle(enabled: boolean) {
    await fetch("/config/wandb", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    });
    setWandbEnabled(enabled);
  }

  // renders a <select> bound to handleSwitch, a checkbox bound to handleWandbToggle,
  // disables the model select while switching is true (a reload takes a few real seconds)
}
```

## Touch list

| Path | Change |
|---|---|
| `src/api/routes/config/` (new package) | `endpoints.py`, `schemas.py` — `GET /config/models`, `POST /config/model`, `POST /config/wandb` |
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
