# CLAUDE.md — Bert Tunning

## Project Purpose

Bert Tunning fine-tunes transformer models on Spanish municipal PDF documents to classify them by document type (decreto, ordenanza, resolución, etc.). Built for Argentine municipalities.

## Current State (June 2026)

Scaffold migration from flat-file layout to `src/` pipeline is complete. All 9 tasks implemented on `feature/scaffold-migration`. Final PR to `master` pending.

Branch strategy:
- `master` — stable, production-ready
- `feature/scaffold-migration` — integration branch, ready for PR to master
- `task/N-*` — per-task branches (all merged into `feature/scaffold-migration`)

## Tech Stack

| Tool | Purpose |
|---|---|
| Python ≥ 3.10 | Runtime — use `X \| Y` union types, not `Optional[X]` |
| PyTorch (CUDA 11.8) | Training backend |
| HuggingFace Transformers ≥ 4.46 | Model loading, Trainer API |
| XLM-RoBERTa base | Default model — stable multilingual, strong Spanish |
| Pydantic v2 | All config and schema objects — `frozen=True`, `alias_generator=to_camel` on API schemas |
| Click | CLI |
| FastAPI + Uvicorn | Inference API endpoint |
| wandb | Experiment tracking |
| MarkItDown + EasyOCR | PDF text extraction with OCR fallback |
| Docker | API containerization |
| uv | Package manager |
| ruff | Lint + format (line-length=100, ALL rules minus D/N/CPY) |
| mypy (strict) | Type checking |
| pytest | Testing |
| poethepoet (poe) | Task runner |

## Development Commands

```powershell
uv sync                  # install all deps including dev
uv run poe check         # lint + typecheck + test — run before every commit
uv run poe lint          # ruff check + format --check
uv run poe fmt           # auto-format with ruff
uv run poe typecheck     # mypy strict
uv run poe test          # pytest
uv run poe coverage      # pytest with HTML coverage report
```

## CLI

```powershell
uv run python main.py train --docs-root "C:\path\to\downloads"
uv run python main.py train --model beto --max-docs-per-class 100
uv run python main.py predict path/to/doc.pdf
uv run python main.py predict-folder path/to/folder
uv run python main.py serve --model-path ./models/bert_tunning_model/final
uv run python main.py clean
```

## Docker

```powershell
# Build
docker build -t bert-tunning .

# Run (mount trained model)
docker run -p 8000:8000 -v ./models/bert_tunning_model:/app/models/bert_tunning_model bert-tunning
```

The container starts the API via `python -m src` → `run_api()` → `create_app(Settings.default_model_path)`. Port 8000. No env vars required — `OUTPUT_DIR` in `src/settings.py` controls which model loads.

## Architecture (`src/` pipeline)

```
src/
├── __init__.py         package entry — exports BertTunningError, Settings, __version__
├── __main__.py         python -m src entry — spawns run_api() via multiprocessing.Process
├── settings.py         all configuration (Pydantic BaseSettings, overridable via .env)
├── schema.py           shared Pydantic schemas (PredictResult, Hyperparams, ReportDict)
├── exceptions.py       BertTunningError base
├── logger.py           setup_logging() → per-run timestamped log file
├── ingestion/
│   ├── extract.py      extract_pdf() — MarkItDown + EasyOCR fallback chain
│   ├── scan.py         folder walk, label mapping
│   ├── cache.py        parquet load/save
│   ├── pipeline.py     orchestrator: scan → extract → cache → DataFrame
│   └── extractors/
│       ├── _base.py    ExtractorBase ABC
│       ├── markitdown.py
│       └── ocr.py      OCRExtractor — thread-safe lazy init (double-checked locking)
├── training/
│   ├── models/
│   │   ├── __init__.py ModelConfig + MODEL_REGISTRY + get_model_config()
│   │   ├── config.py   ModelConfig (Pydantic, frozen)
│   │   ├── xlm_roberta.py
│   │   ├── beto.py
│   │   └── minilm.py
│   ├── options.py      TrainingRequest (Pydantic, frozen)
│   ├── split.py        stratified train/val/test split
│   ├── tokenize.py     BertTunningDataset, prepare_text
│   ├── trainer.py      WeightedTrainer, compute_metrics
│   ├── evaluate.py     run_evaluation() → EvaluationResult (Pydantic)
│   ├── reporting.py    generate_html_report() → reports/
│   ├── wandb_logger.py WandbLogger class
│   └── pipeline.py     orchestrates split → tokenize → train → evaluate
├── inference/
│   ├── classify.py     BertTunningClassifier, predict_text
│   └── pipeline.py     predict_pdf, predict_folder
├── api/
│   ├── app.py          create_app(model_path, threshold) → FastAPI
│   ├── schema.py       BaseSchema (camelCase aliases)
│   ├── __init__.py     run_api() — reads Settings.default_model_path
│   └── routes/
│       ├── health/     GET / and GET /health
│       └── predict/    POST /predict — multipart PDF upload
└── cli/
    ├── train.py        @click.command "train" — TrainOptions (Pydantic)
    ├── predict.py      @click.command "predict" + "predict-folder"
    └── clean.py        @click.command "clean"

Dockerfile              multi-stage build: uv builder + python:3.10-slim-bookworm runtime
.dockerignore           excludes .venv/, data/, models/, logs/, reports/, tests/, docs/
main.py                 Click group — train, predict, predict-folder, serve, clean
```

## Model Registry

Models are registered in `src/training/models/`. Each model file exports a `config: ModelConfig` instance.

```python
from src.training.models import MODEL_REGISTRY, get_model_config

cfg = get_model_config("xlm-roberta")  # raises KeyError with helpful msg if unknown
cfg = get_model_config("beto")
cfg = get_model_config("minilm")
```

Available keys: `xlm-roberta`, `beto`, `minilm`

To add a model: create `src/training/models/my_model.py` with `config = ModelConfig(...)`, then register it in `_build_registry()` in `src/training/models/__init__.py`.

`MODEL_KEY` in `src/settings.py` sets the default model for the CLI.

## Settings (`src/settings.py`)

All settings live in `_Settings(BaseSettings)` and are overridable via `.env`:

| Variable | Default | Description |
|---|---|---|
| `DOCS_ROOT` | *(set this)* | Root folder with labeled PDF subfolders |
| `MODEL_KEY` | `"xlm-roberta"` | Key into `MODEL_REGISTRY` |
| `OUTPUT_DIR` | `./models/bert_tunning_model` | Fine-tuned model output — `default_model_path` returns `OUTPUT_DIR/final` |
| `EPOCHS` | `15` | Max training epochs |
| `EARLY_STOP_PATIENCE` | `5` | Patience (eval epochs) for macro-F1 early stopping |
| `CHUNK_STRATEGY` | `"first"` | `"first"` or `"middle"` (first 256 + last 256 tokens) |
| `SEED` | `42` | Random seed |
| `CACHE_PATH` | `./data/bert_tunning_cache.parquet` | Parquet dataset cache |
| `WANDB_ENTITY` | `leonardo-a-heis` | W&B account |
| `WANDB_PROJECT` | `bert_tunning` | W&B project name |
| `API_PORT` | `8000` | Uvicorn port |
| `HOST` | `127.0.0.1` | Uvicorn host |
| `THRESHOLD` | `0.70` | Confidence threshold for `certain` flag |

Model hyperparameters (`lr`, `batch_size`, `grad_accum`, `max_tokens`, `force_fp32`) live in `ModelConfig`, not in settings.

## Logging

Each run creates a timestamped log file: `logs/bert_tunning_{YYYYMMDD_HHMMSS}.log`. `poe clean` deletes all `bert_tunning_*.log` files. The log path is returned by `setup_logging()` and emitted as the first log line of every CLI command.

## Document Labels

PDF subfolders map to labels via `FOLDER_TO_LABEL` in `src/settings.py`:

```
decretos/                         → decreto
decreto_concejo_municipal/        → decreto_concejo_municipal
ordenanzas/                       → ordenanza
decreto_ordenanzas/               → decreto_ordenanza
resoluciones/                     → resolucion
resoluciones_concejo_municipal/   → resolucion_concejo_municipal
declaraciones_concejo_municipal/  → declaracion_concejo_municipal
convenios/                        → convenio  (excluded by default)
```

## Key Technical Decisions

**XLM-RoBERTa over DeBERTa**
DeBERTa-v3 produces NaN gradients even in fp32. XLM-RoBERTa is numerically stable and handles Spanish well with `xlm-roberta-base`.

**`inputs.pop("labels")` in WeightedTrainer**
Must use `pop`, not `get`. Leaving labels in the inputs dict causes the model to compute its own CE loss on top of ours → doubled gradient graph → NaN loss.

**`num_items_in_batch` normalization**
Transformers ≥ 4.46 expects custom `compute_loss` to scale `loss * local_batch / num_items_in_batch` during gradient accumulation. Without it, effective loss is ~16× too large → NaN gradients.

**Pydantic v2 for all schemas**
All config and schema objects use `BaseModel` with `frozen=True`. API response schemas add `alias_generator=to_camel` for camelCase JSON serialization. Use `model_copy(update={...})` to derive modified instances from frozen models.

**`asyncio.to_thread()` in FastAPI predict endpoint**
`extract_pdf` and `clf.predict_text` are synchronous blocking functions (file I/O + PyTorch inference). Calling them directly in an `async` endpoint freezes the event loop. `asyncio.to_thread()` offloads them to the thread pool — the event loop stays free for concurrent requests.

**`run_api()` uses `Settings.default_model_path` directly**
`create_app(model_path, threshold)` requires `model_path` as an explicit argument. The uvicorn `factory=True` string approach can't pass arguments, so `run_api()` calls `create_app` directly and passes the resulting app object to `uvicorn.run()`.

**OCR thread safety via double-checked locking**
`OCRExtractor._reader` is lazily initialized on first use. Training (batch processing) and API (concurrent requests) both call it. A `threading.Lock()` with double-checked locking ensures EasyOCR is initialized exactly once across threads without holding the lock on every call.

**`[tool.uv] package = false`**
This is a script project, not an installable package. Without this, uv tries to build it with hatchling and fails.

**`pythonpath = ["."]` in pytest**
Enables `from src.ingestion.extract import ...` imports in tests without installing the package.

**MAX_TOKENS hard limit**
XLM-RoBERTa and BETO have a 512-token architectural maximum (positional embeddings). The median document in the corpus is ~654 tokens. `CHUNK_STRATEGY="middle"` (first 256 + last 256) captures more signal than `"first"` for longer documents (ordenanzas, resoluciones).

## Git Workflow

```bash
# Start a task branch from the integration branch
git worktree add -b task/N-name ../bert_tunning-taskN feature/scaffold-migration
cd ../bert_tunning-taskN
# implement, test, commit...
uv run poe check
git push -u origin task/N-name
gh pr create --base feature/scaffold-migration --title "Task N: ..."

# Clean up worktree after PR is open
cd "c:/Users/leona/source/repos/bert_tunning"
git worktree remove ../bert_tunning-taskN
```

## Git-Ignored Directories

```
data/       parquet cache files
models/     base model weights + fine-tuned checkpoints
reports/    HTML evaluation reports
logs/       training logs
samples/    sample PDFs for quick tests
.claude/    Claude session state
```

## Known Lint Suppressions

| Suppression | Location | Reason |
|---|---|---|
| `# noqa: FBT001, FBT002` | `src/training/trainer.py:25` | Boolean arg in WeightedTrainer — matches HuggingFace Trainer signature |
| `# noqa: PLC0415` | `src/training/models/__init__.py:_build_registry` | Deferred import avoids circular import with model submodules |
