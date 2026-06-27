# CLAUDE.md — Classiflow

## Project Purpose

Classiflow fine-tunes transformer models on Spanish municipal PDF documents to classify them by document type (decreto, ordenanza, resolución, etc.). Built for Argentine municipalities.

## Current State (June 2026)

The project is in active scaffold migration from a flat-file layout to a `src/` pipeline with Click CLI, FastAPI endpoint, and a model registry. Full plan: `docs/superpowers/plans/2026-06-27-classiflow-scaffold-migration.md`.

Branch strategy:
- `master` — stable, production-ready
- `feature/scaffold-migration` — integration branch (task PRs merge here first)
- `task/N-*` — per-task branches, each opens a PR to `feature/scaffold-migration`
- Final PR: `feature/scaffold-migration` → `master` after all 9 tasks pass review

## Tech Stack

| Tool | Purpose |
|---|---|
| Python ≥ 3.10 | Runtime — use `X \| Y` union types, not `Optional[X]` |
| PyTorch (CUDA 11.8) | Training backend |
| HuggingFace Transformers ≥ 4.46 | Model loading, Trainer API |
| XLM-RoBERTa base | Default model — stable multilingual, strong Spanish |
| Pydantic v2 | ModelConfig schemas (frozen, validated) |
| Click | CLI |
| FastAPI + Uvicorn | Inference API endpoint |
| wandb | Experiment tracking |
| MarkItDown + EasyOCR | PDF text extraction with OCR fallback |
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

## CLI — Current (flat-file, replaced in Task 8)

```powershell
uv run python main.py --mode train --docs_root "C:\path\to\downloads"
uv run python main.py --mode train --max_docs_per_class 100
uv run python main.py --mode predict --pdf path/to/doc.pdf
uv run python main.py --mode predict_folder --folder path/to/folder
uv run python main.py --mode clean
uv run python main.py --help
```

## CLI — Target (after Task 8 migration)

```powershell
uv run python main.py train --docs-root "C:\path\to\downloads"
uv run python main.py train --model beto --max-docs-per-class 100
uv run python main.py predict path/to/doc.pdf
uv run python main.py predict-folder path/to/folder
uv run python main.py serve --model-path ./models/classiflow_model/final
uv run python main.py clean
```

## Target Architecture (`src/` pipeline)

```
src/
├── ingestion/
│   ├── extract.py      PDF → text (MarkItDown + EasyOCR fallback)
│   ├── scan.py         folder walk, label mapping
│   ├── cache.py        parquet load/save
│   └── pipeline.py     orchestrator: scan → extract → cache → DataFrame
├── training/
│   ├── models/
│   │   ├── __init__.py ModelConfig (Pydantic) + MODEL_REGISTRY + get_model_config()
│   │   ├── xlm_roberta.py
│   │   └── beto.py
│   ├── split.py        stratified train/val/test split
│   ├── tokenize.py     ClassiflowDataset, prepare_text
│   ├── trainer.py      WeightedTrainer, compute_metrics
│   ├── evaluate.py     classification_report, confusion_matrix, HTML report
│   └── pipeline.py     orchestrates split → tokenize → train → evaluate
├── inference/
│   ├── classify.py     ClassiflowClassifier, predict_text
│   └── pipeline.py     predict_pdf, predict_folder
├── api/
│   ├── app.py          FastAPI create_app()
│   └── routes/
│       └── predict.py  POST /predict — multipart PDF upload
└── cli/
    ├── train.py        @click.command "train"
    ├── predict.py      @click.command "predict" + "predict-folder"
    └── clean.py        @click.command "clean"

config.py               constants + MODEL_KEY (stays at root)
wandb_logger.py         WandbLogger class
reporting.py            HTML report generation
logger.py               logging setup
main.py                 Click group with "serve" subcommand
```

Legacy flat files deleted in Task 9: `extraction.py`, `dataset.py`, `training.py`, `classifier.py`

## Model Registry

Models are registered in `src/training/models/`. Each model file exports a `config: ModelConfig` instance.

```python
from src.training.models import MODEL_REGISTRY, get_model_config

cfg = get_model_config("xlm-roberta")  # raises KeyError with helpful msg if unknown
cfg = get_model_config("beto")
```

Available keys: `xlm-roberta`, `beto`

To add a model: create `src/training/models/my_model.py` with `config = ModelConfig(...)`, then register it in `_build_registry()` in `src/training/models/__init__.py`.

`MODEL_KEY` in `config.py` sets the default model for the CLI.

## Config (`config.py`)

| Variable | Default | Description |
|---|---|---|
| `DOCS_ROOT` | *(set this)* | Root folder with labeled PDF subfolders |
| `MODEL_KEY` | `"xlm-roberta"` | Key into `MODEL_REGISTRY` |
| `OUTPUT_DIR` | `./models/classiflow_model` | Fine-tuned model output path |
| `EPOCHS` | `15` | Max training epochs |
| `EARLY_STOP_PATIENCE` | `5` | Patience (eval epochs) for macro-F1 early stopping |
| `CHUNK_STRATEGY` | `"first"` | `"first"` or `"middle"` (first 256 + last 256 tokens) |
| `SEED` | `42` | Random seed |
| `CACHE_PATH` | `./data/classiflow_cache.parquet` | Parquet dataset cache |
| `WANDB_ENTITY` | `leonardo-a-heis` | W&B account |
| `WANDB_PROJECT` | `bert_tunning` | W&B project name |

Model hyperparameters (`lr`, `batch_size`, `grad_accum`, `max_tokens`, `force_fp32`) live in `ModelConfig`, not in `config.py`.

## Document Labels

PDF subfolders map to labels via `FOLDER_TO_LABEL` in `config.py`:

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

**Pydantic v2 for ModelConfig**
All model configs use `BaseModel` with `frozen=True` (via `ConfigDict`) for runtime validation and immutability. Prefer Pydantic over dataclasses for all config/schema objects in this project.

**`[tool.uv] package = false`**
This is a script project, not an installable package. Without this, uv tries to build it with hatchling and fails (no package directory exists).

**`pythonpath = ["."]` in pytest**
Enables `from src.ingestion.extract import ...` imports in tests without installing the package.

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
| `# noqa: FBT001, FBT002` | `training.py:train()` | Legacy boolean args, removed in Task 9 |
| `# noqa: PLW0603` | `extract.py:_get_ocr_reader` | Intentional global singleton for EasyOCR |
| `# noqa: BLE001` | `extract.py:_ocr_fallback` | Broad catch is intentional — OCR must not crash the pipeline |
| `# noqa: PLC0415` | `models/__init__.py:_build_registry` | Deferred import avoids circular import with model submodules |
