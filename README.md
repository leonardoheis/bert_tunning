# Bert Tunning

Fine-tunes transformer models on Spanish municipal PDF documents to classify them by document type (decreto, ordenanza, resolución, etc.).

The default model is [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) — stable multilingual model with strong Spanish support. Additional models (e.g. BETO) are available via the model registry in `src/training/models/`.

## How it works

```
PDF files
   └── src/ingestion/    →  extract text (MarkItDown + EasyOCR fallback), scan folders, cache to Parquet
         └── src/training/   →  fine-tune model, save best checkpoint by macro-F1, log to W&B
               └── src/inference/  →  load saved model, classify new PDFs
                     └── src/api/        →  expose POST /predict via FastAPI
```

## Project structure

```
src/
├── ingestion/     extract.py · scan.py · cache.py · pipeline.py
├── training/
│   ├── models/    __init__.py (ModelConfig + registry) · xlm_roberta.py · beto.py
│   └──            options.py · split.py · tokenize.py · trainer.py · evaluate.py · pipeline.py
├── inference/     classify.py · pipeline.py
├── api/           app.py · schema.py · routes/predict/ · routes/health/
└── cli/           train.py · predict.py · clean.py

src/settings.py    all configuration (overridable via .env)
wandb_logger.py    WandbLogger class
reporting.py       HTML evaluation report
logger.py          logging setup
main.py            Click CLI entry point
```

## Requirements

- Python ≥ 3.10
- CUDA-capable GPU (tested on RTX 8 GB VRAM)
- [uv](https://docs.astral.sh/uv/)

## Installation

```powershell
uv sync
```

> `torch` is pulled from the PyTorch CUDA 11.8 index automatically via `[tool.uv.sources]` in `pyproject.toml`.

## Configuration

All settings live in `src/settings.py` and can be overridden via a `.env` file at the project root:

```ini
# .env (optional — values shown are the defaults)
DOCS_ROOT=C:\path\to\downloads
MODEL_KEY=xlm-roberta
OUTPUT_DIR=./models/bert_tunning_model
EPOCHS=15
EARLY_STOP_PATIENCE=5
CHUNK_STRATEGY=first
SEED=42
WANDB_ENTITY=your-wandb-entity
WANDB_PROJECT=bert_tunning
```

| Variable | Default | Description |
|---|---|---|
| `DOCS_ROOT` | *(set this)* | Root folder containing labeled subfolders of PDFs |
| `MODEL_KEY` | `xlm-roberta` | Default model registry key |
| `OUTPUT_DIR` | `./models/bert_tunning_model` | Where the fine-tuned model is saved |
| `EPOCHS` | `15` | Max training epochs |
| `EARLY_STOP_PATIENCE` | `5` | Epochs without macro-F1 improvement before stopping |
| `CHUNK_STRATEGY` | `first` | `first` = first 512 tokens; `middle` = first 256 + last 256 |

Model hyperparameters (lr, batch size, etc.) live in the model registry — see `src/training/models/xlm_roberta.py`.

### Available models

| Key | Model | Notes |
|---|---|---|
| `xlm-roberta` | `xlm-roberta-base` | Default — stable, strong Spanish support |
| `beto` | `dccuchile/bert-base-spanish-wwm-cased` | Spanish-only BERT |

### Adding a new model

**1. Create `src/training/models/my_model.py`:**

```python
from src.training.models import ModelConfig

config = ModelConfig(
    name="my-model-name",          # display name
    hf_id="org/model-id",          # HuggingFace model ID
    max_tokens=512,
    lr=2e-5,
    batch_size=8,
    grad_accum=8,
    force_fp32=False,              # set True if the model produces NaN in fp16/bf16
)
```

**2. Register it in `src/training/models/__init__.py`** inside `_build_registry()`:

```python
def _build_registry() -> dict[str, ModelConfig]:
    from src.training.models import beto, my_model, xlm_roberta  # noqa: PLC0415

    return {
        "xlm-roberta": xlm_roberta.config,
        "beto": beto.config,
        "my-model": my_model.config,   # ← add this line
    }
```

**3. Train with it:**

```powershell
uv run python main.py train --docs-root "C:\path\to\downloads" --model my-model
```

## Expected folder structure

```
downloads/
├── decretos/
├── decreto_concejo_municipal/
├── ordenanzas/
├── decreto_ordenanzas/
├── resoluciones/
├── resoluciones_concejo_municipal/
├── declaraciones_concejo_municipal/
└── convenios/                        ← excluded by default
```

## Usage

Commands are available via `poe` (short form) or `python main.py` (full form).

### Train

```powershell
# Default model (xlm-roberta)
uv run poe train --docs-root "C:\path\to\downloads"

# Use BETO instead of XLM-RoBERTa
uv run poe train --docs-root "C:\path\to\downloads" --model beto

# Quick test run — cap at 100 docs per class
uv run poe train --docs-root "C:\path\to\downloads" --max-docs-per-class 100

# Force re-extraction (ignore cached parquet)
uv run poe train --docs-root "C:\path\to\downloads" --rebuild-cache

# Disable W&B logging
uv run poe train --docs-root "C:\path\to\downloads" --no-wandb
```

### Classify

```powershell
# Single PDF
uv run poe predict path/to/documento.pdf

# Folder of PDFs → saves results to bert_tunning_predictions.csv
uv run python main.py predict-folder path/to/folder
```

Output:
```
──────────────────────────────────────────────────
  File      : documento.pdf
  Label     : decreto
  Confidence: 97.32%
  Certain   : True

  All scores:
    decreto                                0.9732  ████████████████████████████████████████
    ordenanza                              0.0121  ▌
```

### Serve inference API

```powershell
uv run poe serve
```

Or with custom options:

```powershell
uv run python main.py serve --model-path ./models/bert_tunning_model/final --port 8080
```

- `POST /predict` — upload a PDF, returns JSON with label, confidence, and all scores
- `GET /health` — returns `{"status": "healthy", ...}`

API docs at `http://localhost:8000/docs` (Swagger UI).

### Clean state

```powershell
uv run python main.py clean
```

Deletes logs, dataset cache, and model checkpoints. Base model weights in `models/hub/` are preserved.

### All options

```powershell
uv run python main.py --help
uv run python main.py train --help
uv run python main.py predict --help
```

## Development

```powershell
uv run poe check      # lint + typecheck + test (run before every commit)
uv run poe fmt        # auto-format with ruff
uv run poe lint       # lint + format check only
uv run poe typecheck  # mypy strict
uv run poe test       # pytest
uv run poe coverage   # pytest with HTML coverage report
```

## Logging

Every run appends to `logs/bert_tunning.log`:

```
2026-06-27 14:32:01 [INFO] training — Training started — 15 epochs, effective batch 64
2026-06-27 14:32:01 [WARNING] extraction — Skipping doc.pdf — could not extract usable text
```
