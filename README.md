# Classiflow

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
│   └──            split.py · tokenize.py · trainer.py · evaluate.py · pipeline.py
├── inference/     classify.py · pipeline.py
├── api/           app.py · routes/predict.py
└── cli/           train.py · predict.py · clean.py

config.py          constants and MODEL_KEY
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

Edit `config.py` before running:

| Variable | Default | Description |
|---|---|---|
| `DOCS_ROOT` | *(set this)* | Root folder containing labeled subfolders of PDFs |
| `MODEL_KEY` | `xlm-roberta` | Model registry key — see `src/training/models/` |
| `OUTPUT_DIR` | `./models/classiflow_model` | Where the fine-tuned model is saved |
| `EPOCHS` | `15` | Max training epochs |
| `EARLY_STOP_PATIENCE` | `5` | Epochs without macro-F1 improvement before stopping |
| `CHUNK_STRATEGY` | `first` | `first` = first 512 tokens; `middle` = first 256 + last 256 |

Model hyperparameters (lr, batch size, etc.) live in the model registry — see `src/training/models/xlm_roberta.py`.

### Available models

| Key | Model | Notes |
|---|---|---|
| `xlm-roberta` | `xlm-roberta-base` | Default — stable, strong Spanish support |
| `beto` | `dccuchile/bert-base-spanish-wwm-cased` | Spanish-only BERT |

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

### Train

```powershell
uv run python main.py train --docs-root "C:\path\to\downloads"

# Use BETO instead of XLM-RoBERTa
uv run python main.py train --docs-root "C:\path\to\downloads" --model beto

# Quick test run (100 docs per class)
uv run python main.py train --docs-root "C:\path\to\downloads" --max-docs-per-class 100

# Force re-extraction
uv run python main.py train --docs-root "C:\path\to\downloads" --rebuild-cache

# Disable W&B logging
uv run python main.py train --docs-root "C:\path\to\downloads" --no-wandb
```

### Classify

```powershell
# Single PDF
uv run python main.py predict path/to/documento.pdf

# Folder of PDFs (saves results to classiflow_predictions.csv)
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
uv run python main.py serve --model-path ./models/classiflow_model/final
```

- `POST /predict` — upload a PDF, returns JSON with label, confidence, and all scores
- `GET /health` — returns `{"status": "ok"}`

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
uv run python main.py serve --help
```

## Development

```powershell
uv run poe check      # lint + typecheck + test
uv run poe fmt        # auto-format
uv run poe coverage   # test coverage report
```

## Logging

Every run appends to `logs/classiflow.log`:

```
2026-06-27 14:32:01 [INFO] training — Training started — 15 epochs, effective batch 64
2026-06-27 14:32:01 [WARNING] extraction — Skipping doc.pdf — could not extract usable text
```
