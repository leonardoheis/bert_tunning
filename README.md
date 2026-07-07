# Bert Tunning

Fine-tunes transformer models on Spanish municipal PDF documents to classify them by document type (decreto, ordenanza, resolución, etc.).

The default model is [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) — stable multilingual model with strong Spanish support. Additional models (BETO, MiniLM) are available via the model registry in `src/training/models/`.

## Training results

All runs used effective batch size 64, bf16 precision, early stopping on macro-F1, and a stratified 70/15/15 train/val/test split.

| Run | Model | HuggingFace | Train docs | Classes | Macro F1 | Accuracy |
|---|---|---|---|---|---|---|
| xlm-roberta v1 | XLM-RoBERTa base | [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) | 515 | 8 | 0.782 | 0.811 |
| xlm-roberta v2 | XLM-RoBERTa base | [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) | 1,311 | 8 | 0.862 | 0.872 |
| beto v1 | BETO (Spanish BERT) | [bert-base-spanish-wwm-cased](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased) | 1,311 | 8 | **0.976** | **0.972** |
| beto v2 | BETO (Spanish BERT) | [bert-base-spanish-wwm-cased](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased) | 1,344 | 9 | 0.961 | 0.962 |
| minilm v1 | Multilingual MiniLM | [Multilingual-MiniLM-L12-H384](https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384) | 1,311 | 8 | 0.867 | 0.883 |

**BETO v1** is the best model — Spanish-native pretraining outperforms multilingual models on this corpus.

**XLM-RoBERTa v1 → v2** shows that more training data (+796 docs) improves macro F1 by +8 points regardless of architecture.

**BETO v1 → v2**: adding the `otro` class (46 documents, highly variable content) reduced macro F1 by 1.5 points — expected given the class imbalance and broad definition.

### Classes used

**8-class runs** (v1 / xlm-roberta v2 / minilm v1):
`boletines`, `declaracion_concejo_municipal`, `decreto`, `decreto_ordenanza`, `decretos_concejo_municipal`, `ordenanza`, `resolucion`, `resolucion_concejo_municipal`

**9-class run** (beto v2): same as above + `otro`

### Dataset summary

| Cache file | Docs | Max per class | Classes |
|---|---|---|---|
| `bert_tunning_cache_100.parquet` | 737 | 100 | 8 |
| `bert_tunning_cache_300.parquet` | 1,874 | 300 | 8 |
| `bert_tunning_cache_con_otro_300.parquet` | 1,920 | 300 | 9 |

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
├── __init__.py        package entry — exports BertTunningError, Settings, __version__
├── __main__.py        python -m src entry — spawns run_api() via multiprocessing.Process
├── settings.py        all configuration (Pydantic BaseSettings, overridable via .env)
├── schema.py          shared Pydantic schemas (PredictResult, ExtractionMetadata, ClassEmbeddingStats, Hyperparams, ReportDict)
├── exceptions.py      BertTunningError base
├── logger.py          setup_logging() — per-run timestamped log file
├── ingestion/         extract.py (extract_pdf_with_metadata) · scan.py · cache.py · pipeline.py · extractors/
├── training/
│   ├── models/        __init__.py (ModelConfig + registry) · xlm_roberta.py · beto.py · minilm.py
│   └──                options.py · split.py · tokenize.py · trainer.py · evaluate.py · pipeline.py
│                      reporting.py · wandb_logger.py
├── inference/         classify.py (BertTunningClassifier) · ood.py (Mahalanobis/cosine scoring) · pipeline.py (predict_pdf, predict_folder → list[PredictResult])
├── api/               app.py · schema.py · __init__.py · routes/predict/ · routes/health/
└── cli/               train.py · predict.py · ood_stats.py (compute-ood-stats) · ood_calibration.py (evaluate-ood-calibration) · clean.py

Dockerfile             multi-stage: uv builder + python:3.10-slim-bookworm runtime
main.py                Click CLI entry point
```

## Requirements

- Python ≥ 3.10
- CUDA-capable GPU (tested on NVIDIA RTX A4000 Laptop, 8 GB VRAM, CUDA 11.8)
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
API_PORT=8000
HOST=127.0.0.1
THRESHOLD=0.70
```

| Variable | Default | Description |
|---|---|---|
| `DOCS_ROOT` | *(set this)* | Root folder containing labeled subfolders of PDFs |
| `MODEL_KEY` | `xlm-roberta` | Default model registry key |
| `OUTPUT_DIR` | `./models/bert_tunning_model` | Where the fine-tuned model is saved (`/final` is the inference path) |
| `EPOCHS` | `15` | Max training epochs |
| `EARLY_STOP_PATIENCE` | `5` | Epochs without macro-F1 improvement before stopping |
| `CHUNK_STRATEGY` | `first` | `first` = first 512 tokens; `middle` = first 256 + last 256 |

Model hyperparameters (lr, batch size, etc.) live in the model registry — see `src/training/models/xlm_roberta.py`.

### Available models

| Key | Model | Notes |
|---|---|---|
| `xlm-roberta` | `xlm-roberta-base` | Default — stable, strong Spanish support |
| `beto` | `dccuchile/bert-base-spanish-wwm-cased` | Spanish-only BERT |
| `minilm` | `microsoft/Multilingual-MiniLM-L12-H384` | Lightweight — faster inference, lower accuracy |

### Adding a new model

**1. Create `src/training/models/my_model.py`:**

```python
from src.training.models.config import ModelConfig

config = ModelConfig(
    name="my-model-name",
    hf_id="org/model-id",
    max_tokens=512,
    lr=2e-5,
    batch_size=8,
    grad_accum=8,
    force_fp32=False,  # set True if the model produces NaN in fp16/bf16
)
```

**2. Register it in `src/training/models/__init__.py`** inside `_build_registry()`:

```python
def _build_registry() -> dict[str, ModelConfig]:
    from src.training.models import beto, minilm, my_model, xlm_roberta  # noqa: PLC0415

    return {
        "xlm-roberta": xlm_roberta.config,
        "beto": beto.config,
        "minilm": minilm.config,
        "my-model": my_model.config,
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

### Train

```powershell
# Default model (xlm-roberta)
uv run python main.py train --docs-root "C:\path\to\downloads"

# Use BETO instead of XLM-RoBERTa
uv run python main.py train --docs-root "C:\path\to\downloads" --model beto

# Quick test run — cap at 100 docs per class (minimum 10)
uv run python main.py train --docs-root "C:\path\to\downloads" --max-docs-per-class 100

# Custom epoch count
uv run python main.py train --docs-root "C:\path\to\downloads" --epochs 20

# Force re-extraction (ignore cached parquet)
uv run python main.py train --docs-root "C:\path\to\downloads" --rebuild-cache

# Disable W&B logging
uv run python main.py train --docs-root "C:\path\to\downloads" --no-wandb
```

### Classify

```powershell
# Single PDF
uv run python main.py predict path/to/documento.pdf

# Folder of PDFs → saves results to bert_tunning_predictions.csv
uv run python main.py predict-folder path/to/folder --output results.csv
```

Documents that yield no usable text (blank, corrupted, or below `MIN_USABLE_TEXT` characters) are reported as `label: null, error: "empty/unreadable document"` instead of a spurious classification — this applies to both commands and the API.

Output:
```
──────────────────────────────────────────────────
  File      : documento.pdf
  Label     : decreto
  Confidence: 97.32%
  Certain   : True
  Extractor : MarkItDownExtractor
  Extracted text (first 200 chars): 'DECRETO N° 123/2026...'

  All scores:
    decreto                                0.9732  ████████████████████████████████████████
    ordenanza                              0.0121  ▌
```

### Out-of-distribution detection

If the loaded model directory contains `ood_stats.npz` (generated automatically
during `train`, or backfilled for an existing model — see below), predictions
include three extra fields:

```json
{
  "label": "boletines",
  "confidence": 0.9429,
  "mahalanobisPValue": 0.0003,
  "cosineZ": 1.1,
  "inDistribution": false
}
```

`mahalanobisPValue`/`cosineZ` are reported separately rather than combined
into one score — note they point in **opposite directions**: a LOW
`mahalanobisPValue` (below `OOD_MAHALANOBIS_P_THRESHOLD`, default `0.01`) is
anomalous, while a HIGH `cosineZ` (above `OOD_COSINE_THRESHOLD`) is anomalous.
Either one alone is enough to set `inDistribution: false`. This means
`inDistribution: false` doesn't hide *which* signal fired: a human reviewing
predictions can see whether Mahalanobis, cosine, or both flagged the document.
Treat `inDistribution: false` as "do not trust `label` for this document"
regardless of how high `confidence` is — this is the mechanism that catches
documents (e.g. payment receipts) that were never in any training class,
including `otro`.

Backfill `ood_stats.npz` for an already-trained model (no retraining):

```powershell
uv run python main.py compute-ood-stats --model-path ./models/bert_tunning_model/final --model xlm-roberta --cache-path ./data/bert_tunning_cache.parquet
```

Measure the empirical false-positive rate of the OOD thresholds against the
model's own held-out test split, and get a suggested better-calibrated
threshold if the defaults don't match your target:

```powershell
uv run python main.py evaluate-ood-calibration --model-path ./models/bert_tunning_model/final --model xlm-roberta --cache-path ./data/bert_tunning_cache.parquet
```

### Extraction metadata (what the model actually saw)

Every prediction also records which extractor produced the text and the text
itself:

```json
{
  "label": "boletines",
  "extractorUsed": "OCRExtractor",
  "extractedText": "..."
}
```

`predict-folder`'s CSV output includes `extractedText`/`extractorUsed`
columns automatically. Use this to check *what was actually extracted* from
a misclassified document — e.g. confirming whether a wrong classification is
an extraction-quality problem (garbled OCR output) versus a genuine
out-of-category document with clean, correctly-extracted text.

### Serve inference API

**Local (CLI):**

```powershell
uv run python main.py serve --model-path ./models/bert_tunning_model/final
```

**Local (module):**

```powershell
python -m src
```

**Docker:**

```powershell
docker build -t bert-tunning .
docker run -p 8000:8000 -v ./models/bert_tunning_model:/app/models/bert_tunning_model bert-tunning
```

Available endpoints:
- `POST /predict` — upload a PDF, returns JSON with label, confidence, all scores, OOD signals, and extraction metadata
- `GET /health` — returns `{"status": "healthy", ...}`

API docs at `http://localhost:8000/docs` (Swagger UI).

**Example response:**

```json
{
  "filename": "decreto_123.pdf",
  "label": "decreto",
  "confidence": 0.9431,
  "certain": true,
  "allScores": {
    "decreto": 0.9431,
    "ordenanza": 0.0312,
    "resolucion": 0.0257
  },
  "error": null,
  "mahalanobisPValue": 0.42,
  "cosineZ": 0.8,
  "inDistribution": true,
  "extractedText": "DECRETO N° 123/2026...",
  "extractorUsed": "MarkItDownExtractor"
}
```

`mahalanobisPValue`/`cosineZ`/`inDistribution` are only present (non-null) when the loaded model directory has an `ood_stats.npz` — see "Out-of-distribution detection" above.

### Clean state

```powershell
uv run python main.py clean
```

Deletes logs, dataset cache, and model checkpoints. Base model weights in `models/hub/` are preserved.

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

Each run writes to a dedicated timestamped log file:

```
logs/bert_tunning_20260627_143201.log
logs/bert_tunning_20260627_150432.log
```

`poe clean` deletes all log files. The log path is printed as the first line of every CLI command.
