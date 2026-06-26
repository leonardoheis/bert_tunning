# Classiflow

Fine-tunes [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) on municipal PDF documents to classify them by document type (decreto, ordenanza, resolución, convenio, etc.).

## How it works

```
PDF files
   └── extraction.py   →  MarkItDown extracts text; EasyOCR fallback for scanned pages
         └── dataset.py    →  builds a labeled DataFrame, caches to Parquet
               └── training.py   →  fine-tunes DeBERTa-v3-base, saves best checkpoint
                     └── classifier.py →  loads saved model, runs inference on new PDFs
```

Base model weights download once to `models/hub/`. Fine-tuned weights are saved to `models/classiflow_deberta_model/final/`. All runs are logged to `logs/classiflow.log` and the console.

## Project structure

```
.
├── main.py                  # CLI entry point
├── config.py                # all constants and hyperparameters
├── extraction.py            # PDF → text (MarkItDown + EasyOCR fallback)
├── dataset.py               # dataset builder, Parquet cache, PyTorch Dataset
├── training.py              # training loop, evaluation, model saving
├── classifier.py            # inference wrapper for trained model
├── logger.py                # logging setup (console + file)
├── models/                  # base model cache + fine-tuned checkpoints (git-ignored)
├── logs/                    # run logs (git-ignored)
└── pyproject.toml
```

## Requirements

- Python ≥ 3.10
- CUDA-capable GPU (tested on RTX 8 GB VRAM)
- [uv](https://docs.astral.sh/uv/)

## Installation

```powershell
uv sync
```

> `torch` and `torchvision` are pulled from the PyTorch CUDA 11.8 index automatically via `[tool.uv.sources]` in `pyproject.toml`.

## Configuration

Edit `config.py` before running:

| Variable | Default | Description |
|---|---|---|
| `DOCS_ROOT` | *(set this)* | Root folder containing labeled subfolders of PDFs |
| `MODEL_NAME` | `microsoft/deberta-v3-base` | Base HuggingFace model |
| `OUTPUT_DIR` | `./models/classiflow_deberta_model` | Where fine-tuned model is saved |
| `BATCH_SIZE` | `8` | Per-device batch size (8 fits in 8 GB VRAM with fp16) |
| `GRAD_ACCUM` | `8` | Gradient accumulation steps (effective batch = 64) |
| `EPOCHS` | `5` | Training epochs |
| `CHUNK_STRATEGY` | `first` | `first` = first 512 tokens; `middle` = first 256 + last 256 |

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
├── convenios/
├── boletines/                ← excluded
└── compendios_de_boletines/  ← excluded
```

Each subfolder name maps to a label via `FOLDER_TO_LABEL` in `config.py`.

## Usage

### Train

```powershell
uv run main.py --mode train --docs_root "C:\path\to\downloads"
```

On the first run, text is extracted from all PDFs and cached to `classiflow_cache.parquet` (one-time cost, several hours for large corpora). Subsequent runs load from cache. The best checkpoint by macro-F1 is saved automatically.

```powershell
# Force re-extraction (e.g. after adding new documents)
uv run main.py --mode train --rebuild_cache

# Skip OCR fallback for faster extraction (digital PDFs only)
uv run main.py --mode train --no_ocr
```

### Quick test run (sample mode)

Use `--max_docs_per_class` to cap how many PDFs are read per class. Useful for verifying the pipeline end-to-end without waiting hours for full extraction. Cache is bypassed automatically so the full dataset cache is never overwritten.

```powershell
uv run main.py --mode train --docs_root "C:\path\to\downloads" --max_docs_per_class 30
```

### Clean state

Wipes logs, dataset cache, and model checkpoints so the next run starts completely fresh.

```powershell
# Reset everything, then train
uv run main.py --mode train --docs_root "C:\path\to\downloads" --clean

# Quick clean + sample run to verify the pipeline
uv run main.py --mode train --docs_root "C:\path\to\downloads" --clean --max_docs_per_class 30

# Just wipe state without running anything
uv run main.py --mode clean
```

What gets deleted:

| Target | Path |
|---|---|
| Log file | `logs/classiflow.log` |
| Dataset cache | `classiflow_cache.parquet` |
| Model checkpoints | `models/classiflow_deberta_model/` |

> The base model weights in `models/hub/` are **not** deleted — re-downloading 371 MB on every clean would be wasteful.

### Classify a single PDF

```powershell
uv run main.py --mode predict --pdf "C:\path\to\documento.pdf"
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
    ...
```

### Classify a folder of PDFs

```powershell
uv run main.py --mode predict_folder --folder "C:\path\to\folder"
```

Results are saved to `classiflow_predictions.csv`.

### Common flags

| Flag | Description |
|---|---|
| `--threshold 0.85` | Confidence threshold to mark a prediction as certain (default `0.70`) |
| `--model_path ./models/...` | Path to a specific saved model (default: `OUTPUT_DIR/final`) |
| `--max_docs_per_class 30` | Cap docs per class for quick test runs — bypasses cache |
| `--no_ocr` | Skip OCR fallback (faster if all PDFs are digital) |
| `--rebuild_cache` | Force re-extraction even if cache exists |
| `--debug` | Enable DEBUG level logging |

## Logging

Every run appends to `logs/classiflow.log`. The same output is shown in the console:

```
2026-06-25 14:32:01 [INFO    ] training — Training started — 5 epochs, effective batch 64
2026-06-25 14:32:01 [WARNING ] extraction — Skipping doc.pdf — could not extract usable text
```
