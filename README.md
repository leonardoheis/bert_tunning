# Classiflow

Fine-tunes a transformer model on municipal PDF documents to classify them by document type (decreto, ordenanza, resolución, etc.).

The default model is [xlm-roberta-base](https://huggingface.co/xlm-roberta-base), a stable multilingual model with strong Spanish support. Alternative Spanish models can be selected in `config.py`.

## How it works

```
PDF files
   └── extraction.py   →  MarkItDown extracts text; EasyOCR fallback for scanned pages
         └── dataset.py    →  builds a labeled DataFrame, caches full text to Parquet (data/)
               └── training.py   →  fine-tunes the model, saves best checkpoint by macro-F1
                     └── classifier.py →  loads saved model, runs inference on new PDFs
```

Base model weights download once to `models/hub/`. Fine-tuned weights are saved to `models/classiflow_model/final/`. All runs are logged to `logs/classiflow.log` and the console.

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
├── data/                    # Parquet cache files (git-ignored)
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
| `MODEL_NAME` | `xlm-roberta-base` | Base HuggingFace model (see options below) |
| `OUTPUT_DIR` | `./models/classiflow_model` | Where the fine-tuned model is saved |
| `BATCH_SIZE` | `8` | Per-device batch size |
| `GRAD_ACCUM` | `8` | Gradient accumulation steps (effective batch = 64) |
| `EPOCHS` | `15` | Max training epochs |
| `EARLY_STOP_PATIENCE` | `5` | Epochs without macro-F1 improvement before stopping |
| `LR` | `2e-5` | Learning rate |
| `CHUNK_STRATEGY` | `first` | `first` = first 512 tokens; `middle` = first 256 + last 256 |

### Model options

```python
# config.py
MODEL_NAME = "xlm-roberta-base"                       # recommended: stable, strong Spanish
# MODEL_NAME = "PlanTL-GOB-ES/roberta-base-bne"       # Spanish-only RoBERTa
# MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased" # BETO: original Spanish BERT
# MODEL_NAME = "microsoft/deberta-v3-base"             # high accuracy, numerically unstable
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
└── convenios/                        ← excluded by default (see EXCLUDE_LABELS in config.py)
```

Each subfolder name maps to a label via `FOLDER_TO_LABEL` in `config.py`.

## Usage

### Train

```powershell
uv run python main.py --mode train --docs_root "C:\path\to\downloads"
```

On the first run, text is extracted from all PDFs and cached to `data/classiflow_cache.parquet` (one-time cost — can take several hours for large corpora). Subsequent runs load from cache. The best checkpoint by macro-F1 is saved automatically and early stopping halts training if no improvement is seen for `EARLY_STOP_PATIENCE` epochs.

```powershell
# Force re-extraction (e.g. after adding new documents)
uv run python main.py --mode train --rebuild_cache

# Skip OCR fallback for faster extraction (digital PDFs only)
uv run python main.py --mode train --no_ocr
```

### Quick test run (sample mode)

Cap how many PDFs are read per class to verify the pipeline end-to-end without waiting for full extraction. A separate cache file is created for each cap value so the full dataset cache is never overwritten.

```powershell
uv run python main.py --mode train --docs_root "C:\path\to\downloads" --max_docs_per_class 100
```

### Clean state

Wipes logs, dataset cache, and model checkpoints so the next run starts completely fresh.

```powershell
# Reset everything, then train
uv run python main.py --mode train --docs_root "C:\path\to\downloads" --clean

# Quick clean + sample run to verify the pipeline
uv run python main.py --mode train --docs_root "C:\path\to\downloads" --clean --max_docs_per_class 100

# Just wipe state without running anything
uv run python main.py --mode clean
```

What gets deleted:

| Target | Path |
|---|---|
| Log file | `logs/classiflow.log` |
| Dataset cache | `data/classiflow_cache*.parquet` |
| Model checkpoints | `models/classiflow_model/` |

> The base model weights in `models/hub/` are **not** deleted — re-downloading hundreds of MB on every clean would be wasteful.

### Classify a single PDF

```powershell
uv run python main.py --mode predict --pdf "C:\path\to\documento.pdf"
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
uv run python main.py --mode predict_folder --folder "C:\path\to\folder"
```

Results are saved to `classiflow_predictions.csv`.

### All flags

```powershell
uv run python main.py --help
```

| Flag | Default | Description |
|---|---|---|
| `--mode` | `train` | `train`, `predict`, `predict_folder`, or `clean` |
| `--docs_root` | *(config)* | Root folder with labeled PDF subfolders (train mode) |
| `--model_path` | `OUTPUT_DIR/final` | Path to saved model (predict modes) |
| `--pdf` | — | Single PDF to classify |
| `--folder` | — | Folder of PDFs to classify |
| `--max_docs_per_class` | — | Cap docs per class for quick test runs |
| `--rebuild_cache` | off | Force re-extraction even if cache exists |
| `--no_ocr` | off | Skip OCR fallback (faster, digital PDFs only) |
| `--threshold` | `0.70` | Min confidence to mark a prediction as certain |
| `--clean` | off | Wipe logs, cache and model before running |
| `--no_wandb` | off | Disable Weights & Biases logging |
| `--debug` | off | Enable DEBUG level logging |

## Logging

Every run appends to `logs/classiflow.log`. The same output is shown in the console:

```
2026-06-25 14:32:01 [INFO    ] training — Training started — 15 epochs, effective batch 64
2026-06-25 14:32:01 [WARNING ] extraction — Skipping doc.pdf — could not extract usable text
```
