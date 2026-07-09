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

### Prediction flow (`predict` / `predict-folder` / `POST /predict`)

Every entry point funnels into the same two calls:

```
predict_pdf(model_path, pdf_path)
   │
   ├─ extract_pdf_with_metadata(pdf_path)
   │     tries MarkItDownExtractor first; falls back to OCRExtractor only if the
   │     extracted text is shorter than MIN_TEXT_FOR_OCR. Returns ExtractionMetadata
   │     (text, extractor_used, char_count) — text is None if nothing usable came out
   │     of any extractor in the chain.
   │
   └─ BertTunningClassifier.predict_text(text)
         │
         ├─ tokenize → model forward pass → softmax → label + confidence + all_scores
         │
         └─ if the model directory has an ood_stats.npz (loaded once, at __init__):
               compute three independent signals on the [CLS] embedding —
               mahalanobis_p_value, cosine_z_score, knn_mean_distance — and
               OR them together into in_distribution (see below for the math)
         │
         ▼
   PredictResult (label, confidence, certain, all_scores, the three OOD fields,
                  extracted_text, extractor_used)
```

If `ood_stats.npz` is missing, the OOD fields simply stay `None` — no error, prediction still works, just without the anomaly signal.

### Training flow (`train`)

```
src/training/pipeline.py::run()
   1. LabelEncoder fits the class labels → label_id / label2id / id2label
   2. make_split(df, seed) — stratified 70/15/15 train/val/test
        (falls back to a plain random split, still seeded, if a class is too
        small to stratify — logs a warning rather than crashing)
   3. compute_class_weight("balanced", classes=np.arange(num_labels), ...)
        deliberately np.arange(num_labels), not np.unique(train_df) — guarantees
        every class gets a weight even if one is entirely absent from the train split
   4. Tokenize all three splits, load the base model fresh from the HF hub id
   5. Pick precision automatically: bf16 → fp16 → fp32, whichever the GPU supports
   6. WeightedTrainer.train() — the actual fine-tuning loop, with early stopping
        on macro-F1
   7. AFTER training completes, on the SAME model/tokenizer/device that was just
      fine-tuned:
          extract_embeddings(...) over the train split
             → compute_class_stats(embeddings, labels, class_names)
             → saved as ood_stats.npz right next to the model checkpoint
        (this is why compute-ood-stats, the backfill command for older models,
        has to reconstruct the exact same split via the same --seed — OOD stats
        are always computed from a specific model's own training embeddings)
   8. run_evaluation() on the held-out test split → macro F1 / accuracy
   9. Model + tokenizer + ood_stats.npz all saved to <output_dir>/final/
```

### OOD scoring internals

All three signals live in `src/ood.py` and share one PCA projection (`_project()`, `OOD_PCA_COMPONENTS` dimensions, fit once at training time):

| Signal | What it measures | Anomalous when | How to read the value |
|---|---|---|---|
| `mahalanobis_p_value` | Squared Mahalanobis distance to the nearest class centroid, converted to a p-value via `chi2.sf(df=n_components)`. Uses **one shared covariance matrix** across all classes (not per-class), regularized before inverting so it doesn't blow up on near-singular data. | **LOW** p-value (< `OOD_MAHALANOBIS_P_THRESHOLD`, default `0.01`) | A genuine p-value, `0.0`–`1.0`. `0.5` means "about as far from its class as a typical in-distribution document"; `0.0001` means the document is far enough that a real member of that class would essentially never land there. Because it assumes one Gaussian shape per class, it can misfire on heterogeneous classes (e.g. a broad `otro` catch-all) — the reason two other signals exist alongside it. |
| `cosine_z_score` | Cosine distance to the nearest centroid, z-scored against the training set's own distribution of that same metric. | **HIGH** z-score (> `OOD_COSINE_THRESHOLD`) | A standard z-score, so it **can be negative** — that's not an error, it just means this document's cosine distance to its centroid is *below* the training set's average distance (i.e. more typical than average, even more so than most training documents). Only large *positive* values are anomalous; `0` is average, negative is unremarkable. |
| `knn_mean_distance` | Mean Euclidean distance (PCA space) to the `k` nearest training documents — filtered to only the **predicted class**, not the whole training set. Makes no assumption that a class has one coherent shape, which is what makes it useful for a broad, heterogeneous class like `otro`. Returns `NaN` (logged as a warning) if the predicted class has zero training points; treated as anomalous, fail-safe, rather than silently passing. | **HIGH** distance (> `OOD_KNN_DISTANCE_THRESHOLD`) | A raw distance in PCA-space units — there's no natural "average is 0" reference point like the other two, so read it only relative to its calibrated threshold, not as a standalone percentage or probability. Thresholds are calibrated per model (`evaluate-ood-calibration`); a value calibrated for one trained model has no meaning against another. |

The three are deliberately **not combined into one score** — `in_distribution=False` fires if *any* one of them crosses its threshold. This means a document can be `certain=True` (softmax is confident) and `in_distribution=False` (the embedding doesn't resemble anything trained on) at the same time — the exact failure mode this feature exists to catch, and a human reviewing predictions can always see which specific signal fired.

**Is a signal actually calibrated?** Run `evaluate-ood-calibration` (see "Out-of-distribution detection" below) against the model's own held-out test split — known in-distribution by construction. It reports each signal's *empirical false-positive rate*: the fraction of genuinely in-distribution test documents the current threshold would incorrectly flag. A signal is calibrated when that rate is close to the target (`--target-fp-rate`, default 1%). If it's far off (Mahalanobis routinely runs 20–30% on this corpus, due to the shared-covariance assumption breaking down on heterogeneous classes like `otro`), don't blindly adopt the tool's `suggested_*_threshold` — check first whether the suggestion is itself degenerate (e.g. a suggested Mahalanobis p-value threshold of `0.0`, which would just disable the signal). Only update `settings.py` when the suggested value is a real, usable threshold.

### Review routing

`PredictResult.review_route` turns the confidence (`certain`) and OOD (`in_distribution`) signals into one of three actionable lanes, so a human doesn't have to eyeball every field on every prediction to decide what to do with it:

| `certain` | `in_distribution` | `review_route` | Rationale |
|---|---|---|---|
| — | `False` | `human_review` | An OOD signal fired — the document doesn't resemble anything the model trained on. An LLM judge can't be trusted to catch what already fooled the classifier, so this always routes to a human, regardless of how confident the softmax was. |
| `True` | `True` or `None` (no `ood_stats.npz`) | `accept` | Confident, and no evidence the document is out of distribution. Auto-accept. |
| `False` | `True` or `None` | `llm_judge` | The document looks like a known type, but the model itself is unsure which one — a cheap LLM second opinion is proportionate to the ambiguity. |

`decide_review_route()` (`src/inference/classify.py`) implements this and is unit-tested directly. It's attached to every `PredictResult`, so it shows up in the `predict-folder` CSV, the single-`predict` CLI output, the `/predict` API response, and the W&B predictions table with no extra flags needed. An unreadable/unextractable document (empty text) always gets `human_review` — there's no prediction to be confident about.

**This routing rule is intentionally coarse** (3 lanes from a boolean AND a boolean) — it doesn't distinguish "barely crossed one OOD threshold" from "every signal fired hard." If you need finer-grained triage within `human_review`, count how many of the three raw signals fired (0–3) as an ordinal severity score for queue ordering; that's not currently computed anywhere, so you'd read `mahalanobis_p_value`/`cosine_z`/`knn_distance` directly against their thresholds from the CSV/API output.

## Project structure

```
src/
├── __init__.py        package entry — exports BertTunningError, Settings, __version__
├── __main__.py        python -m src entry — spawns run_api() via multiprocessing.Process
├── settings.py        all configuration (Pydantic BaseSettings, overridable via .env)
├── schema.py          shared Pydantic schemas (PredictResult, ExtractionMetadata, ClassEmbeddingStats, CalibrationReport, Hyperparams, ReportDict)
├── wandb.py            all W&B interaction — WandbLogger (training) + log_predict_folder_results/log_ood_calibration_results (--log-wandb)
├── ood.py              OOD math — compute_class_stats, mahalanobis_p_value, cosine_z_score, knn_mean_distance, extract_embeddings, save_stats/load_stats.
│                       Lives at top level, not under training/ or inference/, since it's used by both (training-time stats computation, inference-time scoring)
├── exceptions.py      BertTunningError base
├── logger.py          setup_logging() — per-run timestamped log file
├── ingestion/         extract.py (extract_pdf_with_metadata) · scan.py · cache.py · pipeline.py · extractors/
├── training/
│   ├── models/        __init__.py (ModelConfig + registry) · xlm_roberta.py · beto.py · minilm.py
│   └──                options.py · split.py · tokenize.py · trainer.py · evaluate.py · pipeline.py · reporting.py
├── inference/         classify.py (BertTunningClassifier — mahalanobis/cosine/k-NN scoring) · pipeline.py (predict_pdf, predict_folder → list[PredictResult])
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
OUTPUT_DIR=./models/bert_tunning_model_beto_v2
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
| `MODEL_KEY` | `xlm-roberta` | Default model registry key (used by `train`; `predict`/`predict-folder`/`serve` load whatever `--model-path` points at directly, model-agnostic) |
| `OUTPUT_DIR` | `./models/bert_tunning_model_beto_v2` | Where the fine-tuned model is saved (`/final` is the inference path) — currently points at the best-performing checkpoint (BETO v2) |
| `EPOCHS` | `15` | Max training epochs |
| `EARLY_STOP_PATIENCE` | `5` | Epochs without macro-F1 improvement before stopping |
| `CHUNK_STRATEGY` | `first` | `first` = first 512 tokens; `middle` = first 256 + last 256 |

See `CLAUDE.md`'s Settings table for the full list, including the OOD-related settings (`OOD_MAHALANOBIS_P_THRESHOLD`, `OOD_COSINE_THRESHOLD`, `OOD_KNN_NEIGHBORS`, `OOD_KNN_DISTANCE_THRESHOLD`) described in the "Out-of-distribution detection" section below.

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

# Folder of PDFs → saves results to <folder>/bert_tunning_predictions.csv by default
uv run python main.py predict-folder path/to/folder

# Explicit output path overrides the default
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
include four extra fields:

```json
{
  "label": "boletines",
  "confidence": 0.9429,
  "mahalanobisPValue": 0.0003,
  "cosineZ": 1.1,
  "knnDistance": 31.4,
  "inDistribution": false
}
```

`mahalanobisPValue`/`cosineZ`/`knnDistance` are reported separately rather
than combined into one score — note `mahalanobisPValue` points in the
**opposite direction** from the other two: a LOW `mahalanobisPValue` (below
`OOD_MAHALANOBIS_P_THRESHOLD`, default `0.01`) is anomalous, while a HIGH
`cosineZ` (above `OOD_COSINE_THRESHOLD`) or a HIGH `knnDistance` (above
`OOD_KNN_DISTANCE_THRESHOLD`) is anomalous. Any one of the three alone is
enough to set `inDistribution: false`. This means `inDistribution: false`
doesn't hide *which* signal fired: a human reviewing predictions can see
whether Mahalanobis, cosine, k-NN, or several flagged the document.
Treat `inDistribution: false` as "do not trust `label` for this document"
regardless of how high `confidence` is — this is the mechanism that catches
documents (e.g. payment receipts) that were never in any training class,
including `otro`.

`knnDistance` is the mean distance (in PCA space) to the `OOD_KNN_NEIGHBORS`
(default `10`) nearest training documents *of the class the model just
predicted* — unlike Mahalanobis/cosine, it makes no assumption that a class
has one coherent "shape," which matters for broad, heterogeneous classes
like `otro`.

Backfill `ood_stats.npz` for an already-trained model (no retraining):

```powershell
uv run python main.py compute-ood-stats --model-path ./models/bert_tunning_model/final --model xlm-roberta --cache-path ./data/bert_tunning_cache.parquet
```

Measure the empirical false-positive rate of all three OOD thresholds against
the model's own held-out test split, and get a suggested better-calibrated
threshold for each if the defaults don't match your target:

```powershell
uv run python main.py evaluate-ood-calibration --model-path ./models/bert_tunning_model/final --model xlm-roberta --cache-path ./data/bert_tunning_cache.parquet
```

### Logging predictions and calibration runs to W&B

Both `predict-folder` and `evaluate-ood-calibration` accept `--log-wandb` to
additionally log their results to Weights & Biases (project/entity from
`Settings.WANDB_PROJECT`/`WANDB_ENTITY`), on top of writing the usual
CSV/console output — nothing changes about the local output when the flag is
omitted (the default).

```powershell
uv run python main.py predict-folder path/to/folder --log-wandb
uv run python main.py evaluate-ood-calibration --model-path ./models/bert_tunning_model/final --model xlm-roberta --cache-path ./data/bert_tunning_cache.parquet --log-wandb
```

`predict-folder --log-wandb` logs a `predictions` table (one row per
document: filename, label, confidence, certain, the four OOD fields,
extractor used, error) to a run tagged `job_type=predict-folder`.
`evaluate-ood-calibration --log-wandb` logs the empirical false-positive
rates and suggested thresholds for all three signals to a run tagged
`job_type=ood-calibration`, so calibration history is trackable across
models/thresholds over time instead of only living in a console log.

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

`predict-folder`'s CSV output includes `extracted_text`/`extractor_used`
columns automatically (the CSV is built from `model_dump()`, which uses the
Python field names, not the camelCase aliases the API returns). Use this to
check *what was actually extracted* from a misclassified document — e.g.
confirming whether a wrong classification is an extraction-quality problem
(garbled OCR output) versus a genuine
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
  "knnDistance": 9.7,
  "inDistribution": true,
  "extractedText": "DECRETO N° 123/2026...",
  "extractorUsed": "MarkItDownExtractor"
}
```

`mahalanobisPValue`/`cosineZ`/`knnDistance`/`inDistribution` are only present (non-null) when the loaded model directory has an `ood_stats.npz` — see "Out-of-distribution detection" above.

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
