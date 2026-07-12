# CLAUDE.md — Bert Tunning

## Project Purpose

Bert Tunning fine-tunes transformer models on Spanish municipal PDF documents to classify them by document type (decreto, ordenanza, resolución, etc.). Built for Argentine municipalities.

## Current State (July 2026)

Scaffold migration from flat-file layout to `src/` pipeline is complete and merged to `master`.

The out-of-distribution (Mahalanobis/cosine) detection feature (`docs/superpowers/plans/2026-07-04-ood-mahalanobis-detection.md`) is complete — all 9 tasks (1, 2, 3, 3b, 4, 5, 6, 7, 8) are merged into `feature/ood-detection`. A third OOD signal, class-conditional k-NN mean distance (`docs/superpowers/plans/2026-07-07-knn-ood-detection.md`), is also complete — all 6 tasks are merged. `feature/ood-detection` is ready to merge to `master`. See "Out-of-distribution (OOD) detection", "k-NN class-conditional distance", and "Extraction metadata" under Key Technical Decisions below for what shipped.

Branch strategy:
- `master` — stable, production-ready
- `feature/ood-detection` — integration branch, merged from `feature/scaffold-migration`, ready to merge to master
- `task/N-*` — per-task branches (merged into whichever integration branch was active when the task started)

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->

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
uv run python main.py train --docs-root "C:\path\to\downloads" --epochs 20
uv run python main.py predict path/to/doc.pdf
uv run python main.py predict-folder path/to/folder
uv run python main.py serve --model-path ./models/bert_tunning_model/final
uv run python main.py compute-ood-stats --model-path ./models/bert_tunning_model/final --model xlm-roberta --cache-path ./data/bert_tunning_cache.parquet
uv run python main.py evaluate-ood-calibration --model-path ./models/bert_tunning_model/final --model xlm-roberta --cache-path ./data/bert_tunning_cache.parquet
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
├── schema.py           shared Pydantic schemas (PredictResult, ExtractionMetadata, ClassEmbeddingStats, CalibrationReport, Hyperparams, ReportDict)
├── wandb.py            all W&B interaction — WandbLogger (training), log_predict_folder_results/log_ood_calibration_results (--log-wandb on the CLI)
├── ood.py              OOD math — compute_class_stats, mahalanobis_p_value, cosine_z_score, knn_mean_distance, extract_embeddings, save_stats/load_stats. Lives at top level (not under inference/) because compute_class_stats/extract_embeddings/save_stats are training-time producers consumed by src/training/pipeline.py, while mahalanobis_p_value/cosine_z_score/knn_mean_distance/load_stats are inference-time consumers used by src/inference/classify.py — same shared-across-layers treatment as wandb.py
├── exceptions.py       BertTunningError base
├── logger.py           setup_logging() → per-run timestamped log file
├── ingestion/
│   ├── extract.py      extract_pdf_with_metadata() → ExtractionMetadata (text, extractor_used, char_count); extract_pdf() is a thin wrapper — MarkItDown + EasyOCR fallback chain
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
│   ├── evaluate.py     run_evaluation() → EvaluationResult (defined in schema.py; macro_f1/accuracy are @property)
│   ├── reporting.py    generate_html_report() → reports/
│   └── pipeline.py     orchestrates split → tokenize → train → evaluate (uses src/wandb.py's WandbLogger)
├── inference/
│   ├── classify.py     BertTunningClassifier, predict_text — computes mahalanobis_p_value/cosine_z/knn_distance/in_distribution when ood_stats.npz is present
│   └── pipeline.py     predict_pdf() → PredictResult; predict_folder() → list[PredictResult] — both attach extracted_text/extractor_used
├── api/
│   ├── app.py          create_app(model_path, threshold) → FastAPI
│   ├── schema.py       BaseSchema (camelCase aliases, `populate_by_name=True`)
│   ├── __init__.py     run_api() — reads Settings.default_model_path
│   └── routes/
│       ├── health/     GET / and GET /health
│       └── predict/    POST /predict — multipart PDF upload; response includes OOD + extraction-metadata fields
└── cli/
    ├── train.py        @click.command "train" — TrainOptions (Pydantic), supports --epochs
    ├── predict.py      @click.command "predict" + "predict-folder" — prints/exports OOD + extraction fields; predict-folder supports --log-wandb and defaults --output to bert_tunning_predictions.csv inside folder_path
    ├── ood_stats.py     @click.command "compute-ood-stats" — backfills ood_stats.npz for an already-trained model, no retraining
    ├── ood_calibration.py @click.command "evaluate-ood-calibration" — measures empirical FP rate of OOD thresholds against the model's own test split; supports --log-wandb
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
| `PREDICT_THRESHOLD` | `0.70` | Confidence threshold used by `predict`/`predict-folder` CLI commands |
| `PREDICT_CONFIDENCE` | `0.0` | Confidence value reported for unreadable/empty documents |
| `MAX_DOCS_PER_CLASS` | `10` | Minimum allowed value for `--max-docs-per-class` — also the ingestion default when unset |
| `OOD_PCA_COMPONENTS` | `64` | Dimensionality the `[CLS]` embedding is reduced to before Mahalanobis/cosine scoring |
| `OOD_MAHALANOBIS_P_THRESHOLD` | `0.001` | Empirical (rank-based) Mahalanobis p-value below which a document is flagged `in_distribution=False` (low p-value = anomalous) — see the empirical p-value decision below for why this isn't a chi-squared p-value, and for why it isn't the naive 1%-target suggestion either |
| `OOD_COSINE_THRESHOLD` | `2.5` | Cosine z-score above which a document is flagged `in_distribution=False` — uncalibrated, see the OOD detection note below |
| `OOD_KNN_NEIGHBORS` | `10` | Number of same-predicted-class training documents used for the k-NN distance signal |
| `OOD_KNN_DISTANCE_THRESHOLD` | `26.125` | Mean k-NN distance (PCA space) above which a document is flagged `in_distribution=False` — calibrated against BETO v2, see the k-NN detection note below |

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

**Mahalanobis chi-squared p-value + cosine z-score OOD detection, multi-signal OR, alongside the softmax classifier**
Softmax classifiers always output a label — there is no "I don't know." `ood_stats.npz` (generated at training time from the training set's `[CLS]` embeddings, PCA-reduced) stores per-class centroids and a shared covariance matrix. At inference, `BertTunningClassifier.predict_text` computes **independently-scored** signals: (1) the squared Mahalanobis distance to the nearest class centroid, evaluated as a p-value under a chi-squared distribution with `df` = the PCA-reduced dimensionality — the theoretically correct distribution for this metric, assuming class-conditional embeddings are multivariate Gaussian (the same assumption Mahalanobis distance itself relies on) — needing no empirical calibration; and (2) cosine distance to the nearest centroid, z-scored against the training set's own distances (no equivalent theoretical distribution exists for cosine, so this stays empirical). Both are attached to `PredictResult` as `mahalanobis_p_value`/`cosine_z`/`in_distribution` (see the k-NN entry below for the third signal, added later). The scores are deliberately **not combined into one score**: `in_distribution=False` is raised if `mahalanobis_p_value < OOD_MAHALANOBIS_P_THRESHOLD` (a LOW p-value is anomalous) **or** `cosine_z > OOD_COSINE_THRESHOLD` (a HIGH z-score is anomalous) — note the comparison directions are opposite. Combining signals into one weighted mixture score risks a strong signal on one axis being diluted by a weak signal on the other; the OR rule and separately-exposed values also let a human reviewing predictions later see which metric fired. This is independent of `certain` (softmax-confidence-based) — a document can be `certain=True` (the softmax is confident) and `in_distribution=False` (the document doesn't resemble anything the model was trained on) at the same time, which is the payment-document failure mode this was built to catch. `OOD_MAHALANOBIS_P_THRESHOLD` (default `0.01`) has a defensible theoretical grounding since it's a genuine p-value. `OOD_COSINE_THRESHOLD` (default `2.5`) has **not** been statistically validated against a labeled out-of-category corpus. `evaluate-ood-calibration` measures the empirical false-positive rate of both thresholds against the model's own held-out test split (known in-distribution by construction, no retraining) and suggests better-calibrated values — run it before trusting either default in production.

**Mahalanobis empirical p-value replaces the chi² one for the actual decision (2026-07-10)**
A QQ-plot check of the training set's own Mahalanobis distances against the theoretical chi²(df) distribution showed the multivariate-Gaussian/shared-covariance assumption is badly violated for this corpus — observed distances run roughly 5x larger than chi² predicts, which is the direct explanation for Mahalanobis's measured 20-30% empirical false-positive rate against a 1% target (see `evaluate-ood-calibration` history above). `mahalanobis_p_value` is now computed empirically — `mahalanobis_empirical_p_value()` in `src/ood.py` ranks a query's raw Mahalanobis distance (to its **nearest** centroid, via `mahalanobis_min_distance()`) against `compute_train_mahalanobis_distances()`'s array of the training set's own distances (each training document to its **own true label's** centroid — deliberately not nearest-centroid, since that's what `compute_class_stats()`'s covariance estimation itself is built from; using nearest-centroid for the reference would let ambiguous/boundary training points look artificially unremarkable and corrupt the tail). This asymmetry (reference: true-label; query: nearest-centroid) is intentional, not a bug. The rank formula is the standard permutation-test empirical p-value: `(exceed_count + 1) / (N + 1)`, making no distributional assumption at all. `OOD_MAHALANOBIS_P_THRESHOLD` and `is_out_of_distribution()`'s comparison direction (`p < threshold` = anomalous) are unchanged — only how the p-value is computed changed. The original chi²-based value is kept as `mahalanobis_p_value_theoretical`, purely informational, explicitly never compared against a threshold or used in `is_out_of_distribution()` — combining it with the empirical value in the OR logic would be redundant, since both are monotonic transforms of the identical underlying distance. No `ood_stats.npz` format change or backfill was needed — `compute_train_mahalanobis_distances()` is a pure function of fields (`centroids`, `covariance_inv`, `knn_train_embeddings`, `knn_train_labels`) the file already stored for the k-NN signal. `BertTunningClassifier` computes the reference distance array once per process (`functools.cached_property`), not per request.

**k-NN class-conditional distance — a third OOD signal, complementary to Mahalanobis/cosine**
Both Mahalanobis (shared covariance) and cosine (single centroid) assume every class has one coherent "shape." Heterogeneous classes — most notably `otro`, a broad catch-all — violate that assumption, which is why Mahalanobis's empirically-measured false-positive rate for BETO models came back around 20-30% instead of the intended 1% (see `evaluate-ood-calibration` history below). `knn_mean_distance()` (`src/ood.py`) sidesteps the shape assumption entirely: `ood_stats.npz` also stores every training document's PCA-reduced embedding (`knn_train_embeddings`/`knn_train_labels`, not just per-class centroids — this is why the file grew from ~10KB to ~1.1MB), and at inference, `BertTunningClassifier.predict_text` measures the mean distance to the `k=10` (`OOD_KNN_NEIGHBORS`) nearest training documents *of the class the model just predicted* — deliberately class-conditional, not a global nearest-neighbor search, so it directly measures local density around the predicted class's own training examples regardless of how the class's overall shape compares to others. A HIGH distance is anomalous — same direction as cosine. Attached to `PredictResult.knn_distance`, folded into the same OR as the other two signals (`in_distribution=False` if any of the three fire). `OOD_KNN_DISTANCE_THRESHOLD` started as an uncalibrated placeholder (`5.0`) like `OOD_COSINE_THRESHOLD` originally was; `evaluate-ood-calibration` measured a 21.88% empirical false-positive rate at that placeholder against BETO v2's held-out test split (vs. a 1% target), so it was recalibrated to `26.125` — re-run `evaluate-ood-calibration` if the training corpus changes materially. `evaluate-ood-calibration` uses each test document's *true* label (not a live forward-pass prediction) as the "predicted class" for k-NN purposes, since the test split is known in-distribution by construction — a documented approximation that avoids adding a full classification pass to a command that otherwise only extracts embeddings. `compute-ood-stats` backfills `ood_stats.npz` (including the k-NN fields) for already-trained models the same way it always has, no retraining required; `ood_stats.npz` files generated before this feature landed are missing the k-NN fields and must be regenerated (`load_stats` raises `KeyError` on an old file). `ood_stats.npz` for BETO v1/v2 is now committed to the repo (previously git-ignored and regenerated on demand) after it silently disappeared once — regenerable, but it broke OOD scoring with no error until noticed, because it was un-ignored but never actually committed.

**Extraction metadata (`extract_pdf_with_metadata`) — which extractor produced the text, alongside the text itself**
When a document is misclassified, there was previously no way to see what text was actually extracted or which extractor (MarkItDown vs. OCR) produced it. `extract_pdf_with_metadata()` returns an `ExtractionMetadata` (`text`, `extractor_used`, `char_count`); `extract_pdf()` is now a thin wrapper around it (`.text`), so its signature and behavior are unchanged and `src/ingestion/scan.py` needed no modification. `predict_pdf`/`predict_folder` attach `extracted_text`/`extractor_used` onto `PredictResult`, so the `predict-folder` CSV, the CLI `predict` output, and the `/predict` API response all surface what the model actually saw — essential for telling apart an extraction-quality problem (garbled OCR) from a genuine out-of-category document with clean text.

**One `src/wandb.py` module for all W&B interaction — training class + one-shot CLI reporting functions**
`src/training/wandb_logger.py`'s `WandbLogger` class (stateful: `init`/`log_results`/`finish` tied to a single `Trainer` run's lifecycle) and a separate `src/wandb_logging.py` (one-shot functions for `predict-folder`/`evaluate-ood-calibration`'s `--log-wandb` flag) were briefly two files with near-identical names before being merged into one `src/wandb.py`. They intentionally were **not** unified into one shared abstraction beyond that — `WandbLogger` and the two `log_*_results()` functions have genuinely different call shapes (a long-lived class vs. fire-and-forget `init → log → finish` in one function body) and forcing them into a common interface would either bolt unrelated methods onto one class or throw away the type safety of `Hyperparams`/`EvaluationResult`/`CalibrationReport`/`PredictResult`. `predict-folder --log-wandb` logs a `wandb.Table` of per-document predictions (`job_type="predict-folder"`); `evaluate-ood-calibration --log-wandb` logs the empirical FP-rate/suggested-threshold summary (`job_type="ood-calibration"`). Both flags default to `False` — nothing changes about local CSV/console output when omitted.

**`populate_by_name=True` is required on every Pydantic model built via direct snake_case keyword construction, not just CLI-facing ones**
The gotcha documented below for CLI options classes turned out to be broader: `PredictResponse` (`alias_generator=to_camel`, no `populate_by_name`) is constructed in `src/api/routes/predict/endpoints.py` with snake_case kwargs (`mahalanobis_p_value=...`, `extracted_text=...`) — Pydantic v2 silently drops any kwarg that isn't the field's alias instead of raising, so the `/predict` API response returned `null`/default values for the OOD and extraction-metadata fields regardless of what was actually computed, undetected until a real end-to-end API test was added. The same pattern was found in `PredictResult` itself: `classify.py` constructs `PredictResult(all_scores={...})` directly, silently dropping `all_scores` to `{}` on every real prediction (the OOD fields on that same class were unaffected because they're set via `model_copy`, which bypasses the alias check entirely). Both `PredictResult` and `BaseSchema` now set `populate_by_name=True`. **Rule of thumb:** any Pydantic model with `alias_generator=to_camel` that is ever constructed with keyword arguments matching its Python field names (directly, or via `model_validate`/a dict) — as opposed to exclusively via `model_copy(update=...)` — must set `populate_by_name=True`, or fields whose alias differs from the field name (any multi-word snake_case name) will silently keep their default value with no error.

**`[tool.uv] package = false`**
This is a script project, not an installable package. Without this, uv tries to build it with hatchling and fails.

**`pythonpath = ["."]` in pytest**
Enables `from src.ingestion.extract import ...` imports in tests without installing the package.

**MAX_TOKENS hard limit**
XLM-RoBERTa and BETO have a 512-token architectural maximum (positional embeddings). The median document in the corpus is ~654 tokens. `CHUNK_STRATEGY="middle"` (first 256 + last 256) captures more signal than `"first"` for longer documents (ordenanzas, resoluciones). Note: `val`/`test` splits in `training/pipeline.py` currently hardcode `"first"` regardless of `CHUNK_STRATEGY` — only `train` respects the configured strategy. Harmless while `CHUNK_STRATEGY="first"` is the only strategy in active use.

**`populate_by_name=True` required alongside `alias_generator=to_camel` on CLI-facing Pydantic models**
`TrainOptions` and `PredictFolderOptions` set `alias_generator=to_camel` so they can also accept camelCase JSON. But Click always passes kwargs in snake_case (the Python parameter name), and Pydantic v2's `model_validate()` only accepts the alias form when an `alias_generator` is set — silently falling back to field defaults for any key it doesn't recognize as an alias, without raising, unless the field is required (in which case it raises `Field required` for the alias name, which is confusing since the caller passed the snake_case name). Any new Pydantic options class fed from Click must set `populate_by_name=True` or its CLI flags will be silently ignored.

**`extract_pdf` raises `BertTunningError` when every extractor in the chain fails**
Previously returned `None` silently. Now: individual extractor failures are logged at `WARNING` and the chain continues to the next extractor; only if *all* extractors raise does `extract_pdf` raise `BertTunningError`. `src/ingestion/scan.py`'s `build_dataset()` wraps the `extract_pdf` call in a try/except on `BertTunningError`, logging and skipping the document (incrementing `skipped`) instead of aborting the whole scan — a single totally-unreadable PDF during a `train` ingestion run no longer takes down the run.

**`compute_class_weight` uses `np.arange(num_labels)`, not `np.unique(train_df["label_id"])`**
If a class is absent from the training split (small per-class counts + stratified split can produce this), computing weights only over classes present in `train_df` produces a weight tensor shorter than `num_labels`, which crashes `CrossEntropyLoss`. `np.arange(num_labels)` guarantees every class gets a weight (sklearn assigns `1.0` to any class absent from `y`).

## Git Workflow

```bash
# Start a task branch from whichever integration branch is currently active
git worktree add -b task/N-name ../bert_tunning-taskN feature/ood-detection
cd ../bert_tunning-taskN
# implement, test...
uv run poe check
# commit, push, and gh pr create are the human's action by default — see note below
git push -u origin task/N-name
gh pr create --base feature/ood-detection --title "Task N: ..."

# Clean up worktree after PR is open
cd "c:/Users/leona/source/repos/bert_tunning"
git worktree remove ../bert_tunning-taskN
```

**Commit/push ownership:** applying file changes and running `uv run poe check` is fine for Claude to do on its own within a task. Committing (`git commit`), pushing (`git push`), and opening/editing a PR (`gh pr create`/`gh pr edit`) are the human's actions by default — Claude should report the change is ready and stop, only performing those git operations when explicitly asked to for that specific instance.

**Review every generated PR:** whenever a PR is created (by Claude when explicitly asked, or by the human), dispatch an agent to evaluate it using the repo's code-review skill and post comments on the PR only if there's something worth flagging — don't post a comment just to say "looks fine."

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
