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
uv run python main.py compute-svm-classifiers --model-path ./models/bert_tunning_model/final --model xlm-roberta --cache-path ./data/bert_tunning_cache.parquet
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
├── ood.py              OOD math — compute_class_stats, mahalanobis_p_value, cosine_z_score, knn_mean_distance, save_stats/load_stats. Lives at top level (not under inference/) because compute_class_stats/save_stats are training-time producers consumed by src/training/pipeline.py, while mahalanobis_p_value/cosine_z_score/knn_mean_distance/load_stats are inference-time consumers used by src/inference/classify.py — same shared-across-layers treatment as wandb.py. Everything here moves in lockstep with one trigger: the evolution of ClassEmbeddingStats — model-forward-pass mechanics live separately in embeddings.py (see below), since that's a different axis of change with a different (torch/transformers-heavy) dependency surface
├── embeddings.py       LoadedModel, extract_embeddings, extract_embeddings_and_predictions — running a model's forward pass to get [CLS] embeddings (+ optionally predictions). Split out of ood.py: used by both src/training/pipeline.py and src/cli/_ood_common.py (training and CLI are sibling layers, so this is top-level, not owned by either), and it doesn't change when ClassEmbeddingStats/OOD math changes — isolating it also keeps torch/transformers out of anything that only needs ood.py's stats/persistence functions
├── svm_reviewer.py      fit_svm_classifiers/save_svm_classifiers/load_svm_classifiers/svm_scores — a fifth, independent signal (per-class one-vs-rest SVM on the raw [CLS] embedding), never folded into the OOD ensemble. Top-level for the same used-by-both-layers reason as ood.py/embeddings.py: fit at training time (src/training/pipeline.py), scored at inference time (src/inference/classify.py). See "SVM independent reviewer" below and docs/superpowers/specs/2026-07-15-svm-independent-reviewer-design.md
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
    ├── svm_classifiers.py @click.command "compute-svm-classifiers" — backfills svm_classifiers.joblib for an already-trained model, no retraining; independent of compute-ood-stats
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
| `OOD_TFIDF_COSINE_THRESHOLD` | `2.5` | TF-IDF cosine z-score above which a document is flagged `in_distribution=False` — see the TF-IDF detection note below |
| `OOD_TFIDF_MAX_FEATURES` | `5000` | Vocabulary size cap for the TF-IDF vectorizer fitted at training time |
| `MAX_UPLOAD_SIZE_BYTES` | `26214400` (25 MB) | `/predict` rejects uploads larger than this with a 413, read in bounded chunks |

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
Both Mahalanobis (shared covariance) and cosine (single centroid) assume every class has one coherent "shape." Heterogeneous classes — most notably `otro`, a broad catch-all — violate that assumption, which is why Mahalanobis's empirically-measured false-positive rate for BETO models came back around 20-30% instead of the intended 1% (see `evaluate-ood-calibration` history below). `knn_mean_distance()` (`src/ood.py`) sidesteps the shape assumption entirely: `ood_stats.npz` also stores every training document's PCA-reduced embedding (`knn_train_embeddings`/`knn_train_labels`, not just per-class centroids — this is why the file grew from ~10KB to ~1.1MB), and at inference, `BertTunningClassifier.predict_text` measures the mean distance to the `k=10` (`OOD_KNN_NEIGHBORS`) nearest training documents *of the class the model just predicted* — deliberately class-conditional, not a global nearest-neighbor search, so it directly measures local density around the predicted class's own training examples regardless of how the class's overall shape compares to others. A HIGH distance is anomalous — same direction as cosine. Attached to `PredictResult.knn_distance`, folded into the same OR as the other two signals (`in_distribution=False` if any of the three fire). `OOD_KNN_DISTANCE_THRESHOLD` started as an uncalibrated placeholder (`5.0`) like `OOD_COSINE_THRESHOLD` originally was; `evaluate-ood-calibration` measured a 21.88% empirical false-positive rate at that placeholder against BETO v2's held-out test split (vs. a 1% target), so it was recalibrated to `26.125` — re-run `evaluate-ood-calibration` if the training corpus changes materially. `evaluate-ood-calibration` originally used each test document's *true* label as the "predicted class" for k-NN purposes, since the test split is known in-distribution by construction — a documented approximation that avoided adding a full classification pass to a command that otherwise only extracted embeddings. This approximation was corrected on 2026-07-12 (see "k-NN calibration scores against the model's predicted label, not the true label" below) — it now runs a real forward pass and uses the model's actual prediction, matching `predict_text()` exactly. `compute-ood-stats` backfills `ood_stats.npz` (including the k-NN fields) for already-trained models the same way it always has, no retraining required; `ood_stats.npz` files generated before this feature landed are missing the k-NN fields and must be regenerated (`load_stats` raises `KeyError` on an old file). `ood_stats.npz` for BETO v1/v2 is now committed to the repo (previously git-ignored and regenerated on demand) after it silently disappeared once — regenerable, but it broke OOD scoring with no error until noticed, because it was un-ignored but never actually committed.

**TF-IDF cosine-centroid — a fourth OOD signal, catching lexical divergence the embedding-based signals cannot (mostly)**
Mahalanobis, cosine, and k-NN all operate on the `[CLS]` embedding's semantic "shape" — which means a document sharing the same document-type genre (e.g. a decree) but naming a different municipality than any in the training corpus can be nearly indistinguishable from a genuine in-distribution document in that space, since BERT's embedding compresses away the specific place name in favor of "this is decree-shaped text." `compute_tfidf_stats()` (`src/ood.py`) fits a `TfidfVectorizer` + per-class centroids directly on training text's raw vocabulary instead, and `tfidf_cosine_z_score()` scores new documents the same way `cosine_z_score` already does (cosine distance to nearest centroid, z-scored against the training set) — just in TF-IDF space, where a different city's name is in principle a distinguishing feature rather than noise the embedding smooths over. Persisted through the existing `ood_stats.npz` with no new artifact file: `TfidfVectorizer.vocabulary_`/`.idf_` round-trip through two plain arrays (`tfidf_vocabulary_terms`, `tfidf_idf`), verified to reconstruct a vectorizer whose `.transform()` output is bit-identical to the originally-fitted one. Five of the six new `ClassEmbeddingStats` fields default to an empty/placeholder sentinel rather than `None` (only `tfidf_threshold` stays `Optional`, matching the other three thresholds' independently-calibrated-later precedent) — an `ood_stats.npz` predating this feature has the signal skipped entirely (not treated as anomalous), and must be regenerated via `compute-ood-stats` to gain it. Folded into the same OR as the other three signals (`in_distribution=False` if any of the four fire) — never blended into one score, same rationale as the original Mahalanobis/cosine decision. `OOD_TFIDF_COSINE_THRESHOLD` calibrated immediately close to its uncalibrated placeholder (`2.5` → `2.5164` for BETO v2, `2.5` → `2.5384` for BETO v1) via `evaluate-ood-calibration --write-thresholds`, landing almost exactly at the 1% target FP rate on both models' held-out test splits.

**Known limitation, found at verification time — TF-IDF did not catch the cross-jurisdiction case it was built for.** Testing against a small real-document sample (`samples/`, BETO v2) confirmed the signal works correctly end-to-end and adds real value for at least one case (a payment-receipt document scores `tfidf_cosine_z=3.28` against a `2.52` threshold, correctly reinforcing what Mahalanobis already flagged) — but it did **not** fire for either of the two known cross-jurisdiction decrees in that sample (`document_predict_2.pdf`, a Córdoba tax-code document, scored `-0.835`; `document_predict_3.pdf`, a Santa Fe decree, scored `0.469`; both far below threshold — an earlier version of this note had these two swapped, corrected once the actual extracted text was checked), which is the exact failure mode this signal was motivated by. The likely explanation: a full legal decree shares hundreds of boilerplate legal-vocabulary tokens with in-distribution training documents, and TF-IDF's cosine distance is computed over the whole vector — one or two distinguishing tokens (a city name mentioned once or twice) get diluted by the shared boilerplate rather than dominating the distance. A follow-up tried excluding shared boilerplate from the vocabulary via `OOD_TFIDF_MAX_DF` (below) — it tightened calibration meaningfully but still didn't close this specific gap (`document_predict_3.pdf`'s score barely moved, `0.469→0.588`, while the threshold itself dropped from `2.52→1.54`). This confirmed the dilution problem isn't fixable by vocabulary tuning alone; a keyword/entity check for known municipality names (`detect_foreign_municipality`, below) was built instead as the more direct fix. TF-IDF cosine stays in place as a fourth, independent signal that may still catch other kinds of lexical drift (as the payment-receipt case shows) even though it underperformed on its original motivating case.

**`OOD_TFIDF_MAX_DF` — exclude shared boilerplate from the TF-IDF vocabulary**
With no `max_df` set, `TfidfVectorizer`'s 5000-feature budget was dominated by whatever's most frequent corpus-wide — for legal decrees, that's shared boilerplate ("considerando", "por cuanto", "el intendente municipal"), not the rarer terms (like city names) that could actually distinguish an out-of-jurisdiction document. `OOD_TFIDF_MAX_DF` (default `0.5`) excludes any term appearing in over half the training corpus from the vocabulary entirely, freeing up feature budget for rarer terms and stopping boilerplate from diluting the cosine-distance calculation. Recalibrating after this change moved `tfidf_threshold` meaningfully (BETO v1 `2.538→1.461`, BETO v2 `2.516→1.543`), confirming it does tighten the in-distribution cluster — but see the known-limitation paragraph above for why this alone didn't solve the cross-jurisdiction case that originally motivated trying it.

**`detect_foreign_municipality` — a deterministic, independent signal outside the OOD ensemble**
Every embedding/lexical OOD signal (Mahalanobis, cosine, k-NN, TF-IDF) answers a *statistical* question — "how far is this document from what's been seen" — which a single distinguishing token (one city name amid hundreds of shared boilerplate words) can't reliably move, as the TF-IDF limitation above demonstrates concretely. `detect_foreign_municipality()` (`src/ingestion/_text.py`) answers a *categorical* question instead: does this document's text explicitly name a municipality other than `OOD_TRAINED_MUNICIPALITY` (`"rosario"` — this corpus is mono-jurisdictional; verified directly against `bert_tunning_cache_con_otro_300.parquet`, where "rosario" appears in 1714/1920 training documents). It matches the literal `"Municipalidad de <Name>"` phrase (tolerant of a `"la"`/`"ciudad de"` prefix, and of stray `|` characters — `MarkItDown` sometimes renders a PDF letterhead as a malformed single-cell markdown table, splitting the phrase mid-word) rather than a bare city-name substring search, because a bare substring search is too loose: "Córdoba" and "Santa Fe" both appear in genuinely in-distribution Rosario training documents too — as a street name (`"Calle Córdoba"`, a real street in Rosario) and as the name of the province Rosario belongs to (`"Provincia de Santa Fe"`), respectively. Anchoring to the actual jurisdiction-claim phrase avoids both false positives, at the cost of staying silent (`None`, not `False`) for the ~11% of training documents that never name their municipality this way at all. Returns `None` when no such phrase is found, or when every phrase found matches the known municipality; the matched foreign name otherwise. Attached to `PredictResult.foreign_municipality`, computed once per document in `_attach_metadata()` (`src/inference/pipeline.py`) and the `/predict` API route — deliberately **not** folded into the OR-based `in_distribution` ensemble in `src/inference/classify.py`: a categorical fact ("this document names a city we've never trained on") is stronger and more literal than a continuous z-score, and blending it into the same OR would treat it as just another vote instead of the direct evidence it actually is. Verified against `samples/`: correctly flags `document_predict_2.pdf` (`"Córdoba"`, also independently caught by Mahalanobis) and, notably, `document_predict_3.pdf` (`"SANTA"`) — the exact case every other signal, including TF-IDF after the `max_df` follow-up, missed.

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

**Per-model OOD thresholds, not a single global config (2026-07-12)**
`Settings.OOD_MAHALANOBIS_P_THRESHOLD`/`OOD_COSINE_THRESHOLD`/`OOD_KNN_DISTANCE_THRESHOLD` were being applied identically regardless of which model was loaded — but they were calibrated specifically against BETO v2's embedding space, corpus size, and empirical rank floor (see the Mahalanobis resolution-floor paragraph above). A different model (a freshly trained XLM-RoBERTa or MiniLM, or a BETO v2 retrained on a different corpus) would receive apparently-valid but statistically unrelated OOD decisions. `ClassEmbeddingStats` (`src/schema.py`) now carries optional `mahalanobis_p_threshold`/`cosine_threshold`/`knn_distance_threshold` fields, `None` by default. `resolve_ood_thresholds()` (`src/ood.py`) reads them from whichever `ood_stats.npz` is loaded, falling back to `Settings.OOD_*` only when they're unset — backward compatible with artifacts that predate this change. `is_out_of_distribution()` now takes an explicit `OodThresholds` instead of reading `Settings` directly. `evaluate-ood-calibration --write-thresholds` persists its suggested values back into that exact model's `ood_stats.npz`, refusing to write a Mahalanobis threshold at or below that model's own empirical resolution floor (the same degenerate-suggestion class of bug this project hit once already) — cosine/k-NN still get written even when Mahalanobis is refused.

**k-NN calibration scores against the model's predicted label, not the true label (2026-07-12)**
`evaluate-ood-calibration` was passing each held-out test document's *true* label into `knn_mean_distance()`, while `predict_text()` in production always passes the model's *predicted* label (`pred_idx = argmax(probs)`). For a misclassified in-distribution document, these differ — and misclassified documents are exactly the ones likely to have large k-NN distances against the wrong class's neighbors, so the true-label shortcut made the reported empirical false-positive rate optimistic relative to what production actually experiences. `extract_embeddings_and_predictions()` (`src/ood.py`) now runs the model's real forward pass (mirroring `predict_text`'s own `output_hidden_states=True` call, not the `base_model`-only path `extract_embeddings` uses for training/`compute-ood-stats`, which never need predictions) and returns each document's `argmax` prediction alongside its embedding. `embed_texts_and_predict()` (`src/cli/_ood_common.py`) wraps it for `evaluate-ood-calibration`'s use; Mahalanobis and cosine calibration are unaffected, since neither takes a predicted label as input.

**`ood_stats.npz` class mapping validated at classifier construction (2026-07-12)**
`BertTunningClassifier` previously loaded whatever `ood_stats.npz` sat next to a model's checkpoint with zero validation that it actually belonged to that model — a copied, stale, or reordered file would silently score embeddings against the wrong centroids, and `knn_mean_distance()` would interpret a predicted label id against the wrong class entirely (it indexes `stats.knn_train_labels` directly by that id). `_validate_ood_stats_class_mapping()` now compares `stats.class_names` (ordered) against `model.config.id2label` (ordered by index) once, at classifier construction — raising `BertTunningError` immediately rather than corrupting every subsequent prediction's OOD scores silently. The existing class-mismatch check in `src/cli/_ood_common.py` (used by `compute-ood-stats`/`evaluate-ood-calibration`) only ever covered the CLI backfill/calibration path, never the ordinary `predict`/`predict-folder`/`serve` path that most traffic actually goes through.

**Follow-up hardening: model identity fingerprint, atomic writes, uncalibrated-threshold visibility, W&B parity (2026-07-13)**
A critical review of the per-model OOD threshold work above found four gaps. (1) `_validate_ood_stats_class_mapping()` only compared `class_names`, which is a property of the training corpus, not the model — since this project's three registered models (`xlm-roberta`, `beto`, `minilm`) are commonly trained on the identical label set, a `beto` model's `ood_stats.npz` copied next to an `xlm-roberta` checkpoint would pass that check trivially. `ClassEmbeddingStats` now also carries an optional `model_type`/`model_hidden_size` fingerprint (from `model.config.model_type`/`.hidden_size` at `compute_class_stats()` time), validated by a new `_validate_ood_stats_model_identity()` — skipped entirely for stats predating the field, enforced when present. (2) `save_stats()` wrote directly to `ood_stats.npz` with no atomicity; an interrupted write could corrupt the only copy a running server reads from. It now writes to a temp file, verifies the temp file loads back with `load_stats()`, then `os.replace()`s it onto the real path. (3) `resolve_ood_thresholds()`'s Settings fallback was completely silent — a freshly trained, never-calibrated model would inherit BETO v2's thresholds with no visibility. `BertTunningClassifier` now logs one `WARNING` at construction naming which specific thresholds are falling back. This is deliberately a warning, not a startup failure or a disabled signal: BETO v2's own `mahalanobis_p_threshold` is `None` because `--write-thresholds`'s degenerate-guard correctly refused to persist a floor-adjacent value, not because it was never calibrated — `Settings.OOD_MAHALANOBIS_P_THRESHOLD` genuinely is BETO v2's calibrated value in that case, so failing startup would break a correctly-configured model. A follow-up (2026-07-13) replaced this prose-only distinction with an explicit `mahalanobis_threshold_status` field (`"not_calibrated"` / `"calibrated"` / `"refused_degenerate"`) on `ClassEmbeddingStats`, so `_warn_on_uncalibrated_thresholds` logs an actionable `WARNING` only for the genuinely-uncalibrated case and a separate non-actionable `INFO` line for the expected-refusal case, instead of one `WARNING` whose own text had to explain both possibilities. (4) `log_ood_calibration_results()` logged `Settings.OOD_*` directly instead of the resolved per-model thresholds (the same gap already fixed in `evaluate-ood-calibration`'s own console/log output during PR #43's review), and never logged a k-NN threshold at all — both fixed, threading the same `OodThresholds` the CLI's log lines use.

**SVM independent reviewer — a fifth signal, deliberately outside the OOD ensemble (2026-07-16)**
`evaluate_svm_classifiers()` (`src/svm_reviewer.py`) reports each class's held-out **balanced accuracy** (not plain accuracy — each one-vs-rest task is itself imbalanced, one class positive against every other class negative) scored against the val split, never the training data the classifiers were fit on. Both `train` (`src/training/pipeline.py`) and `compute-svm-classifiers` log it right after fitting — a real generalization signal, not just a "the backfill ran" record. (An earlier version of this logged a one-shot placeholder summary to W&B instead — just a class count, no accuracy; reverted once it became clear the number itself carried no information.)

**SVM reviewer results in W&B — per-class table, not a placeholder count**
`_svm_results_payload()` (`src/wandb.py`) builds one shared payload — a `svm/per_class_accuracy` `wandb.Table` with **class, training-sample count, and held-out balanced accuracy together in the same row** — used by both `WandbLogger.log_svm_results()` (logs into a `train` run's already-open W&B session, a no-op when `wb` is disabled) and the standalone `log_svm_classifiers_results()` (`compute-svm-classifiers --log-wandb`'s own one-shot `init`/`log`/`finish`). Training sample count travels alongside accuracy specifically so a low score is explainable in the same table row (e.g. `otro`'s ~0.60 balanced accuracy sitting next to its 37 training documents) rather than an isolated number inviting "the SVM is bad at this class" instead of the real story, "this class barely had any training data." `svm/mean_balanced_accuracy`/`svm/min_balanced_accuracy` give an at-a-glance summary; one `svm/balanced_accuracy/<class>` scalar per class supports charting a specific class's trend across retraining runs, which a table cell alone can't do in W&B's UI.
Every OOD signal (Mahalanobis, cosine, k-NN, TF-IDF) and `detect_foreign_municipality` feed into either `in_distribution` or a standalone categorical field — both are decisions made *in this repo*. This project's downstream consumer, **Classiflow** (an agentic workflow consuming this trained model plus a conjunction of OOD and independent classifiers), needs raw per-class evidence to weigh itself instead. `fit_svm_classifiers()` (`src/svm_reviewer.py`) fits one one-vs-rest `sklearn.svm.SVC` (RBF kernel, `class_weight="balanced"`) per class on the training split's **raw, pre-PCA** `[CLS]` embedding — matching Peña et al. 2023's validated config for imbalanced topic classification (particularly relevant given `otro`=46/`declaracion_concejo_municipal`=37 doc counts), and deliberately not coupled to `OOD_PCA_COMPONENTS` (a dimensionality chosen for Mahalanobis's covariance-estimation needs, which this signal doesn't share). Persisted as `svm_classifiers.joblib` next to `ood_stats.npz` (same atomic-write pattern as `save_stats()` — temp file, verify load-back, `os.replace()`), backfillable via a standalone `compute-svm-classifiers` command independent of `compute-ood-stats`. `PredictResult.svm_scores: dict[str, float]` exposes every class's decision-function margin (mirroring `all_scores`' shape, not `OodMetrics`' nested/flattened treatment) — `{}`, not `None`, when the artifact is missing (computed from the same forward pass already used for softmax, no extra inference). **Deliberately never folded into `in_distribution`**, and deliberately has no threshold, no calibration, no `Settings.OOD_SVM_*` config — there is nothing to calibrate when nothing in this repo makes a decision from the raw margins themselves. (**Update 2026-07-16:** `svm_scores` *is* now used for one derived decision — see the disagreement entry below — the "never folded into `review_route`" half of this claim no longer holds; only the `in_distribution`/no-threshold parts still do. **Update 2026-07-16 (2):** `svm_scores` changed from `dict[str, float] | None` to a plain `dict[str, float] = {}` — "no scores" has a natural empty-collection representation, matching `all_scores` itself never being `Optional` in the same class; see the disagreement entry's `svm_predicted_label` discussion for the fuller `/stop-using-none` reasoning, which applied equally here.) `BertTunningClassifier` validates the classifiers dict's class *set* (not order — it's a dict keyed by name, not an array indexed by label id like `ood_stats.npz`'s `class_names`) against `model.config.id2label` at construction, mirroring `_validate_ood_stats_class_mapping()`'s fail-fast-once rationale. Flows into the `predict-folder` CSV automatically, with no new plumbing, since `flatten_predict_result()`'s `model_dump()`-based row forwards every top-level `PredictResult` field. **Correction (2026-07-16): this claim was wrong for the `--log-wandb` table specifically.** `log_predict_folder_results()`'s `wandb.Table` doesn't use the flattened row directly — it cherry-picks a fixed `_PREDICTION_COLUMNS` list (`src/wandb.py`), so a field absent from that list is silently dropped from the W&B table even though it's present in the CSV, discovered only when `svm_scores` didn't show up in a real `--log-wandb` run despite showing up in the CSV from the same run. `svm_scores` is now in `_PREDICTION_COLUMNS`. Any future `PredictResult` field intended for the W&B predictions table needs the same explicit addition — the CSV and the W&B table do not share one automatic path, only the CSV does. See `docs/superpowers/specs/2026-07-15-svm-independent-reviewer-design.md` for the full design rationale, including why this sits at `src/` top level and the option (misclassification among *known* classes) this signal does not address.

**SVM/softmax classifier disagreement — a second-opinion check for misclassification among known classes (2026-07-16)**
The SVM independent reviewer answers "does this document look like anything trained on" (Option A, OOD) — but it also incidentally produces a second, independently-trained opinion on *which* known class a document belongs to, which nothing previously checked (Option B: the document genuinely is a known class, softmax just picked the wrong one). `svm_top_label()` (`src/svm_reviewer.py`) returns the class with the highest `svm_scores` margin — the SVM reviewer's own "prediction." `predict_text()` (`src/inference/classify.py`) compares it against softmax's own pick (`label`) and attaches two new `PredictResult` fields: `svm_predicted_label: str` (what the SVM picked instead, `""` — not `None` — when no `svm_classifiers.joblib` is loaded; a real class name is never empty, so `""` can't collide with genuine data, and it stays type-compatible with `label`'s own comparison without a `None` check) and `svm_agrees_with_prediction: bool` (**plain `bool`, not `bool | None`** — a tri-state here would encode two different questions in one field: "was there SVM evidence" and "did it agree"; the first already has an owner, `svm_predicted_label` being `""`, and no caller ever branches differently on "no signal" vs. "agreed," so a `None` state wasn't earning its complexity; defaults to `True`, the same permissive-default-on-missing-artifact pattern as `OodEvidence.from_in_distribution(None)` → `NOT_ANOMALOUS`). Originally shipped as `str | None`/`bool | None` — revised same-day after review pushed on whether `None` was earning its complexity here at all; landed on empty-string/plain-bool instead of `Literal[*class_names, "sentinel"]` because the class list isn't fixed at type-checking time (varies per trained model), and a same-typed string sentinel (e.g. `"not_a_class"`) would lose `mypy`'s type-level None-vs-str distinction and risk colliding with a real future class name — `""` avoids both, since it's structurally impossible as a real (folder-derived) class name.

`decide_review_route()`'s signature grew a `classifier_disagreement: bool = False` parameter (default preserves every existing caller's behavior) — a disagreement routes to `human_review` **regardless of confidence**, the identical rationale OOD firing already gets, since a confident-but-wrong prediction is the dangerous case either way. This is **not** folded into `in_distribution`/`OodEvidence` — a document can be perfectly in-distribution and still trigger a disagreement (the two classifiers agreeing on genre but not on which specific known class), a different question than "does this resemble anything trained on at all." Both are independent triggers for the same `human_review` lane, not mutually exclusive, mirroring the OOD ensemble's own "OR, not one blended score" philosophy. Both `predict_text()` call sites of `decide_review_route()` pass `classifier_disagreement` — including the early-return path taken when `self._ood_stats is None`, which would otherwise silently drop the disagreement signal for any model without `ood_stats.npz`. `svm_predicted_label`/`svm_agrees_with_prediction` need explicit additions to `_PREDICTION_COLUMNS` (`src/wandb.py`) to reach the `predict-folder --log-wandb` table — the CSV gets them automatically via `flatten_predict_result()`, the W&B table does not, the exact gap already found once for `svm_scores` itself. See `docs/superpowers/specs/2026-07-16-svm-softmax-disagreement-design.md`.

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
| `# noqa: PLR0913` | `src/cli/ood_calibration.py:build_calibration_report` | One parameter per OOD signal (Mahalanobis/cosine/k-NN/TF-IDF), all required — no natural grouping without a wrapper object that would obscure the calibration math |
