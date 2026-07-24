# Bert Tunning

Fine-tunes transformer models (default: [BETO](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased), Spanish-native BERT) on Spanish municipal PDF documents to classify them by document type — `decreto`, `ordenanza`, `resolución`, etc. — for Argentine municipalities. Ships a CLI (train/predict/serve), a FastAPI `/predict` endpoint, and a React frontend for batch PDF classification with results review.

**For everything not covered here** — OOD detection internals, calibration math, training results/dataset breakdown, the SVM independent reviewer, W&B logging, the full API contract, adding a new model — see **[REFERENCE.md](REFERENCE.md)**.

## Requirements

- Python ≥ 3.10
- CUDA-capable GPU (tested on NVIDIA RTX A4000 Laptop, 8 GB VRAM, CUDA 11.8) — CPU works, much slower
- [uv](https://docs.astral.sh/uv/)

## Install

```powershell
uv sync
```

`torch` pulls from the PyTorch CUDA 11.8 index automatically via `[tool.uv.sources]` in `pyproject.toml`.

## Quick start

```powershell
# Classify a single PDF against the committed default model (BETO v2)
uv run python main.py predict path/to/documento.pdf

# Classify a whole folder → CSV
uv run python main.py predict-folder path/to/folder

# Serve the API (+ bundled frontend, if built) at http://localhost:8000
uv run python main.py serve --model-path ./models/bert_tunning_model_beto_v2/final

# Train a new model from scratch
uv run python main.py train --docs-root "C:\path\to\downloads"
```

Or with Docker (builds the frontend too):

```powershell
docker build -t bert-tunning .
docker run -p 8000:8000 -v ./models/bert_tunning_model_beto_v2:/app/models/bert_tunning_model_beto_v2 bert-tunning
```

## Critical parts to know before touching this codebase

- **Out-of-distribution (OOD) detection** — four independent signals (Mahalanobis, cosine, k-NN, TF-IDF), deliberately *not* blended into one score, OR'd together instead. A document can be confidently, wrongly classified (`certain=True`) and still get flagged (`inDistribution=false`) — this is the mechanism that catches documents that don't belong to any trained class. Thresholds are calibrated per model, not global constants. → [REFERENCE.md § OOD scoring internals](REFERENCE.md#ood-scoring-internals)
- **Review routing** — every prediction gets a `reviewRoute` (`accept`/`llm_judge`/`human_review`) computed from confidence + OOD + the SVM reviewer's agreement, so nothing downstream has to re-derive "is this trustworthy" from raw fields. → [REFERENCE.md § Review routing](REFERENCE.md#review-routing)
- **SVM independent reviewer** — a fifth, deliberately separate signal (never folded into OOD or `inDistribution`) that catches misclassification *among* known classes, not "is this any known class at all." → [REFERENCE.md § SVM independent reviewer](REFERENCE.md#svm-independent-reviewer)
- **Settings live in `.env`, not hardcoded** — `src/settings.py` (`Settings`) is a Pydantic `BaseSettings`; every value is overridable via a project-root `.env` file, never committed. → [REFERENCE.md § Configuration](REFERENCE.md#configuration)
- **`/predict` is job-polling, not synchronous** — `POST /predict` returns a job id immediately; the frontend (and any other client) polls `GET /predict/status/{job_id}` for progress and the final result. CLI commands (`predict`/`predict-folder`) bypass this entirely and call the inference pipeline directly. → [REFERENCE.md § Prediction job polling](REFERENCE.md#prediction-job-polling)
- **Run `uv run poe check` before every commit** — lint + mypy strict + pytest, in that order.

## Project structure (top level)

```
src/           pipeline: ingestion → training → inference → api / cli
frontend/      React + TypeScript + Vite SPA for /predict
docs/          design specs (docs/superpowers/specs/)
Dockerfile     multi-stage: node frontend-builder + uv builder + python runtime
main.py        Click CLI entry point
```

Full breakdown of every module → [REFERENCE.md § Project structure](REFERENCE.md#project-structure).

## Development

```powershell
uv run poe check      # lint + typecheck + test (run before every commit)
uv run poe fmt        # auto-format with ruff
uv run poe test       # pytest
```

See also: [CLAUDE.md](CLAUDE.md) (architecture + every technical decision's rationale, for AI assistants and humans alike) and [REFERENCE.md](REFERENCE.md) (the full user-facing reference this README was split out of).
