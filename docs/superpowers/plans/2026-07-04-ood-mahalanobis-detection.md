# Out-of-Distribution Detection (Mahalanobis + Cosine Mixture) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect documents that belong to none of the trained classes (e.g. a payment document run through a model trained only on decreto/ordenanza/resolución/etc.) instead of letting the softmax classifier confidently force them into the nearest known class.

**Architecture:** After training, extract the `[CLS]` embedding for every training document, PCA-reduce it, and compute per-class centroids plus a shared (tied) covariance matrix. At inference time, project a new document's embedding into the same PCA space and score it by (a) Mahalanobis distance to the nearest class centroid and (b) cosine distance to the nearest class centroid. Combine both into one z-scored mixture score. If the score exceeds a threshold, the document is flagged `in_distribution=False` regardless of what the softmax classifier says. This runs alongside the existing classifier — no retraining of the classification head, no change to the 5 already-trained models' weights.

**Tech Stack:** NumPy, scikit-learn (`PCA`, already a dependency), PyTorch/Transformers (reused from the existing `BertTunningClassifier`), Pydantic v2 for the stats schema.

## Global Constraints

- Python ≥ 3.10, `X | Y` union types, not `Optional[X]` (per repo `CLAUDE.md`)
- Pydantic v2, `frozen=True` on all schema/config objects (per repo `CLAUDE.md`)
- `uv run poe check` (lint + typecheck + test) must pass before every commit
- No change to the training loop, model architecture, or any of the 5 already-trained model checkpoints — this is purely additive
- New artifact (`ood_stats.npz`) is optional at load time — `BertTunningClassifier` must work unchanged for any model directory that doesn't have one (backward compatibility with existing checkpoints)
- The OOD threshold cannot be statistically validated without labeled out-of-category documents (this is a known limitation for a thesis PoC with no negative-class corpus) — Task 6 documents this explicitly rather than pretending the default threshold is validated

---

## File Structure

| File | Responsibility |
|---|---|
| `src/inference/ood.py` (new) | All OOD math: PCA reduction, Mahalanobis distance, cosine distance, mixture score, batched embedding extraction, save/load of stats to `.npz` |
| `src/schema.py` (modify) | Add `ClassEmbeddingStats` (the artifact schema) and extend `PredictResult` with `ood_score`/`in_distribution` fields |
| `src/settings.py` (modify) | Add `OOD_PCA_COMPONENTS`, `OOD_MAHALANOBIS_WEIGHT`, `OOD_THRESHOLD` |
| `src/inference/classify.py` (modify) | `BertTunningClassifier` lazy-loads `ood_stats.npz` next to the model; `predict_text` computes and attaches the OOD fields in the same forward pass |
| `src/training/pipeline.py` (modify) | After `trainer.train()`, extract training-set embeddings and persist `ood_stats.npz` next to the saved model |
| `src/cli/ood_stats.py` (new) | Standalone `compute-ood-stats` command — backfills `ood_stats.npz` for the 5 already-trained models without retraining |
| `main.py` (modify) | Register `compute-ood-stats` command |
| `src/cli/predict.py` (modify) | Print the OOD fields in `predict_cmd` output |
| `src/api/routes/predict/schemas.py`, `src/api/routes/predict/endpoints.py` (modify) | Surface `oodScore`/`inDistribution` on `PredictResponse` |
| `tests/inference/test_ood.py` (new) | Unit tests for the math module — synthetic data, no model loading |
| `tests/inference/test_pipeline.py` (modify) | Extend the existing mocked-classifier tests to cover stats-present / stats-absent cases |
| `tests/api/test_predict.py`, `tests/cli/test_commands.py` (modify) | Cover the new response fields |

---

## Task 1: OOD math module + `ClassEmbeddingStats` schema

**Files:**
- Create: `src/inference/ood.py`
- Modify: `src/schema.py`
- Test: `tests/inference/test_ood.py` (new)

**Interfaces:**
- Produces: `ClassEmbeddingStats` (Pydantic model in `src/schema.py`) with fields `class_names: list[str]`, `pca_mean: npt.NDArray[np.float64]`, `pca_components: npt.NDArray[np.float64]`, `centroids: npt.NDArray[np.float64]`, `covariance_inv: npt.NDArray[np.float64]`, `maha_calibration_mean: float`, `maha_calibration_std: float`, `cosine_calibration_mean: float`, `cosine_calibration_std: float`
- Produces: `compute_class_stats(embeddings: npt.NDArray[np.float64], labels: list[int], class_names: list[str], *, n_components: int = 64, covariance_epsilon: float = 1e-6) -> ClassEmbeddingStats`
- Produces: `mahalanobis_min_distance(embedding: npt.NDArray[np.float64], stats: ClassEmbeddingStats) -> float`
- Produces: `cosine_min_distance(embedding: npt.NDArray[np.float64], stats: ClassEmbeddingStats) -> float`
- Produces: `ood_score(embedding: npt.NDArray[np.float64], stats: ClassEmbeddingStats, *, mahalanobis_weight: float = 0.7) -> float`
- Produces: `save_stats(stats: ClassEmbeddingStats, path: Path) -> None` and `load_stats(path: Path) -> ClassEmbeddingStats`
- Produces: `extract_embeddings(model: torch.nn.Module, tokenizer: PreTrainedTokenizerBase, texts: list[str], *, max_length: int, device: str, batch_size: int = 16) -> npt.NDArray[np.float64]`

- [ ] **Step 1: Write the failing tests**

Create `tests/inference/test_ood.py`:

```python
import numpy as np
import torch
from unittest.mock import MagicMock

from src.inference.ood import (
    compute_class_stats,
    cosine_min_distance,
    extract_embeddings,
    mahalanobis_min_distance,
    ood_score,
    save_stats,
    load_stats,
)


def _synthetic_embeddings() -> tuple[np.ndarray, list[int], list[str]]:
    rng = np.random.default_rng(42)
    class_a = rng.normal(loc=0.0, scale=0.1, size=(20, 16))
    class_b = rng.normal(loc=5.0, scale=0.1, size=(20, 16))
    embeddings = np.vstack([class_a, class_b])
    labels = [0] * 20 + [1] * 20
    return embeddings, labels, ["class_a", "class_b"]


def test_compute_class_stats_shapes() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    assert stats.centroids.shape == (2, 8)
    assert stats.covariance_inv.shape == (8, 8)


def test_in_distribution_point_has_lower_mahalanobis_distance_than_far_point() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    known_point = embeddings[0]
    far_point = np.full(16, 100.0)
    known_distance = mahalanobis_min_distance(known_point, stats)
    far_distance = mahalanobis_min_distance(far_point, stats)
    assert far_distance > known_distance


def test_in_distribution_point_has_lower_cosine_distance_than_far_point() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    known_point = embeddings[0]
    far_point = np.full(16, -100.0)
    known_distance = cosine_min_distance(known_point, stats)
    far_distance = cosine_min_distance(far_point, stats)
    assert far_distance > known_distance


def test_ood_score_is_higher_for_far_point() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    known_point = embeddings[0]
    far_point = np.full(16, 100.0)
    assert ood_score(far_point, stats) > ood_score(known_point, stats)


def test_save_and_load_stats_roundtrip(tmp_path) -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    path = tmp_path / "ood_stats.npz"
    save_stats(stats, path)
    loaded = load_stats(path)
    assert loaded.class_names == stats.class_names
    np.testing.assert_allclose(loaded.centroids, stats.centroids)
    np.testing.assert_allclose(loaded.covariance_inv, stats.covariance_inv)
    assert loaded.maha_calibration_mean == stats.maha_calibration_mean


def test_extract_embeddings_returns_correct_shape() -> None:
    tokenizer = MagicMock()
    tokenizer.return_value.to.return_value = {
        "input_ids": torch.zeros(2, 8, dtype=torch.long),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    model = MagicMock()
    model.base_model.return_value.last_hidden_state = torch.zeros(2, 8, 16)

    embeddings = extract_embeddings(
        model, tokenizer, ["doc one", "doc two"], max_length=8, device="cpu", batch_size=2
    )
    assert embeddings.shape == (2, 16)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/inference/test_ood.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.inference.ood'`

- [ ] **Step 3: Add `ClassEmbeddingStats` to `src/schema.py`**

Add this class after `EvaluationResult` (which ends at line 43) and before `class Hyperparams(BaseModel):`:

```python
class ClassEmbeddingStats(BaseModel):
    """Per-class embedding centroids + shared covariance for Mahalanobis/cosine OOD scoring."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    class_names: list[str]
    pca_mean: npt.NDArray[np.float64]
    pca_components: npt.NDArray[np.float64]
    centroids: npt.NDArray[np.float64]
    covariance_inv: npt.NDArray[np.float64]
    maha_calibration_mean: float
    maha_calibration_std: float
    cosine_calibration_mean: float
    cosine_calibration_std: float
```

`np`/`npt` are already imported at the top of `src/schema.py` (used by `EvaluationResult`), so no new imports are needed there.

- [ ] **Step 4: Write `src/inference/ood.py`**

```python
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from sklearn.decomposition import PCA
from transformers import PreTrainedTokenizerBase

from src.schema import ClassEmbeddingStats


def _reduce_dimensionality(
    embeddings: npt.NDArray[np.float64], n_components: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    capped = min(n_components, embeddings.shape[0] - 1, embeddings.shape[1])
    pca = PCA(n_components=capped)
    reduced = pca.fit_transform(embeddings)
    return reduced, pca.mean_.astype(np.float64), pca.components_.astype(np.float64)


def _project(embedding: npt.NDArray[np.float64], stats: ClassEmbeddingStats) -> npt.NDArray[np.float64]:
    return (embedding - stats.pca_mean) @ stats.pca_components.T


def _mahalanobis_min_distance_raw(
    point: npt.NDArray[np.float64],
    centroids: npt.NDArray[np.float64],
    covariance_inv: npt.NDArray[np.float64],
) -> float:
    diffs = centroids - point
    distances = np.einsum("kd,de,ke->k", diffs, covariance_inv, diffs)
    return float(np.min(distances))


def _cosine_min_distance_raw(
    point: npt.NDArray[np.float64], centroids: npt.NDArray[np.float64]
) -> float:
    point_norm = point / (np.linalg.norm(point) + 1e-9)
    centroid_norms = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
    similarities = centroid_norms @ point_norm
    return float(np.min(1.0 - similarities))


def compute_class_stats(
    embeddings: npt.NDArray[np.float64],
    labels: list[int],
    class_names: list[str],
    *,
    n_components: int = 64,
    covariance_epsilon: float = 1e-6,
) -> ClassEmbeddingStats:
    reduced, pca_mean, pca_components = _reduce_dimensionality(embeddings, n_components)
    labels_arr = np.asarray(labels)

    centroids = np.stack(
        [reduced[labels_arr == k].mean(axis=0) for k in range(len(class_names))]
    )
    centered = reduced - centroids[labels_arr]
    covariance = (centered.T @ centered) / reduced.shape[0]
    covariance_reg = covariance + covariance_epsilon * np.eye(covariance.shape[0])
    covariance_inv = np.linalg.inv(covariance_reg)

    maha_scores = np.array(
        [
            _mahalanobis_min_distance_raw(reduced[i], centroids, covariance_inv)
            for i in range(reduced.shape[0])
        ]
    )
    cosine_scores = np.array(
        [_cosine_min_distance_raw(reduced[i], centroids) for i in range(reduced.shape[0])]
    )

    return ClassEmbeddingStats(
        class_names=class_names,
        pca_mean=pca_mean,
        pca_components=pca_components,
        centroids=centroids,
        covariance_inv=covariance_inv,
        maha_calibration_mean=float(maha_scores.mean()),
        maha_calibration_std=float(maha_scores.std() + 1e-9),
        cosine_calibration_mean=float(cosine_scores.mean()),
        cosine_calibration_std=float(cosine_scores.std() + 1e-9),
    )


def mahalanobis_min_distance(embedding: npt.NDArray[np.float64], stats: ClassEmbeddingStats) -> float:
    point = _project(embedding, stats)
    return _mahalanobis_min_distance_raw(point, stats.centroids, stats.covariance_inv)


def cosine_min_distance(embedding: npt.NDArray[np.float64], stats: ClassEmbeddingStats) -> float:
    point = _project(embedding, stats)
    return _cosine_min_distance_raw(point, stats.centroids)


def ood_score(
    embedding: npt.NDArray[np.float64],
    stats: ClassEmbeddingStats,
    *,
    mahalanobis_weight: float = 0.7,
) -> float:
    maha_raw = mahalanobis_min_distance(embedding, stats)
    cosine_raw = cosine_min_distance(embedding, stats)
    maha_z = (maha_raw - stats.maha_calibration_mean) / stats.maha_calibration_std
    cosine_z = (cosine_raw - stats.cosine_calibration_mean) / stats.cosine_calibration_std
    return mahalanobis_weight * maha_z + (1 - mahalanobis_weight) * cosine_z


def save_stats(stats: ClassEmbeddingStats, path: Path) -> None:
    np.savez(
        str(path),
        class_names=np.array(stats.class_names),
        pca_mean=stats.pca_mean,
        pca_components=stats.pca_components,
        centroids=stats.centroids,
        covariance_inv=stats.covariance_inv,
        maha_calibration_mean=stats.maha_calibration_mean,
        maha_calibration_std=stats.maha_calibration_std,
        cosine_calibration_mean=stats.cosine_calibration_mean,
        cosine_calibration_std=stats.cosine_calibration_std,
    )


def load_stats(path: Path) -> ClassEmbeddingStats:
    data = np.load(str(path), allow_pickle=False)
    return ClassEmbeddingStats(
        class_names=data["class_names"].tolist(),
        pca_mean=data["pca_mean"],
        pca_components=data["pca_components"],
        centroids=data["centroids"],
        covariance_inv=data["covariance_inv"],
        maha_calibration_mean=float(data["maha_calibration_mean"]),
        maha_calibration_std=float(data["maha_calibration_std"]),
        cosine_calibration_mean=float(data["cosine_calibration_mean"]),
        cosine_calibration_std=float(data["cosine_calibration_std"]),
    )


def extract_embeddings(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    *,
    max_length: int,
    device: str,
    batch_size: int = 16,
) -> npt.NDArray[np.float64]:
    model.eval()
    batches: list[npt.NDArray[np.float64]] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(
                batch,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            hidden = model.base_model(**inputs).last_hidden_state
            batches.append(hidden[:, 0, :].cpu().numpy().astype(np.float64))
    return np.vstack(batches)
```

`save_stats`/`load_stats` require `path` to already end in `.npz` — the caller (Task 3) is responsible for that.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/inference/test_ood.py -v`
Expected: PASS (all 7 tests)

- [ ] **Step 6: Run full check**

Run: `uv run poe check`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/inference/ood.py src/schema.py tests/inference/test_ood.py
git commit -m "feat: add Mahalanobis/cosine OOD scoring math module"
```

---

## Task 2: Settings + `PredictResult` fields

**Files:**
- Modify: `src/settings.py`
- Modify: `src/schema.py`
- Test: `tests/training/test_model_registry.py` pattern not applicable here — this task has no new testable logic (pure config/schema additions), verified via the existing test suite still passing plus a small schema default-value test.

**Interfaces:**
- Consumes: nothing new
- Produces: `Settings.OOD_PCA_COMPONENTS: int` (default `64`), `Settings.OOD_MAHALANOBIS_WEIGHT: float` (default `0.7`), `Settings.OOD_THRESHOLD: float` (default `2.5`)
- Produces: `PredictResult.ood_score: float | None = None`, `PredictResult.in_distribution: bool | None = None`

- [ ] **Step 1: Write the failing test**

Add to `tests/training/test_model_registry.py`'s neighbor — create `tests/test_settings_ood.py`:

```python
from src.schema import PredictResult
from src.settings import Settings


def test_ood_settings_have_expected_defaults() -> None:
    assert Settings.OOD_PCA_COMPONENTS == 64
    assert Settings.OOD_MAHALANOBIS_WEIGHT == 0.7
    assert Settings.OOD_THRESHOLD == 2.5


def test_predict_result_ood_fields_default_to_none() -> None:
    result = PredictResult(label="decreto", confidence=0.9, certain=True)
    assert result.ood_score is None
    assert result.in_distribution is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_settings_ood.py -v`
Expected: FAIL with `AttributeError: OOD_PCA_COMPONENTS`

- [ ] **Step 3: Add settings to `src/settings.py`**

Add these three lines after `PREDICT_CONFIDENCE: float = 0.0` (line 63):

```python
    OOD_PCA_COMPONENTS: int = 64
    OOD_MAHALANOBIS_WEIGHT: float = 0.7
    OOD_THRESHOLD: float = 2.5
```

- [ ] **Step 4: Add fields to `PredictResult` in `src/schema.py`**

Modify the `PredictResult` class to add two fields after `error: str = ""`:

```python
class PredictResult(BaseModel):
    """Return value from BertTunningClassifier.predict_text and predict_pdf."""

    model_config = ConfigDict(alias_generator=to_camel, arbitrary_types_allowed=True, frozen=True)

    label: str | None = None
    confidence: float = 0.0
    certain: bool = False
    all_scores: dict[str, float] = {}
    filename: str = ""
    error: str = ""
    ood_score: float | None = None
    in_distribution: bool | None = None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_settings_ood.py -v`
Expected: PASS

- [ ] **Step 6: Run full check**

Run: `uv run poe check`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/settings.py src/schema.py tests/test_settings_ood.py
git commit -m "feat: add OOD settings and PredictResult fields"
```

---

## Task 3: Generate + persist `ood_stats.npz` during training

**Files:**
- Modify: `src/training/pipeline.py`
- Test: no new test file — `training/pipeline.py`'s `run()` has no existing unit test (it requires a real transformers training loop), consistent with current test coverage. This task is verified manually per Step 3 below, not via pytest.

**Interfaces:**
- Consumes: `extract_embeddings`, `compute_class_stats`, `save_stats` from `src/inference/ood.py` (Task 1); `Settings.OOD_PCA_COMPONENTS` (Task 2)
- Produces: `{output_dir}/final/ood_stats.npz` written as a side effect of `run()`

- [ ] **Step 1: Modify `src/training/pipeline.py`**

Add the import:

```python
from src.inference.ood import compute_class_stats, extract_embeddings, save_stats
```

Modify the block after `trainer.train()` (currently at lines 150-160) from:

```python
    Path(request.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.train()
    log.info("Training complete")

    result = run_evaluation(trainer, test_ds, le, hyperparams)
    wb.log_results(result, list(le.classes_))
    wb.finish()

    save_path = Path(request.output_dir) / "final"
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    log.info("Model saved to %s", save_path)

    return trainer, le
```

to:

```python
    Path(request.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.train()
    log.info("Training complete")

    train_embeddings = extract_embeddings(
        model,
        tokenizer,
        _texts(train_df, request.chunk_strategy),
        max_length=model_cfg.max_tokens,
        device=str(model.device),
    )
    ood_stats = compute_class_stats(
        train_embeddings,
        train_df["label_id"].tolist(),
        list(le.classes_),
        n_components=Settings.OOD_PCA_COMPONENTS,
    )
    log.info("Computed OOD stats from %d training embeddings", train_embeddings.shape[0])

    result = run_evaluation(trainer, test_ds, le, hyperparams)
    wb.log_results(result, list(le.classes_))
    wb.finish()

    save_path = Path(request.output_dir) / "final"
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    save_stats(ood_stats, save_path / "ood_stats.npz")
    log.info("Model saved to %s", save_path)

    return trainer, le
```

Add `Settings` to the existing import from `src.settings` — check the top of the file; if `Settings` isn't already imported there, add:

```python
from src.settings import Settings
```

- [ ] **Step 2: Run the full test suite to confirm nothing broke**

Run: `uv run poe check`
Expected: PASS (this task adds no new unit tests, so this just confirms the modified file still type-checks and lints correctly, and no existing test regresses)

- [ ] **Step 3: Manual verification (requires a GPU and real training data)**

This step cannot be automated in CI without real PDF documents and a GPU, so verify manually:

```powershell
uv run python main.py train --docs-root "C:\path\to\downloads" --model beto --max-docs-per-class 20
```

Expected: training completes as before, and a new file appears:

```powershell
Test-Path .\models\bert_tunning_model_beto_v2\final\ood_stats.npz
# should print True
```

- [ ] **Step 4: Commit**

```bash
git add src/training/pipeline.py
git commit -m "feat: generate and persist OOD stats after training"
```

---

## Task 3b: Standalone `compute-ood-stats` command (backfill existing models)

This task exists because 5 models are already trained and saved without `ood_stats.npz` — retraining them just to get this artifact would be wasteful and could change their results (different run, different resource state). This command reuses Task 1's functions verbatim against an already-trained checkpoint and its original training cache, with no training loop involved — only forward passes through a model that's already trained.

**Precondition confirmed by the user:** all 5 existing training runs (xlm-roberta v1/v2, beto v1/v2, minilm v1) used the same `SEED` (`Settings.SEED`, default `42`), and `--seed` has never been an exposed CLI flag on `train`, so `make_split(df, seed=Settings.SEED)` reproduces the exact same train/val/test split each of those runs actually used, given the same cache file.

**Files:**
- Create: `src/cli/ood_stats.py`
- Modify: `main.py`
- Test: `tests/cli/test_ood_stats.py` (new)

**Interfaces:**
- Consumes: `extract_embeddings`, `compute_class_stats`, `save_stats` from `src/inference/ood.py` (Task 1); `get_model_config` from `src/training/models`; `make_split` from `src/training/split`; `prepare_text` from `src/training/tokenize`
- Produces: a `compute-ood-stats` CLI command that writes `{model_path}/ood_stats.npz`

- [ ] **Step 1: Write the failing test**

Create `tests/cli/test_ood_stats.py`:

```python
from click.testing import CliRunner

from src.cli.ood_stats import compute_ood_stats_cmd


def test_compute_ood_stats_cmd_help() -> None:
    result = CliRunner().invoke(compute_ood_stats_cmd, ["--help"])
    assert result.exit_code == 0
    assert "ood_stats" in result.output.lower() or "retraining" in result.output.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_ood_stats.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.cli.ood_stats'`

- [ ] **Step 3: Write `src/cli/ood_stats.py`**

```python
import logging
from pathlib import Path

import click
import pandas as pd
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.inference.ood import compute_class_stats, extract_embeddings, save_stats
from src.logger import setup_logging
from src.settings import Settings
from src.training.models import get_model_config
from src.training.split import make_split
from src.training.tokenize import prepare_text

log = logging.getLogger(__name__)


class ComputeOodStatsOptions(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        arbitrary_types_allowed=True,
        frozen=True,
        populate_by_name=True,
    )

    model_path: str
    model_key: str
    cache_path: str
    chunk_strategy: str = Settings.CHUNK_STRATEGY
    seed: int = Settings.SEED
    debug: bool = False


def _run_compute_ood_stats(opts: ComputeOodStatsOptions) -> None:
    log_file = setup_logging(level=logging.DEBUG if opts.debug else logging.INFO)
    log.info("Logging to %s", log_file)

    model_cfg = get_model_config(opts.model_key)
    df = pd.read_parquet(opts.cache_path)

    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])
    log.info("%d classes: %s", len(le.classes_), list(le.classes_))

    train_df, _val_df, _test_df = make_split(df, seed=opts.seed)
    log.info("Reconstructed train split: %d docs", len(train_df))

    tokenizer = AutoTokenizer.from_pretrained(opts.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(opts.model_path)
    model.eval()

    texts = [prepare_text(t, tokenizer, opts.chunk_strategy) for t in train_df["text"]]
    embeddings = extract_embeddings(
        model, tokenizer, texts, max_length=model_cfg.max_tokens, device=str(model.device)
    )
    stats = compute_class_stats(
        embeddings,
        train_df["label_id"].tolist(),
        list(le.classes_),
        n_components=Settings.OOD_PCA_COMPONENTS,
    )

    out_path = Path(opts.model_path) / "ood_stats.npz"
    save_stats(stats, out_path)
    log.info("Saved OOD stats → %s", out_path)


@click.command("compute-ood-stats")
@click.option("--model-path", required=True, help="Path to an already-trained model directory")
@click.option(
    "--model",
    "model_key",
    required=True,
    help="Model registry key used for that model (e.g. beto, xlm-roberta, minilm)",
)
@click.option(
    "--cache-path", required=True, help="Path to the exact parquet cache used to train that model"
)
@click.option("--chunk-strategy", default=Settings.CHUNK_STRATEGY, show_default=True)
@click.option(
    "--seed",
    default=Settings.SEED,
    show_default=True,
    help="Must match the seed used for the original training run, or the reconstructed train split will differ",
)
@click.option("--debug", is_flag=True, default=False)
def compute_ood_stats_cmd(**kwargs: str | int | bool) -> None:
    """Backfill ood_stats.npz for an already-trained model, without retraining it."""
    _run_compute_ood_stats(ComputeOodStatsOptions.model_validate(kwargs))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cli/test_ood_stats.py -v`
Expected: PASS

- [ ] **Step 5: Register the command in `main.py`**

Add the import:

```python
from src.cli.ood_stats import compute_ood_stats_cmd
```

Add the registration line after `cli.add_command(clean_cmd, name="clean")`:

```python
cli.add_command(compute_ood_stats_cmd, name="compute-ood-stats")
```

- [ ] **Step 6: Run full check**

Run: `uv run poe check`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/cli/ood_stats.py main.py tests/cli/test_ood_stats.py
git commit -m "feat: add compute-ood-stats command to backfill existing models"
```

- [ ] **Step 8: Manual verification against your 5 existing models**

Run once per already-trained model, using the cache/model-key mapping below (confirmed from the training results table in `README.md`):

```powershell
# xlm-roberta v1
uv run python main.py compute-ood-stats --model-path ./models/<xlm_roberta_v1_output_dir>/final --model xlm-roberta --cache-path ./data/bert_tunning_cache_100.parquet

# xlm-roberta v2
uv run python main.py compute-ood-stats --model-path ./models/<xlm_roberta_v2_output_dir>/final --model xlm-roberta --cache-path ./data/bert_tunning_cache_300.parquet

# beto v1
uv run python main.py compute-ood-stats --model-path ./models/bert_tunning_model_beto_v2/final --model beto --cache-path ./data/bert_tunning_cache_300.parquet

# beto v2 (9-class run, includes otro)
uv run python main.py compute-ood-stats --model-path ./models/<beto_v2_output_dir>/final --model beto --cache-path ./data/bert_tunning_cache_con_otro_300.parquet

# minilm v1
uv run python main.py compute-ood-stats --model-path ./models/<minilm_v1_output_dir>/final --model minilm --cache-path ./data/bert_tunning_cache_300.parquet
```

Replace `<..._output_dir>` with the actual `OUTPUT_DIR` used for each run (check `logs/` or your own records — these aren't tracked in this repo since `models/` is git-ignored). Expected after each run: `ood_stats.npz` appears in that model's `final/` directory.

---

## Task 4: Load stats + score at inference time

**Files:**
- Modify: `src/inference/classify.py`
- Test: `tests/inference/test_pipeline.py` (existing file, despite testing `classify.py` — matches the codebase's current, if misleadingly named, convention)

**Interfaces:**
- Consumes: `ClassEmbeddingStats`, `load_stats`, `ood_score` from Task 1; `Settings.OOD_MAHALANOBIS_WEIGHT`, `Settings.OOD_THRESHOLD` from Task 2
- Produces: `BertTunningClassifier.predict_text` populates `ood_score`/`in_distribution` on the returned `PredictResult` when stats are available; both stay `None` when no `ood_stats.npz` exists next to `model_path` (backward compatible with the 5 already-trained models, none of which have this artifact yet)

- [ ] **Step 1: Write the failing tests**

Add to `tests/inference/test_pipeline.py`, after the existing `_make_mock_classifier` helper and its two tests:

```python
from src.inference.ood import ClassEmbeddingStats


def _make_stats() -> ClassEmbeddingStats:
    return ClassEmbeddingStats(
        class_names=["decreto", "ordenanza"],
        pca_mean=np.zeros(8),
        pca_components=np.eye(8),
        centroids=np.array([[0.0] * 8, [5.0] * 8]),
        covariance_inv=np.eye(8),
        maha_calibration_mean=0.0,
        maha_calibration_std=1.0,
        cosine_calibration_mean=0.0,
        cosine_calibration_std=1.0,
    )


def test_predict_text_without_stats_leaves_ood_fields_none() -> None:
    clf = _make_mock_classifier()
    clf._ood_stats = None
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        clf.model.return_value.hidden_states = [torch.zeros(1, 512, 8)]
        result = clf.predict_text("anything")
    assert result.ood_score is None
    assert result.in_distribution is None


def test_predict_text_with_stats_populates_ood_fields() -> None:
    clf = _make_mock_classifier()
    clf._ood_stats = _make_stats()
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        clf.model.return_value.hidden_states = [torch.zeros(1, 512, 8)]
        result = clf.predict_text("anything")
    assert isinstance(result.ood_score, float)
    assert isinstance(result.in_distribution, bool)
```

Add `import numpy as np` to the top of `tests/inference/test_pipeline.py` (only `torch` is currently imported there).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/inference/test_pipeline.py -v`
Expected: FAIL — `AttributeError: 'BertTunningClassifier' object has no attribute '_ood_stats'`

- [ ] **Step 3: Modify `src/inference/classify.py`**

Replace the full file contents with:

```python
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.ingestion.extract import clean_text
from src.inference.ood import ClassEmbeddingStats, load_stats, ood_score
from src.schema import PredictResult
from src.settings import Settings

log = logging.getLogger(__name__)


class BertTunningClassifier:
    def __init__(self, model_path: str, *, confidence_threshold: float = 0.70) -> None:
        log.info("Loading classifier from %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.threshold = confidence_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        self.model.to(self.device)
        self.max_length = min(
            self.tokenizer.model_max_length,
            self.model.config.max_position_embeddings,
        )
        self._ood_stats = self._load_ood_stats(model_path)
        log.info("Classifier ready on %s (max_length=%d)", self.device, self.max_length)

    @staticmethod
    def _load_ood_stats(model_path: str) -> ClassEmbeddingStats | None:
        stats_path = Path(model_path) / "ood_stats.npz"
        if not stats_path.exists():
            log.info("No ood_stats.npz found at %s — OOD scoring disabled", stats_path)
            return None
        log.info("Loaded OOD stats from %s", stats_path)
        return load_stats(stats_path)

    def predict_text(self, text: str) -> PredictResult:
        inputs = self.tokenizer(
            clean_text(text),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
            cls_embedding = (
                outputs.hidden_states[-1][:, 0, :][0].cpu().numpy().astype(np.float64)
            )

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        label = self.model.config.id2label[pred_idx]

        result = PredictResult(
            label=label,
            confidence=round(confidence, 4),
            certain=confidence >= self.threshold,
            all_scores={
                self.model.config.id2label[i]: round(float(p), 4) for i, p in enumerate(probs)
            },
        )

        if self._ood_stats is None:
            return result

        score = ood_score(
            cls_embedding, self._ood_stats, mahalanobis_weight=Settings.OOD_MAHALANOBIS_WEIGHT
        )
        return result.model_copy(
            update={
                "ood_score": round(score, 4),
                "in_distribution": score <= Settings.OOD_THRESHOLD,
            }
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/inference/test_pipeline.py -v`
Expected: PASS (all tests, including the 2 new ones)

- [ ] **Step 5: Run full check**

Run: `uv run poe check`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/inference/classify.py tests/inference/test_pipeline.py
git commit -m "feat: score OOD at inference time via Mahalanobis/cosine mixture"
```

---

## Task 5: Surface OOD fields through CLI and API

**Files:**
- Modify: `src/cli/predict.py`
- Modify: `src/api/routes/predict/schemas.py`
- Modify: `src/api/routes/predict/endpoints.py`
- Test: `tests/cli/test_commands.py`, `tests/api/test_predict.py` (existing files)

**Interfaces:**
- Consumes: `PredictResult.ood_score`/`in_distribution` (Task 2/4)
- Produces: `predict_cmd` prints `OOD Score` / `In Distribution` lines; `PredictResponse.ood_score: float | None`, `PredictResponse.in_distribution: bool | None`

- [ ] **Step 1: Write the failing test**

Add to `tests/api/test_predict.py`:

```python
def test_predict_response_has_ood_fields() -> None:
    from src.api.routes.predict.schemas import PredictResponse

    response = PredictResponse(
        filename="doc.pdf", label="decreto", confidence=0.9, certain=True
    )
    assert response.ood_score is None
    assert response.in_distribution is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/api/test_predict.py -v`
Expected: FAIL — `AttributeError: 'PredictResponse' object has no attribute 'ood_score'`

- [ ] **Step 3: Modify `src/api/routes/predict/schemas.py`**

```python
from pydantic import Field

from src.api.schema import BaseSchema


class PredictResponse(BaseSchema):
    filename: str
    label: str | None
    confidence: float
    certain: bool
    all_scores: dict[str, float] = Field(default_factory=dict)
    error: str | None = None
    ood_score: float | None = None
    in_distribution: bool | None = None
```

- [ ] **Step 4: Modify `src/api/routes/predict/endpoints.py`**

Change the final `return PredictResponse(...)` block (currently lines 51-58) from:

```python
    return PredictResponse(
        filename=data["filename"],
        label=data["label"],
        confidence=data["confidence"],
        certain=data["certain"],
        all_scores=data["all_scores"],
        error=data["error"] or None,
    )
```

to:

```python
    return PredictResponse(
        filename=data["filename"],
        label=data["label"],
        confidence=data["confidence"],
        certain=data["certain"],
        all_scores=data["all_scores"],
        error=data["error"] or None,
        ood_score=data["ood_score"],
        in_distribution=data["in_distribution"],
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/api/test_predict.py -v`
Expected: PASS

- [ ] **Step 6: Modify `src/cli/predict.py`**

In `predict_cmd`, change the output block (currently lines 46-54) from:

```python
    click.echo(f"\n{'─' * 50}")
    click.echo(f"  File      : {result.filename or pdf_path}")
    click.echo(f"  Label     : {result.label}")
    click.echo(f"  Confidence: {result.confidence:.2%}")
    click.echo(f"  Certain   : {result.certain}")
    click.echo("\n  All scores:")
```

to:

```python
    click.echo(f"\n{'─' * 50}")
    click.echo(f"  File      : {result.filename or pdf_path}")
    click.echo(f"  Label     : {result.label}")
    click.echo(f"  Confidence: {result.confidence:.2%}")
    click.echo(f"  Certain   : {result.certain}")
    if result.ood_score is not None:
        click.echo(f"  OOD Score : {result.ood_score:.4f}")
        click.echo(f"  In-Dist.  : {result.in_distribution}")
    click.echo("\n  All scores:")
```

- [ ] **Step 7: Run the CLI test suite**

Run: `uv run pytest tests/cli/test_commands.py -v`
Expected: PASS (no behavior change to `--help` output, which is all this suite currently checks)

- [ ] **Step 8: Run full check**

Run: `uv run poe check`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/cli/predict.py src/api/routes/predict/schemas.py src/api/routes/predict/endpoints.py tests/api/test_predict.py
git commit -m "feat: surface OOD score and in-distribution flag in CLI and API output"
```

---

## Task 6: Documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing (documentation only)
- Produces: nothing new — describes what Tasks 1-5 built

- [ ] **Step 1: Update `CLAUDE.md`**

Add a new subsection under "Key Technical Decisions" (after the "OCR thread safety via double-checked locking" entry):

```markdown
**Mahalanobis/cosine OOD detection alongside the softmax classifier**
Softmax classifiers always output a label — there is no "I don't know."
`ood_stats.npz` (generated at training time from the training set's `[CLS]`
embeddings, PCA-reduced) stores per-class centroids and a shared covariance
matrix. At inference, `BertTunningClassifier.predict_text` computes a mixture
of Mahalanobis distance and cosine distance to the nearest centroid and
attaches `ood_score`/`in_distribution` to `PredictResult`. This is separate
from `certain` (softmax-confidence-based) — a document can be `certain=True`
(the softmax is confident) and `in_distribution=False` (the document doesn't
resemble anything the model was trained on) at the same time, which is
exactly the payment-document failure mode this was built to catch.
`OOD_THRESHOLD` (default `2.5`, a z-score-like cutoff) has **not** been
statistically validated against a labeled out-of-category corpus — there
is no such corpus for this project. Treat it as a starting point requiring
manual calibration against real novel documents as they're encountered.
```

Add to the Settings table:

```markdown
| `OOD_PCA_COMPONENTS` | `64` | Dimensionality the `[CLS]` embedding is reduced to before Mahalanobis/cosine scoring |
| `OOD_MAHALANOBIS_WEIGHT` | `0.7` | Weight given to the Mahalanobis z-score vs. cosine z-score in the mixture (`1 - this` goes to cosine) |
| `OOD_THRESHOLD` | `2.5` | Mixture score above which a document is flagged `in_distribution=False` — uncalibrated, see note above |
```

- [ ] **Step 2: Update `README.md`**

Add a new subsection after "### Classify":

```markdown
### Out-of-distribution detection

If the loaded model directory contains `ood_stats.npz` (generated automatically
during `train`), predictions include two extra fields:

```json
{
  "label": "boletines",
  "confidence": 0.9429,
  "oodScore": 4.1,
  "inDistribution": false
}
```

`inDistribution: false` means the document doesn't resemble anything in the
training set, even though the classifier still picked its best guess for
`label`/`confidence`. Treat `inDistribution: false` as "do not trust `label`
for this document" regardless of how high `confidence` is — this is the
mechanism that catches documents (e.g. payment receipts) that were never in
any training class, including `otro`.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: document Mahalanobis/cosine OOD detection"
```

---

## Self-Review

**Spec coverage:**
- Mahalanobis distance instead of pure cosine → Task 1 (`mahalanobis_min_distance`)
- "Or maybe a mixture" → Task 1 (`ood_score` combines both via weighted z-score)
- Approve per PR each of those changes → 7 independently mergeable tasks (1, 2, 3, 3b, 4, 5, 6), each ending in a commit; each maps 1:1 to a worktree + PR in this project's existing git workflow
- Never knowing the true universe of out-of-category documents → the design doesn't try to enumerate negative classes; it flags "far from everything known," which generalizes to unseen document types by construction, unlike expanding the `otro` class
- No retraining of existing 5 models → confirmed, Tasks 1/2 add no training-time behavior; Task 3 only affects *future* training runs; Tasks 4/5 gracefully no-op (`None` fields) for model directories without `ood_stats.npz`
- What about the 4 other already-trained models (not just BETO v1) → Task 3b backfills `ood_stats.npz` for all 5 existing checkpoints (xlm-roberta v1/v2, beto v1/v2, minilm v1) by reconstructing each run's exact train split via the confirmed-constant `SEED`, with no retraining

**Placeholder scan:** no TBD/TODO, no "add error handling" placeholders — all code blocks are complete and runnable as written. Task 3b's manual verification step has `<..._output_dir>` placeholders, but those are explicitly called out as values the user must substitute from their own run records (not tracked in git since `models/` is git-ignored) — not a plan placeholder.

**Type consistency:** `ClassEmbeddingStats`, `compute_class_stats`, `mahalanobis_min_distance`, `cosine_min_distance`, `ood_score`, `save_stats`, `load_stats`, `extract_embeddings` are defined once in Task 1 and referenced with identical names/signatures in Tasks 3, 3b, and 4. `PredictResult.ood_score`/`in_distribution` defined in Task 2, consumed identically in Task 4 (classify.py) and Task 5 (CLI/API). Task 3b's `_run_compute_ood_stats` reconstructs the exact same `le.fit_transform` / `make_split` / `prepare_text` sequence that `training/pipeline.py`'s `run()` uses, so the two code paths produce consistent `label_id` ordering and text preprocessing.
