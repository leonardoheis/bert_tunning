# Classiflow Scaffold Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the flat-file Classiflow project into a `src/` pipeline layout with a Click CLI, FastAPI inference endpoint, and a model registry supporting multiple transformer models.

**Architecture:** Each domain (ingestion, training, inference) becomes an isolated pipeline module under `src/` with its own steps and a thin `pipeline.py` orchestrator. Both the Click CLI and FastAPI routes call the same pipeline functions — no logic lives in the interface layer. A model registry in `src/training/models/` makes adding new models a one-file change.

**Tech Stack:** Python 3.10+, PyTorch, HuggingFace Transformers, Click, FastAPI, Uvicorn, python-multipart, uv

## Global Constraints

- Python `>=3.10` — use `X | Y` union types, not `Optional[X]`
- `package = false` in `[tool.uv]` — this is not an installable package, imports use `src.` prefix
- `pythonpath = ["."]` in pytest — tests import as `from src.ingestion.extract import ...`
- All trained model weights go to `models/` (root, git-ignored)
- All parquet caches go to `data/` (root, git-ignored)
- All HTML reports go to `reports/` (root, git-ignored)
- `ruff`, `mypy`, `pytest` must all pass after each task
- Each task ends with a `git commit` and a PR targeting **`feature/scaffold-migration`** (not `master`)
- Worktrees branch off `feature/scaffold-migration`: `git worktree add -b task/N-name ../bert_tunning-taskN feature/scaffold-migration`
- Final PR merges `feature/scaffold-migration` → `master` after all tasks are reviewed

---

## File Map

```
src/
├── __init__.py
├── ingestion/
│   ├── __init__.py
│   ├── extract.py          ← PDF→text, OCR, clean_text  (was: extraction.py)
│   ├── scan.py             ← folder scan, label mapping   (was: dataset.py:build_dataset)
│   ├── cache.py            ← parquet load/save            (was: dataset.py:load_or_build_dataset)
│   └── pipeline.py         ← thin orchestrator: scan→extract→cache→DataFrame
│
├── training/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py     ← ModelConfig dataclass + MODEL_REGISTRY dict
│   │   ├── xlm_roberta.py  ← XLM-R config entry
│   │   └── beto.py         ← BETO config entry
│   ├── split.py            ← stratified train/val/test split
│   ├── tokenize.py         ← ClassiflowDataset, prepare_text
│   ├── trainer.py          ← WeightedTrainer, compute_metrics
│   ├── evaluate.py         ← classification_report, confusion_matrix, HTML report
│   └── pipeline.py         ← orchestrates split→tokenize→train→evaluate
│
├── inference/
│   ├── __init__.py
│   ├── classify.py         ← ClassiflowClassifier._infer(), predict_text()
│   └── pipeline.py         ← predict_pdf(), predict_folder()
│
├── api/
│   ├── __init__.py
│   ├── app.py              ← FastAPI app factory (create_app)
│   └── routes/
│       ├── __init__.py
│       └── predict.py      ← POST /predict (single PDF upload)
│
└── cli/
    ├── __init__.py
    ├── train.py            ← @click.command "train"
    ├── predict.py          ← @click.command "predict" + "predict-folder"
    └── clean.py            ← @click.command "clean"

config.py                   ← constants + model registry key (stays at root)
wandb_logger.py             ← stays at root (shared)
reporting.py                ← stays at root (shared)
logger.py                   ← stays at root (shared)
main.py                     ← Click group wiring (rewritten)

models/                     ← git-ignored (trained weights)
data/                       ← git-ignored (parquet cache)
reports/                    ← git-ignored (HTML reports)
docs/                       ← plans, ADRs
tests/
├── ingestion/
│   └── test_cache.py
├── training/
│   ├── test_model_registry.py
│   ├── test_split.py
│   └── test_tokenize.py
├── inference/
│   └── test_pipeline.py
├── cli/
│   └── test_commands.py
└── api/
    └── test_predict.py
```

**Files to delete after migration (Task 9):**
- `extraction.py`
- `dataset.py`
- `training.py`
- `classifier.py`

---

## Task 1: Project scaffold + dependency updates

**Files:**
- Create: `src/__init__.py`
- Create: `src/ingestion/__init__.py`
- Create: `src/training/__init__.py`
- Create: `src/training/models/__init__.py`
- Create: `src/inference/__init__.py`
- Create: `src/api/__init__.py`
- Create: `src/api/routes/__init__.py`
- Create: `src/cli/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/ingestion/__init__.py`
- Create: `tests/training/__init__.py`
- Create: `tests/inference/__init__.py`
- Create: `tests/cli/__init__.py`
- Create: `tests/api/__init__.py`
- Modify: `pyproject.toml` — add click, fastapi, uvicorn, python-multipart
- Modify: `pyproject.toml` — update testpaths to `tests/`

**Interfaces:**
- Produces: `src/` package tree importable as `from src.ingestion.extract import ...`

- [ ] **Step 1: Create all `__init__.py` files**

```powershell
$dirs = @(
  "src", "src/ingestion", "src/training", "src/training/models",
  "src/inference", "src/api", "src/api/routes", "src/cli",
  "tests", "tests/ingestion", "tests/training", "tests/inference",
  "tests/cli", "tests/api"
)
foreach ($d in $dirs) {
  New-Item -ItemType Directory -Force $d | Out-Null
  New-Item -ItemType File -Force "$d/__init__.py" | Out-Null
}
```

- [ ] **Step 2: Add new dependencies to `pyproject.toml`**

In the `dependencies` list, add:
```toml
"click>=8.1.0",
"fastapi>=0.111.0",
"uvicorn[standard]>=0.29.0",
"python-multipart>=0.0.9",
```

In `[dependency-groups] dev`, add:
```toml
"httpx>=0.27.0",
```

Update `[tool.pytest.ini_options]`:
```toml
[tool.pytest.ini_options]
addopts = ["-r=A", "-vvv"]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 3: Run `uv sync` to verify deps resolve**

```powershell
uv sync
```
Expected: resolves without error.

- [ ] **Step 4: Verify imports work**

```powershell
uv run python -c "from src.ingestion import __init__; print('OK')"
```
Expected: prints `OK`.

- [ ] **Step 5: Run `uv run poe check`**

```powershell
uv run poe check
```
Expected: lint passes (empty `__init__.py` files have nothing to lint), mypy passes, 6 scaffold tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/ tests/ pyproject.toml
git commit -m "chore: scaffold src/ package layout and test tree"
```

---

## Task 2: Migrate ingestion pipeline

**Files:**
- Create: `src/ingestion/extract.py`
- Create: `src/ingestion/scan.py`
- Create: `src/ingestion/cache.py`
- Create: `src/ingestion/pipeline.py`
- Create: `tests/ingestion/test_cache.py`

**Interfaces:**
- Consumes: `config.EXCLUDE_LABELS`, `config.FOLDER_TO_LABEL`, `config.CACHE_PATH`
- Produces:
  - `extract.extract_pdf(pdf_path: str, *, use_ocr_fallback: bool = True) -> str | None`
  - `extract.clean_text(text: str) -> str`
  - `scan.build_dataset(docs_root: str, *, use_ocr: bool = True, max_docs_per_class: int | None = None) -> pd.DataFrame`
  - `cache.load_or_build(docs_root: str, *, cache_path: str, use_ocr: bool, rebuild: bool, max_docs_per_class: int | None) -> pd.DataFrame`
  - `pipeline.run(docs_root: str, *, cache_path: str, use_ocr: bool, rebuild: bool, max_docs_per_class: int | None) -> pd.DataFrame`

- [ ] **Step 1: Create `src/ingestion/extract.py`**

Move the full content of root `extraction.py` here, then fix the import — change `from config import ...` to keep it as `from config import ...` (config stays at root, `pythonpath=["."]` covers it). Also make `use_ocr_fallback` keyword-only:

```python
import logging
import re
from pathlib import Path
from typing import Optional

import fitz
import easyocr
import numpy as np
import torch
from markitdown import MarkItDown

log = logging.getLogger(__name__)

_md = MarkItDown()
_ocr_reader: Optional[easyocr.Reader] = None

_MIN_TEXT_FOR_OCR = 50   # chars below which we assume scanned PDF
_MIN_USABLE_TEXT  = 20   # chars below which we discard entirely


def _get_ocr_reader() -> easyocr.Reader:
    global _ocr_reader  # noqa: PLW0603
    if _ocr_reader is None:
        log.info("Initializing EasyOCR reader (first use — may take ~10s)")
        _ocr_reader = easyocr.Reader(["es"], gpu=torch.cuda.is_available())
        log.info("EasyOCR reader ready")
    return _ocr_reader


def _ocr_fallback(pdf_path: str) -> str:
    try:
        reader = _get_ocr_reader()
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            results = reader.readtext(img, detail=0, paragraph=True)
            text += " ".join(results) + "\n"
        doc.close()
        return text.strip()
    except Exception:
        log.exception("OCR error on %s", Path(pdf_path).name)
        return ""


def clean_text(text: str) -> str:
    text = text.replace("\f", " ").replace("\xa0", " ")
    text = re.sub(r'\|[-: ]+\|[-: |]+', '', text)
    text = re.sub(r'^\|.*\|$', '', text, flags=re.M)
    text = re.sub(r'#+ ', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def extract_pdf(pdf_path: str, *, use_ocr_fallback: bool = True) -> str | None:
    try:
        result = _md.convert(pdf_path)
        text = clean_text(result.text_content or "")
    except Exception:
        log.warning("MarkItDown failed on %s — trying OCR", Path(pdf_path).name)
        text = ""

    if len(text) < _MIN_TEXT_FOR_OCR and use_ocr_fallback:
        log.info("Scanned PDF detected, running OCR: %s", Path(pdf_path).name)
        text = clean_text(_ocr_fallback(pdf_path))

    if len(text) < _MIN_USABLE_TEXT:
        log.warning("Skipping %s — could not extract usable text", Path(pdf_path).name)
        return None

    return text
```

- [ ] **Step 2: Create `src/ingestion/scan.py`**

```python
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from config import EXCLUDE_LABELS, FOLDER_TO_LABEL
from src.ingestion.extract import extract_pdf

log = logging.getLogger(__name__)


def build_dataset(
    docs_root: str,
    *,
    use_ocr: bool = True,
    max_docs_per_class: Optional[int] = None,
) -> pd.DataFrame:
    root = Path(docs_root)
    records: list[dict] = []
    skipped = 0

    if max_docs_per_class:
        log.info("Sample mode: capped at %d docs per class", max_docs_per_class)

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        folder_name = folder.name.lower()
        if folder_name in EXCLUDE_LABELS:
            log.info("Skipping excluded folder: %s", folder_name)
            continue

        label = FOLDER_TO_LABEL.get(folder_name, folder_name)
        pdfs = list(folder.glob("*.pdf"))
        if max_docs_per_class:
            pdfs = pdfs[:max_docs_per_class]
        log.info("Processing '%s' → '%s' (%d PDFs)", folder_name, label, len(pdfs))

        for i, pdf_path in enumerate(pdfs, start=1):
            log.info("[%d/%d] %s", i, len(pdfs), pdf_path.name)
            text = extract_pdf(str(pdf_path), use_ocr_fallback=use_ocr)
            if text is None:
                skipped += 1
                continue
            records.append({"text": text, "label": label, "filename": pdf_path.name})
            log.info("  extracted %d chars", len(text))

    df = pd.DataFrame(records)
    log.info("Dataset built: %d docs, %d skipped", len(df), skipped)
    if len(df):
        log.info("Class distribution:\n%s", df["label"].value_counts().to_string())
    return df
```

- [ ] **Step 3: Create `src/ingestion/cache.py`**

```python
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.ingestion.scan import build_dataset

log = logging.getLogger(__name__)


def _resolve_path(base_path: str, max_docs_per_class: Optional[int]) -> Path:
    p = Path(base_path)
    if max_docs_per_class:
        return p.parent / f"{p.stem}_{max_docs_per_class}{p.suffix}"
    return p


def load_or_build(
    docs_root: str,
    *,
    cache_path: str,
    use_ocr: bool = True,
    rebuild: bool = False,
    max_docs_per_class: Optional[int] = None,
) -> pd.DataFrame:
    cache = _resolve_path(cache_path, max_docs_per_class)

    if not rebuild and cache.exists():
        log.info("Cache found — loading from %s", cache)
        df = pd.read_parquet(cache)
        log.info("Loaded %d docs from cache", len(df))
        return df

    if rebuild and cache.exists():
        log.info("rebuild=True — removing cache: %s", cache)
        cache.unlink()

    log.info("Extracting PDFs from: %s", docs_root)
    df = build_dataset(docs_root, use_ocr=use_ocr, max_docs_per_class=max_docs_per_class)

    if len(df) > 0:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
        log.info("Cached %d docs → %s (%.1f MB)", len(df), cache, cache.stat().st_size / 1_048_576)

    return df
```

- [ ] **Step 4: Create `src/ingestion/pipeline.py`**

```python
from typing import Optional

import pandas as pd

from src.ingestion.cache import load_or_build


def run(
    docs_root: str,
    *,
    cache_path: str,
    use_ocr: bool = True,
    rebuild: bool = False,
    max_docs_per_class: Optional[int] = None,
) -> pd.DataFrame:
    return load_or_build(
        docs_root,
        cache_path=cache_path,
        use_ocr=use_ocr,
        rebuild=rebuild,
        max_docs_per_class=max_docs_per_class,
    )
```

- [ ] **Step 5: Write and run tests for cache path resolution**

Create `tests/ingestion/test_cache.py`:

```python
from pathlib import Path

from src.ingestion.cache import _resolve_path


def test_resolve_path_no_cap():
    result = _resolve_path("./data/classiflow_cache.parquet", None)
    assert result == Path("./data/classiflow_cache.parquet")


def test_resolve_path_with_cap():
    result = _resolve_path("./data/classiflow_cache.parquet", 100)
    assert result == Path("./data/classiflow_cache_100.parquet")


def test_resolve_path_preserves_extension():
    result = _resolve_path("./data/cache.parquet", 50)
    assert result.suffix == ".parquet"
    assert "50" in result.name
```

```powershell
uv run pytest tests/ingestion/test_cache.py -v
```
Expected: 3 passed.

- [ ] **Step 6: Run `uv run poe check`**

```powershell
uv run poe check
```
Expected: lint passes for `src/ingestion/`, mypy passes for the new module, ingestion tests pass. Existing warnings in legacy flat files (`training.py`, `dataset.py`, etc.) are acceptable until Task 9.

- [ ] **Step 7: Commit**

```bash
git add src/ingestion/ tests/ingestion/
git commit -m "feat: migrate ingestion pipeline to src/ingestion"
```

---

## Task 3: Model registry

**Files:**
- Create: `src/training/models/__init__.py`
- Create: `src/training/models/xlm_roberta.py`
- Create: `src/training/models/beto.py`
- Create: `tests/training/test_model_registry.py`

**Interfaces:**
- Produces:
  - `ModelConfig` dataclass with fields: `name`, `hf_id`, `max_tokens`, `lr`, `batch_size`, `grad_accum`, `force_fp32`
  - `MODEL_REGISTRY: dict[str, ModelConfig]`
  - `get_model_config(key: str) -> ModelConfig` — raises `KeyError` with helpful message

- [ ] **Step 1: Create `src/training/models/xlm_roberta.py`**

```python
from src.training.models import ModelConfig

config = ModelConfig(
    name="xlm-roberta-base",
    hf_id="xlm-roberta-base",
    max_tokens=512,
    lr=2e-5,
    batch_size=8,
    grad_accum=8,
    force_fp32=False,
)
```

- [ ] **Step 2: Create `src/training/models/beto.py`**

```python
from src.training.models import ModelConfig

config = ModelConfig(
    name="beto",
    hf_id="dccuchile/bert-base-spanish-wwm-cased",
    max_tokens=512,
    lr=3e-5,
    batch_size=16,
    grad_accum=4,
    force_fp32=False,
)
```

- [ ] **Step 3: Create `src/training/models/__init__.py`**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hf_id: str
    max_tokens: int
    lr: float
    batch_size: int
    grad_accum: int
    force_fp32: bool


def _build_registry() -> dict[str, ModelConfig]:
    from src.training.models import xlm_roberta, beto  # noqa: PLC0415
    return {
        "xlm-roberta": xlm_roberta.config,
        "beto": beto.config,
    }


MODEL_REGISTRY: dict[str, ModelConfig] = _build_registry()


def get_model_config(key: str) -> ModelConfig:
    if key not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY)
        msg = f"Unknown model '{key}'. Available: {available}"
        raise KeyError(msg)
    return MODEL_REGISTRY[key]
```

- [ ] **Step 4: Update `config.py` — replace `MODEL_NAME` with `MODEL_KEY`**

In `config.py`, replace:
```python
MODEL_NAME = "xlm-roberta-base"
```
With:
```python
MODEL_KEY = "xlm-roberta"  # key into src.training.models.MODEL_REGISTRY
```

Remove `FORCE_FP32`, `BATCH_SIZE`, `GRAD_ACCUM`, `LR`, `MAX_TOKENS` — these now live in `ModelConfig`. Keep `EPOCHS`, `EARLY_STOP_PATIENCE`, `GRAD_ACCUM` (global training overrides), `CHUNK_STRATEGY`, `SEED`, `OUTPUT_DIR`, `CACHE_PATH`, `WANDB_*`, `DOCS_ROOT`, `EXCLUDE_LABELS`, `FOLDER_TO_LABEL`.

- [ ] **Step 5: Write and run registry tests**

Create `tests/training/test_model_registry.py`:

```python
import pytest

from src.training.models import MODEL_REGISTRY, get_model_config


def test_registry_has_xlm_roberta():
    assert "xlm-roberta" in MODEL_REGISTRY


def test_registry_has_beto():
    assert "beto" in MODEL_REGISTRY


def test_get_model_config_returns_correct_hf_id():
    cfg = get_model_config("xlm-roberta")
    assert cfg.hf_id == "xlm-roberta-base"


def test_get_model_config_raises_on_unknown():
    with pytest.raises(KeyError, match="Unknown model"):
        get_model_config("nonexistent-model")


def test_model_config_is_immutable():
    cfg = get_model_config("xlm-roberta")
    with pytest.raises(Exception):  # frozen dataclass
        cfg.name = "other"  # type: ignore[misc]
```

```powershell
uv run pytest tests/training/test_model_registry.py -v
```
Expected: 5 passed.

- [ ] **Step 6: Run `uv run poe check`**

```powershell
uv run poe check
```
Expected: lint passes for `src/training/models/`, mypy validates the frozen dataclass and registry types, 5 registry tests pass. Existing warnings in legacy flat files are acceptable until Task 9.

- [ ] **Step 7: Commit**

```bash
git add src/training/models/ tests/training/test_model_registry.py config.py
git commit -m "feat: add model registry with XLM-R and BETO configs"
```

---

## Task 4: Migrate training pipeline

**Files:**
- Create: `src/training/split.py`
- Create: `src/training/tokenize.py`
- Create: `src/training/trainer.py`
- Create: `src/training/evaluate.py`
- Create: `src/training/pipeline.py`
- Create: `tests/training/test_split.py`
- Create: `tests/training/test_tokenize.py`

**Interfaces:**
- Consumes: `ModelConfig` from Task 3, `WandbLogger` from `wandb_logger.py`, `generate_html_report` from `reporting.py`
- Produces:
  - `split.make_split(df, *, seed) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`
  - `tokenize.ClassiflowDataset(texts, labels, tokenizer, max_length)`
  - `tokenize.prepare_text(text, tokenizer, strategy) -> str`
  - `trainer.WeightedTrainer` (Trainer subclass)
  - `trainer.compute_metrics(eval_pred: tuple) -> dict`
  - `evaluate.run_evaluation(trainer, test_ds, le) -> tuple[dict, np.ndarray, list[int]]`
  - `pipeline.run(df, model_cfg, *, epochs, early_stop_patience, chunk_strategy, seed, output_dir, use_wandb) -> tuple[Trainer, LabelEncoder]`

- [ ] **Step 1: Create `src/training/split.py`**

```python
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)


def make_split(
    df: pd.DataFrame,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        train_df, temp_df = train_test_split(
            df, test_size=0.30, stratify=df["label_id"], random_state=seed
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, stratify=temp_df["label_id"], random_state=seed
        )
    except ValueError as e:
        log.warning("Stratified split failed (%s) — using random split", e)
        train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed)
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed)

    log.info("Split — train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df
```

- [ ] **Step 2: Create `src/training/tokenize.py`**

```python
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer


def prepare_text(text: str, tokenizer: AutoTokenizer, strategy: str = "first") -> str:
    if strategy == "first":
        return text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if strategy == "middle":
        max_t = tokenizer.model_max_length
        if len(tokens) <= max_t - 2:
            return text
        half = (max_t - 2) // 2
        selected = tokens[:half] + tokens[-half:]
        return tokenizer.decode(selected, skip_special_tokens=True)
    return text


class ClassiflowDataset(TorchDataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }
```

- [ ] **Step 3: Create `src/training/trainer.py`**

```python
import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import Trainer


class WeightedTrainer(Trainer):
    """Trainer subclass that applies per-class weights to cross-entropy loss."""

    def __init__(
        self, *args: object, class_weights: torch.Tensor | None = None, **kwargs: object
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict,
        return_outputs: bool = False,  # noqa: FBT001, FBT002
        num_items_in_batch: int | None = None,
        **_kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, object]:
        labels = inputs.pop("labels", None)
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None and labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weights.to(device=logits.device, dtype=logits.dtype)
            )
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs.loss

        if num_items_in_batch is not None and self.args.gradient_accumulation_steps > 1:
            loss = loss * logits.shape[0] / num_items_in_batch

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred: tuple) -> dict:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "accuracy": float((predictions == labels).mean()),
    }
```

- [ ] **Step 4: Create `src/training/evaluate.py`**

```python
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from transformers import Trainer

from reporting import generate_html_report
from src.training.tokenize import ClassiflowDataset

log = logging.getLogger(__name__)


def run_evaluation(
    trainer: Trainer,
    test_ds: ClassiflowDataset,
    le: LabelEncoder,
    hyperparams: dict,
) -> tuple[dict, np.ndarray, list[int]]:
    preds_out = trainer.predict(test_ds)
    y_pred = np.argmax(preds_out.predictions, axis=-1)
    y_true: list[int] = preds_out.label_ids.tolist()

    report_str = classification_report(
        y_true, y_pred, target_names=le.classes_, digits=3, zero_division=0
    )
    report_dict = classification_report(
        y_true, y_pred, target_names=le.classes_, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

    log.info("Per-class report:\n%s", report_str)
    log.info("Confusion matrix:\n%s", cm_df.to_string())

    generate_html_report(
        label_names=list(le.classes_),
        y_true=y_true,
        y_pred=y_pred,
        cm=cm,
        report_dict=report_dict,
        hyperparams=hyperparams,
    )

    return report_dict, y_pred, y_true
```

- [ ] **Step 5: Create `src/training/pipeline.py`**

```python
import logging

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

import pandas as pd
from src.training.models import ModelConfig
from src.training.split import make_split
from src.training.tokenize import ClassiflowDataset, prepare_text
from src.training.trainer import WeightedTrainer, compute_metrics
from src.training.evaluate import run_evaluation
from wandb_logger import WandbLogger

log = logging.getLogger(__name__)


def run(  # noqa: PLR0913
    df: pd.DataFrame,
    model_cfg: ModelConfig,
    *,
    epochs: int,
    early_stop_patience: int,
    chunk_strategy: str,
    seed: int,
    output_dir: str,
    use_wandb: bool = True,
) -> tuple[Trainer, LabelEncoder]:
    log.info("=" * 60)
    log.info("CLASSIFLOW — FINE-TUNING %s", model_cfg.hf_id)
    log.info("=" * 60)

    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])
    label2id = {cls: int(i) for i, cls in enumerate(le.classes_)}
    id2label = {int(i): cls for cls, i in label2id.items()}
    num_labels = len(le.classes_)
    log.info("%d classes: %s", num_labels, list(le.classes_))

    train_df, val_df, test_df = make_split(df, seed=seed)

    raw_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["label_id"]),
        y=train_df["label_id"].to_numpy(),
    )
    class_weights = torch.tensor(raw_weights, dtype=torch.float)
    log.info("Class weights: %s", dict(zip(le.classes_, raw_weights.round(3), strict=True)))

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.hf_id)

    def _texts(split_df: pd.DataFrame, strategy: str) -> list[str]:
        return [prepare_text(t, tokenizer, strategy) for t in split_df["text"]]

    train_ds = ClassiflowDataset(_texts(train_df, chunk_strategy), train_df["label_id"].tolist(), tokenizer, model_cfg.max_tokens)
    val_ds   = ClassiflowDataset(_texts(val_df,   "first"),         val_df["label_id"].tolist(),   tokenizer, model_cfg.max_tokens)
    test_ds  = ClassiflowDataset(_texts(test_df,  "first"),         test_df["label_id"].tolist(),  tokenizer, model_cfg.max_tokens)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg.hf_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    use_bf16 = not model_cfg.force_fp32 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = not model_cfg.force_fp32 and torch.cuda.is_available() and not use_bf16
    precision = "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32"
    log.info("Mixed precision: %s", precision)

    steps_per_epoch = max(1, len(train_ds) // (model_cfg.batch_size * model_cfg.grad_accum))
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * 0.1))

    hyperparams = {
        "model": model_cfg.hf_id,
        "epochs": epochs,
        "batch_size": model_cfg.batch_size,
        "grad_accum": model_cfg.grad_accum,
        "effective_batch": model_cfg.batch_size * model_cfg.grad_accum,
        "learning_rate": model_cfg.lr,
        "warmup_steps": warmup_steps,
        "precision": precision,
        "train_docs": len(train_df),
        "num_classes": num_labels,
    }

    wb = WandbLogger(enabled=use_wandb)
    wb.init(hyperparams)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=model_cfg.batch_size,
        per_device_eval_batch_size=model_cfg.batch_size,
        gradient_accumulation_steps=model_cfg.grad_accum,
        learning_rate=model_cfg.lr,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=use_bf16,
        fp16=use_fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=10,
        report_to=wb.report_to,
        seed=seed,
        dataloader_num_workers=4,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stop_patience)],
    )
    log.info("Early stopping: patience=%d epochs on macro_f1", early_stop_patience)
    log.info("Training — %d epochs, effective batch %d", epochs, model_cfg.batch_size * model_cfg.grad_accum)

    trainer.train()
    log.info("Training complete")

    report_dict, y_pred, y_true = run_evaluation(trainer, test_ds, le, hyperparams)
    wb.log_results(report_dict, y_true, y_pred, list(le.classes_))
    wb.finish()

    save_path = f"{output_dir}/final"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    log.info("Model saved to %s", save_path)

    return trainer, le
```

- [ ] **Step 6: Write and run split tests**

Create `tests/training/test_split.py`:

```python
import pandas as pd
import pytest

from src.training.split import make_split


@pytest.fixture()
def balanced_df() -> pd.DataFrame:
    rows = []
    for label in ["a", "b", "c"]:
        for i in range(20):
            rows.append({"text": f"doc {i}", "label": label, "label_id": ["a", "b", "c"].index(label)})
    return pd.DataFrame(rows)


def test_split_sizes(balanced_df):
    train, val, test = make_split(balanced_df, seed=42)
    total = len(balanced_df)
    assert len(train) + len(val) + len(test) == total


def test_split_no_overlap(balanced_df):
    train, val, test = make_split(balanced_df, seed=42)
    train_idx = set(train.index)
    val_idx = set(val.index)
    test_idx = set(test.index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)


def test_split_is_deterministic(balanced_df):
    t1, v1, _ = make_split(balanced_df, seed=42)
    t2, v2, _ = make_split(balanced_df, seed=42)
    assert list(t1.index) == list(t2.index)
    assert list(v1.index) == list(v2.index)
```

```powershell
uv run pytest tests/training/test_split.py -v
```
Expected: 3 passed.

- [ ] **Step 7: Run `uv run poe check`**

```powershell
uv run poe check
```
Expected: lint passes for all `src/training/` files, mypy validates `WeightedTrainer` and `make_split` signatures, split + tokenize tests pass. Existing warnings in legacy flat files are acceptable until Task 9.

- [ ] **Step 8: Commit**

```bash
git add src/training/ tests/training/
git commit -m "feat: migrate training pipeline to src/training with model registry"
```

---

## Task 5: Migrate inference pipeline

**Files:**
- Create: `src/inference/classify.py`
- Create: `src/inference/pipeline.py`
- Create: `tests/inference/test_pipeline.py`

**Interfaces:**
- Consumes: `src/ingestion/extract.py:extract_pdf`, `src/ingestion/extract.py:clean_text`
- Produces:
  - `classify.ClassiflowClassifier(model_path: str, *, confidence_threshold: float = 0.70)`
  - `classify.ClassiflowClassifier.predict_text(text: str) -> dict`
  - `pipeline.predict_pdf(model_path: str, pdf_path: str, *, threshold: float, use_ocr: bool) -> dict`
  - `pipeline.predict_folder(model_path: str, folder_path: str, *, threshold: float, use_ocr: bool) -> pd.DataFrame`

- [ ] **Step 1: Create `src/inference/classify.py`**

```python
import logging

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.ingestion.extract import clean_text

log = logging.getLogger(__name__)


class ClassiflowClassifier:
    def __init__(self, model_path: str, *, confidence_threshold: float = 0.70) -> None:
        log.info("Loading classifier from %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.threshold = confidence_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        self.model.to(self.device)
        log.info("Classifier ready on %s", self.device)

    def predict_text(self, text: str) -> dict:
        inputs = self.tokenizer(
            clean_text(text),
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self.model(**inputs).logits, dim=-1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        label = self.model.config.id2label[pred_idx]

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "certain": confidence >= self.threshold,
            "all_scores": {
                self.model.config.id2label[i]: round(float(p), 4)
                for i, p in enumerate(probs)
            },
        }
```

- [ ] **Step 2: Create `src/inference/pipeline.py`**

```python
import logging
from pathlib import Path

import pandas as pd

from src.inference.classify import ClassiflowClassifier
from src.ingestion.extract import extract_pdf

log = logging.getLogger(__name__)


def predict_pdf(
    model_path: str,
    pdf_path: str,
    *,
    threshold: float = 0.70,
    use_ocr: bool = True,
) -> dict:
    clf = ClassiflowClassifier(model_path, confidence_threshold=threshold)
    log.info("Classifying: %s", Path(pdf_path).name)
    text = extract_pdf(pdf_path, use_ocr_fallback=use_ocr)

    if text is None:
        log.warning("Could not extract text from %s", Path(pdf_path).name)
        return {
            "label": None,
            "confidence": 0.0,
            "certain": False,
            "error": "empty/unreadable document",
            "filename": Path(pdf_path).name,
        }

    result = clf.predict_text(text)
    result["filename"] = Path(pdf_path).name
    log.info("%s → %s (%.2f%%)", Path(pdf_path).name, result["label"], result["confidence"] * 100)
    return result


def predict_folder(
    model_path: str,
    folder_path: str,
    *,
    threshold: float = 0.70,
    use_ocr: bool = True,
) -> pd.DataFrame:
    clf = ClassiflowClassifier(model_path, confidence_threshold=threshold)
    pdfs = sorted(Path(folder_path).glob("*.pdf"))
    log.info("Classifying %d PDFs in %s", len(pdfs), folder_path)

    results = []
    for pdf in pdfs:
        text = extract_pdf(str(pdf), use_ocr_fallback=use_ocr)
        if text is None:
            results.append({"filename": pdf.name, "label": None, "confidence": 0.0, "certain": False})
            continue
        r = clf.predict_text(text)
        r["filename"] = pdf.name
        results.append(r)

    log.info("Folder classification complete")
    return pd.DataFrame(results)
```

- [ ] **Step 3: Write inference unit tests (mocked model)**

Create `tests/inference/test_pipeline.py`:

```python
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from src.inference.classify import ClassiflowClassifier


def _make_mock_classifier() -> ClassiflowClassifier:
    with patch("src.inference.classify.AutoTokenizer.from_pretrained") as mock_tok, \
         patch("src.inference.classify.AutoModelForSequenceClassification.from_pretrained") as mock_model, \
         patch("torch.cuda.is_available", return_value=False):

        mock_tok.return_value = MagicMock()
        model = MagicMock()
        model.config.id2label = {0: "decreto", 1: "ordenanza"}
        logits = torch.tensor([[2.0, 0.5]])
        model.return_value.logits = logits
        mock_model.return_value = model

        clf = ClassiflowClassifier("fake/path", confidence_threshold=0.70)
        clf.tokenizer.return_value = {
            "input_ids": torch.zeros(1, 512, dtype=torch.long),
            "attention_mask": torch.ones(1, 512, dtype=torch.long),
        }
        clf.tokenizer.return_value.__class__ = dict
        return clf


def test_predict_text_returns_label():
    clf = _make_mock_classifier()
    result = clf.predict_text("Decreto numero 123")
    assert "label" in result
    assert "confidence" in result
    assert "certain" in result
    assert "all_scores" in result


def test_predict_text_certain_above_threshold():
    clf = _make_mock_classifier()
    # softmax([2.0, 0.5]) → [0.818, 0.182], which is above 0.70
    result = clf.predict_text("anything")
    assert result["certain"] is True
```

```powershell
uv run pytest tests/inference/test_pipeline.py -v
```
Expected: 2 passed.

- [ ] **Step 4: Run `uv run poe check`**

```powershell
uv run poe check
```
Expected: lint passes for `src/inference/`, mypy validates `ClassiflowClassifier` types, 2 inference tests pass. Existing warnings in legacy flat files are acceptable until Task 9.

- [ ] **Step 5: Commit**

```bash
git add src/inference/ tests/inference/
git commit -m "feat: migrate inference pipeline to src/inference"
```

---

## Task 6: Click CLI

**Files:**
- Create: `src/cli/train.py`
- Create: `src/cli/predict.py`
- Create: `src/cli/clean.py`
- Create: `tests/cli/test_commands.py`

**Interfaces:**
- Consumes: `src/ingestion/pipeline.run`, `src/training/pipeline.run`, `src/inference/pipeline.predict_pdf`, `src/inference/pipeline.predict_folder`
- Produces: Click commands `train`, `predict`, `predict-folder`, `clean` — importable by `main.py`

- [ ] **Step 1: Create `src/cli/clean.py`**

```python
import logging
import shutil
from pathlib import Path

import click

from config import CACHE_PATH, OUTPUT_DIR
from logger import setup_logging

log = logging.getLogger(__name__)

_LOG_FILE = Path("logs/classiflow.log")
_CACHE = Path(CACHE_PATH)
_MODEL_DIR = Path(OUTPUT_DIR)


def _release_log_file() -> None:
    root = logging.getLogger()
    for handler in root.handlers[:]:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == _LOG_FILE:
            handler.close()
            root.removeHandler(handler)


@click.command("clean")
def clean_cmd() -> None:
    """Wipe logs, dataset cache, and model checkpoints."""
    setup_logging()
    targets = [(_LOG_FILE, "log file"), (_CACHE, "dataset cache"), (_MODEL_DIR, "model checkpoints")]
    for path, label in targets:
        if not path.exists():
            log.info("Clean: %s not found, skipping", label)
            continue
        if path == _LOG_FILE:
            _release_log_file()
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        log.info("Clean: deleted %s (%s)", label, path)
    setup_logging()
    log.info("Clean complete")
```

- [ ] **Step 2: Create `src/cli/train.py`**

```python
import logging
from typing import Optional

import click

from config import CACHE_PATH, DOCS_ROOT, EARLY_STOP_PATIENCE, EPOCHS, MODEL_KEY, OUTPUT_DIR, SEED, CHUNK_STRATEGY
from logger import setup_logging
from src.training.models import get_model_config
from src.ingestion.pipeline import run as ingest
from src.training.pipeline import run as train_run
from src.cli.clean import clean_cmd

log = logging.getLogger(__name__)


@click.command("train")
@click.option("--docs-root", default=DOCS_ROOT, show_default=True, help="Root folder with labeled PDF subfolders")
@click.option("--model", "model_key", default=MODEL_KEY, show_default=True, help="Model registry key (e.g. xlm-roberta, beto)")
@click.option("--max-docs-per-class", type=int, default=None, help="Cap docs per class for quick test runs")
@click.option("--rebuild-cache", is_flag=True, default=False, help="Force re-extraction even if cache exists")
@click.option("--no-ocr", is_flag=True, default=False, help="Skip OCR fallback")
@click.option("--no-wandb", is_flag=True, default=False, help="Disable W&B logging")
@click.option("--clean", "do_clean", is_flag=True, default=False, help="Wipe state before training")
@click.option("--debug", is_flag=True, default=False, help="Enable DEBUG logging")
def train_cmd(
    docs_root: str,
    model_key: str,
    max_docs_per_class: Optional[int],
    rebuild_cache: bool,
    no_ocr: bool,
    no_wandb: bool,
    do_clean: bool,
    debug: bool,
) -> None:
    """Fine-tune a transformer model on municipal PDF documents."""
    setup_logging(level=logging.DEBUG if debug else logging.INFO)

    if do_clean:
        from click.testing import CliRunner
        CliRunner().invoke(clean_cmd)
        setup_logging(level=logging.DEBUG if debug else logging.INFO)

    model_cfg = get_model_config(model_key)
    log.info("Using model: %s (%s)", model_cfg.name, model_cfg.hf_id)

    df = ingest(docs_root, cache_path=CACHE_PATH, use_ocr=not no_ocr, rebuild=rebuild_cache, max_docs_per_class=max_docs_per_class)

    if len(df) == 0:
        log.error("No documents found. Check --docs-root: %s", docs_root)
        return

    train_run(
        df,
        model_cfg,
        epochs=EPOCHS,
        early_stop_patience=EARLY_STOP_PATIENCE,
        chunk_strategy=CHUNK_STRATEGY,
        seed=SEED,
        output_dir=OUTPUT_DIR,
        use_wandb=not no_wandb,
    )
```

- [ ] **Step 3: Create `src/cli/predict.py`**

```python
import logging

import click

from config import OUTPUT_DIR
from logger import setup_logging
from src.inference.pipeline import predict_folder, predict_pdf

log = logging.getLogger(__name__)

_DEFAULT_MODEL = f"{OUTPUT_DIR}/final"


@click.command("predict")
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--model-path", default=_DEFAULT_MODEL, show_default=True)
@click.option("--threshold", default=0.70, show_default=True, help="Confidence threshold")
@click.option("--no-ocr", is_flag=True, default=False)
@click.option("--debug", is_flag=True, default=False)
def predict_cmd(pdf_path: str, model_path: str, threshold: float, no_ocr: bool, debug: bool) -> None:
    """Classify a single PDF document."""
    setup_logging(level=logging.DEBUG if debug else logging.INFO)
    result = predict_pdf(model_path, pdf_path, threshold=threshold, use_ocr=not no_ocr)

    click.echo(f"\n{'─' * 50}")
    click.echo(f"  File      : {result.get('filename', pdf_path)}")
    click.echo(f"  Label     : {result['label']}")
    click.echo(f"  Confidence: {result['confidence']:.2%}")
    click.echo(f"  Certain   : {result['certain']}")
    click.echo("\n  All scores:")
    for lbl, sc in sorted(result.get("all_scores", {}).items(), key=lambda x: -x[1]):
        bar = "█" * int(sc * 40)
        click.echo(f"    {lbl:<38} {sc:.4f}  {bar}")


@click.command("predict-folder")
@click.argument("folder_path", type=click.Path(exists=True, file_okay=False))
@click.option("--model-path", default=_DEFAULT_MODEL, show_default=True)
@click.option("--threshold", default=0.70, show_default=True)
@click.option("--no-ocr", is_flag=True, default=False)
@click.option("--output", default="classiflow_predictions.csv", show_default=True)
@click.option("--debug", is_flag=True, default=False)
def predict_folder_cmd(
    folder_path: str, model_path: str, threshold: float, no_ocr: bool, output: str, debug: bool
) -> None:
    """Classify all PDFs in a folder and save results to CSV."""
    setup_logging(level=logging.DEBUG if debug else logging.INFO)
    df_out = predict_folder(model_path, folder_path, threshold=threshold, use_ocr=not no_ocr)
    df_out.to_csv(output, index=False)
    log.info("Results saved to %s", output)
```

- [ ] **Step 4: Write Click CLI tests**

Create `tests/cli/test_commands.py`:

```python
from click.testing import CliRunner

from src.cli.clean import clean_cmd
from src.cli.predict import predict_cmd, predict_folder_cmd
from src.cli.train import train_cmd


def test_train_cmd_help():
    runner = CliRunner()
    result = runner.invoke(train_cmd, ["--help"])
    assert result.exit_code == 0
    assert "Fine-tune" in result.output


def test_predict_cmd_help():
    runner = CliRunner()
    result = runner.invoke(predict_cmd, ["--help"])
    assert result.exit_code == 0
    assert "single PDF" in result.output.lower() or "classify" in result.output.lower()


def test_predict_folder_cmd_help():
    runner = CliRunner()
    result = runner.invoke(predict_folder_cmd, ["--help"])
    assert result.exit_code == 0
    assert "folder" in result.output.lower()


def test_clean_cmd_help():
    runner = CliRunner()
    result = runner.invoke(clean_cmd, ["--help"])
    assert result.exit_code == 0
    assert "Wipe" in result.output or "clean" in result.output.lower()
```

```powershell
uv run pytest tests/cli/test_commands.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Run `uv run poe check`**

```powershell
uv run poe check
```
Expected: lint passes for `src/cli/`, mypy validates Click command signatures, 4 CLI help tests pass. Existing warnings in legacy flat files are acceptable until Task 9.

- [ ] **Step 6: Commit**

```bash
git add src/cli/ tests/cli/
git commit -m "feat: replace argparse with Click CLI"
```

---

## Task 7: FastAPI inference API

**Files:**
- Create: `src/api/app.py`
- Create: `src/api/routes/predict.py`
- Create: `tests/api/test_predict.py`

**Interfaces:**
- Consumes: `src/inference/pipeline.predict_pdf`
- Produces:
  - `POST /predict` — accepts `multipart/form-data` with `file: UploadFile`, returns `PredictResponse`
  - `GET /health` — returns `{"status": "ok"}`
  - `create_app(model_path: str, threshold: float) -> FastAPI`

- [ ] **Step 1: Create `src/api/routes/predict.py`**

```python
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.inference.pipeline import predict_pdf

router = APIRouter()


class PredictResponse(BaseModel):
    filename: str
    label: str | None
    confidence: float
    certain: bool
    all_scores: dict[str, float]
    error: str | None = None


@router.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    threshold: float = 0.70,
) -> PredictResponse:
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = predict_pdf(
            router.state.model_path,  # type: ignore[attr-defined]
            tmp_path,
            threshold=threshold,
            use_ocr=True,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return PredictResponse(**result)
```

- [ ] **Step 2: Create `src/api/app.py`**

```python
from fastapi import FastAPI

from src.api.routes.predict import router as predict_router


def create_app(model_path: str, threshold: float = 0.70) -> FastAPI:
    app = FastAPI(
        title="Classiflow API",
        description="Classifies Spanish municipal PDF documents",
        version="0.1.0",
    )

    # Store config on router state so the route can access it
    predict_router.state = type("State", (), {"model_path": model_path, "threshold": threshold})()  # type: ignore[attr-defined]

    app.include_router(predict_router)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app
```

- [ ] **Step 3: Write API tests**

Create `tests/api/test_predict.py`:

```python
from fastapi.testclient import TestClient

from src.api.app import create_app


def test_health_endpoint():
    app = create_app(model_path="fake/path")
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_rejects_non_pdf():
    app = create_app(model_path="fake/path")
    client = TestClient(app)
    response = client.post(
        "/predict",
        files={"file": ("document.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]
```

```powershell
uv run pytest tests/api/test_predict.py -v
```
Expected: 2 passed.

- [ ] **Step 4: Run `uv run poe check`**

```powershell
uv run poe check
```
Expected: lint passes for `src/api/`, mypy validates FastAPI route types and `PredictResponse` model, 2 API tests pass. Existing warnings in legacy flat files are acceptable until Task 9.

- [ ] **Step 5: Commit**

```bash
git add src/api/ tests/api/
git commit -m "feat: add FastAPI inference endpoint at POST /predict"
```

---

## Task 8: Rewrite `main.py`

**Files:**
- Modify: `main.py` — replace argparse with Click group wiring
- Modify: `config.py` — add `API_HOST`, `API_PORT` constants

**Interfaces:**
- Consumes: all four Click commands from `src/cli/`
- Produces: `classiflow` CLI group + optional `serve` subcommand for the API

- [ ] **Step 1: Add API config to `config.py`**

```python
API_HOST = "0.0.0.0"
API_PORT = 8000
```

- [ ] **Step 2: Rewrite `main.py`**

```python
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).parent / "models"))

import click

from src.cli.clean import clean_cmd
from src.cli.predict import predict_cmd, predict_folder_cmd
from src.cli.train import train_cmd


@click.group()
def cli() -> None:
    """Classiflow — Spanish municipal document classifier."""


cli.add_command(train_cmd,          name="train")
cli.add_command(predict_cmd,        name="predict")
cli.add_command(predict_folder_cmd, name="predict-folder")
cli.add_command(clean_cmd,          name="clean")


@cli.command("serve")
@click.option("--model-path", required=True, help="Path to saved model")
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True)
@click.option("--threshold", default=0.70, show_default=True)
def serve_cmd(model_path: str, host: str, port: int, threshold: float) -> None:
    """Start the FastAPI inference server."""
    import uvicorn
    from src.api.app import create_app
    app = create_app(model_path=model_path, threshold=threshold)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli()
```

- [ ] **Step 3: Verify `--help` works**

```powershell
uv run python main.py --help
uv run python main.py train --help
uv run python main.py serve --help
```
Expected: each prints usage without import errors.

- [ ] **Step 4: Run `uv run poe check`**

```powershell
uv run poe check
```
Expected: lint passes for `main.py`, mypy validates the Click group wiring, all tests still pass. Existing warnings in legacy flat files are acceptable until Task 9.

- [ ] **Step 5: Commit**

```bash
git add main.py config.py
git commit -m "feat: rewrite main.py as Click group with serve subcommand"
```

---

## Task 9: Delete old flat files + update docs

**Files:**
- Delete: `extraction.py`, `dataset.py`, `training.py`, `classifier.py`
- Modify: `.gitignore` — confirm `src/` not ignored
- Modify: `README.md` — update structure, commands, model list

- [ ] **Step 1: Run the full test suite to confirm nothing breaks**

```powershell
uv run pytest -v
```
Expected: all tests pass.

- [ ] **Step 2: Delete old files**

```powershell
Remove-Item extraction.py, dataset.py, training.py, classifier.py
```

- [ ] **Step 3: Run tests again to confirm no hidden dependency on old files**

```powershell
uv run pytest -v
```
Expected: all tests still pass.

- [ ] **Step 4: Update `README.md` — new structure section**

Replace the project structure section with:

```
src/
├── ingestion/         extract.py · scan.py · cache.py · pipeline.py
├── training/
│   ├── models/        xlm_roberta.py · beto.py · __init__.py (registry)
│   └──               split.py · tokenize.py · trainer.py · evaluate.py · pipeline.py
├── inference/         classify.py · pipeline.py
├── api/               app.py · routes/predict.py
└── cli/               train.py · predict.py · clean.py

config.py · wandb_logger.py · reporting.py · logger.py · main.py
models/   (git-ignored — trained weights)
data/     (git-ignored — parquet cache)
reports/  (git-ignored — HTML reports)
```

Update the Usage section with Click commands:
```powershell
uv run python main.py train --docs-root "C:\path\to\downloads"
uv run python main.py train --model beto --max-docs-per-class 100
uv run python main.py predict path/to/doc.pdf
uv run python main.py predict-folder path/to/folder
uv run python main.py serve --model-path ./models/classiflow_model/final
uv run python main.py clean
```

- [ ] **Step 5: Run `uv run poe check` — full clean-codebase gate**

```powershell
uv run poe check
```
Expected: lint, mypy, and all tests pass with **zero** errors now that the legacy flat files are gone. This is the final quality gate — no acceptable warnings remain.

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "refactor: complete scaffold migration to src/ pipeline layout"
```

---

## Self-Review

**Spec coverage:**
- ✅ `src/` pipeline layout per image
- ✅ Click CLI replacing argparse
- ✅ FastAPI inference endpoint
- ✅ Multiple model support via registry
- ✅ `models/` and `data/` at root (not inside `assets/`)
- ✅ Ingestion and extraction separated into focused files
- ✅ CLI and API both call the same pipeline layer

**Placeholder scan:** None found — all steps contain actual code.

**Type consistency:**
- `predict_pdf` in `src/inference/pipeline.py` matches usage in `src/cli/predict.py` and `src/api/routes/predict.py`
- `ModelConfig` fields (`batch_size`, `grad_accum`, `lr`, etc.) used consistently in `src/training/pipeline.py`
- `WandbLogger(enabled=use_wandb)` keyword-only call matches `wandb_logger.py` signature
