# Training Output Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent silent data loss and user confusion around the training output directory by adding early write validation, separating HuggingFace checkpoints from the final model, and fixing the .gitignore to cover all versioned output directories.

**Architecture:** Three independent fixes in two files. `src/training/pipeline.py` gets an early `Path.mkdir()` call that fails fast on permission errors, and its `TrainingArguments.output_dir` is redirected to a `checkpoints/` subdirectory so that `final/` stays clean. `.gitignore` gets a glob entry replacing the hardcoded versioned directory name.

**Tech Stack:** Python ≥ 3.10, pathlib, HuggingFace Transformers `TrainingArguments`, pytest, ruff, mypy

## Global Constraints

- Python ≥ 3.10 — use `X | Y` union types, not `Optional[X]`
- No inline `# noqa:` or `# type: ignore` unless already present in the file
- `uv run poe check` must pass (ruff + mypy + pytest) after each task
- Commit after every task — do not batch tasks into one commit
- `src/training/pipeline.py` uses pathlib (`Path`) elsewhere — keep consistent

---

### Task 1: Fix .gitignore glob for versioned model directories

**Files:**
- Modify: `.gitignore:209-210`

**Interfaces:**
- Consumes: nothing
- Produces: nothing (pure config change)

- [ ] **Step 1: Open `.gitignore` and locate the models section**

Find this block (around line 206-210):
```
# Keep the models/ folder structure but ignore downloaded/trained weights
models/xet
models/hub/
models/bert_tunning_model/
models/bert_tunning_model_xlmroberta_v1/
```

- [ ] **Step 2: Replace the hardcoded versioned entry with a glob**

Replace that block with:
```
# Keep the models/ folder structure but ignore downloaded/trained weights
models/xet
models/hub/
models/bert_tunning_model/
models/bert_tunning_model_*/
```

The trailing `models/bert_tunning_model/` entry stays — it covers the default output path.  
`models/bert_tunning_model_*/` covers any versioned variant (`_beto_v1/`, `_xlmroberta_v2/`, etc.) without requiring a manual edit per run.

- [ ] **Step 3: Verify git sees the change correctly**

Run:
```powershell
git check-ignore -v models/bert_tunning_model_xlmroberta_v1
git check-ignore -v models/bert_tunning_model_beto_v2
```

Expected: both lines report `.gitignore` as the matching rule.

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "fix: replace hardcoded versioned model dir with glob in .gitignore"
```

---

### Task 2: Separate HuggingFace checkpoints from the final model directory

**Files:**
- Modify: `src/training/pipeline.py:108-109` and `src/training/pipeline.py:155`
- Test: `tests/training/test_pipeline_paths.py` (new file)

**Interfaces:**
- Consumes: `TrainingRequest.output_dir: str` (defined in `src/training/options.py`)
- Produces: `TrainingArguments(output_dir=str)` receives `<output_dir>/checkpoints`; `trainer.save_model()` receives `<output_dir>/final`

**Context:** HuggingFace `Trainer` with `save_strategy="epoch"` writes `checkpoint-N/` subdirectories into `TrainingArguments.output_dir`. Currently that equals `request.output_dir`, so checkpoints and `final/` both land in the same directory. The fix redirects checkpoints to `<output_dir>/checkpoints/` so `final/` is the only thing at the top level of the user-specified path.

- [ ] **Step 1: Write the failing test**

Create `tests/training/test_pipeline_paths.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.training.options import TrainingRequest


def _make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"text": ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"],
         "label": ["a", "a", "a", "b", "b", "b"]}
    )


@pytest.fixture()
def mock_model_cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.hf_id = "xlm-roberta-base"
    cfg.max_tokens = 512
    cfg.batch_size = 2
    cfg.grad_accum = 1
    cfg.lr = 2e-5
    cfg.force_fp32 = True
    return cfg


def test_training_arguments_output_dir_is_checkpoints_subdir(
    mock_model_cfg: MagicMock, tmp_path: Path
) -> None:
    """TrainingArguments.output_dir must point to <output_dir>/checkpoints, not output_dir."""
    captured: dict[str, str] = {}

    def fake_training_args(**kwargs: object) -> MagicMock:
        captured["output_dir"] = str(kwargs.get("output_dir", ""))
        m = MagicMock()
        m.output_dir = captured["output_dir"]
        m.gradient_accumulation_steps = 1
        return m

    request = TrainingRequest(output_dir=str(tmp_path / "my_model"), use_wandb=False)

    with (
        patch("src.training.pipeline.AutoTokenizer.from_pretrained", return_value=MagicMock()),
        patch("src.training.pipeline.AutoModelForSequenceClassification.from_pretrained", return_value=MagicMock()),
        patch("src.training.pipeline.TrainingArguments", side_effect=fake_training_args),
        patch("src.training.pipeline.WeightedTrainer", return_value=MagicMock()),
        patch("src.training.pipeline.make_split", return_value=(_make_df(), _make_df(), _make_df())),
        patch("src.training.pipeline.run_evaluation", return_value=({}, [], [])),
        patch("src.training.pipeline.WandbLogger", return_value=MagicMock()),
    ):
        from src.training.pipeline import run
        run(_make_df(), mock_model_cfg, request)

    assert captured["output_dir"] == str(tmp_path / "my_model" / "checkpoints")


def test_save_model_path_is_final_subdir(
    mock_model_cfg: MagicMock, tmp_path: Path
) -> None:
    """trainer.save_model and tokenizer.save_pretrained must receive <output_dir>/final."""
    save_paths: list[str] = []
    mock_trainer = MagicMock()
    mock_trainer.save_model.side_effect = lambda p: save_paths.append(str(p))

    mock_tokenizer = MagicMock()
    mock_tokenizer.save_pretrained.side_effect = lambda p: save_paths.append(str(p))

    request = TrainingRequest(output_dir=str(tmp_path / "my_model"), use_wandb=False)

    with (
        patch("src.training.pipeline.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        patch("src.training.pipeline.AutoModelForSequenceClassification.from_pretrained", return_value=MagicMock()),
        patch("src.training.pipeline.TrainingArguments", return_value=MagicMock()),
        patch("src.training.pipeline.WeightedTrainer", return_value=mock_trainer),
        patch("src.training.pipeline.make_split", return_value=(_make_df(), _make_df(), _make_df())),
        patch("src.training.pipeline.run_evaluation", return_value=({}, [], [])),
        patch("src.training.pipeline.WandbLogger", return_value=MagicMock()),
    ):
        from src.training.pipeline import run
        run(_make_df(), mock_model_cfg, request)

    expected = str(tmp_path / "my_model" / "final")
    assert all(p == expected for p in save_paths), f"save paths were {save_paths}"
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
uv run pytest tests/training/test_pipeline_paths.py -v
```

Expected: both tests FAIL — `captured["output_dir"]` will equal `str(tmp_path / "my_model")` instead of the `checkpoints` subdir.

- [ ] **Step 3: Update `src/training/pipeline.py`**

Change line 109 (TrainingArguments `output_dir`) and line 155 (save_path). The `output_dir` for `TrainingArguments` becomes `<output_dir>/checkpoints`; `save_path` stays `<output_dir>/final` but uses `Path`:

```python
# line 108 — TrainingArguments block, change output_dir argument:
    args = TrainingArguments(
        output_dir=str(Path(request.output_dir) / "checkpoints"),
        ...
    )

# line 155 — save block, use Path:
    save_path = Path(request.output_dir) / "final"
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    log.info("Model saved to %s", save_path)
```

The full edited section at line 108:
```python
    args = TrainingArguments(
        output_dir=str(Path(request.output_dir) / "checkpoints"),
        num_train_epochs=request.epochs,
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
        seed=request.seed,
        dataloader_num_workers=4,
    )
```

The full edited save block at line 155:
```python
    save_path = Path(request.output_dir) / "final"
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    log.info("Model saved to %s", save_path)
```

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
uv run pytest tests/training/test_pipeline_paths.py -v
```

Expected: both PASS.

- [ ] **Step 5: Run full check**

```powershell
uv run poe check
```

Expected: all checks pass.

- [ ] **Step 6: Commit**

```bash
git add src/training/pipeline.py tests/training/test_pipeline_paths.py
git commit -m "fix: separate HF checkpoints from final model into checkpoints/ subdir"
```

---

### Task 3: Fail fast when output_dir is not writable

**Files:**
- Modify: `src/training/pipeline.py` (add write check before `trainer.train()`)
- Test: `tests/training/test_pipeline_paths.py` (add to existing file from Task 2)

**Interfaces:**
- Consumes: `TrainingRequest.output_dir: str`, `Path.mkdir(parents=True, exist_ok=True)`
- Produces: raises `PermissionError` (or `OSError`) before training starts if the directory cannot be created or written to

**Context:** `TrainingArguments` construction does not check write permissions. The first actual write happens at the first checkpoint save during training. A bad path (read-only, full disk) would only surface after the full training run completes. The fix is to call `Path(output_dir).mkdir(parents=True, exist_ok=True)` — which fails immediately on permission errors — right before `trainer.train()`.

- [ ] **Step 1: Write the failing test**

Add to `tests/training/test_pipeline_paths.py`:

```python
import stat
import sys


@pytest.mark.skipif(sys.platform == "win32" and not _running_as_admin(), reason="chmod restricted on Windows without admin")
def test_run_raises_before_training_if_output_dir_not_writable(
    mock_model_cfg: MagicMock, tmp_path: Path
) -> None:
    """run() must raise OSError before trainer.train() if output_dir is not writable."""
    locked_dir = tmp_path / "locked"
    locked_dir.mkdir()
    locked_dir.chmod(stat.S_IREAD | stat.S_IEXEC)  # read + exec only, no write

    mock_trainer = MagicMock()
    request = TrainingRequest(output_dir=str(locked_dir / "output"), use_wandb=False)

    try:
        with (
            patch("src.training.pipeline.AutoTokenizer.from_pretrained", return_value=MagicMock()),
            patch("src.training.pipeline.AutoModelForSequenceClassification.from_pretrained", return_value=MagicMock()),
            patch("src.training.pipeline.TrainingArguments", return_value=MagicMock()),
            patch("src.training.pipeline.WeightedTrainer", return_value=mock_trainer),
            patch("src.training.pipeline.make_split", return_value=(_make_df(), _make_df(), _make_df())),
            patch("src.training.pipeline.WandbLogger", return_value=MagicMock()),
        ):
            from src.training.pipeline import run
            with pytest.raises(OSError):
                run(_make_df(), mock_model_cfg, request)

        mock_trainer.train.assert_not_called()
    finally:
        locked_dir.chmod(stat.S_IRWXU)  # restore so tmp_path cleanup works


def _running_as_admin() -> bool:
    import ctypes
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False
```

- [ ] **Step 2: Run the test to verify it fails**

```powershell
uv run pytest tests/training/test_pipeline_paths.py::test_run_raises_before_training_if_output_dir_not_writable -v
```

Expected: FAIL — `mock_trainer.train` IS called (no early check exists yet).

- [ ] **Step 3: Add the early write check to `src/training/pipeline.py`**

Add immediately before `trainer.train()` (line 148). The full block:

```python
    # Fail fast — verify output_dir is writable before the training run starts.
    output_path = Path(request.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trainer.train()
```

This creates the directory (and all parents) immediately. If `request.output_dir` is inside a read-only parent, `mkdir` raises `PermissionError` (a subclass of `OSError`) before a single training step runs.

- [ ] **Step 4: Run all pipeline tests**

```powershell
uv run pytest tests/training/test_pipeline_paths.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Run full check**

```powershell
uv run poe check
```

Expected: all checks pass.

- [ ] **Step 6: Commit**

```bash
git add src/training/pipeline.py tests/training/test_pipeline_paths.py
git commit -m "fix: validate output_dir is writable before training starts"
```
