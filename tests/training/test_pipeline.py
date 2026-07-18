from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.embeddings import LoadedModel
from src.ood import load_stats
from src.svm_reviewer import load_svm_classifiers
from src.training.models import ModelConfig
from src.training.options import TrainingRequest
from src.training.pipeline import run


@pytest.fixture
def balanced_df() -> pd.DataFrame:
    # 3 classes x 20 docs, mirroring tests/training/test_split.py's own balanced_df --
    # unique-per-doc text (not one repeated string) so compute_class_stats' TF-IDF fitting
    # survives OOD_TFIDF_MAX_DF pruning (an identical-everywhere corpus has 100% document
    # frequency for every term, which max_df=0.5 would prune to an empty vocabulary).
    labels = ["decreto", "ordenanza", "resolucion"]
    rows = [
        {"text": f"{label} rosario municipal intendente doc {i}", "label": label}
        for label in labels
        for i in range(20)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def model_cfg() -> ModelConfig:
    return ModelConfig(
        name="test-model",
        hf_id="test/test-model",
        max_tokens=16,
        lr=5e-5,
        batch_size=4,
        grad_accum=1,
        force_fp32=True,
    )


def test_run_orchestrates_training_pipeline_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    balanced_df: pd.DataFrame,
    model_cfg: ModelConfig,
    tmp_path: Path,
) -> None:
    # Characterization test: locks down run()'s current orchestration behavior (call
    # order, what's passed to what, what's persisted where) BEFORE the extraction into
    # _encode_labels/_split_and_weight/_build_datasets/_build_model/_compute_hyperparams/
    # _build_trainer/_generate_auxiliary_artifacts/_persist_artifacts -- and must keep
    # passing unchanged after it, or the extraction silently changed behavior.
    #
    # Mocking boundary: only mock what's genuinely expensive/network-bound (tokenizer and
    # model download, the real HF training loop) or has no dedicated test file elsewhere
    # (run_evaluation). Everything else -- label encoding, split, class weights, dataset
    # construction, OOD stats, SVM fitting, W&B's real no-op path, artifact persistence --
    # runs for real against this small synthetic fixture, since that's exactly the wiring
    # this test exists to verify.
    tokenizer = MagicMock()
    from_pretrained_tokenizer = MagicMock(return_value=tokenizer)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", from_pretrained_tokenizer)

    model = MagicMock()
    model.config.model_type = "test-model"
    model.config.hidden_size = 8
    model.device = "cpu"
    from_pretrained_model = MagicMock(return_value=model)
    monkeypatch.setattr(
        AutoModelForSequenceClassification, "from_pretrained", from_pretrained_model
    )

    trainer_cls = MagicMock()
    trainer = trainer_cls.return_value
    # The real HF Trainer.save_model() creates its target directory as a side effect --
    # save_stats/save_svm_classifiers (real, unmocked) depend on that directory already
    # existing, same as they do in production.
    trainer.save_model.side_effect = lambda path: Path(path).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("src.training.pipeline.WeightedTrainer", trainer_cls)

    rng = np.random.default_rng(0)

    def _fake_extract_embeddings(
        loaded: LoadedModel,  # noqa: ARG001
        texts: list[str],
        *,
        max_length: int,  # noqa: ARG001
        batch_size: int = 16,  # noqa: ARG001
    ) -> npt.NDArray[np.float64]:
        return rng.normal(size=(len(texts), 8))

    monkeypatch.setattr("src.training.pipeline.extract_embeddings", _fake_extract_embeddings)

    mock_result = MagicMock()
    run_evaluation = MagicMock(return_value=mock_result)
    monkeypatch.setattr("src.training.pipeline.run_evaluation", run_evaluation)

    request = TrainingRequest(
        epochs=1,
        early_stop_patience=1,
        chunk_strategy="first",
        seed=42,
        output_dir=str(tmp_path),
        use_wandb=False,
    )

    result_trainer, le = run(balanced_df, model_cfg, request)

    # 1. Return value.
    assert result_trainer is trainer
    assert list(le.classes_) == ["decreto", "ordenanza", "resolucion"]

    # 2. Model constructed with the label encoding run() itself computed.
    call_kwargs = from_pretrained_model.call_args.kwargs
    assert call_kwargs["num_labels"] == 3  # noqa: PLR2004
    assert call_kwargs["id2label"] == {0: "decreto", 1: "ordenanza", 2: "resolucion"}
    assert call_kwargs["label2id"] == {"decreto": 0, "ordenanza": 1, "resolucion": 2}

    # 3. Training happened.
    cast("MagicMock", trainer).train.assert_called_once()

    # 4. Persistence targeted the expected path.
    save_path = tmp_path / "final"
    cast("MagicMock", trainer).save_model.assert_called_once_with(str(save_path))
    tokenizer.save_pretrained.assert_called_once_with(str(save_path))

    # 5. Auxiliary artifacts were actually written and are loadable.
    ood_stats = load_stats(save_path / "ood_stats.npz")
    assert ood_stats.class_names == ["decreto", "ordenanza", "resolucion"]

    svm_classifiers = load_svm_classifiers(save_path / "svm_classifiers.joblib")
    assert svm_classifiers is not None
    assert set(svm_classifiers.keys()) == {"decreto", "ordenanza", "resolucion"}

    run_evaluation.assert_called_once()
