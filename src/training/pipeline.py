import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from src.embeddings import LoadedModel, extract_embeddings
from src.ood import compute_class_stats, save_stats
from src.schema import Hyperparams, OodArtifact
from src.settings import Settings
from src.svm_reviewer import evaluate_svm_classifiers, fit_svm_classifiers, save_svm_classifiers
from src.training.evaluate import run_evaluation
from src.training.models import ModelConfig
from src.training.options import TrainingRequest
from src.training.split import make_split
from src.training.tokenize import BertTunningDataset, prepare_text
from src.training.trainer import WeightedTrainer, compute_metrics
from src.wandb import WandbLogger

log = logging.getLogger(__name__)


def _fit_svm_reviewer(  # noqa: PLR0913 -- one param per input this needs from run(), all
    # required; bundling into a NamedTuple purely to dodge the arg-count limit here would
    # be more ceremony than the limit is worth for a single-caller helper.
    loaded: LoadedModel,
    model_cfg: ModelConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    class_names: list[str],
    train_embeddings: npt.NDArray[np.float64],
    wb: WandbLogger,
) -> dict[str, SVC]:
    """Fits the SVM reviewer and logs its held-out balanced accuracy (val split, never the
    training data it was fit on) -- pulled out of run() to keep it under ruff's statement
    limit, not because this logic is reused elsewhere. Logs into wb's already-open training
    run (a no-op if wb is disabled), mirroring compute-svm-classifiers' own --log-wandb."""
    classifiers = fit_svm_classifiers(train_embeddings, train_df["label_id"].tolist(), class_names)
    val_embeddings = extract_embeddings(
        loaded,
        [prepare_text(t, loaded.tokenizer, "first") for t in val_df["text"]],
        max_length=model_cfg.max_tokens,
    )
    svm_val_accuracy = evaluate_svm_classifiers(
        classifiers, val_embeddings, val_df["label_id"].tolist(), class_names
    )
    log.info(
        "SVM reviewer held-out balanced accuracy (val split): %s",
        {k: round(v, 4) for k, v in svm_val_accuracy.items()},
    )
    train_labels = train_df["label_id"]
    train_class_counts = {
        name: int((train_labels == idx).sum()) for idx, name in enumerate(class_names)
    }
    wb.log_svm_results(svm_val_accuracy, train_class_counts)
    return classifiers


class _LabelEncoding(NamedTuple):
    le: LabelEncoder
    label2id: dict[str, int]
    id2label: dict[int, str]
    num_labels: int


def _encode_labels(df: pd.DataFrame) -> _LabelEncoding:
    """Mutates df["label_id"] in place -- run() has always had this side effect; preserved
    here rather than returning a copy, since callers downstream (make_split, class weight
    computation) all expect df["label_id"] to already exist."""
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])
    label2id = {cls: int(i) for i, cls in enumerate(le.classes_)}
    id2label = {int(i): cls for cls, i in label2id.items()}
    num_labels = len(le.classes_)
    return _LabelEncoding(le=le, label2id=label2id, id2label=id2label, num_labels=num_labels)


class _SplitBundle(NamedTuple):
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    class_weights: torch.Tensor


def _split_and_weight(
    df: pd.DataFrame, num_labels: int, class_names: list[str], seed: int
) -> _SplitBundle:
    train_df, val_df, test_df = make_split(df, seed=seed)
    raw_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_labels),
        y=train_df["label_id"].to_numpy(),
    )
    class_weights = torch.tensor(raw_weights, dtype=torch.float)
    log.info("Class weights: %s", dict(zip(class_names, raw_weights.round(3), strict=True)))
    return _SplitBundle(
        train_df=train_df, val_df=val_df, test_df=test_df, class_weights=class_weights
    )


class _DatasetBundle(NamedTuple):
    tokenizer: PreTrainedTokenizerBase
    train_ds: BertTunningDataset
    val_ds: BertTunningDataset
    test_ds: BertTunningDataset


def _build_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_cfg: ModelConfig,
    chunk_strategy: str,
) -> _DatasetBundle:
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_cfg.hf_id)

    def _texts(split_df: pd.DataFrame, strategy: str) -> list[str]:
        return [prepare_text(t, tokenizer, strategy) for t in split_df["text"]]

    train_ds = BertTunningDataset(
        _texts(train_df, chunk_strategy),
        train_df["label_id"].tolist(),
        tokenizer,
        model_cfg.max_tokens,
    )
    val_ds = BertTunningDataset(
        _texts(val_df, "first"), val_df["label_id"].tolist(), tokenizer, model_cfg.max_tokens
    )
    test_ds = BertTunningDataset(
        _texts(test_df, "first"), test_df["label_id"].tolist(), tokenizer, model_cfg.max_tokens
    )
    return _DatasetBundle(tokenizer=tokenizer, train_ds=train_ds, val_ds=val_ds, test_ds=test_ds)


def _build_model(
    model_cfg: ModelConfig, num_labels: int, id2label: dict[int, str], label2id: dict[str, int]
) -> PreTrainedModel:
    # AutoModelForSequenceClassification.from_pretrained's stub returns Any -- narrowing to
    # the function's own declared PreTrainedModel return type, same as every other
    # from_pretrained call site in this codebase relies on implicitly.
    return AutoModelForSequenceClassification.from_pretrained(  # type: ignore[no-any-return]
        model_cfg.hf_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )


class _HyperparamsBundle(NamedTuple):
    hyperparams: Hyperparams
    use_bf16: bool
    use_fp16: bool
    warmup_steps: int


def _compute_hyperparams(
    model_cfg: ModelConfig,
    request: TrainingRequest,
    train_ds: BertTunningDataset,
    train_df: pd.DataFrame,
    num_labels: int,
) -> _HyperparamsBundle:
    use_bf16 = (
        not model_cfg.force_fp32 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    )
    use_fp16 = not model_cfg.force_fp32 and torch.cuda.is_available() and not use_bf16
    precision = "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32"
    log.info("Mixed precision: %s", precision)

    steps_per_epoch = max(1, len(train_ds) // (model_cfg.batch_size * model_cfg.grad_accum))
    total_steps = steps_per_epoch * request.epochs
    warmup_steps = max(1, int(total_steps * 0.1))

    hyperparams = Hyperparams(
        model=model_cfg.hf_id,
        epochs=request.epochs,
        batch_size=model_cfg.batch_size,
        grad_accum=model_cfg.grad_accum,
        effective_batch=model_cfg.batch_size * model_cfg.grad_accum,
        learning_rate=model_cfg.lr,
        warmup_steps=warmup_steps,
        precision=precision,
        train_docs=len(train_df),
        num_classes=num_labels,
    )
    return _HyperparamsBundle(
        hyperparams=hyperparams, use_bf16=use_bf16, use_fp16=use_fp16, warmup_steps=warmup_steps
    )


def _build_trainer(  # noqa: PLR0913 -- one param per input TrainingArguments/WeightedTrainer
    # need, all required; matches this file's existing precedent (_fit_svm_reviewer above)
    # for a single-caller helper where a wrapper NamedTuple would be more ceremony than the
    # limit is worth.
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_ds: BertTunningDataset,
    val_ds: BertTunningDataset,
    class_weights: torch.Tensor,
    model_cfg: ModelConfig,
    request: TrainingRequest,
    hp_bundle: _HyperparamsBundle,
    report_to: str,
) -> Trainer:
    args = TrainingArguments(
        output_dir=str(Path(request.output_dir) / "checkpoints"),
        num_train_epochs=request.epochs,
        per_device_train_batch_size=model_cfg.batch_size,
        per_device_eval_batch_size=model_cfg.batch_size,
        gradient_accumulation_steps=model_cfg.grad_accum,
        learning_rate=model_cfg.lr,
        warmup_steps=hp_bundle.warmup_steps,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=hp_bundle.use_bf16,
        fp16=hp_bundle.use_fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=10,
        report_to=report_to,
        seed=request.seed,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=request.early_stop_patience)],
    )
    log.info("Early stopping: patience=%d epochs on macro_f1", request.early_stop_patience)
    return trainer


class _AuxiliaryArtifacts(NamedTuple):
    ood_stats: OodArtifact
    svm_classifiers: dict[str, SVC]


def _generate_auxiliary_artifacts(  # noqa: PLR0913 -- one param per input this needs from
    # run(), all required; same rationale as _fit_svm_reviewer/_build_trainer above.
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    model_cfg: ModelConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    class_names: list[str],
    chunk_strategy: str,
    wb: WandbLogger,
) -> _AuxiliaryArtifacts:
    loaded = LoadedModel(model=model, tokenizer=tokenizer, device=str(model.device))
    train_embeddings = extract_embeddings(
        loaded,
        [prepare_text(t, tokenizer, chunk_strategy) for t in train_df["text"]],
        max_length=model_cfg.max_tokens,
    )
    ood_stats = compute_class_stats(
        train_embeddings,
        train_df["label_id"].tolist(),
        class_names,
        texts=train_df["text"].tolist(),
        n_components=Settings.OOD_PCA_COMPONENTS,
        model_type=model.config.model_type,
        model_hidden_size=model.config.hidden_size,
        max_tfidf_features=Settings.OOD_TFIDF_MAX_FEATURES,
        max_tfidf_max_df=Settings.OOD_TFIDF_MAX_DF,
    )
    log.info("Computed OOD stats from %d training embeddings", train_embeddings.shape[0])

    svm_classifiers = _fit_svm_reviewer(
        loaded, model_cfg, train_df, val_df, class_names, train_embeddings, wb
    )
    return _AuxiliaryArtifacts(ood_stats=ood_stats, svm_classifiers=svm_classifiers)


def _persist_artifacts(
    trainer: Trainer,
    tokenizer: PreTrainedTokenizerBase,
    ood_stats: OodArtifact,
    svm_classifiers: dict[str, SVC],
    save_path: Path,
) -> None:
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    save_stats(ood_stats, save_path / "ood_stats.npz")
    save_svm_classifiers(svm_classifiers, save_path / "svm_classifiers.joblib")
    log.info("Model saved to %s", save_path)


def run(
    df: pd.DataFrame,
    model_cfg: ModelConfig,
    request: TrainingRequest,
) -> tuple[Trainer, LabelEncoder]:
    log.info("=" * 60)
    log.info("BERT TUNNING — FINE-TUNING %s", model_cfg.hf_id)
    log.info("=" * 60)

    encoding = _encode_labels(df)
    log.info("%d classes: %s", encoding.num_labels, list(encoding.le.classes_))
    class_names = list(encoding.le.classes_)

    split = _split_and_weight(df, encoding.num_labels, class_names, request.seed)

    datasets = _build_datasets(
        split.train_df, split.val_df, split.test_df, model_cfg, request.chunk_strategy
    )

    model = _build_model(model_cfg, encoding.num_labels, encoding.id2label, encoding.label2id)

    hp_bundle = _compute_hyperparams(
        model_cfg, request, datasets.train_ds, split.train_df, encoding.num_labels
    )

    wb = WandbLogger(enabled=request.use_wandb)
    wb.init(hp_bundle.hyperparams)

    trainer = _build_trainer(
        model,
        datasets.tokenizer,
        datasets.train_ds,
        datasets.val_ds,
        split.class_weights,
        model_cfg,
        request,
        hp_bundle,
        wb.report_to,
    )
    log.info(
        "Training — %d epochs, effective batch %d",
        request.epochs,
        model_cfg.batch_size * model_cfg.grad_accum,
    )

    Path(request.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.train()
    log.info("Training complete")

    aux = _generate_auxiliary_artifacts(
        model,
        datasets.tokenizer,
        model_cfg,
        split.train_df,
        split.val_df,
        class_names,
        request.chunk_strategy,
        wb,
    )

    result = run_evaluation(trainer, datasets.test_ds, encoding.le, hp_bundle.hyperparams)
    wb.log_results(result, class_names)
    wb.finish()

    save_path = Path(request.output_dir) / "final"
    _persist_artifacts(trainer, datasets.tokenizer, aux.ood_stats, aux.svm_classifiers, save_path)

    return trainer, encoding.le
