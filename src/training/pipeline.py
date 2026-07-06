import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from src.inference.ood import compute_class_stats, extract_embeddings, save_stats
from src.schema import Hyperparams
from src.settings import Settings
from src.training.evaluate import run_evaluation
from src.training.models import ModelConfig
from src.training.options import TrainingRequest
from src.training.split import make_split
from src.training.tokenize import BertTunningDataset, prepare_text
from src.training.trainer import WeightedTrainer, compute_metrics
from src.training.wandb_logger import WandbLogger

log = logging.getLogger(__name__)


def run(
    df: pd.DataFrame,
    model_cfg: ModelConfig,
    request: TrainingRequest,
) -> tuple[Trainer, LabelEncoder]:
    log.info("=" * 60)
    log.info("BERT TUNNING — FINE-TUNING %s", model_cfg.hf_id)
    log.info("=" * 60)

    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])
    label2id = {cls: int(i) for i, cls in enumerate(le.classes_)}
    id2label = {int(i): cls for cls, i in label2id.items()}
    num_labels = len(le.classes_)
    log.info("%d classes: %s", num_labels, list(le.classes_))

    train_df, val_df, test_df = make_split(df, seed=request.seed)

    raw_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_labels),
        y=train_df["label_id"].to_numpy(),
    )
    class_weights = torch.tensor(raw_weights, dtype=torch.float)
    log.info("Class weights: %s", dict(zip(le.classes_, raw_weights.round(3), strict=True)))

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_cfg.hf_id)

    def _texts(split_df: pd.DataFrame, strategy: str) -> list[str]:
        return [prepare_text(t, tokenizer, strategy) for t in split_df["text"]]

    train_ds = BertTunningDataset(
        _texts(train_df, request.chunk_strategy),
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

    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg.hf_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

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

    wb = WandbLogger(enabled=request.use_wandb)
    wb.init(hyperparams)

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
    log.info(
        "Training — %d epochs, effective batch %d",
        request.epochs,
        model_cfg.batch_size * model_cfg.grad_accum,
    )

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
