import logging
from typing import TYPE_CHECKING

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

from src.training.evaluate import run_evaluation
from src.training.models import ModelConfig
from src.training.split import make_split
from src.training.tokenize import BertTunningDataset, prepare_text
from src.training.trainer import WeightedTrainer, compute_metrics
from wandb_logger import WandbLogger

if TYPE_CHECKING:
    from src.schema import Hyperparams

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
    log.info("BERT TUNNING — FINE-TUNING %s", model_cfg.hf_id)
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
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * 0.1))

    hyperparams: Hyperparams = {
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
    log.info(
        "Training — %d epochs, effective batch %d",
        epochs,
        model_cfg.batch_size * model_cfg.grad_accum,
    )

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
