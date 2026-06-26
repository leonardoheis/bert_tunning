import logging

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from config import (
    BATCH_SIZE,
    CHUNK_STRATEGY,
    EPOCHS,
    GRAD_ACCUM,
    LR,
    MAX_TOKENS,
    MODEL_NAME,
    OUTPUT_DIR,
    SEED,
)
from dataset import ClassiflowDataset, prepare_text

log = logging.getLogger(__name__)


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "accuracy": float((predictions == labels).mean()),
    }


def train(df: pd.DataFrame) -> tuple[Trainer, LabelEncoder]:
    log.info("=" * 60)
    log.info("CLASSIFLOW — DEBERTA FINE-TUNING")
    log.info("=" * 60)

    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])
    label2id   = {l: int(i) for i, l in enumerate(le.classes_)}
    id2label   = {int(i): l for l, i in label2id.items()}
    num_labels = len(le.classes_)
    log.info("%d classes: %s", num_labels, list(le.classes_))

    try:
        train_df, temp_df = train_test_split(
            df, test_size=0.30, stratify=df["label_id"], random_state=SEED
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, stratify=temp_df["label_id"], random_state=SEED
        )
    except ValueError as e:
        log.warning("Stratified split failed (%s) — using random split", e)
        train_df, temp_df = train_test_split(df, test_size=0.30, random_state=SEED)
        val_df, test_df   = train_test_split(temp_df, test_size=0.50, random_state=SEED)

    log.info("Split — train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))

    log.info("Loading tokenizer: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_texts = [prepare_text(t, tokenizer, CHUNK_STRATEGY) for t in train_df["text"]]
    val_texts   = [prepare_text(t, tokenizer, "first")        for t in val_df["text"]]
    test_texts  = [prepare_text(t, tokenizer, "first")        for t in test_df["text"]]

    train_ds = ClassiflowDataset(train_texts, train_df["label_id"].tolist(), tokenizer)
    val_ds   = ClassiflowDataset(val_texts,   val_df["label_id"].tolist(),   tokenizer)
    test_ds  = ClassiflowDataset(test_texts,  test_df["label_id"].tolist(),  tokenizer)

    log.info("Loading model: %s", MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    log.info("Mixed precision: %s", "bf16" if use_bf16 else "fp16" if use_fp16 else "none (CPU)")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=use_bf16,
        fp16=use_fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=SEED,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    log.info("Training started — %d epochs, effective batch %d", EPOCHS, BATCH_SIZE * GRAD_ACCUM)
    trainer.train()
    log.info("Training complete")

    log.info("Evaluating on held-out test set...")
    preds_out = trainer.predict(test_ds)
    y_pred = np.argmax(preds_out.predictions, axis=-1)
    y_true = test_df["label_id"].tolist()

    report = classification_report(y_true, y_pred, target_names=le.classes_, digits=3, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

    log.info("Per-class report:\n%s", report)
    log.info("Confusion matrix:\n%s", cm_df.to_string())

    save_path = f"{OUTPUT_DIR}/final"
    log.info("Saving model to %s", save_path)
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    log.info("Model saved successfully")

    return trainer, le
