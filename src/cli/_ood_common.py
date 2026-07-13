from typing import NamedTuple

import click
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.embeddings import LoadedModel, extract_embeddings, extract_embeddings_and_predictions
from src.training.split import make_split
from src.training.tokenize import prepare_text


def load_model_and_verify_classes(model_path: str, cache_labels: set[str]) -> LoadedModel:
    """Load a trained model + tokenizer for an OOD command, on the right device, and verify
    its classes match the cache the caller is about to reconstruct a split from — shared by
    compute-ood-stats and evaluate-ood-calibration, which both need exactly this."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    model.to(device)

    model_labels = set(model.config.id2label.values())
    if model_labels != cache_labels:
        msg = (
            f"Cache classes {sorted(cache_labels)} do not match model classes "
            f"{sorted(model_labels)} — wrong --cache-path or --model-path?"
        )
        raise click.ClickException(msg)

    return LoadedModel(model=model, tokenizer=tokenizer, device=device)


class ReconstructedSplit(NamedTuple):
    """The loaded model plus the exact train/val/test split reconstruction shared by
    compute-ood-stats (uses train_df) and evaluate-ood-calibration (uses test_df) — one
    place defining "the split" so both commands can't quietly diverge on what it means."""

    loaded: LoadedModel
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    classes: list[str]


def reconstruct_split_and_load_model(
    *, model_path: str, cache_path: str, seed: int
) -> ReconstructedSplit:
    df = pd.read_parquet(cache_path)
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])
    train_df, val_df, test_df = make_split(df, seed=seed)
    loaded = load_model_and_verify_classes(model_path, set(le.classes_))
    return ReconstructedSplit(loaded, train_df, val_df, test_df, list(le.classes_))


def embed_texts(
    loaded: LoadedModel, df: pd.DataFrame, *, chunk_strategy: str, max_tokens: int
) -> npt.NDArray[np.float64]:
    texts = [prepare_text(t, loaded.tokenizer, chunk_strategy) for t in df["text"]]
    return extract_embeddings(loaded, texts, max_length=max_tokens)


def embed_texts_and_predict(
    loaded: LoadedModel, df: pd.DataFrame, *, chunk_strategy: str, max_tokens: int
) -> tuple[npt.NDArray[np.float64], list[int]]:
    """Sibling to embed_texts() for callers that also need each document's predicted label
    -- currently only evaluate-ood-calibration, for reproducing predict_text()'s exact k-NN
    scoring input (the model's own prediction, not the document's true label)."""
    texts = [prepare_text(t, loaded.tokenizer, chunk_strategy) for t in df["text"]]
    return extract_embeddings_and_predictions(loaded, texts, max_length=max_tokens)
