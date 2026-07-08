import click
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.ood import LoadedModel


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
