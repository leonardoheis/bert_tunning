from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import torch
from transformers import PreTrainedTokenizerBase


class LoadedModel(NamedTuple):
    """A trained model + its tokenizer + the device it's on — these three always travel
    together at every call site, so bundling them is what actually gets extract_embeddings
    under ruff's 5-argument limit, not a noqa."""

    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase
    device: str


def select_device() -> str:
    """cuda -> mps -> cpu, in order of preference — mps covers Apple Silicon Macs, which
    have no CUDA support but do have a GPU worth using."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extract_embeddings(
    loaded: LoadedModel,
    texts: list[str],
    *,
    max_length: int,
    batch_size: int = 16,
) -> npt.NDArray[np.float64]:
    loaded.model.eval()
    batches: list[npt.NDArray[np.float64]] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = loaded.tokenizer(
                batch,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            ).to(loaded.device)
            hidden = loaded.model.base_model(**inputs).last_hidden_state  # type: ignore[operator]
            batches.append(hidden[:, 0, :].cpu().numpy().astype(np.float64))
    return np.vstack(batches)


def extract_embeddings_and_predictions(
    loaded: LoadedModel,
    texts: list[str],
    *,
    max_length: int,
    batch_size: int = 16,
) -> tuple[npt.NDArray[np.float64], list[int]]:
    """Like extract_embeddings, but also returns each document's predicted label id from
    the same forward pass -- for evaluate-ood-calibration, which must score the k-NN signal
    against the model's actual prediction (mirroring predict_text exactly), not the
    document's true label. extract_embeddings alone can't do this: it calls
    loaded.model.base_model(...) to skip the classification head entirely, since its other
    callers (compute-ood-stats, training) only ever need embeddings, never predictions."""
    loaded.model.eval()
    embedding_batches: list[npt.NDArray[np.float64]] = []
    predicted_ids: list[int] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = loaded.tokenizer(
                batch,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            ).to(loaded.device)
            outputs = loaded.model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            embedding_batches.append(hidden[:, 0, :].cpu().numpy().astype(np.float64))
            predicted_ids.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
    return np.vstack(embedding_batches), predicted_ids
