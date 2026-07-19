from collections.abc import Iterator
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase


class LoadedModel(NamedTuple):
    """A trained model + its tokenizer + the device it's on — these three always travel
    together at every call site, so bundling them is what actually gets extract_embeddings
    under ruff's 5-argument limit, not a noqa."""

    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase
    device: str


def _batched_inputs(
    loaded: LoadedModel, texts: list[str], *, max_length: int, batch_size: int
) -> Iterator[BatchEncoding]:
    """Yields tokenized, device-moved inputs one batch at a time -- the identical
    batching/tokenization/device-transfer machinery both extract_embeddings and
    extract_embeddings_and_predictions need. Does NOT run the forward pass -- the two
    callers deliberately use different forward-pass strategies (see module-level note on
    why extract_embeddings skips the classification head) and must stay free to do so."""
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        yield loaded.tokenizer(
            batch,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        ).to(loaded.device)


def _cls_embedding(hidden_state: torch.Tensor) -> npt.NDArray[np.float64]:
    return hidden_state[:, 0, :].cpu().numpy().astype(np.float64)


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
        for inputs in _batched_inputs(loaded, texts, max_length=max_length, batch_size=batch_size):
            hidden = loaded.model.base_model(**inputs).last_hidden_state  # type: ignore[operator]
            batches.append(_cls_embedding(hidden))
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
        for inputs in _batched_inputs(loaded, texts, max_length=max_length, batch_size=batch_size):
            outputs = loaded.model(**inputs, output_hidden_states=True)
            embedding_batches.append(_cls_embedding(outputs.hidden_states[-1]))
            predicted_ids.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
    return np.vstack(embedding_batches), predicted_ids
