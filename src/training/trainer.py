from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import Trainer


class WeightedTrainer(Trainer):
    """Trainer subclass that applies per-class weights to cross-entropy loss."""

    def __init__(
        self, *args: Any, class_weights: torch.Tensor | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(  # type: ignore[override]
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,  # noqa: FBT001, FBT002
        num_items_in_batch: int | None = None,
        **_kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        labels = inputs.pop("labels", None)
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None and labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weights.to(device=logits.device, dtype=logits.dtype)
            )
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))  # type: ignore[union-attr]
        else:
            loss = outputs.loss

        if num_items_in_batch is not None and self.args.gradient_accumulation_steps > 1:
            loss = loss * logits.shape[0] / num_items_in_batch

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred: tuple[Any, Any]) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "accuracy": float((predictions == labels).mean()),
    }
