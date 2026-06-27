import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import wandb

from config import WANDB_ENTITY, WANDB_PROJECT

log = logging.getLogger(__name__)


class WandbLogger:
    """Encapsulates all Weights & Biases interactions for a training run."""

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._entity = WANDB_ENTITY
        self._project = WANDB_PROJECT

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def report_to(self) -> str:
        return "wandb" if self._enabled else "none"

    def init(self, hyperparams: dict[str, Any]) -> None:
        if not self._enabled:
            return
        wandb.init(entity=self._entity, project=self._project, config=hyperparams)
        log.info("W&B run started: %s/%s", self._entity, self._project)

    def log_results(
        self,
        report_dict: dict[str, Any],
        y_true: list[int],
        y_pred: npt.NDArray[np.int_],
        class_names: list[str],
    ) -> None:
        if not self._enabled:
            return
        wandb.log(
            {
                "test/macro_f1": report_dict["macro avg"]["f1-score"],
                "test/accuracy": report_dict["accuracy"],
                "confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=y_true,
                    preds=y_pred.tolist(),
                    class_names=class_names,
                ),
            }
        )

    def finish(self) -> None:
        if not self._enabled:
            return
        wandb.finish()
