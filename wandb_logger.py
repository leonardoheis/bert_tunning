import logging

import numpy as np
import numpy.typing as npt

import wandb
from src.schema import Hyperparams, ReportDict
from src.settings import Settings

log = logging.getLogger(__name__)


class WandbLogger:
    """Encapsulates all Weights & Biases interactions for a training run."""

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._entity = Settings.WANDB_ENTITY
        self._project = Settings.WANDB_PROJECT

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def report_to(self) -> str:
        return "wandb" if self._enabled else "none"

    def init(self, hyperparams: Hyperparams) -> None:
        if not self._enabled:
            return
        wandb.init(entity=self._entity, project=self._project, config=hyperparams.model_dump())
        log.info("W&B run started: %s/%s", self._entity, self._project)

    def log_results(
        self,
        report_dict: ReportDict,
        y_true: list[int],
        y_pred: npt.NDArray[np.int_],
        class_names: list[str],
    ) -> None:
        if not self._enabled:
            return
        macro_raw = report_dict.get("macro avg", {})
        macro_f1 = float(macro_raw["f1-score"]) if isinstance(macro_raw, dict) else 0.0
        accuracy_raw = report_dict.get("accuracy", 0.0)
        accuracy = float(accuracy_raw) if isinstance(accuracy_raw, float) else 0.0
        wandb.log(
            {
                "test/macro_f1": macro_f1,
                "test/accuracy": accuracy,
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
