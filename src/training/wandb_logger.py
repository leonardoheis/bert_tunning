import logging

import wandb

from src.schema import EvaluationResult, Hyperparams
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

    def log_results(self, result: EvaluationResult, class_names: list[str]) -> None:
        if not self._enabled:
            return
        wandb.log(
            {
                "test/macro_f1": result.macro_f1,
                "test/accuracy": result.accuracy,
                "confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=result.y_true,
                    preds=result.y_pred.tolist(),
                    class_names=class_names,
                ),
            }
        )

    def finish(self) -> None:
        if not self._enabled:
            return
        wandb.finish()
