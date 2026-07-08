import logging

import wandb
from src.schema import CalibrationReport, EvaluationResult, Hyperparams, PredictResult
from src.settings import Settings

log = logging.getLogger(__name__)

_PREDICTION_COLUMNS = [
    "filename",
    "label",
    "confidence",
    "certain",
    "mahalanobis_p_value",
    "cosine_z",
    "knn_distance",
    "in_distribution",
    "extractor_used",
    "error",
]


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


def log_predict_folder_results(
    results: list[PredictResult], *, model_path: str, folder_path: str
) -> None:
    """Log a predict-folder run's per-document predictions as a W&B Table."""
    wandb.init(
        entity=Settings.WANDB_ENTITY,
        project=Settings.WANDB_PROJECT,
        job_type="predict-folder",
        config={"model_path": model_path, "folder_path": folder_path},
    )
    table = wandb.Table(columns=_PREDICTION_COLUMNS)
    for r in results:
        row = r.model_dump()
        table.add_data(*(row[col] for col in _PREDICTION_COLUMNS))
    wandb.log({"predictions": table})
    wandb.finish()
    log.info(
        "Logged %d predictions to W&B (%s/%s)",
        len(results),
        Settings.WANDB_ENTITY,
        Settings.WANDB_PROJECT,
    )


def log_ood_calibration_results(
    report: CalibrationReport, *, model_path: str, cache_path: str, target_fp_rate: float
) -> None:
    """Log an evaluate-ood-calibration run's summary metrics to W&B."""
    wandb.init(
        entity=Settings.WANDB_ENTITY,
        project=Settings.WANDB_PROJECT,
        job_type="ood-calibration",
        config={
            "model_path": model_path,
            "cache_path": cache_path,
            "target_fp_rate": target_fp_rate,
            "current_mahalanobis_threshold": Settings.OOD_MAHALANOBIS_P_THRESHOLD,
            "current_cosine_threshold": Settings.OOD_COSINE_THRESHOLD,
        },
    )
    wandb.log(
        {
            "ood/fp_rate_mahalanobis": report.fp_rate_maha,
            "ood/fp_rate_cosine": report.fp_rate_cosine,
            "ood/suggested_mahalanobis_threshold": report.suggested_maha_threshold,
            "ood/suggested_cosine_threshold": report.suggested_cosine_threshold,
            "ood/fp_rate_knn": report.fp_rate_knn,
            "ood/suggested_knn_threshold": report.suggested_knn_threshold,
        }
    )
    wandb.finish()
    log.info(
        "Logged OOD calibration results to W&B (%s/%s)",
        Settings.WANDB_ENTITY,
        Settings.WANDB_PROJECT,
    )
