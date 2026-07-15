import logging

import wandb
from src.ood import OodThresholds
from src.schema import (
    CalibrationReport,
    EvaluationResult,
    Hyperparams,
    PredictResult,
    flatten_predict_result,
)
from src.settings import Settings

log = logging.getLogger(__name__)

_PREDICTION_COLUMNS = [
    "filename",
    "label",
    "confidence",
    "certain",
    "mahalanobis_p_value",
    "mahalanobis_p_value_theoretical",
    "cosine_z",
    "knn_distance",
    "tfidf_cosine_z",
    "in_distribution",
    "review_route",
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
        row = flatten_predict_result(r)
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
    report: CalibrationReport,
    *,
    model_path: str,
    cache_path: str,
    target_fp_rate: float,
    thresholds: OodThresholds,
) -> None:
    """Log an evaluate-ood-calibration run's summary metrics to W&B.

    `thresholds` must be the resolved per-model OodThresholds (resolve_ood_thresholds(stats)),
    not Settings.OOD_* directly -- otherwise a W&B dashboard comparing calibration runs across
    models with different per-model thresholds shows the identical "current threshold" for
    every model, silently wrong once any model has calibrated values written via
    --write-thresholds. Also logs the k-NN threshold, which this function previously omitted
    entirely.
    """
    wandb.init(
        entity=Settings.WANDB_ENTITY,
        project=Settings.WANDB_PROJECT,
        job_type="ood-calibration",
        config={
            "model_path": model_path,
            "cache_path": cache_path,
            "target_fp_rate": target_fp_rate,
            "current_mahalanobis_threshold": thresholds.mahalanobis_p,
            "current_cosine_threshold": thresholds.cosine_z,
            "current_knn_threshold": thresholds.knn_distance,
            "current_tfidf_threshold": thresholds.tfidf_cosine_z,
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
            "ood/fp_rate_tfidf": report.fp_rate_tfidf,
            "ood/suggested_tfidf_threshold": report.suggested_tfidf_threshold,
        }
    )
    wandb.finish()
    log.info(
        "Logged OOD calibration results to W&B (%s/%s)",
        Settings.WANDB_ENTITY,
        Settings.WANDB_PROJECT,
    )
