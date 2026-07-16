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
    "foreign_municipality",
    "foreign_municipality_context",
    "review_route",
    "extractor_used",
    "error",
    "svm_scores",
    "svm_predicted_label",
    "svm_agrees_with_prediction",
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

    def log_svm_results(
        self, svm_val_accuracy: dict[str, float], train_class_counts: dict[str, int]
    ) -> None:
        """Logs into the training run already opened by init() -- unlike
        log_svm_classifiers_results below, which is compute-svm-classifiers' standalone,
        one-shot equivalent (its own init/finish, no prior training run to attach to)."""
        if not self._enabled:
            return
        wandb.log(_svm_results_payload(svm_val_accuracy, train_class_counts))

    def finish(self) -> None:
        if not self._enabled:
            return
        wandb.finish()


def _svm_results_payload(
    svm_val_accuracy: dict[str, float], train_class_counts: dict[str, int]
) -> dict[str, object]:
    """Shared by WandbLogger.log_svm_results (a training run's already-open W&B run) and
    log_svm_classifiers_results (compute-svm-classifiers' own one-shot run) -- same
    per-class table + scalar metrics either way, so a dashboard comparing a `train` run
    against a `compute-svm-classifiers` backfill run for the same model sees identical
    shapes. train_class_counts sits alongside the accuracy in the same table because a low
    accuracy is only explainable next to how many training examples that class actually
    had -- an unexplained number invites the wrong conclusion (e.g. "the SVM is bad at this
    class") when the real story is "this class had 37 training documents."""
    table = wandb.Table(columns=["class", "train_samples", "held_out_balanced_accuracy"])
    for class_name, accuracy in svm_val_accuracy.items():
        table.add_data(class_name, train_class_counts[class_name], accuracy)
    payload: dict[str, object] = {
        "svm/per_class_accuracy": table,
        "svm/mean_balanced_accuracy": sum(svm_val_accuracy.values()) / len(svm_val_accuracy),
        "svm/min_balanced_accuracy": min(svm_val_accuracy.values()),
    }
    payload.update({f"svm/balanced_accuracy/{name}": acc for name, acc in svm_val_accuracy.items()})
    return payload


def log_svm_classifiers_results(
    *,
    model_path: str,
    cache_path: str,
    model_key: str,
    svm_val_accuracy: dict[str, float],
    train_class_counts: dict[str, int],
) -> None:
    """Log a compute-svm-classifiers backfill run's per-class held-out balanced accuracy
    (and training sample counts, for context) to W&B."""
    wandb.init(
        entity=Settings.WANDB_ENTITY,
        project=Settings.WANDB_PROJECT,
        job_type="compute-svm-classifiers",
        config={"model_path": model_path, "cache_path": cache_path, "model_key": model_key},
    )
    wandb.log(_svm_results_payload(svm_val_accuracy, train_class_counts))
    wandb.finish()
    log.info(
        "Logged SVM reviewer results to W&B (%s/%s)",
        Settings.WANDB_ENTITY,
        Settings.WANDB_PROJECT,
    )


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
