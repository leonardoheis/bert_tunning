"""Shared Pydantic model definitions used across the Bert Tunning pipeline."""

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class PredictResult(BaseModel):
    """Return value from BertTunningClassifier.predict_text and predict_pdf."""

    model_config = ConfigDict(alias_generator=to_camel, arbitrary_types_allowed=True, frozen=True)

    label: str | None = None
    confidence: float = 0.0
    certain: bool = False
    all_scores: dict[str, float] = {}
    filename: str = ""
    error: str = ""


# classification_report(output_dict=True) returns per-class dicts and scalar floats.
ReportDict = dict[str, "dict[str, float] | float"]


class EvaluationResult(BaseModel):
    """Return value from run_evaluation — carries the raw report plus derived scalars."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    report_dict: ReportDict
    y_pred: npt.NDArray[np.int_]
    y_true: list[int]

    @property
    def macro_f1(self) -> float:
        macro_raw = self.report_dict.get("macro avg", {})
        return float(macro_raw["f1-score"]) if isinstance(macro_raw, dict) else 0.0

    @property
    def accuracy(self) -> float:
        accuracy_raw = self.report_dict.get("accuracy", 0.0)
        return float(accuracy_raw) if isinstance(accuracy_raw, float) else 0.0


class ClassEmbeddingStats(BaseModel):
    """Per-class embedding centroids + shared covariance for Mahalanobis/cosine OOD scoring."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    class_names: list[str]
    pca_mean: npt.NDArray[np.float64]
    pca_components: npt.NDArray[np.float64]
    centroids: npt.NDArray[np.float64]
    covariance_inv: npt.NDArray[np.float64]
    maha_calibration_mean: float
    maha_calibration_std: float
    cosine_calibration_mean: float
    cosine_calibration_std: float


class Hyperparams(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, arbitrary_types_allowed=True, frozen=True)

    model: str
    epochs: int
    batch_size: int
    grad_accum: int
    effective_batch: int
    learning_rate: float
    warmup_steps: int
    precision: str
    train_docs: int
    num_classes: int
