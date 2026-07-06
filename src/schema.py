"""Shared Pydantic model definitions used across the Bert Tunning pipeline."""

from typing import Annotated

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, BeforeValidator, ConfigDict
from pydantic.alias_generators import to_camel


def _as_float64_array(value: object) -> npt.NDArray[np.float64]:
    return np.asarray(value, dtype=np.float64)


# Coerces/validates to a float64 ndarray at construction time — arbitrary_types_allowed=True
# alone only checks isinstance(value, np.ndarray), silently accepting any dtype/shape.
Float64Array = Annotated[npt.NDArray[np.float64], BeforeValidator(_as_float64_array)]


class PredictResult(BaseModel):
    """Return value from BertTunningClassifier.predict_text and predict_pdf."""

    model_config = ConfigDict(alias_generator=to_camel, arbitrary_types_allowed=True, frozen=True)

    label: str | None = None
    confidence: float = 0.0
    certain: bool = False
    all_scores: dict[str, float] = {}
    filename: str = ""
    error: str = ""
    mahalanobis_z: float | None = None
    cosine_z: float | None = None
    in_distribution: bool | None = None


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
    pca_mean: Float64Array
    pca_components: Float64Array
    centroids: Float64Array
    covariance_inv: Float64Array
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
