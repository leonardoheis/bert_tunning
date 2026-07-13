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

    model_config = ConfigDict(
        alias_generator=to_camel,
        arbitrary_types_allowed=True,
        frozen=True,
        populate_by_name=True,
    )

    label: str | None = None
    confidence: float = 0.0
    certain: bool = False
    all_scores: dict[str, float] = {}
    filename: str = ""
    error: str = ""
    mahalanobis_p_value: float | None = None
    mahalanobis_p_value_theoretical: float | None = None
    cosine_z: float | None = None
    knn_distance: float | None = None
    in_distribution: bool | None = None
    extracted_text: str = ""
    extractor_used: str = ""
    review_route: str = ""


class CalibrationReport(BaseModel):
    """Return value from build_calibration_report — empirical OOD threshold calibration."""

    model_config = ConfigDict(frozen=True)

    fp_rate_maha: float
    fp_rate_cosine: float
    fp_rate_knn: float
    suggested_maha_threshold: float
    suggested_cosine_threshold: float
    suggested_knn_threshold: float


class ExtractionMetadata(BaseModel):
    """Return value from extract_pdf_with_metadata — extracted text plus provenance."""

    model_config = ConfigDict(frozen=True)

    text: str | None
    extractor_used: str | None
    char_count: int


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
    """Per-class embedding centroids + shared covariance for Mahalanobis/cosine OOD scoring,
    plus the raw per-class training embeddings needed for k-NN local-density scoring."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    class_names: list[str]
    pca_mean: Float64Array
    pca_components: Float64Array
    centroids: Float64Array
    covariance_inv: Float64Array
    cosine_calibration_mean: float
    cosine_calibration_std: float
    knn_train_embeddings: Float64Array  # (n_train_docs, n_components), PCA-reduced
    knn_train_labels: list[int]  # length n_train_docs, parallel to knn_train_embeddings
    # Per-model calibrated thresholds -- written by `evaluate-ood-calibration --write-thresholds`
    # (src/cli/ood_calibration.py), read via resolve_ood_thresholds() (src/ood.py). None means
    # "not yet calibrated for this specific model" -- resolve_ood_thresholds() falls back to
    # Settings.OOD_* in that case. Fixes thresholds calibrated for one model (e.g. BETO v2)
    # being silently applied to a different model's differently-scaled embedding space.
    mahalanobis_p_threshold: float | None = None
    cosine_threshold: float | None = None
    knn_distance_threshold: float | None = None
    # Coarse per-model identity fingerprint -- written by compute_class_stats() from the
    # model that produced the embeddings, validated at classifier construction in
    # BertTunningClassifier._validate_ood_stats_model_identity(). class_names alone can't
    # distinguish two different model architectures trained on the same corpus/label set
    # (true for every model in this project's registry). None means "predates this field" --
    # the identity check is skipped entirely, not enforced as absent.
    model_type: str | None = None
    model_hidden_size: int | None = None


class Hyperparams(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        arbitrary_types_allowed=True,
        frozen=True,
        populate_by_name=True,
    )

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
