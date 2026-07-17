"""Shared Pydantic model definitions used across the Bert Tunning pipeline."""

from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from pydantic.alias_generators import to_camel


def _as_float64_array(value: object) -> npt.NDArray[np.float64]:
    return np.asarray(value, dtype=np.float64)


# Coerces/validates to a float64 ndarray at construction time — arbitrary_types_allowed=True
# alone only checks isinstance(value, np.ndarray), silently accepting any dtype/shape.
Float64Array = Annotated[npt.NDArray[np.float64], BeforeValidator(_as_float64_array)]


class OodMetrics(BaseModel):
    """Out-of-distribution scoring results — only present when a model has ood_stats.npz
    loaded. Nested on PredictResult (rather than five flat Optional fields) so that
    PredictResult.ood_metrics is None means exactly one thing — no ood_stats.npz loaded
    for this model — and tfidf_cosine_z's own None means exactly one different thing —
    this specific stats file predates the TF-IDF signal — instead of both reasons
    colliding on the same flat None. See
    docs/superpowers/specs/2026-07-15-ood-metrics-nesting-design.md for the full
    rationale (found via a /stop-using-none audit)."""

    model_config = ConfigDict(alias_generator=to_camel, frozen=True, populate_by_name=True)

    mahalanobis_p_value: float
    mahalanobis_p_value_theoretical: float
    cosine_z: float
    knn_distance: float
    tfidf_cosine_z: float | None = None
    in_distribution: bool


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
    ood_metrics: OodMetrics | None = None
    extracted_text: str = ""
    extractor_used: str = ""
    review_route: str = ""
    foreign_municipality: str | None = None
    foreign_municipality_context: str | None = None
    # Per-class one-vs-rest SVM decision-function margins, independent of ood_metrics --
    # not an OOD signal, evidence about per-class membership for the downstream Classiflow
    # agent. Empty dict, not None, when svm_classifiers.joblib isn't present next to the
    # loaded model -- "nothing here" has a natural empty-collection representation, same as
    # all_scores above. See docs/superpowers/specs/2026-07-15-svm-independent-reviewer-design.md.
    svm_scores: dict[str, float] = {}
    # The SVM reviewer's own top pick (max-margin class) -- "" only when svm_scores itself
    # is empty (no svm_classifiers.joblib); a real class name is never empty, so "" cannot
    # collide with genuine data. svm_agrees_with_prediction is a plain bool, not bool | None:
    # "was there SVM evidence at all" already has an owner (this field being ""), so the
    # agreement flag doesn't need to also carry that meaning -- it defaults to True (no
    # disagreement) when there's no SVM signal, the same permissive-default-on-missing-
    # artifact pattern as OodEvidence.from_in_distribution(None). See
    # docs/superpowers/specs/2026-07-16-svm-softmax-disagreement-design.md.
    svm_predicted_label: str = ""
    svm_agrees_with_prediction: bool = True


_OOD_METRIC_FIELDS = (
    "mahalanobis_p_value",
    "mahalanobis_p_value_theoretical",
    "cosine_z",
    "knn_distance",
    "tfidf_cosine_z",
    "in_distribution",
)


def flatten_predict_result(result: PredictResult) -> dict[str, object]:
    """Flattens PredictResult.ood_metrics back into individual top-level keys, for
    consumers that need one flat row per prediction (the predict-folder CSV, the W&B
    predictions table) rather than a nested object -- pandas/wandb.Table don't
    recursively flatten a nested dict column/cell, so without this every OOD score would
    collapse into one unreadable stringified-dict value. None-fills every OOD field when
    ood_metrics itself is None (no ood_stats.npz loaded), matching the same shape a
    caller would have seen from the flat fields this replaced."""
    row = result.model_dump(exclude={"ood_metrics"})
    metrics = result.ood_metrics.model_dump() if result.ood_metrics is not None else {}
    for field in _OOD_METRIC_FIELDS:
        row[field] = metrics.get(field)
    return row


class CalibrationReport(BaseModel):
    """Return value from build_calibration_report — empirical OOD threshold calibration."""

    model_config = ConfigDict(frozen=True)

    fp_rate_maha: float
    fp_rate_cosine: float
    fp_rate_knn: float
    suggested_maha_threshold: float
    suggested_cosine_threshold: float
    suggested_knn_threshold: float
    fp_rate_tfidf: float = 0.0
    suggested_tfidf_threshold: float = 0.0


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


class EmbeddingStats(BaseModel):
    """Per-class embedding centroids + shared covariance for Mahalanobis/cosine OOD scoring,
    plus the raw per-class training embeddings needed for k-NN local-density scoring. One
    of OodArtifact's sections -- see
    docs/superpowers/specs/2026-07-16-ood-artifact-schema-versioning-design.md."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    pca_mean: Float64Array
    pca_components: Float64Array
    centroids: Float64Array
    covariance_inv: Float64Array
    cosine_calibration_mean: float
    cosine_calibration_std: float
    knn_train_embeddings: Float64Array  # (n_train_docs, n_components), PCA-reduced
    knn_train_labels: list[int]  # length n_train_docs, parallel to knn_train_embeddings


class LexicalStats(BaseModel):
    """TF-IDF cosine-centroid signal -- a fourth OOD signal, independent of the
    embedding-space ones above, operating on raw lexical vocabulary instead. Catches
    divergence the embedding-based signals structurally cannot (e.g. a document naming a
    different municipality but sharing the same document-type shape). Always present
    (never Optional) -- an empty vocabulary means "this artifact predates the TF-IDF
    signal" or "not yet fitted," an unambiguous sentinel since a real fitted vectorizer
    always has >=1 term (see stop-using-none's "Nothing here" case). is_fitted() replaces
    scattered `if not stats.tfidf_vocabulary_terms` checks with one named predicate."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    vocabulary_terms: list[str] = []  # ordered; index = TF-IDF feature id
    idf: Float64Array = Field(default_factory=lambda: np.zeros(0))
    centroids: Float64Array = Field(default_factory=lambda: np.zeros((0, 0)))
    # Plain floats (0.0/1.0 sentinels), NOT Optional, unlike CalibratedThresholds.tfidf_cosine
    # -- always produced together with vocabulary_terms/idf/centroids in one
    # compute_tfidf_stats() call, never independently set or checked, so a None here would
    # just be a second, redundant way to express what vocabulary_terms' emptiness already
    # expresses. tfidf_cosine (on CalibratedThresholds) differs because it's set
    # independently, later, by a separate calibration step.
    cosine_calibration_mean: float = 0.0
    cosine_calibration_std: float = 1.0  # never 0.0 -- avoids a divide-by-zero if ever read
    # ungated, though every real caller already gates on is_fitted()

    def is_fitted(self) -> bool:
        return bool(self.vocabulary_terms)


class CalibratedThresholds(BaseModel):
    """Per-model calibrated OOD thresholds -- written by
    `evaluate-ood-calibration --write-thresholds` (src/cli/ood_calibration.py), read via
    resolve_ood_thresholds() (src/ood.py). Each field is independently Optional, because
    each threshold is calibrated independently -- the degenerate-threshold guard can
    refuse to write mahalanobis_p while cosine/knn_distance still get written in the same
    run, so this section can't collapse to one Optional the way ArtifactMetadata does.
    None means "not yet calibrated for this specific model" -- resolve_ood_thresholds()
    falls back to Settings.OOD_* in that case. Fixes thresholds calibrated for one model
    (e.g. BETO v2) being silently applied to a different model's differently-scaled
    embedding space."""

    model_config = ConfigDict(frozen=True)

    mahalanobis_p: float | None = None
    cosine: float | None = None
    knn_distance: float | None = None
    tfidf_cosine: float | None = None
    # Distinguishes *why* mahalanobis_p is None -- "not_calibrated" (nobody has run
    # evaluate-ood-calibration --write-thresholds yet, an operator should) from
    # "refused_degenerate" (it WAS run, but the degenerate-threshold guard in
    # cli/ood_calibration.py correctly refused to persist a floor-adjacent value --
    # expected, no action needed, will keep recurring). Without this, both states collapse
    # to the same None and BertTunningClassifier's startup warning can't tell them apart --
    # see OodScorer.warn_if_uncalibrated. cosine/knn_distance don't need an equivalent
    # field: the degenerate guard only ever applies to the Mahalanobis threshold.
    mahalanobis_status: Literal["not_calibrated", "calibrated", "refused_degenerate"] = (
        "not_calibrated"
    )


class ArtifactMetadata(BaseModel):
    """Coarse per-model identity fingerprint -- written by compute_class_stats() from the
    model that produced the embeddings, validated at classifier construction in
    OodScorer._validate_model_identity(). class_names alone can't distinguish two
    different model architectures trained on the same corpus/label set (true for every
    model in this project's registry). Unlike CalibratedThresholds, model_type and
    model_hidden_size are always set or unset TOGETHER (compute_class_stats() passes both
    or neither), so the WHOLE section is Optional on OodArtifact (None = "predates
    identity fingerprinting, skip the check entirely") rather than two independently-
    nullable fields that are only ever used as a pair."""

    model_config = ConfigDict(frozen=True)

    model_type: str
    model_hidden_size: int


class OodArtifact(BaseModel):
    """Replaces the former flat ClassEmbeddingStats -- one artifact, four independently-
    evolvable sections, one shared class taxonomy. Adding a fifth signal type means adding
    one new section class + a save/load pair for it, not editing this class or the
    monolithic save_stats()/load_stats() functions that used to hold every field's
    backward-compatibility branch inline. See
    docs/superpowers/specs/2026-07-16-ood-artifact-schema-versioning-design.md."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    format_version: int
    class_names: list[str]
    embedding: EmbeddingStats
    lexical: LexicalStats = LexicalStats()
    thresholds: CalibratedThresholds = CalibratedThresholds()
    metadata: ArtifactMetadata | None = None


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
