import logging
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import numpy.typing as npt
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerBase

if TYPE_CHECKING:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import SVC

from src.exceptions import BertTunningError
from src.ingestion.extract import clean_text
from src.ood import (
    OodThresholds as OodThresholds,  # noqa: PLC0414 -- explicit re-export: mypy strict's
)
from src.ood import (
    # no_implicit_reexport otherwise blocks tests from importing OodThresholds via this module
    build_tfidf_vectorizer,
    compute_train_mahalanobis_distances,
    cosine_z_score,
    empirical_survival_p_value,
    knn_mean_distance,
    load_stats,
    mahalanobis_chi2_p_value_from_distance,
    mahalanobis_min_distance,
    resolve_ood_thresholds,
    tfidf_cosine_z_score,
)
from src.schema import ClassEmbeddingStats, OodMetrics, PredictResult
from src.settings import Settings
from src.svm_reviewer import load_svm_classifiers, svm_top_label
from src.svm_reviewer import svm_scores as compute_svm_scores

log = logging.getLogger(__name__)


class ConfidenceTier(Enum):
    """Named alternative to a bare `certain: bool` at the decide_review_route boundary."""

    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"

    @classmethod
    def from_confidence(cls, confidence: float, threshold: float) -> "ConfidenceTier":
        return cls.CONFIDENT if confidence >= threshold else cls.UNCERTAIN


class OodEvidence(Enum):
    """Named tri-state for `in_distribution`, so "no ood_stats.npz loaded" reads as an
    explicit state rather than a bare `None` a caller has to know to interpret."""

    NOT_ANOMALOUS = "not_anomalous"  # in_distribution=True, or no ood_stats.npz loaded
    ANOMALOUS = "anomalous"  # in_distribution=False

    @classmethod
    def from_in_distribution(cls, *, in_distribution: bool | None) -> "OodEvidence":
        return cls.ANOMALOUS if in_distribution is False else cls.NOT_ANOMALOUS


class OodScores(NamedTuple):
    """The four OOD signals -- always computed together, passed together, never used
    independently. tfidf_cosine_z uses the same NaN-sentinel convention as knn_distance,
    not Optional -- keeps this NamedTuple a uniform tuple of plain floats. The two NaNs
    mean different things and are handled with OPPOSITE polarity in is_out_of_distribution:
    knn_distance's NaN means "this document's predicted class has zero training points"
    and fails CLOSED (anomalous); tfidf_cosine_z's NaN means "this whole model's
    ood_stats.npz predates the TF-IDF signal" and fails OPEN (not anomalous, same as
    _ood_stats being None disables OOD scoring entirely for the other three signals)."""

    mahalanobis_p: float
    cosine_z: float
    knn_distance: float
    tfidf_cosine_z: float = float("nan")


def is_out_of_distribution(scores: OodScores, thresholds: OodThresholds) -> bool:
    """Any one of the four OOD signals firing is enough -- a deliberate OR, not a
    weighted blend (see README's "OOD scoring internals" for why). NaN in knn_distance
    means the predicted class had zero training points to compare against; treated as
    anomalous, fail-safe, since `nan > threshold` would otherwise silently pass. NaN in
    tfidf_cosine_z means this model's ood_stats.npz predates the TF-IDF signal -- treated
    as NOT anomalous, fail-open, the opposite polarity, since there the signal simply
    doesn't exist for this model rather than having failed to compute for this document.
    `thresholds` comes from resolve_ood_thresholds(stats) -- per-model calibrated values
    when available, Settings.OOD_* fallback otherwise. Never reads Settings directly here,
    or a model's decisions silently use whichever thresholds happen to be configured for a
    completely different model.
    """
    maha_anomalous = scores.mahalanobis_p < thresholds.mahalanobis_p
    cosine_anomalous = scores.cosine_z > thresholds.cosine_z
    knn_anomalous = (
        bool(np.isnan(scores.knn_distance)) or scores.knn_distance > thresholds.knn_distance
    )
    # Opposite of knn_anomalous's NaN handling on purpose -- see OodScores' docstring.
    tfidf_anomalous = (
        not np.isnan(scores.tfidf_cosine_z) and scores.tfidf_cosine_z > thresholds.tfidf_cosine_z
    )
    log.debug(
        "OOD signals: mahalanobis_p=%.6f (threshold=%.6f, anomalous=%s), "
        "cosine_z=%.4f (threshold=%.4f, anomalous=%s), "
        "knn_distance=%.4f (threshold=%.4f, anomalous=%s), "
        "tfidf_cosine_z=%s (threshold=%.4f, anomalous=%s)",
        scores.mahalanobis_p,
        thresholds.mahalanobis_p,
        maha_anomalous,
        scores.cosine_z,
        thresholds.cosine_z,
        cosine_anomalous,
        scores.knn_distance,
        thresholds.knn_distance,
        knn_anomalous,
        scores.tfidf_cosine_z,
        thresholds.tfidf_cosine_z,
        tfidf_anomalous,
    )
    return maha_anomalous or cosine_anomalous or knn_anomalous or tfidf_anomalous


def decide_review_route(
    *,
    confidence_tier: ConfidenceTier,
    ood_evidence: OodEvidence,
    classifier_disagreement: bool = False,
) -> str:
    """Route a prediction to "accept", "llm_judge", or "human_review".

    An OOD signal firing (ANOMALOUS) always wins and routes to a human -- an LLM judge
    can't be trusted to catch what already fooled the classifier itself. A classifier
    disagreement (the SVM reviewer's top pick doesn't match softmax's) is a second,
    independent trigger for the same lane -- a different failure mode than OOD (the
    document IS a known class, the two classifiers just disagree which one), but the
    same "route to a human regardless of confidence" rationale applies: a confident-but-
    wrong prediction is the dangerous case either way. Otherwise the softmax confidence
    tier alone decides: confident predictions are accepted, uncertain ones get a cheap
    LLM-judge second opinion. See "Review routing" in README.md for the full decision
    table and rationale.
    """
    if ood_evidence is OodEvidence.ANOMALOUS or classifier_disagreement:
        return "human_review"
    return "accept" if confidence_tier is ConfidenceTier.CONFIDENT else "llm_judge"


class BertTunningClassifier:
    def __init__(
        self,
        model_path: str,
        *,
        confidence_threshold: float = 0.70,
        tokenizer: PreTrainedTokenizerBase | None = None,
        model: torch.nn.Module | None = None,
    ) -> None:
        log.info("Loading classifier from %s", model_path)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_path)
        self.model: Any = model or AutoModelForSequenceClassification.from_pretrained(model_path)
        self.threshold = confidence_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        self.model.to(self.device)
        self.max_length = min(  # type: ignore[type-var]
            self.tokenizer.model_max_length,
            self.model.config.max_position_embeddings,  # type: ignore[union-attr]
        )
        self._ood_stats = self._load_ood_stats(model_path)
        self._validate_ood_stats_class_mapping()
        self._validate_ood_stats_model_identity()
        self._warn_on_uncalibrated_thresholds()
        self._svm_classifiers = self._load_svm_classifiers(model_path)
        self._validate_svm_classifiers_class_mapping()
        log.info("Classifier ready on %s (max_length=%d)", self.device, self.max_length)

    @staticmethod
    def _load_ood_stats(model_path: str) -> ClassEmbeddingStats | None:
        stats_path = Path(model_path) / "ood_stats.npz"
        if not stats_path.exists():
            log.info("No ood_stats.npz found at %s — OOD scoring disabled", stats_path)
            return None
        log.info("Loaded OOD stats from %s", stats_path)
        return load_stats(stats_path)

    @staticmethod
    def _load_svm_classifiers(model_path: str) -> "dict[str, SVC] | None":
        classifiers_path = Path(model_path) / "svm_classifiers.joblib"
        classifiers = load_svm_classifiers(classifiers_path)
        if classifiers is None:
            log.info(
                "No svm_classifiers.joblib found at %s — SVM reviewer disabled",
                classifiers_path,
            )
            return None
        log.info("Loaded SVM reviewer classifiers from %s", classifiers_path)
        return classifiers

    def _validate_ood_stats_class_mapping(self) -> None:
        """ood_stats.npz's class_names must match this model's id2label -- by count AND by
        ordered index, since knn_mean_distance() indexes stats.knn_train_labels directly by
        the model's own predicted label id (see predict_text). A silently mismatched or
        stale ood_stats.npz would score every prediction's k-NN signal against the wrong
        class's neighbors with no error. Fails fast here, once, at classifier construction
        (server startup or CLI invocation) -- not per-request, so a bad artifact can't reach
        production traffic at all rather than corrupting scores silently."""
        if self._ood_stats is None:
            return
        id2label: dict[int, str] = self.model.config.id2label
        expected = [id2label[i] for i in range(len(id2label))]
        if self._ood_stats.class_names != expected:
            msg = (
                f"ood_stats.npz class_names {self._ood_stats.class_names} do not match "
                f"this model's id2label {expected} (order matters, not just the set) -- "
                "OOD scoring would silently score against the wrong classes. Regenerate "
                "ood_stats.npz for this exact model with compute-ood-stats."
            )
            raise BertTunningError(msg)

    def _validate_ood_stats_model_identity(self) -> None:
        """class_names alone can't distinguish two different model architectures when both
        were trained on the identical corpus/label set -- true for every model in this
        project's registry (xlm-roberta/beto/minilm all commonly train on the same
        FOLDER_TO_LABEL classes). model_type + hidden_size is a coarse fingerprint that
        catches the realistic mistake (copying one model's ood_stats.npz next to a
        different architecture) without a new CLI flag or fragile checkpoint-metadata
        introspection. Skipped entirely when the stats predate this field (both None) --
        this is an additional check layered on top of the class-mapping one, not a
        replacement for it, and not a hard requirement for older artifacts."""
        if self._ood_stats is None:
            return
        if self._ood_stats.model_type is None or self._ood_stats.model_hidden_size is None:
            return
        actual_type = self.model.config.model_type
        actual_hidden_size = self.model.config.hidden_size
        if (
            self._ood_stats.model_type != actual_type
            or self._ood_stats.model_hidden_size != actual_hidden_size
        ):
            msg = (
                f"ood_stats.npz was computed from model_type={self._ood_stats.model_type!r}, "
                f"hidden_size={self._ood_stats.model_hidden_size}, but the loaded model is "
                f"model_type={actual_type!r}, hidden_size={actual_hidden_size} -- this "
                "ood_stats.npz belongs to a different model architecture. Regenerate it for "
                "this exact model with compute-ood-stats."
            )
            raise BertTunningError(msg)

    def _validate_svm_classifiers_class_mapping(self) -> None:
        """svm_classifiers.joblib's class set must match this model's id2label -- a stale
        or mismatched artifact (e.g. fit for a corpus with a different class set) would
        silently produce svm_scores keyed by the wrong classes, or missing some entirely.
        Unlike ood_stats.npz's class_names (an ordered list indexed positionally by label
        id, since knn_mean_distance indexes into it directly), svm_classifiers is a dict
        keyed by class NAME -- only the set needs to match, not the order -- but the same
        fail-fast-at-construction rationale as _validate_ood_stats_class_mapping applies."""
        if self._svm_classifiers is None:
            return
        id2label: dict[int, str] = self.model.config.id2label
        expected = set(id2label.values())
        actual = set(self._svm_classifiers.keys())
        if actual != expected:
            msg = (
                f"svm_classifiers.joblib classes {sorted(actual)} do not match this "
                f"model's id2label classes {sorted(expected)} -- svm_scores would be "
                "computed for the wrong classes. Regenerate svm_classifiers.joblib for "
                "this exact model with compute-svm-classifiers."
            )
            raise BertTunningError(msg)

    def _warn_on_uncalibrated_thresholds(self) -> None:
        """resolve_ood_thresholds()'s silent Settings.OOD_* fallback is intentional backward
        compatibility, not something to hide from whoever operates this service -- a model
        that's never been through `evaluate-ood-calibration --write-thresholds` silently
        inherits whichever model Settings.OOD_* happens to be calibrated for. This does not
        fail startup -- an uncalibrated model is still usable, just with potentially
        miscalibrated OOD decisions -- but it must not be silent either. Runs once at
        construction, not per-request. mahalanobis_threshold_status distinguishes "never
        calibrated" (this WARNING) from "calibration ran, degenerate-threshold guard
        correctly refused to persist a value" (a separate, non-actionable INFO line below) --
        collapsing both into one message here is exactly the ambiguity that field exists to
        remove."""
        if self._ood_stats is None:
            return
        uncalibrated = [
            name
            for name, value in (
                ("cosine_threshold", self._ood_stats.cosine_threshold),
                ("knn_distance_threshold", self._ood_stats.knn_distance_threshold),
            )
            if value is None
        ]
        if self._ood_stats.mahalanobis_threshold_status == "not_calibrated":
            uncalibrated.append("mahalanobis_p_threshold")
        if self._ood_stats.tfidf_centroids.size > 0 and self._ood_stats.tfidf_threshold is None:
            uncalibrated.append("tfidf_threshold")
        if uncalibrated:
            log.warning(
                "ood_stats.npz has no per-model value for %s -- falling back to Settings.OOD_* "
                "(calibrated for a specific model, not necessarily this one). Run "
                "evaluate-ood-calibration --write-thresholds for this model to silence this.",
                ", ".join(uncalibrated),
            )
        if self._ood_stats.mahalanobis_threshold_status == "refused_degenerate":
            log.info(
                "mahalanobis_p_threshold falls back to Settings.OOD_MAHALANOBIS_P_THRESHOLD "
                "because evaluate-ood-calibration's degenerate-threshold guard correctly "
                "refused to persist a floor-adjacent value for this model -- expected, no "
                "action needed."
            )

    @cached_property
    def _train_mahalanobis_distances(self) -> npt.NDArray[np.float64] | None:
        """Computed lazily on first access (not in __init__), cached for the process
        lifetime -- avoids recomputing 1300+ training-point distances on every single
        predict_text() call. None when there's no ood_stats.npz to compute it from."""
        if self._ood_stats is None:
            return None
        return compute_train_mahalanobis_distances(self._ood_stats)

    @cached_property
    def _tfidf_vectorizer(self) -> "TfidfVectorizer | None":
        """Reconstructed once per process lifetime, not per predict_text() call -- mirrors
        _train_mahalanobis_distances' caching rationale. None when ood_stats.npz predates
        the TF-IDF signal or there's no ood_stats.npz at all."""
        if self._ood_stats is None:
            return None
        return build_tfidf_vectorizer(self._ood_stats)

    def predict_text(self, text: str) -> PredictResult:
        inputs = self.tokenizer(
            clean_text(text),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
            cls_embedding = outputs.hidden_states[-1][:, 0, :][0].cpu().numpy().astype(np.float64)

        id2label = self.model.config.id2label
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        label = id2label[pred_idx]

        confidence_tier = ConfidenceTier.from_confidence(confidence, self.threshold)
        certain = confidence_tier is ConfidenceTier.CONFIDENT
        if self._svm_classifiers is None:
            svm_scores_result: dict[str, float] = {}
            svm_predicted_label, svm_agrees_with_prediction = "", True
        else:
            svm_scores_result = compute_svm_scores(cls_embedding, self._svm_classifiers)
            svm_predicted_label = svm_top_label(svm_scores_result)
            svm_agrees_with_prediction = svm_predicted_label == label
        classifier_disagreement = not svm_agrees_with_prediction
        result = PredictResult(
            label=label,
            confidence=round(confidence, 4),
            certain=certain,
            all_scores={id2label[i]: round(float(p), 4) for i, p in enumerate(probs)},
            review_route=decide_review_route(
                confidence_tier=confidence_tier,
                ood_evidence=OodEvidence.NOT_ANOMALOUS,
                classifier_disagreement=classifier_disagreement,
            ),
            svm_scores=svm_scores_result,
            svm_predicted_label=svm_predicted_label,
            svm_agrees_with_prediction=svm_agrees_with_prediction,
        )

        if self._ood_stats is None:
            return result

        train_distances = self._train_mahalanobis_distances
        assert train_distances is not None
        if len(train_distances) == 0:
            log.warning(
                "ood_stats.npz has no k-NN training data (empty knn_train_embeddings) — "
                "OOD scoring disabled for this prediction"
            )
            return result

        tfidf_z = (
            tfidf_cosine_z_score(text, self._ood_stats, self._tfidf_vectorizer)
            if self._tfidf_vectorizer is not None
            else float("nan")
        )
        squared_distance = mahalanobis_min_distance(cls_embedding, self._ood_stats)
        scores = OodScores(
            mahalanobis_p=empirical_survival_p_value(squared_distance, train_distances),
            cosine_z=cosine_z_score(cls_embedding, self._ood_stats),
            knn_distance=knn_mean_distance(
                cls_embedding, self._ood_stats, pred_idx, k=Settings.OOD_KNN_NEIGHBORS
            ),
            tfidf_cosine_z=tfidf_z,
        )
        maha_p_theoretical = mahalanobis_chi2_p_value_from_distance(
            squared_distance, self._ood_stats
        )
        thresholds = resolve_ood_thresholds(self._ood_stats)
        in_distribution = not is_out_of_distribution(scores, thresholds)
        ood_metrics = OodMetrics(
            mahalanobis_p_value=round(scores.mahalanobis_p, 6),
            mahalanobis_p_value_theoretical=round(maha_p_theoretical, 6),
            cosine_z=round(scores.cosine_z, 4),
            knn_distance=round(scores.knn_distance, 4),
            tfidf_cosine_z=(
                None if np.isnan(scores.tfidf_cosine_z) else round(scores.tfidf_cosine_z, 4)
            ),
            in_distribution=in_distribution,
        )
        return result.model_copy(
            update={
                "ood_metrics": ood_metrics,
                "review_route": decide_review_route(
                    confidence_tier=confidence_tier,
                    ood_evidence=OodEvidence.from_in_distribution(in_distribution=in_distribution),
                    classifier_disagreement=classifier_disagreement,
                ),
            }
        )
