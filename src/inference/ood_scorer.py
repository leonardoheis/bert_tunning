"""OOD scoring, extracted from BertTunningClassifier -- the artifact this project's four
embedding/lexical out-of-distribution signals (Mahalanobis, cosine, k-NN, TF-IDF) are
computed from. See docs/superpowers/specs/2026-07-16-ood-scorer-extraction-design.md for
why this is its own module and increment (SRP: BertTunningClassifier was accumulating
loading, validation, and scoring logic for every review signal it consumes) -- SVM
reviewer extraction is a deferred follow-up increment, not part of this file."""

import logging
from functools import cached_property
from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

# sklearn is already an unconditional runtime dependency elsewhere in this project
# (src/ood.py) -- gating this import behind TYPE_CHECKING buys nothing real here.
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: TC002

from src.exceptions import BertTunningError
from src.ood import (
    OodCalibrationStatus,
    OodThresholds,
    build_tfidf_vectorizer,
    compute_train_mahalanobis_distances,
    cosine_z_score,
    empirical_survival_p_value,
    knn_mean_distance,
    load_stats,
    mahalanobis_chi2_p_value_from_distance,
    mahalanobis_min_distance,
    resolve_ood_calibration_status,
    resolve_ood_thresholds,
    tfidf_cosine_z_score,
)
from src.schema import OodArtifact, OodMetrics
from src.settings import Settings

log = logging.getLogger(__name__)


class OodScores(NamedTuple):
    """The four OOD signals -- always computed together, passed together, never used
    independently. tfidf_cosine_z uses the same NaN-sentinel convention as knn_distance,
    not Optional -- keeps this NamedTuple a uniform tuple of plain floats. The two NaNs
    mean different things and are handled with OPPOSITE polarity in is_out_of_distribution:
    knn_distance's NaN means "this document's predicted class has zero training points"
    and fails CLOSED (anomalous); tfidf_cosine_z's NaN means "this whole model's
    ood_stats.npz predates the TF-IDF signal" and fails OPEN (not anomalous, same as
    OodScorer being absent disables OOD scoring entirely for the other three signals)."""

    mahalanobis_p: float
    cosine_z: float
    knn_distance: float
    tfidf_cosine_z: float = float("nan")


_ALL_CALIBRATED = OodCalibrationStatus(
    mahalanobis="calibrated",
    cosine="calibrated",
    knn_distance="calibrated",
    tfidf_cosine="calibrated",
)


def is_out_of_distribution(
    scores: OodScores,
    thresholds: OodThresholds,
    calibration_status: OodCalibrationStatus = _ALL_CALIBRATED,
    *,
    allow_uncalibrated_fallback: bool = True,
) -> bool:
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

    calibration_status/allow_uncalibrated_fallback default to "fully permissive" (every
    signal calibrated / fallback allowed) so callers that don't care about this gating
    behave exactly as before. When allow_uncalibrated_fallback is False, a signal whose
    calibration_status is "not_calibrated" is excluded from the OR -- its score is still
    computed by the caller and reported, it just can't flip in_distribution to False on
    its own. "refused_degenerate" (mahalanobis only) is NEVER excluded regardless of the
    flag -- that's a legitimate calibration outcome (the degenerate-threshold guard
    correctly declined to persist a floor-adjacent value), not a calibration gap; both
    committed production models rely on this exact fallback for Mahalanobis today. The
    existing NaN-based knn_distance/tfidf_cosine_z fail-closed/fail-open rules are
    unchanged -- this is an additional `and`, not a replacement.
    """
    maha_blocked = (
        not allow_uncalibrated_fallback and calibration_status.mahalanobis == "not_calibrated"
    )
    maha_anomalous = not maha_blocked and scores.mahalanobis_p < thresholds.mahalanobis_p
    cosine_blocked = (
        not allow_uncalibrated_fallback and calibration_status.cosine == "not_calibrated"
    )
    cosine_anomalous = not cosine_blocked and scores.cosine_z > thresholds.cosine_z
    knn_blocked = (
        not allow_uncalibrated_fallback and calibration_status.knn_distance == "not_calibrated"
    )
    knn_anomalous = not knn_blocked and (
        bool(np.isnan(scores.knn_distance)) or scores.knn_distance > thresholds.knn_distance
    )
    tfidf_blocked = (
        not allow_uncalibrated_fallback and calibration_status.tfidf_cosine == "not_calibrated"
    )
    # Opposite of knn_anomalous's NaN handling on purpose -- see OodScores' docstring.
    tfidf_anomalous = not tfidf_blocked and (
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


class OodScorer:
    """Owns everything derived from a loaded ood_stats.npz: validation against the model
    it's paired with, the uncalibrated-threshold warning, and per-document scoring. One
    instance per BertTunningClassifier, built once at construction via load(). Unlike the
    classifier-level attributes this replaces, there is no None-state to check on every
    method here -- an OodScorer only ever exists when stats are genuinely loaded; the
    "no ood_stats.npz" case is represented by load() returning None, once, not by every
    method re-checking an Optional field."""

    def __init__(self, stats: OodArtifact) -> None:
        self._stats = stats

    @staticmethod
    def load(model_path: str) -> "OodScorer | None":
        """None when ood_stats.npz isn't present next to the model -- mirrors
        BertTunningClassifier's previous _load_ood_stats exactly, just wrapped in the
        scorer instead of returning a bare OodArtifact."""
        stats_path = Path(model_path) / "ood_stats.npz"
        if not stats_path.exists():
            log.info("No ood_stats.npz found at %s — OOD scoring disabled", stats_path)
            return None
        log.info("Loaded OOD stats from %s", stats_path)
        return OodScorer(load_stats(stats_path))

    def validate(self, id2label: dict[int, str], model_type: str, model_hidden_size: int) -> None:
        """Both of the model-compatibility checks a loaded ood_stats.npz must pass before
        it's trusted -- class-mapping first, then model-identity."""
        self._validate_class_mapping(id2label)
        self._validate_model_identity(model_type, model_hidden_size)

    def _validate_class_mapping(self, id2label: dict[int, str]) -> None:
        """ood_stats.npz's class_names must match this model's id2label -- by count AND by
        ordered index, since knn_mean_distance() indexes stats.knn_train_labels directly by
        the model's own predicted label id (see OodScorer.score). A silently mismatched or
        stale ood_stats.npz would score every prediction's k-NN signal against the wrong
        class's neighbors with no error. Fails fast here, once, at classifier construction
        (server startup or CLI invocation) -- not per-request, so a bad artifact can't reach
        production traffic at all rather than corrupting scores silently."""
        expected = [id2label[i] for i in range(len(id2label))]
        if self._stats.class_names != expected:
            msg = (
                f"ood_stats.npz class_names {self._stats.class_names} do not match "
                f"this model's id2label {expected} (order matters, not just the set) -- "
                "OOD scoring would silently score against the wrong classes. Regenerate "
                "ood_stats.npz for this exact model with compute-ood-stats."
            )
            raise BertTunningError(msg)

    def _validate_model_identity(self, model_type: str, model_hidden_size: int) -> None:
        """class_names alone can't distinguish two different model architectures when both
        were trained on the identical corpus/label set -- true for every model in this
        project's registry (xlm-roberta/beto/minilm all commonly train on the same
        FOLDER_TO_LABEL classes). model_type + hidden_size is a coarse fingerprint that
        catches the realistic mistake (copying one model's ood_stats.npz next to a
        different architecture) without a new CLI flag or fragile checkpoint-metadata
        introspection. Skipped entirely when the stats predate this field (metadata is
        None) -- this is an additional check layered on top of the class-mapping one, not
        a replacement for it, and not a hard requirement for older artifacts."""
        metadata = self._stats.metadata
        if metadata is None:
            return
        mismatched = (
            metadata.model_type != model_type or metadata.model_hidden_size != model_hidden_size
        )
        if mismatched:
            msg = (
                f"ood_stats.npz was computed from model_type={metadata.model_type!r}, "
                f"hidden_size={metadata.model_hidden_size}, but the loaded model is "
                f"model_type={model_type!r}, hidden_size={model_hidden_size} -- this "
                "ood_stats.npz belongs to a different model architecture. Regenerate it for "
                "this exact model with compute-ood-stats."
            )
            raise BertTunningError(msg)

    def warn_if_uncalibrated(self) -> None:
        """resolve_ood_thresholds()'s silent Settings.OOD_* fallback is intentional backward
        compatibility, not something to hide from whoever operates this service -- a model
        that's never been through `evaluate-ood-calibration --write-thresholds` silently
        inherits whichever model Settings.OOD_* happens to be calibrated for. This does not
        fail startup -- an uncalibrated model is still usable, just with potentially
        miscalibrated OOD decisions unless OOD_ALLOW_UNCALIBRATED_FALLBACK=False -- but it
        must not be silent either. Runs once at construction, not per-request. Reuses
        resolve_ood_calibration_status() rather than re-deriving "never calibrated" here,
        so this warning and score()'s actual gating can never disagree about which signals
        are uncalibrated. mahalanobis's three-way status distinguishes "never calibrated"
        (this WARNING) from "calibration ran, degenerate-threshold guard correctly refused
        to persist a value" (a separate, non-actionable INFO line below) -- collapsing both
        into one message here is exactly the ambiguity that field exists to remove."""
        status = resolve_ood_calibration_status(self._stats)
        uncalibrated = [
            name
            for name, value in (
                ("mahalanobis_p_threshold", status.mahalanobis),
                ("cosine_threshold", status.cosine),
                ("knn_distance_threshold", status.knn_distance),
                ("tfidf_threshold", status.tfidf_cosine),
            )
            if value == "not_calibrated"
        ]
        if uncalibrated:
            if Settings.OOD_ALLOW_UNCALIBRATED_FALLBACK:
                log.warning(
                    "ood_stats.npz has no per-model value for %s -- falling back to "
                    "Settings.OOD_* (calibrated for a specific model, not necessarily this "
                    "one). Run evaluate-ood-calibration --write-thresholds for this model to "
                    "silence this.",
                    ", ".join(uncalibrated),
                )
            else:
                log.warning(
                    "ood_stats.npz has no per-model value for %s -- DISABLED under strict "
                    "mode (OOD_ALLOW_UNCALIBRATED_FALLBACK=False), not falling back to "
                    "Settings.OOD_*. Run evaluate-ood-calibration --write-thresholds for "
                    "this model to enable it.",
                    ", ".join(uncalibrated),
                )
        if status.mahalanobis == "refused_degenerate":
            log.info(
                "mahalanobis_p_threshold falls back to Settings.OOD_MAHALANOBIS_P_THRESHOLD "
                "because evaluate-ood-calibration's degenerate-threshold guard correctly "
                "refused to persist a floor-adjacent value for this model -- expected, no "
                "action needed."
            )

    @cached_property
    def _train_mahalanobis_distances(self) -> npt.NDArray[np.float64]:
        """Computed lazily on first access (not at construction), cached for the process
        lifetime -- avoids recomputing 1300+ training-point distances on every single
        score() call."""
        return compute_train_mahalanobis_distances(self._stats)

    @cached_property
    def _tfidf_vectorizer(self) -> "TfidfVectorizer | None":
        """Reconstructed once per process lifetime, not per score() call -- mirrors
        _train_mahalanobis_distances' caching rationale. None when ood_stats.npz predates
        the TF-IDF signal -- a genuinely different, second reason than "no ood_stats.npz at
        all" (which this class as a whole doesn't exist to represent; see load())."""
        return build_tfidf_vectorizer(self._stats)

    def score(
        self, text: str, embedding: npt.NDArray[np.float64], pred_idx: int
    ) -> OodMetrics | None:
        """None when this specific prediction can't be scored (empty knn_train_embeddings --
        the predicted class, or the whole model, has zero stored k-NN training points),
        distinct from OodScorer itself being absent (load() returning None for "no
        ood_stats.npz at all"). Otherwise the four OOD signals, thresholded and rounded."""
        train_distances = self._train_mahalanobis_distances
        if len(train_distances) == 0:
            log.warning(
                "ood_stats.npz has no k-NN training data (empty knn_train_embeddings) — "
                "OOD scoring disabled for this prediction"
            )
            return None
        tfidf_z = (
            tfidf_cosine_z_score(text, self._stats, self._tfidf_vectorizer)
            if self._tfidf_vectorizer is not None
            else float("nan")
        )
        squared_distance = mahalanobis_min_distance(embedding, self._stats)
        scores = OodScores(
            mahalanobis_p=empirical_survival_p_value(squared_distance, train_distances),
            cosine_z=cosine_z_score(embedding, self._stats),
            knn_distance=knn_mean_distance(
                embedding, self._stats, pred_idx, k=Settings.OOD_KNN_NEIGHBORS
            ),
            tfidf_cosine_z=tfidf_z,
        )
        maha_p_theoretical = mahalanobis_chi2_p_value_from_distance(squared_distance, self._stats)
        thresholds = resolve_ood_thresholds(self._stats)
        calibration_status = resolve_ood_calibration_status(self._stats)
        in_distribution = not is_out_of_distribution(
            scores,
            thresholds,
            calibration_status,
            allow_uncalibrated_fallback=Settings.OOD_ALLOW_UNCALIBRATED_FALLBACK,
        )
        return OodMetrics(
            mahalanobis_p_value=round(scores.mahalanobis_p, 6),
            mahalanobis_p_value_theoretical=round(maha_p_theoretical, 6),
            cosine_z=round(scores.cosine_z, 4),
            knn_distance=round(scores.knn_distance, 4),
            tfidf_cosine_z=(
                None if np.isnan(scores.tfidf_cosine_z) else round(scores.tfidf_cosine_z, 4)
            ),
            in_distribution=in_distribution,
            mahalanobis_calibration_status=calibration_status.mahalanobis,
            cosine_calibration_status=calibration_status.cosine,
            knn_distance_calibration_status=calibration_status.knn_distance,
            tfidf_calibration_status=calibration_status.tfidf_cosine,
        )
