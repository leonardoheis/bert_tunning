import logging
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

from src.exceptions import BertTunningError
from src.ingestion.extract import clean_text
from src.schema import ClassEmbeddingStats
from src.settings import Settings

log = logging.getLogger(__name__)


class _PcaReduction(NamedTuple):
    """Internal return type for _reduce_dimensionality — not part of the public schema.
    A NamedTuple, not a Pydantic model: this is trusted internal data straight out of
    sklearn's PCA, not external input that needs validation/coercion."""

    reduced: npt.NDArray[np.float64]
    mean: npt.NDArray[np.float64]
    components: npt.NDArray[np.float64]


class _TfidfStats(NamedTuple):
    """Internal return type for compute_tfidf_stats -- merged into ClassEmbeddingStats by
    the caller (compute_class_stats), same convention as _PcaReduction."""

    vocabulary_terms: list[str]
    idf: npt.NDArray[np.float64]
    centroids: npt.NDArray[np.float64]
    cosine_calibration_mean: float
    cosine_calibration_std: float


def _reduce_dimensionality(embeddings: npt.NDArray[np.float64], n_components: int) -> _PcaReduction:
    capped = min(n_components, embeddings.shape[0] - 1, embeddings.shape[1])
    pca = PCA(n_components=capped)
    reduced = pca.fit_transform(embeddings)
    return _PcaReduction(reduced=reduced, mean=pca.mean_, components=pca.components_)


def _project(
    embedding: npt.NDArray[np.float64], stats: ClassEmbeddingStats
) -> npt.NDArray[np.float64]:
    return (embedding - stats.pca_mean) @ stats.pca_components.T


def _cosine_min_distance_raw(
    point: npt.NDArray[np.float64], centroids: npt.NDArray[np.float64]
) -> float:
    return float(cosine_distances(point.reshape(1, -1), centroids).min())


def compute_class_stats(  # noqa: PLR0913 -- model_type/model_hidden_size are an optional
    # identity fingerprint threaded through at the two call sites (training/pipeline.py,
    # cli/ood_stats.py); bundling them into a NamedTuple for two rarely-varying trailing
    # kwargs would be more ceremony than the limit is worth here.
    embeddings: npt.NDArray[np.float64],
    labels: list[int],
    class_names: list[str],
    *,
    n_components: int = 64,
    covariance_epsilon: float = 1e-6,
    model_type: str | None = None,
    model_hidden_size: int | None = None,
) -> ClassEmbeddingStats:
    pca_result = _reduce_dimensionality(embeddings, n_components)
    reduced = pca_result.reduced
    labels_arr = np.asarray(labels)

    centroids = np.stack([reduced[labels_arr == k].mean(axis=0) for k in range(len(class_names))])
    centered = reduced - centroids[labels_arr]
    covariance = (centered.T @ centered) / reduced.shape[0]
    covariance_reg = covariance + covariance_epsilon * np.eye(covariance.shape[0])
    covariance_inv = np.linalg.inv(covariance_reg)

    cosine_scores = np.array(
        [_cosine_min_distance_raw(reduced[i], centroids) for i in range(reduced.shape[0])]
    )

    return ClassEmbeddingStats(
        class_names=class_names,
        pca_mean=pca_result.mean,
        pca_components=pca_result.components,
        centroids=centroids,
        covariance_inv=covariance_inv,
        cosine_calibration_mean=float(cosine_scores.mean()),
        cosine_calibration_std=float(cosine_scores.std() + 1e-9),
        knn_train_embeddings=reduced,
        knn_train_labels=labels_arr.tolist(),
        model_type=model_type,
        model_hidden_size=model_hidden_size,
    )


def compute_tfidf_stats(
    texts: list[str], labels: list[int], class_names: list[str], *, max_features: int = 5000
) -> _TfidfStats:
    """Fits a TF-IDF vectorizer + per-class centroids on raw training text -- a signal
    independent of compute_class_stats' BERT-embedding space, operating on surface
    vocabulary instead. Catches lexical divergence (e.g. a different municipality's name)
    that a shared document-type "shape" in embedding space cannot separate."""
    cleaned = [clean_text(t) for t in texts]
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(cleaned).toarray()
    labels_arr = np.asarray(labels)

    centroids = np.stack([X[labels_arr == k].mean(axis=0) for k in range(len(class_names))])
    cosine_scores = np.array([_cosine_min_distance_raw(X[i], centroids) for i in range(X.shape[0])])

    return _TfidfStats(
        vocabulary_terms=vectorizer.get_feature_names_out().tolist(),
        idf=vectorizer.idf_,
        centroids=centroids,
        cosine_calibration_mean=float(cosine_scores.mean()),
        cosine_calibration_std=float(cosine_scores.std() + 1e-9),
    )


def build_tfidf_vectorizer(stats: ClassEmbeddingStats) -> TfidfVectorizer | None:
    """Reconstructs a fixed-vocabulary TfidfVectorizer from the two arrays load_stats/
    save_stats round-trip through ood_stats.npz -- verified to produce bit-identical
    .transform() output to the originally-fitted vectorizer. Returns None when this
    model's ood_stats.npz predates the TF-IDF signal (tfidf_vocabulary_terms is empty),
    so callers can treat the signal as disabled rather than crash on missing data."""
    if not stats.tfidf_vocabulary_terms:  # empty list = not fitted, see Task 1's field comment
        return None
    vocabulary = {term: i for i, term in enumerate(stats.tfidf_vocabulary_terms)}
    vectorizer = TfidfVectorizer(vocabulary=vocabulary)
    vectorizer.idf_ = stats.tfidf_idf
    return vectorizer


def tfidf_cosine_z_score(
    text: str, stats: ClassEmbeddingStats, vectorizer: TfidfVectorizer
) -> float:
    """Cosine distance to the nearest TF-IDF centroid, z-scored against the training set --
    same technique as cosine_z_score, different vector space. Caller must have already
    confirmed build_tfidf_vectorizer(stats) is not None."""
    point = vectorizer.transform([clean_text(text)]).toarray()[0]
    cosine_raw = _cosine_min_distance_raw(point, stats.tfidf_centroids)
    return (cosine_raw - stats.tfidf_cosine_calibration_mean) / stats.tfidf_cosine_calibration_std


def mahalanobis_min_distance(
    embedding: npt.NDArray[np.float64], stats: ClassEmbeddingStats
) -> float:
    point = _project(embedding, stats)
    diffs = stats.centroids - point
    distances = np.einsum("kd,de,ke->k", diffs, stats.covariance_inv, diffs)
    return float(np.min(distances))


def cosine_min_distance(embedding: npt.NDArray[np.float64], stats: ClassEmbeddingStats) -> float:
    point = _project(embedding, stats)
    return _cosine_min_distance_raw(point, stats.centroids)


def mahalanobis_chi2_p_value_from_distance(
    squared_distance: float, stats: ClassEmbeddingStats
) -> float:
    """Same as mahalanobis_chi2_p_value, but takes an already-computed squared distance --
    for callers (e.g. BertTunningClassifier.predict_text) that also need
    mahalanobis_empirical_p_value's distance and would otherwise recompute
    mahalanobis_min_distance (a PCA projection + centroid search) twice per call."""
    degrees_of_freedom = stats.centroids.shape[1]
    return float(chi2.sf(squared_distance, df=degrees_of_freedom))


def mahalanobis_chi2_p_value(
    embedding: npt.NDArray[np.float64], stats: ClassEmbeddingStats
) -> float:
    """Theoretical p-value under the assumption that class-conditional embeddings are
    multivariate Gaussian with one shared covariance matrix: the squared Mahalanobis
    distance of such a point follows a chi-squared distribution with `df` equal to the
    embedding dimensionality. Kept as a transparent, purely informational value --
    NOT used to decide in_distribution. A QQ-plot check (see the playground notebook)
    showed this assumption is badly violated for this corpus (observed distances run
    ~5x larger than chi2 predicts), which is why mahalanobis_empirical_p_value exists
    and is the one that actually drives the anomaly decision. A LOW value here still
    means "far from centroid," it just isn't a trustworthy probability."""
    squared_distance = mahalanobis_min_distance(embedding, stats)
    return mahalanobis_chi2_p_value_from_distance(squared_distance, stats)


def empirical_survival_p_value(distance: float, reference: npt.NDArray[np.float64]) -> float:
    """The standard permutation-test empirical p-value: the fraction of `reference` values
    at least as extreme as `distance`, with the usual +1/+1 correction so the result is
    never exactly 0. Raises if `reference` is empty -- silently returning 1.0 ("maximally
    normal") for no reference data would be a fail-open bug, backwards for an
    anomaly-detection signal."""
    if len(reference) == 0:
        msg = "empirical_survival_p_value: reference array is empty, cannot rank against it"
        raise ValueError(msg)
    exceed_count = int(np.sum(reference >= distance))
    return (exceed_count + 1) / (len(reference) + 1)


def compute_train_mahalanobis_distances(stats: ClassEmbeddingStats) -> npt.NDArray[np.float64]:
    """Squared Mahalanobis distance from every training document (stats.knn_train_embeddings,
    already PCA-reduced) to its OWN TRUE class centroid (via stats.knn_train_labels) -- not
    the nearest centroid. This intentionally mirrors compute_class_stats()'s own covariance
    estimation (`centered = reduced - centroids[labels_arr]`), built from each point's
    deviation from its labeled class, not whichever centroid happens to be closest. Using
    nearest-centroid distance here would let ambiguous/boundary training points look
    artificially unremarkable (nearest distance <= true-label distance, always), corrupting
    the reference distribution's tail -- exactly where "how extreme is extreme" matters most.
    mahalanobis_empirical_p_value() below still scores a QUERY point's distance via
    mahalanobis_min_distance() (nearest centroid) -- inference has no true label to measure
    against, and nearest-centroid is that function's existing, unchanged, preserved
    behavior (NOT a distance to some "predicted-class centroid" -- no such per-class
    distance is computed at inference time). This asymmetry (reference: true-label
    distance; query: nearest-centroid distance) is intentional -- see "Global Constraints"
    in this plan."""
    labels_arr = np.asarray(stats.knn_train_labels)
    distances = np.empty(len(stats.knn_train_embeddings), dtype=np.float64)
    for i, point in enumerate(stats.knn_train_embeddings):
        centroid = stats.centroids[labels_arr[i]]
        diff = centroid - point
        distances[i] = float(diff @ stats.covariance_inv @ diff)
    return distances


def mahalanobis_empirical_p_value(
    embedding: npt.NDArray[np.float64],
    stats: ClassEmbeddingStats,
    train_distances: npt.NDArray[np.float64],
) -> float:
    """Empirical (rank-based) p-value for a query embedding: ranks its Mahalanobis distance
    to the nearest class centroid (mahalanobis_min_distance) against train_distances (each
    training document's distance to its own TRUE class centroid -- see
    compute_train_mahalanobis_distances). Makes no distributional assumption, unlike
    mahalanobis_chi2_p_value -- this is the value that drives is_out_of_distribution(). A LOW
    p-value means the document is anomalous, same comparison direction as the chi2 version
    it replaces."""
    distance = mahalanobis_min_distance(embedding, stats)
    return empirical_survival_p_value(distance, train_distances)


def cosine_z_score(embedding: npt.NDArray[np.float64], stats: ClassEmbeddingStats) -> float:
    """Cosine distance to the nearest centroid, z-scored against the training set."""
    cosine_raw = cosine_min_distance(embedding, stats)
    return (cosine_raw - stats.cosine_calibration_mean) / stats.cosine_calibration_std


def knn_mean_distance(
    embedding: npt.NDArray[np.float64],
    stats: ClassEmbeddingStats,
    predicted_label_id: int,
    *,
    k: int = 10,
) -> float:
    """Mean Euclidean distance, in PCA space, to the k nearest training documents that share
    the predicted class. Unlike Mahalanobis (global shared covariance, assumes one Gaussian
    shape) and cosine (distance to a single centroid), this makes no assumption about the
    class's shape — it directly measures local density around the predicted class's own
    training examples, which matters for heterogeneous classes (e.g. a broad `otro`
    catch-all) that a single centroid represents poorly. A HIGH distance means anomalous —
    same comparison direction as cosine_z_score. Returns NaN if predicted_label_id has zero
    training points — callers comparing this against a threshold must handle that explicitly,
    since `nan > threshold` silently evaluates to False in Python and would never flag as
    anomalous."""
    point = _project(embedding, stats)
    labels_arr = np.array(stats.knn_train_labels)
    class_points = stats.knn_train_embeddings[labels_arr == predicted_label_id]
    if class_points.shape[0] == 0:
        log.warning(
            "knn_mean_distance: class %d has zero training points — returning NaN",
            predicted_label_id,
        )
        return float("nan")
    k_eff = min(k, class_points.shape[0])
    distances = np.linalg.norm(class_points - point, axis=1)
    nearest = np.partition(distances, k_eff - 1)[:k_eff]
    return float(nearest.mean())


class OodThresholds(NamedTuple):
    """Resolved OOD decision thresholds for one specific model -- either the values
    evaluate-ood-calibration wrote into that model's own ood_stats.npz (via
    --write-thresholds), or Settings.OOD_* as a fallback for stats files that predate
    per-model calibration. is_out_of_distribution() must never read Settings.OOD_* directly
    again -- always go through resolve_ood_thresholds(), or a freshly trained model silently
    inherits whichever model's thresholds happen to be in Settings."""

    mahalanobis_p: float
    cosine_z: float
    knn_distance: float


def resolve_ood_thresholds(stats: ClassEmbeddingStats) -> OodThresholds:
    """Falls back to Settings.OOD_* per-field, only for whichever threshold
    evaluate-ood-calibration hasn't written yet (None) -- a stats file with all three set
    never touches Settings at all."""
    return OodThresholds(
        mahalanobis_p=stats.mahalanobis_p_threshold
        if stats.mahalanobis_p_threshold is not None
        else Settings.OOD_MAHALANOBIS_P_THRESHOLD,
        cosine_z=stats.cosine_threshold
        if stats.cosine_threshold is not None
        else Settings.OOD_COSINE_THRESHOLD,
        knn_distance=stats.knn_distance_threshold
        if stats.knn_distance_threshold is not None
        else Settings.OOD_KNN_DISTANCE_THRESHOLD,
    )


def save_stats(stats: ClassEmbeddingStats, path: Path) -> None:
    # npz has no native "missing key" for a single scalar the way a dict does, and no None --
    # NaN is the serialization sentinel for "not yet calibrated," round-tripped back to None
    # by load_stats's _optional_threshold.
    maha_threshold = (
        np.nan if stats.mahalanobis_p_threshold is None else stats.mahalanobis_p_threshold
    )
    cosine_thresh = np.nan if stats.cosine_threshold is None else stats.cosine_threshold
    knn_thresh = np.nan if stats.knn_distance_threshold is None else stats.knn_distance_threshold
    # "" / -1 are the None-sentinels for a string/int field, the same role NaN plays for the
    # threshold floats above -- npz has no native optional-scalar support.
    model_type = "" if stats.model_type is None else stats.model_type
    model_hidden_size = -1 if stats.model_hidden_size is None else stats.model_hidden_size
    tfidf_threshold = np.nan if stats.tfidf_threshold is None else stats.tfidf_threshold

    # Write to a temp file first, verify it actually loads back, then atomically replace the
    # real path -- np.savez writing directly to `path` left a window where a crash, disk-full,
    # or kill mid-write corrupts the ONLY copy of ood_stats.npz, which predict/serve also read
    # from. Path.replace() (os.replace() under the hood) is atomic on both POSIX and Windows
    # when src/dst are on the same filesystem (always true here -- same directory).
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        # A file handle, not a string/Path, is required here: np.savez auto-appends ".npz" to
        # string/Path filenames that don't already end in it, which would silently save this
        # as "....npz.tmp.npz" instead of the tmp_path we opened. A handle bypasses that.
        with tmp_path.open("wb") as f:
            np.savez(
                f,
                class_names=np.array(stats.class_names),
                pca_mean=stats.pca_mean,
                pca_components=stats.pca_components,
                centroids=stats.centroids,
                covariance_inv=stats.covariance_inv,
                cosine_calibration_mean=stats.cosine_calibration_mean,
                cosine_calibration_std=stats.cosine_calibration_std,
                knn_train_embeddings=stats.knn_train_embeddings,
                knn_train_labels=np.array(stats.knn_train_labels),
                mahalanobis_p_threshold=maha_threshold,
                cosine_threshold=cosine_thresh,
                knn_distance_threshold=knn_thresh,
                model_type=model_type,
                model_hidden_size=model_hidden_size,
                mahalanobis_threshold_status=stats.mahalanobis_threshold_status,
                tfidf_vocabulary_terms=np.array(stats.tfidf_vocabulary_terms),
                tfidf_idf=stats.tfidf_idf,
                tfidf_centroids=stats.tfidf_centroids,
                tfidf_cosine_calibration_mean=stats.tfidf_cosine_calibration_mean,
                tfidf_cosine_calibration_std=stats.tfidf_cosine_calibration_std,
                tfidf_threshold=tfidf_threshold,
            )
        load_stats(tmp_path)  # fail fast on a corrupt/incomplete write, before touching `path`
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _optional_threshold(data: npt.NDArray[np.float64]) -> float | None:
    value = float(data)
    return None if np.isnan(value) else value


def _optional_str(data: npt.NDArray[np.str_]) -> str | None:
    value = str(data)
    return None if value == "" else value


def _optional_int(data: npt.NDArray[np.int_]) -> int | None:
    value = int(data)
    return None if value == -1 else value


def _threshold_status(
    data: npt.NDArray[np.str_],
) -> Literal["not_calibrated", "calibrated", "refused_degenerate"]:
    value = str(data)
    match value:
        case "not_calibrated" | "calibrated" | "refused_degenerate":
            return value
        case _:
            msg = f"ood_stats.npz has an unrecognized mahalanobis_threshold_status: {value!r}"
            raise BertTunningError(msg)


def load_stats(path: Path) -> ClassEmbeddingStats:
    data = np.load(str(path), allow_pickle=False)
    return ClassEmbeddingStats(
        class_names=data["class_names"].tolist(),
        pca_mean=data["pca_mean"],
        pca_components=data["pca_components"],
        centroids=data["centroids"],
        covariance_inv=data["covariance_inv"],
        cosine_calibration_mean=float(data["cosine_calibration_mean"]),
        cosine_calibration_std=float(data["cosine_calibration_std"]),
        knn_train_embeddings=data["knn_train_embeddings"],
        knn_train_labels=data["knn_train_labels"].tolist(),
        # "in data.files" -- not data.get() (npz's NpzFile has no .get) -- lets a
        # pre-this-change ood_stats.npz (missing these keys entirely, not just NaN) still
        # load instead of KeyError-ing every predict/serve call until it's regenerated.
        mahalanobis_p_threshold=_optional_threshold(data["mahalanobis_p_threshold"])
        if "mahalanobis_p_threshold" in data.files
        else None,
        cosine_threshold=_optional_threshold(data["cosine_threshold"])
        if "cosine_threshold" in data.files
        else None,
        knn_distance_threshold=_optional_threshold(data["knn_distance_threshold"])
        if "knn_distance_threshold" in data.files
        else None,
        model_type=_optional_str(data["model_type"]) if "model_type" in data.files else None,
        model_hidden_size=_optional_int(data["model_hidden_size"])
        if "model_hidden_size" in data.files
        else None,
        mahalanobis_threshold_status=_threshold_status(data["mahalanobis_threshold_status"])
        if "mahalanobis_threshold_status" in data.files
        else "not_calibrated",
        tfidf_vocabulary_terms=(
            data["tfidf_vocabulary_terms"].tolist()
            if "tfidf_vocabulary_terms" in data.files
            else []
        ),
        tfidf_idf=(data["tfidf_idf"] if "tfidf_idf" in data.files else np.zeros(0)),
        tfidf_centroids=(
            data["tfidf_centroids"] if "tfidf_centroids" in data.files else np.zeros((0, 0))
        ),
        tfidf_cosine_calibration_mean=(
            float(data["tfidf_cosine_calibration_mean"])
            if "tfidf_cosine_calibration_mean" in data.files
            else 0.0
        ),
        tfidf_cosine_calibration_std=(
            float(data["tfidf_cosine_calibration_std"])
            if "tfidf_cosine_calibration_std" in data.files
            else 1.0
        ),
        tfidf_threshold=(
            _optional_threshold(data["tfidf_threshold"])
            if "tfidf_threshold" in data.files
            else None
        ),
    )
