import logging
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import numpy.typing as npt
from numpy.lib.npyio import NpzFile
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

from src.exceptions import BertTunningError
from src.ingestion.extract import clean_text
from src.schema import (
    ArtifactMetadata,
    CalibratedThresholds,
    EmbeddingStats,
    LexicalStats,
    OodArtifact,
)
from src.settings import Settings

log = logging.getLogger(__name__)

# Bumped whenever OodArtifact's section shape changes. 1 is retroactively assigned to mean
# "no format_version key in the .npz at all" -- every already-committed ood_stats.npz today.
# See docs/superpowers/specs/2026-07-16-ood-artifact-schema-versioning-design.md.
CURRENT_OOD_ARTIFACT_VERSION = 2


class _PcaReduction(NamedTuple):
    """Internal return type for _reduce_dimensionality — not part of the public schema.
    A NamedTuple, not a Pydantic model: this is trusted internal data straight out of
    sklearn's PCA, not external input that needs validation/coercion."""

    reduced: npt.NDArray[np.float64]
    mean: npt.NDArray[np.float64]
    components: npt.NDArray[np.float64]


class _TfidfStats(NamedTuple):
    """Internal return type for compute_tfidf_stats -- merged into OodArtifact.lexical by
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


def _project(embedding: npt.NDArray[np.float64], stats: OodArtifact) -> npt.NDArray[np.float64]:
    return (embedding - stats.embedding.pca_mean) @ stats.embedding.pca_components.T


def _cosine_min_distance_raw(
    point: npt.NDArray[np.float64], centroids: npt.NDArray[np.float64]
) -> float:
    return float(cosine_distances(point.reshape(1, -1), centroids).min())


def compute_class_stats(  # noqa: PLR0913 -- model_type/model_hidden_size/texts/
    # max_tfidf_features/max_tfidf_max_df are optional trailing kwargs threaded through from
    # the two call sites (training/pipeline.py, cli/ood_stats.py); bundling them into a
    # NamedTuple for rarely-varying trailing kwargs would be more ceremony than the limit is
    # worth here.
    embeddings: npt.NDArray[np.float64],
    labels: list[int],
    class_names: list[str],
    *,
    texts: list[str],
    n_components: int = 64,
    covariance_epsilon: float = 1e-6,
    model_type: str | None = None,
    model_hidden_size: int | None = None,
    max_tfidf_features: int = Settings.OOD_TFIDF_MAX_FEATURES,
    max_tfidf_max_df: float = Settings.OOD_TFIDF_MAX_DF,
) -> OodArtifact:
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
    tfidf = compute_tfidf_stats(
        texts, labels, class_names, max_features=max_tfidf_features, max_df=max_tfidf_max_df
    )

    metadata = (
        ArtifactMetadata(model_type=model_type, model_hidden_size=model_hidden_size)
        if model_type is not None and model_hidden_size is not None
        else None
    )

    return OodArtifact(
        format_version=CURRENT_OOD_ARTIFACT_VERSION,
        class_names=class_names,
        embedding=EmbeddingStats(
            pca_mean=pca_result.mean,
            pca_components=pca_result.components,
            centroids=centroids,
            covariance_inv=covariance_inv,
            cosine_calibration_mean=float(cosine_scores.mean()),
            cosine_calibration_std=float(cosine_scores.std() + 1e-9),
            knn_train_embeddings=reduced,
            knn_train_labels=labels_arr.tolist(),
        ),
        lexical=LexicalStats(
            vocabulary_terms=tfidf.vocabulary_terms,
            idf=tfidf.idf,
            centroids=tfidf.centroids,
            cosine_calibration_mean=tfidf.cosine_calibration_mean,
            cosine_calibration_std=tfidf.cosine_calibration_std,
        ),
        metadata=metadata,
    )


def compute_tfidf_stats(
    texts: list[str],
    labels: list[int],
    class_names: list[str],
    *,
    max_features: int = Settings.OOD_TFIDF_MAX_FEATURES,
    max_df: float = Settings.OOD_TFIDF_MAX_DF,
) -> _TfidfStats:
    """Fits a TF-IDF vectorizer + per-class centroids on raw training text -- a signal
    independent of compute_class_stats' BERT-embedding space, operating on surface
    vocabulary instead. Catches lexical divergence (e.g. a different municipality's name)
    that a shared document-type "shape" in embedding space cannot separate. max_df excludes
    terms above that document-frequency fraction from the vocabulary entirely -- without it,
    shared legal boilerplate dominates the 5000-feature budget and dilutes the cosine
    distance a rare, genuinely distinguishing term (like a city name) could otherwise carry."""
    cleaned = [clean_text(t) for t in texts]
    vectorizer = TfidfVectorizer(max_features=max_features, max_df=max_df)
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


def build_tfidf_vectorizer(stats: OodArtifact) -> TfidfVectorizer | None:
    """Reconstructs a fixed-vocabulary TfidfVectorizer from the two arrays load_stats/
    save_stats round-trip through ood_stats.npz -- verified to produce bit-identical
    .transform() output to the originally-fitted vectorizer. Returns None when this
    model's ood_stats.npz predates the TF-IDF signal (tfidf_vocabulary_terms is empty),
    so callers can treat the signal as disabled rather than crash on missing data."""
    if not stats.lexical.is_fitted():
        return None
    vocabulary = {term: i for i, term in enumerate(stats.lexical.vocabulary_terms)}
    vectorizer = TfidfVectorizer(vocabulary=vocabulary)
    vectorizer.idf_ = stats.lexical.idf
    return vectorizer


def tfidf_cosine_z_score(text: str, stats: OodArtifact, vectorizer: TfidfVectorizer) -> float:
    """Cosine distance to the nearest TF-IDF centroid, z-scored against the training set --
    same technique as cosine_z_score, different vector space. Caller must have already
    confirmed build_tfidf_vectorizer(stats) is not None."""
    point = vectorizer.transform([clean_text(text)]).toarray()[0]
    cosine_raw = _cosine_min_distance_raw(point, stats.lexical.centroids)
    lexical = stats.lexical
    return (cosine_raw - lexical.cosine_calibration_mean) / lexical.cosine_calibration_std


def mahalanobis_min_distance(embedding: npt.NDArray[np.float64], stats: OodArtifact) -> float:
    point = _project(embedding, stats)
    diffs = stats.embedding.centroids - point
    distances = np.einsum("kd,de,ke->k", diffs, stats.embedding.covariance_inv, diffs)
    return float(np.min(distances))


def cosine_min_distance(embedding: npt.NDArray[np.float64], stats: OodArtifact) -> float:
    point = _project(embedding, stats)
    return _cosine_min_distance_raw(point, stats.embedding.centroids)


def mahalanobis_chi2_p_value_from_distance(squared_distance: float, stats: OodArtifact) -> float:
    """Same as mahalanobis_chi2_p_value, but takes an already-computed squared distance --
    for callers (e.g. BertTunningClassifier.predict_text) that also need
    mahalanobis_empirical_p_value's distance and would otherwise recompute
    mahalanobis_min_distance (a PCA projection + centroid search) twice per call."""
    degrees_of_freedom = stats.embedding.centroids.shape[1]
    return float(chi2.sf(squared_distance, df=degrees_of_freedom))


def mahalanobis_chi2_p_value(embedding: npt.NDArray[np.float64], stats: OodArtifact) -> float:
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


def compute_train_mahalanobis_distances(stats: OodArtifact) -> npt.NDArray[np.float64]:
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
    labels_arr = np.asarray(stats.embedding.knn_train_labels)
    distances = np.empty(len(stats.embedding.knn_train_embeddings), dtype=np.float64)
    for i, point in enumerate(stats.embedding.knn_train_embeddings):
        centroid = stats.embedding.centroids[labels_arr[i]]
        diff = centroid - point
        distances[i] = float(diff @ stats.embedding.covariance_inv @ diff)
    return distances


def mahalanobis_empirical_p_value(
    embedding: npt.NDArray[np.float64],
    stats: OodArtifact,
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


def cosine_z_score(embedding: npt.NDArray[np.float64], stats: OodArtifact) -> float:
    """Cosine distance to the nearest centroid, z-scored against the training set."""
    cosine_raw = cosine_min_distance(embedding, stats)
    mean, std = stats.embedding.cosine_calibration_mean, stats.embedding.cosine_calibration_std
    return (cosine_raw - mean) / std


def knn_mean_distance(
    embedding: npt.NDArray[np.float64],
    stats: OodArtifact,
    predicted_label_id: int,
    *,
    k: int = Settings.OOD_KNN_NEIGHBORS,
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
    labels_arr = np.array(stats.embedding.knn_train_labels)
    class_points = stats.embedding.knn_train_embeddings[labels_arr == predicted_label_id]
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
    tfidf_cosine_z: float = Settings.OOD_TFIDF_COSINE_THRESHOLD


def resolve_ood_thresholds(stats: OodArtifact) -> OodThresholds:
    """Falls back to Settings.OOD_* per-field, only for whichever threshold
    evaluate-ood-calibration hasn't written yet (None) -- a stats file with all three set
    never touches Settings at all."""
    return OodThresholds(
        mahalanobis_p=stats.thresholds.mahalanobis_p
        if stats.thresholds.mahalanobis_p is not None
        else Settings.OOD_MAHALANOBIS_P_THRESHOLD,
        cosine_z=stats.thresholds.cosine
        if stats.thresholds.cosine is not None
        else Settings.OOD_COSINE_THRESHOLD,
        knn_distance=stats.thresholds.knn_distance
        if stats.thresholds.knn_distance is not None
        else Settings.OOD_KNN_DISTANCE_THRESHOLD,
        tfidf_cosine_z=stats.thresholds.tfidf_cosine
        if stats.thresholds.tfidf_cosine is not None
        else Settings.OOD_TFIDF_COSINE_THRESHOLD,
    )


def save_stats(stats: OodArtifact, path: Path) -> None:
    """The .npz file's own keys stay exactly as they've always been -- flat, unnamespaced --
    this function and load_stats are the only place that know OodArtifact's Python-side
    shape is nested. See "Versioning strategy" in
    docs/superpowers/specs/2026-07-16-ood-artifact-schema-versioning-design.md."""
    # npz has no native "missing key" for a single scalar the way a dict does, and no None --
    # NaN is the serialization sentinel for "not yet calibrated," round-tripped back to None
    # by load_stats's _optional_threshold.
    maha_threshold = (
        np.nan if stats.thresholds.mahalanobis_p is None else stats.thresholds.mahalanobis_p
    )
    cosine_thresh = np.nan if stats.thresholds.cosine is None else stats.thresholds.cosine
    knn_thresh = np.nan if stats.thresholds.knn_distance is None else stats.thresholds.knn_distance
    # "" / -1 are the None-sentinels for a string/int field, the same role NaN plays for the
    # threshold floats above -- npz has no native optional-scalar support.
    model_type = "" if stats.metadata is None else stats.metadata.model_type
    model_hidden_size = -1 if stats.metadata is None else stats.metadata.model_hidden_size
    tfidf_threshold = (
        np.nan if stats.thresholds.tfidf_cosine is None else stats.thresholds.tfidf_cosine
    )

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
                format_version=stats.format_version,
                class_names=np.array(stats.class_names),
                pca_mean=stats.embedding.pca_mean,
                pca_components=stats.embedding.pca_components,
                centroids=stats.embedding.centroids,
                covariance_inv=stats.embedding.covariance_inv,
                cosine_calibration_mean=stats.embedding.cosine_calibration_mean,
                cosine_calibration_std=stats.embedding.cosine_calibration_std,
                knn_train_embeddings=stats.embedding.knn_train_embeddings,
                knn_train_labels=np.array(stats.embedding.knn_train_labels),
                mahalanobis_p_threshold=maha_threshold,
                cosine_threshold=cosine_thresh,
                knn_distance_threshold=knn_thresh,
                model_type=model_type,
                model_hidden_size=model_hidden_size,
                mahalanobis_threshold_status=stats.thresholds.mahalanobis_status,
                tfidf_vocabulary_terms=np.array(stats.lexical.vocabulary_terms),
                tfidf_idf=stats.lexical.idf,
                tfidf_centroids=stats.lexical.centroids,
                tfidf_cosine_calibration_mean=stats.lexical.cosine_calibration_mean,
                tfidf_cosine_calibration_std=stats.lexical.cosine_calibration_std,
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


def _load_embedding_stats(data: NpzFile) -> EmbeddingStats:
    return EmbeddingStats(
        pca_mean=data["pca_mean"],
        pca_components=data["pca_components"],
        centroids=data["centroids"],
        covariance_inv=data["covariance_inv"],
        cosine_calibration_mean=float(data["cosine_calibration_mean"]),
        cosine_calibration_std=float(data["cosine_calibration_std"]),
        knn_train_embeddings=data["knn_train_embeddings"],
        knn_train_labels=data["knn_train_labels"].tolist(),
    )


def _load_lexical_stats(data: NpzFile) -> LexicalStats:
    # "in data.files" -- not data.get() (npz's NpzFile has no .get) -- lets a
    # pre-TF-IDF ood_stats.npz (missing these keys entirely, not just empty) still load
    # instead of KeyError-ing every predict/serve call until it's regenerated.
    if "tfidf_vocabulary_terms" not in data.files:
        return LexicalStats()
    return LexicalStats(
        vocabulary_terms=data["tfidf_vocabulary_terms"].tolist(),
        idf=data["tfidf_idf"],
        centroids=data["tfidf_centroids"],
        cosine_calibration_mean=float(data["tfidf_cosine_calibration_mean"]),
        cosine_calibration_std=float(data["tfidf_cosine_calibration_std"]),
    )


def _load_thresholds(data: NpzFile) -> CalibratedThresholds:
    return CalibratedThresholds(
        mahalanobis_p=_optional_threshold(data["mahalanobis_p_threshold"])
        if "mahalanobis_p_threshold" in data.files
        else None,
        cosine=_optional_threshold(data["cosine_threshold"])
        if "cosine_threshold" in data.files
        else None,
        knn_distance=_optional_threshold(data["knn_distance_threshold"])
        if "knn_distance_threshold" in data.files
        else None,
        tfidf_cosine=_optional_threshold(data["tfidf_threshold"])
        if "tfidf_threshold" in data.files
        else None,
        mahalanobis_status=_threshold_status(data["mahalanobis_threshold_status"])
        if "mahalanobis_threshold_status" in data.files
        else "not_calibrated",
    )


def _load_metadata(data: NpzFile) -> ArtifactMetadata | None:
    if "model_type" not in data.files or "model_hidden_size" not in data.files:
        return None
    model_type = _optional_str(data["model_type"])
    model_hidden_size = _optional_int(data["model_hidden_size"])
    if model_type is None or model_hidden_size is None:
        return None
    return ArtifactMetadata(model_type=model_type, model_hidden_size=model_hidden_size)


def load_stats(path: Path) -> OodArtifact:
    """The .npz file's own keys are flat, exactly as they've always been -- this function
    (and save_stats) is the only place that translates them into OodArtifact's nested
    Python shape. format_version is absent on every ood_stats.npz written before this
    change; those are treated as version 1 and loaded via the same per-field
    backward-compatibility checks this code has always had (see each _load_* helper
    above) -- no already-committed artifact needs regenerating because of this refactor."""
    data = np.load(str(path), allow_pickle=False)
    format_version = int(data["format_version"]) if "format_version" in data.files else 1
    return OodArtifact(
        format_version=format_version,
        class_names=data["class_names"].tolist(),
        embedding=_load_embedding_stats(data),
        lexical=_load_lexical_stats(data),
        thresholds=_load_thresholds(data),
        metadata=_load_metadata(data),
    )
