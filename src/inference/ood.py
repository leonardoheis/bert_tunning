from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from pydantic import BaseModel, ConfigDict
from sklearn.decomposition import PCA
from transformers import PreTrainedTokenizerBase

from src.schema import ClassEmbeddingStats, Float64Array


class _PcaReduction(BaseModel):
    """Internal return type for _reduce_dimensionality — not part of the public schema."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    reduced: Float64Array
    mean: Float64Array
    components: Float64Array


def _reduce_dimensionality(embeddings: npt.NDArray[np.float64], n_components: int) -> _PcaReduction:
    capped = min(n_components, embeddings.shape[0] - 1, embeddings.shape[1])
    pca = PCA(n_components=capped)
    reduced = pca.fit_transform(embeddings)
    return _PcaReduction(reduced=reduced, mean=pca.mean_, components=pca.components_)


def _project(
    embedding: npt.NDArray[np.float64], stats: ClassEmbeddingStats
) -> npt.NDArray[np.float64]:
    return (embedding - stats.pca_mean) @ stats.pca_components.T


def _mahalanobis_min_distance_raw(
    point: npt.NDArray[np.float64],
    centroids: npt.NDArray[np.float64],
    covariance_inv: npt.NDArray[np.float64],
) -> float:
    diffs = centroids - point
    distances = np.einsum("kd,de,ke->k", diffs, covariance_inv, diffs)
    return float(np.min(distances))


def _cosine_min_distance_raw(
    point: npt.NDArray[np.float64], centroids: npt.NDArray[np.float64]
) -> float:
    point_norm = point / (np.linalg.norm(point) + 1e-9)
    centroid_norms = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
    similarities = centroid_norms @ point_norm
    return float(np.min(1.0 - similarities))


def compute_class_stats(
    embeddings: npt.NDArray[np.float64],
    labels: list[int],
    class_names: list[str],
    *,
    n_components: int = 64,
    covariance_epsilon: float = 1e-6,
) -> ClassEmbeddingStats:
    pca_result = _reduce_dimensionality(embeddings, n_components)
    reduced = pca_result.reduced
    labels_arr = np.asarray(labels)

    centroids = np.stack([reduced[labels_arr == k].mean(axis=0) for k in range(len(class_names))])
    centered = reduced - centroids[labels_arr]
    covariance = (centered.T @ centered) / reduced.shape[0]
    covariance_reg = covariance + covariance_epsilon * np.eye(covariance.shape[0])
    covariance_inv = np.linalg.inv(covariance_reg)

    maha_scores = np.array(
        [
            _mahalanobis_min_distance_raw(reduced[i], centroids, covariance_inv)
            for i in range(reduced.shape[0])
        ]
    )
    cosine_scores = np.array(
        [_cosine_min_distance_raw(reduced[i], centroids) for i in range(reduced.shape[0])]
    )

    return ClassEmbeddingStats(
        class_names=class_names,
        pca_mean=pca_result.mean,
        pca_components=pca_result.components,
        centroids=centroids,
        covariance_inv=covariance_inv,
        maha_calibration_mean=float(maha_scores.mean()),
        maha_calibration_std=float(maha_scores.std() + 1e-9),
        cosine_calibration_mean=float(cosine_scores.mean()),
        cosine_calibration_std=float(cosine_scores.std() + 1e-9),
    )


def mahalanobis_min_distance(
    embedding: npt.NDArray[np.float64], stats: ClassEmbeddingStats
) -> float:
    point = _project(embedding, stats)
    return _mahalanobis_min_distance_raw(point, stats.centroids, stats.covariance_inv)


def cosine_min_distance(embedding: npt.NDArray[np.float64], stats: ClassEmbeddingStats) -> float:
    point = _project(embedding, stats)
    return _cosine_min_distance_raw(point, stats.centroids)


def mahalanobis_z_score(embedding: npt.NDArray[np.float64], stats: ClassEmbeddingStats) -> float:
    """Mahalanobis distance to the nearest centroid, z-scored against the training set."""
    maha_raw = mahalanobis_min_distance(embedding, stats)
    return (maha_raw - stats.maha_calibration_mean) / stats.maha_calibration_std


def cosine_z_score(embedding: npt.NDArray[np.float64], stats: ClassEmbeddingStats) -> float:
    """Cosine distance to the nearest centroid, z-scored against the training set."""
    cosine_raw = cosine_min_distance(embedding, stats)
    return (cosine_raw - stats.cosine_calibration_mean) / stats.cosine_calibration_std


def save_stats(stats: ClassEmbeddingStats, path: Path) -> None:
    np.savez(
        str(path),
        class_names=np.array(stats.class_names),
        pca_mean=stats.pca_mean,
        pca_components=stats.pca_components,
        centroids=stats.centroids,
        covariance_inv=stats.covariance_inv,
        maha_calibration_mean=stats.maha_calibration_mean,
        maha_calibration_std=stats.maha_calibration_std,
        cosine_calibration_mean=stats.cosine_calibration_mean,
        cosine_calibration_std=stats.cosine_calibration_std,
    )


def load_stats(path: Path) -> ClassEmbeddingStats:
    data = np.load(str(path), allow_pickle=False)
    return ClassEmbeddingStats(
        class_names=data["class_names"].tolist(),
        pca_mean=data["pca_mean"],
        pca_components=data["pca_components"],
        centroids=data["centroids"],
        covariance_inv=data["covariance_inv"],
        maha_calibration_mean=float(data["maha_calibration_mean"]),
        maha_calibration_std=float(data["maha_calibration_std"]),
        cosine_calibration_mean=float(data["cosine_calibration_mean"]),
        cosine_calibration_std=float(data["cosine_calibration_std"]),
    )


def extract_embeddings(  # noqa: PLR0913
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    *,
    max_length: int,
    device: str,
    batch_size: int = 16,
) -> npt.NDArray[np.float64]:
    model.eval()
    batches: list[npt.NDArray[np.float64]] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(
                batch,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            hidden = model.base_model(**inputs).last_hidden_state  # type: ignore[operator]
            batches.append(hidden[:, 0, :].cpu().numpy().astype(np.float64))
    return np.vstack(batches)
