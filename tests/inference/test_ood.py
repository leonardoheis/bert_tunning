from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import torch

from src.inference.ood import (
    compute_class_stats,
    cosine_min_distance,
    extract_embeddings,
    load_stats,
    mahalanobis_min_distance,
    ood_score,
    save_stats,
)


def _synthetic_embeddings() -> tuple[npt.NDArray[np.float64], list[int], list[str]]:
    rng = np.random.default_rng(42)
    class_a = rng.normal(loc=0.0, scale=0.1, size=(20, 16))
    class_b = rng.normal(loc=5.0, scale=0.1, size=(20, 16))
    embeddings = np.vstack([class_a, class_b])
    labels = [0] * 20 + [1] * 20
    return embeddings, labels, ["class_a", "class_b"]


def test_compute_class_stats_shapes() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    assert stats.centroids.shape == (2, 8)
    assert stats.covariance_inv.shape == (8, 8)


def test_in_distribution_point_has_lower_mahalanobis_distance_than_far_point() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    known_point = embeddings[0]
    far_point = np.full(16, 100.0)
    known_distance = mahalanobis_min_distance(known_point, stats)
    far_distance = mahalanobis_min_distance(far_point, stats)
    assert far_distance > known_distance


def test_in_distribution_point_has_lower_cosine_distance_than_far_point() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    known_point = embeddings[0]
    # A uniform far_point (e.g. all -100) projects almost entirely onto PC1 — the
    # axis separating the two synthetic classes, since they differ by a uniform
    # shift across all dims — which makes it MORE cosine-aligned with a centroid
    # than an in-distribution point (whose direction is perturbed by noise on the
    # other PCA axes). An alternating-sign vector is off that axis and genuinely
    # far in cosine terms.
    far_point = np.array([100.0 if i % 2 == 0 else -100.0 for i in range(16)])
    known_distance = cosine_min_distance(known_point, stats)
    far_distance = cosine_min_distance(far_point, stats)
    assert far_distance > known_distance


def test_ood_score_is_higher_for_far_point() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    known_point = embeddings[0]
    far_point = np.full(16, 100.0)
    assert ood_score(far_point, stats) > ood_score(known_point, stats)


def test_save_and_load_stats_roundtrip(tmp_path: Path) -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    path = tmp_path / "ood_stats.npz"
    save_stats(stats, path)
    loaded = load_stats(path)
    assert loaded.class_names == stats.class_names
    np.testing.assert_allclose(loaded.centroids, stats.centroids)
    np.testing.assert_allclose(loaded.covariance_inv, stats.covariance_inv)
    assert loaded.maha_calibration_mean == stats.maha_calibration_mean


def test_extract_embeddings_returns_correct_shape() -> None:
    tokenizer = MagicMock()
    tokenizer.return_value.to.return_value = {
        "input_ids": torch.zeros(2, 8, dtype=torch.long),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    model = MagicMock()
    model.base_model.return_value.last_hidden_state = torch.zeros(2, 8, 16)

    embeddings = extract_embeddings(
        model, tokenizer, ["doc one", "doc two"], max_length=8, device="cpu", batch_size=2
    )
    assert embeddings.shape == (2, 16)
