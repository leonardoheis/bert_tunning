import logging
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest
import torch

from src.ood import (
    LoadedModel,
    compute_class_stats,
    compute_train_mahalanobis_distances,
    cosine_min_distance,
    cosine_z_score,
    empirical_survival_p_value,
    extract_embeddings,
    extract_embeddings_and_predictions,
    knn_mean_distance,
    load_stats,
    mahalanobis_chi2_p_value,
    mahalanobis_chi2_p_value_from_distance,
    mahalanobis_empirical_p_value,
    mahalanobis_min_distance,
    resolve_ood_thresholds,
    save_stats,
)
from src.schema import ClassEmbeddingStats
from src.settings import Settings


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


def test_mahalanobis_chi2_p_value_is_lower_for_far_point() -> None:
    # A low p-value means "unlikely to be in-distribution" — the far point should
    # score LOWER (more anomalous), not higher, unlike a distance/z-score metric.
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    known_point = embeddings[0]
    far_point = np.full(16, 100.0)
    maha_p_far = mahalanobis_chi2_p_value(far_point, stats)
    maha_p_known = mahalanobis_chi2_p_value(known_point, stats)
    assert maha_p_far < maha_p_known


def test_mahalanobis_chi2_p_value_is_bounded_between_zero_and_one() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    known_point = embeddings[0]
    far_point = np.full(16, 100.0)
    for point in (known_point, far_point):
        p_value = mahalanobis_chi2_p_value(point, stats)
        assert 0.0 <= p_value <= 1.0


def test_mahalanobis_chi2_p_value_from_distance_matches_embedding_based_call() -> None:
    # Feeding the from_distance variant a pre-computed distance must give the identical
    # result as the original embedding-based function -- the refactor that split the
    # distance computation out must not change the result.
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    far_point = np.full(16, 100.0)
    squared_distance = mahalanobis_min_distance(far_point, stats)
    assert mahalanobis_chi2_p_value_from_distance(squared_distance, stats) == pytest.approx(
        mahalanobis_chi2_p_value(far_point, stats)
    )


def test_empirical_survival_p_value_matches_hand_computed_rank() -> None:
    reference = np.array([1.0, 3.0, 5.0, 9.0])
    # 2 of 4 reference values are >= 5.0, so p = (2 + 1) / (4 + 1) = 0.6.
    assert empirical_survival_p_value(5.0, reference) == pytest.approx(0.6)


def test_empirical_survival_p_value_raises_on_empty_reference() -> None:
    # Silently returning 1.0 ("maximally normal") for no reference data would be a
    # fail-open bug — exactly backwards for an anomaly-detection signal.
    with pytest.raises(ValueError, match="empty"):
        empirical_survival_p_value(5.0, np.array([]))


def test_compute_train_mahalanobis_distances_returns_one_value_per_training_doc() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    distances = compute_train_mahalanobis_distances(stats)
    assert distances.shape == (len(embeddings),)
    assert np.all(distances >= 0.0)


def test_compute_train_mahalanobis_distances_uses_true_label_not_nearest_centroid() -> None:
    # A training point labeled class_a but physically much closer to class_b's centroid --
    # its distance must be measured against its TRUE centroid (class_a, far -> large
    # distance), not whichever centroid is nearest (class_b, close -> small distance).
    # This mirrors compute_class_stats()'s own covariance estimation
    # (centered = reduced - centroids[labels_arr]), which uses true labels too.
    stats = ClassEmbeddingStats(
        class_names=["class_a", "class_b"],
        pca_mean=np.zeros(2),
        pca_components=np.eye(2),
        centroids=np.array([[0.0, 0.0], [10.0, 0.0]]),
        covariance_inv=np.eye(2),
        cosine_calibration_mean=0.0,
        cosine_calibration_std=1.0,
        # labeled class_a (centroid [0,0]) but sits right next to class_b's centroid [10,0].
        knn_train_embeddings=np.array([[9.0, 0.0]]),
        knn_train_labels=[0],
    )
    distances = compute_train_mahalanobis_distances(stats)
    # True-label (class_a) squared distance: 9^2 = 81. Nearest-centroid (class_b) would
    # have been 1^2 = 1 -- if this assertion sees 1.0 instead of 81.0, the implementation
    # is using nearest-centroid instead of true-label distance.
    assert distances[0] == pytest.approx(81.0)


def test_mahalanobis_empirical_p_value_is_lower_for_far_point() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    train_distances = compute_train_mahalanobis_distances(stats)
    known_point = embeddings[0]
    far_point = np.full(16, 100.0)
    p_far = mahalanobis_empirical_p_value(far_point, stats, train_distances)
    p_known = mahalanobis_empirical_p_value(known_point, stats, train_distances)
    assert p_far < p_known


def test_mahalanobis_empirical_p_value_is_bounded_between_zero_and_one() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    train_distances = compute_train_mahalanobis_distances(stats)
    known_point = embeddings[0]
    far_point = np.full(16, 100.0)
    for point in (known_point, far_point):
        p_value = mahalanobis_empirical_p_value(point, stats, train_distances)
        assert 0.0 < p_value <= 1.0


def test_cosine_z_score_is_higher_for_far_point() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    known_point = embeddings[0]
    far_point = np.array([100.0 if i % 2 == 0 else -100.0 for i in range(16)])
    assert cosine_z_score(far_point, stats) > cosine_z_score(known_point, stats)


def test_save_and_load_stats_roundtrip(tmp_path: Path) -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    path = tmp_path / "ood_stats.npz"
    save_stats(stats, path)
    loaded = load_stats(path)
    assert loaded.class_names == stats.class_names
    np.testing.assert_allclose(loaded.centroids, stats.centroids)
    np.testing.assert_allclose(loaded.covariance_inv, stats.covariance_inv)
    assert loaded.cosine_calibration_mean == stats.cosine_calibration_mean


def test_knn_mean_distance_is_zero_for_a_training_point_itself() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    # embeddings[0] is a class_a point; its own class-conditional 10-NN distance
    # should be small (it's one of its own neighbors, distance 0 to itself).
    dist = knn_mean_distance(embeddings[0], stats, predicted_label_id=0, k=10)
    assert dist >= 0.0
    assert dist < 1.0  # class_a cluster has scale=0.1, so neighbor distances are small


def test_knn_mean_distance_is_larger_for_a_far_point() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    far_point = np.full(16, 100.0)
    near_dist = knn_mean_distance(embeddings[0], stats, predicted_label_id=0, k=10)
    far_dist = knn_mean_distance(far_point, stats, predicted_label_id=0, k=10)
    assert far_dist > near_dist


def test_knn_mean_distance_handles_k_larger_than_class_size() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    # class_a has 20 members in the fixture; request more neighbors than exist.
    dist = knn_mean_distance(embeddings[0], stats, predicted_label_id=0, k=1000)
    assert dist >= 0.0  # falls back to using all available class members, not an error


def test_knn_mean_distance_logs_warning_when_class_has_no_training_points(
    caplog: pytest.LogCaptureFixture,
) -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    with caplog.at_level(logging.WARNING, logger="src.ood"):
        dist = knn_mean_distance(embeddings[0], stats, predicted_label_id=99, k=10)
    assert np.isnan(dist)
    assert any("zero training points" in record.message for record in caplog.records)


def test_save_and_load_stats_roundtrip_includes_knn_fields(tmp_path: Path) -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    path = tmp_path / "ood_stats.npz"
    save_stats(stats, path)
    loaded = load_stats(path)
    np.testing.assert_allclose(loaded.knn_train_embeddings, stats.knn_train_embeddings)
    assert loaded.knn_train_labels == stats.knn_train_labels


def test_extract_embeddings_returns_correct_shape() -> None:
    tokenizer = MagicMock()
    tokenizer.return_value.to.return_value = {
        "input_ids": torch.zeros(2, 8, dtype=torch.long),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    model = MagicMock()
    model.base_model.return_value.last_hidden_state = torch.zeros(2, 8, 16)

    loaded = LoadedModel(model=model, tokenizer=tokenizer, device="cpu")
    embeddings = extract_embeddings(loaded, ["doc one", "doc two"], max_length=8, batch_size=2)
    assert embeddings.shape == (2, 16)


def test_save_and_load_stats_roundtrip_includes_thresholds() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8).model_copy(
        update={
            "mahalanobis_p_threshold": 0.001,
            "cosine_threshold": 13.7366,
            "knn_distance_threshold": 26.125,
        }
    )
    path = Path("test_stats_thresholds.npz")
    try:
        save_stats(stats, path)
        loaded = load_stats(path)
        assert loaded.mahalanobis_p_threshold == pytest.approx(0.001)
        assert loaded.cosine_threshold == pytest.approx(13.7366)
        assert loaded.knn_distance_threshold == pytest.approx(26.125)
    finally:
        path.unlink(missing_ok=True)


def test_save_and_load_stats_roundtrip_thresholds_default_to_none() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    path = Path("test_stats_no_thresholds.npz")
    try:
        save_stats(stats, path)
        loaded = load_stats(path)
        assert loaded.mahalanobis_p_threshold is None
        assert loaded.cosine_threshold is None
        assert loaded.knn_distance_threshold is None
    finally:
        path.unlink(missing_ok=True)


def test_save_stats_leaves_original_file_untouched_if_write_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    path = Path("test_stats_atomic_original.npz")
    try:
        # A real, previously-good file already exists at `path`.
        save_stats(stats, path)
        original_bytes = path.read_bytes()

        # Simulate a crash mid-write: np.savez succeeds but the written file is corrupt/
        # incomplete, so load_stats(tmp_path) inside save_stats must raise before the
        # original file is ever touched.
        def _broken_savez(*_args: object, **_kwargs: object) -> None:
            msg = "simulated write failure"
            raise OSError(msg)

        monkeypatch.setattr(np, "savez", _broken_savez)
        with pytest.raises(OSError, match="simulated write failure"):
            save_stats(stats, path)

        assert path.read_bytes() == original_bytes  # untouched
        assert not path.with_name(path.name + ".tmp").exists()  # tmp file cleaned up
    finally:
        path.unlink(missing_ok=True)
        path.with_name(path.name + ".tmp").unlink(missing_ok=True)


def test_save_stats_leaves_original_file_untouched_when_load_back_verification_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Unlike test_save_stats_leaves_original_file_untouched_if_write_fails (which fails
    # np.savez itself, before any bytes hit disk), this exercises the actual gap the
    # load-back verification step exists to close: np.savez succeeds and writes real bytes
    # to the tmp file, but load_stats(tmp_path) -- called from inside save_stats() to catch
    # a corrupt/incomplete write before it's ever promoted to the real path -- fails. A
    # regression that silently dropped that verification call would still leave both other
    # atomic-write tests passing; this one would catch it.
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    path = Path("test_stats_atomic_verify_fails.npz")
    try:
        save_stats(stats, path)
        original_bytes = path.read_bytes()

        real_load_stats = load_stats

        def _load_stats_that_rejects_tmp_files(p: Path) -> ClassEmbeddingStats:
            if str(p).endswith(".tmp"):
                msg = "simulated corrupt tmp file"
                raise ValueError(msg)
            return real_load_stats(p)

        monkeypatch.setattr("src.ood.load_stats", _load_stats_that_rejects_tmp_files)
        with pytest.raises(ValueError, match="simulated corrupt tmp file"):
            save_stats(stats, path)

        assert path.read_bytes() == original_bytes  # untouched -- no bad replace happened
        assert not path.with_name(path.name + ".tmp").exists()  # tmp file cleaned up
    finally:
        path.unlink(missing_ok=True)
        path.with_name(path.name + ".tmp").unlink(missing_ok=True)


def test_save_stats_does_not_leave_tmp_file_on_success() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    path = Path("test_stats_atomic_success.npz")
    try:
        save_stats(stats, path)
        assert path.exists()
        assert not path.with_name(path.name + ".tmp").exists()
    finally:
        path.unlink(missing_ok=True)


def test_load_stats_handles_legacy_file_without_threshold_fields() -> None:
    # A pre-this-change ood_stats.npz has no threshold keys at all (not even as NaN) --
    # load_stats must not KeyError, and must resolve all three to None.
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    path = Path("test_stats_legacy.npz")
    try:
        np.savez(
            str(path),
            class_names=np.array(stats.class_names),
            pca_mean=stats.pca_mean,
            pca_components=stats.pca_components,
            centroids=stats.centroids,
            covariance_inv=stats.covariance_inv,
            cosine_calibration_mean=stats.cosine_calibration_mean,
            cosine_calibration_std=stats.cosine_calibration_std,
            knn_train_embeddings=stats.knn_train_embeddings,
            knn_train_labels=np.array(stats.knn_train_labels),
        )
        loaded = load_stats(path)
        assert loaded.mahalanobis_p_threshold is None
        assert loaded.cosine_threshold is None
        assert loaded.knn_distance_threshold is None
    finally:
        path.unlink(missing_ok=True)


def test_resolve_ood_thresholds_falls_back_to_settings_when_stats_thresholds_none() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    thresholds = resolve_ood_thresholds(stats)
    assert thresholds.mahalanobis_p == Settings.OOD_MAHALANOBIS_P_THRESHOLD
    assert thresholds.cosine_z == Settings.OOD_COSINE_THRESHOLD
    assert thresholds.knn_distance == Settings.OOD_KNN_DISTANCE_THRESHOLD


def test_resolve_ood_thresholds_uses_stats_values_when_present() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8).model_copy(
        update={
            "mahalanobis_p_threshold": 0.002,
            "cosine_threshold": 5.0,
            "knn_distance_threshold": 10.0,
        }
    )
    thresholds = resolve_ood_thresholds(stats)
    assert thresholds.mahalanobis_p == pytest.approx(0.002)
    assert thresholds.cosine_z == pytest.approx(5.0)
    assert thresholds.knn_distance == pytest.approx(10.0)


def test_extract_embeddings_and_predictions_returns_matching_lengths() -> None:
    tokenizer = MagicMock()
    tokenizer.return_value.to.return_value = {
        "input_ids": torch.zeros(2, 8, dtype=torch.long),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    model = MagicMock()
    model.return_value.hidden_states = [torch.zeros(2, 8, 4)]
    model.return_value.logits = torch.tensor([[2.0, 0.5], [0.1, 3.0]])

    loaded = LoadedModel(model=model, tokenizer=tokenizer, device="cpu")
    embeddings, predicted_ids = extract_embeddings_and_predictions(
        loaded, ["doc one", "doc two"], max_length=8
    )
    assert embeddings.shape == (2, 4)
    assert predicted_ids == [0, 1]  # argmax of each row above
