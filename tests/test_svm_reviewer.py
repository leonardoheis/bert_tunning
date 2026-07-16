from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from src.svm_reviewer import (
    evaluate_svm_classifiers,
    fit_svm_classifiers,
    load_svm_classifiers,
    save_svm_classifiers,
    svm_scores,
)


def _synthetic_embeddings() -> tuple[npt.NDArray[np.float64], list[int], list[str]]:
    rng = np.random.default_rng(42)
    class_names = ["decreto", "ordenanza"]
    n_per_class = 20
    decreto = rng.normal(loc=0.0, scale=0.1, size=(n_per_class, 8))
    ordenanza = rng.normal(loc=5.0, scale=0.1, size=(n_per_class, 8))
    embeddings = np.vstack([decreto, ordenanza])
    labels = [0] * n_per_class + [1] * n_per_class
    return embeddings, labels, class_names


def test_fit_svm_classifiers_returns_one_per_class() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    classifiers = fit_svm_classifiers(embeddings, labels, class_names)
    assert set(classifiers.keys()) == set(class_names)


def test_svm_scores_returns_one_score_per_class() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    classifiers = fit_svm_classifiers(embeddings, labels, class_names)
    scores = svm_scores(np.zeros(8), classifiers)
    assert set(scores.keys()) == set(class_names)
    assert all(isinstance(v, float) for v in scores.values())


def test_svm_scores_prefers_the_class_the_point_actually_resembles() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    classifiers = fit_svm_classifiers(embeddings, labels, class_names)
    # A point near the "decreto" cluster (centered at 0) should score higher for decreto
    # than for ordenanza (centered at 5).
    scores = svm_scores(np.zeros(8), classifiers)
    assert scores["decreto"] > scores["ordenanza"]


def test_evaluate_svm_classifiers_returns_one_score_per_class() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    classifiers = fit_svm_classifiers(embeddings, labels, class_names)
    # Scored against the same well-separated clusters used for fitting -- not the point of
    # this test (that's the next one), just confirms the shape of the return value.
    scores = evaluate_svm_classifiers(classifiers, embeddings, labels, class_names)
    assert set(scores.keys()) == set(class_names)
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_evaluate_svm_classifiers_scores_high_on_well_separated_held_out_data() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    classifiers = fit_svm_classifiers(embeddings, labels, class_names)
    # A fresh sample from the same well-separated distributions as training -- held-out
    # balanced accuracy should be high, confirming the metric reflects real generalization,
    # not just training-set memorization.
    rng = np.random.default_rng(99)
    held_out_decreto = rng.normal(loc=0.0, scale=0.1, size=(10, 8))
    held_out_ordenanza = rng.normal(loc=5.0, scale=0.1, size=(10, 8))
    held_out_embeddings = np.vstack([held_out_decreto, held_out_ordenanza])
    held_out_labels = [0] * 10 + [1] * 10

    scores = evaluate_svm_classifiers(
        classifiers, held_out_embeddings, held_out_labels, class_names
    )
    assert all(v > 0.9 for v in scores.values())  # noqa: PLR2004


def test_save_and_load_svm_classifiers_roundtrip(tmp_path: Path) -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    classifiers = fit_svm_classifiers(embeddings, labels, class_names)
    path = tmp_path / "svm_classifiers.joblib"
    save_svm_classifiers(classifiers, path)

    loaded = load_svm_classifiers(path)
    assert loaded is not None
    assert set(loaded.keys()) == set(class_names)
    original_scores = svm_scores(np.zeros(8), classifiers)
    loaded_scores = svm_scores(np.zeros(8), loaded)
    assert original_scores == loaded_scores


def test_load_svm_classifiers_returns_none_when_missing(tmp_path: Path) -> None:
    assert load_svm_classifiers(tmp_path / "does_not_exist.joblib") is None


def test_save_svm_classifiers_leaves_original_file_untouched_if_write_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    classifiers = fit_svm_classifiers(embeddings, labels, class_names)
    path = tmp_path / "svm_classifiers.joblib"
    save_svm_classifiers(classifiers, path)
    original_bytes = path.read_bytes()

    def _broken_dump(*_args: object, **_kwargs: object) -> None:
        msg = "simulated write failure"
        raise OSError(msg)

    monkeypatch.setattr("src.svm_reviewer.joblib.dump", _broken_dump)
    with pytest.raises(OSError, match="simulated write failure"):
        save_svm_classifiers(classifiers, path)

    assert path.read_bytes() == original_bytes  # untouched
    assert not path.with_name(path.name + ".tmp").exists()  # tmp file cleaned up
