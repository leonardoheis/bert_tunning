import contextlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.ood import OodThresholds
from src.schemas import SmellThresholds
from src.smell_thresholds import (
    load_smell_thresholds,
    resolve_smell_thresholds,
    save_smell_thresholds,
)

_DECISION = OodThresholds(
    mahalanobis_p=0.001, cosine_z=2.5, knn_distance=26.125, tfidf_cosine_z=2.5
)


def test_resolve_smell_thresholds_returns_decision_thresholds_when_empty() -> None:
    """No smell_thresholds.json (an empty SmellThresholds()) -> smells must be computed
    identically to the pre-existing decision-only behavior -- zero silent change for anyone
    who hasn't opted in."""
    resolved = resolve_smell_thresholds(SmellThresholds(), _DECISION)
    assert resolved == _DECISION


def test_resolve_smell_thresholds_overrides_only_set_keys() -> None:
    smell_thresholds = SmellThresholds(thresholds={"cosine": 1.5})
    resolved = resolve_smell_thresholds(smell_thresholds, _DECISION)
    assert resolved.cosine_z == 1.5  # noqa: PLR2004
    # Every other signal keeps the decision threshold unchanged.
    assert resolved.mahalanobis_p == _DECISION.mahalanobis_p
    assert resolved.knn_distance == _DECISION.knn_distance
    assert resolved.tfidf_cosine_z == _DECISION.tfidf_cosine_z


def test_resolve_smell_thresholds_respects_a_genuine_zero_value() -> None:
    """dict.get(key, default), not `or` -- a calibrated smell threshold of exactly 0.0 must
    not be silently treated as unset and fall back to the decision threshold."""
    smell_thresholds = SmellThresholds(thresholds={"mahalanobis_p": 0.0})
    resolved = resolve_smell_thresholds(smell_thresholds, _DECISION)
    assert resolved.mahalanobis_p == 0.0


def test_save_and_load_smell_thresholds_roundtrip(tmp_path: Path) -> None:
    thresholds = SmellThresholds(
        thresholds={"cosine": 1.5, "knn_distance": 12.0, "svm_margin": 0.2},
        mahalanobis_status="calibrated",
    )
    save_smell_thresholds(thresholds, str(tmp_path))
    loaded = load_smell_thresholds(str(tmp_path))
    assert loaded == thresholds


def test_load_smell_thresholds_returns_empty_when_missing(tmp_path: Path) -> None:
    assert load_smell_thresholds(str(tmp_path)) == SmellThresholds()


def test_smell_thresholds_rejects_an_unknown_key(tmp_path: Path) -> None:
    """A typo'd key (e.g. "cosine_z" instead of "cosine") must raise, not silently no-op
    through dict.get()'s fallback default with no error anywhere -- the whole reason
    SmellThresholds.thresholds is typed dict[SmellSignalKey, float], not dict[str, float]."""
    path = tmp_path / "smell_thresholds.json"
    path.write_text('{"thresholds": {"cosine_z": 1.0}, "mahalanobis_status": "calibrated"}')
    with pytest.raises(ValidationError):
        load_smell_thresholds(str(tmp_path))


def test_save_smell_thresholds_leaves_original_file_untouched_if_write_fails(
    tmp_path: Path,
) -> None:
    original = SmellThresholds(thresholds={"cosine": 1.0})
    save_smell_thresholds(original, str(tmp_path))

    class _Boom(SmellThresholds):
        def model_dump_json(self, **_kwargs: object) -> str:
            msg = "boom"
            raise RuntimeError(msg)

    broken = _Boom(thresholds={"cosine": 99.0})
    with contextlib.suppress(RuntimeError):
        save_smell_thresholds(broken, str(tmp_path))

    assert load_smell_thresholds(str(tmp_path)) == original
    assert not (tmp_path / "smell_thresholds.json.tmp").exists()
