"""smell_thresholds.json -- a second, deliberately decoupled threshold profile used only to
compute PredictResult.smells, never in_distribution/review_route. See
docs/superpowers/specs/2026-07-26-smell-thresholds-design.md for the full rationale. Its own
module, not folded into ood.py/OodScorer, for the same reason svm_reviewer.py is its own
module: a genuinely independent concern with a different artifact and a different lifecycle
from ood_stats.npz."""

import logging
from pathlib import Path

from src.ood import OodThresholds
from src.schemas import SmellThresholds

log = logging.getLogger(__name__)

_FILENAME = "smell_thresholds.json"


def load_smell_thresholds(model_path: str) -> SmellThresholds:
    """An empty SmellThresholds() (thresholds={}) when smell_thresholds.json isn't present --
    not Optional. An empty `thresholds` dict already falls back to the decision threshold for
    every key via resolve_smell_thresholds()'s dict.get(), the same way PredictResult.svm_scores
    defaults to {} rather than None elsewhere in this codebase -- there's no second thing an
    absent file needs to mean beyond "nothing customized", so a None state would only add a
    redundant check every caller has to repeat (see the classify.py Null Check smell finding
    docs/superpowers/specs/2026-07-26-smell-thresholds-design.md was correcting)."""
    path = Path(model_path) / _FILENAME
    if not path.exists():
        log.info(
            "No smell_thresholds.json found at %s — smell thresholds fall back to this "
            "model's decision thresholds",
            path,
        )
        return SmellThresholds()
    return SmellThresholds.model_validate_json(path.read_text())


def save_smell_thresholds(thresholds: SmellThresholds, model_path: str) -> None:
    """Atomic write -- temp file, verify load-back, replace -- same pattern as
    save_stats()/save_svm_classifiers(), same reason: this file is read at every
    predict/predict-folder/serve startup, an interrupted write must never corrupt it."""
    path = Path(model_path) / _FILENAME
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(thresholds.model_dump_json(indent=2))
        SmellThresholds.model_validate_json(tmp_path.read_text())  # fail fast on a bad write
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def resolve_smell_thresholds(
    smell_thresholds: SmellThresholds, decision_thresholds: OodThresholds
) -> OodThresholds:
    """Per-key fallback to the DECISION thresholds (not Settings.OOD_* directly) when a key is
    missing from `smell_thresholds.thresholds` -- an empty dict (no smell_thresholds.json, or
    a key simply never customized) falls back to every decision-threshold value, so a model
    with no smell_thresholds.json produces IDENTICAL smells to pre-existing behavior (the
    decision breakdown), zero silent change for anyone who hasn't opted in. dict.get() with a
    fallback default, not `or` -- a calibrated value of exactly 0.0 must not be treated as
    unset. svm_margin has no slot in OodThresholds (a 4-field NamedTuple, the OOD ensemble's
    own shape) -- callers needing it read smell_thresholds.thresholds.get("svm_margin") directly;
    a missing key there means "no low_svm_margin smell is ever emitted", not a fallback to
    some other value."""
    thresholds = smell_thresholds.thresholds
    return OodThresholds(
        mahalanobis_p=thresholds.get("mahalanobis_p", decision_thresholds.mahalanobis_p),
        cosine_z=thresholds.get("cosine", decision_thresholds.cosine_z),
        knn_distance=thresholds.get("knn_distance", decision_thresholds.knn_distance),
        tfidf_cosine_z=thresholds.get("tfidf_cosine", decision_thresholds.tfidf_cosine_z),
    )
