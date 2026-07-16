"""Per-class one-vs-rest SVM reviewer -- a fifth, independent signal, never folded into
the OOD ensemble (in_distribution). Exposes a full per-class score dict, mirroring
all_scores, for the downstream Classiflow agent to weigh itself. See
docs/superpowers/specs/2026-07-15-svm-independent-reviewer-design.md for the full
design rationale -- notably why this sits at src/ top level (used by both
training/pipeline.py and inference/classify.py, same reasoning as ood.py/embeddings.py)
and why there is deliberately no calibration/threshold here."""

from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC


def fit_svm_classifiers(
    embeddings: npt.NDArray[np.float64], labels: list[int], class_names: list[str]
) -> dict[str, SVC]:
    """One one-vs-rest SVC per class, trained on the raw (non-PCA-reduced) [CLS]
    embedding -- matches Peña et al. 2023's validated config for imbalanced topic
    classification, and avoids coupling this signal to OOD_PCA_COMPONENTS (a
    dimensionality chosen for Mahalanobis's covariance-estimation needs, which this
    signal doesn't share). class_weight="balanced" mirrors this project's existing
    imbalance handling (compute_class_weight("balanced", ...) in training/pipeline.py),
    relevant given classes like otro/declaracion_concejo_municipal have few samples."""
    labels_arr = np.asarray(labels)
    classifiers: dict[str, SVC] = {}
    for idx, name in enumerate(class_names):
        binary_labels = (labels_arr == idx).astype(int)
        svc = SVC(kernel="rbf", class_weight="balanced")
        svc.fit(embeddings, binary_labels)
        classifiers[name] = svc
    return classifiers


def save_svm_classifiers(classifiers: dict[str, SVC], path: Path) -> None:
    # Same atomic-write pattern as ood.py's save_stats -- a crash/kill mid-write must not
    # corrupt the only copy of this artifact, which predict/serve also read from.
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        joblib.dump(classifiers, tmp_path)
        load_svm_classifiers(tmp_path)  # fail fast on a corrupt/incomplete write
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_svm_classifiers(path: Path) -> dict[str, SVC] | None:
    if not path.exists():
        return None
    classifiers: dict[str, SVC] = joblib.load(path)
    return classifiers


def evaluate_svm_classifiers(
    classifiers: dict[str, SVC],
    embeddings: npt.NDArray[np.float64],
    labels: list[int],
    class_names: list[str],
) -> dict[str, float]:
    """Held-out balanced accuracy per class's one-vs-rest SVM, scored against embeddings
    the classifier was NOT fit on (the val split) -- a meaningful signal of how well each
    class's boundary generalizes, unlike in-sample accuracy which a class_weight="balanced"
    SVM can trivially inflate on its own training data. Balanced accuracy (average of
    per-class recall), not plain accuracy, because each one-vs-rest task is itself
    imbalanced (one class positive, every other class negative)."""
    labels_arr = np.asarray(labels)
    scores: dict[str, float] = {}
    for idx, name in enumerate(class_names):
        binary_labels = (labels_arr == idx).astype(int)
        predictions = classifiers[name].predict(embeddings)
        scores[name] = float(balanced_accuracy_score(binary_labels, predictions))
    return scores


def svm_scores(embedding: npt.NDArray[np.float64], classifiers: dict[str, SVC]) -> dict[str, float]:
    """Each class's one-vs-rest decision-function margin for this embedding -- positive
    means inside that class's SVM boundary, negative means outside. Not a probability,
    not calibrated, not combined into any decision here -- raw evidence for the
    downstream Classiflow agent to weigh itself."""
    point = embedding.reshape(1, -1)
    return {name: float(svc.decision_function(point)[0]) for name, svc in classifiers.items()}


def svm_top_label(scores: dict[str, float]) -> str:
    """The class whose one-vs-rest SVM scored this embedding highest -- the SVM
    reviewer's own "prediction," for comparison against softmax's argmax. See
    docs/superpowers/specs/2026-07-16-svm-softmax-disagreement-design.md."""
    return max(scores, key=lambda name: scores[name])
