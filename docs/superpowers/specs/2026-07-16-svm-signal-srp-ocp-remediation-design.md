# SVM Signal SRP/OCP Remediation — Design Spec

**Scope note:** this spec covers the interface design (Option C) only — the `EvidenceSignal` shape, the loader extraction, and how disagreement fits in. It does not cover the line-by-line implementation; that's the next step (`writing-plans`), once this design is approved.

## Motivation

A SOLID review of PR #51 (the SVM independent reviewer + classifier disagreement signal) found two related issues in `BertTunningClassifier` (`src/inference/classify.py`):

- **SRP:** the class owns model/tokenizer lifecycle, OOD-stats load+validate (two validators), SVM-classifiers load+validate, threshold-uncalibrated warnings, *and* prediction orchestration across three independent signals. Each is a separate reason to change.
- **OCP:** adding the SVM signal required directly editing `__init__` (two new lines) and `predict_text()`'s body (a new inline branch) — the same edit pattern every signal before it (Mahalanobis, cosine, k-NN, TF-IDF) already went through. Nothing about the class is "closed for modification" when a new signal shows up.

This spec addresses both, for all three signal-shaped things currently in `predict_text()`: the OOD ensemble, the SVM reviewer, and the disagreement check.

## Why disagreement isn't a peer of OOD/SVM

The OOD ensemble and the SVM reviewer share a real shape: given an embedding (and, for k-NN, the predicted class index), produce evidence. Disagreement doesn't fit that shape — it needs `label` (softmax's own pick) *and* the SVM signal's already-computed result, to compare them. It's a comparison between two other results, not an independent producer of evidence from the embedding. Forcing it into the same interface as OOD/SVM (tried as "Option A/B" in discussion) means either a fat context parameter most signals ignore, or a fake peer relationship where one "signal" secretly depends on another. Neither is honest about what the code does.

## Design

### `EvidenceSignal[T]` — a generic Protocol, for OOD ensemble and SVM only

```python
from typing import Protocol, TypeVar

T = TypeVar("T")

class EvidenceSignal(Protocol[T]):
    @staticmethod
    def load(model_path: str) -> "EvidenceSignal[T] | None": ...
    def validate(self, id2label: dict[int, str]) -> None: ...
    def score(self, embedding: npt.NDArray[np.float64], pred_idx: int) -> T: ...
```

Generic over its own result type `T`, so `OodEnsembleSignal` implements `EvidenceSignal[OodScoreResult]` and `SvmReviewSignal` implements `EvidenceSignal[SvmScoreResult]` — each independently typed, sharing only the structural contract (method names/shapes), not a forced-common return type. `load()` returns `None` when the backing artifact (`ood_stats.npz` / `svm_classifiers.joblib`) isn't present, mirroring the existing `_load_ood_stats`/`_load_svm_classifiers` behavior. `validate()` raises `BertTunningError` on a class-mapping mismatch, mirroring the existing `_validate_*` methods — moved onto each signal, not left on the classifier.

**Disclosed, accepted imperfection:** `score()`'s shared signature takes `pred_idx`, which `SvmReviewSignal` doesn't use (only the OOD ensemble's k-NN component needs it). This is the honest cost of sharing a signature across exactly two real implementers — a third parameter tier to eliminate one unused argument would be over-engineering for two callers. Rejected in favor of disclosing it plainly here rather than pretending the interface is perfectly clean.

`OodScoreResult` wraps what `OodMetrics` + the `in_distribution` bool already carry today — no new information, just owned by the signal instead of computed inline in `predict_text()`. `SvmScoreResult` wraps `svm_scores` (the dict) + `top_label` (today's `svm_top_label()` call), computed together since `top_label` is always derived from the same `score()` call's output.

### Disagreement: a plain function, not a class

```python
def check_svm_agreement(svm_result: SvmScoreResult | None, predicted_label: str) -> bool:
    """True if there's no SVM signal, or its top pick matches predicted_label."""
    return svm_result is None or svm_result.top_label == predicted_label
```

Not part of `EvidenceSignal` — it's a comparison consuming another signal's already-computed result, called explicitly in `predict_text()` *after* the SVM signal has been scored, not looped over generically. This keeps the honest asymmetry visible in the code instead of hidden inside a forced-uniform interface.

### SRP fix: extracting the loader

A free function (matching this project's existing functional-core convention — `ood.py`, `embeddings.py`, `svm_reviewer.py` are all plain function modules, not injected service classes):

```python
@dataclass(frozen=True)
class LoadedSignals:
    ood: OodEnsembleSignal | None
    svm: SvmReviewSignal | None

def load_evidence_signals(model_path: str, id2label: dict[int, str]) -> LoadedSignals:
    ...
```

Does the load → validate → (for OOD) warn-if-uncalibrated dance for each signal, reusing each signal's own `load()`/`validate()` methods. `_warn_on_uncalibrated_thresholds()`'s logic moves onto `OodEnsembleSignal` itself (it only ever reads OOD-stats fields), called from inside the loader right after `validate()` — so all OOD/SVM-artifact-specific logic leaves `BertTunningClassifier` entirely, not just the load/validate calls.

`BertTunningClassifier.__init__` shrinks to: load tokenizer/model (unchanged), then one call — `self._signals = load_evidence_signals(model_path, self.model.config.id2label)` — replacing today's five separate load/validate/warn calls.

### `predict_text()` after the refactor (illustrative, not final)

```python
# softmax forward pass -> label, pred_idx, confidence (unchanged)
ood_result = self._signals.ood.score(embedding, pred_idx) if self._signals.ood else None
svm_result = self._signals.svm.score(embedding, pred_idx) if self._signals.svm else None
agrees = check_svm_agreement(svm_result, label)
review_route = decide_review_route(
    confidence_tier=confidence_tier,
    ood_evidence=OodEvidence.from_result(ood_result),
    classifier_disagreement=not agrees,
)
```

Each signal is still called by name (`self._signals.ood`, `self._signals.svm`), not via a blind loop over a list — their result types differ (`OodScoreResult` needs unpacking into `OodMetrics` for `PredictResult`; `SvmScoreResult` needs unpacking into `svm_scores`/`svm_predicted_label`), so `predict_text()` still knows and handles each concretely. The OCP win is that *loading* a new signal type doesn't touch `__init__`/`load_evidence_signals`'s call sites beyond adding one field to `LoadedSignals` and one line to the loader — scoring orchestration in `predict_text()` will still need a new line for a genuinely new signal, because genuinely new signals produce genuinely different result shapes that need genuinely different handling. This spec doesn't claim to eliminate that; it claims to stop the *loading/validation* boilerplate (which really is uniform) from being duplicated by hand for every signal, and to get each signal's own logic out of `BertTunningClassifier`.

## Out of scope

- `foreign_municipality` — computed outside `predict_text()` entirely (in `_attach_metadata()`), unrelated to this refactor.
- No behavior change. `PredictResult`'s fields, `review_route`'s decision table, and every existing test's assertions on public behavior stay identical — this is a pure internal restructuring. Tests touching private attributes (`clf._ood_stats`, `clf._svm_classifiers`) will need updating to the new `clf._signals.ood`/`clf._signals.svm` shape, but their assertions on `PredictResult`/`review_route` don't change.
- Not attempting a fully uniform scoring loop (rejected Option A/B) — see "Why disagreement isn't a peer" above.
- Not touching `WandbLogger`/`log_svm_classifiers_results` (`src/wandb.py`) — out of scope, not implicated by the SOLID review.

## Testing (shape, not exhaustive)

- Each `EvidenceSignal` implementation (`OodEnsembleSignal`, `SvmReviewSignal`) unit-testable independently — `load()`/`validate()`/`score()` without constructing a full `BertTunningClassifier` or mocking a tokenizer/model. This is the concrete testability win from the SRP split.
- `load_evidence_signals()` tested for its own load/skip/validate/raise/warn behavior, decoupled from classifier construction.
- `check_svm_agreement()` tested directly as a pure function (already effectively covered today via `predict_text()`-level tests; this makes it testable in isolation too).
- `predict_text()`/`decide_review_route()` existing test suite should still pass with only private-attribute-path updates (`clf._ood_stats = ...` → `clf._signals = LoadedSignals(ood=..., svm=...)` or equivalent) — no assertion on `PredictResult` or `review_route` should need to change.
