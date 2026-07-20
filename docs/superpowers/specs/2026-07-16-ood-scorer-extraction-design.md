# OOD Scorer Extraction — Design Spec (Increment 1 of SRP/OCP Remediation)

## Motivation

A follow-up code review of `BertTunningClassifier` (`src/inference/classify.py`, 275 lines) confirmed the same SRP/OCP tension already spec'd in `docs/superpowers/specs/2026-07-16-svm-signal-srp-ocp-remediation-design.md`, framed as a **Large Class** / **Divergent Change** smell: the class handles transformer loading+inference, OOD artifact loading/validation/calibration-warnings, four OOD scores, SVM scoring+disagreement, threshold resolution, review routing, and response construction — eight distinct reasons to change in one class.

The review's recommendation is to extract `OodScorer`, an SVM reviewer component, and artifact loading/validation functions — **incrementally, beginning with the pure OOD-scoring block in `predict_text()`.** This spec covers exactly that first increment: `OodScorer` only. SVM extraction is explicitly deferred to a follow-up spec/increment, not bundled in here.

## Why this narrows the earlier spec, not replaces it

The earlier spec (`2026-07-16-svm-signal-srp-ocp-remediation-design.md`) designed a shared `EvidenceSignal[T]` Protocol covering *both* the OOD ensemble and the SVM reviewer, plus a unifying `load_evidence_signals()` loader. Building that shared Protocol now, with only one real implementer (`OodScorer`), would be premature — a Protocol's value comes from a second implementer proving its shape is actually right, and the earlier spec already flagged that OOD's `score()` needs differ from SVM's (OOD needs `text` for TF-IDF and `pred_idx` for k-NN; SVM needs neither). Extracting `OodScorer` alone first, then extracting the SVM component in a follow-up increment, lets the *real* shared shape emerge from two concrete implementations instead of being guessed upfront. `EvidenceSignal[T]`/`load_evidence_signals()` remain the target for increment 2; this increment doesn't build them yet.

## What "pure OOD-scoring block" means, concretely

Everything in the current `BertTunningClassifier` that exists only because `ood_stats.npz` exists:

- `_load_ood_stats()` (staticmethod)
- `_validate_ood_stats_class_mapping()` / `_validate_ood_stats_model_identity()`
- `_warn_on_uncalibrated_thresholds()`
- `_train_mahalanobis_distances` / `_tfidf_vectorizer` (both `@cached_property`, lazily computed once per process lifetime)
- The entire second half of `predict_text()` (`src/inference/classify.py:373-423`) — everything from `if self._ood_stats is None: return result` through the final `result.model_copy(...)` call

None of this touches SVM state, softmax, or the transformer model directly (only the embedding/text/predicted-index `predict_text()` already computed). It's a clean, self-contained slice.

## Design

### New class: `OodScorer` (`src/inference/classify.py`, or a new `src/inference/ood_scorer.py` — see open question below)

```python
class OodScorer:
    def __init__(self, stats: ClassEmbeddingStats) -> None:
        self._stats = stats

    @staticmethod
    def load(model_path: str) -> "OodScorer | None":
        """None when ood_stats.npz isn't present -- mirrors today's _load_ood_stats exactly,
        just wrapped in the scorer instead of returning a bare ClassEmbeddingStats."""
        stats_path = Path(model_path) / "ood_stats.npz"
        if not stats_path.exists():
            log.info("No ood_stats.npz found at %s — OOD scoring disabled", stats_path)
            return None
        log.info("Loaded OOD stats from %s", stats_path)
        return OodScorer(load_stats(stats_path))

    def validate(self, id2label: dict[int, str]) -> None:
        """Both today's _validate_ood_stats_class_mapping and _validate_ood_stats_model_identity,
        combined into one call -- same BertTunningError-raising behavior, same messages."""
        ...

    def warn_if_uncalibrated(self) -> None:
        """Today's _warn_on_uncalibrated_thresholds, verbatim logic, moved here since it only
        ever reads self._stats fields."""
        ...

    @cached_property
    def _train_mahalanobis_distances(self) -> npt.NDArray[np.float64]:
        """Same laziness as today's classifier-level cached_property -- computed once per
        OodScorer instance's lifetime (== once per loaded model, since one OodScorer is built
        at classifier construction), not per predict_text() call."""
        return compute_train_mahalanobis_distances(self._stats)

    @cached_property
    def _tfidf_vectorizer(self) -> "TfidfVectorizer | None":
        return build_tfidf_vectorizer(self._stats)

    def score(
        self, text: str, embedding: npt.NDArray[np.float64], pred_idx: int
    ) -> OodMetrics | None:
        """None when this specific prediction can't be scored (empty knn_train_embeddings --
        today's "OOD scoring disabled for this prediction" case), distinct from OodScorer
        itself being absent (load() returning None). Otherwise identical math and rounding
        to today's inline block."""
        train_distances = self._train_mahalanobis_distances
        if len(train_distances) == 0:
            log.warning(
                "ood_stats.npz has no k-NN training data (empty knn_train_embeddings) — "
                "OOD scoring disabled for this prediction"
            )
            return None
        tfidf_z = (
            tfidf_cosine_z_score(text, self._stats, self._tfidf_vectorizer)
            if self._tfidf_vectorizer is not None
            else float("nan")
        )
        squared_distance = mahalanobis_min_distance(embedding, self._stats)
        scores = OodScores(
            mahalanobis_p=empirical_survival_p_value(squared_distance, train_distances),
            cosine_z=cosine_z_score(embedding, self._stats),
            knn_distance=knn_mean_distance(embedding, self._stats, pred_idx, k=Settings.OOD_KNN_NEIGHBORS),
            tfidf_cosine_z=tfidf_z,
        )
        maha_p_theoretical = mahalanobis_chi2_p_value_from_distance(squared_distance, self._stats)
        thresholds = resolve_ood_thresholds(self._stats)
        in_distribution = not is_out_of_distribution(scores, thresholds)
        return OodMetrics(
            mahalanobis_p_value=round(scores.mahalanobis_p, 6),
            mahalanobis_p_value_theoretical=round(maha_p_theoretical, 6),
            cosine_z=round(scores.cosine_z, 4),
            knn_distance=round(scores.knn_distance, 4),
            tfidf_cosine_z=(None if np.isnan(scores.tfidf_cosine_z) else round(scores.tfidf_cosine_z, 4)),
            in_distribution=in_distribution,
        )
```

`score()` returning `OodMetrics | None` (rather than a separate wrapper type) works because `OodMetrics` already carries `in_distribution` as a field — `predict_text()` needs nothing else from this call. `None` here means something different from `OodScorer.load()` returning `None`: the scorer exists and is valid, but *this particular document* can't be scored (empty k-NN reference set) — the same distinction the current code already makes, just made explicit by the return type instead of an early `return result` buried in the middle of `predict_text()`.

### `BertTunningClassifier.__init__` (after)

```python
self._ood_scorer = OodScorer.load(model_path)
if self._ood_scorer is not None:
    self._ood_scorer.validate(self.model.config.id2label)
    self._ood_scorer.warn_if_uncalibrated()
```

Replaces today's `self._ood_stats = ...` / `_validate_ood_stats_class_mapping()` / `_validate_ood_stats_model_identity()` / `_warn_on_uncalibrated_thresholds()` — four call sites become one field assignment plus two conditional calls. SVM loading (`self._svm_classifiers = ...`, `_validate_svm_classifiers_class_mapping()`) is **untouched** in this increment — it stays exactly as it is today, since SVM extraction is the next increment, not this one.

### `predict_text()` (after)

```python
# ... softmax forward pass, svm scoring, result construction -- all unchanged ...

ood_metrics = (
    self._ood_scorer.score(text, cls_embedding, pred_idx) if self._ood_scorer is not None else None
)
if ood_metrics is None:
    return result
return result.model_copy(
    update={
        "ood_metrics": ood_metrics,
        "review_route": decide_review_route(
            confidence_tier=confidence_tier,
            ood_evidence=OodEvidence.from_in_distribution(in_distribution=ood_metrics.in_distribution),
            classifier_disagreement=classifier_disagreement,
        ),
    }
)
```

The `if ood_metrics is None: return result` line covers *both* today's early-return cases (`self._ood_stats is None` and the empty-k-NN-data case) with one check, since `OodScorer.score()` already collapses them into a single `None` result — `predict_text()` doesn't need to know or care which of the two reasons applies, only whether OOD evidence is available at all. This is itself a small, deliberate simplification enabled by the extraction (previously two separate `if`/`return result` blocks for what `predict_text()` treats identically either way).

## Open question for your revision

**File placement:** should `OodScorer` live in `src/inference/classify.py` itself (smallest diff, everything OOD-and-inference-related still in one file) or a new `src/inference/ood_scorer.py` (matches this project's existing pattern of one-concept-per-file — `svm_reviewer.py`, `embeddings.py` — and keeps `classify.py` itself shrinking, which is the whole point)? Leaning toward the new file, since `classify.py` being 275 lines is the problem being solved, and leaving `OodScorer` in the same file only partially addresses that — but flagging it as a decision rather than assuming.

## Out of scope (deferred to increment 2)

- SVM reviewer extraction (`SvmReviewer`/`SvmReviewSignal` from the earlier spec).
- `check_svm_agreement()` extraction.
- The shared `EvidenceSignal[T]` Protocol and `load_evidence_signals()` loader — premature with only one real implementer; revisit once increment 2 makes the SVM shape concrete too.
- Any change to `PredictResult`, `review_route`'s decision table, or any observable behavior — this is a pure internal restructuring, same principle as the earlier spec.

## Testing

- `OodScorer.load()`: returns `None` when `ood_stats.npz` absent, a valid instance otherwise — testable without constructing `BertTunningClassifier`.
- `OodScorer.validate()`: raises `BertTunningError` on class-mapping/model-identity mismatch, same two existing test scenarios, now exercisable directly against `OodScorer` instead of through a full classifier.
- `OodScorer.score()`: returns `None` on empty k-NN training data; returns correct `OodMetrics` otherwise — directly portable from today's `predict_text()`-level OOD tests, now targeting `OodScorer` instead.
- `BertTunningClassifier`/`predict_text()`'s existing test suite should pass with only the private-attribute path updated (`clf._ood_stats = ...` → `clf._ood_scorer = OodScorer(...)` or equivalent) — no assertion on `PredictResult`/`review_route` should need to change.
