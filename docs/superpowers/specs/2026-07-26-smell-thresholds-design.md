# Smell Thresholds: A Second, Decoupled Threshold Profile — Design Spec

**Amendment (post-implementation):** every `threshold_overrides`/`OodThresholdOverrides`/
`--ood-mahalanobis-p`/`--ood-cosine-z`/`--ood-knn-distance`/`--ood-tfidf-z` reference below
was removed during implementation — the user decided the ad hoc per-run CLI override layer
wasn't needed once `smell_thresholds.json` (the persisted profile) covers the real workflow.
`SmellThresholds.thresholds` is the only mechanism now; `OodScorer.score()` takes a single
`smell_thresholds: SmellThresholds | None` parameter, no override dict. Everything else in
this spec (the second breakdown, `resolve_smell_thresholds()`'s per-key fallback,
`low_svm_margin`, `--write-smell-thresholds`) shipped as designed below.

## Motivation

Piece 1 (`2026-07-23-1-risk-score-and-smells-design.md`) built `smells`/`risk_score` from the
*same* breakdown that drives `in_distribution` — deliberately, to guarantee the two could
never disagree. In real use against the ordenanzas batch, that guarantee turned out to be
too strong: the user found documents where the calibrated decision (`in_distribution=True`,
softmax confident) is almost certainly wrong, but every signal sits just inside its
calibrated threshold — e.g. a Mahalanobis empirical p-value of `0.008922` against a
calibrated/fallback threshold of `0.001`. The signal is *pointing* at the problem, just not
strongly enough to flip the official decision without also making the decision noisier for
everything else.

The user's own analysis (`review_router.py`, a standalone script built outside this repo)
already approximates a fix: a second, more permissive threshold set applied to the same raw
scores, surfacing documents worth a second look without changing what the model officially
decided. That script's thresholds are hand-picked and disconnected from real calibration,
which is the actual problem this spec fixes — give the same idea a real home: a second
threshold profile, calibrated the same way the decision thresholds are, but persisted and
used independently.

**What must NOT change:** `in_distribution`/`review_route` — the official decision — stay
computed exactly as today, from the existing calibrated thresholds in `ood_stats.npz`. This
spec only changes what `smells` is computed from.

**Also in scope:** a new `low_svm_margin` smell. The SVM reviewer's per-class margins
(`PredictResult.svm_scores`) are surfaced today but never thresholded — the user observed
cases where the SVM's own margin for its predicted class is weak, but nothing flags it
because `svm_agrees_with_prediction` is a binary top-label comparison, not a confidence
check on the margin itself.

## Prior art this spec is grounded in (surveyed before writing)

- **`ood_stats.npz` vs. a separate file** (`2026-07-16-ood-artifact-schema-versioning-design.md`):
  that spec explicitly keeps `svm_classifiers.joblib` out of the npz — *"a completely
  separate artifact, untouched by this refactor"* — because it's a genuinely independent
  concern. Smell thresholds are the same shape of decision: they don't describe the model's
  embedding space (what `OodArtifact` is for), they describe a *presentational, decoupled*
  policy layered on top of it. **This spec adds a new sibling file, not new `OodArtifact`
  fields.**
- **`OodScorer`'s SRP** (`2026-07-16-ood-scorer-extraction-design.md`): its stated reason to
  exist is *"everything that exists only because `ood_stats.npz` exists."* A second file with
  a different shape and a different lifecycle doesn't belong inside it — mirrors why the SVM
  reviewer got its own module (`svm_reviewer.py`) instead of being bolted onto `ood.py`. **A
  new small module, `src/smell_thresholds.py`, owns loading/saving/resolving this file.**
- **SVM margins are deliberately unthresholded today** (`2026-07-15-svm-independent-reviewer-design.md`):
  *"No calibration, no threshold... this signal doesn't gate any decision in this repo... raw
  evidence for Classiflow, not a decision made in this repo."* This spec is a considered,
  narrow reversal of that stance for exactly one thing: an optional, presentational smell.
  It still doesn't gate `in_distribution`/`review_route`, and still adds no
  `Settings.OOD_SVM_*` config — the threshold only exists if the user opts in by writing one
  to `smell_thresholds.json`. The original rationale ("nothing in this repo makes a decision
  from the raw margins") still holds for every *decision*; `smells` was never a decision, it's
  a label.
- **Per-signal calibration-status pattern** (`2026-07-18-ood-uncalibrated-threshold-visibility-design.md`):
  `OodCalibrationStatus` is a separate, parallel type from the threshold values themselves —
  this spec's `SmellThresholds` reuses that shape (values + independent per-signal status),
  not a single collapsed flag.
- **Registry specs** (`2026-07-23-2/3/4`, `-5`): no repository/service layering exists in this
  codebase yet (confirmed by `-5`, an explicitly non-implementable exploration doc) — this
  spec follows the same plain-module-per-concern convention as `svm_reviewer.py`/`ood.py`,
  not a new abstraction.

## Scope

- New file `smell_thresholds.json`, written next to `ood_stats.npz`/`svm_classifiers.joblib`
  (same directory as the model checkpoint). Optional artifact — its total absence changes
  nothing (see Backward compatibility).
- New module `src/smell_thresholds.py`: `SmellThresholds` load/save/resolve, mirroring
  `svm_reviewer.py`'s self-contained shape and `save_stats()`'s atomic-write pattern
  (temp file → verify load-back → `os.replace()`).
- `OodScorer.score()` computes `smells` from a **second** `ood_signal_breakdown()` call
  against the resolved smell thresholds — `in_distribution` keeps using the first
  (decision-threshold) breakdown, completely unchanged.
- New `low_svm_margin` smell, computed in `BertTunningClassifier.predict_text()` (where
  `svm_scores` is already computed), gated on `smell_thresholds.svm_margin` being set.
- `evaluate-ood-calibration` gets a new `--write-smell-thresholds` flag, sibling to the
  existing `--write-thresholds`, writing the same `--target-fp-rate`-derived suggestions
  into `smell_thresholds.json` instead of `ood_stats.npz`. Independently runnable — you can
  calibrate decision thresholds at 1% and smell thresholds at 5% in two separate invocations,
  or skip one entirely.
- **Correction to already-shipped code:** the `--ood-mahalanobis-p`/`--ood-cosine-z`/
  `--ood-knn-distance`/`--ood-tfidf-z` CLI flags added to `predict`/`predict-folder` in the
  prior session were wired to override the *decision* thresholds — which was the wrong
  target per this spec's own "decision must not change" premise. This spec re-points them to
  override the *resolved smell thresholds* instead, so a CLI override can only ever move
  `smells`, never `in_distribution`.

**Out of scope:**
- Auto-calibrating `svm_margin` from `evaluate-ood-calibration`. That command's held-out test
  split has embeddings/predictions but never loads `svm_classifiers.joblib` today, and
  wiring that in is a real, separate addition (loading a second artifact into a command that
  currently only touches `ood_stats.npz`'s inputs). `svm_margin` starts out settable only by
  hand-editing `smell_thresholds.json` or via a CLI override on a single `predict` run — a
  calibration command for it is a natural follow-up once the manual value proves useful, not
  bundled here.
- Any change to `review_router.py` (the user's standalone script) — out of this repo's scope
  regardless.
- Persistence of `smells`/`risk_score` to a database (that's the separate, already-scoped
  registry work, pieces 2-4).

## Design

### `SmellThresholds` (`src/schemas.py`, next to `CalibratedThresholds`)

```python
class SmellThresholds(BaseModel):
    """A second, deliberately decoupled threshold profile -- read from smell_thresholds.json
    (src/smell_thresholds.py), used ONLY to compute PredictResult.smells. Never read by
    is_out_of_distribution()/ood_signal_breakdown() when deciding in_distribution -- that
    still reads exclusively from OodArtifact.thresholds (CalibratedThresholds) via
    resolve_ood_thresholds(). Shape mirrors CalibratedThresholds deliberately (same four OOD
    fields, same independently-Optional/status pattern) plus one field CalibratedThresholds
    doesn't have: svm_margin, since the SVM reviewer's margin has never been thresholded
    anywhere until now -- see the SVM survey note in this spec's Motivation for why that's a
    deliberate, narrow exception, not scope creep back into the OOD ensemble."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    mahalanobis_p: float | None = None
    cosine: float | None = None
    knn_distance: float | None = None
    tfidf_cosine: float | None = None
    svm_margin: float | None = None
    # Mirrors CalibratedThresholds.mahalanobis_status -- same reason (the degenerate-floor
    # guard can refuse the suggested Mahalanobis value here too, on the same corpus).
    mahalanobis_status: Literal["not_calibrated", "calibrated", "refused_degenerate"] = (
        "not_calibrated"
    )
```

### `src/smell_thresholds.py` (new module)

```python
_FILENAME = "smell_thresholds.json"


def load_smell_thresholds(model_path: str) -> SmellThresholds | None:
    """None when smell_thresholds.json isn't present -- mirrors OodScorer.load()/
    load_svm_classifiers()'s identical "optional artifact" convention exactly."""
    path = Path(model_path) / _FILENAME
    if not path.exists():
        log.info("No smell_thresholds.json found at %s — smell thresholds fall back to this "
                  "model's decision thresholds", path)
        return None
    return SmellThresholds.model_validate_json(path.read_text())


def save_smell_thresholds(thresholds: SmellThresholds, model_path: str) -> None:
    """Atomic write -- temp file, verify load-back, os.replace() -- same pattern as
    save_stats()/save_svm_classifiers(), same reason: this file is read at every
    predict/predict-folder/serve startup, an interrupted write must never corrupt it."""
    ...


def resolve_smell_thresholds(
    smell_thresholds: SmellThresholds | None, decision_thresholds: OodThresholds
) -> OodThresholds:
    """Per-field fallback to the DECISION thresholds (not Settings.OOD_* directly) when
    smell_thresholds.json is absent or a specific field is unset -- so a model with no
    smell_thresholds.json produces IDENTICAL smells to today's behavior (the decision
    breakdown), zero silent change for anyone who hasn't opted in. svm_margin has no
    fallback slot in OodThresholds (a 4-field NamedTuple) -- callers needing it read
    smell_thresholds.svm_margin directly; None there means "no low_svm_margin smell is ever
    emitted", not a fallback to some other value (there's nothing today to fall back to)."""
    if smell_thresholds is None:
        return decision_thresholds
    return OodThresholds(
        mahalanobis_p=smell_thresholds.mahalanobis_p or decision_thresholds.mahalanobis_p,
        cosine_z=smell_thresholds.cosine or decision_thresholds.cosine_z,
        knn_distance=smell_thresholds.knn_distance or decision_thresholds.knn_distance,
        tfidf_cosine_z=smell_thresholds.tfidf_cosine or decision_thresholds.tfidf_cosine_z,
    )
```

### `OodScorer.score()` — a second breakdown, `in_distribution` untouched

```python
def score(
    self,
    text: str,
    embedding: npt.NDArray[np.float64],
    pred_idx: int,
    threshold_overrides: OodThresholdOverrides = NO_OOD_THRESHOLD_OVERRIDES,
    smell_thresholds: SmellThresholds | None = None,
) -> OodMetrics | None:
    ...
    decision_thresholds = resolve_ood_thresholds(self._stats)
    calibration_status = resolve_ood_calibration_status(self._stats)
    breakdown = ood_signal_breakdown(scores, decision_thresholds, calibration_status, ...)
    in_distribution = not any(breakdown)  # UNCHANGED -- reads only decision_thresholds

    effective_smell_thresholds = apply_ood_threshold_overrides(
        resolve_smell_thresholds(smell_thresholds, decision_thresholds), threshold_overrides
    )
    smell_breakdown = ood_signal_breakdown(scores, effective_smell_thresholds, calibration_status, ...)
    smells = _smells_from_breakdown(smell_breakdown)  # now driven by the SECOND breakdown
    ...
```

`threshold_overrides` (the CLI flags built in the prior session) move from decorating
`decision_thresholds` to decorating `effective_smell_thresholds` — the one-line change that
corrects last session's wrong target. Everything downstream of `OodThresholdOverrides` (the
plain `dict[str, float]` type, the CLI flag collection, `apply_ood_threshold_overrides()`)
is reused unchanged; only *which* thresholds it's layered onto moves.

`ood_signal_breakdown()` itself is unchanged — it already takes `thresholds` as a parameter,
this just calls it twice with two different `OodThresholds` values instead of once.

### `low_svm_margin` — `BertTunningClassifier.predict_text()`

```python
if smell_thresholds_obj is not None and smell_thresholds_obj.svm_margin is not None:
    predicted_margin = svm_scores_result.get(label)
    if predicted_margin is not None and predicted_margin < smell_thresholds_obj.svm_margin:
        smells.append("low_svm_margin")
```

Placed alongside the existing `low_confidence` smell (both are non-OOD, computed directly in
`predict_text()` rather than via `OodScorer`, since neither depends on `ood_stats.npz`).
`_SMELL_WEIGHTS` (`src/inference/pipeline.py`) gains `"low_svm_margin": 2` — same tier as
`foreign_municipality` (strong-but-not-OOD evidence), pending the user's confirmation this
weight feels right once tested against real documents.

### `evaluate-ood-calibration --write-smell-thresholds`

Reuses `build_calibration_report()`'s already-computed `suggested_maha_threshold`/
`suggested_cosine_threshold`/`suggested_knn_threshold`/`suggested_tfidf_threshold` (computed
once per run, at whatever `--target-fp-rate` was passed) — writes them into
`smell_thresholds.json` via `save_smell_thresholds()` instead of (or alongside) the existing
`_write_calibrated_thresholds()` call into `ood_stats.npz`. Same degenerate-floor guard on
Mahalanobis, same reasoning, applied to this second file's own `mahalanobis_status`.

Typical usage the user asked for:
```bash
# Decision thresholds: strict, 1% (unchanged, existing workflow)
uv run python main.py evaluate-ood-calibration ... --target-fp-rate 0.01 --write-thresholds

# Smell thresholds: permissive, 5%, separate file, does not touch the above
uv run python main.py evaluate-ood-calibration ... --target-fp-rate 0.05 --write-smell-thresholds
```

## How to test this piece

- A document identical to the `ordenanza_5720_1993.pdf` case from this session (Mahalanobis
  p just inside the decision threshold) should show `in_distribution=True` (unchanged) but
  `smells` containing `low_mahalanobis_p` once a smell-threshold profile calibrated at a
  looser target is written — the exact decoupling this spec exists for.
- A model with no `smell_thresholds.json` must produce byte-identical `smells` to today's
  behavior (piece 1) — the resolve-to-decision-thresholds fallback is the regression guard.
- `--ood-cosine-z` (etc.) on `predict`/`predict-folder` must change `smells` but never
  `in_distribution`/`review_route` for the same document — the specific bug this spec fixes
  in last session's shipped CLI flags.
- `low_svm_margin` appears only when `smell_thresholds.json` sets `svm_margin`, and only for
  the predicted class's own margin, not any other class's.

## Backward compatibility

- No existing file changes shape: `ood_stats.npz`/`CalibratedThresholds`/`OodArtifact` are
  completely untouched by this spec.
- `smell_thresholds.json` absent (every model today) → `resolve_smell_thresholds()` returns
  `decision_thresholds` unchanged → `smells` computed exactly as piece 1 shipped it. Existing
  `predict`/`predict-folder`/`serve` output is unaffected until a user opts in.
- `PredictResult.smells`/`risk_score` schema fields are unchanged (still `list[str]`/`int`) —
  only which threshold values decide membership in that list changes, and only when opted in.

## Open questions for spec review

1. **`low_svm_margin` weight** (`_SMELL_WEIGHTS`) — proposed `2`, same tier as
   `foreign_municipality`. Confirm, or wait until tested against real documents to pick a
   number.
2. **Should `--write-smell-thresholds` be its own flag, or should `--target-fp-rate` accept
   two values** (`--target-fp-rate 0.01 --smell-target-fp-rate 0.05`) so one invocation writes
   both files at once? This spec proposes two separate flags/invocations for simplicity and
   independent control (matches "confirm one, do the other later" from this session) — a
   combined flag is a small follow-up if two invocations prove annoying in practice.
3. **`svm_margin` calibration** — confirmed out of scope per above (manual/CLI-only for now).
   Flagging again here since it's the one piece of "both together" that isn't actually
   auto-calibrated in this pass, only settable by hand or per-run override.

## Touch list

| Path | Change |
|---|---|
| `src/schemas.py` | New `SmellThresholds` model |
| `src/smell_thresholds.py` | New module: load/save/resolve, atomic write |
| `src/ood.py` | No change to `OodArtifact`/`resolve_ood_thresholds()` — confirmed out of scope |
| `src/inference/ood_scorer.py` | `OodScorer.score()` gains `smell_thresholds` param, computes a second `ood_signal_breakdown()` call for `smells`; `threshold_overrides` re-targeted onto the smell breakdown instead of the decision one |
| `src/inference/classify.py` | `BertTunningClassifier` loads `smell_thresholds.json` at construction (mirrors `_load_svm_classifiers`); `predict_text()` computes `low_svm_margin` |
| `src/inference/pipeline.py` | `_SMELL_WEIGHTS` gains `"low_svm_margin"` |
| `src/cli/ood_calibration.py` | New `--write-smell-thresholds` flag |
| `src/cli/predict.py` | No new flags — existing `--ood-*` flags' target changes (now decorate smell thresholds, via `OodScorer.score()`'s re-pointed `threshold_overrides` param), no CLI-surface change |
| Tests | `tests/test_smell_thresholds.py` (new), `tests/inference/test_pipeline.py` (decoupling regression test), `tests/cli/test_ood_calibration.py` (`--write-smell-thresholds`) |
