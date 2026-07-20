# OOD Uncalibrated-Threshold Visibility & Control — Design Spec

## Motivation

A PR #53 review comment flagged: `resolve_ood_thresholds()` (`src/ood.py`) silently
substitutes `Settings.OOD_*` per-field whenever a specific model's `ood_stats.npz` lacks a
calibrated threshold for that signal. `OodScorer.warn_if_uncalibrated()`
(`src/inference/ood_scorer.py`) logs one `WARNING` at classifier construction, but every
subsequent `predict_text()` call still evaluates and can flag `in_distribution=False` using
a threshold calibrated for a *different* model — with zero visibility to anything
downstream of the log line: the API response, the `predict-folder` CSV, the W&B table.
Reviewer's ask: represent per-signal calibration state explicitly (not just in logs), and
give operators a way to require calibration before a signal is allowed to fire, rather than
relying on logging alone to surface a hidden dependency.

## Two goals, confirmed with the user before writing this spec

1. **Visibility** — calibration state must be observable on `predict_text()`'s actual
   output (`OodMetrics`/`PredictResult`), not only in logs.
2. **Control** — an operator must be able to require calibration before an uncalibrated
   signal is allowed to contribute to the `in_distribution` decision. Default behavior must
   stay backward compatible (today's silent-fallback behavior, unchanged) unless the
   operator opts in.

## Rejected shape: `OodThresholds` fields becoming `float | None`

An earlier draft of this design considered making each `OodThresholds` field `float | None`
where `None` means "disabled by policy." Rejected after review: this overloads one field to
answer two unrelated questions — "what number to compare against" (always answerable, worst
case via the `Settings.OOD_*` fallback) and "is this specific model trustworthy enough on
this signal to act on" (a policy question, layered on top). `OodThresholds` has **21**
existing call sites (`src/wandb.py`, `src/cli/ood_calibration.py`,
`src/inference/ood_scorer.py`, tests) that all assume a concrete `float` today — forcing
every one of them to add a null-check for a policy concern most of them don't care about is
unnecessary churn for a problem a second, orthogonal type solves cleanly. This is the same
"don't let one field encode two kinds of missing" rule this codebase already applies
elsewhere (`svm_predicted_label`, `LexicalStats.is_fitted()`).

**Adopted shape:** leave `OodThresholds`/`resolve_ood_thresholds()` untouched — still four
concrete floats, always resolved. Add a **separate, parallel** per-signal calibration-status
type, generalizing the `mahalanobis_status` field that already exists on
`CalibratedThresholds` (`src/schema.py`) to all four signals instead of just one.

## Complete touch list

| File | What changes |
|---|---|
| `src/settings.py` | New `OOD_ALLOW_UNCALIBRATED_FALLBACK: bool = True` |
| `src/ood.py` | New `OodCalibrationStatus` NamedTuple + `resolve_ood_calibration_status()`; `is_out_of_distribution()` gains two new parameters (defaulted, backward compatible) |
| `src/schema.py` | `OodMetrics` gains four new fields (defaulted to `"calibrated"`/`None`, backward compatible) |
| `src/inference/ood_scorer.py` | `OodScorer.score()` computes and passes calibration status; `warn_if_uncalibrated()` refactored to reuse `resolve_ood_calibration_status()` instead of duplicating its own detection logic, and its message wording changes under strict mode |
| `src/wandb.py` | `_PREDICTION_COLUMNS` needs the four new field names added, or they silently won't appear in the `predict-folder --log-wandb` table (the CSV gets them automatically via `flatten_predict_result()` — this exact gap has bitten `svm_scores` and `svm_predicted_label` before) |
| `tests/inference/test_pipeline.py`, `tests/test_ood.py` | New tests for strict-mode gating |
| `tests/api/test_predict.py`, `tests/cli/test_commands.py`, `tests/test_schema.py`, `tests/test_wandb.py` | Contain the 9 existing direct `OodMetrics(...)`/`is_out_of_distribution(...)` call sites — confirmed via grep, none need edits since the new fields/params are all defaulted |

**Not touched:** `src/cli/ood_calibration.py` — `evaluate-ood-calibration` calls
`resolve_ood_thresholds()` directly for its own FP-rate measurement, never
`is_out_of_distribution()`, so it's unaffected by the new gating parameters.

## Design

### `OodCalibrationStatus` (new, `src/ood.py`)

```python
class OodCalibrationStatus(NamedTuple):
    """Companion to OodThresholds -- describes HOW each threshold was resolved (this
    model's own calibration vs. a Settings.OOD_* fallback), not the resolved number
    itself. Kept separate from OodThresholds so its 21 existing call sites, which only
    ever need the number, never have to learn about this orthogonal policy concept.
    tfidf_cosine is None (not "not_calibrated") when this model's lexical stats were
    never fitted at all -- mirrors OodMetrics.tfidf_cosine_z's own existing Optional
    convention for the identical reason."""

    mahalanobis: Literal["calibrated", "not_calibrated", "refused_degenerate"]
    cosine: Literal["calibrated", "not_calibrated"]
    knn_distance: Literal["calibrated", "not_calibrated"]
    tfidf_cosine: Literal["calibrated", "not_calibrated"] | None


def resolve_ood_calibration_status(stats: OodArtifact) -> OodCalibrationStatus:
    thresholds = stats.thresholds
    return OodCalibrationStatus(
        mahalanobis=thresholds.mahalanobis_status,
        cosine="calibrated" if thresholds.cosine is not None else "not_calibrated",
        knn_distance="calibrated" if thresholds.knn_distance is not None else "not_calibrated",
        tfidf_cosine=(
            None
            if not stats.lexical.is_fitted()
            else ("calibrated" if thresholds.tfidf_cosine is not None else "not_calibrated")
        ),
    )
```

### `is_out_of_distribution()` gains two backward-compatible parameters

```python
def is_out_of_distribution(
    scores: OodScores,
    thresholds: OodThresholds,
    calibration_status: OodCalibrationStatus = _ALL_CALIBRATED,
    *,
    allow_uncalibrated_fallback: bool = True,
) -> bool:
    ...
    maha_blocked = (
        not allow_uncalibrated_fallback and calibration_status.mahalanobis == "not_calibrated"
    )
    maha_anomalous = not maha_blocked and scores.mahalanobis_p < thresholds.mahalanobis_p

    cosine_blocked = (
        not allow_uncalibrated_fallback and calibration_status.cosine == "not_calibrated"
    )
    cosine_anomalous = not cosine_blocked and scores.cosine_z > thresholds.cosine_z

    knn_blocked = (
        not allow_uncalibrated_fallback and calibration_status.knn_distance == "not_calibrated"
    )
    knn_anomalous = not knn_blocked and (
        bool(np.isnan(scores.knn_distance)) or scores.knn_distance > thresholds.knn_distance
    )

    tfidf_blocked = (
        not allow_uncalibrated_fallback and calibration_status.tfidf_cosine == "not_calibrated"
    )
    tfidf_anomalous = not tfidf_blocked and (
        not np.isnan(scores.tfidf_cosine_z) and scores.tfidf_cosine_z > thresholds.tfidf_cosine_z
    )
    ...
```

Both new parameters default to "fully permissive" — `calibration_status` defaults to a
module-level constant,
`_ALL_CALIBRATED = OodCalibrationStatus(mahalanobis="calibrated", cosine="calibrated", knn_distance="calibrated", tfidf_cosine="calibrated")`,
and `allow_uncalibrated_fallback` defaults to `True` — so every one of the 8 existing test
call sites (which test the OR-firing mechanics themselves, not this new gating feature)
keeps passing unmodified.

**Refused-degenerate is never blocked, regardless of the flag.** `maha_blocked` only
matches `calibration_status.mahalanobis == "not_calibrated"` — `"refused_degenerate"` never
equals that, so Mahalanobis keeps firing via the `Settings.OOD_*` fallback under strict mode
too. This is intentional: `refused_degenerate` means calibration genuinely ran and its
degenerate-threshold guard correctly declined to persist a floor-adjacent value — a
legitimate, deliberate outcome, not a "never calibrated" gap. Both committed production
models (BETO v1/v2) are in this exact state for Mahalanobis today; blocking it under strict
mode would silently disable the one signal it's most reliably tuned for on the very models
that motivated building it.

**The k-NN NaN and TF-IDF NaN/None fail-open/fail-closed rules are untouched** — they're
gated by an *additional*, separate `and`, not replaced. A model whose lexical stats were
never fitted still fails open on TF-IDF the same way it does today, independent of the new
strict-mode flag.

### `OodMetrics` gains four new fields (`src/schema.py`)

```python
mahalanobis_calibration_status: Literal["calibrated", "not_calibrated", "refused_degenerate"] = (
    "calibrated"
)
cosine_calibration_status: Literal["calibrated", "not_calibrated"] = "calibrated"
knn_distance_calibration_status: Literal["calibrated", "not_calibrated"] = "calibrated"
tfidf_calibration_status: Literal["calibrated", "not_calibrated"] | None = None
```

Flat fields, not one nested status object — mirrors how `OodMetrics` already flattens
`OodScores`' four signal values (`mahalanobis_p_value`, `cosine_z`, `knn_distance`,
`tfidf_cosine_z`) rather than nesting them, so this follows the existing shape of the same
class instead of introducing a second nesting convention inside it.
`tfidf_calibration_status` mirrors `tfidf_cosine_z`'s own existing `Optional` — `None` for
the same reason (model predates/never fitted the TF-IDF signal), not a fifth state. Defaults
to `"calibrated"`/`None` so the 9 existing direct `OodMetrics(...)` test constructors need no
changes.

### `OodScorer.score()` (`src/inference/ood_scorer.py`)

```python
calibration_status = resolve_ood_calibration_status(self._stats)
thresholds = resolve_ood_thresholds(self._stats)
in_distribution = not is_out_of_distribution(
    scores,
    thresholds,
    calibration_status,
    allow_uncalibrated_fallback=Settings.OOD_ALLOW_UNCALIBRATED_FALLBACK,
)
return OodMetrics(
    ...,  # existing fields unchanged
    mahalanobis_calibration_status=calibration_status.mahalanobis,
    cosine_calibration_status=calibration_status.cosine,
    knn_distance_calibration_status=calibration_status.knn_distance,
    tfidf_calibration_status=calibration_status.tfidf_cosine,
)
```

The raw score values (`mahalanobis_p_value`, `cosine_z`, `knn_distance`, `tfidf_cosine_z`)
are **always computed and reported**, even for a blocked signal under strict mode — blocking
only prevents that signal from contributing to `in_distribution`, it doesn't hide the number.
This preserves the visibility goal even when control is exercised: an operator running
strict mode can still see what an uncalibrated signal *would have* said.

### `warn_if_uncalibrated()` refactor

Reuses `resolve_ood_calibration_status()` instead of its own hand-rolled `None`/status
checks (removes duplicated detection logic between the warning path and the new scoring
path). Message wording changes based on `Settings.OOD_ALLOW_UNCALIBRATED_FALLBACK`: today's
wording ("falling back to Settings.OOD_*") stays accurate when the flag is `True`; under
strict mode (`False`), the message changes to state the signal is **disabled**, since
falling back to `Settings.OOD_*` is no longer what actually happens for a `not_calibrated`
signal.

## Backward compatibility

- `OOD_ALLOW_UNCALIBRATED_FALLBACK` defaults to `True` — zero behavior change for BETO v1/v2
  or any existing deployment unless an operator explicitly opts into strict mode.
- New `OodMetrics` fields are additive with defaults — no existing field changes type, no
  existing direct `OodMetrics(...)` construction (9 test call sites) needs updating.
- New `is_out_of_distribution()` parameters are defaulted and backward compatible — no
  existing call site (8 in tests, 1 in production) needs updating unless it wants to
  exercise the new gating behavior specifically.
- `resolve_ood_thresholds()`/`OodThresholds` are completely untouched.

## Known follow-up, out of scope for this spec

New `OodMetrics` fields need an explicit addition to `_PREDICTION_COLUMNS`
(`src/wandb.py`) to appear in the `predict-folder --log-wandb` table — the CSV gets them
automatically via `flatten_predict_result()`'s `model_dump()`-based row, the W&B table does
not. This is the identical gap already hit twice before (`svm_scores`, then
`svm_predicted_label`/`svm_agrees_with_prediction`) — included in the touch list above as a
required part of this change, not deferred.
