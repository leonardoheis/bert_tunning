# Smell Review Suggested Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `PredictResult.smell_review_suggested: bool` — `True` when a document's
`risk_score` exceeds a new configurable threshold, independent of `review_route`, so a
human reviewing batch results can filter/sort on "the smell profile disagrees with the
official decision" directly instead of reading the `risk_score` column by hand.

**Architecture:** One new pure function (`_compute_smell_review_suggested()`) in
`src/inference/pipeline.py`, called from the existing `attach_metadata()` seam where
`risk_score` is already finalized. The new boolean then follows the exact same
already-established fan-out path `risk_score`/`smells` took: `PredictResult` →
`PredictResponse` (API) → `_PREDICTION_COLUMNS` (W&B) → `predict-folder` CSV (automatic)
→ frontend (`types/api.ts`/`utils/flatten.ts`/`utils/csv.ts`) → `predict` CLI output.

**Tech Stack:** Python 3.10+, Pydantic v2, pytest, Click, FastAPI, TypeScript/React
(frontend, typecheck-only — no frontend test runner exists in this repo, `tsc -b` +
`eslint` are the verification step for that task).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-27-smell-review-suggested-design.md` — every task
  below implements one row of that spec's Touch List, plus one gap the spec's touch list
  missed (see Task 3).
- `risk_score > Settings.SMELL_REVIEW_RISK_SCORE_THRESHOLD` — **strict** `>`, not `>=`.
- `Settings.SMELL_REVIEW_RISK_SCORE_THRESHOLD: int = 5`, env-overridable, not placed under
  the `OOD_*` prefix (it's a review-policy value, not an OOD signal).
- `smell_review_suggested` stays `False` whenever `ood_metrics is not None and
  ood_metrics.in_distribution is False` — that path already gets
  `review_route="human_review"` today, so this flag adds nothing there.
  `ood_metrics is None` (no `ood_stats.npz` loaded) behaves like `in_distribution=True`.
- Deliberately independent of `review_route`/`classifier_disagreement` — do not read
  either when computing this field.
- Run `uv run poe check` (lint + mypy strict + pytest) after every Python task before
  committing. Every existing test must keep passing — this is a purely additive field.

---

### Task 1: `Settings` threshold + `PredictResult` field (schema only, no logic yet)

**Files:**
- Modify: `src/settings.py` (add one setting, near line 98-108's threshold block)
- Modify: `src/schemas.py:120` (add one field, right after `risk_score`)
- Test: `tests/test_settings_ood.py`, `tests/test_schemas.py`

**Interfaces:**
- Produces: `Settings.SMELL_REVIEW_RISK_SCORE_THRESHOLD: int` (default `5`) —
  `_compute_smell_review_suggested()` (Task 2) reads this.
- Produces: `PredictResult.smell_review_suggested: bool` (default `False`) — every later
  task reads/forwards this field by exactly this name.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_settings_ood.py`:

```python
def test_smell_review_risk_score_threshold_default() -> None:
    assert Settings.SMELL_REVIEW_RISK_SCORE_THRESHOLD == 5  # noqa: PLR2004
```

Add to `tests/test_schemas.py` (near `test_predict_result_svm_predicted_label_defaults_to_empty_string`):

```python
def test_predict_result_smell_review_suggested_defaults_to_false() -> None:
    result = PredictResult(label="decreto", confidence=0.9, certain=True)
    assert result.smell_review_suggested is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_settings_ood.py::test_smell_review_risk_score_threshold_default tests/test_schemas.py::test_predict_result_smell_review_suggested_defaults_to_false -v`
Expected: both FAIL — `AttributeError: 'PredictResult' object has no attribute
'smell_review_suggested'` and `AttributeError: type object '_Settings' has no attribute
'SMELL_REVIEW_RISK_SCORE_THRESHOLD'`.

- [ ] **Step 3: Add the setting**

In `src/settings.py`, add this line right after `TARGET_FP_RATE: float = 0.01` (line 99),
before the `OOD_TFIDF_COSINE_THRESHOLD` block — it's a sibling threshold value, not an
OOD-prefixed one:

```python
    # Review-policy cutoff over the AGGREGATE risk_score (weighted sum of smells), not a
    # per-signal OOD threshold -- deliberately not under the OOD_* prefix. See
    # docs/superpowers/specs/2026-07-27-smell-review-suggested-design.md.
    SMELL_REVIEW_RISK_SCORE_THRESHOLD: int = 5
```

- [ ] **Step 4: Add the field**

In `src/schemas.py`, right after `risk_score: int = 0` (the last line of the
`PredictResult` class, currently line 120), add:

```python
    # True when risk_score alone (independent of review_route/classifier_disagreement)
    # crosses Settings.SMELL_REVIEW_RISK_SCORE_THRESHOLD, for documents the official
    # decision didn't already send to a human. False is the not-yet-computed default, same
    # convention as risk_score's own 0. See
    # docs/superpowers/specs/2026-07-27-smell-review-suggested-design.md.
    smell_review_suggested: bool = False
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_settings_ood.py::test_smell_review_risk_score_threshold_default tests/test_schemas.py::test_predict_result_smell_review_suggested_defaults_to_false -v`
Expected: both PASS

- [ ] **Step 6: Full check and commit**

Run: `uv run poe check`
Expected: all lint/mypy/tests pass (this is a purely additive field with a default, so
every existing `PredictResult(...)` construction keeps working).

```bash
git add src/settings.py src/schemas.py tests/test_settings_ood.py tests/test_schemas.py
git commit -m "feat: add SMELL_REVIEW_RISK_SCORE_THRESHOLD setting and PredictResult.smell_review_suggested field"
```

---

### Task 2: `_compute_smell_review_suggested()` + wire into `attach_metadata()`

**Files:**
- Modify: `src/inference/pipeline.py` (new function + `attach_metadata()`, lines 51-68)
- Test: `tests/inference/test_pipeline.py`

**Interfaces:**
- Consumes: `Settings.SMELL_REVIEW_RISK_SCORE_THRESHOLD` (Task 1), `PredictResult`,
  `OodMetrics` (`src/schemas.py`, already exists).
- Produces: `_compute_smell_review_suggested(risk_score: int, ood_metrics: OodMetrics |
  None) -> bool` — a private, pure function. Not tested directly (matching this
  codebase's own convention — `_compute_risk_score` is never unit-tested in isolation
  either, only through `predict_pdf`'s public entry point); tested here via `predict_pdf`.

This codebase has no test runner wired to call private helpers directly for this module —
follow the existing pattern in `tests/inference/test_pipeline.py`
(`test_predict_pdf_adds_foreign_municipality_smell_and_risk_score`,
`test_predict_pdf_returns_extraction_failed_result_when_text_missing`): mock
`BertTunningClassifier.predict_text` to return a `PredictResult` with `smells`/`ood_metrics`
pre-set, run it through `predict_pdf()`, assert on the final `result.smell_review_suggested`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/inference/test_pipeline.py`. First, add `OodMetrics` to the existing
`from src.schemas import (...)` block (it currently imports `ArtifactMetadata,
EmbeddingStats, ExtractionMetadata, LexicalStats, OodArtifact, PredictResult,
SmellThresholds` — add `OodMetrics` alphabetically after `LexicalStats`). Then add:

```python
def test_predict_pdf_smell_review_suggested_true_above_threshold_when_in_distribution() -> None:
    fake_extraction = ExtractionMetadata(
        text="hola mundo", extractor_used="MarkItDownExtractor", char_count=10
    )
    fake_result = PredictResult(
        label="decreto",
        confidence=0.9,
        certain=True,
        smells=["low_mahalanobis_p", "high_cosine_z"],  # weights 3+3=6, over the default 5
        ood_metrics=OodMetrics(
            mahalanobis_p_value=0.0001,
            mahalanobis_p_value_theoretical=0.0001,
            cosine_z=5.0,
            knn_distance=1.0,
            in_distribution=True,
        ),
    )
    with (
        patch("src.inference.pipeline.extract_pdf_with_metadata", return_value=fake_extraction),
        patch("src.inference.pipeline.BertTunningClassifier") as mock_clf_cls,
    ):
        mock_clf = MagicMock()
        mock_clf.predict_text.return_value = fake_result
        mock_clf_cls.return_value = mock_clf
        result = predict_pdf("fake/model", "doc.pdf")

    assert result.risk_score == 6  # noqa: PLR2004
    assert result.smell_review_suggested is True


def test_predict_pdf_smell_review_suggested_false_at_exact_threshold() -> None:
    fake_extraction = ExtractionMetadata(
        text="hola mundo", extractor_used="MarkItDownExtractor", char_count=10
    )
    fake_result = PredictResult(
        label="decreto",
        confidence=0.9,
        certain=True,
        smells=["low_mahalanobis_p", "low_svm_margin"],  # weights 3+2=5, exactly the threshold
        ood_metrics=OodMetrics(
            mahalanobis_p_value=0.0001,
            mahalanobis_p_value_theoretical=0.0001,
            cosine_z=0.0,
            knn_distance=1.0,
            in_distribution=True,
        ),
    )
    with (
        patch("src.inference.pipeline.extract_pdf_with_metadata", return_value=fake_extraction),
        patch("src.inference.pipeline.BertTunningClassifier") as mock_clf_cls,
    ):
        mock_clf = MagicMock()
        mock_clf.predict_text.return_value = fake_result
        mock_clf_cls.return_value = mock_clf
        result = predict_pdf("fake/model", "doc.pdf")

    assert result.risk_score == 5  # noqa: PLR2004
    assert result.smell_review_suggested is False


def test_predict_pdf_smell_review_suggested_false_when_in_distribution_false() -> None:
    fake_extraction = ExtractionMetadata(
        text="hola mundo", extractor_used="MarkItDownExtractor", char_count=10
    )
    fake_result = PredictResult(
        label="decreto",
        confidence=0.9,
        certain=True,
        smells=["low_mahalanobis_p", "high_cosine_z"],  # weights 3+3=6, over the default 5
        ood_metrics=OodMetrics(
            mahalanobis_p_value=0.0001,
            mahalanobis_p_value_theoretical=0.0001,
            cosine_z=5.0,
            knn_distance=1.0,
            in_distribution=False,
        ),
    )
    with (
        patch("src.inference.pipeline.extract_pdf_with_metadata", return_value=fake_extraction),
        patch("src.inference.pipeline.BertTunningClassifier") as mock_clf_cls,
    ):
        mock_clf = MagicMock()
        mock_clf.predict_text.return_value = fake_result
        mock_clf_cls.return_value = mock_clf
        result = predict_pdf("fake/model", "doc.pdf")

    assert result.risk_score == 6  # noqa: PLR2004
    assert result.smell_review_suggested is False


def test_predict_pdf_smell_review_suggested_true_when_no_ood_metrics() -> None:
    fake_extraction = ExtractionMetadata(
        text="hola mundo", extractor_used="MarkItDownExtractor", char_count=10
    )
    fake_result = PredictResult(
        label="decreto",
        confidence=0.9,
        certain=True,
        smells=["low_mahalanobis_p", "high_cosine_z"],  # weights 3+3=6, over the default 5
        ood_metrics=None,
    )
    with (
        patch("src.inference.pipeline.extract_pdf_with_metadata", return_value=fake_extraction),
        patch("src.inference.pipeline.BertTunningClassifier") as mock_clf_cls,
    ):
        mock_clf = MagicMock()
        mock_clf.predict_text.return_value = fake_result
        mock_clf_cls.return_value = mock_clf
        result = predict_pdf("fake/model", "doc.pdf")

    assert result.smell_review_suggested is True


def test_predict_pdf_extraction_failed_smell_review_suggested_stays_false() -> None:
    fake_extraction = ExtractionMetadata(text=None, extractor_used=None, char_count=0)
    with (
        patch("src.inference.pipeline.extract_pdf_with_metadata", return_value=fake_extraction),
        patch("src.inference.pipeline.BertTunningClassifier"),
    ):
        result = predict_pdf("fake/model", "doc.pdf")

    assert result.risk_score == 3  # noqa: PLR2004
    assert result.smell_review_suggested is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/inference/test_pipeline.py -k smell_review_suggested -v`
Expected: all 5 FAIL — the first 4 with `assert False is True` (field defaults to `False`,
`attach_metadata()` never sets it yet), the last one currently already passes trivially
(field defaults to `False`) — confirm it still passes for the right reason after Step 4,
not before.

- [ ] **Step 3: Implement `_compute_smell_review_suggested()`**

In `src/inference/pipeline.py`, add this function right after `_compute_risk_score()`
(currently lines 29-30):

```python
def _compute_smell_review_suggested(risk_score: int, ood_metrics: OodMetrics | None) -> bool:
    """True when risk_score alone crosses Settings.SMELL_REVIEW_RISK_SCORE_THRESHOLD, for
    documents the official decision didn't already send to a human. Deliberately
    independent of review_route/classifier_disagreement -- see
    docs/superpowers/specs/2026-07-27-smell-review-suggested-design.md. ood_metrics being
    None (no ood_stats.npz loaded) behaves the same as in_distribution=True; only an
    explicit False excludes it, since that path already gets review_route="human_review"."""
    if ood_metrics is not None and ood_metrics.in_distribution is False:
        return False
    return risk_score > Settings.SMELL_REVIEW_RISK_SCORE_THRESHOLD
```

Update the import at the top of the same file — change:

```python
from src.schemas import ExtractionMetadata, PredictResult
```

to:

```python
from src.schemas import ExtractionMetadata, OodMetrics, PredictResult
```

- [ ] **Step 4: Wire it into `attach_metadata()`**

In `src/inference/pipeline.py`, replace the current `attach_metadata()` body (lines 51-68):

```python
def attach_metadata(
    result: PredictResult, filename: str, extraction: ExtractionMetadata
) -> PredictResult:
    foreign_match = detect_foreign_municipality(extraction.text or "")
    smells = list(result.smells)
    if foreign_match is not None:
        smells.append("foreign_municipality")
    return result.model_copy(
        update={
            "filename": filename,
            "extracted_text": extraction.text,
            "extractor_used": extraction.extractor_used or "",
            "foreign_municipality": foreign_match.name if foreign_match else None,
            "foreign_municipality_context": foreign_match.context if foreign_match else None,
            "smells": smells,
            "risk_score": _compute_risk_score(smells),
        }
    )
```

with:

```python
def attach_metadata(
    result: PredictResult, filename: str, extraction: ExtractionMetadata
) -> PredictResult:
    foreign_match = detect_foreign_municipality(extraction.text or "")
    smells = list(result.smells)
    if foreign_match is not None:
        smells.append("foreign_municipality")
    risk_score = _compute_risk_score(smells)
    return result.model_copy(
        update={
            "filename": filename,
            "extracted_text": extraction.text,
            "extractor_used": extraction.extractor_used or "",
            "foreign_municipality": foreign_match.name if foreign_match else None,
            "foreign_municipality_context": foreign_match.context if foreign_match else None,
            "smells": smells,
            "risk_score": risk_score,
            "smell_review_suggested": _compute_smell_review_suggested(
                risk_score, result.ood_metrics
            ),
        }
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/inference/test_pipeline.py -k smell_review_suggested -v`
Expected: all 5 PASS

- [ ] **Step 6: Full check and commit**

Run: `uv run poe check`
Expected: all lint/mypy/tests pass (258 existing + 5 new).

```bash
git add src/inference/pipeline.py tests/inference/test_pipeline.py
git commit -m "feat: compute smell_review_suggested in attach_metadata()"
```

---

### Task 3: API surfacing — `PredictResponse` field + `_to_predict_response()` (spec gap)

**Files:**
- Modify: `src/api/routes/predict/schemas.py:26` (add field to `PredictResponse`)
- Modify: `src/api/routes/predict/endpoints.py:63-81` (`_to_predict_response()`)
- Test: `tests/api/test_predict.py`

**Interfaces:**
- Consumes: `PredictResult.smell_review_suggested` (Task 1/2).
- Produces: `PredictResponse.smell_review_suggested: bool`, surfaced in the `/predict`
  JSON response as `smellReviewSuggested` (camelCase, via `BaseSchema`'s
  `alias_generator=to_camel`).

**Why this task exists even though it's not in the spec's own Touch List:**
`_to_predict_response()` (`src/api/routes/predict/endpoints.py:63`) builds `PredictResponse`
from an explicit, hand-listed set of `data["..."]` kwargs — it does **not** forward every
`PredictResult` field automatically. Adding the field to the `PredictResponse` schema alone
is necessary but not sufficient: without also adding it to this function's explicit kwarg
list, the API response would silently keep `smellReviewSuggested: false` regardless of the
real value. This is the exact class of bug `CLAUDE.md` documents happening twice already
(`_PREDICTION_COLUMNS`/`RESULT_COLUMNS` drift) — catch it here before shipping a third
instance.

- [ ] **Step 1: Write the failing test**

Add to `tests/api/test_predict.py`, near `test_predict_endpoint_returns_review_route`:

```python
def test_predict_endpoint_returns_smell_review_suggested() -> None:
    app = create_app(model_path="fake/path")
    mock_clf = MagicMock()
    mock_clf.predict_text.return_value = PredictResult(
        label="decreto",
        confidence=0.9,
        certain=True,
        smells=["low_mahalanobis_p", "high_cosine_z"],
        risk_score=6,
        smell_review_suggested=True,
    )
    app.state.clf = mock_clf

    fake_extraction = ExtractionMetadata(
        text="hola mundo", extractor_used="OCRExtractor", char_count=10
    )
    with patch(
        "src.api.routes.predict.endpoints.extract_pdf_with_metadata", return_value=fake_extraction
    ):
        client = TestClient(app)
        result = _predict_and_await_result(
            client, files={"file": ("doc.pdf", b"%PDF-1.4 fake content", "application/pdf")}
        )

    assert result["smellReviewSuggested"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/api/test_predict.py::test_predict_endpoint_returns_smell_review_suggested -v`
Expected: FAIL with `KeyError: 'smellReviewSuggested'` (the response body has no such key
yet).

- [ ] **Step 3: Add the field to `PredictResponse`**

In `src/api/routes/predict/schemas.py`, add this line right after `risk_score: int = 0`
(currently the last field of `PredictResponse`, line 26):

```python
    smell_review_suggested: bool = False
```

- [ ] **Step 4: Wire it into `_to_predict_response()`**

In `src/api/routes/predict/endpoints.py`, add one line to the explicit kwarg list — right
after `risk_score=data["risk_score"],` (currently line 80):

```python
        smell_review_suggested=data["smell_review_suggested"],
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/api/test_predict.py::test_predict_endpoint_returns_smell_review_suggested -v`
Expected: PASS

- [ ] **Step 6: Full check and commit**

Run: `uv run poe check`
Expected: all lint/mypy/tests pass.

```bash
git add src/api/routes/predict/schemas.py src/api/routes/predict/endpoints.py tests/api/test_predict.py
git commit -m "feat: surface smell_review_suggested in the /predict API response"
```

---

### Task 4: W&B `_PREDICTION_COLUMNS`

**Files:**
- Modify: `src/wandb.py:16-41` (`_PREDICTION_COLUMNS` list)
- Test: `tests/test_wandb.py`

**Interfaces:**
- Consumes: `PredictResult.smell_review_suggested` (Task 1/2), the existing
  `flatten_predict_result()` (`src/schemas.py`, unchanged — it already forwards every
  top-level `PredictResult` field via `model_dump()`, so no change needed there).
- Produces: nothing new for later tasks — this is a leaf.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_wandb.py`, near `test_log_predict_folder_results_table_includes_svm_disagreement_columns`:

```python
def test_log_predict_folder_results_table_includes_smell_review_suggested_column() -> None:
    results = [
        PredictResult(
            filename="a.pdf",
            label="decreto",
            confidence=0.9,
            certain=True,
            smells=["low_mahalanobis_p", "high_cosine_z"],
            risk_score=6,
            smell_review_suggested=True,
        ),
    ]
    mock_table = MagicMock()
    with (
        patch("src.wandb.wandb.init"),
        patch("src.wandb.wandb.Table", return_value=mock_table) as mock_table_cls,
        patch("src.wandb.wandb.log"),
        patch("src.wandb.wandb.finish"),
    ):
        log_predict_folder_results(results, model_path="fake/model", folder_path="fake/folder")

    assert _logged_row(mock_table_cls, mock_table)["smell_review_suggested"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_wandb.py::test_log_predict_folder_results_table_includes_smell_review_suggested_column -v`
Expected: FAIL with `KeyError: 'smell_review_suggested'` (`_logged_row` looks up a column
`_PREDICTION_COLUMNS` doesn't have yet).

- [ ] **Step 3: Add the column**

In `src/wandb.py`, add `"smell_review_suggested"` to `_PREDICTION_COLUMNS` right after
`"risk_score",` (currently the last entry, line 40):

```python
    "risk_score",
    "smell_review_suggested",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_wandb.py::test_log_predict_folder_results_table_includes_smell_review_suggested_column -v`
Expected: PASS

- [ ] **Step 5: Full check and commit**

Run: `uv run poe check`
Expected: all lint/mypy/tests pass.

```bash
git add src/wandb.py tests/test_wandb.py
git commit -m "feat: add smell_review_suggested to the W&B predictions table"
```

---

### Task 5: `predict` CLI single-document output

**Files:**
- Modify: `src/cli/predict.py:69-70` (`predict_cmd`)
- Test: `tests/cli/test_commands.py`

**Interfaces:**
- Consumes: `PredictResult.smell_review_suggested` (Task 1/2).
- Produces: nothing new for later tasks — this is a leaf. (`predict-folder`'s CSV needs no
  change at all — `flatten_predict_result()` already forwards every top-level field
  automatically, confirmed by Task 4's note above.)

- [ ] **Step 1: Write the failing test**

Add to `tests/cli/test_commands.py`, near `test_predict_cmd_prints_ood_metrics_when_present`:

```python
def test_predict_cmd_prints_smell_review_suggested(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake content")
    fake_result = PredictResult(
        label="decreto",
        confidence=0.9,
        certain=True,
        smells=["low_mahalanobis_p", "high_cosine_z"],
        risk_score=6,
        smell_review_suggested=True,
    )

    with patch("src.cli.predict.predict_pdf", return_value=fake_result):
        result = CliRunner().invoke(predict_cmd, [str(pdf_path)])

    assert result.exit_code == 0
    assert "Smell review suggested: True" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_commands.py::test_predict_cmd_prints_smell_review_suggested -v`
Expected: FAIL — `"Smell review suggested: True" not in result.output`.

- [ ] **Step 3: Add the echo line**

In `src/cli/predict.py`, add one line right after the existing `Smells` line (currently
line 70):

```python
    click.echo(f"  Smells       : {', '.join(result.smells) if result.smells else '-'}")
    click.echo(f"  Smell review suggested: {result.smell_review_suggested}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cli/test_commands.py::test_predict_cmd_prints_smell_review_suggested -v`
Expected: PASS

- [ ] **Step 5: Full check and commit**

Run: `uv run poe check`
Expected: all lint/mypy/tests pass.

```bash
git add src/cli/predict.py tests/cli/test_commands.py
git commit -m "feat: print smell_review_suggested in predict CLI output"
```

---

### Task 6: Frontend — types, flatten, CSV/table column

**Files:**
- Modify: `frontend/src/types/api.ts:16-34` (`PredictResponse` interface)
- Modify: `frontend/src/utils/flatten.ts` (`FlatResultRow`, `NULL_ROW`, `flattenResult()`)
- Modify: `frontend/src/utils/csv.ts:7-31` (`RESULT_COLUMNS`)

**Interfaces:**
- Consumes: the API's `smellReviewSuggested: boolean` field (Task 3).
- Produces: nothing new for later tasks — this is the last leaf. `RESULT_COLUMNS`
  automatically drives both the CSV export and `PredictionsTable.tsx`'s on-screen columns
  (`FIXED_COLUMNS` derives from `RESULT_COLUMNS` — no separate table-component change
  needed, confirmed by the existing `svmPredictedLabel`/`svmAgreesWithPrediction` pattern).

No frontend test runner exists in this repo (`frontend/package.json` has no `test` script
— only `build` which runs `tsc -b`, and `lint` which runs `eslint`). Verification for this
task is typecheck + lint, not a unit test.

- [ ] **Step 1: Add the field to the `PredictResponse` interface**

In `frontend/src/types/api.ts`, add this line right after `riskScore: number;` (currently
the last field, line 33):

```typescript
  smellReviewSuggested: boolean;
```

- [ ] **Step 2: Add the field to `FlatResultRow`**

In `frontend/src/utils/flatten.ts`, add this line right after `riskScore: number | null;`
in the `FlatResultRow` interface (currently line 28):

```typescript
  smellReviewSuggested: boolean | null;
```

- [ ] **Step 3: Add the default to `NULL_ROW`**

In the same file, add this line right after `riskScore: null,` in `NULL_ROW` (currently
line 53):

```typescript
  smellReviewSuggested: null,
```

- [ ] **Step 4: Populate it in `flattenResult()`**

In the same file, add this line right after `riskScore: outcome.riskScore,` at the end of
`flattenResult()` (currently line 91):

```typescript
    smellReviewSuggested: outcome.smellReviewSuggested,
```

- [ ] **Step 5: Add the column to `RESULT_COLUMNS`**

In `frontend/src/utils/csv.ts`, add this line right after
`{ key: "riskScore", header: "Risk score" },` (currently the last entry, line 30):

```typescript
  { key: "smellReviewSuggested", header: "Smell review suggested" },
```

- [ ] **Step 6: Typecheck and lint**

Run: `cd frontend && npm run build`
Expected: succeeds with no TypeScript errors (confirms every interface/usage site agrees
on the field name and type).

Run: `cd frontend && npm run lint`
Expected: succeeds with no new lint errors.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/types/api.ts frontend/src/utils/flatten.ts frontend/src/utils/csv.ts
git commit -m "feat: surface smell_review_suggested in the frontend table and CSV export"
```

---

## Final verification

- [ ] Run `uv run poe check` one more time from repo root — full lint + mypy strict +
  pytest, all Python tasks combined.
- [ ] Run `cd frontend && npm run build && npm run lint` one more time — confirms the
  frontend task didn't drift from the backend's final field shape.
- [ ] Manually smoke-test: `uv run python main.py predict <path/to/a/pdf>` on a real
  document and confirm the new `Smell review suggested: ...` line prints.
