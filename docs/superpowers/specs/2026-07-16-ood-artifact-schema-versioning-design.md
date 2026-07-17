# OOD Artifact Schema: Grouped, Versioned Sections — Design Spec

## Motivation

A code review of `ClassEmbeddingStats` (`src/schema.py`) found a Shotgun Surgery smell: the class flatly combines PCA/Mahalanobis/cosine stats, k-NN training data, TF-IDF lexical stats, four independent calibrated thresholds, a threshold-status enum, and model-identity metadata — 20 fields, no internal grouping. Adding the TF-IDF signal (a real, already-shipped change) touched: the schema (6 new fields), `compute_class_stats()` (new kwargs), `save_stats()`/`load_stats()` (new `.npz` keys, each with its own backward-compatibility branch), `build_calibration_report()`, `_write_calibrated_thresholds()`, every OOD scoring function's field access, and tests across five files. Adding a fifth OOD-embedding-space signal, or a second lexical signal, would repeat the exact same shotgun pattern.

**Recommendation being implemented:** group the flat fields into nested immutable sections (embedding stats, lexical stats, thresholds, artifact metadata), add an explicit format version, and keep the `.npz` file itself backward-compatible at the serialization boundary.

## Scope confirmed with reviewer before writing this spec

The top-level container is being renamed alongside the restructuring: `ClassEmbeddingStats` → `OodArtifact`. The name is actively misleading today (it holds lexical/threshold/metadata sections, not just embedding stats) — fixing the structure without fixing the name would leave the same confusion in a tidier shape.

## Complete touch list (verified via grep, not guessed)

| File | What changes |
|---|---|
| `src/schema.py` | The model itself — this is the core of the change |
| `src/ood.py` | `compute_class_stats()`, `save_stats()`, `load_stats()`, `_project()`, `mahalanobis_min_distance()`, `cosine_min_distance()`, `knn_mean_distance()`, `tfidf_cosine_z_score()`, `build_tfidf_vectorizer()`, `compute_train_mahalanobis_distances()`, `resolve_ood_thresholds()` — every one reads `stats.<field>` and needs the new nested path |
| `src/inference/ood_scorer.py` | `OodScorer.validate()`/`_validate_class_mapping()`/`_validate_model_identity()`/`warn_if_uncalibrated()`/`score()` |
| `src/cli/ood_calibration.py` | `_write_calibrated_thresholds()` (currently `stats.model_copy(update={flat fields})`) |
| `src/cli/ood_stats.py`, `src/training/pipeline.py` | Call `compute_class_stats(...)` — external kwargs stay the same, so these need minimal/no changes, confirmed by grep (they only ever pass `model_type=`/`model_hidden_size=` as plain kwargs, never touch the returned object's field layout directly beyond `save_stats(stats, ...)`) |
| `tests/test_ood.py`, `tests/test_schema.py`, `tests/cli/test_ood_calibration.py`, `tests/inference/test_pipeline.py` | Every `ClassEmbeddingStats(...)` construction and every `stats.<field>` access |

**Not touched, confirmed by grep:** `src/wandb.py`, `src/cli/_ood_common.py`, `src/svm_reviewer.py` — none read `ClassEmbeddingStats` fields directly. `wandb.py` only ever sees the already-resolved `OodThresholds` NamedTuple (a separate type, produced by `resolve_ood_thresholds()`), not the raw artifact.

## Design

### New shape (`src/schema.py`)

```python
class EmbeddingStats(BaseModel):
    """PCA + Mahalanobis/cosine/k-NN stats — the BERT-embedding-space signals."""
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    pca_mean: Float64Array
    pca_components: Float64Array
    centroids: Float64Array
    covariance_inv: Float64Array
    cosine_calibration_mean: float
    cosine_calibration_std: float
    knn_train_embeddings: Float64Array
    knn_train_labels: list[int]


class LexicalStats(BaseModel):
    """TF-IDF cosine-centroid stats -- the lexical, non-embedding signal. Always
    present (never Optional), same reasoning as PredictResult.svm_scores: "not fitted"
    has a natural empty-collection representation (empty vocabulary_terms), so there's
    no second, redundant way to express the same fact. is_fitted() replaces today's
    scattered `if not stats.tfidf_vocabulary_terms` checks with one named predicate."""
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    vocabulary_terms: list[str] = []
    idf: Float64Array = Field(default_factory=lambda: np.zeros(0))
    centroids: Float64Array = Field(default_factory=lambda: np.zeros((0, 0)))
    cosine_calibration_mean: float = 0.0
    cosine_calibration_std: float = 1.0

    def is_fitted(self) -> bool:
        return bool(self.vocabulary_terms)


class CalibratedThresholds(BaseModel):
    """Per-model calibrated OOD thresholds -- each independently Optional, because each
    is calibrated independently (the degenerate-threshold guard can refuse to write
    Mahalanobis while cosine/k-NN still get written in the same run). Unlike
    ArtifactMetadata below, these do NOT move as a unit, so they stay individually
    Optional rather than the whole section being Optional."""
    model_config = ConfigDict(frozen=True)

    mahalanobis_p: float | None = None
    cosine: float | None = None
    knn_distance: float | None = None
    tfidf_cosine: float | None = None
    mahalanobis_status: Literal["not_calibrated", "calibrated", "refused_degenerate"] = (
        "not_calibrated"
    )


class ArtifactMetadata(BaseModel):
    """Model-identity fingerprint. Unlike CalibratedThresholds, model_type and
    model_hidden_size are always set or unset TOGETHER (compute_class_stats() passes
    both or neither) -- so the whole section is Optional (None = "predates identity
    fingerprinting") rather than two independently-nullable fields that are only ever
    used as a pair. This collapses _validate_model_identity()'s existing two-field
    None-check into one."""
    model_config = ConfigDict(frozen=True)

    model_type: str
    model_hidden_size: int


class OodArtifact(BaseModel):
    """Replaces ClassEmbeddingStats. One artifact, four independently-evolvable
    sections, one shared class taxonomy. Adding a fifth signal type means adding one
    new section class + a save/load pair for it -- not editing this class or the
    monolithic save_stats()/load_stats() functions."""
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    format_version: int
    class_names: list[str]
    embedding: EmbeddingStats
    lexical: LexicalStats = LexicalStats()
    thresholds: CalibratedThresholds = CalibratedThresholds()
    metadata: ArtifactMetadata | None = None
```

### Field mapping (old flat path → new nested path)

| Old (`ClassEmbeddingStats`) | New (`OodArtifact`) |
|---|---|
| `class_names` | `class_names` (unchanged, top-level — shared class taxonomy indexes both `embedding.centroids` and `lexical.centroids` identically) |
| `pca_mean`, `pca_components`, `centroids`, `covariance_inv`, `cosine_calibration_mean`, `cosine_calibration_std`, `knn_train_embeddings`, `knn_train_labels` | `embedding.<same name>` |
| `tfidf_vocabulary_terms` | `lexical.vocabulary_terms` |
| `tfidf_idf` | `lexical.idf` |
| `tfidf_centroids` | `lexical.centroids` |
| `tfidf_cosine_calibration_mean` | `lexical.cosine_calibration_mean` |
| `tfidf_cosine_calibration_std` | `lexical.cosine_calibration_std` |
| `mahalanobis_p_threshold` | `thresholds.mahalanobis_p` |
| `cosine_threshold` | `thresholds.cosine` |
| `knn_distance_threshold` | `thresholds.knn_distance` |
| `tfidf_threshold` | `thresholds.tfidf_cosine` |
| `mahalanobis_threshold_status` | `thresholds.mahalanobis_status` |
| `model_type`, `model_hidden_size` | `metadata.model_type`, `metadata.model_hidden_size` (both `None` together becomes `metadata is None`) |
| *(none — new)* | `format_version` |

Redundant `tfidf_`/`_threshold` prefixes are dropped inside their sections — `stats.lexical.vocabulary_terms` and `stats.thresholds.tfidf_cosine` already say what they are via the nesting; keeping the old prefix would just be stutter.

### Versioning strategy: `.npz` keys stay flat, only the Python-side shape changes

"Keep `.npz` compatibility at the serialization boundary" is implemented literally: **the actual `.npz` file keys do not change** (still `mahalanobis_p_threshold`, `tfidf_vocabulary_terms`, etc., flat, exactly as today) — only *one* new key is added, `format_version` (an int scalar). `save_stats()`/`load_stats()` become the translation layer between the flat `.npz` file and the nested `OodArtifact` object.

- **`save_stats()`** always writes `format_version=CURRENT_OOD_ARTIFACT_VERSION` (starting at `2` — `1` is retroactively assigned to mean "no `format_version` key present at all", i.e. every already-committed `ood_stats.npz` today). Flattens `stats.embedding.*`/`stats.lexical.*`/etc. back into the same flat key names the file has always used.
- **`load_stats()`** reads `format_version` (absent → `1`). For `format_version == 1`, applies the **exact same granular per-field `"in data.files"` backward-compatibility checks this code already has today** — nothing about existing already-committed BETO v1/v2 artifacts changes, they load exactly as before. For `format_version >= 2`, every section is guaranteed fully present (since `save_stats()` always writes the complete structure going forward), so loading is direct with no per-field presence checks.

**What this actually fixes:** today, `load_stats()` is one ~55-line function with 8+ inline `"x" in data.files` branches for every optional field, all at the same nesting level. After this change, each section gets its own small `_load_<section>(data, version) -> Section` helper (e.g. `_load_lexical_stats(data, version) -> LexicalStats`), and `load_stats()` itself becomes a thin orchestrator calling each one. The next time a genuinely new signal type is added, the future version bump (`format_version == 3`) only needs a check inside *that new section's own* loader — not a new branch threaded through one giant function alongside seven unrelated ones. This is the actual Shotgun Surgery fix: not that changes touch zero files (they still touch the section's own model + its own save/load helper + its own consumers), but that they stop touching everything *else*.

### Consumer updates (mechanical, per the field-mapping table above)

Every `stats.<old_field>` access across `ood.py`'s scoring functions, `OodScorer`, and `_write_calibrated_thresholds()` becomes `stats.<section>.<new_field>`. `_write_calibrated_thresholds()`'s `stats.model_copy(update={...})` becomes a nested update — Pydantic v2's `model_copy` doesn't deep-merge nested models automatically, so this becomes `stats.model_copy(update={"thresholds": stats.thresholds.model_copy(update={...})})`.

`OodScorer._validate_model_identity()` simplifies from two `is None` checks to one: `if stats.metadata is None: return` (skip entirely) replaces `if stats.model_type is None or stats.model_hidden_size is None: return`.

`build_tfidf_vectorizer()`'s `if not stats.tfidf_vocabulary_terms: return None` becomes `if not stats.lexical.is_fitted(): return None`.

## Out of scope

- `svm_classifiers.joblib` / `src/svm_reviewer.py` — a completely separate artifact, untouched by this refactor.
- `PredictResult`/`OodMetrics` (the *inference-time output* schema) — unaffected; this refactor is about the *training-time artifact* (`ood_stats.npz`) only.
- `OodThresholds` (the resolved NamedTuple `resolve_ood_thresholds()` returns, consumed by `wandb.py`/`is_out_of_distribution()`) — stays exactly as it is; it's already a separate, decoupled type from the persisted `CalibratedThresholds` section, and nothing about that boundary needs to change.
- No behavior change to any OOD decision, threshold value, or scoring math — this is a pure structural/serialization refactor.

## Testing

- `tests/test_schema.py`: construction of each new nested section class; `OodArtifact` construction; `metadata is None` when no identity fingerprint; `LexicalStats().is_fitted() is False` for the default (unfitted) instance.
- `tests/test_ood.py`: `save_stats`/`load_stats` round-trip for a `format_version == 2` (current) artifact; round-trip loading of an **actual legacy fixture with no `format_version` key** (constructed the same way today's tests already simulate "predates this field" — via a raw `.npz` written without the new key) to prove `format_version == 1` handling reproduces today's exact backward-compatible defaults; every existing `compute_class_stats`/scoring-function test updated to the new nested field paths, same assertions.
- `tests/inference/test_pipeline.py`, `tests/cli/test_ood_calibration.py`: updated field paths only — no assertions on `PredictResult`, `review_route`, or calibration output values should change.
- A real end-to-end check against the two already-committed artifacts (`models/bert_tunning_model_beto/final/ood_stats.npz`, `models/bert_tunning_model_beto_v2/final/ood_stats.npz`) — both predate `format_version` entirely, so successfully loading them via `OodScorer.load()` and producing identical `predict_text()` output to before this refactor is the real proof the migration path works, not just a synthetic fixture.
