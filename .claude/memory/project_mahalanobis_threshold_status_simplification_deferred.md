---
name: mahalanobis-threshold-status-simplification-deferred
description: bert_tunning's ClassEmbeddingStats.mahalanobis_threshold_status is a 3-state Literal where "calibrated" is redundant/inferable -- a bool simplification was considered and deliberately deferred, not rejected as wrong
metadata:
  type: project
---

`ClassEmbeddingStats.mahalanobis_threshold_status` (`src/schema.py`, PR #43 / branch `feature/ood-detection`) is a `Literal["not_calibrated", "calibrated", "refused_degenerate"]`. A ponytail-review pass (2026-07-13) found "calibrated" is always inferable from `mahalanobis_p_threshold is not None`, and no code path (`src/inference/classify.py`'s `_warn_on_uncalibrated_thresholds`, `src/cli/ood_calibration.py`'s `_write_calibrated_thresholds`) ever branches on the explicit "calibrated" value -- only "not_calibrated" and "refused_degenerate" are ever checked. A `refused_degenerate: bool = False` field would carry the same information with ~20 fewer lines (drops the `_threshold_status()` npz-string parser/validator in `src/ood.py` and its `Literal`/`BertTunningError` machinery).

**Why deferred:** by the time this was found, the code was already merged into `feature/ood-detection` and deployed -- BETO v2's committed `ood_stats.npz` already has the string field written to disk. The user explicitly chose not to reopen already-shipped, already-tested code for a line-count win, confirmed via a grilling-skill interview (2026-07-13) that walked the decision tree (do it now vs. skip vs. other) before settling on skip.

**How to apply:** don't proactively fix this or re-raise it unprompted. If `mahalanobis_threshold_status` (or an equivalent per-signal status field, e.g. for cosine/knn) needs to be touched again for an unrelated reason, mention the bool-simplification option again at that point, since the migration cost would then be shared with whatever other change is already touching that code.
