# Extract Calibration Operation Out of the Click Module — Design Spec

## Motivation

A code review found `_run_ood_calibration()` (`src/cli/ood_calibration.py:153`) mixing six
concerns in one function: CLI setup (logging), model/split reconstruction, four OOD signal
computations, threshold resolution, optional threshold persistence, ~40 lines of console
reporting, and optional W&B logging. `build_calibration_report()` right above it already
went through this exact separation once — its own docstring says "Pure calibration math,
isolated from model/IO for direct unit testing" — this is the same treatment for the
*orchestration* half that already happened to the *math* half.

**Recommendation being implemented:** extract a pure calibration operation returning a
structured result. Click stays responsible only for parsing options, calling the operation,
and rendering its result (console logs, optional W&B).

## Touch list

| File | What changes |
|---|---|
| `src/cli/ood_calibration.py` | New `compute_ood_calibration()` (the extracted operation) + new `OodCalibrationRunResult`; `_run_ood_calibration()` shrinks to logging setup + calling the operation + rendering |
| `tests/cli/test_ood_calibration.py` | Existing tests keep working unchanged (same public `evaluate_ood_calibration_cmd` behavior) — optionally add direct tests against `compute_ood_calibration()` itself, now callable without going through Click |

**Not touched:** `build_calibration_report()` (already pure, unchanged), `_write_calibrated_thresholds()` (already a separate function — the new operation calls it exactly where `_run_ood_calibration()` used to), `OodCalibrationOptions` (Click still constructs it the same way).

## Design

### `OodCalibrationRunResult` (new)

```python
class OodCalibrationRunResult(NamedTuple):
    report: CalibrationReport
    current_thresholds: OodThresholds
```

A plain `NamedTuple`, matching this project's established convention for internal-only
return bundles (`_PcaReduction`/`_TfidfStats` in `src/ood.py`, the bundles added in the
training-pipeline decomposition) — not Pydantic, since this never crosses a validation
boundary; it's produced and consumed entirely within `src/cli/ood_calibration.py`.

### `compute_ood_calibration()` (new — the extracted operation)

```python
def compute_ood_calibration(
    model_path: str,
    model_key: str,
    cache_path: str,
    chunk_strategy: str,
    seed: int,
    target_fp_rate: float,
    write_thresholds: bool,
) -> OodCalibrationRunResult:
```

Takes the operational parameters only — deliberately **excludes** `log_wandb` and `debug`
from `OodCalibrationOptions`, since those are rendering/CLI-setup concerns, not inputs the
calibration math itself needs. A signature that can't reach for CLI-only flags is a
stronger boundary than passing the whole `OodCalibrationOptions` object through and trusting
the function not to touch the fields it shouldn't.

Body is `_run_ood_calibration()`'s current lines 157–237 (stats loading, split
reconstruction, all four signal computations, `resolve_ood_thresholds()`,
`build_calibration_report()`, and the `if write_thresholds: _write_calibrated_thresholds(...)`
call) — moved verbatim, no logic changes. Returns `OodCalibrationRunResult(report, current_thresholds)`.

Persistence (`_write_calibrated_thresholds`) stays **inside** the operation, not pushed to
Click — writing calibrated thresholds back to `ood_stats.npz` is core domain behavior (`--write-thresholds`
is what `evaluate-ood-calibration` *does*, not how its result gets displayed), unlike the
console-log block and W&B call which are genuinely about presentation.

**Error handling stays exactly as today**: `click.ClickException` raised directly for the
three validation failures (missing stats file, no training data, no test docs with
same-class training points). This matches existing precedent already in this
codebase — `src/cli/_ood_common.py`'s `reconstruct_split_and_load_model()` is a non-Click
helper function that also raises `click.ClickException` directly, not `BertTunningError`
translated at a boundary. Inventing a new exception-translation layer here would be
inconsistent with that existing convention, not an improvement.

### `_run_ood_calibration()` (shrinks to setup + call + render)

```python
def _run_ood_calibration(opts: OodCalibrationOptions) -> None:
    log_file = setup_logging(level=logging.DEBUG if opts.debug else logging.INFO)
    log.info("Logging to %s", log_file)

    result = compute_ood_calibration(
        model_path=opts.model_path,
        model_key=opts.model_key,
        cache_path=opts.cache_path,
        chunk_strategy=opts.chunk_strategy,
        seed=opts.seed,
        target_fp_rate=opts.target_fp_rate,
        write_thresholds=opts.write_thresholds,
    )

    log.info("=" * 60)
    log.info("OOD threshold calibration — %s", opts.model_path)
    log.info("=" * 60)
    # ...the same ~30 lines of log.info(...) calls as today, reading result.report /
    # result.current_thresholds instead of local variables...

    if opts.log_wandb:
        log_ood_calibration_results(
            result.report,
            model_path=opts.model_path,
            cache_path=opts.cache_path,
            target_fp_rate=opts.target_fp_rate,
            thresholds=result.current_thresholds,
        )
```

The console-log block's content and order are byte-for-byte unchanged — only the source of
`report`/`current_thresholds` changes (from local variables to `result.report`/
`result.current_thresholds`).

## Backward compatibility

- `evaluate_ood_calibration_cmd`'s CLI behavior (flags, output, exit codes) is unchanged —
  this is a pure internal reshuffle behind the same public command.
- All existing tests in `tests/cli/test_ood_calibration.py` exercise
  `evaluate_ood_calibration_cmd`/`_run_ood_calibration` and should pass unmodified, since the
  observable behavior (console output, W&B calls, `ood_stats.npz` writes) doesn't change.
