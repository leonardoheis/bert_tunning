import logging
from pathlib import Path
from typing import Literal

import click
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from src.cli._ood_common import embed_texts_and_predict, reconstruct_split_and_load_model
from src.logger import setup_logging
from src.ood import (
    OodThresholds,
    compute_train_mahalanobis_distances,
    cosine_z_score,
    knn_mean_distance,
    load_stats,
    mahalanobis_empirical_p_value,
    resolve_ood_thresholds,
    save_stats,
)
from src.schema import CalibrationReport, ClassEmbeddingStats
from src.settings import Settings
from src.training.models import get_model_config
from src.wandb import log_ood_calibration_results

log = logging.getLogger(__name__)


def build_calibration_report(
    p_values: npt.NDArray[np.float64],
    z_scores: npt.NDArray[np.float64],
    knn_distances: npt.NDArray[np.float64],
    target_fp_rate: float,
    thresholds: OodThresholds,
) -> CalibrationReport:
    """Pure calibration math, isolated from model/IO for direct unit testing.

    Mahalanobis: LOW p-value = anomalous, so the threshold for a target false-positive
    rate is the `target_fp_rate`-th percentile of in-distribution p-values. Cosine and
    k-NN mean distance: HIGH value = anomalous, so their thresholds are the
    `(1 - target_fp_rate)`-th percentile.

    `thresholds` must come from resolve_ood_thresholds(stats) at the call site, not
    Settings.OOD_* directly -- otherwise re-running this command on a model that already
    has --write-thresholds-persisted per-model thresholds reports the empirical
    false-positive rate of thresholds production isn't even using, silently defeating the
    per-model calibration this module exists to support.
    """
    return CalibrationReport(
        fp_rate_maha=float(np.mean(p_values < thresholds.mahalanobis_p)),
        fp_rate_cosine=float(np.mean(z_scores > thresholds.cosine_z)),
        fp_rate_knn=float(np.mean(knn_distances > thresholds.knn_distance)),
        suggested_maha_threshold=float(np.percentile(p_values, target_fp_rate * 100)),
        suggested_cosine_threshold=float(np.percentile(z_scores, (1 - target_fp_rate) * 100)),
        suggested_knn_threshold=float(np.percentile(knn_distances, (1 - target_fp_rate) * 100)),
    )


def _write_calibrated_thresholds(
    stats: ClassEmbeddingStats,
    stats_path: Path,
    report: CalibrationReport,
    n_train: int,
) -> None:
    """Writes evaluate-ood-calibration's suggested thresholds back into this model's own
    ood_stats.npz, so resolve_ood_thresholds() uses per-model calibrated values instead of
    falling back to Settings.OOD_* -- the fix for thresholds calibrated against one model
    being silently applied to every other model. Refuses to write a Mahalanobis threshold at
    or below the empirical p-value's own resolution floor (1/(n_train+1)) -- that threshold
    would be mathematically unreachable (the signal could never fire), the exact bug this
    project hit once already with an unchecked suggested value."""
    floor = 1 / (n_train + 1)
    suggested_maha_threshold = report.suggested_maha_threshold
    maha_threshold: float | None = suggested_maha_threshold
    maha_status: Literal["calibrated", "refused_degenerate"] = "calibrated"
    if suggested_maha_threshold <= floor:
        log.warning(
            "Refusing to write suggested Mahalanobis threshold %.6f: at or below this "
            "model's empirical resolution floor %.6f (n_train=%d). The signal would never "
            "fire. Keeping the existing value (%s).",
            suggested_maha_threshold,
            floor,
            n_train,
            stats.mahalanobis_p_threshold,
        )
        maha_threshold = stats.mahalanobis_p_threshold
        # A kept prior value is still "calibrated" -- only truly unset (never calibrated
        # before, and now also refused) becomes "refused_degenerate".
        maha_status = "calibrated" if maha_threshold is not None else "refused_degenerate"

    updated = stats.model_copy(
        update={
            "mahalanobis_p_threshold": maha_threshold,
            "mahalanobis_threshold_status": maha_status,
            "cosine_threshold": report.suggested_cosine_threshold,
            "knn_distance_threshold": report.suggested_knn_threshold,
        }
    )
    save_stats(updated, stats_path)
    log.info(
        "Wrote calibrated thresholds to %s: mahalanobis_p=%s, cosine=%.4f, knn_distance=%.4f",
        stats_path,
        maha_threshold,
        report.suggested_cosine_threshold,
        report.suggested_knn_threshold,
    )


class OodCalibrationOptions(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        arbitrary_types_allowed=True,
        frozen=True,
        populate_by_name=True,
    )

    model_path: str
    model_key: str
    cache_path: str
    chunk_strategy: str = Settings.CHUNK_STRATEGY
    seed: int = Settings.SEED
    target_fp_rate: float = Settings.TARGET_FP_RATE
    write_thresholds: bool = False
    log_wandb: bool = False
    debug: bool = False


def _run_ood_calibration(opts: OodCalibrationOptions) -> None:
    log_file = setup_logging(level=logging.DEBUG if opts.debug else logging.INFO)
    log.info("Logging to %s", log_file)

    # Checked before touching the model/split (both expensive) so a missing stats file
    # fails fast, same as before this function's setup was consolidated.
    stats_path = Path(opts.model_path) / "ood_stats.npz"
    if not stats_path.exists():
        msg = f"No ood_stats.npz found at {stats_path} — run compute-ood-stats first"
        raise click.ClickException(msg)
    stats = load_stats(stats_path)

    model_cfg = get_model_config(opts.model_key)
    # Same split-reconstruction as Task 3b — the test split is known in-distribution
    # by construction, since it's real training documents held out from the train split.
    split = reconstruct_split_and_load_model(
        model_path=opts.model_path, cache_path=opts.cache_path, seed=opts.seed
    )

    train_distances = compute_train_mahalanobis_distances(stats)
    if len(train_distances) == 0:
        msg = "No training data — cannot calibrate Mahalanobis"
        raise click.ClickException(msg)
    log.info("Reconstructed test split: %d docs (known in-distribution)", len(split.test_df))
    log.info("Extracting embeddings on %s", split.loaded.device)
    embeddings, predicted_ids = embed_texts_and_predict(
        split.loaded,
        split.test_df,
        chunk_strategy=opts.chunk_strategy,
        max_tokens=model_cfg.max_tokens,
    )

    p_values = np.array(
        [mahalanobis_empirical_p_value(e, stats, train_distances) for e in embeddings]
    )
    z_scores = np.array([cosine_z_score(e, stats) for e in embeddings])
    # Predicted label, not the document's true label -- predict_text() in production always
    # scores knn_mean_distance against the model's own prediction, so calibration must
    # reproduce exactly that, including the k-NN penalty a misclassified in-distribution
    # document actually gets in production when it's scored against the wrong class's
    # neighbors. Using the true label here would understate the real false-positive rate.
    knn_distances = np.array(
        [
            knn_mean_distance(e, stats, pred_id, k=Settings.OOD_KNN_NEIGHBORS)
            for e, pred_id in zip(embeddings, predicted_ids, strict=True)
        ]
    )
    # knn_mean_distance returns NaN when a document's class has zero training points in
    # the reconstructed split — exclude those from the k-NN calibration rather than let
    # NaN silently propagate into np.mean/np.percentile and corrupt the whole report.
    knn_valid = ~np.isnan(knn_distances)
    if not knn_valid.all():
        log.warning(
            "Skipping %d/%d test docs with no same-class training points for k-NN calibration",
            int((~knn_valid).sum()),
            len(knn_distances),
        )
    if not knn_valid.any():
        msg = "No test documents have same-class training points — cannot calibrate k-NN"
        raise click.ClickException(msg)
    # The model's actually-deployed thresholds -- per-model calibrated values from `stats`
    # if evaluate-ood-calibration --write-thresholds already ran for this model, falling
    # back to Settings.OOD_* otherwise. Reading Settings.OOD_* unconditionally here would
    # report the empirical false-positive rate of thresholds production isn't even using
    # once a model has its own per-model thresholds persisted.
    current_thresholds = resolve_ood_thresholds(stats)
    report = build_calibration_report(
        p_values, z_scores, knn_distances[knn_valid], opts.target_fp_rate, current_thresholds
    )
    if opts.write_thresholds:
        _write_calibrated_thresholds(stats, stats_path, report, len(train_distances))

    log.info("=" * 60)
    log.info("OOD threshold calibration — %s", opts.model_path)
    log.info("=" * 60)
    log.info(
        "Mahalanobis — current threshold=%.4f, empirical false-positive rate=%.2f%%",
        current_thresholds.mahalanobis_p,
        report.fp_rate_maha * 100,
    )
    log.info(
        "Mahalanobis — suggested threshold for %.1f%% target FP rate: %.6f",
        opts.target_fp_rate * 100,
        report.suggested_maha_threshold,
    )
    log.info(
        "Cosine — current threshold=%.4f, empirical false-positive rate=%.2f%%",
        current_thresholds.cosine_z,
        report.fp_rate_cosine * 100,
    )
    log.info(
        "Cosine — suggested threshold for %.1f%% target FP rate: %.4f",
        opts.target_fp_rate * 100,
        report.suggested_cosine_threshold,
    )
    log.info(
        "k-NN — current threshold=%.4f, empirical false-positive rate=%.2f%%",
        current_thresholds.knn_distance,
        report.fp_rate_knn * 100,
    )
    log.info(
        "k-NN — suggested threshold for %.1f%% target FP rate: %.4f",
        opts.target_fp_rate * 100,
        report.suggested_knn_threshold,
    )

    if opts.log_wandb:
        log_ood_calibration_results(
            report,
            model_path=opts.model_path,
            cache_path=opts.cache_path,
            target_fp_rate=opts.target_fp_rate,
            thresholds=current_thresholds,
        )


@click.command("evaluate-ood-calibration")
@click.option(
    "--model-path",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Path to an already-trained model directory",
)
@click.option(
    "--model",
    "model_key",
    required=True,
    help="Model registry key used for that model (e.g. beto, xlm-roberta, minilm)",
)
@click.option(
    "--cache-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the exact parquet cache used to train that model",
)
@click.option("--chunk-strategy", default=Settings.CHUNK_STRATEGY, show_default=True)
@click.option(
    "--seed",
    default=Settings.SEED,
    show_default=True,
    help="Must match the seed used for the original training run",
)
@click.option(
    "--target-fp-rate",
    type=click.FloatRange(0.0, 1.0, min_open=True, max_open=True),
    default=Settings.TARGET_FP_RATE,
    show_default=True,
    help="Target false-positive rate used to compute the suggested threshold",
)
@click.option(
    "--write-thresholds",
    is_flag=True,
    default=False,
    help=(
        "Persist the suggested thresholds into this model's ood_stats.npz, so "
        "predict/predict-folder/serve use them instead of falling back to Settings.OOD_*"
    ),
)
@click.option(
    "--log-wandb",
    is_flag=True,
    default=False,
    help="Log calibration summary metrics to W&B",
)
@click.option("--debug", is_flag=True, default=False)
def evaluate_ood_calibration_cmd(**kwargs: str | float | bool) -> None:
    """Measure the empirical false-positive rate of OOD thresholds against the test split,
    and suggest better-calibrated thresholds if the current ones don't match your target."""
    _run_ood_calibration(OodCalibrationOptions.model_validate(kwargs))
