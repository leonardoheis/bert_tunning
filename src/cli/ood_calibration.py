import logging
from pathlib import Path

import click
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from src.cli._ood_common import embed_texts, reconstruct_split_and_load_model
from src.logger import setup_logging
from src.ood import (
    compute_train_mahalanobis_distances,
    cosine_z_score,
    knn_mean_distance,
    load_stats,
    mahalanobis_empirical_p_value,
)
from src.schema import CalibrationReport
from src.settings import Settings
from src.training.models import get_model_config
from src.wandb import log_ood_calibration_results

log = logging.getLogger(__name__)


def build_calibration_report(
    p_values: npt.NDArray[np.float64],
    z_scores: npt.NDArray[np.float64],
    knn_distances: npt.NDArray[np.float64],
    target_fp_rate: float,
) -> CalibrationReport:
    """Pure calibration math, isolated from model/IO for direct unit testing.

    Mahalanobis: LOW p-value = anomalous, so the threshold for a target false-positive
    rate is the `target_fp_rate`-th percentile of in-distribution p-values. Cosine and
    k-NN mean distance: HIGH value = anomalous, so their thresholds are the
    `(1 - target_fp_rate)`-th percentile.
    """
    return CalibrationReport(
        fp_rate_maha=float(np.mean(p_values < Settings.OOD_MAHALANOBIS_P_THRESHOLD)),
        fp_rate_cosine=float(np.mean(z_scores > Settings.OOD_COSINE_THRESHOLD)),
        fp_rate_knn=float(np.mean(knn_distances > Settings.OOD_KNN_DISTANCE_THRESHOLD)),
        suggested_maha_threshold=float(np.percentile(p_values, target_fp_rate * 100)),
        suggested_cosine_threshold=float(np.percentile(z_scores, (1 - target_fp_rate) * 100)),
        suggested_knn_threshold=float(np.percentile(knn_distances, (1 - target_fp_rate) * 100)),
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
    embeddings = embed_texts(
        split.loaded,
        split.test_df,
        chunk_strategy=opts.chunk_strategy,
        max_tokens=model_cfg.max_tokens,
    )

    p_values = np.array(
        [mahalanobis_empirical_p_value(e, stats, train_distances) for e in embeddings]
    )
    z_scores = np.array([cosine_z_score(e, stats) for e in embeddings])
    label_ids = split.test_df["label_id"].to_numpy()
    knn_distances = np.array(
        [
            knn_mean_distance(e, stats, int(lbl), k=Settings.OOD_KNN_NEIGHBORS)
            for e, lbl in zip(embeddings, label_ids, strict=True)
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
    report = build_calibration_report(
        p_values, z_scores, knn_distances[knn_valid], opts.target_fp_rate
    )

    log.info("=" * 60)
    log.info("OOD threshold calibration — %s", opts.model_path)
    log.info("=" * 60)
    log.info(
        "Mahalanobis — current threshold=%.4f, empirical false-positive rate=%.2f%%",
        Settings.OOD_MAHALANOBIS_P_THRESHOLD,
        report.fp_rate_maha * 100,
    )
    log.info(
        "Mahalanobis — suggested threshold for %.1f%% target FP rate: %.6f",
        opts.target_fp_rate * 100,
        report.suggested_maha_threshold,
    )
    log.info(
        "Cosine — current threshold=%.4f, empirical false-positive rate=%.2f%%",
        Settings.OOD_COSINE_THRESHOLD,
        report.fp_rate_cosine * 100,
    )
    log.info(
        "Cosine — suggested threshold for %.1f%% target FP rate: %.4f",
        opts.target_fp_rate * 100,
        report.suggested_cosine_threshold,
    )
    log.info(
        "k-NN — current threshold=%.4f, empirical false-positive rate=%.2f%%",
        Settings.OOD_KNN_DISTANCE_THRESHOLD,
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
