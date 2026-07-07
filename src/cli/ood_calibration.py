import logging
from pathlib import Path

import click
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.inference.ood import (
    cosine_z_score,
    extract_embeddings,
    load_stats,
    mahalanobis_p_value,
)
from src.logger import setup_logging
from src.schema import CalibrationReport
from src.settings import Settings
from src.training.models import get_model_config
from src.training.split import make_split
from src.training.tokenize import prepare_text
from src.wandb import log_ood_calibration_results

log = logging.getLogger(__name__)


def build_calibration_report(
    p_values: npt.NDArray[np.float64],
    z_scores: npt.NDArray[np.float64],
    target_fp_rate: float,
) -> CalibrationReport:
    """Pure calibration math, isolated from model/IO for direct unit testing.

    Mahalanobis: LOW p-value = anomalous, so the threshold for a target false-positive
    rate is the `target_fp_rate`-th percentile of in-distribution p-values. Cosine: HIGH
    z-score = anomalous, so its threshold is the `(1 - target_fp_rate)`-th percentile.
    """
    return CalibrationReport(
        fp_rate_maha=float(np.mean(p_values < Settings.OOD_MAHALANOBIS_P_THRESHOLD)),
        fp_rate_cosine=float(np.mean(z_scores > Settings.OOD_COSINE_THRESHOLD)),
        suggested_maha_threshold=float(np.percentile(p_values, target_fp_rate * 100)),
        suggested_cosine_threshold=float(np.percentile(z_scores, (1 - target_fp_rate) * 100)),
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

    model_cfg = get_model_config(opts.model_key)
    df = pd.read_parquet(opts.cache_path)

    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])

    # Same split-reconstruction as Task 3b — the test split is known in-distribution
    # by construction, since it's real training documents held out from the train split.
    _train_df, _val_df, test_df = make_split(df, seed=opts.seed)
    log.info("Reconstructed test split: %d docs (known in-distribution)", len(test_df))

    stats_path = Path(opts.model_path) / "ood_stats.npz"
    if not stats_path.exists():
        msg = f"No ood_stats.npz found at {stats_path} — run compute-ood-stats first"
        raise click.ClickException(msg)
    stats = load_stats(stats_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(opts.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(opts.model_path)
    model.eval()
    model.to(device)
    log.info("Extracting embeddings on %s", device)

    model_labels = set(model.config.id2label.values())
    cache_labels = set(le.classes_)
    if model_labels != cache_labels:
        msg = (
            f"Cache classes {sorted(cache_labels)} do not match model classes "
            f"{sorted(model_labels)} — wrong --cache-path or --model-path?"
        )
        raise click.ClickException(msg)

    texts = [prepare_text(t, tokenizer, opts.chunk_strategy) for t in test_df["text"]]
    embeddings = extract_embeddings(
        model, tokenizer, texts, max_length=model_cfg.max_tokens, device=device
    )

    p_values = np.array([mahalanobis_p_value(e, stats) for e in embeddings])
    z_scores = np.array([cosine_z_score(e, stats) for e in embeddings])
    report = build_calibration_report(p_values, z_scores, opts.target_fp_rate)

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
