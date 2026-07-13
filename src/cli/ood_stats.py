import logging
from pathlib import Path

import click
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from src.cli._ood_common import embed_texts, reconstruct_split_and_load_model
from src.logger import setup_logging
from src.ood import compute_class_stats, save_stats
from src.settings import Settings
from src.training.models import get_model_config

log = logging.getLogger(__name__)


class ComputeOodStatsOptions(BaseModel):
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
    debug: bool = False


def _run_compute_ood_stats(opts: ComputeOodStatsOptions) -> None:
    log_file = setup_logging(level=logging.DEBUG if opts.debug else logging.INFO)
    log.info("Logging to %s", log_file)

    model_cfg = get_model_config(opts.model_key)
    split = reconstruct_split_and_load_model(
        model_path=opts.model_path, cache_path=opts.cache_path, seed=opts.seed
    )
    log.info("%d classes: %s", len(split.classes), split.classes)
    log.info("Reconstructed train split: %d docs", len(split.train_df))
    log.info("Extracting embeddings on %s", split.loaded.device)

    embeddings = embed_texts(
        split.loaded,
        split.train_df,
        chunk_strategy=opts.chunk_strategy,
        max_tokens=model_cfg.max_tokens,
    )
    stats = compute_class_stats(
        embeddings,
        split.train_df["label_id"].tolist(),
        split.classes,
        n_components=Settings.OOD_PCA_COMPONENTS,
        model_type=split.loaded.model.config.model_type,  # type: ignore[union-attr,arg-type]
        model_hidden_size=split.loaded.model.config.hidden_size,  # type: ignore[union-attr,arg-type]
    )

    out_path = Path(opts.model_path) / "ood_stats.npz"
    save_stats(stats, out_path)
    log.info("Saved OOD stats -> %s", out_path)


@click.command("compute-ood-stats")
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
    help=(
        "Must match the seed used for the original training run, or the reconstructed "
        "train split will differ"
    ),
)
@click.option("--debug", is_flag=True, default=False)
def compute_ood_stats_cmd(**kwargs: str | int | bool) -> None:
    """Backfill ood_stats.npz for an already-trained model, without retraining it."""
    _run_compute_ood_stats(ComputeOodStatsOptions.model_validate(kwargs))
