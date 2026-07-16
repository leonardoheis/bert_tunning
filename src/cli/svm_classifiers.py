import logging
from pathlib import Path

import click
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from src.cli._ood_common import embed_texts, reconstruct_split_and_load_model
from src.logger import setup_logging
from src.settings import Settings
from src.svm_reviewer import evaluate_svm_classifiers, fit_svm_classifiers, save_svm_classifiers
from src.training.models import get_model_config
from src.wandb import log_svm_classifiers_results

log = logging.getLogger(__name__)


class ComputeSvmClassifiersOptions(BaseModel):
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
    log_wandb: bool = False
    debug: bool = False


def _run_compute_svm_classifiers(opts: ComputeSvmClassifiersOptions) -> None:
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
    classifiers = fit_svm_classifiers(
        embeddings, split.train_df["label_id"].tolist(), split.classes
    )
    log.info("Fit %d one-vs-rest SVM reviewers", len(classifiers))

    # "first", not opts.chunk_strategy -- mirrors training/pipeline.py's val split, which
    # always uses "first" regardless of the training chunk strategy.
    val_embeddings = embed_texts(
        split.loaded, split.val_df, chunk_strategy="first", max_tokens=model_cfg.max_tokens
    )
    svm_val_accuracy = evaluate_svm_classifiers(
        classifiers, val_embeddings, split.val_df["label_id"].tolist(), split.classes
    )
    log.info(
        "SVM reviewer held-out balanced accuracy (val split): %s",
        {k: round(v, 4) for k, v in svm_val_accuracy.items()},
    )

    out_path = Path(opts.model_path) / "svm_classifiers.joblib"
    save_svm_classifiers(classifiers, out_path)
    log.info("Saved SVM reviewer classifiers -> %s", out_path)

    if opts.log_wandb:
        train_labels = split.train_df["label_id"]
        train_class_counts = {
            name: int((train_labels == idx).sum()) for idx, name in enumerate(split.classes)
        }
        log_svm_classifiers_results(
            model_path=opts.model_path,
            cache_path=opts.cache_path,
            model_key=opts.model_key,
            svm_val_accuracy=svm_val_accuracy,
            train_class_counts=train_class_counts,
        )


@click.command("compute-svm-classifiers")
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
@click.option(
    "--log-wandb", is_flag=True, default=False, help="Log per-class held-out accuracy to W&B"
)
@click.option("--debug", is_flag=True, default=False)
def compute_svm_classifiers_cmd(**kwargs: str | int | bool) -> None:
    """Backfill svm_classifiers.joblib for an already-trained model, without retraining it."""
    _run_compute_svm_classifiers(ComputeSvmClassifiersOptions.model_validate(kwargs))
