import logging
from pathlib import Path

import click
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.inference.ood import compute_class_stats, extract_embeddings, save_stats
from src.logger import setup_logging
from src.settings import Settings
from src.training.models import get_model_config
from src.training.split import make_split
from src.training.tokenize import prepare_text

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
    df = pd.read_parquet(opts.cache_path)

    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])
    log.info("%d classes: %s", len(le.classes_), list(le.classes_))

    train_df, _val_df, _test_df = make_split(df, seed=opts.seed)
    log.info("Reconstructed train split: %d docs", len(train_df))

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

    texts = [prepare_text(t, tokenizer, opts.chunk_strategy) for t in train_df["text"]]
    embeddings = extract_embeddings(
        model, tokenizer, texts, max_length=model_cfg.max_tokens, device=device
    )
    stats = compute_class_stats(
        embeddings,
        train_df["label_id"].tolist(),
        list(le.classes_),
        n_components=Settings.OOD_PCA_COMPONENTS,
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
