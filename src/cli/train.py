import logging

import click

from config import (
    CACHE_PATH,
    CHUNK_STRATEGY,
    DOCS_ROOT,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    MODEL_KEY,
    OUTPUT_DIR,
    SEED,
)
from logger import setup_logging
from src.ingestion.pipeline import run as ingest
from src.training.models import get_model_config
from src.training.pipeline import run as train_run

log = logging.getLogger(__name__)


@click.command("train")
@click.option(
    "--docs-root",
    default=DOCS_ROOT,
    show_default=True,
    help="Root folder with labeled PDF subfolders",
)
@click.option(
    "--model",
    "model_key",
    default=MODEL_KEY,
    show_default=True,
    help="Model registry key (e.g. xlm-roberta, beto)",
)
@click.option(
    "--max-docs-per-class", type=int, default=None, help="Cap docs per class for quick test runs"
)
@click.option(
    "--rebuild-cache", is_flag=True, default=False, help="Force re-extraction even if cache exists"
)
@click.option("--no-ocr", is_flag=True, default=False, help="Skip OCR fallback")
@click.option("--no-wandb", is_flag=True, default=False, help="Disable W&B logging")
@click.option("--debug", is_flag=True, default=False, help="Enable DEBUG logging")
def train_cmd(  # noqa: PLR0913
    docs_root: str,
    model_key: str,
    max_docs_per_class: int | None,
    rebuild_cache: bool,  # noqa: FBT001
    no_ocr: bool,  # noqa: FBT001
    no_wandb: bool,  # noqa: FBT001
    debug: bool,  # noqa: FBT001
) -> None:
    """Fine-tune a transformer model on municipal PDF documents."""
    setup_logging(level=logging.DEBUG if debug else logging.INFO)

    model_cfg = get_model_config(model_key)
    log.info("Using model: %s (%s)", model_cfg.name, model_cfg.hf_id)

    df = ingest(
        docs_root,
        cache_path=CACHE_PATH,
        use_ocr=not no_ocr,
        rebuild=rebuild_cache,
        max_docs_per_class=max_docs_per_class,
    )

    if len(df) == 0:
        log.error("No documents found. Check --docs-root: %s", docs_root)
        return

    train_run(
        df,
        model_cfg,
        epochs=EPOCHS,
        early_stop_patience=EARLY_STOP_PATIENCE,
        chunk_strategy=CHUNK_STRATEGY,
        seed=SEED,
        output_dir=OUTPUT_DIR,
        use_wandb=not no_wandb,
    )
