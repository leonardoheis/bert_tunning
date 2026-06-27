import logging

import click
from pydantic import BaseModel

from logger import setup_logging
from src.ingestion.pipeline import run as ingest
from src.settings import Settings
from src.training.models import get_model_config
from src.training.options import TrainingRequest
from src.training.pipeline import run as train_run

log = logging.getLogger(__name__)


class TrainOptions(BaseModel):
    docs_root: str = Settings.DOCS_ROOT
    model_key: str = Settings.MODEL_KEY
    max_docs_per_class: int | None = None
    rebuild_cache: bool = False
    no_ocr: bool = False
    no_wandb: bool = False
    debug: bool = False


def _run_train(opts: TrainOptions) -> None:
    setup_logging(level=logging.DEBUG if opts.debug else logging.INFO)

    model_cfg = get_model_config(opts.model_key)
    log.info("Using model: %s (%s)", model_cfg.name, model_cfg.hf_id)

    df = ingest(
        opts.docs_root,
        cache_path=Settings.CACHE_PATH,
        use_ocr=not opts.no_ocr,
        rebuild=opts.rebuild_cache,
        max_docs_per_class=opts.max_docs_per_class,
    )

    if len(df) == 0:
        log.error("No documents found. Check --docs-root: %s", opts.docs_root)
        return

    train_run(df, model_cfg, TrainingRequest(use_wandb=not opts.no_wandb))


@click.command("train")
@click.option(
    "--docs-root",
    default=Settings.DOCS_ROOT,
    show_default=True,
    help="Root folder with labeled PDF subfolders",
)
@click.option(
    "--model",
    "model_key",
    default=Settings.MODEL_KEY,
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
def train_cmd(**kwargs: str | int | bool | None) -> None:
    """Fine-tune a transformer model on municipal PDF documents."""
    _run_train(TrainOptions.model_validate(kwargs))
