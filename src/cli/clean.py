import logging
import shutil
from pathlib import Path

import click

from logger import setup_logging
from src.settings import Settings

log = logging.getLogger(__name__)

_LOG_DIR = Path(__file__).parent.parent.parent / "logs"
_CACHE = Path(Settings.CACHE_PATH)
_MODEL_DIR = Path(Settings.OUTPUT_DIR)


def _release_log_handlers() -> None:
    root = logging.getLogger()
    for handler in root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            root.removeHandler(handler)


@click.command("clean")
def clean_cmd() -> None:
    """Wipe logs, dataset cache, and model checkpoints."""
    setup_logging()
    log_files = sorted(_LOG_DIR.glob("bert_tunning_*.log"))
    if log_files:
        _release_log_handlers()
        for f in log_files:
            f.unlink()
        log.info("Clean: deleted %d log file(s) from %s", len(log_files), _LOG_DIR)

    for path, label in [(_CACHE, "dataset cache"), (_MODEL_DIR, "model checkpoints")]:
        if not path.exists():
            log.info("Clean: %s not found, skipping", label)
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        log.info("Clean: deleted %s (%s)", label, path)
    log.info("Clean complete")
