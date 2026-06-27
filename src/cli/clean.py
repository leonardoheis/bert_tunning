import logging
import shutil
from pathlib import Path

import click

from config import CACHE_PATH, OUTPUT_DIR
from logger import setup_logging

log = logging.getLogger(__name__)

_LOG_FILE = Path("logs/classiflow.log")
_CACHE = Path(CACHE_PATH)
_MODEL_DIR = Path(OUTPUT_DIR)


def _release_log_file() -> None:
    root = logging.getLogger()
    for handler in root.handlers[:]:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == _LOG_FILE:
            handler.close()
            root.removeHandler(handler)


@click.command("clean")
def clean_cmd() -> None:
    """Wipe logs, dataset cache, and model checkpoints."""
    setup_logging()
    targets = [
        (_LOG_FILE, "log file"),
        (_CACHE, "dataset cache"),
        (_MODEL_DIR, "model checkpoints"),
    ]
    for path, label in targets:
        if not path.exists():
            log.info("Clean: %s not found, skipping", label)
            continue
        if path == _LOG_FILE:
            _release_log_file()
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        log.info("Clean: deleted %s (%s)", label, path)
    setup_logging()
    log.info("Clean complete")
