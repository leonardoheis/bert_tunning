import logging
import sys
from pathlib import Path

_LOG_DIR  = Path(__file__).parent / "logs"
_LOG_FILE = _LOG_DIR / "classiflow.log"

_FMT    = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    _LOG_DIR.mkdir(exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any handlers added by third-party imports (transformers, torch, etc.)
    # before we had a chance to configure logging — basicConfig would silently no-op otherwise.
    root.handlers.clear()

    fmt = logging.Formatter(_FMT, datefmt=_DATEFMT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)

    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(fmt)

    root.addHandler(console)
    root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in (
        "transformers", "accelerate", "datasets", "filelock", "urllib3",
        "httpx", "httpcore", "huggingface_hub", "huggingface_hub.utils._http",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
