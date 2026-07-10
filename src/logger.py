import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_LOG_DIR = Path(__file__).parent.parent / "logs"

_FMT = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> Path:
    _LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = _LOG_DIR / f"bert_tunning_{timestamp}.log"

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    fmt = logging.Formatter(_FMT, datefmt=_DATEFMT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)

    root.addHandler(console)
    root.addHandler(file_handler)

    for noisy in (
        "transformers",
        "accelerate",
        "datasets",
        "filelock",
        "urllib3",
        "httpx",
        "httpcore",
        "huggingface_hub",
        "huggingface_hub.utils._http",
        "pdfminer",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return log_file
