import logging

import uvicorn

from src.logger import setup_logging
from src.settings import Settings

from .app import create_app

log = logging.getLogger(__name__)


def run_api() -> None:
    log_file = setup_logging()
    log.info("Logging to %s", log_file)
    app = create_app(model_path=Settings.default_model_path, threshold=Settings.model_threshold)
    uvicorn.run(app, host=Settings.HOST, port=Settings.API_PORT)


__all__ = ["create_app", "run_api"]
