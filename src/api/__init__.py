import uvicorn

from src.settings import Settings

from .app import create_app


def run_api() -> None:
    app = create_app(model_path=Settings.default_model_path, threshold=Settings.model_threshold)
    uvicorn.run(app, host=Settings.HOST, port=Settings.API_PORT)


__all__ = ["create_app", "run_api"]
