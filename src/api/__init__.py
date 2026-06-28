import uvicorn

from src.settings import Settings

from .app import create_app


def run_api() -> None:
    uvicorn.run(
        "app.api:create_app",
        factory=True,
        host=Settings.HOST,
        port=Settings.API_PORT,
    )


__all__ = ["create_app", "run_api"]
