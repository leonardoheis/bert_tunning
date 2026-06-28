from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes.predict import router as predict_router
from src.inference.classify import BertTunningClassifier


def create_app(model_path: str, threshold: float = 0.70) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.clf = BertTunningClassifier(model_path, confidence_threshold=threshold)
        yield

    app = FastAPI(
        title="Bert Tunning API",
        description="Classifies Spanish municipal PDF documents",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(predict_router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
