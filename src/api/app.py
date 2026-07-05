from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.inference.classify import BertTunningClassifier

from .routes import ROUTERS


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

    for router in ROUTERS:
        app.include_router(router)

    return app
