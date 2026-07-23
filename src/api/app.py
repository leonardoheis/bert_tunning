import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.inference.classify import BertTunningClassifier
from src.ingestion.extract import warm_ocr_reader

from .routes import ROUTERS

# repo root / frontend / dist -- built by `npm run build` (or the Dockerfile's frontend
# build stage). Not present in a plain `uv run` dev/test environment, hence the is_dir()
# guard below: this mount is additive, API-only behavior (tests, CLI, Swagger UI) is
# unaffected when the frontend hasn't been built.
_FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"


def create_app(model_path: str, threshold: float = 0.70) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        app.state.clf = BertTunningClassifier(model_path, confidence_threshold=threshold)
        await asyncio.to_thread(warm_ocr_reader)
        yield

    app = FastAPI(
        title="Bert Tunning API",
        description="Classifies Spanish municipal PDF documents",
        version="0.1.0",
        lifespan=lifespan,
    )

    for router in ROUTERS:
        app.include_router(router)

    if _FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")

    return app
