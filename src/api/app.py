from fastapi import FastAPI

from src.api.routes.predict import configure
from src.api.routes.predict import router as predict_router


def create_app(model_path: str, threshold: float = 0.70) -> FastAPI:
    configure(model_path, threshold)

    app = FastAPI(
        title="Classiflow API",
        description="Classifies Spanish municipal PDF documents",
        version="0.1.0",
    )

    app.include_router(predict_router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
