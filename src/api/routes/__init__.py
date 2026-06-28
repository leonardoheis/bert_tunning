from collections.abc import Iterable

from fastapi import APIRouter

from .health import health_router
from .predict import PredictResponse, prediction_router

ROUTERS: Iterable[APIRouter] = (
    health_router,
    prediction_router,
)

__all__ = ["ROUTERS", "PredictResponse", "health_router", "prediction_router"]
