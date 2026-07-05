from pydantic import Field

from src.api.schema import BaseSchema


class PredictResponse(BaseSchema):
    filename: str
    label: str | None
    confidence: float
    certain: bool
    all_scores: dict[str, float] = Field(default_factory=dict)
    error: str | None = None
