from pydantic import Field

from src.api.schema import BaseSchema


class PredictResponse(BaseSchema):
    filename: str
    label: str | None
    confidence: float
    certain: bool
    all_scores: dict[str, float] = Field(default_factory=dict)
    error: str | None = None
    mahalanobis_p_value: float | None = None
    mahalanobis_p_value_theoretical: float | None = None
    cosine_z: float | None = None
    knn_distance: float | None = None
    in_distribution: bool | None = None
    extracted_text: str = ""
    extractor_used: str = ""
    review_route: str = ""
    foreign_municipality: str | None = None
    foreign_municipality_context: str | None = None
