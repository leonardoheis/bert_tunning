from pydantic import Field

from src.api.schema import BaseSchema
from src.schema import OodMetrics, ReviewRoute


class PredictResponse(BaseSchema):
    filename: str
    label: str | None
    confidence: float
    certain: bool
    all_scores: dict[str, float] = Field(default_factory=dict)
    error: str | None = None
    ood_metrics: OodMetrics | None = None
    extracted_text: str = ""
    extractor_used: str = ""
    review_route: ReviewRoute = ""
    foreign_municipality: str | None = None
    foreign_municipality_context: str | None = None
    svm_scores: dict[str, float] = Field(default_factory=dict)
    svm_predicted_label: str = ""
    svm_agrees_with_prediction: bool = True
