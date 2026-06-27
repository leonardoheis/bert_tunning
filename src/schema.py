"""Shared Pydantic model definitions used across the Bert Tunning pipeline."""

from pydantic import BaseModel


class PredictResult(BaseModel):
    """Return value from BertTunningClassifier.predict_text and predict_pdf."""

    label: str | None = None
    confidence: float = 0.0
    certain: bool = False
    all_scores: dict[str, float] = {}
    filename: str = ""
    error: str = ""


# classification_report(output_dict=True) returns per-class dicts and scalar floats.
ReportDict = dict[str, "dict[str, float] | float"]

# Hyperparameters forwarded to W&B / HTML report — scalar values only.
Hyperparams = dict[str, "str | int | float | bool"]
