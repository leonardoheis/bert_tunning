"""Shared Pydantic model definitions used across the Bert Tunning pipeline."""

from pydantic import BaseModel, ConfigDict


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


class Hyperparams(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    epochs: int
    batch_size: int
    grad_accum: int
    effective_batch: int
    learning_rate: float
    warmup_steps: int
    precision: str
    train_docs: int
    num_classes: int
