import logging
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerBase

from src.ingestion.extract import clean_text
from src.ood import cosine_z_score, knn_mean_distance, load_stats, mahalanobis_p_value
from src.schema import ClassEmbeddingStats, PredictResult
from src.settings import Settings

log = logging.getLogger(__name__)


class ConfidenceTier(Enum):
    """Named alternative to a bare `certain: bool` at the decide_review_route boundary."""

    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"

    @classmethod
    def from_confidence(cls, confidence: float, threshold: float) -> "ConfidenceTier":
        return cls.CONFIDENT if confidence >= threshold else cls.UNCERTAIN


class OodEvidence(Enum):
    """Named tri-state for `in_distribution`, so "no ood_stats.npz loaded" reads as an
    explicit state rather than a bare `None` a caller has to know to interpret."""

    NOT_ANOMALOUS = "not_anomalous"  # in_distribution=True, or no ood_stats.npz loaded
    ANOMALOUS = "anomalous"  # in_distribution=False

    @classmethod
    def from_in_distribution(cls, *, in_distribution: bool | None) -> "OodEvidence":
        return cls.ANOMALOUS if in_distribution is False else cls.NOT_ANOMALOUS


class OodScores(NamedTuple):
    """The three OOD signals -- always computed together, passed together, never used
    independently. Matches LoadedModel/_PcaReduction's convention in src/ood.py."""

    mahalanobis_p: float
    cosine_z: float
    knn_distance: float


def is_out_of_distribution(scores: OodScores) -> bool:
    """Any one of the three OOD signals firing is enough -- a deliberate OR, not a
    weighted blend (see README's "OOD scoring internals" for why). NaN in knn_distance
    means the predicted class had zero training points to compare against; treated as
    anomalous, fail-safe, since `nan > threshold` would otherwise silently pass.
    """
    maha_anomalous = scores.mahalanobis_p < Settings.OOD_MAHALANOBIS_P_THRESHOLD
    cosine_anomalous = scores.cosine_z > Settings.OOD_COSINE_THRESHOLD
    knn_anomalous = (
        bool(np.isnan(scores.knn_distance))
        or scores.knn_distance > Settings.OOD_KNN_DISTANCE_THRESHOLD
    )
    log.debug(
        "OOD signals: mahalanobis_p=%.6f (threshold=%.6f, anomalous=%s), "
        "cosine_z=%.4f (threshold=%.4f, anomalous=%s), "
        "knn_distance=%.4f (threshold=%.4f, anomalous=%s)",
        scores.mahalanobis_p,
        Settings.OOD_MAHALANOBIS_P_THRESHOLD,
        maha_anomalous,
        scores.cosine_z,
        Settings.OOD_COSINE_THRESHOLD,
        cosine_anomalous,
        scores.knn_distance,
        Settings.OOD_KNN_DISTANCE_THRESHOLD,
        knn_anomalous,
    )
    return maha_anomalous or cosine_anomalous or knn_anomalous


def decide_review_route(*, confidence_tier: ConfidenceTier, ood_evidence: OodEvidence) -> str:
    """Route a prediction to "accept", "llm_judge", or "human_review".

    An OOD signal firing (ANOMALOUS) always wins and routes to a human -- an LLM judge
    can't be trusted to catch what already fooled the classifier itself. Otherwise the
    softmax confidence tier alone decides: confident predictions are accepted, uncertain
    ones get a cheap LLM-judge second opinion. See "Review routing" in README.md for the
    full decision table and rationale.
    """
    if ood_evidence is OodEvidence.ANOMALOUS:
        return "human_review"
    return "accept" if confidence_tier is ConfidenceTier.CONFIDENT else "llm_judge"


class BertTunningClassifier:
    def __init__(
        self,
        model_path: str,
        *,
        confidence_threshold: float = 0.70,
        tokenizer: PreTrainedTokenizerBase | None = None,
        model: torch.nn.Module | None = None,
    ) -> None:
        log.info("Loading classifier from %s", model_path)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_path)
        self.model: Any = model or AutoModelForSequenceClassification.from_pretrained(model_path)
        self.threshold = confidence_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        self.model.to(self.device)
        self.max_length = min(  # type: ignore[type-var]
            self.tokenizer.model_max_length,
            self.model.config.max_position_embeddings,  # type: ignore[union-attr]
        )
        self._ood_stats = self._load_ood_stats(model_path)
        log.info("Classifier ready on %s (max_length=%d)", self.device, self.max_length)

    @staticmethod
    def _load_ood_stats(model_path: str) -> ClassEmbeddingStats | None:
        stats_path = Path(model_path) / "ood_stats.npz"
        if not stats_path.exists():
            log.info("No ood_stats.npz found at %s — OOD scoring disabled", stats_path)
            return None
        log.info("Loaded OOD stats from %s", stats_path)
        return load_stats(stats_path)

    def predict_text(self, text: str) -> PredictResult:
        inputs = self.tokenizer(
            clean_text(text),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
            cls_embedding = outputs.hidden_states[-1][:, 0, :][0].cpu().numpy().astype(np.float64)

        id2label = self.model.config.id2label
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        label = id2label[pred_idx]

        confidence_tier = ConfidenceTier.from_confidence(confidence, self.threshold)
        certain = confidence_tier is ConfidenceTier.CONFIDENT
        result = PredictResult(
            label=label,
            confidence=round(confidence, 4),
            certain=certain,
            all_scores={id2label[i]: round(float(p), 4) for i, p in enumerate(probs)},
            review_route=decide_review_route(
                confidence_tier=confidence_tier, ood_evidence=OodEvidence.NOT_ANOMALOUS
            ),
        )

        if self._ood_stats is None:
            return result

        scores = OodScores(
            mahalanobis_p=mahalanobis_p_value(cls_embedding, self._ood_stats),
            cosine_z=cosine_z_score(cls_embedding, self._ood_stats),
            knn_distance=knn_mean_distance(
                cls_embedding, self._ood_stats, pred_idx, k=Settings.OOD_KNN_NEIGHBORS
            ),
        )
        in_distribution = not is_out_of_distribution(scores)
        return result.model_copy(
            update={
                "mahalanobis_p_value": round(scores.mahalanobis_p, 6),
                "cosine_z": round(scores.cosine_z, 4),
                "knn_distance": round(scores.knn_distance, 4),
                "in_distribution": in_distribution,
                "review_route": decide_review_route(
                    confidence_tier=confidence_tier,
                    ood_evidence=OodEvidence.from_in_distribution(in_distribution=in_distribution),
                ),
            }
        )
