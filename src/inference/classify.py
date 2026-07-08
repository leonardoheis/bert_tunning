import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.ingestion.extract import clean_text
from src.ood import cosine_z_score, knn_mean_distance, load_stats, mahalanobis_p_value
from src.schema import ClassEmbeddingStats, PredictResult
from src.settings import Settings

log = logging.getLogger(__name__)


class BertTunningClassifier:
    def __init__(self, model_path: str, *, confidence_threshold: float = 0.70) -> None:
        log.info("Loading classifier from %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.threshold = confidence_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        self.model.to(self.device)
        self.max_length = min(
            self.tokenizer.model_max_length,
            self.model.config.max_position_embeddings,
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

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        label = self.model.config.id2label[pred_idx]

        result = PredictResult(
            label=label,
            confidence=round(confidence, 4),
            certain=confidence >= self.threshold,
            all_scores={
                self.model.config.id2label[i]: round(float(p), 4) for i, p in enumerate(probs)
            },
        )

        if self._ood_stats is None:
            return result

        maha_p = mahalanobis_p_value(cls_embedding, self._ood_stats)
        cosine_z = cosine_z_score(cls_embedding, self._ood_stats)
        knn_dist = knn_mean_distance(
            cls_embedding, self._ood_stats, pred_idx, k=Settings.OOD_KNN_NEIGHBORS
        )
        maha_anomalous = maha_p < Settings.OOD_MAHALANOBIS_P_THRESHOLD
        cosine_anomalous = cosine_z > Settings.OOD_COSINE_THRESHOLD
        # NaN means the predicted class had zero training points to compare against — treat
        # that as anomalous (fail safe) rather than let `nan > threshold` silently evaluate
        # to False and never flag it.
        knn_anomalous = bool(np.isnan(knn_dist)) or knn_dist > Settings.OOD_KNN_DISTANCE_THRESHOLD
        out_of_distribution = maha_anomalous or cosine_anomalous or knn_anomalous
        return result.model_copy(
            update={
                "mahalanobis_p_value": round(maha_p, 6),
                "cosine_z": round(cosine_z, 4),
                "knn_distance": round(knn_dist, 4),
                "in_distribution": not out_of_distribution,
            }
        )
