import logging
from enum import Enum
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

# sklearn is already an unconditional runtime dependency elsewhere in this project
# (svm_reviewer.py) -- gating this import behind TYPE_CHECKING buys nothing real here.
from sklearn.svm import SVC  # noqa: TC002
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerBase

from src.exceptions import BertTunningError
from src.inference.ood_scorer import OodScorer
from src.ingestion.extract import clean_text
from src.schema import PredictResult
from src.settings import Settings
from src.svm_reviewer import load_svm_classifiers, svm_top_label
from src.svm_reviewer import svm_scores as compute_svm_scores

log = logging.getLogger(__name__)


class TransformerModelConfig(Protocol):
    """The subset of a loaded transformers model's .config this classifier reads."""

    id2label: dict[int, str]
    model_type: str
    hidden_size: int
    max_position_embeddings: int


class TransformerModelOutput(Protocol):
    """The subset of a forward-pass output this classifier reads."""

    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...]


class TransformerModel(Protocol):
    """The subset of a loaded transformers model this classifier actually depends on --
    named explicitly instead of typed Any. A bare torch.nn.Module doesn't have .config
    (that's transformers-specific), and transformers' own stubs for forward-pass outputs
    are too loose (Tensor | Module, via nn.Module.__getattr__) to type-check cleanly
    against directly -- a minimal Protocol naming exactly what's used sidesteps both."""

    config: TransformerModelConfig

    def eval(self) -> "TransformerModel": ...
    def to(self, device: str) -> "TransformerModel": ...
    def __call__(self, **kwargs: torch.Tensor | bool) -> TransformerModelOutput: ...


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


def decide_review_route(
    *,
    confidence_tier: ConfidenceTier,
    ood_evidence: OodEvidence,
    classifier_disagreement: bool = False,
) -> str:
    """Route a prediction to "accept", "llm_judge", or "human_review".

    An OOD signal firing (ANOMALOUS) always wins and routes to a human -- an LLM judge
    can't be trusted to catch what already fooled the classifier itself. A classifier
    disagreement (the SVM reviewer's top pick doesn't match softmax's) is a second,
    independent trigger for the same lane -- a different failure mode than OOD (the
    document IS a known class, the two classifiers just disagree which one), but the
    same "route to a human regardless of confidence" rationale applies: a confident-but-
    wrong prediction is the dangerous case either way. Otherwise the softmax confidence
    tier alone decides: confident predictions are accepted, uncertain ones get a cheap
    LLM-judge second opinion. See "Review routing" in README.md for the full decision
    table and rationale.
    """
    if ood_evidence is OodEvidence.ANOMALOUS or classifier_disagreement:
        return "human_review"
    return "accept" if confidence_tier is ConfidenceTier.CONFIDENT else "llm_judge"


class BertTunningClassifier:
    def __init__(
        self,
        model_path: str,
        *,
        confidence_threshold: float = Settings.model_threshold,
        tokenizer: PreTrainedTokenizerBase | None = None,
        model: TransformerModel | None = None,
    ) -> None:
        log.info("Loading classifier from %s", model_path)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_path)
        self.model: TransformerModel = model or AutoModelForSequenceClassification.from_pretrained(
            model_path
        )
        self.threshold = confidence_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        self.model.to(self.device)
        self.max_length = min(
            self.tokenizer.model_max_length, self.model.config.max_position_embeddings
        )
        self._ood_scorer = OodScorer.load(model_path)
        if self._ood_scorer is not None:
            self._ood_scorer.validate(
                self.model.config.id2label,
                self.model.config.model_type,
                self.model.config.hidden_size,
            )
            self._ood_scorer.warn_if_uncalibrated()
        self._svm_classifiers = self._load_svm_classifiers(model_path)
        self._validate_svm_classifiers_class_mapping()
        log.info("Classifier ready on %s (max_length=%d)", self.device, self.max_length)

    @staticmethod
    def _load_svm_classifiers(model_path: str) -> "dict[str, SVC] | None":
        classifiers_path = Path(model_path) / "svm_classifiers.joblib"
        classifiers = load_svm_classifiers(classifiers_path)
        if classifiers is None:
            log.info(
                "No svm_classifiers.joblib found at %s — SVM reviewer disabled",
                classifiers_path,
            )
            return None
        log.info("Loaded SVM reviewer classifiers from %s", classifiers_path)
        return classifiers

    def _validate_svm_classifiers_class_mapping(self) -> None:
        """svm_classifiers.joblib's class set must match this model's id2label -- a stale
        or mismatched artifact (e.g. fit for a corpus with a different class set) would
        silently produce svm_scores keyed by the wrong classes, or missing some entirely.
        Unlike ood_stats.npz's class_names (an ordered list indexed positionally by label
        id, since knn_mean_distance indexes into it directly), svm_classifiers is a dict
        keyed by class NAME -- only the set needs to match, not the order -- but the same
        fail-fast-at-construction rationale as _validate_ood_stats_class_mapping applies."""
        if self._svm_classifiers is None:
            return
        id2label: dict[int, str] = self.model.config.id2label
        expected = set(id2label.values())
        actual = set(self._svm_classifiers.keys())
        if actual != expected:
            msg = (
                f"svm_classifiers.joblib classes {sorted(actual)} do not match this "
                f"model's id2label classes {sorted(expected)} -- svm_scores would be "
                "computed for the wrong classes. Regenerate svm_classifiers.joblib for "
                "this exact model with compute-svm-classifiers."
            )
            raise BertTunningError(msg)

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
        if self._svm_classifiers is None:
            svm_scores_result: dict[str, float] = {}
            svm_predicted_label, svm_agrees_with_prediction = "", True
        else:
            svm_scores_result = compute_svm_scores(cls_embedding, self._svm_classifiers)
            svm_predicted_label = svm_top_label(svm_scores_result)
            svm_agrees_with_prediction = svm_predicted_label == label
        classifier_disagreement = not svm_agrees_with_prediction
        result = PredictResult(
            label=label,
            confidence=round(confidence, 4),
            certain=certain,
            all_scores={id2label[i]: round(float(p), 4) for i, p in enumerate(probs)},
            review_route=decide_review_route(
                confidence_tier=confidence_tier,
                ood_evidence=OodEvidence.NOT_ANOMALOUS,
                classifier_disagreement=classifier_disagreement,
            ),
            svm_scores=svm_scores_result,
            svm_predicted_label=svm_predicted_label,
            svm_agrees_with_prediction=svm_agrees_with_prediction,
        )

        if self._ood_scorer is None:
            return result

        ood_metrics = self._ood_scorer.score(text, cls_embedding, pred_idx)
        if ood_metrics is None:
            return result

        return result.model_copy(
            update={
                "ood_metrics": ood_metrics,
                "review_route": decide_review_route(
                    confidence_tier=confidence_tier,
                    ood_evidence=OodEvidence.from_in_distribution(
                        in_distribution=ood_metrics.in_distribution
                    ),
                    classifier_disagreement=classifier_disagreement,
                ),
            }
        )
