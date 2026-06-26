import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import MAX_TOKENS
from extraction import clean_text, extract_pdf

log = logging.getLogger(__name__)


class ClassiflowClassifier:
    """
    Production inference wrapper.

    Usage:
        clf = ClassiflowClassifier("./models/classiflow_deberta_model/final")
        print(clf.predict_pdf("path/to/documento.pdf"))
        print(clf.predict_text("Licitación Pública para construcción de puente..."))
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.70):
        log.info("Loading classifier from %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.threshold = confidence_threshold
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        self.model.to(self.device)
        log.info("Classifier ready on %s", self.device)

    def _infer(self, text: str) -> dict:
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_TOKENS,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self.model(**inputs).logits, dim=-1)[0].cpu().numpy()

        pred_idx   = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        label      = self.model.config.id2label[pred_idx]

        log.debug("Inference result: label=%s confidence=%.4f", label, confidence)

        return {
            "label":      label,
            "confidence": round(confidence, 4),
            "certain":    confidence >= self.threshold,
            "all_scores": {
                self.model.config.id2label[i]: round(float(p), 4)
                for i, p in enumerate(probs)
            },
        }

    def predict_text(self, text: str) -> dict:
        return self._infer(clean_text(text))

    def predict_pdf(self, pdf_path: str) -> dict:
        log.info("Classifying: %s", Path(pdf_path).name)
        text = extract_pdf(pdf_path, use_ocr_fallback=True)
        if text is None:
            log.warning("Could not extract text from %s", Path(pdf_path).name)
            return {
                "label": None,
                "confidence": 0.0,
                "certain": False,
                "error": "empty/unreadable document",
                "filename": Path(pdf_path).name,
            }
        result = self._infer(text)
        result["filename"] = Path(pdf_path).name
        log.info("%s → %s (%.2f%%)", Path(pdf_path).name, result["label"], result["confidence"] * 100)
        return result

    def predict_folder(self, folder_path: str) -> pd.DataFrame:
        pdfs = sorted(Path(folder_path).glob("*.pdf"))
        log.info("Classifying %d PDFs in %s", len(pdfs), folder_path)
        results = []
        for pdf in pdfs:
            r = self.predict_pdf(str(pdf))
            results.append(r)
        log.info("Folder classification complete")
        return pd.DataFrame(results)
