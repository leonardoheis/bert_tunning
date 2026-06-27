import logging

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from transformers import Trainer

from reporting import generate_html_report
from src.training.tokenize import ClassiflowDataset

log = logging.getLogger(__name__)


def run_evaluation(
    trainer: Trainer,
    test_ds: ClassiflowDataset,
    le: LabelEncoder,
    hyperparams: dict,
) -> tuple[dict, np.ndarray, list[int]]:
    preds_out = trainer.predict(test_ds)
    y_pred = np.argmax(preds_out.predictions, axis=-1)
    y_true: list[int] = preds_out.label_ids.tolist()

    report_str = classification_report(
        y_true, y_pred, target_names=le.classes_, digits=3, zero_division=0
    )
    report_dict = classification_report(
        y_true, y_pred, target_names=le.classes_, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

    log.info("Per-class report:\n%s", report_str)
    log.info("Confusion matrix:\n%s", cm_df.to_string())

    generate_html_report(
        label_names=list(le.classes_),
        y_true=y_true,
        y_pred=y_pred,
        cm=cm,
        report_dict=report_dict,
        hyperparams=hyperparams,
    )

    return report_dict, y_pred, y_true
