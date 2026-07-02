import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from transformers import Trainer

from src.schema import EvaluationResult, Hyperparams, ReportDict
from src.training.reporting import generate_html_report
from src.training.tokenize import BertTunningDataset

log = logging.getLogger(__name__)


def run_evaluation(
    trainer: Trainer,
    test_ds: BertTunningDataset,
    le: LabelEncoder,
    hyperparams: Hyperparams,
) -> EvaluationResult:
    preds_out = trainer.predict(test_ds)
    y_pred: npt.NDArray[np.int_] = np.argmax(preds_out.predictions, axis=-1)
    label_ids: npt.NDArray[np.int_] = np.asarray(preds_out.label_ids)
    y_true: list[int] = label_ids.tolist()

    report_str = classification_report(
        y_true, y_pred, target_names=le.classes_, digits=3, zero_division=0
    )
    report_dict: ReportDict = classification_report(
        y_true, y_pred, target_names=le.classes_, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

    log.info("Per-class report:\n%s", report_str)
    log.info("Confusion matrix:\n%s", cm_df.to_string())

    result = EvaluationResult(report_dict=report_dict, y_pred=y_pred, y_true=y_true)

    generate_html_report(
        label_names=list(le.classes_),
        cm=cm,
        result=result,
        hyperparams=hyperparams,
    )

    return result
