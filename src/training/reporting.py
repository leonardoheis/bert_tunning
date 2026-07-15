import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.schema import EvaluationResult, Hyperparams, ReportDict

log = logging.getLogger(__name__)

_REPORTS_DIR = Path(__file__).parent.parent.parent / "reports"


def _class_metric_pairs(report_dict: ReportDict) -> tuple[list[str], list[dict[str, float]]]:
    """Per-class rows of a classification_report dict, skipping the accuracy/macro/weighted
    summary rows and any other scalar entries -- only the per-class metric dicts remain."""
    class_pairs = [
        (c, v)
        for c in report_dict
        if c not in ("accuracy", "macro avg", "weighted avg")
        and isinstance(v := report_dict[c], dict)
    ]
    classes = [c for c, _v in class_pairs]
    class_dicts = [v for _c, v in class_pairs]
    return classes, class_dicts


def _confusion_matrix_trace(label_names: list[str], cm: npt.NDArray[np.int_]) -> go.Heatmap:
    cm_pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    heatmap_text = [
        [f"{cm[r][c]}<br>({cm_pct[r][c]:.0%})" for c in range(len(label_names))]
        for r in range(len(label_names))
    ]
    return go.Heatmap(
        z=cm_pct,
        x=label_names,
        y=label_names,
        text=heatmap_text,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=False,
    )


def _f1_bar_trace(classes: list[str], class_dicts: list[dict[str, float]]) -> go.Bar:
    f1_scores = [float(m["f1-score"]) for m in class_dicts]
    return go.Bar(x=classes, y=f1_scores, marker_color="steelblue", name="F1")


def _precision_recall_traces(
    classes: list[str], class_dicts: list[dict[str, float]]
) -> tuple[go.Bar, go.Bar]:
    precision = [float(m["precision"]) for m in class_dicts]
    recall = [float(m["recall"]) for m in class_dicts]
    precision_trace = go.Bar(
        x=classes, y=precision, name="Precision", marker_color="cornflowerblue"
    )
    recall_trace = go.Bar(x=classes, y=recall, name="Recall", marker_color="lightcoral")
    return precision_trace, recall_trace


def _hyperparams_table_trace(hyperparams: Hyperparams) -> go.Table:
    return go.Table(
        header={
            "values": ["Parameter", "Value"],
            "fill_color": "steelblue",
            "font_color": "white",
        },
        cells={
            "values": [
                list(hyperparams.model_dump().keys()),
                list(hyperparams.model_dump().values()),
            ],
            "fill_color": "lavender",
        },
    )


def _next_report_path(model: str) -> tuple[Path, str, int, str]:
    """Report output path plus the (model_short, version, timestamp) pieces its title
    string is built from."""
    _REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_short = model.rsplit("/", maxsplit=1)[-1]
    version = len(list(_REPORTS_DIR.glob(f"run_{model_short}_v*.html"))) + 1
    out_path = _REPORTS_DIR / f"run_{model_short}_v{version}_{timestamp}.html"
    return out_path, model_short, version, timestamp


def generate_html_report(
    label_names: list[str],
    cm: npt.NDArray[np.int_],
    result: EvaluationResult,
    hyperparams: Hyperparams,
) -> Path:
    out_path, model_short, version, timestamp = _next_report_path(hyperparams.model)
    classes, class_dicts = _class_metric_pairs(result.report_dict)
    precision_trace, recall_trace = _precision_recall_traces(classes, class_dicts)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Confusion Matrix",
            "Per-class F1",
            "Precision vs Recall",
            "Hyperparameters",
        ),
        specs=[
            [{"type": "heatmap"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "table"}],
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )
    fig.add_trace(_confusion_matrix_trace(label_names, cm), row=1, col=1)
    fig.add_trace(_f1_bar_trace(classes, class_dicts), row=1, col=2)
    fig.add_trace(precision_trace, row=2, col=1)
    fig.add_trace(recall_trace, row=2, col=1)
    fig.add_trace(_hyperparams_table_trace(hyperparams), row=2, col=2)

    fig.update_layout(
        title_text=(
            f"Bert Tunning — Experiment Report  |  "
            f"{model_short} v{version}  |  "
            f"macro F1: {result.macro_f1:.3f}  "
            f"accuracy: {result.accuracy:.3f}  |  {timestamp}"
        ),
        height=900,
        barmode="group",
        showlegend=True,
    )
    fig.update_xaxes(tickangle=-35)

    fig.write_html(str(out_path))
    log.info("HTML report saved → %s", out_path)
    return out_path
