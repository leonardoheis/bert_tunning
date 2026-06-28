import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.schema import Hyperparams, ReportDict

log = logging.getLogger(__name__)

_REPORTS_DIR = Path(__file__).parent / "reports"


def generate_html_report(
    label_names: list[str],
    cm: npt.NDArray[np.int_],
    report_dict: ReportDict,
    hyperparams: Hyperparams,
) -> Path:
    _REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = _REPORTS_DIR / f"run_{timestamp}.html"

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

    # Confusion matrix heatmap
    cm_pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    heatmap_text = [
        [f"{cm[r][c]}<br>({cm_pct[r][c]:.0%})" for c in range(len(label_names))]
        for r in range(len(label_names))
    ]
    fig.add_trace(
        go.Heatmap(
            z=cm_pct,
            x=label_names,
            y=label_names,
            text=heatmap_text,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        ),
        row=1,
        col=1,
    )

    # Per-class F1 bar — skip summary rows and scalar entries
    classes = [k for k in report_dict if k not in ("accuracy", "macro avg", "weighted avg")]
    class_dicts = [v for c in classes if isinstance(v := report_dict[c], dict)]
    f1_scores = [float(m["f1-score"]) for m in class_dicts]
    fig.add_trace(
        go.Bar(x=classes, y=f1_scores, marker_color="steelblue", name="F1"),
        row=1,
        col=2,
    )

    # Precision vs Recall bar
    precision = [float(m["precision"]) for m in class_dicts]
    recall = [float(m["recall"]) for m in class_dicts]
    fig.add_trace(
        go.Bar(x=classes, y=precision, name="Precision", marker_color="cornflowerblue"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=classes, y=recall, name="Recall", marker_color="lightcoral"), row=2, col=1
    )

    # Hyperparameters table
    fig.add_trace(
        go.Table(
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
        ),
        row=2,
        col=2,
    )

    macro_raw = report_dict.get("macro avg", {})
    macro_f1 = float(macro_raw["f1-score"]) if isinstance(macro_raw, dict) else 0.0
    accuracy_raw = report_dict.get("accuracy", 0.0)
    accuracy = float(accuracy_raw) if isinstance(accuracy_raw, float) else 0.0
    fig.update_layout(
        title_text=(
            f"Bert Tunning — Experiment Report  |  "
            f"macro F1: {macro_f1:.3f}  "
            f"accuracy: {accuracy:.3f}  |  {timestamp}"
        ),
        height=900,
        barmode="group",
        showlegend=True,
    )
    fig.update_xaxes(tickangle=-35)

    fig.write_html(str(out_path))
    log.info("HTML report saved → %s", out_path)
    return out_path
