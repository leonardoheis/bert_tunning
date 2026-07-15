from src.schema import Hyperparams, ReportDict
from src.training.reporting import (
    _class_metric_pairs,
    _f1_bar_trace,
    _hyperparams_table_trace,
    _precision_recall_traces,
)


def _make_hyperparams() -> Hyperparams:
    # As of this test's authoring, Hyperparams has alias_generator=to_camel without
    # populate_by_name=True, so only the camelCase alias is accepted at construction --
    # NOT how src/training/pipeline.py actually constructs it (snake_case kwargs), which
    # meant that call was itself broken (see task/47-fix-hyperparams-populate-by-name,
    # filed after this test surfaced the gap). model_validate() here also sidesteps
    # mypy's alias-blind synthesized __init__ signature for the camelCase dict below.
    return Hyperparams.model_validate(
        {
            "model": "beto",
            "epochs": 15,
            "batchSize": 8,
            "gradAccum": 8,
            "effectiveBatch": 64,
            "learningRate": 2e-5,
            "warmupSteps": 100,
            "precision": "bf16",
            "trainDocs": 1344,
            "numClasses": 9,
        }
    )


def test_class_metric_pairs_skips_summary_rows() -> None:
    report_dict: ReportDict = {
        "decreto": {"precision": 0.9, "recall": 0.8, "f1-score": 0.85},
        "ordenanza": {"precision": 0.7, "recall": 0.6, "f1-score": 0.65},
        "accuracy": 0.75,
        "macro avg": {"precision": 0.8, "recall": 0.7, "f1-score": 0.75},
        "weighted avg": {"precision": 0.8, "recall": 0.7, "f1-score": 0.75},
    }

    classes, class_dicts = _class_metric_pairs(report_dict)

    assert classes == ["decreto", "ordenanza"]
    assert class_dicts == [
        {"precision": 0.9, "recall": 0.8, "f1-score": 0.85},
        {"precision": 0.7, "recall": 0.6, "f1-score": 0.65},
    ]


def test_f1_bar_trace_reads_f1_score_field() -> None:
    classes = ["decreto", "ordenanza"]
    class_dicts = [{"f1-score": 0.85}, {"f1-score": 0.65}]

    trace = _f1_bar_trace(classes, class_dicts)

    assert list(trace.x) == classes
    assert list(trace.y) == [0.85, 0.65]


def test_precision_recall_traces_reads_matching_fields() -> None:
    classes = ["decreto", "ordenanza"]
    class_dicts = [
        {"precision": 0.9, "recall": 0.8},
        {"precision": 0.7, "recall": 0.6},
    ]

    precision_trace, recall_trace = _precision_recall_traces(classes, class_dicts)

    assert list(precision_trace.y) == [0.9, 0.7]
    assert list(recall_trace.y) == [0.8, 0.6]


def test_hyperparams_table_trace_includes_every_field() -> None:
    hyperparams = _make_hyperparams()

    trace = _hyperparams_table_trace(hyperparams)

    assert list(trace.cells.values[0]) == list(hyperparams.model_dump().keys())
    assert list(trace.cells.values[1]) == list(hyperparams.model_dump().values())
