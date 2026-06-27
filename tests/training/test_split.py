import pandas as pd
import pytest

from src.training.split import make_split


@pytest.fixture
def balanced_df() -> pd.DataFrame:
    labels = ["a", "b", "c"]
    rows = [
        {"text": f"doc {i}", "label": label, "label_id": labels.index(label)}
        for label in labels
        for i in range(20)
    ]
    return pd.DataFrame(rows)


def test_split_sizes(balanced_df: pd.DataFrame) -> None:
    train, val, test = make_split(balanced_df, seed=42)
    assert len(train) + len(val) + len(test) == len(balanced_df)


def test_split_no_overlap(balanced_df: pd.DataFrame) -> None:
    train, val, test = make_split(balanced_df, seed=42)
    assert set(train.index).isdisjoint(set(val.index))
    assert set(train.index).isdisjoint(set(test.index))
    assert set(val.index).isdisjoint(set(test.index))


def test_split_is_deterministic(balanced_df: pd.DataFrame) -> None:
    t1, v1, _ = make_split(balanced_df, seed=42)
    t2, v2, _ = make_split(balanced_df, seed=42)
    assert list(t1.index) == list(t2.index)
    assert list(v1.index) == list(v2.index)
