import logging

import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)


def make_split(
    df: pd.DataFrame,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        train_df, temp_df = train_test_split(
            df, test_size=0.30, stratify=df["label_id"], random_state=seed
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, stratify=temp_df["label_id"], random_state=seed
        )
    except ValueError as e:
        log.warning("Stratified split failed (%s) — using random split", e)
        train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed)
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed)

    log.info("Split — train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df
