import logging
from pathlib import Path

import pandas as pd

from src.ingestion.scan import build_dataset

log = logging.getLogger(__name__)


def _resolve_path(base_path: str, max_docs_per_class: int | None) -> Path:
    p = Path(base_path)
    if max_docs_per_class:
        return p.parent / f"{p.stem}_{max_docs_per_class}{p.suffix}"
    return p


def load_or_build(
    docs_root: str,
    *,
    cache_path: str,
    use_ocr: bool = True,
    rebuild: bool = False,
    max_docs_per_class: int | None = None,
) -> pd.DataFrame:
    cache = _resolve_path(cache_path, max_docs_per_class)

    if not rebuild and cache.exists():
        log.info("Cache found — loading from %s", cache)
        df = pd.read_parquet(cache)
        log.info("Loaded %d docs from cache", len(df))
        return df

    if rebuild and cache.exists():
        log.info("rebuild=True — removing cache: %s", cache)
        cache.unlink()

    log.info("Extracting PDFs from: %s", docs_root)
    df = build_dataset(docs_root, use_ocr=use_ocr, max_docs_per_class=max_docs_per_class)

    if len(df) > 0:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
        log.info("Cached %d docs → %s (%.1f MB)", len(df), cache, cache.stat().st_size / 1_048_576)

    return df
