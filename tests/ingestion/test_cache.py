from pathlib import Path

from src.ingestion.cache import _resolve_path


def test_resolve_path_no_cap() -> None:
    result = _resolve_path("./data/bert_tunning_cache.parquet", None)
    assert result == Path("./data/bert_tunning_cache.parquet")


def test_resolve_path_with_cap() -> None:
    result = _resolve_path("./data/bert_tunning_cache.parquet", 100)
    assert result == Path("./data/bert_tunning_cache_100.parquet")


def test_resolve_path_preserves_extension() -> None:
    result = _resolve_path("./data/cache.parquet", 50)
    assert result.suffix == ".parquet"
    assert "50" in result.name
