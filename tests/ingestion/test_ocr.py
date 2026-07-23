import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from src.ingestion.extractors.ocr import OCRExtractor


def test_get_reader_only_constructs_once_under_concurrent_calls() -> None:
    """Regression test for the race functools.lru_cache didn't actually prevent: two
    threads racing OCRExtractor's first-time reader init used to both call
    easyocr.Reader(...) concurrently. A sleep inside the mocked constructor widens the
    race window so this test would have caught the old bug."""
    call_count = 0

    def slow_reader(*_args: object, **_kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        time.sleep(0.05)
        return MagicMock()

    extractor = OCRExtractor()
    with (
        patch("src.ingestion.extractors.ocr.easyocr.Reader", side_effect=slow_reader),
        ThreadPoolExecutor(max_workers=8) as pool,
    ):
        list(pool.map(lambda _: extractor._get_reader(), range(8)))  # noqa: SLF001

    assert call_count == 1


def test_get_reader_returns_the_same_instance_every_call() -> None:
    extractor = OCRExtractor()
    with patch("src.ingestion.extractors.ocr.easyocr.Reader", return_value=MagicMock()):
        first = extractor._get_reader()  # noqa: SLF001
        second = extractor._get_reader()  # noqa: SLF001

    assert first is second


def test_warm_initializes_the_reader() -> None:
    extractor = OCRExtractor()
    with patch(
        "src.ingestion.extractors.ocr.easyocr.Reader", return_value=MagicMock()
    ) as mock_reader:
        extractor.warm()

    mock_reader.assert_called_once()
    assert extractor._reader is not None  # noqa: SLF001
