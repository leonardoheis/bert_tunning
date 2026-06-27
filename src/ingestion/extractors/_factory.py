from src.ingestion.extractors._base import ExtractorBase
from src.ingestion.extractors.markitdown import MarkItDownExtractor
from src.ingestion.extractors.ocr import OCRExtractor

_REGISTRY: dict[str, type[ExtractorBase]] = {
    "markitdown": MarkItDownExtractor,
    "ocr": OCRExtractor,
}


def get_extractor(name: str) -> ExtractorBase:
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY)
        msg = f"Unknown extractor '{name}'. Available: {available}"
        raise KeyError(msg)
    return _REGISTRY[name]()
