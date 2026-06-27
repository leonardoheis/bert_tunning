import logging
from dataclasses import dataclass
from pathlib import Path

from src.exceptions import ClassiflowError

log = logging.getLogger(__name__)


@dataclass
class PDFExtractionError(ClassiflowError):
    pdf_path: str
    cause: Exception

    def __post_init__(self) -> None:
        super().__init__(str(self))
        log.warning(str(self))

    def __str__(self) -> str:
        name = Path(self.pdf_path).name
        return f"Text extraction failed for '{name}' ({type(self.cause).__name__}: {self.cause})"


@dataclass
class OCRError(ClassiflowError):
    pdf_path: str
    cause: Exception

    def __post_init__(self) -> None:
        super().__init__(str(self))
        log.exception(str(self))

    def __str__(self) -> str:
        name = Path(self.pdf_path).name
        return f"OCR failed for '{name}' ({type(self.cause).__name__}: {self.cause})"
