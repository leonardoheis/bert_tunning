from abc import ABC, abstractmethod


class ExtractorBase(ABC):
    @abstractmethod
    def extract(self, pdf_path: str) -> str: ...
