
import pandas as pd

from src.ingestion.cache import load_or_build


def run(
    docs_root: str,
    *,
    cache_path: str,
    use_ocr: bool = True,
    rebuild: bool = False,
    max_docs_per_class: int | None = None,
) -> pd.DataFrame:
    return load_or_build(
        docs_root,
        cache_path=cache_path,
        use_ocr=use_ocr,
        rebuild=rebuild,
        max_docs_per_class=max_docs_per_class,
    )
