import logging
from pathlib import Path

import pandas as pd

from config import EXCLUDE_LABELS, FOLDER_TO_LABEL
from src.ingestion.extract import extract_pdf

log = logging.getLogger(__name__)


def build_dataset(
    docs_root: str,
    *,
    use_ocr: bool = True,
    max_docs_per_class: int | None = None,
) -> pd.DataFrame:
    root = Path(docs_root)
    records: list[dict] = []
    skipped = 0

    if max_docs_per_class:
        log.info("Sample mode: capped at %d docs per class", max_docs_per_class)

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        folder_name = folder.name.lower()
        if folder_name in EXCLUDE_LABELS:
            log.info("Skipping excluded folder: %s", folder_name)
            continue

        label = FOLDER_TO_LABEL.get(folder_name, folder_name)
        pdfs = list(folder.glob("*.pdf"))
        if max_docs_per_class:
            pdfs = pdfs[:max_docs_per_class]
        log.info("Processing '%s' → '%s' (%d PDFs)", folder_name, label, len(pdfs))

        for i, pdf_path in enumerate(pdfs, start=1):
            log.info("[%d/%d] %s", i, len(pdfs), pdf_path.name)
            text = extract_pdf(str(pdf_path), use_ocr_fallback=use_ocr)
            if text is None:
                skipped += 1
                continue
            records.append({"text": text, "label": label, "filename": pdf_path.name})
            log.info("  extracted %d chars", len(text))

    df = pd.DataFrame(records)
    log.info("Dataset built: %d docs, %d skipped", len(df), skipped)
    if len(df):
        log.info("Class distribution:\n%s", df["label"].value_counts().to_string())
    return df
