import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer

from config import MAX_TOKENS, EXCLUDE_LABELS, FOLDER_TO_LABEL
from extraction import extract_pdf

log = logging.getLogger(__name__)


def prepare_text(text: str, tokenizer: AutoTokenizer, strategy: str = "first") -> str:
    if strategy == "first":
        return text

    tokens = tokenizer.encode(text, add_special_tokens=False)
    if strategy == "middle":
        if len(tokens) <= MAX_TOKENS - 2:
            return text
        half = (MAX_TOKENS - 2) // 2
        selected = tokens[:half] + tokens[-half:]
        return tokenizer.decode(selected, skip_special_tokens=True)

    return text


class ClassiflowDataset(TorchDataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer: AutoTokenizer, max_length: int = MAX_TOKENS):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


def build_dataset(
    docs_root: str,
    use_ocr: bool = True,
    max_docs_per_class: Optional[int] = None,
) -> pd.DataFrame:
    root = Path(docs_root)
    records = []
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
        pdfs  = list(folder.glob("*.pdf"))
        if max_docs_per_class:
            pdfs = pdfs[:max_docs_per_class]
        total = len(pdfs)
        log.info("Processing folder '%s' → label '%s' (%d PDFs)", folder_name, label, total)

        for i, pdf_path in enumerate(pdfs, start=1):
            log.info("[%d/%d] %s", i, total, pdf_path.name)
            text = extract_pdf(str(pdf_path), use_ocr_fallback=use_ocr)
            if text is None:
                skipped += 1
                continue
            records.append({"text": text, "label": label, "filename": pdf_path.name})
            log.info("  ✓ extracted %d chars", len(text))

    df = pd.DataFrame(records)
    log.info("Dataset built: %d docs loaded, %d skipped", len(df), skipped)
    log.info("Class distribution:\n%s", df["label"].value_counts().to_string())
    return df


def load_or_build_dataset(
    docs_root: str,
    cache_path: str = "./classiflow_cache.parquet",
    use_ocr: bool = True,
    rebuild: bool = False,
    max_docs_per_class: Optional[int] = None,
) -> pd.DataFrame:
    cache = Path(cache_path)

    # Skip cache entirely in sample mode — avoid polluting the full cache
    if max_docs_per_class:
        log.info("Sample mode active — cache disabled")
        return build_dataset(docs_root, use_ocr=use_ocr, max_docs_per_class=max_docs_per_class)

    if not rebuild and cache.exists():
        log.info("Cache found — loading from %s", cache_path)
        df = pd.read_parquet(cache_path)
        log.info("Loaded %d docs from cache", len(df))
        log.info("Class distribution (cached):\n%s", df["label"].value_counts().to_string())
        return df

    if rebuild and cache.exists():
        log.info("--rebuild_cache flag set — removing existing cache: %s", cache_path)
        cache.unlink()

    log.info("Extracting PDFs from: %s", docs_root)
    log.info("One-time extraction — may take several hours for large corpora")
    df = build_dataset(docs_root, use_ocr=use_ocr)

    if len(df) > 0:
        df.to_parquet(cache_path, index=False)
        size_mb = Path(cache_path).stat().st_size / 1_048_576
        log.info("Cached %d docs → %s (%.1f MB)", len(df), cache_path, size_mb)

    return df
