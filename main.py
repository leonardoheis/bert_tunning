import argparse
import logging
import os
import shutil
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).parent / "models"))

from classifier import ClassiflowClassifier
from config import CACHE_PATH, DOCS_ROOT, MODEL_NAME, OUTPUT_DIR
from dataset import load_or_build_dataset
from logger import setup_logging
from training import train

log = logging.getLogger(__name__)

_LOG_FILE  = Path(__file__).parent / "logs" / "classiflow.log"
_CACHE     = Path(CACHE_PATH)
_MODEL_DIR = Path(OUTPUT_DIR)


def _release_log_file() -> None:
    root = logging.getLogger()
    for handler in root.handlers[:]:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == _LOG_FILE:
            handler.close()
            root.removeHandler(handler)


def clean_state() -> None:
    targets = [
        (_LOG_FILE,  "log file"),
        (_CACHE,     "dataset cache"),
        (_MODEL_DIR, "model checkpoints"),
    ]
    for path, label in targets:
        if not path.exists():
            log.info("Clean: %s not found, skipping", label)
            continue
        if path == _LOG_FILE:
            _release_log_file()
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        log.info("Clean: deleted %s (%s)", label, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Classiflow — Spanish municipal document classifier (model: {MODEL_NAME})"
    )
    parser.add_argument("--mode", choices=["train", "predict", "predict_folder", "clean"],
                        default="train")
    parser.add_argument("--docs_root",          default=DOCS_ROOT,
                        help="Root folder with labeled subfolders (train mode)")
    parser.add_argument("--model_path",         default=f"{OUTPUT_DIR}/final",
                        help="Saved model path (predict/predict_folder mode)")
    parser.add_argument("--pdf",                help="Single PDF to classify")
    parser.add_argument("--folder",             help="Folder of PDFs to classify")
    parser.add_argument("--no_ocr",             action="store_true",
                        help="Skip OCR fallback for scanned PDFs")
    parser.add_argument("--rebuild_cache",      action="store_true",
                        help="Force re-extraction even if cache exists")
    parser.add_argument("--max_docs_per_class", type=int, default=None,
                        help="Limit docs per class for quick test runs (skips cache)")
    parser.add_argument("--threshold",          type=float, default=0.70,
                        help="Confidence threshold (default 0.70)")
    parser.add_argument("--clean",              action="store_true",
                        help="Wipe logs, cache and model checkpoints before running")
    parser.add_argument("--no_wandb",           action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--debug",              action="store_true",
                        help="Enable DEBUG level logging")
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)

    if args.clean or args.mode == "clean":
        log.info("Cleaning state...")
        clean_state()
        setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
        log.info("Clean complete")
        if args.mode == "clean":
            return

    log.info("Mode: %s", args.mode)

    if args.mode == "train":
        df = load_or_build_dataset(
            args.docs_root,
            cache_path=CACHE_PATH,
            use_ocr=not args.no_ocr,
            rebuild=args.rebuild_cache,
            max_docs_per_class=args.max_docs_per_class,
        )
        if len(df) == 0:
            log.error("No documents found. Check --docs_root path: %s", args.docs_root)
        else:
            train(df, use_wandb=not args.no_wandb)

    elif args.mode == "predict":
        if not args.pdf:
            log.error("Provide --pdf in predict mode")
            return
        clf = ClassiflowClassifier(args.model_path, args.threshold)
        result = clf.predict_pdf(args.pdf)
        print(f"\n{'─' * 50}")
        print(f"  File      : {result.get('filename', args.pdf)}")
        print(f"  Label     : {result['label']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Certain   : {result['certain']}")
        print("\n  All scores:")
        for lbl, sc in sorted(result["all_scores"].items(), key=lambda x: -x[1]):
            bar = "█" * int(sc * 40)
            print(f"    {lbl:<38} {sc:.4f}  {bar}")

    elif args.mode == "predict_folder":
        if not args.folder:
            log.error("Provide --folder in predict_folder mode")
            return
        clf = ClassiflowClassifier(args.model_path, args.threshold)
        df_out = clf.predict_folder(args.folder)
        out_csv = "classiflow_predictions.csv"
        df_out.to_csv(out_csv, index=False)
        log.info("Results saved to %s", out_csv)


if __name__ == "__main__":
    main()
