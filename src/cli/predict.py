import logging

import click
import pandas as pd
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from src.inference.pipeline import predict_folder, predict_pdf
from src.logger import setup_logging
from src.settings import Settings

log = logging.getLogger(__name__)


class PredictFolderOptions(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, arbitrary_types_allowed=True, frozen=True)

    folder_path: str
    model_path: str = Settings.default_model_path
    threshold: float = Settings.THRESHOLD
    no_ocr: bool = False
    output: str = "bert_tunning_predictions.csv"
    debug: bool = False


@click.command("predict")
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--model-path", default=Settings.default_model_path, show_default=True)
@click.option(
    "--threshold", default=Settings.THRESHOLD, show_default=True, help="Confidence threshold"
)
@click.option("--no-ocr", is_flag=True, default=False)
@click.option("--debug", is_flag=True, default=False)
def predict_cmd(
    pdf_path: str,
    model_path: str,
    threshold: float,
    *,
    no_ocr: bool,
    debug: bool,
) -> None:
    """Classify a single PDF document."""
    log_file = setup_logging(level=logging.DEBUG if debug else logging.INFO)
    log.info("Logging to %s", log_file)
    result = predict_pdf(model_path, pdf_path, threshold=threshold, use_ocr=not no_ocr)

    click.echo(f"\n{'─' * 50}")
    click.echo(f"  File      : {result.filename or pdf_path}")
    click.echo(f"  Label     : {result.label}")
    click.echo(f"  Confidence: {result.confidence:.2%}")
    click.echo(f"  Certain   : {result.certain}")
    click.echo("\n  All scores:")
    for lbl, sc in sorted(result.all_scores.items(), key=lambda x: -x[1]):
        bar = "█" * int(sc * 40)
        click.echo(f"    {lbl:<38} {sc:.4f}  {bar}")


def _run_predict_folder(opts: PredictFolderOptions) -> None:
    log_file = setup_logging(level=logging.DEBUG if opts.debug else logging.INFO)
    log.info("Logging to %s", log_file)

    results = predict_folder(
        opts.model_path, opts.folder_path, threshold=opts.threshold, use_ocr=not opts.no_ocr
    )
    df = pd.DataFrame([r.model_dump() for r in results])
    df.insert(1, "model", opts.model_path)
    df.to_csv(opts.output, index=False)
    log.info("Results saved to %s", opts.output)


@click.command("predict-folder")
@click.argument("folder_path", type=click.Path(exists=True, file_okay=False))
@click.option("--model-path", default=Settings.default_model_path, show_default=True)
@click.option(
    "--threshold", default=Settings.THRESHOLD, show_default=True, help="Confidence threshold"
)
@click.option("--no-ocr", is_flag=True, default=False)
@click.option("--output", default="bert_tunning_predictions.csv", show_default=True)
@click.option("--debug", is_flag=True, default=False)
def predict_folder_cmd(**kwargs: str | float | bool) -> None:
    """Classify all PDFs in a folder and save results to CSV."""
    _run_predict_folder(PredictFolderOptions.model_validate(kwargs))
