import logging

import click

from logger import setup_logging
from src.inference.pipeline import predict_folder, predict_pdf
from src.settings import Settings

log = logging.getLogger(__name__)

_DEFAULT_MODEL = f"{Settings.OUTPUT_DIR}/final"


@click.command("predict")
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--model-path", default=_DEFAULT_MODEL, show_default=True)
@click.option("--threshold", default=0.70, show_default=True, help="Confidence threshold")
@click.option("--no-ocr", is_flag=True, default=False)
@click.option("--debug", is_flag=True, default=False)
def predict_cmd(
    pdf_path: str,
    model_path: str,
    threshold: float,
    no_ocr: bool,  # noqa: FBT001
    debug: bool,  # noqa: FBT001
) -> None:
    """Classify a single PDF document."""
    setup_logging(level=logging.DEBUG if debug else logging.INFO)
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


@click.command("predict-folder")
@click.argument("folder_path", type=click.Path(exists=True, file_okay=False))
@click.option("--model-path", default=_DEFAULT_MODEL, show_default=True)
@click.option("--threshold", default=0.70, show_default=True)
@click.option("--no-ocr", is_flag=True, default=False)
@click.option("--output", default="classiflow_predictions.csv", show_default=True)
@click.option("--debug", is_flag=True, default=False)
def predict_folder_cmd(  # noqa: PLR0913
    folder_path: str,
    model_path: str,
    threshold: float,
    no_ocr: bool,  # noqa: FBT001
    output: str,
    debug: bool,  # noqa: FBT001
) -> None:
    """Classify all PDFs in a folder and save results to CSV."""
    setup_logging(level=logging.DEBUG if debug else logging.INFO)
    df_out = predict_folder(model_path, folder_path, threshold=threshold, use_ocr=not no_ocr)
    df_out.to_csv(output, index=False)
    log.info("Results saved to %s", output)
