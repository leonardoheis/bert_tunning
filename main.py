import logging

import click
import uvicorn

from src.api.app import create_app
from src.cli.clean import clean_cmd
from src.cli.ood_calibration import evaluate_ood_calibration_cmd
from src.cli.ood_stats import compute_ood_stats_cmd
from src.cli.predict import predict_cmd, predict_folder_cmd
from src.cli.svm_classifiers import compute_svm_classifiers_cmd
from src.cli.train import train_cmd
from src.logger import setup_logging
from src.settings import Settings

log = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    """Bert Tunning — Spanish municipal document classifier."""


cli.add_command(train_cmd, name="train")
cli.add_command(predict_cmd, name="predict")
cli.add_command(predict_folder_cmd, name="predict-folder")
cli.add_command(clean_cmd, name="clean")
cli.add_command(compute_ood_stats_cmd, name="compute-ood-stats")
cli.add_command(evaluate_ood_calibration_cmd, name="evaluate-ood-calibration")
cli.add_command(compute_svm_classifiers_cmd, name="compute-svm-classifiers")


@cli.command("serve")
@click.option("--model-path", required=True, help="Path to saved model directory")
@click.option("--host", default=Settings.HOST, show_default=True)
@click.option("--port", default=Settings.API_PORT, show_default=True)
@click.option("--threshold", default=Settings.model_threshold, show_default=True)
def serve_cmd(model_path: str, host: str, port: int, threshold: float) -> None:
    """Start the FastAPI inference server."""
    log_file = setup_logging()
    log.info("Logging to %s", log_file)
    app = create_app(model_path=model_path, threshold=threshold)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli()
