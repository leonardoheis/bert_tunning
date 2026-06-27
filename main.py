import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).parent / "models"))

import click

from src.cli.clean import clean_cmd
from src.cli.predict import predict_cmd, predict_folder_cmd
from src.cli.train import train_cmd


@click.group()
def cli() -> None:
    """Classiflow — Spanish municipal document classifier."""


cli.add_command(train_cmd, name="train")
cli.add_command(predict_cmd, name="predict")
cli.add_command(predict_folder_cmd, name="predict-folder")
cli.add_command(clean_cmd, name="clean")


@cli.command("serve")
@click.option("--model-path", required=True, help="Path to saved model directory")
@click.option("--host", default="0.0.0.0", show_default=True)  # noqa: S104
@click.option("--port", default=8000, show_default=True)
@click.option("--threshold", default=0.70, show_default=True)
def serve_cmd(model_path: str, host: str, port: int, threshold: float) -> None:
    """Start the FastAPI inference server."""
    import uvicorn  # noqa: PLC0415

    from src.api.app import create_app  # noqa: PLC0415

    app = create_app(model_path=model_path, threshold=threshold)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli()
