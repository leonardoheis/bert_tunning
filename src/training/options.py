from pydantic import BaseModel, ConfigDict

from src.settings import Settings


class TrainingRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    epochs: int = Settings.EPOCHS
    early_stop_patience: int = Settings.EARLY_STOP_PATIENCE
    chunk_strategy: str = Settings.CHUNK_STRATEGY
    seed: int = Settings.SEED
    output_dir: str = Settings.OUTPUT_DIR
    use_wandb: bool = True
