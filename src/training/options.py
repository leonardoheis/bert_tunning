from pydantic import BaseModel, ConfigDict, field_validator

from src.settings import Settings


class TrainingRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    epochs: int = Settings.EPOCHS
    early_stop_patience: int = Settings.EARLY_STOP_PATIENCE
    chunk_strategy: str = Settings.CHUNK_STRATEGY
    seed: int = Settings.SEED
    output_dir: str = Settings.OUTPUT_DIR
    use_wandb: bool = True

    @field_validator("max_docs_per_class")
    @classmethod
    def validate_max_docs_per_class(cls, v: int | None) -> int | None:
        if v is not None and v <= Settings.MAX_DOCS_PER_CLASS:
            msg = f"max_docs_per_class must be greater than {Settings.MAX_DOCS_PER_CLASS}, got {v}"
            raise ValueError(msg)
        return v
