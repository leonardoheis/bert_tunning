import os
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).parent.parent


class _Settings(BaseSettings):
    # ── Server ────────────────────────────────────────────────────────────────
    API_PORT: int = 8000
    HOST: str = "127.0.0.1"
    THRESHOLD: float = 0.70
    HF_HOME: str = str(_PROJECT_ROOT / "models")

    MAX_LENGTH: int = 512

    # ── Paths ─────────────────────────────────────────────────────────────────
    DOCS_ROOT: str = r"C:\Users\YourUser\Downloads\downloadsdocs\downloads"
    OUTPUT_DIR: str = "./models/bert_tunning_model"
    CACHE_PATH: str = "./data/bert_tunning_cache.parquet"

    # ── Model ─────────────────────────────────────────────────────────────────
    MODEL_NAME: str = "xlm-roberta-base"
    MODEL_KEY: str = "xlm-roberta"

    # ── Training ──────────────────────────────────────────────────────────────
    MAX_TOKENS: int = 512
    CHUNK_STRATEGY: str = "first"
    BATCH_SIZE: int = 8
    GRAD_ACCUM: int = 8
    EPOCHS: int = 15
    LR: float = 2e-5
    FORCE_FP32: bool = False
    EARLY_STOP_PATIENCE: int = 5
    SEED: int = 42

    # ── W&B ───────────────────────────────────────────────────────────────────
    WANDB_ENTITY: str = "leonardo-a-heis"
    WANDB_PROJECT: str = "bert_tunning"

    # ── Extraction ────────────────────────────────────────────────────────────
    MIN_TEXT_FOR_OCR: int = 50
    MIN_USABLE_TEXT: int = 20

    # ── Label mapping ─────────────────────────────────────────────────────────
    EXCLUDE_LABELS: set[str] = {"convenios"}
    FOLDER_TO_LABEL: dict[str, str] = {
        "decretos": "decreto",
        "decreto_concejo_municipal": "decreto_concejo_municipal",
        "ordenanzas": "ordenanza",
        "decreto_ordenanzas": "decreto_ordenanza",
        "resoluciones": "resolucion",
        "resoluciones_concejo_municipal": "resolucion_concejo_municipal",
        "declaraciones_concejo_municipal": "declaracion_concejo_municipal",
        "convenios": "convenio",
    }

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def _apply_hf_home(self) -> "_Settings":
        os.environ.setdefault("HF_HOME", self.HF_HOME)
        return self

    @property
    def model_threshold(self) -> float:
        return self.THRESHOLD

    @property
    def max_length(self) -> int:
        return self.MAX_LENGTH

    @property
    def default_model_path(self) -> str:
        return str(Path(self.OUTPUT_DIR) / "final")


Settings = _Settings()
