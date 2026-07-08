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
    OUTPUT_DIR: str = "./models/bert_tunning_model_beto_v2"
    CACHE_PATH: str = "./data/bert_tunning_cache.parquet"

    # ── Model ─────────────────────────────────────────────────────────────────
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
        "otros": "otro",
    }

    MAX_DOCS_PER_CLASS: int = 10
    PREDICT_THRESHOLD: float = 0.70
    PREDICT_CONFIDENCE: float = 0.0
    OOD_PCA_COMPONENTS: int = 64
    OOD_MAHALANOBIS_P_THRESHOLD: float = 0.01
    OOD_COSINE_THRESHOLD: float = 2.5
    OOD_KNN_NEIGHBORS: int = 10
    # Calibrated 2026-07-08 against BETO v2 (bert_tunning_model_beto_v2) via
    # evaluate-ood-calibration: the prior placeholder (5.0) gave a 21.88% empirical
    # false-positive rate vs. the 1% target; 26.125 is the suggested threshold for a
    # 1% target FP rate on that run. Same caveat as OOD_COSINE_THRESHOLD — re-run
    # evaluate-ood-calibration if the training corpus changes materially.
    OOD_KNN_DISTANCE_THRESHOLD: float = 26.125
    TARGET_FP_RATE: float = 0.01

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

    @property
    def max_docs_per_class(self) -> int:
        return self.MAX_DOCS_PER_CLASS

    @property
    def predict_confidence(self) -> float:
        return self.PREDICT_CONFIDENCE

    @property
    def predict_threshold(self) -> float:
        return self.PREDICT_THRESHOLD


Settings = _Settings()
