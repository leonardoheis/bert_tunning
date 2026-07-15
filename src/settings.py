import os
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).parent.parent


class _Settings(BaseSettings):
    # ── Server ────────────────────────────────────────────────────────────────
    API_PORT: int = 8000
    HOST: str = "127.0.0.1"
    # 25 MB -- municipal decree/ordinance PDFs in this corpus are small (a few hundred KB
    # typical); this is generous headroom, not a tight budget. Enforced by
    # src/api/routes/predict/endpoints.py's chunked read, not by FastAPI/uvicorn, which
    # impose no default body-size limit on their own.
    MAX_UPLOAD_SIZE_BYTES: int = 25 * 1024 * 1024
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
    # Calibrated 2026-07-12 against BETO v2 (bert_tunning_model_beto_v2) via
    # evaluate-ood-calibration, after switching to the empirical (rank-based)
    # p-value: the prior placeholder (0.01) gave a 9.38% empirical false-positive
    # rate. The naive 1%-target suggestion (0.000743) turned out to be exactly the
    # empirical p-value's own resolution floor for this model — 1/(N_train+1) with
    # N_train=1344 — so that threshold is mathematically unreachable and the signal
    # would never fire (0.00% empirical FP rate, i.e. permanently inert). Achievable
    # p-values are quantized in steps of 1/(N_train+1) ≈ 0.0007435; 17/288 (5.90%) of
    # the held-out test set ties exactly at the floor, so any threshold above the
    # floor and at/below the next step (2/(N_train+1) ≈ 0.001487) captures that same
    # 5.90% — the practical minimum FP rate this training-set size can resolve, well
    # above the original 1% target. 0.001 sits in that window. Re-run
    # evaluate-ood-calibration (and re-derive this quantization math) if the training
    # corpus changes materially — a larger corpus lowers the floor and may support a
    # tighter target again.
    OOD_MAHALANOBIS_P_THRESHOLD: float = 0.001
    OOD_COSINE_THRESHOLD: float = 13.7366
    OOD_KNN_NEIGHBORS: int = 10
    # Calibrated 2026-07-08 against BETO v2 (bert_tunning_model_beto_v2) via
    # evaluate-ood-calibration: the prior placeholder (5.0) gave a 21.88% empirical
    # false-positive rate vs. the 1% target; 26.125 is the suggested threshold for a
    # 1% target FP rate on that run. Same caveat as OOD_COSINE_THRESHOLD — re-run
    # evaluate-ood-calibration if the training corpus changes materially.
    OOD_KNN_DISTANCE_THRESHOLD: float = 26.125
    TARGET_FP_RATE: float = 0.01
    # Uncalibrated placeholder, matching how OOD_COSINE_THRESHOLD/OOD_KNN_DISTANCE_THRESHOLD
    # started before their first evaluate-ood-calibration --write-thresholds run. Run that
    # command for any model using this signal before trusting it in production.
    OOD_TFIDF_COSINE_THRESHOLD: float = 2.5
    OOD_TFIDF_MAX_FEATURES: int = 5000
    # Excludes terms appearing in over half the training corpus (shared legal boilerplate --
    # "considerando", "por cuanto", etc.) from the TF-IDF vocabulary, so cosine distance isn't
    # diluted by tokens every document has regardless of municipality or type.
    OOD_TFIDF_MAX_DF: float = 0.5

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
