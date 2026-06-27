from pydantic import BaseModel, ConfigDict


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    hf_id: str
    max_tokens: int
    lr: float
    batch_size: int
    grad_accum: int
    force_fp32: bool


def _build_registry() -> dict[str, ModelConfig]:
    from src.training.models import beto, xlm_roberta  # noqa: PLC0415
    return {
        "xlm-roberta": xlm_roberta.config,
        "beto": beto.config,
    }


MODEL_REGISTRY: dict[str, ModelConfig] = _build_registry()


def get_model_config(key: str) -> ModelConfig:
    if key not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY)
        msg = f"Unknown model '{key}'. Available: {available}"
        raise KeyError(msg)
    return MODEL_REGISTRY[key]
