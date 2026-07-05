from src.training.models import beto, minilm, xlm_roberta
from src.training.models.config import ModelConfig

__all__ = ["MODEL_REGISTRY", "ModelConfig", "get_model_config"]


def _build_registry() -> dict[str, ModelConfig]:
    return {
        "xlm-roberta": xlm_roberta.config,
        "beto": beto.config,
        "minilm": minilm.config,
    }


MODEL_REGISTRY: dict[str, ModelConfig] = _build_registry()


def get_model_config(key: str) -> ModelConfig:
    if key not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY)
        msg = f"Unknown model '{key}'. Available: {available}"
        raise KeyError(msg)
    return MODEL_REGISTRY[key]
