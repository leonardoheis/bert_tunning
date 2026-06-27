import pytest

from src.training.models import MODEL_REGISTRY, get_model_config


def test_registry_has_xlm_roberta() -> None:
    assert "xlm-roberta" in MODEL_REGISTRY


def test_registry_has_beto() -> None:
    assert "beto" in MODEL_REGISTRY


def test_get_model_config_returns_correct_hf_id() -> None:
    cfg = get_model_config("xlm-roberta")
    assert cfg.hf_id == "xlm-roberta-base"


def test_get_model_config_raises_on_unknown() -> None:
    with pytest.raises(KeyError, match="Unknown model"):
        get_model_config("nonexistent-model")


def test_model_config_is_immutable() -> None:
    cfg = get_model_config("xlm-roberta")
    with pytest.raises(AttributeError, match="cannot assign to field"):  # frozen dataclass
        cfg.name = "other"  # type: ignore[misc]
