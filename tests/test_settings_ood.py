from src.schema import PredictResult
from src.settings import Settings


def test_ood_settings_have_expected_defaults() -> None:
    assert Settings.OOD_PCA_COMPONENTS == 64  # noqa: PLR2004
    assert Settings.OOD_MAHALANOBIS_P_THRESHOLD == 0.01  # noqa: PLR2004
    assert Settings.OOD_COSINE_THRESHOLD == 2.5  # noqa: PLR2004


def test_predict_result_ood_fields_default_to_none() -> None:
    result = PredictResult(label="decreto", confidence=0.9, certain=True)
    assert result.mahalanobis_p_value is None
    assert result.cosine_z is None
    assert result.in_distribution is None
