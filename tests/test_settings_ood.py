from src.schema import PredictResult
from src.settings import Settings


def test_ood_settings_have_expected_defaults() -> None:
    assert Settings.OOD_PCA_COMPONENTS == 64  # noqa: PLR2004
    assert Settings.OOD_MAHALANOBIS_WEIGHT == 0.7  # noqa: PLR2004
    assert Settings.OOD_THRESHOLD == 2.5  # noqa: PLR2004


def test_predict_result_ood_fields_default_to_none() -> None:
    result = PredictResult(label="decreto", confidence=0.9, certain=True)
    assert result.ood_score is None
    assert result.in_distribution is None
