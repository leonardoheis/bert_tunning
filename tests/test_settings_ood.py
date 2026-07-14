from src.schema import PredictResult
from src.settings import Settings


def test_ood_settings_have_expected_defaults() -> None:
    assert Settings.OOD_PCA_COMPONENTS == 64  # noqa: PLR2004
    # Calibrated against BETO v2 via evaluate-ood-calibration -- see settings.py's comment
    # for the run this value came from. Not the placeholder 0.01, and not the naive
    # 1%-target suggestion (0.000743), which sits at the empirical p-value's own
    # resolution floor for this model and is mathematically unreachable.
    assert Settings.OOD_MAHALANOBIS_P_THRESHOLD == 0.001  # noqa: PLR2004
    # Calibrated against BETO v2 via evaluate-ood-calibration -- see settings.py's comment
    # for the run this value came from. Not the theoretically-grounded 2.5 placeholder.
    assert Settings.OOD_COSINE_THRESHOLD == 13.7366  # noqa: PLR2004


def test_predict_result_ood_fields_default_to_none() -> None:
    result = PredictResult(label="decreto", confidence=0.9, certain=True)
    assert result.mahalanobis_p_value is None
    assert result.cosine_z is None
    assert result.in_distribution is None


def test_ood_tfidf_settings_have_defaults() -> None:
    assert Settings.OOD_TFIDF_COSINE_THRESHOLD > 0
    assert Settings.OOD_TFIDF_MAX_FEATURES > 0
