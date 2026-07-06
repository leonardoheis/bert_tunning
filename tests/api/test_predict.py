from http import HTTPStatus
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.routes.predict.schemas import PredictResponse


def test_health_endpoint() -> None:
    app = create_app(model_path="fake/path")
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"status": "healthy", "message": "The API is running smoothly."}


def test_predict_rejects_non_pdf() -> None:
    app = create_app(model_path="fake/path")
    app.state.clf = MagicMock()  # satisfy dependency; classifier is irrelevant to this assertion
    client = TestClient(app)
    response = client.post(
        "/predict",
        files={"file": ("document.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert "PDF" in response.json()["detail"]


def test_predict_response_has_ood_fields() -> None:
    response = PredictResponse(filename="doc.pdf", label="decreto", confidence=0.9, certain=True)
    assert response.mahalanobis_p_value is None
    assert response.cosine_z is None
    assert response.in_distribution is None
