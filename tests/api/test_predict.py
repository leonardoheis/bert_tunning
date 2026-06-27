from http import HTTPStatus

from fastapi.testclient import TestClient

from src.api.app import create_app


def test_health_endpoint() -> None:
    app = create_app(model_path="fake/path")
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"status": "ok"}


def test_predict_rejects_non_pdf() -> None:
    app = create_app(model_path="fake/path")
    client = TestClient(app)
    response = client.post(
        "/predict",
        files={"file": ("document.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert "PDF" in response.json()["detail"]
