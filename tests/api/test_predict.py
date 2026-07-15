from http import HTTPStatus
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.routes.predict.schemas import PredictResponse
from src.schema import ExtractionMetadata, OodMetrics, PredictResult


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


def test_predict_response_has_ood_metrics_field() -> None:
    response = PredictResponse(filename="doc.pdf", label="decreto", confidence=0.9, certain=True)
    assert response.ood_metrics is None


def test_predict_endpoint_returns_extraction_metadata() -> None:
    app = create_app(model_path="fake/path")
    mock_clf = MagicMock()
    mock_clf.predict_text.return_value = PredictResult(
        label="decreto", confidence=0.9, certain=True
    )
    app.state.clf = mock_clf

    fake_extraction = ExtractionMetadata(
        text="hola mundo", extractor_used="OCRExtractor", char_count=10
    )
    with patch(
        "src.api.routes.predict.endpoints.extract_pdf_with_metadata", return_value=fake_extraction
    ):
        client = TestClient(app)
        response = client.post(
            "/predict",
            files={"file": ("doc.pdf", b"%PDF-1.4 fake content", "application/pdf")},
        )

    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body["extractedText"] == "hola mundo"
    assert body["extractorUsed"] == "OCRExtractor"


def test_predict_endpoint_returns_ood_metrics() -> None:
    expected_knn_distance = 4.2
    app = create_app(model_path="fake/path")
    mock_clf = MagicMock()
    mock_clf.predict_text.return_value = PredictResult(
        label="decreto",
        confidence=0.9,
        certain=True,
        ood_metrics=OodMetrics(
            mahalanobis_p_value=0.5,
            mahalanobis_p_value_theoretical=0.6,
            cosine_z=1.0,
            knn_distance=expected_knn_distance,
            in_distribution=True,
        ),
    )
    app.state.clf = mock_clf

    fake_extraction = ExtractionMetadata(
        text="hola mundo", extractor_used="OCRExtractor", char_count=10
    )
    with patch(
        "src.api.routes.predict.endpoints.extract_pdf_with_metadata", return_value=fake_extraction
    ):
        client = TestClient(app)
        response = client.post(
            "/predict",
            files={"file": ("doc.pdf", b"%PDF-1.4 fake content", "application/pdf")},
        )

    assert response.status_code == HTTPStatus.OK
    assert response.json()["oodMetrics"]["knnDistance"] == expected_knn_distance


def test_predict_endpoint_returns_review_route() -> None:
    app = create_app(model_path="fake/path")
    mock_clf = MagicMock()
    mock_clf.predict_text.return_value = PredictResult(
        label="decreto", confidence=0.9, certain=True, review_route="accept"
    )
    app.state.clf = mock_clf

    fake_extraction = ExtractionMetadata(
        text="hola mundo", extractor_used="OCRExtractor", char_count=10
    )
    with patch(
        "src.api.routes.predict.endpoints.extract_pdf_with_metadata", return_value=fake_extraction
    ):
        client = TestClient(app)
        response = client.post(
            "/predict",
            files={"file": ("doc.pdf", b"%PDF-1.4 fake content", "application/pdf")},
        )

    assert response.status_code == HTTPStatus.OK
    assert response.json()["reviewRoute"] == "accept"


def test_predict_endpoint_routes_unreadable_document_to_human_review() -> None:
    app = create_app(model_path="fake/path")
    app.state.clf = MagicMock()
    fake_extraction = ExtractionMetadata(text=None, extractor_used=None, char_count=0)
    with patch(
        "src.api.routes.predict.endpoints.extract_pdf_with_metadata", return_value=fake_extraction
    ):
        client = TestClient(app)
        response = client.post(
            "/predict",
            files={"file": ("doc.pdf", b"%PDF-1.4 fake content", "application/pdf")},
        )

    assert response.status_code == HTTPStatus.OK
    assert response.json()["reviewRoute"] == "human_review"


def test_predict_endpoint_returns_theoretical_mahalanobis_p_value() -> None:
    expected_theoretical_p = 0.1708
    app = create_app(model_path="fake/path")
    mock_clf = MagicMock()
    mock_clf.predict_text.return_value = PredictResult(
        label="decreto",
        confidence=0.9,
        certain=True,
        ood_metrics=OodMetrics(
            mahalanobis_p_value=0.5,
            mahalanobis_p_value_theoretical=expected_theoretical_p,
            cosine_z=1.0,
            knn_distance=2.0,
            in_distribution=True,
        ),
    )
    app.state.clf = mock_clf

    fake_extraction = ExtractionMetadata(
        text="hola mundo", extractor_used="OCRExtractor", char_count=10
    )
    with patch(
        "src.api.routes.predict.endpoints.extract_pdf_with_metadata", return_value=fake_extraction
    ):
        client = TestClient(app)
        response = client.post(
            "/predict",
            files={"file": ("doc.pdf", b"%PDF-1.4 fake content", "application/pdf")},
        )

    assert response.status_code == HTTPStatus.OK
    assert response.json()["oodMetrics"]["mahalanobisPValueTheoretical"] == expected_theoretical_p


def test_predict_endpoint_rejects_upload_exceeding_max_size() -> None:
    app = create_app(model_path="fake/path")
    app.state.clf = MagicMock()  # satisfy dependency; classifier is irrelevant to this assertion
    with patch("src.api.routes.predict.endpoints.Settings.MAX_UPLOAD_SIZE_BYTES", 10):
        client = TestClient(app)
        response = client.post(
            "/predict",
            files={"file": ("doc.pdf", b"%PDF-1.4 fake content", "application/pdf")},
        )
    assert response.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
