from http import HTTPStatus
from typing import Any
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.routes.predict.schemas import PredictResponse
from src.schema import ExtractionMetadata, OodMetrics, PredictResult


def _predict_and_await_result(client: TestClient, files: dict[str, tuple[str, bytes, str]]) -> Any:
    """POST /predict now returns a job id immediately (see PredictJob) -- the actual
    prediction runs in a BackgroundTask, which TestClient drives to completion before
    the POST call returns, so a single status poll right after is always already terminal."""
    created = client.post("/predict", files=files)
    assert created.status_code == HTTPStatus.OK
    job_id = created.json()["jobId"]
    job = client.get(f"/predict/status/{job_id}").json()
    assert job["stage"] == "done", job
    return job["result"]


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


def test_predict_response_has_svm_scores_field() -> None:
    response = PredictResponse(filename="doc.pdf", label="decreto", confidence=0.9, certain=True)
    assert response.svm_scores == {}
    assert response.svm_predicted_label == ""


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
        result = _predict_and_await_result(
            client, files={"file": ("doc.pdf", b"%PDF-1.4 fake content", "application/pdf")}
        )

    assert result["extractedText"] == "hola mundo"
    assert result["extractorUsed"] == "OCRExtractor"


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
        result = _predict_and_await_result(
            client, files={"file": ("doc.pdf", b"%PDF-1.4 fake content", "application/pdf")}
        )

    assert result["oodMetrics"]["knnDistance"] == expected_knn_distance


def test_predict_endpoint_returns_svm_scores() -> None:
    app = create_app(model_path="fake/path")
    mock_clf = MagicMock()
    mock_clf.predict_text.return_value = PredictResult(
        label="decreto",
        confidence=0.9,
        certain=True,
        svm_scores={"decreto": 1.2, "ordenanza": -0.5},
    )
    app.state.clf = mock_clf

    fake_extraction = ExtractionMetadata(
        text="hola mundo", extractor_used="OCRExtractor", char_count=10
    )
    with patch(
        "src.api.routes.predict.endpoints.extract_pdf_with_metadata", return_value=fake_extraction
    ):
        client = TestClient(app)
        result = _predict_and_await_result(
            client, files={"file": ("doc.pdf", b"%PDF-1.4 fake content", "application/pdf")}
        )

    assert result["svmScores"] == {"decreto": 1.2, "ordenanza": -0.5}


def test_predict_endpoint_returns_svm_disagreement_fields() -> None:
    app = create_app(model_path="fake/path")
    mock_clf = MagicMock()
    mock_clf.predict_text.return_value = PredictResult(
        label="decreto",
        confidence=0.9,
        certain=True,
        svm_scores={"decreto": -0.5, "ordenanza": 1.2},
        svm_predicted_label="ordenanza",
        svm_agrees_with_prediction=False,
    )
    app.state.clf = mock_clf

    fake_extraction = ExtractionMetadata(
        text="hola mundo", extractor_used="OCRExtractor", char_count=10
    )
    with patch(
        "src.api.routes.predict.endpoints.extract_pdf_with_metadata", return_value=fake_extraction
    ):
        client = TestClient(app)
        result = _predict_and_await_result(
            client, files={"file": ("doc.pdf", b"%PDF-1.4 fake content", "application/pdf")}
        )

    assert result["svmPredictedLabel"] == "ordenanza"
    assert result["svmAgreesWithPrediction"] is False


def test_predict_response_svm_agrees_with_prediction_defaults_to_true() -> None:
    response = PredictResponse(filename="doc.pdf", label="decreto", confidence=0.9, certain=True)
    assert response.svm_agrees_with_prediction is True


def test_predict_endpoint_returns_foreign_municipality() -> None:
    app = create_app(model_path="fake/path")
    mock_clf = MagicMock()
    mock_clf.predict_text.return_value = PredictResult(
        label="decreto", confidence=0.9, certain=True
    )
    app.state.clf = mock_clf

    fake_extraction = ExtractionMetadata(
        text="Municipalidad de Cordoba informa", extractor_used="OCRExtractor", char_count=30
    )
    with patch(
        "src.api.routes.predict.endpoints.extract_pdf_with_metadata", return_value=fake_extraction
    ):
        client = TestClient(app)
        result = _predict_and_await_result(
            client, files={"file": ("doc.pdf", b"%PDF-1.4 fake content", "application/pdf")}
        )

    assert result["foreignMunicipality"] == "Cordoba"
    assert "Municipalidad de Cordoba" in result["foreignMunicipalityContext"]


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
        result = _predict_and_await_result(
            client, files={"file": ("doc.pdf", b"%PDF-1.4 fake content", "application/pdf")}
        )

    assert result["reviewRoute"] == "accept"


def test_predict_endpoint_routes_unreadable_document_to_human_review() -> None:
    app = create_app(model_path="fake/path")
    app.state.clf = MagicMock()
    fake_extraction = ExtractionMetadata(text=None, extractor_used=None, char_count=0)
    with patch(
        "src.api.routes.predict.endpoints.extract_pdf_with_metadata", return_value=fake_extraction
    ):
        client = TestClient(app)
        result = _predict_and_await_result(
            client, files={"file": ("doc.pdf", b"%PDF-1.4 fake content", "application/pdf")}
        )

    assert result["reviewRoute"] == "human_review"


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
        result = _predict_and_await_result(
            client, files={"file": ("doc.pdf", b"%PDF-1.4 fake content", "application/pdf")}
        )

    assert result["oodMetrics"]["mahalanobisPValueTheoretical"] == expected_theoretical_p


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


def test_predict_status_returns_404_for_unknown_job_id() -> None:
    app = create_app(model_path="fake/path")
    client = TestClient(app)
    response = client.get("/predict/status/does-not-exist")
    assert response.status_code == HTTPStatus.NOT_FOUND


def test_predict_endpoint_job_lands_in_error_stage_on_exception() -> None:
    app = create_app(model_path="fake/path")
    mock_clf = MagicMock()
    mock_clf.predict_text.side_effect = RuntimeError("boom")
    app.state.clf = mock_clf

    fake_extraction = ExtractionMetadata(
        text="hola mundo", extractor_used="OCRExtractor", char_count=10
    )
    with patch(
        "src.api.routes.predict.endpoints.extract_pdf_with_metadata", return_value=fake_extraction
    ):
        client = TestClient(app)
        created = client.post(
            "/predict",
            files={"file": ("doc.pdf", b"%PDF-1.4 fake content", "application/pdf")},
        )
        assert created.status_code == HTTPStatus.OK
        job_id = created.json()["jobId"]
        job = client.get(f"/predict/status/{job_id}").json()

    assert job["stage"] == "error"
    assert "boom" in job["error"]
    assert job["result"] is None
