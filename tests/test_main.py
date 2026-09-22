import pytest
from fastapi.testclient import TestClient

from proj_1 import app
from proj_1.dtos import PredictionResponse

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def endpoint():
    # Entering the context runs startup and loads the real ONNX model.
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] == "True", (
            "Real integration tests require a loadable ONNX model and its weights. "
            "Check the model path in configs/config.yaml and the startup output."
        )
        yield client


@pytest.mark.parametrize("data", [b"", b"not an image"])
def test_endpoint_rejects_invalid_upload(endpoint, data):
    response = endpoint.post(
        "/predict/image", files={"image": ("broken.png", data, "image/png")}
    )
    assert response.status_code == 400
    expected_detail = (
        "Uploaded image file is empty."
        if not data
        else "Uploaded file is not a valid image."
    )
    assert response.json()["detail"] == expected_detail


def test_endpoint_rejects_truncated_image(endpoint, image_bytes):
    response = endpoint.post(
        "/predict/image",
        files={"image": ("broken.png", image_bytes()[:50], "image/png")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is not a valid image."


def test_predict_dtos(endpoint, image_bytes):
    response = endpoint.post(
        "/predict/image",
        files={"image": ("image.png", image_bytes(), "image/png")},
    )

    assert response.status_code == 200, response.text
    prediction = PredictionResponse.model_validate(response.json())
    assert prediction.status == "success"
