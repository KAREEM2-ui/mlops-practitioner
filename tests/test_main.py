from importlib import import_module
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

main = import_module("proj_1.main")


pytestmark = pytest.mark.unit


@pytest.fixture
def endpoint(monkeypatch):
    model = Mock()
    model.predict.return_value = []
    monkeypatch.setattr(main, "_model_instance", model)
    return TestClient(main.app), model


@pytest.mark.parametrize("data", [b"", b"not an image"])
def test_endpoint_rejects_invalid_upload_before_inference(endpoint, data):
    client, model = endpoint
    response = client.post(
        "/predict/image", files={"image": ("fake.png", data, "image/png")}
    )
    assert response.status_code == 400
    model.predict.assert_not_called()


def test_endpoint_rejects_truncated_image(endpoint, image_bytes):
    client, model = endpoint
    data = image_bytes()[:50]
    response = client.post(
        "/predict/image", files={"image": ("broken.png", data, "image/png")}
    )
    assert response.status_code == 400
    model.predict.assert_not_called()


def test_endpoint_passes_decoded_image_to_model(endpoint, image_bytes):
    client, model = endpoint

    def predict(image):
        assert isinstance(image, Image.Image)
        assert image.size == (80, 40)
        assert image.getpixel((0, 0)) == (255, 128, 0)
        return []

    model.predict.side_effect = predict
    response = client.post(
        "/predict/image", files={"image": ("image.png", image_bytes(), "image/png")}
    )
    assert response.status_code == 200
    model.predict.assert_called_once()
