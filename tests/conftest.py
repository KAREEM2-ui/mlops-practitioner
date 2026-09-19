import io

import pytest
from PIL import Image


@pytest.fixture
def image_bytes():
    """Create small PNGs in memory without downloading a dataset."""

    def make_image(mode="RGB", size=(80, 40), color=(255, 128, 0)):
        buffer = io.BytesIO()
        Image.new(mode, size, color).save(buffer, format="PNG")
        return buffer.getvalue()

    return make_image


@pytest.fixture
def get_original_size():
    return (80, 40)
