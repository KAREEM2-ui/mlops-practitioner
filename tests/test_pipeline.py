import numpy as np
import pytest
from PIL import UnidentifiedImageError

from proj_1.pipeline import ImagePreprocessingPipeline as Pipeline
import pytest

# mark all tests as unit tests
pytestmark = pytest.mark.unit



def make_outputs(detections):
    """Each row contains cx, cy, width, height, confidence; masks have zero logits."""
    output = np.zeros((1, 37, len(detections)), dtype=np.float32)
    for index, detection in enumerate(detections):
        output[0, :5, index] = detection
    return [output, np.zeros((1, 32, 16, 16), dtype=np.float32)]


def test_preprocess_resizes_normalizes_and_preserves_original_size(image_bytes,get_original_size):
    
    original_size, tensor = Pipeline.preprocess(image_bytes(), (1, 3, 20, 30))
    assert original_size == get_original_size
    assert tensor.shape == (1, 3, 20, 30)
    assert tensor.dtype == np.float32
    np.testing.assert_allclose(tensor[0, 0], 1.0) # red 
    np.testing.assert_allclose(tensor[0, 1], 128 / 255) # green
    np.testing.assert_allclose(tensor[0, 2], 0.0) # blue






@pytest.mark.parametrize("data", [b"", b"not an image"])
def test_preprocess_rejects_invalid_image(data):
    with pytest.raises(UnidentifiedImageError):
        Pipeline.preprocess(data, (1, 3, 20, 30))


@pytest.mark.parametrize("shape", [None, (), (3, 20, 30), (1, 3, 20, 30, 1)])
def test_preprocess_rejects_invalid_shape(image_bytes, shape):
    with pytest.raises(ValueError, match="4D shape"):
        Pipeline.preprocess(image_bytes(), shape)


@pytest.mark.parametrize("detections", [[], [[320, 320, 100, 100, 0.1]]])
def test_postprocess_returns_empty_without_qualifying_detections(detections,get_original_size):
    assert Pipeline.postprocess(make_outputs(detections), get_original_size) == []



def test_single_detection_scales_box_and_crops_mask(get_original_size):
    
    original_size = get_original_size
    
    predictions = Pipeline.postprocess(make_outputs([[320, 320, 320, 320, 0.9]]), original_size)
    assert len(predictions) == 1
    prediction = predictions[0]
    assert prediction.confidence == pytest.approx(0.9)
    assert prediction.box.model_dump() == dict(x_min=original_size[0] // 4, y_min=original_size[1] // 4, x_max=original_size[0] * 3/4, y_max=original_size[1] * 3/4)
    mask = np.asarray(prediction.mask.mask)
    assert mask.shape == (40, 80)   
    assert np.isfinite(mask).all()
    assert 0 <= mask.min() <= mask.max() <= 1
    # Zero logits yield sigmoid 0.5, quantized to uint8 before resizing.
    assert mask[20, 40] == pytest.approx(127 / 255)
    assert not mask[:5].any()
    assert not mask[35:].any()
    assert not mask[:, :10].any()
    assert not mask[:, 70:].any()


def test_postprocess_clips_box_to_image_boundaries():
    result = Pipeline.postprocess(make_outputs([[320, 320, 800, 800, 0.9]]), (80, 40))
    assert result[0].box.model_dump() == dict(x_min=0, y_min=0, x_max=80, y_max=40)


def test_postprocess_filters_confidence_and_suppresses_overlapping_boxes():
    outputs = make_outputs([
        [160, 160, 80, 80, 0.6],
        [160, 160, 80, 80, 0.9],
        [480, 480, 80, 80, 0.8],
        [320, 320, 80, 80, 0.1],
    ])
    result = Pipeline.postprocess(outputs, (80, 40))
    assert [item.confidence for item in result] == pytest.approx([0.9, 0.8])
    assert [item.box.x_min for item in result] == [15, 55]


def test_confidence_threshold_includes_exact_boundary():
    outputs = make_outputs([[320, 320, 80, 80, 0.5]])
    assert len(Pipeline.postprocess(outputs, (80, 40), confidence_threshold=0.5)) == 1
    assert Pipeline.postprocess(outputs, (80, 40), confidence_threshold=0.51) == []


@pytest.mark.parametrize("other,expected", [
    ((0, 0, 10, 10), 1),
    ((20, 20, 30, 30), 0),
    ((5, 0, 15, 10), 1 / 3),
    ((0, 0, 0, 0), 0),
])
def test_iou(other, expected):
    assert Pipeline._compute_iou((0, 0, 10, 10), other) == pytest.approx(expected)
