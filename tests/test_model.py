from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from proj_1.model import ONNXModel
from proj_1.pipeline import ImagePreprocessingPipeline

# mark all tests as unit tests
pytestmark = pytest.mark.unit


@pytest.fixture
def model_setup(monkeypatch):
    session = Mock()
    session.get_inputs.return_value = [
        SimpleNamespace(name="image_input", shape=[1, 3, 20, 30])
    ]

    # replace InferenceSession (runtime of onnx) with a mock
    monkeypatch.setattr("proj_1.model.ort.InferenceSession", Mock(return_value=session))

    pipeline = Mock(spec=ImagePreprocessingPipeline)
    return ONNXModel("unused.onnx", pipeline=pipeline), session, pipeline


def test_predict_passes_data_through_pipeline(model_setup):
    model, session, pipeline = model_setup
    data = Image.new("RGB", (80, 40))
    tensor = np.zeros((1, 3, 20, 30), dtype=np.float32)
    pipeline.preprocess.return_value = ((80, 40), tensor)
    expected = [object()]
    pipeline.postprocess.return_value = expected

    assert model.predict(data) is expected

    pipeline.preprocess.assert_called_once_with(data, target_shape=[1, 3, 20, 30])
    session.run.assert_called_once()
    names, inputs = session.run.call_args.args
    assert names == ["detection", "prototype"]
    assert list(inputs) == ["image_input"]
    assert inputs["image_input"] is tensor
    pipeline.postprocess.assert_called_once_with(
        session.run.return_value, original_size=(80, 40)
    )


def test_inference_failure_propagates(model_setup):
    model, session, pipeline = model_setup
    pipeline.preprocess.return_value = ((80, 40), np.zeros((1, 3, 20, 30)))
    session.run.side_effect = RuntimeError("inference failed")
    with pytest.raises(RuntimeError, match="inference failed"):
        model.predict(Image.new("RGB", (80, 40)))
    pipeline.postprocess.assert_not_called()
