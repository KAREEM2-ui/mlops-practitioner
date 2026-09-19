from abc import ABC, abstractmethod
from typing import Any

import onnxruntime as ort
from PIL import Image

from proj_1.pipeline import ImagePreprocessingPipeline


class IModel(ABC):
    """Abstract Base Class defining the model interface."""

    @abstractmethod
    def predict(self, image_data: Image.Image) -> dict[str, Any]:
        """Accepts a decoded PIL image and returns inference prediction results."""


class ONNXModel(IModel):
    """Concrete implementation of IModel using ONNX Runtime for inference."""

    def __init__(
        self,
        model_path: str = "model.onnx",
        pipeline: ImagePreprocessingPipeline | None = None,
        providers: list[str] | None = None,
    ) -> None:
        self.model_path = model_path
        self.pipeline = pipeline or ImagePreprocessingPipeline()

        # Initialize ONNX inference session
        self._model = ort.InferenceSession(
            self.model_path,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.model = self._model

    def predict(self, image_data: Image.Image) -> dict[str, Any]:
        """Runs inference on a decoded PIL image using the ONNX session."""

        # Discover model input details
        input_meta = self._model.get_inputs()[0]
        input_name = input_meta.name
        input_shape = input_meta.shape

        # Preprocess the image to match model input shape
        original_size, input_tensor = self.pipeline.preprocess(
            image_data, target_shape=input_shape
        )

        # Run ONNX inference
        outputs = self._model.run(
            ["detection", "prototype"], {input_name: input_tensor}
        )

        # Postprocess predictions
        return self.pipeline.postprocess(
            outputs, original_size=original_size
        )  # Assuming (batch, channels, height, width)
