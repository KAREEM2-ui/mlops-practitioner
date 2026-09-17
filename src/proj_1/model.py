from abc import ABC, abstractmethod
from typing import Any
import onnxruntime as ort
from proj_1.pipeline import ImagePreprocessingPipeline


class IModel(ABC):
    """Abstract Base Class defining the model interface."""

    @abstractmethod
    def predict(self, image_data: bytes) -> dict[str, Any]:
        """Accepts raw image bytes and returns inference prediction results."""
        pass


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

    def predict(self, image_data: bytes) -> dict[str, Any]:
        """Runs inference on raw image bytes using the ONNX session."""
        if not image_data:
            raise ValueError("Input image_data cannot be empty.")

        # Discover model input details
        input_meta = self._model.get_inputs()[0]
        input_name = input_meta.name
        input_shape = input_meta.shape

        # Preprocess bytes to match model input shape
        original_size, input_tensor = self.pipeline.preprocess(image_data, target_shape=input_shape)

        # Run ONNX inference
        outputs = self._model.run(["detection","prototype"], {input_name: input_tensor})
        

        # Postprocess predictions
        return self.pipeline.postprocess(outputs, original_size=original_size)  # Assuming (batch, channels, height, width)
