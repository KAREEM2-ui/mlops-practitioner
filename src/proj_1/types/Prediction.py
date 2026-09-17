
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Represents a bounding box in an image."""
    x_min: float = Field(default=0.0, description="Minimum x-coordinate of the bounding box")
    y_min: float = Field(default=0.0, description="Minimum y-coordinate of the bounding box")
    x_max: float = Field(default=0.0, description="Maximum x-coordinate of the bounding box")
    y_max: float = Field(default=0.0, description="Maximum y-coordinate of the bounding box")


class MaskPrediction(BaseModel):
    """Represents a mask prediction for an object in an image."""
    mask: list[list[float]] = Field(default_factory=list, description="2D array representing the predicted mask")


class ObjectPrediction(BaseModel):
    class_id: int = Field(default=0, description="Predicted class ID")
    confidence: float = Field(default=0.0, description="Confidence score of the prediction")
    box: BoundingBox = Field(default_factory=BoundingBox, description="Optional bounding box for object detection tasks")
    mask: MaskPrediction | None = Field(default=None, description="Segmentation mask prediction")
