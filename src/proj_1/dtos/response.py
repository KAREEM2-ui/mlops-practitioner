from pydantic import BaseModel, Field

from proj_1.types.Prediction import ObjectPrediction


class PredictionResponse(BaseModel):
    """Response DTO for model inference results."""

    status: str = Field(
        default="success", description="Status of the inference request"
    )
    prediction: list[ObjectPrediction] = Field(..., description="Inference output ")
