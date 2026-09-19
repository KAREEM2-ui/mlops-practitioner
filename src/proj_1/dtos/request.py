from fastapi import UploadFile
from fastapi.params import File
from pydantic import BaseModel


class PredictionImageRequest(BaseModel):
    """Request DTO containing an image file."""

    image: UploadFile = File(..., description="Image file to be sent for inference")
