from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from proj_1.dtos.response import PredictionResponse
from proj_1.model import IModel, ONNXModel
from proj_1.utils import load_config

# Global model instance holder
_model_instance: IModel | None = None


def get_model() -> IModel:
    """Returns the loaded model instance or raises an HTTP 503 if not ready."""
    if _model_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not initialized or model file not found.",
        )
    return _model_instance


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context for model loading and resource management."""
    global _model_instance
    config = load_config()
    model_cfg = config.get("model", {})
    raw_path = model_cfg.get("path", "model.onnx") 

    candidates = [
        Path(raw_path),
        Path("src/proj_1") / Path(raw_path).name,
        Path(__file__).parent / Path(raw_path).name,
        Path("model.onnx"),
    ]
    resolved_path = next((p for p in candidates if p.is_file()), None)

    if resolved_path and resolved_path.exists():
        try:
            _model_instance = ONNXModel(model_path=str(resolved_path))
            print(f"Info: Loaded ONNX model from '{resolved_path}'")
        except Exception as e:
            print(f"Warning: Failed to load ONNX model from '{resolved_path}': {e}")
            _model_instance = None
    else:
        print(f"Warning: Model file '{raw_path}' not found at startup. Waiting for model to be provided.")
        _model_instance = None

    yield
    _model_instance = None


app = FastAPI(
    title="MLOps ONNX Model Inference API",
    description="FastAPI service for image inference using ONNX Runtime.",
    version="0.1.0",
    lifespan=lifespan,
)



@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Health check endpoint."""

    return {"status": "healthy", "model_loaded": str(_model_instance is not None)}


@app.post(
    "/predict/image",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
async def predict_file(
    file: UploadFile | None = File(None),
    image: UploadFile | None = File(None),
) -> PredictionResponse:
    """Inference endpoint accepting multipart/form-data image file upload."""
    uploaded = file or image
    if uploaded is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image uploaded. Please supply a multipart field named 'file' or 'image'.",
        )

    model = get_model()
    try:
        image_bytes = await uploaded.read()
        if not image_bytes:
            raise ValueError("Uploaded file is empty.")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {e}",
        )

    try:
        result = model.predict(image_bytes)
        predictions = result if isinstance(result, list) else result.get("prediction", [])
        return PredictionResponse(
            status="success",
            prediction=predictions,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference failed: {e}",
        )


def main() -> None:
    """Entry point for running the API with Uvicorn."""
    import uvicorn

    config = load_config()
    server_config = config.get("server", {})
    host = server_config.get("host", "0.0.0.0")
    port = server_config.get("port", 8000)
    uvicorn.run("proj_1.main:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    main()

