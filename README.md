# MLOps Image Inference Service

An end-to-end MLOps project skeleton providing an ONNX model inference pipeline and FastAPI web service.

## Project Structure

```
my_ml_project/
├── configs/
│   └── config.yaml          # YAML configuration
├── notebooks/
│   └── exploration.ipynb    # Jupyter notebook for exploration
├── src/
│   └── proj_1/
│       ├── __init__.py
│       ├── dtos/            # Pydantic request & response schemas
│       │   ├── __init__.py
│       │   ├── request.py
│       │   └── response.py
│       ├── model.py         # IModel abstract base class & ONNXModel
│       ├── pipeline.py      # Image preprocessing & postprocessing
│       ├── utils.py         # Utilities and config loader
│       └── main.py          # FastAPI application & entrypoint
├── tests/
│   └── test_model.py        # Pytest test suite
├── Dockerfile               # Container definition
├── pyproject.toml           # Project dependencies & packaging
└── README.md
```

## Quick Start with `uv`

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Run tests:**
   ```bash
   uv run pytest
   ```

3. **Start the FastAPI server:**
   ```bash
   uv run uvicorn proj_1.main:app --reload --port 8000
   ```

4. **API Endpoints:**
   - `GET /health` - Service health status
   - `POST /predict` - Base64 image inference
   - `POST /predict/file` - Multipart image file upload inference
