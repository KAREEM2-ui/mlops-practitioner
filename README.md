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

## Quick Start

### 1. Pull the Docker image

```bash
docker pull kareemooo/mlopsprac:0.1.0
```

### 2. Run it

```bash
docker run --rm -p 8000:8000 kareemooo/mlopsprac:0.1.0
```

### 3. Make a prediction

```bash
curl -X POST http://localhost:8000/predict -F "file=@pothole.jpg"

replace the file with specfic image path locally
```

