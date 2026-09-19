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

## Continuous Integration

[The CI workflow](.github/workflows/ci.yml) runs on pushes to every branch, including
branches with `/` in their names. It uses Python 3.11 and locked uv dependencies to:

1. Check lint and formatting for the application code in `src/` with Ruff.
2. Run the existing pytest suite and require at least 80% total coverage of
   `src/proj_1`. The XML coverage report is saved as a workflow artifact for 14 days.
3. Build the existing `Dockerfile` after all quality checks pass and push the
   image to `kareemooo/mlopsprac` on Docker Hub.

Add `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` under the GitHub repository's
Actions secrets. Use a Docker Hub access token for `DOCKERHUB_TOKEN`, not the
account password. Each branch push publishes one image tagged with the branch
name. A semantic Git tag publishes one exact version tag: pushing `v0.1.0`
publishes only `kareemooo/mlopsprac:0.1.0`.

```bash
git tag v0.1.0
git push origin v0.1.0
```

Tests use generated images and mocked inference, so CI does not download the
dataset, train a model, or require model weights.

Run the same quality checks locally:

```bash
uv sync --locked --group dev
uv run --no-sync ruff check src/
uv run --no-sync ruff format --check src/
uv run --no-sync pytest --cov=proj_1 --cov-report=term-missing --cov-report=xml:coverage.xml --cov-fail-under=80
```

To fix formatting locally, run `uv run --no-sync ruff format src/`, then commit
the formatted files. CI checks formatting without changing files.

The setup follows the official [uv GitHub Actions integration](https://docs.astral.sh/uv/guides/integration/github/)
and uses Docker's [build action](https://github.com/docker/build-push-action).

## Download the Dataset and Train the Model

Run these commands from the repository root on your local machine, with Python 3.11 and uv installed.

### 1. Install training dependencies

```bash
uv sync --group train
```


### 2. Download and extract the dataset

```bash
uv run --group train python dataset_install_script.py
```

The script downloads `farzadnekouei/pothole-image-segmentation-dataset` from Kaggle and extracts it into `dataset/`. Before training, check that this file exists:

```text
dataset/Pothole_Segmentation_YOLOv8/data.yaml
```

### 3. Train the model

```bash
uv run --group train python train.py
```
