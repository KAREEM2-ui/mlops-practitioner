FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy configuration and project definitions
COPY pyproject.toml uv.lock* README.md ./

# Install project dependencies
RUN uv sync --frozen --no-install-project

# Copy application source and configs
COPY src/ ./src/
COPY configs/ ./configs/

# Install the application
RUN uv sync --frozen

EXPOSE 8000

# Run FastAPI app with Uvicorn
CMD ["uv", "run", "uvicorn", "proj_1.main:app", "--host", "0.0.0.0", "--port", "8000"]
