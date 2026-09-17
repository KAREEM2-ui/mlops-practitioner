# ==========================================
# Stage 1: Build & Dependency Installation
# ==========================================
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

# Enable bytecode compilation and copy mode for uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy configuration and project definitions for layer caching
COPY pyproject.toml uv.lock* README.md ./

# Install only production dependencies (excluding dev/train groups)
RUN uv sync --frozen --no-install-project --no-dev --no-editable

# Copy application source and configs
COPY src/ ./src/
COPY configs/ ./configs/

# Install the application without development dependencies
RUN uv sync --frozen --no-dev --no-editable

# ==========================================
# Stage 2: Runtime Stage (Minimal Image)
# ==========================================
FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

# Ensure Python doesn't buffer stdout/stderr and configure PATH for virtualenv
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Copy only the compiled virtual environment (without uv or dev dependencies)
COPY --from=builder /app/.venv /app/.venv

# Copy source code and runtime configurations
COPY --from=builder /app/src ./src
COPY --from=builder /app/configs ./configs

# Expose the API port
EXPOSE 8000

# Run FastAPI app directly with Uvicorn from the virtual environment
CMD ["uvicorn", "proj_1.main:app", "--host", "0.0.0.0", "--port", "8000"]

