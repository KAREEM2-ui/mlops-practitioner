from pathlib import Path
from typing import Any
import yaml


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Loads a YAML configuration file."""
    path = Path(config_path)
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def validate_image_bytes(image_bytes: bytes) -> bool:
    """Basic validation for non-empty image bytes."""
    return bool(image_bytes and len(image_bytes) > 0)
