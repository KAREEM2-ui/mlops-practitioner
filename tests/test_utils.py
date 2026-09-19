import pytest
import yaml

from proj_1.utils import load_config

# mark all tests as unit tests
pytestmark = pytest.mark.unit


def test_load_config_reads_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "server:\n  port: 8000\nmodel:\n  path: model.onnx\n", encoding="utf-8"
    )
    assert load_config(path) == {
        "server": {"port": 8000},
        "model": {"path": "model.onnx"},
    }


def test_load_config_missing_file(tmp_path):
    assert load_config(tmp_path / "missing.yaml") == {}


def test_load_config_empty_file(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config(path) == {}


def test_load_config_invalid_yaml(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("model: [", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        load_config(path)
