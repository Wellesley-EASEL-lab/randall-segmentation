from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from randall_pipeline.inference import resolve_adapter_files, validate_adapter_config
from randall_pipeline.io import safe_join
from scripts.audit_public_release import audit_files


ROOT = Path(__file__).resolve().parents[1]


def test_readme_script_modules_exist():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    modules = set(re.findall(r"python -m (scripts\.[a-z_]+)", readme))
    assert modules
    for module in modules:
        assert importlib.util.find_spec(module) is not None, module


def test_local_model_files_take_precedence(tmp_path, monkeypatch):
    files = [tmp_path / name for name in ("model.pth", "config.json", "id2label.json")]
    for path in files:
        path.write_text("fixture", encoding="utf-8")

    def unexpected_download(*args, **kwargs):
        raise AssertionError("local overrides should prevent Hub downloads")

    monkeypatch.setattr("randall_pipeline.inference.hf_hub_download", unexpected_download)
    assert resolve_adapter_files(
        adapter_checkpoint=files[0], config_path=files[1], id2label_path=files[2]
    ) == tuple(files)


def test_hosted_adapter_contract_is_strict():
    config = {
        "model_type": "mapsam-dora-sam-vit-l", "standalone": False,
        "num_classes": 8,
        "adapter": {"rank": 8, "spgen_variant": "depthwise"},
        "preprocessing": {"image_size": 1024},
    }
    validate_adapter_config(config)
    config["adapter"]["rank"] = 4
    with pytest.raises(ValueError, match="rank"):
        validate_adapter_config(config)


def test_root_containment_rejects_traversal(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        safe_join(tmp_path, "../outside.png")


def test_repository_passes_local_release_audit():
    assert audit_files() == []
