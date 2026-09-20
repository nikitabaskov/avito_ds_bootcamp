"""Контракт конфигурации, модели и контрольных сумм итоговой отправки."""

import json
from pathlib import Path

import pytest

from candgen.core.submission import sha256_file
from candgen.workflows.predict import expected_features, feature_spec, model_config, verify_file


def test_final_feature_contract():
    meta = json.loads(Path("configs/final.json").read_text())
    config = model_config(meta)
    assert expected_features(feature_spec(meta), config) == meta["features"]
    assert len(meta["features"]) == len(set(meta["features"])) == 52
    assert meta["ranker"]["random_seed"] == 42
    assert meta["ranker"]["iterations"] == 500


def test_verify_file_rejects_changed_and_missing_inputs(tmp_path):
    path = tmp_path / "input.parquet"
    path.write_bytes(b"original data")
    digest = sha256_file(path)
    verify_file(path, digest)
    path.write_bytes(b"changed data")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_file(path, digest)
    path.unlink()
    with pytest.raises(FileNotFoundError, match="Missing"):
        verify_file(path, digest)
