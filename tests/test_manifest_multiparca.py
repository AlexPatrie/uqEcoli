"""Tests for manifest multi-parca info recording (PR2 §0.3)."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from uq.workflow import _write_manifest


def _make_mock_cache(tmp_path: Path, with_parca_variants: bool = False, with_conditions: bool = False):
    """Create a minimal mock cache for manifest testing."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    np.save(cache_dir / "X.npy", np.random.randn(10, 3))
    np.save(cache_dir / "Y.npy", np.random.randn(10, 5))

    if with_parca_variants:
        batch_dir = cache_dir / "_batch"
        batch_dir.mkdir()
        wf_config = {
            "sim_data_path": "/tmp/simData.cPickle",
            "parca_variants": [
                {"rnaseq_basal_dataset_id": "glucose_minimal"},
                {"rnaseq_basal_dataset_id": "glucose_rich"},
            ],
        }
        (batch_dir / "workflow_config.json").write_text(json.dumps(wf_config))

    if with_conditions:
        cond_meta = {"conditions": ["glucose_minimal", "glucose_rich"], "n_samples": 10}
        (cache_dir / "conditions.json").write_text(json.dumps(cond_meta))

    mock = MagicMock()
    mock.cache_dir = cache_dir
    mock.X = np.random.randn(10, 3)
    mock.Y = np.random.randn(10, 5)
    mock.parameter_names = ["param_a", "param_b", "param_c"]
    return mock


class TestManifestMultiParca:
    """Test that manifest.json records multi-parca info when present."""

    def test_manifest_without_parca_variants(self, tmp_path):
        """Standard single-parca: no parca_variants in manifest."""
        cache = _make_mock_cache(tmp_path)
        export_dir = tmp_path / "export"
        export_dir.mkdir()

        _write_manifest(export_dir, cache)

        manifest = json.loads((export_dir / "manifest.json").read_text())
        assert "parca_variants" not in manifest
        assert "conditions" not in manifest
        assert manifest["n_samples"] == 10
        assert manifest["n_parameters"] == 3

    def test_manifest_with_parca_variants(self, tmp_path):
        """Multi-parca config → parca_variants recorded in manifest."""
        cache = _make_mock_cache(tmp_path, with_parca_variants=True)
        export_dir = tmp_path / "export"
        export_dir.mkdir()

        _write_manifest(export_dir, cache)

        manifest = json.loads((export_dir / "manifest.json").read_text())
        assert "parca_variants" in manifest
        assert len(manifest["parca_variants"]) == 2
        assert manifest["parca_variants"][0]["rnaseq_basal_dataset_id"] == "glucose_minimal"

    def test_manifest_with_conditions(self, tmp_path):
        """Multi-condition cache → conditions metadata recorded in manifest."""
        cache = _make_mock_cache(tmp_path, with_conditions=True)
        export_dir = tmp_path / "export"
        export_dir.mkdir()

        _write_manifest(export_dir, cache)

        manifest = json.loads((export_dir / "manifest.json").read_text())
        assert "conditions" in manifest
        assert manifest["conditions"]["conditions"] == ["glucose_minimal", "glucose_rich"]

    def test_manifest_has_standard_fields(self, tmp_path):
        """All standard manifest fields are present regardless of multi-parca."""
        cache = _make_mock_cache(tmp_path)
        export_dir = tmp_path / "export"
        export_dir.mkdir()

        _write_manifest(export_dir, cache)

        manifest = json.loads((export_dir / "manifest.json").read_text())
        assert "timestamp" in manifest
        assert "hostname" in manifest
        assert "python_version" in manifest
        assert "package_versions" in manifest
        assert "git_sha" in manifest
        assert "data_hashes" in manifest
        assert "parameter_names" in manifest

    def test_manifest_data_hashes_valid(self, tmp_path):
        """Data hashes are valid hex strings (not 'missing')."""
        cache = _make_mock_cache(tmp_path)
        export_dir = tmp_path / "export"
        export_dir.mkdir()

        _write_manifest(export_dir, cache)

        manifest = json.loads((export_dir / "manifest.json").read_text())
        for key in ["X.npy", "Y.npy"]:
            h = manifest["data_hashes"][key]
            assert h != "missing"
            assert len(h) == 64  # SHA-256 hex length
