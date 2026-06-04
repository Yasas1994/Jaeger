"""Tests for pyproject.toml packaging configuration."""

import json
import zipfile
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestPyprojectConfig:
    """Test pyproject.toml configuration."""

    def test_pyproject_exists(self):
        """pyproject.toml should exist."""
        pyproject = PROJECT_ROOT / "pyproject.toml"
        assert pyproject.exists(), "pyproject.toml not found"

    def test_pyproject_has_version(self):
        """pyproject.toml should contain a version."""
        pyproject = PROJECT_ROOT / "pyproject.toml"
        content = pyproject.read_text()
        assert 'version = "' in content, "No version found in pyproject.toml"

    def test_pyproject_uses_pdm_backend(self):
        """pyproject.toml should use pdm-backend."""
        pyproject = PROJECT_ROOT / "pyproject.toml"
        content = pyproject.read_text()
        assert "pdm-backend" in content, "pdm-backend not in build-system"
        assert "pdm.backend" in content, "pdm.backend not as build-backend"

    def test_pdm_build_includes_data_files(self):
        """[tool.pdm.build] should include data file patterns."""
        pyproject = PROJECT_ROOT / "pyproject.toml"
        content = pyproject.read_text()
        assert "[tool.pdm.build]" in content, "[tool.pdm.build] section missing"
        assert "includes" in content, "includes key missing in [tool.pdm.build]"
        assert "*.fasta" in content, ".fasta files not included"
        assert "*.h5" in content, ".h5 files not included"
        assert "*.json" in content, ".json files not included"

    def test_no_setuptools_package_data(self):
        """Old broken [tool.setuptools.package-data] should not exist."""
        pyproject = PROJECT_ROOT / "pyproject.toml"
        content = pyproject.read_text()
        assert "[tool.setuptools.package-data]" not in content, \
            "Old setuptools package-data config still present"


class TestWheelContents:
    """Test that built wheel contains required data files."""

    def test_wheel_contains_test_fastas(self):
        """Wheel should contain test FASTA files."""
        dist_dir = PROJECT_ROOT / "dist"
        if not dist_dir.exists():
            pytest.skip("No dist/ directory found; run 'python -m build' first")

        wheels = list(dist_dir.glob("*.whl"))
        if not wheels:
            pytest.skip("No wheel found in dist/")

        wheel = wheels[0]
        with zipfile.ZipFile(wheel) as zf:
            names = zf.namelist()
            assert any("test_short.fasta" in n for n in names), \
                "test_short.fasta not in wheel"
            assert any("test_empty.fasta" in n for n in names), \
                "test_empty.fasta not in wheel"
            assert any("test_contigs.fasta" in n for n in names), \
                "test_contigs.fasta not in wheel"

    def test_wheel_contains_config_json(self):
        """Wheel should contain config.json."""
        dist_dir = PROJECT_ROOT / "dist"
        if not dist_dir.exists():
            pytest.skip("No dist/ directory found")

        wheels = list(dist_dir.glob("*.whl"))
        if not wheels:
            pytest.skip("No wheel found in dist/")

        wheel = wheels[0]
        with zipfile.ZipFile(wheel) as zf:
            names = zf.namelist()
            assert any("config.json" in n for n in names), \
                "config.json not in wheel"

    def test_wheel_contains_model_weights(self):
        """Wheel should contain default model weights."""
        dist_dir = PROJECT_ROOT / "dist"
        if not dist_dir.exists():
            pytest.skip("No dist/ directory found")

        wheels = list(dist_dir.glob("*.whl"))
        if not wheels:
            pytest.skip("No wheel found in dist/")

        wheel = wheels[0]
        with zipfile.ZipFile(wheel) as zf:
            names = zf.namelist()
            assert any(".h5" in n for n in names), \
                "No .h5 weights files in wheel"
