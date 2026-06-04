"""Tests for the jaeger health command."""

import sys
import platform
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Skip all tests if tensorflow is not available
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False


pytestmark = pytest.mark.skipif(not HAS_TF, reason="tensorflow not installed")


class TestHealthDiagnostics:
    """Test the diagnostic info gathering in health command."""

    def test_python_version_reported(self):
        """Check that Python version is accessible."""
        assert platform.python_version() is not None
        assert "." in platform.python_version()

    def test_python_executable_exists(self):
        """Check that Python executable path exists."""
        assert Path(sys.executable).exists()

    def test_platform_info(self):
        """Check that platform info is non-empty."""
        assert platform.platform()
        assert platform.machine()


class TestHealthDependencies:
    """Test that core dependencies are importable and have versions."""

    def test_jaeger_bio_importable(self):
        """jaeger-bio package should be importable."""
        import jaeger
        assert jaeger.__file__ is not None

    def test_tensorflow_importable(self):
        """tensorflow should be importable."""
        import tensorflow as tf
        assert tf.__version__ is not None

    def test_numpy_importable(self):
        """numpy should be importable."""
        import numpy as np
        assert np.__version__ is not None

    def test_click_importable(self):
        """click should be importable."""
        import click
        assert click.__version__ is not None

    def test_parasail_importable(self):
        """parasail should be importable."""
        import parasail
        assert parasail.__version__ is not None

    def test_pyfastx_importable(self):
        """pyfastx should be importable."""
        import pyfastx
        # pyfastx doesn't expose a simple version attribute; just check it loads
        assert pyfastx.Fasta is not None

    def test_pydustmasker_importable(self):
        """pydustmasker should be importable."""
        import pydustmasker
        assert pydustmasker.__version__ is not None

    def test_sklearn_importable(self):
        """scikit-learn should be importable."""
        import sklearn
        assert sklearn.__version__ is not None

    def test_polars_importable(self):
        """polars should be importable."""
        import polars
        assert polars.__version__ is not None

    def test_pandas_importable(self):
        """pandas should be importable."""
        import pandas
        assert pandas.__version__ is not None

    def test_matplotlib_importable(self):
        """matplotlib should be importable."""
        import matplotlib
        assert matplotlib.__version__ is not None

    def test_ruptures_importable(self):
        """ruptures should be importable."""
        import ruptures
        assert ruptures.__version__ is not None

    def test_pycirclize_importable(self):
        """pycirclize should be importable."""
        import pycirclize
        assert pycirclize.__version__ is not None

    def test_biopython_importable(self):
        """biopython should be importable."""
        import Bio
        assert Bio.__version__ is not None


class TestHealthTensorFlow:
    """Test TensorFlow-specific health checks."""

    def test_tf_can_list_devices(self):
        """TensorFlow should be able to list physical devices."""
        import tensorflow as tf
        cpus = tf.config.list_physical_devices("CPU")
        assert len(cpus) >= 1, "At least one CPU should be available"

    def test_tf_matrix_multiplication(self):
        """TensorFlow should be able to perform basic matrix ops."""
        import tensorflow as tf
        import numpy as np

        a = np.random.rand(10, 10).astype(np.float32)
        b = np.random.rand(10, 10).astype(np.float32)

        with tf.device("CPU:0"):
            result = tf.matmul(a, b)

        assert result.shape == (10, 10)

    def test_tf_version_parsable(self):
        """TensorFlow version should be a non-empty string."""
        import tensorflow as tf
        assert tf.__version__
        assert "." in tf.__version__


class TestHealthModels:
    """Test model discovery in health command."""

    def test_config_json_exists(self):
        """config.json should exist in jaeger.data."""
        from importlib.resources import files
        config_path = files("jaeger.data").joinpath("config.json")
        assert config_path.exists(), "config.json not found in jaeger.data"

    def test_config_json_loadable(self):
        """config.json should be valid JSON with expected keys."""
        import json
        from importlib.resources import files

        config_path = files("jaeger.data").joinpath("config.json")
        config = json.loads(config_path.read_text())
        assert "default" in config
        assert "model_paths" in config

    def test_default_weights_exist(self):
        """Default model weights should exist."""
        import json
        from importlib.resources import files

        config_path = files("jaeger.data").joinpath("config.json")
        config = json.loads(config_path.read_text())
        weights_name = config["default"]["weights"]
        weights_path = files("jaeger.data.models.default").joinpath(weights_name)
        assert weights_path.exists(), f"Default weights {weights_name} not found"

    def test_test_fasta_files_exist(self):
        """Test FASTA files should exist in jaeger.data.test."""
        from importlib.resources import files

        for fname in ["test_short.fasta", "test_empty.fasta", "test_contigs.fasta"]:
            fpath = files("jaeger.data.test").joinpath(fname)
            assert fpath.exists(), f"{fname} not found in jaeger.data.test"


class TestHealthCoreFunction:
    """Test the health_core function directly."""

    def test_health_core_runs_without_error(self, tmp_path, caplog):
        """health_core should run without raising exceptions."""
        import logging
        from jaeger.commands.health import health_core

        # Change to tmp_path so test_log is created there
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            # Run with minimal verbosity to reduce output
            health_core(verbose=0)
        finally:
            os.chdir(original_cwd)

    def test_health_core_creates_log(self, tmp_path):
        """health_core should create a log file."""
        import os
        from jaeger.commands.health import health_core

        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            health_core(verbose=0)
            log_dir = tmp_path / "test_log"
            assert log_dir.exists(), "test_log directory not created"
            log_files = list(log_dir.glob("*test_jaeger.log"))
            assert len(log_files) > 0, "No log file created"
        finally:
            os.chdir(original_cwd)

    def test_health_core_all_tests_pass(self, tmp_path):
        """health_core should report 5/5 tests passed."""
        import os
        import logging
        from jaeger.commands.health import health_core

        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            health_core(verbose=0)
            log_dir = tmp_path / "test_log"
            log_files = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
            assert log_files, "No log file found"
            log_content = log_files[-1].read_text()
            assert "5/5 tests passed!" in log_content, \
                f"Expected 5/5 tests passed, got:\n{log_content[-500:]}"
        finally:
            os.chdir(original_cwd)
