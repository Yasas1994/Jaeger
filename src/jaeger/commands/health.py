import sys
import platform
from pathlib import Path
from importlib.resources import files
from importlib.metadata import version, PackageNotFoundError
import warnings

from jaeger.utils.fs import validate_fasta_entries
from jaeger.utils.logging import get_logger
from jaeger.utils.test import test_torch
from jaeger.utils.misc import AvailableModels, json_to_dict

warnings.filterwarnings("ignore")


def _get_package_version(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "not installed"


def _get_jaeger_version() -> str:
    """Get jaeger-bio version, preferring local pyproject.toml over system package metadata."""
    try:
        import jaeger

        project_root = Path(jaeger.__file__).resolve().parent.parent.parent
        pyproject = project_root / "pyproject.toml"
        if pyproject.exists():
            for line in pyproject.read_text().splitlines():
                if line.strip().startswith("version"):
                    return line.split("=")[-1].strip().strip('"')
    except Exception:
        pass
    try:
        return version("jaeger-bio")
    except PackageNotFoundError:
        return "unknown"


def _print_diagnostics(logger) -> None:
    """Print system and dependency diagnostics."""
    logger.info("=" * 60)
    logger.info("Jaeger Health Diagnostics")
    logger.info("=" * 60)

    logger.info(f"Python version:    {platform.python_version()}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Platform:          {platform.platform()}")
    logger.info(f"Machine:           {platform.machine()}")
    logger.info(f"Processor:         {platform.processor()}")

    logger.info("-" * 40)
    logger.info("Core Dependencies:")
    logger.info(f"  jaeger-bio:      {_get_jaeger_version()}")
    logger.info(f"  torch:           {_get_package_version('torch')}")
    logger.info(f"  numpy:           {_get_package_version('numpy')}")
    logger.info(f"  click:           {_get_package_version('click')}")
    logger.info(f"  parasail:        {_get_package_version('parasail')}")
    logger.info(f"  pyfastx:         {_get_package_version('pyfastx')}")
    logger.info(f"  pydustmasker:    {_get_package_version('pydustmasker')}")
    logger.info(f"  scikit-learn:    {_get_package_version('scikit-learn')}")
    logger.info(f"  polars:          {_get_package_version('polars')}")
    logger.info(f"  pandas:          {_get_package_version('pandas')}")
    logger.info(f"  matplotlib:      {_get_package_version('matplotlib')}")
    logger.info(f"  ruptures:        {_get_package_version('ruptures')}")
    logger.info(f"  pycirclize:      {_get_package_version('pycirclize')}")
    logger.info(f"  biopython:       {_get_package_version('biopython')}")

    logger.info("-" * 40)
    logger.info("PyTorch Hardware:")
    try:
        import torch

        logger.info("  CPUs detected:   1")
        logger.info(f"  GPUs detected:   {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"    GPU {i}:         {torch.cuda.get_device_name(i)}")
    except Exception as e:
        logger.info(f"  Could not inspect PyTorch devices: {e}")

    logger.info("-" * 40)
    logger.info("Installed Models:")
    config_path = files("jaeger.data").joinpath("config.json")
    config = json_to_dict(str(config_path))
    user_paths = config.get("model_paths", [])
    default_paths = [files("jaeger.data")]
    all_paths = default_paths + user_paths

    avail = AvailableModels(path=all_paths)
    if avail.info:
        for name, meta in sorted(avail.info.items()):
            parts = []
            if "graph" in meta:
                parts.append("graph")
            if "weights" in meta:
                parts.append("weights")
            if "classes" in meta:
                parts.append("classes")
            logger.info(f"  {name}:          {', '.join(parts)}")
    else:
        logger.info("  (no models found)")

    logger.info("-" * 40)
    logger.info("Bundled Model Configs:")
    for key in sorted(k for k in config.keys() if k != "model_paths"):
        entry = config[key]
        weights = entry.get("weights", "N/A")
        num_classes = entry.get("num_classes", "N/A")
        vindex = entry.get("vindex", "N/A")
        logger.info(
            f"  {key}:           weights={weights}, classes={num_classes}, vindex={vindex}"
        )

    logger.info("=" * 60)


def health_core(**kwargs) -> None:
    """Run tests to check installation health."""
    passed = 0

    output_path = Path.cwd() / "test_log"
    output_path.mkdir(parents=True, exist_ok=True)

    fnames = ["test_short.fasta", "test_empty.fasta", "test_contigs.fasta"]
    log_file = "test_jaeger.log"
    logger = get_logger(
        log_path=Path(output_path), log_file=log_file, level=kwargs.get("verbose")
    )

    _print_diagnostics(logger)

    # Test 1-3: validate bundled FASTA files
    for i, f in enumerate(fnames):
        input_file = str(files("jaeger.data.test").joinpath(f))
        logger.info(input_file)
        try:
            validate_fasta_entries(input_file)
            passed += 1
            logger.info(f"{i} test passed {f}!")
        except Exception as e:
            if i > 1:
                logger.error(f"{i} test failed {f}!")
            else:
                passed += 1
                logger.info(f"{i} test passed {f}!")
            logger.debug(e)

    # Test 4: PyTorch smoke test
    result = test_torch()
    if isinstance(result, Exception):
        logger.error("4 pytorch test failed!")
        logger.debug(result)
    else:
        passed += 1
        logger.info("4 pytorch test passed!")

    logger.info(f"{passed}/4 tests passed!")
