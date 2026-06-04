import os
os.environ["WRAPT_DISABLE_EXTENSIONS"] = "true"
import json
import sys
import platform
from pathlib import Path
from importlib.resources import files
from importlib.metadata import version, PackageNotFoundError
import tensorflow as tf
from jaeger.nnlib.inference import JaegerModel
from jaeger.nnlib.v1.layers import WRes_model_embeddings
from jaeger.preprocess.v1.convert import process_string
from jaeger.preprocess.fasta import fragment_generator
from jaeger.utils.fs import validate_fasta_entries
from jaeger.utils.logging import get_logger
from jaeger.utils.test import test_tf
from jaeger.utils.misc import AvailableModels, json_to_dict
import warnings
from typing import Generator, Any
from jaeger.utils.misc import track_ms as track

warnings.filterwarnings("ignore")


class InferModel:
    """
    loads a graph given a dict with model graph location and class map
    consumnes batched iterators and returns logits per iterator element
    """

    def __init__(self, graph_path):
        self.loaded_model = tf.saved_model.load(graph_path)
        self.inference_fn = self.loaded_model.signatures["serving_default"]

    @tf.function
    def _predict_step(self, batch):
        # Unpack the data
        x, y = batch[0], batch[1:]
        # set model to inference mode
        y_logits = self.inference_fn(
            inputs=x["forward_1"],
            inputs_1=x["forward_2"],
            inputs_2=x["forward_3"],
            inputs_3=x["reverse_1"],
            inputs_4=x["reverse_2"],
            inputs_5=x["reverse_3"],
        )

        return {"y_hat": y_logits, "meta": y}

    def predict(self, x) -> Generator[Any, Any, Any]:
        accum = []
        for batch in track(x, description="[cyan]Crunching data..."):
            accum.append(self._predict_step(batch))
        return accum


def _get_package_version(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "not installed"


def _get_jaeger_version() -> str:
    """Get jaeger-bio version, preferring local pyproject.toml over system package metadata."""
    # Prefer local editable install's pyproject.toml
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
    # Fallback to importlib.metadata
    try:
        return version("jaeger-bio")
    except PackageNotFoundError:
        return "unknown"


def _print_diagnostics(logger) -> None:
    """Print system and dependency diagnostics."""
    logger.info("=" * 60)
    logger.info("Jaeger Health Diagnostics")
    logger.info("=" * 60)

    # Python environment
    logger.info(f"Python version:    {platform.python_version()}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Platform:          {platform.platform()}")
    logger.info(f"Machine:           {platform.machine()}")
    logger.info(f"Processor:         {platform.processor()}")

    # Core dependency versions
    logger.info("-" * 40)
    logger.info("Core Dependencies:")
    logger.info(f"  jaeger-bio:      {_get_jaeger_version()}")
    logger.info(f"  tensorflow:      {_get_package_version('tensorflow')}")
    logger.info(f"  keras:           {_get_package_version('keras')}")
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

    # TensorFlow hardware info
    logger.info("-" * 40)
    logger.info("TensorFlow Hardware:")
    cpus = tf.config.list_physical_devices("CPU")
    gpus = tf.config.list_physical_devices("GPU")
    logger.info(f"  CPUs detected:   {len(cpus)}")
    logger.info(f"  GPUs detected:   {len(gpus)}")
    for i, gpu in enumerate(gpus):
        logger.info(f"    GPU {i}:         {gpu.name}")
    logger.info(f"  TF built with CUDA: {tf.test.is_built_with_cuda()}")

    # Installed models
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

    # Bundled config entries
    logger.info("-" * 40)
    logger.info("Bundled Model Configs:")
    for key in sorted(k for k in config.keys() if k != "model_paths"):
        entry = config[key]
        weights = entry.get("weights", "N/A")
        num_classes = entry.get("num_classes", "N/A")
        vindex = entry.get("vindex", "N/A")
        logger.info(f"  {key}:           weights={weights}, classes={num_classes}, vindex={vindex}")

    logger.info("=" * 60)


def health_core(**kwargs) -> None:
    """Run tests to check installation health."""
    passed = 0

    output_path = Path.cwd() / "test_log"
    output_path.mkdir(parents=True, exist_ok=True)
    fsize = 2048
    stride = 2048
    batch = 64

    fnames = ["test_short.fasta", "test_empty.fasta", "test_contigs.fasta"]
    log_file = "test_jaeger.log"
    logger = get_logger(
        log_path=Path(output_path), log_file=log_file, level=kwargs.get("verbose")
    )

    # Print diagnostics first
    _print_diagnostics(logger)

    # Test 1-3
    for i, f in enumerate(fnames):
        input_file = str(files("jaeger.data.test").joinpath(f))
        logger.info(input_file)
        try:
            num = validate_fasta_entries(input_file)
            passed += 1
            logger.info(f"{i} test passed {f}!")
        except Exception as e:
            if i > 1:
                logger.error(f"{i} test failed {f}!")
            else:
                passed += 1
                logger.info(f"{i} test passed {f}!")
            logger.debug(e)

    # Test 4
    result = test_tf()
    if isinstance(result, Exception):
        logger.error("4 tensorflow test failed!")
        logger.debug(result)
    else:
        passed += 1
        logger.info("4 tensorflow test passed!")

    # Test 5
    try:
        tf.config.set_soft_device_placement(True)
        config_path = files("jaeger.data").joinpath("config.json")
        config = json.loads(config_path.read_text())
        weights_path = files("jaeger.data.models.default").joinpath(
            config["default"]["weights"]
        )

        input_dataset = tf.data.Dataset.from_generator(
            fragment_generator(
                input_file,
                fragsize=fsize,
                stride=stride,
                num=num,
            ),
            output_signature=(tf.TensorSpec(shape=(), dtype=tf.string)),
        )

        logger.info("loading the dataset")
        idataset = input_dataset.map(
            process_string(crop_size=fsize),
        ).batch(batch)

        inputs, outputs = WRes_model_embeddings(
            input_shape=(None,), dropout_active=False
        )

        logger.info("creating the model")
        model = JaegerModel(inputs=inputs, outputs=outputs)
        model.load_weights(filepath=weights_path)
        model.summary()
        logger.info(files("jaeger.data.models.test").joinpath("jaeger_fragment_graph"))
        tf.saved_model.save(
            model,
            files("jaeger.data.models.test").joinpath("jaeger_fragment_graph"),
        )

        logger.info("loading the model")
        model = InferModel(
            files("jaeger.data.models.test").joinpath("jaeger_fragment_graph")
        )
        logger.info("starting model inference")
        _ = model.predict(idataset)
        logger.info("5 test model passed!")
        passed += 1
    except Exception as e:
        logger.exception("5 test model failed!")
        logger.debug(e)
    finally:
        logger.info(f"{passed}/5 tests passed!")
