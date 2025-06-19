import json
from pathlib import Path
from importlib.resources import files
import tensorflow as tf
from jaeger.nnlib.cmodel import JaegerModel
from jaeger.nnlib.v1.layers import WRes_model_embeddings
from jaeger.preprocess.v1.convert import process_string
from jaeger.preprocess.fasta import fragment_generator
from jaeger.utils.fs import validate_fasta_entries
from jaeger.utils.logging import get_logger
from jaeger.utils.test import test_tf
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


def test_core(**kwargs) -> None:
    """Run tests to check installation"""
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
        logger.info(files("jaeger.data.models.default").joinpath("jaeger_graph"))
        tf.saved_model.save(
            tf.keras.Sequential([tf.keras.layers.Dense(10)]),
            files("jaeger.data.models.default").joinpath("jaeger_graph"),
        )
        tf.saved_model.save(
            model, files("jaeger.data.models.default").joinpath("jaeger_graph")
        )
        # ic(f"model computational graph is written to {path / f"{model_name}_graph"}")

        logger.info("loading the model")
        model = InferModel(
            files("jaeger.data.models.default").joinpath("jaeger_fragment_graph")
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
