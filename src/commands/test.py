import json
from pathlib import Path
from importlib.resources import files
import tensorflow as tf
from nnlib.v1.cmodel import JaegerModel
from nnlib.v1.layers import WRes_model_embeddings
from preprocess.v1.convert import process_string
from preprocess.fasta import fragment_generator
from utils.fs import validate_fasta_entries
from utils.logging import get_logger
from utils.test import test_tf
import warnings

warnings.filterwarnings("ignore")


def test_core(**kwargs) -> None:
    """Run tests to check installation"""
    passed = 0

    output_path = Path.cwd() / "test_log"
    output_path.mkdir(parents=True, exist_ok=True)
    fsize = 2048
    stride = 2048
    batch = 64

    fnames = ['test_short.fasta', 'test_empty.fasta', 'test_contigs.fasta']
    log_file = "test_jaeger.log"
    logger = get_logger(log_path=Path(output_path),
                        log_file=log_file,
                        level=kwargs.get("verbose"))

    # Test 1-3
    for i, f in enumerate(fnames):
        input_file = str(files("data").joinpath(f))
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
        config_path = files("data").joinpath("config.json")
        config = json.loads(config_path.read_text())
        weights_path = files("data").joinpath(config["default"]["weights"])
        
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
        idataset = (
            input_dataset.map(
                process_string(crop_size=fsize),
            ).batch(batch)
        )
        
        inputs, outputs = WRes_model_embeddings(
            input_shape=(None,), dropout_active=False
        )
        
        logger.info("creating the model")
        model = JaegerModel(inputs=inputs, outputs=outputs)
        model.load_weights(filepath=weights_path)
        
        logger.info("starting model inference")
        _ = model.predict(idataset,  verbose=0)

        logger.info("5 test model passed!")
        passed += 1
    except Exception as e:
        logger.exception("5 test model failed!")
        logger.debug(e)
    finally:
        logger.info(f"{passed}/5 tests passed!")