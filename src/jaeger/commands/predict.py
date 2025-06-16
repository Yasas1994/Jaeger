import os
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
import traceback
import sys
import psutil
import time
from importlib.resources import files
from importlib.metadata import version
from pathlib import Path
import tensorflow as tf

from jaeger.nnlib.inference import InferModel
from jaeger.postprocess.collect import write_fasta_from_results, write_output
from jaeger.postprocess.prophages import logits_to_df, plot_scores, prophage_report, segment
from jaeger.preprocess.fasta import fragment_generator
from jaeger.utils.gpu import get_device_name
from jaeger.utils.termini import scan_for_terminal_repeats
from jaeger.utils.fs import validate_fasta_entries
from jaeger.utils.misc import json_to_dict, AvailableModels, get_model_id
from jaeger.utils.logging import description, get_logger

GB_BYTES = 1024 ** 3



def run_core(**kwargs):

    current_process = psutil.Process()
    MODEL = kwargs.get("model")
    MODEL_ID = get_model_id(MODEL)
    DATA_PATH = files("jaeger.data")
    if kwargs.get("config") is None:
        CONFIG_PATH = DATA_PATH / "config.json"
    else:
        CONFIG_PATH = kwargs.get("config")
        
    USER_MODEL_PATHS = json_to_dict(CONFIG_PATH).get("model_paths")
    MODEL_INFO = AvailableModels(path=USER_MODEL_PATHS).info[MODEL]
    
    input_file_path = Path(kwargs.get("input"))
    input_file = input_file_path.name
    file_base = input_file_path.stem
    
    OUTPUT_DIR = Path(kwargs.get("output")) / MODEL_ID
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    log_file = Path(f"{file_base}_jaeger.log")
    logger = get_logger(OUTPUT_DIR, log_file, level=kwargs.get("verbose"))
    logger.info(description(version("jaeger-bio")) + "\n{:-^80}".format("validating parameters"))
    logger.debug(DATA_PATH)
    logger.debug(AvailableModels(path=USER_MODEL_PATHS).info)
    
    try:
        num = validate_fasta_entries(str(input_file_path), min_len=kwargs.get("fsize"))
    except Exception as e:
        logger.error(e)
        logger.debug(traceback.format_exc())
        sys.exit(1)

    output_table_path = OUTPUT_DIR / f"{file_base}.tsv"
    output_phage_table_path = OUTPUT_DIR / f"{file_base}_phages.tsv"
    
    if output_table_path.exists() and not kwargs.get('overwrite'):
        logger.error("output file exists. enable --overwrite option to overwrite the output file.")
        sys.exit(1)
    logger.info(MODEL_INFO)
    
    if not MODEL_INFO["graph"].exists():
        logger.error(f"could not find model graph. please check {USER_MODEL_PATHS}")
        sys.exit(1)
        
    tf.config.set_soft_device_placement(True)
    gpus = tf.config.list_physical_devices("GPU")
    mode = None
    
    if kwargs.get('cpu'):
        mode = "CPU"
        tf.config.set_visible_devices([], "GPU")
        logger.info("CPU only mode selected")
    elif gpus:
        mode = "GPU"
        tf.config.set_visible_devices([gpus[kwargs.get('physicalid')]], "GPU")
        try:
            tf.config.set_logical_device_configuration(
                gpus[kwargs.get('physicalid')],
                [tf.config.LogicalDeviceConfiguration(memory_limit=4096, experimental_device_ordinal=10)],
            )
        except Exception as e:
            logger.error(f"an error {e} occurred during virtual device initialization ")
            logger.debug(traceback.format_exc())
    else:
        mode = "CPU"
        logger.warning("could not find a GPU on the system. For optimal performance run Jaeger on a GPU.")
    
    logger.info(f"tensorflow: {version('tensorflow')}")
    logger.info(f"input file: {input_file}")
    logger.info(f"log file: {log_file.name}")
    logger.info(f"outpath: {OUTPUT_DIR.resolve()}")
    logger.info(f"fragment size: {kwargs.get('fsize')}")
    logger.info(f"stride: {kwargs.get('stride')}")
    logger.info(f"batch size: {kwargs.get('batch')}")
    logger.info(f"mode: {mode}")
    logger.info(f"avail mem: {psutil.virtual_memory().available/(GB_BYTES):.2f}GB")
    logger.info(f"avail cpus: {psutil.cpu_count()}")
    logger.info(f"CPU time(s) : {current_process.cpu_times().user:.2f}")
    logger.info(f"wall time(s) : {time.time() - current_process.create_time():.2f}")
    logger.info(f"memory usage : {current_process.memory_full_info().rss/GB_BYTES:.2f}GB ({current_process.memory_percent():.2f}%)")


    device = tf.config.list_logical_devices(mode)
    device_names = [get_device_name(i) for i in device]
    logger.debug(f"{device}, {device_names}")
    if len(device) > 1:
        logger.info(f"Using MirroredStrategy {device_names}")
        strategy = tf.distribute.MirroredStrategy(device_names)
    else:
        logger.info(f"Using OneDeviceStrategy {device_names}")
        strategy = tf.distribute.OneDeviceStrategy(device_names[0])

    
    term_repeats = scan_for_terminal_repeats(
                            file_path=str(input_file_path),
                            num=num,
                            workers= kwargs.get("workers"),
                            fsize=kwargs.get("fsize") )
    
    model = InferModel(MODEL_INFO)
    string_processor_config = model.string_processor_config
    input_dataset = tf.data.Dataset.from_generator(
    fragment_generator(
        str(input_file_path),
        no_progress=False,
        fragsize=kwargs.get("fsize"),
        stride=kwargs.get("stride"),
        num=num,
    ),
    output_signature=(tf.TensorSpec(shape=(), dtype=tf.string)),)

    from jaeger.preprocess.latest.convert import process_string_inference 
    idataset = (
        input_dataset.map(
            process_string_inference(crop_size=kwargs.get("fsize"),  
                                     codons=string_processor_config.get("codon"),
                                     codon_num=string_processor_config.get("codon_id"),
                                     codon_depth=string_processor_config.get("codon_depth"), 
                                     input_type=string_processor_config.get("input_type"),),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(kwargs.get("batch"), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(25)
    )

    with strategy.scope():

        try:
            logger.info("starting model inference")
            y_pred = model.predict(idataset, no_progress=True)
        except Exception as e:
            logger.debug(traceback.format_exc())
            logger.error(
                f'an error {e} occured during inference on {"|".join(device_names)}! check {log_file} for traceback.'
            )
            sys.exit(1)
        
        from jaeger.postprocess.collect import pred_to_dict, write_output

        
        if kwargs.get("getalllabels"):
            pass
        data, data_full = pred_to_dict(
                            y_pred,
                            num_classes=model.class_map.get('num_classes'),
                            fsize=kwargs.get('fsize'),
                            term_repeats=term_repeats)

        num_written = write_output(
                     data,
                     labels=model.class_map.get('class'),
                     indices=model.class_map.get('index'),
                     output_table_path=output_table_path, 
                     output_phage_table_path=output_phage_table_path)
        
        logger.info(f"processed {num_written}/{num} sequences")
        logger.info(f"CPU time(s) : {current_process.cpu_times().user:.2f}")
        logger.info(f"wall time(s) : {time.time() - current_process.create_time():.2f}")
        logger.info(f"memory usage : {current_process.memory_full_info().rss/GB_BYTES:.2f}GB ({current_process.memory_percent():.2f}%)")