import os
import traceback
import sys
import psutil
import time
import numpy as np
from importlib.resources import files
from importlib.metadata import version
import json
import h5py
import joblib
from pathlib import Path
import tensorflow as tf

from src.nnlib.v1.cmodel import JaegerModel
from src.nnlib.v1.layers import WRes_model_embeddings, create_jaeger_model
from src.postprocess.collect import write_fasta_from_results, write_output
from src.postprocess.prophages import logits_to_df, plot_scores, prophage_report, segment
from src.preprocess.fasta import fragment_generator
from src.utils.gpu import get_device_name
from src.utils.termini import scan_for_terminal_repeats
from src.utils.fs import validate_fasta_entries
from src.utils.logging import description, get_logger

GB_BYTES = 1024 ** 3

def run_core(**kwargs):

    current_process = psutil.Process()
    
    config_path = files("data").joinpath("config.json")
    config = json.loads(config_path.read_text())
    
    input_file_path = Path(kwargs.get("input"))
    input_file = input_file_path.name
    file_base = input_file_path.stem
    
    output_dir = Path(kwargs.get("output")) / file_base
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = f"{file_base}_jaeger.log"
    logger = get_logger(output_dir, log_file)
    
    logger.info(description(version("jaeger-bio")) + "\n{:-^80}".format("validating parameters"))
    logger.debug(files("data"))
    
    try:
        num = validate_fasta_entries(str(input_file_path), min_len=kwargs.get("fsize"))
    except Exception as e:
        logger.error(e)
        logger.debug(traceback.format_exc())
        sys.exit(1)

    output_table_path = output_dir / f"{file_base}_{config[kwargs.get('model')]['suffix']}_jaeger.tsv"
    output_phage_table_path = output_dir / f"{file_base}_{config[kwargs.get('model')]['suffix']}_phages_jaeger.tsv"
    
    if output_table_path.exists() and not kwargs.get('overwrite'):
        logger.error("output file exists. enable --overwrite option to overwrite the output file.")
        sys.exit(1)
    
    weights_path = files("data").joinpath(config[kwargs.get('model')]["weights"])
    num_class = config[kwargs.get('model')]["num_classes"]
    ood_params = None
    
    if not weights_path.exists():
        logger.error("could not find model weights. please check the data dir")
    
    if config[kwargs.get('model')]["ood"]:
        ood_weights_path = files("data").joinpath(config[kwargs.get('model')]["ood"])
        
        if ood_weights_path.suffix == ".h5":
            with h5py.File(ood_weights_path, "r") as hf:
                ood_params = {
                    "type": "params",
                    "coeff": np.array(hf["coeff"]),
                    "intercept": np.array(hf["intercept"]),
                    "batch_mean": np.array(hf["mean_batch"]),
                }
        elif ood_weights_path.suffix == ".pkl":
            mean_file = files("data").joinpath("batch_means.npy")
            std_file = files("data").joinpath("batch_std.npy")
            batch_mean = np.load(mean_file)
            batch_std = np.load(std_file)
            ood_params = {
                "type": "sklearn",
                "model": joblib.load(ood_weights_path),
                "batch_mean": batch_mean,
                "batch_std": batch_std,
            }
    

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
        logger.warn("could not find a GPU on the system. For optimal performance run Jaeger on a GPU.")
    
    logger.info(f"tensorflow: {version('tensorflow')}")
    logger.info(f"input file: {input_file}")
    logger.info(f"log file: {log_file.name}")
    logger.info(f"outpath: {output_dir.resolve()}")
    logger.info(f"fragment size: {kwargs.get('fsize')}")
    logger.info(f"stride: {kwargs.get('stride')}")
    logger.info(f"batch size: {kwargs.get('batch')}")
    logger.info(f"mode: {mode}")
    logger.info(f"avail mem: {psutil.virtual_memory().available/(GB_BYTES):.2f}GB")
    logger.info(f"avail cpus: {psutil.cpu_count()}")
    logger.info(f"CPU time(s) : {current_process.cpu_times().user:.2f}")
    logger.info(f"wall time(s) : {time.time() - current_process.create_time():.2f}")
    logger.info(f"memory usage : {current_process.memory_full_info().rss/GB_BYTES:.2f}GB ({current_process.memory_percent():.2f}%)")


    term_repeats = scan_for_terminal_repeats(
                                file_path=input_file_path,
                                num=num,
                                workers= kwargs.get("workers"),
                                fsize=kwargs.get("fsize") )

    device = tf.config.list_logical_devices(mode)
    device_names = [get_device_name(i) for i in device]
    logger.debug(f"{device}, {device_names}")
    if len(device) > 1:
        logger.info(f"Using MirroredStrategy {device_names}")
        strategy = tf.distribute.MirroredStrategy(device_names)
    else:
        logger.info(f"Using OneDeviceStrategy {device_names}")
        strategy = tf.distribute.OneDeviceStrategy(device_names[0])

    tf.config.set_soft_device_placement(True)
    with strategy.scope():

        input_dataset = tf.data.Dataset.from_generator(
            fragment_generator(
                input_file_path,
                fragsize=kwargs.get("fsize"),
                stride=kwargs.get("stride"),
                num=num,
            ),
            output_signature=(tf.TensorSpec(shape=(), dtype=tf.string)),
        )
        if kwargs.get("model") == "default":
            from preprocess.v1 import process_string
            idataset = (
                input_dataset.map(
                    process_string(crop_size=kwargs.get("fsize")),
                    num_parallel_calls=tf.data.AUTOTUNE,
                )
                .batch(kwargs.get("batch"), num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(25)
            )
            inputs, outputs = WRes_model_embeddings(
                input_shape=(None,), dropout_active=False
            )
        else:
            from preprocess.v2 import process_string
            insize = (int(kwargs.get("fsize")) // 3) - 1
            idataset = (
                input_dataset.map(
                    process_string(crop_size=kwargs.get("fsize")),
                    num_parallel_calls=tf.data.AUTOTUNE,
                )
                .batch(kwargs.get("batch"), num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(50)
            )

            inputs, outputs = create_jaeger_model(
                input_shape=(6, insize, 11), out_shape=num_class
            )
        model = JaegerModel(inputs=inputs, outputs=outputs)

        logger.info("loading model to memory")
        model.load_weights(
            filepath=weights_path
            )
        logger.info(
            f'avail mem : {psutil.virtual_memory().available/(GB_BYTES): .2f}GB\n{"-"*80}'
        )

        if mode == "GPU":
            for device_number, d in enumerate(device_names):
                gpu_mem = tf.config.experimental.get_memory_info(d)
                logger.info(
                    f'GPU {device_number} current : {gpu_mem["current"]/(GB_BYTES): .2f}GB  peak : {gpu_mem["peak"]/(GB_BYTES) : .2f}GB'
                )

        try:
            logger.info("starting model inference")
            y_pred = model.predict(idataset, workers=kwargs.get('workers'), verbose=0)
        except Exception as e:
            logger.debug(traceback.format_exc())
            logger.error(
                f'an error {e} occured during inference on {"|".join(device_names)}! check {log_file} for traceback.'
            )
            sys.exit(1)
        logger.info(f"processed {data.get('headers').shape[0]}/{num} sequences")
        from src.postprocess.collect import pred_to_dict

        data, data_full = pred_to_dict(config,
                            y_pred,
                            model=kwargs.get('model'),
                            fsize=kwargs.get('fsize'),
                            ood_params=ood_params,
                            term_repeats=term_repeats)

        write_output(config,
                     data,
                     output_table_path=output_table_path, 
                     output_phage_table_path=output_phage_table_path)
        
        if kwargs.get('getsequences'):
            output_fasta_file = f"{file_base}_{config[kwargs.get('model')]['suffix']}_phages_jaeger.fasta"
            output_fasta_file_path =output_dir/output_fasta_file
            write_fasta_from_results(input_fasta=input_file_path,
                                     output_tsv=output_phage_table_path,
                                     output_fasta=output_fasta_file_path)
            logger.info(f"{output_fasta_file} created")
        

        if kwargs.get('getalllogits'):
            output_logits = f"{file_base}_{config[kwargs.get('model')]['suffix']}_jaeger.npy"
            output_logits_path = os.path.join(output_dir, output_logits)
            logger.info(f"writing window-wise scores to {output_logits}")
            np.save(
                output_logits_path,
                dict(zip(data.get('headers'), data.get('predictions'))), 
                allow_pickle=True
            )
            logger.info(f"{output_logits} created")

        if kwargs.get('prophage'):
            # still experimental - needs more testing!!!!
            try:
                if logits_df := logits_to_df(
                    config,
                    cutoff_length=kwargs.get('lc'),
                    **data_full
                ):
                    logger.info("identifying prophages")
                    pro_dir = output_dir / f"{file_base}_{config[kwargs.get('model')]['suffix']}_prophages"
                    plots_dir = pro_dir / "plots"
                    for dir in [pro_dir, plots_dir]:
                        dir.mkdir(parents=True, exist_ok=True)

                    phage_cord = segment(
                        logits_df,
                        outdir=plots_dir,
                        cutoff_length=kwargs.get('lc'),
                        sensitivity=kwargs.get('sensitivity'),
                        identifier="phage"
                    )
                    plot_scores(
                        logits_df,
                        config=config,
                        model=kwargs.get("model"),
                        fsize=kwargs.get("fsize"),
                        infile_base=file_base,
                        outdir=plots_dir,
                        phage_cordinates=phage_cord,
                    )
                    prophage_report(fsize=kwargs.get("fsize"),
                                    filehandle=input_file_path,
                                    prophage_cordinates=phage_cord,
                                    outdir=pro_dir)
                else:
                    logger.info("no prophage regions found")
            except Exception as e:
                logger.error(f"an error {e} occured during the prophage prediction step")
                logger.debug(traceback.format_exc())

        logger.info(f"CPU time(s) : {current_process.cpu_times().user:.2f}")
        logger.info(f"wall time(s) : {time.time() - current_process.create_time():.2f}")
        logger.info(
            f"memory usage : {current_process.memory_full_info().rss/1024**3:.2f}Gb ({current_process.memory_percent():.2f}%)"
        )

