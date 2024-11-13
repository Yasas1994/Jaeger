from enum import Enum, auto
from typing import Union
import argparse
import bz2
import gzip
import logging
import lzma
import os
import traceback
import pyfastx
import tensorflow as tf

logger = logging.getLogger("Jaeger")


def safe_divide(numerator, denominator):
    try:
        result = round(numerator / denominator, 2)
    except ZeroDivisionError:
        logger.debug("Error: Division by zero!")
        result = 0
    return result


def description(version):
    return f"""
                  .
               ,'/ \`.
              |\/___\/|
              \'\   /`/          ██╗ █████╗ ███████╗ ██████╗ ███████╗██████╗
               `.\ /,'           ██║██╔══██╗██╔════╝██╔════╝ ██╔════╝██╔══██╗
                  |              ██║███████║█████╗  ██║  ███╗█████╗  ██████╔╝
                  |         ██   ██║██╔══██║██╔══╝  ██║   ██║██╔══╝  ██╔══██╗
                 |=|        ╚█████╔╝██║  ██║███████╗╚██████╔╝███████╗██║  ██║
            /\  ,|=|.  /\    ╚════╝ ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝
        ,'`.  \/ |=| \/  ,'`. 
      ,'    `.|\ `-' /|,'    `. 
    ,'   .-._ \ `---' / _,-.   `.
       ,'    `-`-._,-'-'   `.
      '
    \n\n## Jaeger {version} (yet AnothEr phaGe idEntifier) Deep-learning based
bacteriophage discovery https://github.com/Yasas1994/Jaeger.git

"""


class Compression(Enum):
    gzip = auto()
    bzip2 = auto()
    xz = auto()
    noncompressed = auto()


def is_compressed(filepath):
    """Checks if a file is compressed (gzip, bzip2 or xz)"""
    with open(filepath, "rb") as fin:
        signature = fin.peek(8)[:8]
        if tuple(signature[:2]) == (0x1F, 0x8B):
            return Compression.gzip
        elif tuple(signature[:3]) == (0x42, 0x5A, 0x68):
            return Compression.bzip2
        elif tuple(signature[:7]) == (0xFD,
                                      0x37,
                                      0x7A,
                                      0x58,
                                      0x5A,
                                      0x00,
                                      0x00):
            return Compression.xz
        else:
            return Compression.noncompressed


def get_compressed_file_handle(path):
    filepath_compression = is_compressed(path)
    if filepath_compression == Compression.gzip:
        return gzip.open(path, "rt")
    elif filepath_compression == Compression.bzip2:
        return bz2.open(path, "rt")
    elif filepath_compression == Compression.xz:
        return lzma.open(path, "rt")
    else:
        return open(path, "r")


def remove_directory_recursively(directory):
    # Walk the directory tree in bottom-up order
    for root, dirs, files in os.walk(directory, topdown=False):
        # Remove all files
        for name in files:
            file_path = os.path.join(root, name)
            os.remove(file_path)
        # Remove all subdirectories
        for name in dirs:
            dir_path = os.path.join(root, name)
            os.rmdir(dir_path)
    # Remove the main directory
    os.rmdir(directory)

def delete_all_in_directory(root_path):
    # Check if the root directory exists
    if os.path.exists(root_path):
        # Iterate over all files and directories in the root directory
        for filename in os.listdir(root_path):
            file_path = os.path.join(root_path, filename)
            # If it's a directory, recursively delete its contents
            if os.path.isdir(file_path):
                delete_all_in_directory(file_path)
                os.rmdir(file_path)  # Remove the empty directory
            # If it's a file, delete it
            else:
                os.remove(file_path)
        # Finally, remove the root directory itself
        os.rmdir(root_path)


def signal_fl(it):
    """get a signal at the begining and the end of a iterator"""
    iterable = iter(it)
    yield True, next(iterable)
    ret_var = next(iterable)
    for val in iterable:
        yield 0, ret_var
        ret_var = val
    yield 1, ret_var


def signal_l(it):
    """get a signal at the end of the iterator"""
    iterable = iter(it)
    ret_var = next(iterable)
    for val in iterable:
        yield 0, ret_var
        ret_var = val
    yield 1, ret_var


def dir_path(path):
    """checks path and creates if absent"""
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


def check_file_path(path):
    """checks if file is present"""
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"ERROR:{path} is not a valid file")


def remove_directory(directory):
    # Check if the directory exists
    if os.path.exists(directory):
        # Iterate over the files in the directory
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)
            # Check if it's a file
            if os.path.isfile(file_path):
                # Remove the file
                os.remove(file_path)
            # If it's a directory, recursively remove it
            elif os.path.isdir(file_path):
                remove_directory(file_path)
        # Once all files are removed, remove the directory itself
        os.rmdir(directory)


class LogFormatter(logging.Formatter):

    COLOR_CODES = {
        logging.CRITICAL: "\033[1;35m",  # bright/bold magenta
        logging.ERROR:    "\033[1;31m",  # bright/bold red
        logging.WARNING:  "\033[1;33m",  # bright/bold yellow
        logging.INFO:     "\033[0;37m",  # white / light gray
        logging.DEBUG:    "\033[1;30m"   # bright/bold black / dark gray
    }

    RESET_CODE = "\033[0m"

    def __init__(self, color, *args, **kwargs):
        super(LogFormatter, self).__init__(*args, **kwargs)
        self.color = color

    def format(self, record, *args, **kwargs):
        if (self.color and record.levelno in self.COLOR_CODES):
            record.color_on = self.COLOR_CODES[record.levelno]
            record.color_off = self.RESET_CODE
        else:
            record.color_on = ""
            record.color_off = ""
        return super(LogFormatter, self).format(record, *args, **kwargs)


class JaegerLogger(logging.Logger):
    def __init__(self, args, log_file):
        super().__init__(args)

        log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
        df = "%Y-%m-%d %H:%M:%S"
        logging.addLevelName(
            logging.WARNING,
            logging.getLevelName(logging.WARNING),
        )
        logging.addLevelName(
            logging.ERROR,
            logging.getLevelName(logging.ERROR)
        )
        logging.addLevelName(
            logging.INFO,
            logging.getLevelName(logging.INFO)
        )
        logging.addLevelName(
            logging.DEBUG,
            logging.getLevelName(logging.DEBUG)
        )
        logging.getLogger().addFilter(logging.Filter("Jaeger"))

        logger = logging.getLogger("Jaeger")

        self.file_handler = logging.FileHandler(log_file)
        logger.addHandler(self.file_handler)
        logger.setLevel(logging.DEBUG)

        self.stderr_handler = logging.StreamHandler()
        self.stderr_handler.setLevel(log_levels[args.verbose])

        self.formatter_stdout = LogFormatter(
            fmt="%(color_on)s[%(asctime)s | %(levelname)5s]: %(message)s%(color_off)s",
            datefmt=df,
            color=True
        )
        self.formatter_log = LogFormatter(
            fmt="[%(asctime)s]: %(message)s",
            datefmt=df,
            color=False
        )
        self.formatter_clean = LogFormatter("")

        self.stderr_handler.setFormatter(self.formatter_stdout)
        self.file_handler.setFormatter(self.formatter_log)
        logger.addHandler(self.stderr_handler)
        logger.addHandler(self.file_handler)
        self.logger = logger

    def info(self, message, cleanformat=False):
        if cleanformat:

            self.stderr_handler.setFormatter(self.formatter_clean)
            self.file_handler.setFormatter(self.formatter_clean)
            self.logger.info(message)
            self.reset_handler()
        else:
            self.logger.info(message)

    def warn(self, message, cleanformat=False):
        if cleanformat:

            self.stderr_handler.setFormatter(self.formatter_clean)
            self.file_handler.setFormatter(self.formatter_clean)
            self.logger.warning(message)
            self.reset_handler()
        else:
            self.logger.warning(message)

    def error(self, message, cleanformat=False):
        if cleanformat:

            self.stderr_handler.setFormatter(self.formatter_clean)
            self.file_handler.setFormatter(self.formatter_clean)
            self.logger.error(message)
            self.reset_handler()
        else:
            self.logger.error(message)

    def debug(self, message, cleanformat=False):
        if cleanformat:

            self.stderr_handler.setFormatter(self.formatter_clean)
            self.file_handler.setFormatter(self.formatter_clean)
            self.logger.debug(message)
            self.reset_handler()
        else:
            self.logger.debug(message)

    def reset_handler(self):

        self.stderr_handler.setFormatter(self.formatter_stdout)
        self.file_handler.setFormatter(self.formatter_log)


def configure_multi_gpu_inference(gpus):
    if gpus > 0:
        return tf.distribute.MirroredStrategy()
    else:
        None


def get_device_name(device):
    name = device.name
    return f"{name.split(':',1)[-1]}"


def create_virtual_gpus(logger, num_gpus=2, memory_limit=2048):
    if gpus := tf.config.list_physical_devices("GPU"):
        # Create n virtual GPUs with user defined amount of memory each
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [
                    tf.config.LogicalDeviceConfiguration(
                        memory_limit=memory_limit)
                    for _ in range(num_gpus)
                ],
            )
            logical_gpus = tf.config.list_logical_devices("GPU")
            logger.info(len(gpus), "Physical GPU,", len(logical_gpus),
                        "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            logger.error(e)
            logger.debug(traceback.format_exc())


def format_seconds(seconds):
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes} minutes and {remaining_seconds} seconds"


def validate_fasta_entries(input_file_path: str,
                           min_len: int = 2048) -> Union[int, Exception]:
    num = 0
    gt_min_len = 0
    # try:
    logger.debug("validating fasta file")
    fa = pyfastx.Fasta(input_file_path, build_index=False)
    for seq in fa:
        num += 1
        gt_min_len += 1 if len(seq[1]) >= min_len else 0
    logger.info(f"{gt_min_len}/{num} entries in {input_file_path}")

    if gt_min_len == 0:
        raise Exception(f"all records in {input_file_path} are < {min_len}bp")

    return num

    # except Exception as e:
    #     logger.error(e)
    #     logger.debug(traceback.format_exc())
    #     # sys.exit(1)
    #     return e


class Precision_per_class(tf.keras.metrics.Metric):

    def __init__(self, name="Precision_per_class", num_classes=4, **kwargs):
        super(Precision_per_class, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(
            name="tp", initializer="zeros", shape=num_classes
        )
        self.pred_positives = self.add_weight(
            name="pp", initializer="zeros", shape=num_classes
        )
        self.num_classes = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        y_pred = tf.one_hot(y_pred, self.num_classes)

        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        tp = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        tp = tf.cast(tp, self.dtype)
        pp = tf.equal(y_pred, True)
        pp = tf.cast(pp, self.dtype)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, self.shape)
            tp = tf.multiply(tp, sample_weight)
            pp = tf.multiply(pp, sample_weight)

        self.true_positives.assign_add(tf.reduce_sum(tp, axis=0))
        self.pred_positives.assign_add(tf.reduce_sum(pp, axis=0))

    def reset_state(self):
        self.true_positives.assign(tf.zeros(shape=self.num_classes))
        self.pred_positives.assign(tf.zeros(shape=self.num_classes))

    def result(self):
        result = tf.math.divide_no_nan(self.true_positives,
                                       self.pred_positives)
        return result


class Recall_per_class(tf.keras.metrics.Metric):

    def __init__(self, name="Recall_per_class", num_classes=4, **kwargs):
        super(Recall_per_class, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(
            name="tp", initializer="zeros", shape=num_classes
        )
        self.positives = self.add_weight(
            name="positives", initializer="zeros", shape=num_classes
        )
        self.num_classes = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_pred = tf.math.argmax(y_pred, axis=-1)
        y_pred = tf.one_hot(y_pred, self.num_classes)

        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        tp = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        tp = tf.cast(tp, self.dtype)

        p = tf.equal(y_true, True)
        p = tf.cast(p, self.dtype)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, self.shape)
            tp = tf.multiply(tp, sample_weight)
            p = tf.multiply(p, sample_weight)

        self.true_positives.assign_add(tf.reduce_sum(tp, axis=0))
        self.positives.assign_add(tf.reduce_sum(p, axis=0))

    def reset_state(self):
        self.true_positives.assign(tf.zeros(shape=self.num_classes))
        self.positives.assign(tf.zeros(shape=self.num_classes))

    def result(self):
        result = tf.math.divide_no_nan(self.true_positives, self.positives)

        return result
