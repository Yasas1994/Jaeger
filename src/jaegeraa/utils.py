from typing import Iterable, Any, Tuple
from enum import Enum, auto
import tensorflow as tf
import bz2
import gzip
import logging
import lzma
import os
import argparse
import logging
logger = logging.getLogger(__name__)

class Compression(Enum):
    gzip = auto()
    bzip2 = auto()
    xz = auto()
    noncompressed = auto()
    
def is_compressed(filepath):
    """ Checks if a file is compressed (gzip, bzip2 or xz) """
    with open(filepath, "rb") as fin:
        signature = fin.peek(8)[:8]
        if tuple(signature[:2]) == (0x1F, 0x8B):
            return Compression.gzip
        elif tuple(signature[:3]) == (0x42, 0x5A, 0x68):
            return Compression.bzip2
        elif tuple(signature[:7]) == (0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00, 0x00):
            return Compression.xz
        else:
            return Compression.noncompressed

def get_compressed_file_handle(path):
    filepath_compression = is_compressed(path)
    if filepath_compression == Compression.gzip:
        f = gzip.open(path, "rt")
    elif filepath_compression == Compression.bzip2:
        f = bz2.open(path, "rt")
    elif filepath_compression == Compression.xz:
        f = lzma.open(path, "rt")
    else:
        f = open(path, "r")
    return f

def signal_fl(it):
    '''get a signal at the begining and the end of a iterator'''
    iterable = iter(it)
    yield True, next(iterable)
    ret_var = next(iterable)
    for val in iterable:
        yield 0, ret_var
        ret_var = val
    yield 1, ret_var
def signal_l(it):
    '''get a signal at the end of the iterator'''
    iterable = iter(it)
    ret_var = next(iterable)
    for val in iterable:
        yield 0, ret_var
        ret_var = val
    yield 1, ret_var


def dir_path(path):
    '''checks path and creates if absent'''
    if os.path.isdir(path):
        return path
    else:
        os.mkdir(path)
        return path


def file_path(path):
    '''checks if file is present'''
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

def create_logger(args):
    # Logging config
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]

    logging.addLevelName(logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
    logging.addLevelName(logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
    logging.addLevelName(logging.INFO, "\033[1;42m%s\033[1;0m" % logging.getLevelName(logging.INFO))
    logging.addLevelName(logging.DEBUG, "\033[1;43m%s\033[1;0m" % logging.getLevelName(logging.DEBUG))
    logging.getLogger().addFilter(logging.Filter("Jaeger"))

    logger = logging.getLogger('Jaeger')
    logger.setLevel(log_levels[args.verbose])

    ch = logging.StreamHandler()
    ch.setLevel(log_levels[args.verbose])
    ch.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s:%(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(ch)

    return logger

        
def configure_multi_gpu_inference(gpus):
    if gpus > 0:
            return tf.distribute.MirroredStrategy()
    else:
        None

def get_device_name(device):
    name = device.name 
    return  f"{name.split(':',1)[-1]}"

def create_virtual_gpus(logger, num_gpus=2, memory_limit=2048):
    #creates virtual GPUs for testing multigpu inference
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Create n virtual GPUs with user defined amount of memory each
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit) for i in range(num_gpus)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            logger.info(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            logger.error(e)
            

def fasta_entries(input_file_handle):
    num = 0
    for i in input_file_handle: 
        if i.startswith('>'):
            num+=1
    input_file_handle.seek(0)
    return num