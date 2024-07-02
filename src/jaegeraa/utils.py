from typing import Iterable, Any, Tuple
from enum import Enum, auto
import tensorflow as tf
from datetime import datetime
import bz2
import gzip
import logging
import lzma
import os
import argparse
import logging
logger = logging.getLogger(__name__)

def safe_divide(numerator, denominator):
    try:
        result = round(numerator / denominator, 2)
    except ZeroDivisionError:
        logger.debug("Error: Division by zero!")
        result = 0
    return result

def description(version):
    return f'''
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
    \n\n## Jaeger {version} (yet AnothEr phaGe idEntifier) Deep-learning based bacteriophage discovery \nhttps://github.com/Yasas1994/Jaeger.git'''

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

class LOGGER:
    def __init__(self,args):
        self.input_file_path=args.input
        self.input_file = os.path.basename(self.input_file_path)
        log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]

        logging.addLevelName(logging.WARNING, "\033[1;33m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
        logging.addLevelName(logging.ERROR, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
        logging.addLevelName(logging.INFO, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.INFO))
        logging.addLevelName(logging.DEBUG, "\033[1;33m%s\033[1;0m" % logging.getLevelName(logging.DEBUG))
        logging.getLogger().addFilter(logging.Filter("Jaeger"))

        logger = logging.getLogger('Jaeger')

        file_base = os.path.splitext(os.path.basename(args.input))[0]
        self.file_handler = logging.FileHandler(os.path.join(args.output,f"{file_base}_jaeger.log"))
        logger.addHandler(self.file_handler)
        logger.setLevel(logging.DEBUG)

        self.stderr_handler = logging.StreamHandler()
        self.stderr_handler.setLevel(log_levels[args.verbose])

        self.formatter_stdout = logging.Formatter('[%(asctime)s | %(levelname)s] : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        self.formatter_log =logging.Formatter('[%(asctime)s] : %(message)s' , datefmt='%Y-%m-%d %H:%M:%S')
        self.formatter_clean=logging.Formatter('')

        self.stderr_handler.setFormatter(self.formatter_stdout)
        self.file_handler.setFormatter(self.formatter_log)
        logger.addHandler(self.stderr_handler)
        logger.addHandler(self.file_handler)
        self.logger = logger

    def info(self,message, cleanformat=False):
        if cleanformat == True:

            self.stderr_handler.setFormatter(self.formatter_clean)
            self.file_handler.setFormatter(self.formatter_clean)
            self.logger.info(message)
            self.reset_handler()
        else:
            self.logger.info(message)

    def warn(self,message, cleanformat=False):
        if cleanformat == True:

            self.stderr_handler.setFormatter(self.formatter_clean)
            self.file_handler.setFormatter(self.formatter_clean)
            self.logger.warning(message)
            self.reset_handler()
        else:
            self.logger.warning(message)  

    def error(self,message, cleanformat=False):
        if cleanformat == True:

            self.stderr_handler.setFormatter(self.formatter_clean)
            self.file_handler.setFormatter(self.formatter_clean)
            self.logger.error(message)
            self.reset_handler()
        else:
            self.logger.error(message)  

    def debug(self,message, cleanformat=False):
        if cleanformat == True:

            self.stderr_handler.setFormatter(self.formatter_clean)
            self.file_handler.setFormatter(self.formatter_clean)
            self.logger.debug(message)
            self.reset_handler()
        else:
            self.logger.debug(message)         

    def reset_handler(self):

        self.stderr_handler.setFormatter(self.formatter_stdout)
        self.file_handler.setFormatter(self.formatter_log)

class JaegerLogger(logging.Logger):
    def __init__(self, args):
        super().__init__(args)

        self.input_file_path=args.input
        self.input_file = os.path.basename(self.input_file_path)
        log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]

        logging.addLevelName(logging.WARNING, "\033[1;33m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
        logging.addLevelName(logging.ERROR, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
        logging.addLevelName(logging.INFO, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.INFO))
        logging.addLevelName(logging.DEBUG, "\033[1;33m%s\033[1;0m" % logging.getLevelName(logging.DEBUG))
        logging.getLogger().addFilter(logging.Filter("Jaeger"))

        logger = logging.getLogger('Jaeger')

        file_base = os.path.splitext(os.path.basename(args.input))[0]
        self.file_handler = logging.FileHandler(os.path.join(args.output,f"{file_base}_jaeger.log"))
        logger.addHandler(self.file_handler)
        logger.setLevel(logging.DEBUG)

        self.stderr_handler = logging.StreamHandler()
        self.stderr_handler.setLevel(log_levels[args.verbose])

        self.formatter_stdout = logging.Formatter('[%(asctime)s | %(levelname)s] : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        self.formatter_log =logging.Formatter('[%(asctime)s] : %(message)s' , datefmt='%Y-%m-%d %H:%M:%S')
        self.formatter_clean=logging.Formatter('')

        self.stderr_handler.setFormatter(self.formatter_stdout)
        self.file_handler.setFormatter(self.formatter_log)
        logger.addHandler(self.stderr_handler)
        logger.addHandler(self.file_handler)
        self.logger = logger
    
    def info(self,message, cleanformat=False):
        if cleanformat == True:

            self.stderr_handler.setFormatter(self.formatter_clean)
            self.file_handler.setFormatter(self.formatter_clean)
            self.logger.info(message)
            self.reset_handler()
        else:
            self.logger.info(message)

    def warn(self,message, cleanformat=False):
        if cleanformat == True:

            self.stderr_handler.setFormatter(self.formatter_clean)
            self.file_handler.setFormatter(self.formatter_clean)
            self.logger.warning(message)
            self.reset_handler()
        else:
            self.logger.warning(message)  

    def error(self,message, cleanformat=False):
        if cleanformat == True:

            self.stderr_handler.setFormatter(self.formatter_clean)
            self.file_handler.setFormatter(self.formatter_clean)
            self.logger.error(message)
            self.reset_handler()
        else:
            self.logger.error(message)  

    def debug(self,message, cleanformat=False):
        if cleanformat == True:

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

def format_seconds(seconds):
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f'{minutes} minutes and {remaining_seconds} seconds'            

def fasta_entries(input_file_handle, chunk_size=1024):
    num = 0
    while True:
        chunk = input_file_handle.read(chunk_size)
        if not chunk:
            break
        num += chunk.count('>')

    input_file_handle.seek(0)
    return num

class Precision_per_class(tf.keras.metrics.Metric):

    def __init__(self, name='Precision_per_class',num_classes=4 ,**kwargs):
        super(Precision_per_class, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros',shape=num_classes)
        self.pred_positives = self.add_weight(name='pp', initializer='zeros',shape=num_classes)
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
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            tp = tf.multiply(tp, sample_weight)
            pp = tf.multiply(pp, sample_weight)
            
        self.true_positives.assign_add(tf.reduce_sum(tp,axis=0))
        self.pred_positives.assign_add(tf.reduce_sum(pp,axis=0))
        
    def reset_state(self):
        self.true_positives.assign(tf.zeros(shape=self.num_classes))
        self.pred_positives.assign(tf.zeros(shape=self.num_classes))
        
    def result(self):
        result= tf.math.divide_no_nan(self.true_positives,self.pred_positives)
 
        #return  {'prec_c1':result[0],'prec_c2':result[1],'prec_c3':result[2],'prec_c4':result[3],'prec_c5':result[4],'prec_c6':result[5]}
    
        return result
    
class Recall_per_class(tf.keras.metrics.Metric):

    def __init__(self, name='Recall_per_class',num_classes=4 ,**kwargs):
        super(Recall_per_class, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros',shape=num_classes)
        self.positives = self.add_weight(name='positives', initializer='zeros',shape=num_classes)
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
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            tp = tf.multiply(tp, sample_weight)
            p = tf.multiply(p, sample_weight)
            
        self.true_positives.assign_add(tf.reduce_sum(tp,axis=0))
        self.positives.assign_add(tf.reduce_sum(p,axis=0))
        
    def reset_state(self):
        self.true_positives.assign(tf.zeros(shape=self.num_classes))
        self.positives.assign(tf.zeros(shape=self.num_classes))
        
    def result(self):
        result= tf.math.divide_no_nan(self.true_positives,self.positives)
    
        return result